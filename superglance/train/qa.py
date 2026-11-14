#!/usr/bin/env python3
"""Stages 1 and 2: multiple-choice QA on two option orders (joint Brier policy gradient), KL to the frozen start.

Answers are read by the option letters' LM-head logits at the answer position. The KL anchor is the same model with
the adapter disabled: the base VLM in stage 1, the merged stage-1 model in stage 2.
Run: torchrun --nproc_per_node 8 -m superglance.train.qa --config CONFIG --model MODEL --data DATA --output OUT
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    EvalPrediction,
    set_seed,
)

from superglance.data.qa import candidate_token_ids, native_candidate_logits
from superglance.data.qa_pairs import PermutationCollator, PermutationDataset
from superglance.objectives.permutation import aligned_probabilities, candidate_kl, permutation_loss
from superglance.data.mixture import MixtureSampler
from superglance.model.prompt import validate_answer_boundary

# Qwen3.5: layers i % 4 == 3 are full attention (q/k/v/o); the other 24 are Gated DeltaNet linear attention
# (in_proj_qkv = fused q/k/v, in_proj_z = output multiplier, out_proj = o). in_proj_a / in_proj_b (4096 -> 32,
# decay and beta inputs) are left out: rank-32 LoRA on a 32-wide output is a full-rank update of the recurrence.
Q35_LORA_TARGETS = (r"model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
                    r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate_proj|up_proj|down_proj))")


class FinalCheckpoint(TrainerCallback):
    def __init__(self, evaluate=True):
        self.evaluate = evaluate   # False without dev.jsonl: the last step is saved, not evaluated

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == state.max_steps:
            control.should_evaluate = self.evaluate
            control.should_save = True


class EvalAfterFirstStep(TrainerCallback):
    """Smoke tests: the first optimizer step runs at warm-up lr 0, so this eval equals the untrained model."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


def trainable_parameter_report(model):
    """Trainable parameters by module kind; fails if anything outside the language LoRA adapters trains."""
    groups, total = Counter(), 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" not in name or ".language_model.layers." not in name:
            raise RuntimeError(f"Unexpected trainable parameter outside language LoRA: {name}")
        kind = re.search(r"\.(self_attn|linear_attn|mlp)\.(\w+)\.lora_", name)
        groups[f"{kind.group(1)}.{kind.group(2)}"] += parameter.numel()
        total += parameter.numel()
    modules = [name for name, module in model.named_modules()
               if hasattr(module, "lora_A") and ".language_model.layers." in name]
    layers = {int(re.search(r"\.layers\.(\d+)\.", name).group(1)) for name in modules}
    return {"trainable_parameters": total, "by_module": dict(sorted(groups.items())),
            "lora_modules": len(modules), "layers_with_lora": len(layers)}


def source_metrics(rows):
    sources = np.asarray([row['source'] for row in rows])

    def compute(prediction):
        metrics = decision_metrics(prediction)
        per_source = []
        for source in sorted(set(sources)):
            mask = sources == source
            subset = EvalPrediction(predictions=tuple(p[mask] for p in prediction.predictions),
                                    label_ids=prediction.label_ids[mask])
            result = decision_metrics(subset)
            for name in ['original_accuracy', 'swapped_accuracy', 'both_correct', 'content_consistency', 'qa_count']:
                metrics[f'source_{source}_{name}'] = result[name]
            per_source.append(result['both_correct'])
        metrics['source_macro_both_correct'] = float(np.mean(per_source))
        return metrics
    return compute


class QACountCollator:
    """Expose the original QA batch size to HF's evaluation bookkeeping.

    Model tensors have 2B rows, while labels/predictions have B rows. Trainer
    finds batch size from the first tensor, so labels must precede input_ids.
    This also makes its scalar-loss repetition match the QA-level tail batch.
    """
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, items):
        batch = self.collator(items)
        labels = batch.pop("labels")
        return {"labels": labels, **batch}


def decision_metrics(prediction):
    """All predictions and GT refer to the first view's answer identities."""
    probabilities, winners = prediction.predictions
    probabilities = np.asarray(probabilities, dtype=np.float64)
    winners = np.asarray(winners)
    labels = np.asarray(prediction.label_ids)
    gold = labels[:, 0].astype(np.int64)
    counts = labels[:, 1].astype(np.int64)
    valid = np.arange(probabilities.shape[-1])[None, None] < counts[:, None, None]
    probabilities = np.where(valid, probabilities, 0.0)
    correct = winners == gold[:, None]
    tied_top = (probabilities == probabilities.max(-1, keepdims=True)).sum(-1) > 1
    target = np.eye(probabilities.shape[-1])[gold][:, None]
    gold_prob = np.take_along_axis(probabilities, gold[:, None, None], axis=-1)
    logp = np.log(np.maximum(probabilities, 1e-12))
    mean_prob = probabilities.mean(1, keepdims=True)
    js = (probabilities * (logp - np.log(np.maximum(mean_prob, 1e-12)))).sum(-1).mean(1)
    metrics = {
        "accuracy": float(correct.mean()),
        "original_accuracy": float(correct[:, 0].mean()),
        "swapped_accuracy": float(correct[:, 1].mean()),
        "swap_accuracy_delta": float(correct[:, 1].mean() - correct[:, 0].mean()),
        "minimum_order_accuracy": float(correct.mean(0).min()),
        "correct_to_wrong": int((correct[:, 0] & ~correct[:, 1]).sum()),
        "wrong_to_correct": int((~correct[:, 0] & correct[:, 1]).sum()),
        "both_correct": float(correct.all(-1).mean()),
        "content_consistency": float((winners[:, 0] == winners[:, 1]).mean()),
        "exact_top_probability_tie_row_rate": float(tied_top.mean()),
        "nll": float(-np.log(np.maximum(gold_prob, 1e-12)).mean()),
        "brier": float(((probabilities - target) ** 2).sum(-1).mean()),
        "content_js": float(js.mean()),
        "qa_count": int(len(labels)),
    }
    for count in np.unique(counts):
        subset = correct[counts == count]
        prefix = f"k{count}_"
        metrics.update({
            prefix + "qa_count": int(len(subset)),
            prefix + "original_accuracy": float(subset[:, 0].mean()),
            prefix + "swapped_accuracy": float(subset[:, 1].mean()),
            prefix + "swap_accuracy_delta": float(subset[:, 1].mean() - subset[:, 0].mean()),
            prefix + "both_correct": float(subset.all(-1).mean()),
            prefix + "content_consistency": float((winners[counts == count, 0] == winners[counts == count, 1]).mean()),
            prefix + "correct_to_wrong": int((subset[:, 0] & ~subset[:, 1]).sum()),
            prefix + "wrong_to_correct": int((~subset[:, 0] & subset[:, 1]).sum()),
        })
    return metrics


class PermutationTrainer(Trainer):
    def __init__(self, *args, token_ids, samples=32, consistency_weight=0.0, mixture_config=None, kl_weight=0.0,
                 pair_reward="independent", soft_objective="brier", fp32_readout=False, kl_skip_soft_rows=False,
                 **kwargs):
        kwargs["data_collator"] = QACountCollator(kwargs["data_collator"])
        kwargs.setdefault("compute_metrics", decision_metrics)
        super().__init__(*args, **kwargs)
        self.token_ids = token_ids
        self.samples = samples
        self.consistency_weight = consistency_weight
        self.model_accepts_loss_kwargs = False
        self.mixture_config = mixture_config
        self.kl_weight = kl_weight
        self.pair_reward = pair_reward
        self.soft_objective = soft_objective
        self.fp32_readout = fp32_readout
        self.kl_skip_soft_rows = kl_skip_soft_rows

    def _get_train_sampler(self, train_dataset=None):
        if self.mixture_config is None:
            return super()._get_train_sampler(train_dataset)
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset.pair_rounds != 1:
            raise ValueError('Mixture sampling expects one two-view item per original QA')
        return MixtureSampler(dataset.rows, self.mixture_config, self.args.data_seed)

    def create_optimizer(self, model=None):
        optimizer = super().create_optimizer(model)
        # ZeRO removes empty LoRA groups; do it before the scheduler sees them.
        optimizer.param_groups[:] = [group for group in optimizer.param_groups if group["params"]]
        return optimizer

    @staticmethod
    def split_inputs(inputs):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        view_gold = inputs.pop("view_gold")
        counts = inputs.pop("candidate_count")
        canonical_index = inputs.pop("canonical_index")
        inputs.pop("view_target", None)  # read by compute_loss before the split; not a model input
        inputs.pop("soft_rows", None)
        inputs.pop("view_reward", None)
        inputs.pop("reward_rows", None)
        return inputs, labels, view_gold, counts, canonical_index

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        view_target = inputs.get("view_target")  # one-hot on the gold option unless a row carries soft targets
        soft_rows = inputs.get("soft_rows")
        model_inputs, labels, view_gold, counts, canonical_index = self.split_inputs(inputs)
        logits = native_candidate_logits(model, model_inputs, self.token_ids, fp32=self.fp32_readout)
        loss, _ = permutation_loss(
            logits, view_gold, counts, canonical_index,
            consistency_weight=self.consistency_weight, samples=self.samples,
            pair_reward=self.pair_reward, view_target=view_target, soft_rows=soft_rows, soft_objective=self.soft_objective,
            view_reward=inputs.get("view_reward"), reward_rows=inputs.get("reward_rows"),
        )
        if self.kl_weight:
            # Anchor to the untouched base: same inputs, adapter disabled, no gradient.
            peft_model = self.accelerator.unwrap_model(model)
            keep = torch.ones(len(logits), dtype=torch.bool, device=logits.device)
            if self.kl_skip_soft_rows and soft_rows is not None:
                keep = ~soft_rows.to(logits.device).bool()  # the reference knows nothing about soft-target tasks
            if bool(keep.any()):
                with torch.no_grad(), peft_model.disable_adapter():
                    base_logits = native_candidate_logits(peft_model, model_inputs, self.token_ids, fp32=self.fp32_readout)
                loss = loss + self.kl_weight * candidate_kl(logits[keep], base_logits[keep], counts[keep])
        return (loss, {"logits": logits}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        model_inputs, labels, _, counts, canonical_index = self.split_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            logits = native_candidate_logits(model, model_inputs, self.token_ids, fp32=self.fp32_readout)
            probabilities = aligned_probabilities(logits, canonical_index, counts)
            valid = torch.arange(logits.shape[-1], device=logits.device)[None] < counts[:, None]
            # Preserve the actual decoder's first-letter tie break. Taking
            # argmax after content alignment would quietly change that rule.
            letter_winners = logits.masked_fill(~valid, -torch.inf).argmax(-1)
            content_winners = canonical_index.argsort(1).gather(1, letter_winners[:, None]).reshape(-1, 2)
            target = torch.nn.functional.one_hot(labels[:, 0], probabilities.shape[-1]).float()
            # Deterministic diagnostic: no paired Monte Carlo calls during eval.
            loss = ((probabilities - target[:, None]) ** 2).sum(-1).mean()
        if prediction_loss_only:
            return loss.detach(), None, None
        return loss.detach(), (probabilities.detach(), content_winners.detach()), labels.detach()

    def evaluate(self, *args, **kwargs):
        # Even an eval DataLoader takes a worker seed from the RNG. Preserve the
        # training RNG so eval cadence cannot alter the paired sampling stream.
        python_state, numpy_state = random.getstate(), np.random.get_state()
        devices = [self.args.device.index] if self.args.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                return super().evaluate(*args, **kwargs)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def training_arguments(cfg, output, max_steps=-1, evaluate=True):
    ds = cfg.get("deepspeed", "zero2.json")
    if ds:
        ds = str(Path(__file__).resolve().parents[2] / "configs" / ds)
    return TrainingArguments(
        output_dir=str(output), num_train_epochs=cfg["epochs"], max_steps=max_steps,
        # transformers 5 removed warmup_ratio; a float < 1 is a ratio: ceil(ratio * steps), as before.
        learning_rate=cfg["learning_rate"], warmup_steps=cfg["warmup_ratio"],
        per_device_train_batch_size=cfg["micro_batch_size"],
        per_device_eval_batch_size=cfg["eval_micro_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation"],
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=ds,
        eval_strategy="steps" if evaluate else "no", eval_steps=cfg["eval_steps"],
        save_strategy="steps", save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=evaluate,   # without dev.jsonl the last step is kept
        metric_for_best_model=cfg.get("selection_metric", "both_correct"), greater_is_better=True,
        logging_steps=cfg["logging_steps"], logging_first_step=True,
        remove_unused_columns=False, label_names=["labels"],
        dataloader_num_workers=cfg["workers"], dataloader_drop_last=False,
        report_to="none", seed=cfg["seed"], data_seed=cfg["seed"],
        ddp_find_unused_parameters=False, lr_scheduler_type="cosine",
        optim="adamw_torch", weight_decay=cfg["weight_decay"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True, help="Existing train.jsonl and independent dev.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--consistency-weight", type=float)
    parser.add_argument("--pair-mode", choices=["repeat", "swap", "coverage"])
    parser.add_argument("--pair-rounds", type=int, default=1)
    parser.add_argument("--resume-from-checkpoint", nargs="?", const=True, default=None)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.consistency_weight is not None:
        cfg["consistency_weight"] = args.consistency_weight
    if args.pair_mode is not None:
        cfg["pair_mode"] = args.pair_mode
    set_seed(cfg["seed"])

    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=cfg["min_pixels"], max_pixels=cfg["max_pixels"],
    )
    # Qwen3.5: the Auto class resolves Qwen3_5ForConditionalGeneration (transformers >= 5; dtype= replaces torch_dtype=).
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float32,
        attn_implementation=cfg["attention"],
    )
    model.model.visual.requires_grad_(False)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
        target_modules=cfg.get("lora_target_modules", Q35_LORA_TARGETS),
        task_type="CAUSAL_LM",
    ))
    lora_report = trainable_parameter_report(model)
    train = PermutationDataset(args.data / "train.jsonl", pair_mode=cfg["pair_mode"], seed=cfg["seed"], pair_rounds=args.pair_rounds)
    # dev.jsonl is optional: it only selects the best checkpoint (in stage 1 that was the last step).
    # Without it there is no evaluation and the last step is kept.
    dev = PermutationDataset(args.data / "dev.jsonl", pair_mode="swap", seed=cfg["seed"] + 100) \
        if (args.data / "dev.jsonl").exists() else None
    max_options = max(len(row["answers"]) for dataset in (train, dev) if dataset is not None for row in dataset.rows)
    answer_boundary = validate_answer_boundary(processor, max_options)
    token_ids = candidate_token_ids(processor, max_options)
    trainer = PermutationTrainer(
        model=model, args=training_arguments(cfg, args.output, args.max_steps, evaluate=dev is not None),
        train_dataset=train, eval_dataset=dev,
        data_collator=PermutationCollator(processor, max_options),
        token_ids=token_ids, samples=cfg["reward_samples"],
        consistency_weight=cfg["consistency_weight"],
        mixture_config=cfg.get("mixture_sampling"),
        kl_weight=cfg.get("kl_weight", 0.0),
        pair_reward=cfg.get("pair_reward", "independent"),
        soft_objective=cfg.get("soft_objective", "brier"),
        fp32_readout=cfg.get("fp32_readout", False),
        kl_skip_soft_rows=cfg.get("kl_skip_soft_rows", False),
        compute_metrics=source_metrics(dev.rows) if dev is not None else None,
        callbacks=[FinalCheckpoint(evaluate=dev is not None)] + ([EvalAfterFirstStep()] if cfg.get("eval_after_first_step") else []),
    )
    if trainer.is_world_process_zero():
        args.output.mkdir(parents=True, exist_ok=True)
        model.print_trainable_parameters()
        global_qa = cfg["micro_batch_size"] * cfg["gradient_accumulation"] * trainer.args.world_size
        run = dict(
            cfg, model=args.model, data=str(args.data), max_steps=args.max_steps,
            train_qa=len(train.rows), train_pair_items=len(train), pair_rounds=args.pair_rounds,
            dev_qa=len(dev) if dev is not None else 0, max_options=max_options,
            train_views=2 * len(train), global_qa_batch=global_qa,
            global_view_batch=2 * global_qa, eval_pair_mode="swap",
            objective=("joint_brier_pg (QA correct only if both orders are)" if cfg.get("pair_reward") == "joint"
                       else "paired_brier_pg") + " + consistency_weight * content_js + kl_weight * KL(p || p_base)",
            train_pool=train.pool_report, dev_pool=dev.pool_report if dev is not None else None,
            lora=lora_report, answer_boundary_token_ids=answer_boundary,
        )
        print(json.dumps({"lora": lora_report, "answer_boundary_token_ids": answer_boundary}), flush=True)
        if cfg.get("mixture_sampling"):
            sampler = trainer._get_train_sampler()
            run["mixture_sampling_report"] = sampler.report
            run["train_pair_items"] = len(sampler)
            run["train_views"] = 2 * len(sampler)
        (args.output / "run_config.json").write_text(json.dumps(run, indent=2) + "\n")
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    # Trainer has loaded its best dev checkpoint, if one was saved (without dev.jsonl: the last step).
    trainer.save_model(str(args.output / "final"))
    if dev is not None:
        metrics = trainer.evaluate(metric_key_prefix="dev")
        trainer.save_metrics("dev", metrics)
    if trainer.is_world_process_zero():
        processor.save_pretrained(args.output / "final")
        (args.output / "selection.json").write_text(json.dumps({
            "metric": "eval_" + cfg.get("selection_metric", "both_correct"),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "last_training_step": trainer.state.global_step,
            "final_adapter": str(args.output / "final"),
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
