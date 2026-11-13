"""The stage-3 trainer with the plan written as text by the language-model head (Table 8, "Text, decoded"). Same backbone,
LoRA, pool and QA rows as the supervised model. Each trajectory row's answer is its logged future (superglance.ablations.traj_targets,
TRAJ_DIR/{train,dev}_traj.npy; the trajectory its softmax(-ADE) imitation target used) in ReCogDrive's text format: "Here is the planning
trajectory [PT, (x, y, heading), ...]." with 8 poses, signed, 2 decimals, then <|im_end|>. A missing heading is taken from the
neighbouring points' differences. Trajectory rows train with teacher-forced cross-entropy on the answer tokens; QA rows keep their loss
and the KL anchor unchanged, and each micro-batch weights the two kinds of items by their counts, as permutation_loss does. Trajectory
rows without a recovered trajectory are dropped. The unused Policy Head stays at 0.
usage: TRAJ_DIR=<dir> torchrun ... -m superglance.ablations.text <superglance.train.policy arguments>"""
import math
import os

import numpy as np
import torch

from superglance.data import policy as dd
from superglance.model import history as mf_hist
from superglance.data import policy_pairs as pdata
from superglance.train import policy as tp
from superglance.objectives.permutation import candidate_kl, permutation_loss
from superglance.model.prompt import build_choice_prompt, render_chat

TD = os.environ["TRAJ_DIR"]
_init, _getitem = dd.DecisionDataset.__init__, dd.DecisionDataset.__getitem__


def fmt(n):   # ReCogDrive's format_number
    return 0.0 if abs(round(n, 2)) <= 1e-2 else f"{n:+.2f}"


def with_heading(p):   # [8, 3]; NaN headings from the neighbouring points' differences (the first from the origin)
    p = np.array(p, np.float64)
    if np.isfinite(p[:, 2]).all():
        return p
    prev, last = np.zeros(2), 0.0
    for i in range(8):
        d = p[i, :2] - prev
        if not np.isfinite(p[i, 2]):
            p[i, 2] = math.atan2(d[1], d[0]) if np.hypot(*d) > 0.05 else last
        prev, last = p[i, :2], p[i, 2]
    return p


def answer(p):
    return f"Here is the planning trajectory [PT, {', '.join(f'({fmt(x)}, {fmt(y)}, {fmt(h)})' for x, y, h in p)}]."


def ds_init(self, manifest, limit=None):
    _init(self, manifest, limit)
    T = np.load(f"{TD}/{self.manifest.stem}_traj.npy", mmap_mode="r"); assert len(T) == len(self.rows), (len(T), len(self.rows))
    keep, drop = [], 0
    for i, row in enumerate(self.rows):
        row["_line"] = i
        if "compact" in row and not np.isfinite(T[i][:, :2]).all():
            drop += 1
            continue
        keep.append(row)
    self.rows, self._traj = keep, T
    print(f"{self.manifest.name}: {drop} trajectory rows without a recovered trajectory dropped, {len(keep)} rows kept", flush=True)


def ds_getitem(self, index):
    row = _getitem(self, index)
    if "compact" in row:
        row["answer_text"] = answer(with_heading(self._traj[row["_line"]]))
    return row


dd.DecisionDataset.__init__, dd.DecisionDataset.__getitem__ = ds_init, ds_getitem


class TextCollator(pdata.PermutationCollator):
    """The parent's batch, with every trajectory row's prompt followed by its answer and <|im_end|>: rows are left padded, so a
    row's answer is its last n tokens; answer_len holds n per encoded row (0 for QA rows)."""
    def __init__(self, processor, max_options):
        super().__init__(processor, max_options)
        base = self.base

        def collate(rows):
            texts, images = [], []
            for row in rows:
                prompt = row["question"] if row.get("no_options") else build_choice_prompt(row["question"], row["answers"])
                past = row.get("extra_image_files") or []
                if past:
                    assert row.get("history") and len(past) == len(mf_hist.OFFS), row.get("token")
                    messages = [{"role": "user", "content": mf_hist.content(len(past)) + [{"type": "text", "text": mf_hist.question(prompt)}]}]
                    images += [mf_hist.open_past(p) for p in past]
                else:
                    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                text = render_chat(base.processor, messages)
                if row.get("answer_text"):
                    text += row["answer_text"] + "<|im_end|>"
                texts.append(text)
                with dd.open_decision_image(row) as image:
                    images.append(image.convert("RGB"))
            inputs = base.processor(text=texts, images=images, padding=True, return_tensors="pt")
            tok = base.processor.tokenizer; n = []
            for i, row in enumerate(rows):
                if not row.get("answer_text"):
                    n.append(0); continue
                ids = tok(row["answer_text"] + "<|im_end|>", add_special_tokens=False).input_ids
                assert inputs["input_ids"][i, -len(ids):].tolist() == ids, ("answer tokens differ inside the prompt", row.get("token"))
                n.append(len(ids))
            inputs["gold_index"] = torch.tensor([row["gold_index"] for row in rows])
            inputs["candidate_count"] = torch.tensor([len(row["answers"]) for row in rows])
            inputs["answer_len"] = torch.tensor(n)
            return inputs
        self.base = collate   # the parent calls self.base(rows) for the encoded rows and the KL reference rows

    def __call__(self, items):
        batch = super().__call__(items)
        if "ref_inputs" in batch:
            batch["ref_inputs"].pop("answer_len", None)   # reference rows are QA rows: no answer
        return batch


def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    inputs = dict(inputs)
    alen, view_index, ref_inputs = inputs.pop("answer_len"), inputs.pop("view_index"), inputs.pop("ref_inputs", None)
    inputs.pop("c256_rows", None)
    soft = inputs["soft_rows"].bool(); view_target = inputs["view_target"]
    model_inputs, labels, view_gold, counts, canonical_index = self.split_inputs(inputs)
    m = getattr(model, "module", model); lm = m.get_output_embeddings()
    norm = next(mod for name, mod in m.named_modules() if name.endswith("language_model.norm")); seen = []
    hook = norm.register_forward_hook(lambda module, args, output: seen.append(output))
    try:
        model(**model_inputs, use_cache=False, logits_to_keep=1)
    finally:
        hook.remove()
    H = seen[-1]; h = H[:, -1].float()
    letters = h @ lm.weight[self.token_ids].float().T
    letters = torch.nn.functional.pad(letters, (0, self.max_options - letters.shape[1]))[view_index]
    ids = model_inputs["input_ids"]; parts, weights = [], []
    if bool(soft.any()):
        rows = view_index[soft][::2].tolist(); n = [int(alen[r]) for r in rows]; assert min(n) > 0, n
        # positions T-n-1 .. T-2 predict the answer tokens T-n .. T-1; logits in the model dtype, the loss in fp32
        z = (torch.cat([H[r, -k - 1:-1] for r, k in zip(rows, n)]) @ lm.weight.T).float()
        tok = torch.nn.functional.cross_entropy(z, torch.cat([ids[r, -k:] for r, k in zip(rows, n)]), reduction="none")
        parts.append(torch.stack([x.mean() for x in tok.split(n)]).mean()); weights.append(len(rows))   # per-row mean, then rows
    if not bool(soft.all()):
        hard = ~soft
        parts.append(permutation_loss(letters[hard], view_gold[hard], counts[hard], canonical_index[hard], 0.0, self.samples,
                                      self.pair_reward, view_target[hard])[0])
        weights.append(int(hard[::2].sum()))
    lin = dd.choice_linear(model)   # unused; DDP (find_unused_parameters=False) still needs its gradient on every rank
    loss = sum(w * x for w, x in zip(weights, parts)) / sum(weights) + 0.0 * (lin.weight.sum() + lin.bias.sum())
    if self.kl_weight and bool((~soft).any()):   # superglance.train.policy's anchor on the QA rows, unchanged
        peft_model = self.accelerator.unwrap_model(model)
        with torch.no_grad(), peft_model.disable_adapter():
            base = dd.readout(peft_model, self._prepare_input(ref_inputs), self.token_ids, None, self.max_options)
        loss = loss + self.kl_weight * candidate_kl(letters[~soft], base, counts[~soft])
    return (loss, {"logits": letters}) if return_outputs else loss


_predict = tp.PermutationTrainer.prediction_step


def prediction_step(self, model, inputs, *args, **kwargs):   # dev reads only: the answer lengths are not a model input
    inputs = dict(inputs); inputs.pop("answer_len", None)
    return _predict(self, model, inputs, *args, **kwargs)


tp.PermutationCollator = TextCollator
tp.PermutationTrainer.compute_loss = compute_loss
tp.PermutationTrainer.prediction_step = prediction_step

if __name__ == "__main__":
    tp.main()
