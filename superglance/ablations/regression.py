"""The stage-3 trainer with a regression head (Table 8 and the appendix's One change at a time, "Linear regression"). The backbone and its LoRA are
the supervised model's, and there is one forward pass, as before. The Policy Head is replaced by a linear layer h -> 8 poses x (x, y,
heading), trained with L1 against each trajectory row's logged future: the trajectory its softmax(-ADE) imitation target was measured
against (superglance.ablations.traj_targets, TRAJ_DIR/{train,dev}_traj.npy, one [8, 3] per manifest line). A NaN heading is left out of
the loss. Trajectory rows without a recovered trajectory are dropped. QA rows keep their loss and the KL anchor unchanged. Each
micro-batch's loss weights trajectory and QA items by their counts, as permutation_loss does with soft and hard items. The head's bias
starts at the mean training trajectory and its weight at 0. Everything else is superglance.train.policy's own.
usage: TRAJ_DIR=<dir> torchrun ... -m superglance.ablations.regression <superglance.train.policy arguments>"""
import os

import numpy as np
import torch

from superglance.data import policy as dd
from superglance.data import policy_pairs as pdata
from superglance.train import policy as tp
from superglance.objectives.permutation import candidate_kl, permutation_loss

TD = os.environ["TRAJ_DIR"]
NAN = [float("nan")] * 24
_init, _getitem = dd.DecisionDataset.__init__, dd.DecisionDataset.__getitem__


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
    row["traj"] = [float(x) for x in np.asarray(self._traj[row["_line"]], np.float64).reshape(-1)] if "compact" in row else NAN
    return row


dd.DecisionDataset.__init__, dd.DecisionDataset.__getitem__ = ds_init, ds_getitem


class TrajCollator(pdata.PermutationCollator):
    def __call__(self, items):
        batch = super().__call__(items)
        physical = []
        for item in items:   # the parent's encoded-row order: the first view, and the second only when it differs
            first, second = item["views"]
            physical.append(first)
            if second != first:
                physical.append(second)
        batch["traj"] = torch.tensor([row["traj"] for row in physical], dtype=torch.float32).view(-1, 8, 3)
        return batch


def head(model):
    return dd.choice_linear(model)


def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    inputs = dict(inputs)
    traj, view_index, ref_inputs = inputs.pop("traj"), inputs.pop("view_index"), inputs.pop("ref_inputs", None)
    inputs.pop("c256_rows", None)
    soft = inputs["soft_rows"].bool(); view_target = inputs["view_target"]
    model_inputs, labels, view_gold, counts, canonical_index = self.split_inputs(inputs)
    m = getattr(model, "module", model); lm = m.get_output_embeddings(); seen = []
    hook = lm.register_forward_hook(lambda module, args, output: seen.append(args[0]))
    try:
        model(**model_inputs, use_cache=False, logits_to_keep=1)
    finally:
        hook.remove()
    h = seen[-1][:, -1].float()
    letters = h @ lm.weight[self.token_ids].float().T
    letters = torch.nn.functional.pad(letters, (0, self.max_options - letters.shape[1]))[view_index]
    lin = head(model); parts, weights = [], []
    if bool(soft.any()):
        rows = view_index[soft][::2]   # one encoded row per trajectory item (its two views are the same row)
        pred = (h[rows] @ lin.weight.float().T + lin.bias.float()).view(-1, 8, 3)
        tgt = traj.to(pred.device)[rows]; ok = torch.isfinite(tgt)
        l1 = torch.where(ok, (pred - tgt.nan_to_num()).abs(), torch.zeros_like(pred)).sum((1, 2)) / ok.sum((1, 2))
        parts.append(l1.mean()); weights.append(len(rows))
    else:   # DDP (find_unused_parameters=False) needs a gradient for the head on every rank and step
        parts.append(0.0 * (lin.weight.sum() + lin.bias.sum())); weights.append(0)
    if not bool(soft.all()):
        hard = ~soft
        parts.append(permutation_loss(letters[hard], view_gold[hard], counts[hard], canonical_index[hard], 0.0, self.samples,
                                      self.pair_reward, view_target[hard])[0])
        weights.append(int(hard[::2].sum()))
    loss = sum(w * x for w, x in zip(weights, parts)) / max(sum(weights), 1) + (parts[0] if weights[0] == 0 else 0.0)
    if self.kl_weight and bool((~soft).any()):   # superglance.train.policy's anchor on the QA rows, unchanged
        peft_model = self.accelerator.unwrap_model(model)
        with torch.no_grad(), peft_model.disable_adapter():
            base = dd.readout(peft_model, self._prepare_input(ref_inputs), self.token_ids, None, self.max_options)
        loss = loss + self.kl_weight * candidate_kl(letters[~soft], base, counts[~soft])
    return (loss, {"logits": letters}) if return_outputs else loss


_attach = tp.attach_lora


def attach_reg(model, cfg):
    d = model.config.text_config.hidden_size
    T = np.load(f"{TD}/train_traj.npy", mmap_mode="r"); s = T[::97].astype(np.float64)
    mean = np.nanmean(s, 0); mean = np.where(np.isfinite(mean), mean, 0.0)
    model.choice_head = torch.nn.Linear(d, 24, dtype=model.dtype)
    torch.nn.init.zeros_(model.choice_head.weight)
    with torch.no_grad():
        model.choice_head.bias.copy_(torch.tensor(mean.reshape(-1), dtype=model.dtype))
    print(f"regression head {d} -> 8 x 3, bias = mean trajectory of {len(s)} sampled lines (last pose {mean[-1].round(2).tolist()})", flush=True)
    return _attach(model, cfg)


_predict = tp.PermutationTrainer.prediction_step


def prediction_step(self, model, inputs, *args, **kwargs):   # dev reads only: the targets are not a model input
    inputs = dict(inputs); inputs.pop("traj", None)
    return _predict(self, model, inputs, *args, **kwargs)


tp.attach_lora = attach_reg
tp.PermutationCollator = TrajCollator
tp.PermutationTrainer.compute_loss = compute_loss
tp.PermutationTrainer.prediction_step = prediction_step

if __name__ == "__main__":
    tp.main()
