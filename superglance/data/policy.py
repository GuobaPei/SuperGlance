"""Stage-3 rows: the dataset (compact policy pools and QA replay rows), the processor batch with past frames, and the
float32 readout of the Policy Head and the option letters."""
import json
import io
from functools import lru_cache
from pathlib import Path
import sqlite3

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from superglance.model.prompt import build_choice_prompt, candidate_token_ids, render_chat  # noqa: F401 (candidate_token_ids: imported from here)
from superglance.model import history as mf_hist


@lru_cache(maxsize=4)
def image_store(path):
    return sqlite3.connect(f'file:{path}?mode=ro', uri=True)


def open_decision_image(row):
    if 'image_store_key' in row:
        payload = image_store(row['image_file']).execute(
            'SELECT data FROM images WHERE id = ?', (row['image_store_key'],)).fetchone()[0]
        return Image.open(io.BytesIO(payload))
    return Image.open(row['image_file'])


class DecisionDataset(Dataset):
    def __init__(self, manifest, limit=None):
        self.manifest = Path(manifest)
        with self.manifest.open() as stream:
            self.rows = [json.loads(line) for line in stream if line.strip()]
        if limit is not None:
            self.rows = self.rows[:limit]
        # compact pools (thousands of candidates): the K-way targets / rewards live in memory-mapped float16 arrays next to
        # the manifest; every row shares one answers list, so len(row["answers"]) keeps working everywhere
        self._targets = self._rewards = self._ref = None
        if any("compact" in row for row in self.rows):  # QA option rows may share a compact manifest
            stem = self.manifest.stem
            self._targets = np.load(self.manifest.parent / f"{stem}_targets.npy", mmap_mode="r")
            rewards = self.manifest.parent / f"{stem}_rewards.npy"
            self._rewards = np.load(rewards, mmap_mode="r") if rewards.exists() else None
            ref = self.manifest.parent / f"{stem}_ref.npy"  # RL pools: the supervised model's head logits (KL reference)
            self._ref = np.load(ref, mmap_mode="r") if ref.exists() else None
            k = int(self._targets.shape[1])
            for name, arr in (("rewards", self._rewards), ("ref", self._ref)):  # one width for every array, never zero-padded
                assert arr is None or arr.shape[1] == k, f"{stem}_{name}.npy is {arr.shape}, {stem}_targets.npy {self._targets.shape}"
            answers = [f"t{i:0{3 if k <= 1000 else 4}d}" for i in range(k)]
            for row in self.rows:
                if "compact" not in row:
                    continue  # QA option rows keep their own answers
                assert row["compact"]["k"] == k and 0 <= row["compact"]["row"] < len(self._targets), row.get("token")
                row["answers"] = answers

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = dict(self.rows[index])
        row["image_file"] = str(self.manifest.parent / row["images"][0])
        if len(row["images"]) > 1:  # the past front frames, oldest first
            row["extra_image_files"] = [str(self.manifest.parent / p) for p in row["images"][1:]]
        if "compact" in row:  # the row's per-candidate lists, read from the pool arrays
            j = row["compact"]["row"]
            q = np.asarray(self._targets[j], dtype=np.float64)
            q = q / q.sum()
            row["target_probs"] = [round(float(x), 7) for x in q]
            if self._rewards is not None:
                row["option_rewards"] = [round(float(x), 5) for x in self._rewards[j]]
            if self._ref is not None:
                row["ref_logits"] = [float(x) for x in self._ref[j]]
        return row


def choice_linear(model):
    """The Policy Head (its trainable PEFT modules_to_save copy), or None."""
    m = getattr(model, "module", model)
    base = m.get_base_model() if hasattr(m, "get_base_model") else m
    head = getattr(base, "choice_head", None)
    if head is not None and hasattr(head, "modules_to_save"):
        head = head.modules_to_save[next(iter(head.modules_to_save))]
    return head


def readout(model, inputs, token_ids, c256_rows=None, width=None):
    """fp32 candidate logits from the final hidden state at the last prompt position (inputs LEFT padded): letter rows
    h . W_lm[token_ids], Policy Head rows h . W + b. Returns [B, width]; entries past a row's candidate count
    are masked downstream."""
    m = getattr(model, "module", model)
    head = m.get_output_embeddings()
    assert head.bias is None
    seen = []
    hook = head.register_forward_hook(lambda module, args, output: seen.append(args[0]))
    try:
        model(**inputs, use_cache=False, logits_to_keep=1)
    finally:
        hook.remove()
    h = seen[-1][:, -1].float()
    out = h @ head.weight[token_ids].float().T
    width = width or out.shape[1]
    if width > out.shape[1]:
        out = torch.nn.functional.pad(out, (0, width - out.shape[1]))
    # Also with no policy row in the micro-batch (all-False mask, zero gradient) whenever the model has the head: DDP with
    # find_unused_parameters=False needs a gradient for every trainable parameter on every rank.
    if c256_rows is not None and (bool(c256_rows.any()) or choice_linear(model) is not None):
        lin = choice_linear(model)
        choice = h @ lin.weight.float().T + lin.bias.float()
        choice = torch.nn.functional.pad(choice, (0, width - choice.shape[1]))
        out = torch.where(c256_rows.to(h.device).bool()[:, None], choice, out)
    return out


class DecisionCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "left"

    def __call__(self, rows):
        texts, images = [], []
        for row in rows:
            prompt = row["question"] if row.get("no_options") else build_choice_prompt(row["question"], row["answers"])
            past = row.get("extra_image_files") or []
            if past:  # frame tags, the past frames first, the current frame last
                assert row.get("history") and len(past) == len(mf_hist.OFFS), row.get("token")
                messages = [{"role": "user", "content": mf_hist.content(len(past)) + [
                    {"type": "text", "text": mf_hist.question(prompt)}]}]
                images += [mf_hist.open_past(p) for p in past]
            else:
                messages = [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": prompt}
                ]}]
            texts.append(render_chat(self.processor, messages))
            with open_decision_image(row) as image:
                images.append(image.convert("RGB"))
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        inputs["gold_index"] = torch.tensor([row["gold_index"] for row in rows])
        inputs["candidate_count"] = torch.tensor([len(row["answers"]) for row in rows])
        return inputs
