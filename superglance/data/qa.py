"""Dataset, processor batch, and unchanged native answer-token readout."""
import json
import io
from functools import lru_cache
from pathlib import Path
import sqlite3

import torch
from PIL import Image
from torch.utils.data import Dataset
from superglance.model.prompt import build_choice_prompt, candidate_token_ids, native_candidate_logits, render_chat  # noqa: F401 (re-exported)


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

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = dict(self.rows[index])
        row["image_file"] = str(self.manifest.parent / row["images"][0])
        return row


class DecisionCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "left"

    def __call__(self, rows):
        texts, images = [], []
        for row in rows:
            prompt = build_choice_prompt(row["question"], row["answers"])
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
