"""Loaders shared by the readouts: the features of superglance.eval.features (all shards of a directory) and an
adapter's Policy Head (W [K, d], b [K], float32)."""
import os
import numpy as np


def load_h(xd):
    tok, code, h = [], [], []
    for k in range(64):
        if not os.path.exists(f"{xd}/tok_{k}.txt"):
            continue
        tok += open(f"{xd}/tok_{k}.txt").read().split(); code.append(np.load(f"{xd}/code_{k}.npy")); h.append(np.load(f"{xd}/h_{k}.npy"))
    return tok, np.concatenate(code), np.concatenate(h).astype(np.float32)


def head(ad):
    from safetensors import safe_open
    with safe_open(f"{ad}/adapter_model.safetensors", "pt") as f:
        return [f.get_tensor(f"base_model.model.choice_head.{k}").float().numpy() for k in ("weight", "bias")]
