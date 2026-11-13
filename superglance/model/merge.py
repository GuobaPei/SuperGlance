#!/usr/bin/env python3
"""Merge a LoRA adapter into a safetensors checkpoint without instantiating the model (CPU is enough).

Every adapted weight becomes W' = round_to_dtype(W.float() + (alpha / r) * B.float() @ A.float()), one rounding to the
checkpoint dtype (bf16). All other tensors (fp32 A_log / linear-attention norm weights, the unused MTP head, vision
tower, embeddings, LM head) are copied bit-exactly and all non-weight files (config, tokenizer, processor, chat
template) are copied unchanged, so the merged directory loads exactly like the original. Stage 2 trains a fresh
adapter on this model; its KL anchor (adapter disabled) is then the stage-1 model.
Usage: python -m superglance.model.merge --model BASE_DIR --adapter ADAPTER_DIR --output NEW_DIR
"""
import argparse
import json
import re
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    config = json.loads((args.adapter / "adapter_config.json").read_text())
    assert config["peft_type"] == "LORA" and not config.get("use_rslora") and not config.get("use_dora"), config
    assert not config.get("modules_to_save") and config.get("bias", "none") == "none", config
    scale = config["lora_alpha"] / config["r"]
    pairs = {}
    for key, tensor in load_file(str(args.adapter / "adapter_model.safetensors")).items():
        match = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.weight", key)
        assert match, f"unexpected adapter tensor {key}"
        pairs.setdefault(f"{match.group(1)}.weight", {})[match.group(2)] = tensor
    assert all(set(ab) == {"A", "B"} for ab in pairs.values())
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    missing = [key for key in pairs if key not in index["weight_map"]]
    assert not missing, f"adapter targets missing from the checkpoint: {missing[:5]}"
    args.output.mkdir(parents=True, exist_ok=False)  # never overwrite an existing model directory
    merged, relative_delta = 0, {}
    for shard in sorted(set(index["weight_map"].values())):
        tensors = {}
        with safe_open(str(args.model / shard), framework="pt") as stream:
            metadata = stream.metadata() or {"format": "pt"}
            for key in stream.keys():
                tensor = stream.get_tensor(key)
                if key in pairs:
                    delta = scale * (pairs[key]["B"].float() @ pairs[key]["A"].float())
                    assert delta.shape == tensor.shape, (key, delta.shape, tensor.shape)
                    base = tensor.float()
                    relative_delta[key] = float(delta.norm() / base.norm())
                    tensor = (base + delta).to(tensor.dtype)
                    merged += 1
                tensors[key] = tensor.contiguous()
        save_file(tensors, str(args.output / shard), metadata=metadata)
        print(f"wrote {shard} ({len(tensors)} tensors, merged so far {merged})", flush=True)
    assert merged == len(pairs), (merged, len(pairs))
    for path in args.model.iterdir():
        if path.is_file() and not path.name.endswith(".safetensors"):
            shutil.copy2(path, args.output / path.name)
    kinds = {}
    for key, value in relative_delta.items():
        kind = re.search(r"\.(self_attn|linear_attn|mlp)\.(\w+)\.weight$", key)
        kinds.setdefault(f"{kind.group(1)}.{kind.group(2)}", []).append(value)
    info = {
        "base_model": args.model.name, "adapter": args.adapter.name, "scale": scale, "merged_tensors": merged,
        "rounding": "W' = round_to_checkpoint_dtype(W.float() + scale * B.float() @ A.float()); other tensors copied bit-exactly",
        "relative_delta_norm_by_module": {k: {"n": len(v), "mean": sum(v) / len(v), "max": max(v)} for k, v in sorted(kinds.items())},
        "seconds": round(time.time() - started, 1),
    }
    (args.output / "MERGE_INFO.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
