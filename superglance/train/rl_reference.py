"""The RL reference: the supervised policy's Policy Head logits on every compact row of the RL pool, read through the
trainer's own path (PermutationDataset -> PermutationCollator -> readout, first view; a policy row has two identical
views), so the KL term of exact_rl_loss compares the policy with what the supervised model gives on the same input.
  extract  shard r of n (compact rows r, r + n, ...) -> <out>/ref_<r>.npy (float16 [n_r, K]) and idx_<r>.npy (compact row ids)
  merge    -> <pool>/train_ref.npy (float16 [N, K], every compact row exactly once)
usage: python -m superglance.train.rl_reference extract <pool> <config.json> <model> <adapter> <out> <rank> <world> [bs=8] [workers=2]
       python -m superglance.train.rl_reference merge <pool> <out> <world>"""
import json, sys
import numpy as np


def extract(pool, config, model_dir, adapter, out, rank, world, bs=8, workers=2):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    from superglance.data.policy import readout
    from superglance.data.policy_pairs import PermutationCollator, PermutationDataset
    from superglance.train.policy import PermutationTrainer
    from superglance.model.prompt import candidate_token_ids
    cfg = json.load(open(config)); K = int(cfg["choice_k"]); rank, world, bs, workers = int(rank), int(world), int(bs), int(workers)
    proc = AutoProcessor.from_pretrained(model_dir, min_pixels=cfg["min_pixels"], max_pixels=cfg["max_pixels"])
    m = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation=cfg["attention"])
    m.choice_head = torch.nn.Linear(m.config.text_config.hidden_size, K, dtype=m.dtype)   # replaced by the adapter's copy
    m = PeftModel.from_pretrained(m, adapter).cuda().eval()
    ds = PermutationDataset(f"{pool}/train.jsonl", pair_mode=cfg["pair_mode"], seed=cfg["seed"])
    ds.rows = [r for r in ds.rows if "compact" in r][rank::world]
    col, tid = PermutationCollator(proc, K), candidate_token_ids(proc, 16)
    L, I = np.empty((len(ds.rows), K), np.float16), np.array([r["compact"]["row"] for r in ds.rows])
    # Batches are collated in DataLoader workers ahead of the GPU; workers=0 collates in the main thread (same output).
    batches = [range(a, min(a + bs, len(ds.rows))) for a in range(0, len(ds.rows), bs)]
    dl = torch.utils.data.DataLoader(batches, batch_size=None, collate_fn=lambda r: dict(col([ds[i] for i in r])), num_workers=workers,
                                     prefetch_factor=4 if workers else None)
    with torch.no_grad():
        for a, b in zip(range(0, len(ds.rows), bs), dl):
            vi, c256 = b.pop("view_index"), b.pop("c256_rows"); b.pop("ref_inputs", None)
            x = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in PermutationTrainer.split_inputs(b)[0].items()}
            L[a:a + len(vi) // 2] = readout(m, x, tid, c256, K)[vi][::2].cpu().numpy()
            if a % (100 * bs) == 0:
                print(f"rank {rank}: {a + len(vi) // 2} / {len(ds.rows)}", flush=True)
    np.save(f"{out}/ref_{rank}.npy", L); np.save(f"{out}/idx_{rank}.npy", I); print(f"rank {rank} done: {len(I)} rows", flush=True)


def merge(pool, out, world):
    n = sum(1 for l in open(f"{pool}/train.jsonl") if '"compact"' in l); K = np.load(f"{pool}/train_rewards.npy", mmap_mode="r").shape[1]
    R, seen = np.lib.format.open_memmap(f"{pool}/train_ref.npy.tmp", "w+", np.float16, (n, K)), np.zeros(n, bool)
    for r in range(int(world)):
        I = np.load(f"{out}/idx_{r}.npy"); assert not seen[I].any(); R[I] = np.load(f"{out}/ref_{r}.npy"); seen[I] = True
    assert seen.all() and np.isfinite(np.asarray(R, np.float32)).all(); R.flush(); del R
    import os; os.replace(f"{pool}/train_ref.npy.tmp", f"{pool}/train_ref.npy"); print(f"train_ref.npy: {n} x {K}")


if __name__ == "__main__":
    {"extract": extract, "merge": merge}[sys.argv[1]](*sys.argv[2:])
