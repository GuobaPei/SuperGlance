"""WOD-E2E shard of the stage-3 SFT pool, from all 10 Hz training frames (wod_rows --sel every:1 -> <wod rows>/train1), in the
NAVSIM rows' format: target softmax(-ADE / 1 m) over the 5,121 candidates to the logged future's 8 poses at 0.5-4 s (label_source
softmax_minus_ade_1m); segments with crc32(segment) % 40 == 0 are held out of training and 200 of their rows (seeded draw) are the
dev rows. A standalone shard for merge.py (compact rows from 0), float32 targets, image paths relative to the pool dir as
wod/<segment>/<frame>.jpg (copies.tsv: source -> that relative path); no prompt may carry the road-users block.
usage: python -m superglance.builders.pools.wod_shard <wod rows dir> <cands.npy> <out dir>"""
import glob, json, os, random, sys, zlib
import numpy as np

ROWS, CANDS, OUT = sys.argv[1:4]
C = np.load(CANDS)[:, :, :2].astype(np.float32); K = len(C)
os.makedirs(OUT)   # never overwrite

R = [json.loads(l) for p in sorted(glob.glob(f"{ROWS}/meta/*.jsonl")) for l in open(p)]
assert not any("Road users ahead" in r["question"] for r in R)   # no detector-made text in the model inputs
held = [r for r in R if zlib.crc32(r["segment"].encode()) % 40 == 0]; train = [r for r in R if zlib.crc32(r["segment"].encode()) % 40]
dev = random.Random(20261023).sample(held, min(200, len(held)))
rel = lambda r: f"wod/{r['segment']}/{r['frame']:03d}.jpg"
report = {"wod_rows": len(R), "held_segments_rows": len(held), "segments": len({r["segment"] for r in R}), "k": K}
for split, rs in (("train", train), ("dev", dev)):
    T = np.lib.format.open_memmap(f"{OUT}/{split}_targets.npy.tmp", "w+", np.float32, (len(rs), K)); G = np.empty(len(rs), np.int64)
    for i in range(0, len(rs), 512):   # softmax(-ADE / 1 m), as wod_pool.target
        F8 = np.array([r["future"] for r in rs[i:i + 512]], np.float32)[:, 1:16:2]
        a = np.linalg.norm(C[None] - F8[:, None], axis=-1).mean(-1); q = np.exp(-(a - a.min(1, keepdims=True)))
        T[i:i + 512] = q / q.sum(1, keepdims=True); G[i:i + 512] = a.argmin(1)
    T.flush(); del T; os.replace(f"{OUT}/{split}_targets.npy.tmp", f"{OUT}/{split}_targets.npy")
    with open(f"{OUT}/{split}.jsonl", "w") as f:
        for j, (r, g) in enumerate(zip(rs, G)):
            f.write(json.dumps({"question": r["question"], "no_options": True, "fixed_order": True, "gold_index": int(g), "id": f"wod_{r['token']}",
                                "token": r["token"], "source": "wod", "sampling_task": "wod", "sampling_bucket": "all",
                                "images": [rel(r)], "correct_answer": f"t{int(g):04d}",
                                "label_source": "softmax_minus_ade_1m", "compact": {"k": K, "row": j}}) + "\n")
    report[split] = len(rs)
with open(f"{OUT}/copies.tsv", "w") as f:
    f.writelines(f"{r['images'][0]}\t{rel(r)}\n" for r in train + dev)
assert len({rel(r) for r in train + dev}) == len(train) + len(dev)   # one image per row
json.dump(report, open(f"{OUT}/report.json", "w"), indent=1); print(json.dumps(report))
