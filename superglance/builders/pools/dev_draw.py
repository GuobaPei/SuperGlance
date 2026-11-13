"""Stage-3 dev split: draw n rows (seeded) from the union of shards' val splits into a dev-only shard for merge.py: dev.jsonl +
dev_targets.npy float32, compact rows renumbered from 0, rows otherwise unchanged. The paper's pool: 200 rows, seed 20261023,
from the PAI-AV human shards' val splits (their other val rows stay out of train).
usage: python -m superglance.builders.pools.dev_draw <out dir> <n> <seed> <shard dir>..."""
import json, os, random, sys
import numpy as np

OUT, N, SEED, SH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
os.makedirs(OUT)   # never overwrite
R = {s: [json.loads(l) for l in open(f"{s}/val.jsonl")] for s in SH}
pick = sorted(random.Random(SEED).sample([(k, i) for k, s in enumerate(SH) for i in range(len(R[s]))], N))
T = {s: np.load(f"{s}/val_targets.npy", mmap_mode="r") for s in SH}
M = np.empty((N, T[SH[0]].shape[1]), np.float32)
with open(f"{OUT}/dev.jsonl", "w") as f:
    for j, (k, i) in enumerate(pick):
        r = dict(R[SH[k]][i]); M[j] = T[SH[k]][r["compact"]["row"]]; r["compact"] = dict(r["compact"], row=j)
        f.write(json.dumps(r) + "\n")
np.save(f"{OUT}/dev_targets.npy", M)
json.dump({"dev": N, "seed": SEED, "from": {os.path.basename(s.rstrip("/")): sum(SH[k] == s for k, _ in pick) for s in SH}},
          open(f"{OUT}/report.json", "w"), indent=1)
print(open(f"{OUT}/report.json").read())
