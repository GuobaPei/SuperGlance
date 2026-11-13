"""The clustered candidates: k-means on the human 4-s trajectories of the NAVSIM training ("fit") scenes (8 waypoints x, y),
heading = the cluster's mean heading, plus one standing-still trajectory. K = 4,096 gives the first 4,097 rows of the 5,121-candidate
library; K = 1,024 and 256 the smaller sets of the supplement's coverage table. Same code and seed as
superglance.builders.candidates.study (k-means++ seeding on a 20,000-row sample, 30 Lloyd iterations; an empty cluster keeps its centre).
usage: python -m superglance.builders.candidates.kmeans <navtrain_inputs.jsonl> <K> <out.npy> [compare.npy]"""
import json
import sys

import numpy as np


def assign(X, C):
    c2 = (C ** 2).sum(1)
    return np.concatenate([(c2[None] - 2 * X[s:s + 8192] @ C.T).argmin(1) for s in range(0, len(X), 8192)])


def kmeans(X, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    sample = X[rng.choice(len(X), min(len(X), 20000), replace=False)]
    C = sample[rng.choice(len(sample), 1)]
    d = ((sample - C[0]) ** 2).sum(1)
    for _ in range(1, k):
        C = np.vstack([C, sample[rng.choice(len(sample), p=d / d.sum())]])
        d = np.minimum(d, ((sample - C[-1]) ** 2).sum(1))
    for _ in range(iters):
        lab = assign(X, C)
        sums = np.zeros_like(C); np.add.at(sums, lab, X)
        cnt = np.bincount(lab, minlength=k)
        C = np.where(cnt[:, None] > 0, sums / np.maximum(cnt, 1)[:, None], C)
    return C, assign(X, C)


rows = [json.loads(l) for l in open(sys.argv[1])]
fit = np.stack([np.asarray(json.loads(r["future_poses"]) if isinstance(r["future_poses"], str) else r["future_poses"], float)[:8]
                for r in rows if r["subset"] == "fit"])
k = int(sys.argv[2])
C, lab = kmeans(fit[:, :, :2].reshape(len(fit), -1), k)
head = np.stack([fit[lab == j, :, 2].mean(0) if (lab == j).any() else np.zeros(8) for j in range(k)])
out = np.vstack([np.concatenate([C.reshape(k, 8, 2), head[:, :, None]], -1), np.zeros((1, 8, 3))])
np.save(sys.argv[3], out)
msg = {"K": k, "fit_scenes": len(fit), "out": sys.argv[3], "shape": list(out.shape)}
if len(sys.argv) > 4:
    ref = np.load(sys.argv[4])
    msg["max_abs_diff_vs_" + sys.argv[4].split("/")[-1]] = float(np.abs(ref - out).max()) if ref.shape == out.shape else "shape differs"
print(json.dumps(msg))
