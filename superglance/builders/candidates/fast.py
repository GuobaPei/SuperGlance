"""The 1,024 high-speed candidates: NAVSIM's training logs have few scenes above 13 m/s, so the 4,097 clustered candidates miss
PAI-AV's highway speeds. k-means (k-means++ seeding, Lloyd until no centre moves 1 mm, seed 20261009) over the 4-s futures (x, y of the 8
poses) of the PAI-AV training rows whose mean speed over 4 s is at least vmin (13 m/s); a centre's headings are its members' mean. They
are appended after the 4,097 (rows 0-4,096 keep their meaning; asserted). Every 10th log (crc32 of the log name) is held out, and the
report gives its nearest-candidate ADE with the old and the new set, per speed band.
usage: python -m superglance.builders.candidates.fast <pai_inputs.jsonl> <cands_kmeans4096.npy> 1024 <cands_k4097_fast1024.npy> [vmin=13]"""
import json, math, sys, zlib
import numpy as np

inp, cf, k, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
vmin = float(sys.argv[5]) if len(sys.argv) > 5 else 13.0
old = np.load(cf)
F, Hd = [], []
for l in open(inp):
    try:
        r = json.loads(l)
    except ValueError:                                    # the last line of a file still being written
        continue
    fp = np.array(r["future_poses"], np.float64)
    if math.hypot(*fp[-1, :2]) / 4 >= vmin:
        (Hd if zlib.crc32(r["log_name"].encode()) % 10 == 0 else F).append(fp)
F, Hd = np.stack(F), np.stack(Hd)
X = F[:, :, :2].reshape(len(F), -1)
d2c = lambda A, C: (A ** 2).sum(1)[:, None] - 2 * A @ C.T + (C ** 2).sum(1)[None]
rng = np.random.default_rng(20261009)
C = [X[rng.integers(len(X))]]
d = d2c(X, C[0][None])[:, 0]
for _ in range(k - 1):
    C.append(X[rng.choice(len(X), p=np.clip(d, 0, None) / np.clip(d, 0, None).sum())])
    d = np.minimum(d, d2c(X, C[-1][None])[:, 0])
C = np.stack(C)
for it in range(100):
    a = d2c(X, C).argmin(1)
    N = np.stack([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
    moved = np.abs(N - C).max(); C = N
    if moved < 1e-3:
        break
a = d2c(X, C).argmin(1)
head = np.stack([F[a == j, :, 2].mean(0) for j in range(k)])
new = np.concatenate([C.reshape(k, 8, 2), head[:, :, None]], 2)
allc = np.concatenate([old.astype(np.float64), new]).astype(old.dtype)
assert allc.shape == (len(old) + k, 8, 3) and np.array_equal(allc[:len(old)], old)
np.save(out, allc)


def ade(P, cands):
    best = np.full(len(P), np.inf)
    for s in range(0, len(cands), 512):                   # mean pose distance, in chunks
        diff = P[:, None, :, :2] - cands[None, s:s + 512, :, :2]
        best = np.minimum(best, np.linalg.norm(diff, axis=-1).mean(-1).min(1))
    return best


sp = np.hypot(new[:, -1, 0], new[:, -1, 1]) / 4
print(f"{len(F)} fit rows / {len(Hd)} held-out rows at >= {vmin} m/s; {it + 1} k-means iterations; cluster sizes "
      f"min {np.bincount(a, minlength=k).min()} median {int(np.median(np.bincount(a, minlength=k)))}")
print(f"new candidates: mean speed {sp.min():.1f}-{sp.max():.1f} m/s (median {np.median(sp):.1f}); |y at 4 s| up to {np.abs(new[:, -1, 1]).max():.1f} m")
hv = np.hypot(Hd[:, -1, 0], Hd[:, -1, 1]) / 4
for name, cands in (("old 4,097", old.astype(np.float64)), (f"old + {k}", allc.astype(np.float64))):
    e = ade(Hd, cands)
    print(f"held-out rows, nearest-candidate ADE with {name}: median {np.median(e):.2f} m, mean {e.mean():.2f} m, > 2 m {100 * (e > 2).mean():.1f}%; by speed "
          + ", ".join(f"{lo}-{hi} {np.median(e[(hv >= lo) & (hv < hi)]):.2f}" for lo, hi in ((13, 16), (16, 20), (20, 25), (25, 30), (30, 60)) if ((hv >= lo) & (hv < hi)).any()))
for lo, hi in ((13, 16), (16, 20), (20, 25), (25, 30), (30, 60)):
    m = (sp >= lo) & (sp < hi)
    print(f"new candidates at {lo}-{hi} m/s: {m.sum()} ({(np.abs(new[m, -1, 1]) < 4).sum()} with |y at 4 s| < 4 m)")
print("saved", out, allc.shape)
