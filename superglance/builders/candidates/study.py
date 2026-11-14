"""Coverage of NAVSIM scenes by k-means candidate sets: k-means on the human 4-s trajectories of the fit scenes (8 waypoints x, y;
heading = the cluster's mean), K = 256 / 1,024 / 4,096, each plus a standing-still trajectory. Every candidate is scored on the given
held-out scenes with the NAVSIM v1.1 PDM scorer (superglance.executor.pdm.score_pairs). Per set: the mean distance between neighbouring
candidates (the appendix's "Spacing"), and the mean over scenes of the PDMS of the best candidate, of the candidate nearest the human
trajectory (mean waypoint distance), of the first choice of the target argmax(PDMS x DDC - 0.1 ADE) and of that choice's 3 nearest
candidates. superglance.builders.candidates.coverage gives the table's best and nearest columns on all 10,549 held-out scenes.
usage (NAVSIM environment): python -m superglance.builders.candidates.study <navtrain_inputs.jsonl> <codebook dir | -> <scene tokens>
       <out dir> [procs]   ("-": no earlier 16 x 16 codebook)"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np



def assign(X, C):
    c2 = (C ** 2).sum(1)
    return np.concatenate([(c2[None] - 2 * X[s:s + 8192] @ C.T).argmin(1) for s in range(0, len(X), 8192)])


def kmeans(X, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    sample = X[rng.choice(len(X), min(len(X), 20000), replace=False)]
    C = sample[rng.choice(len(sample), 1)]
    d = ((sample - C[0]) ** 2).sum(1)
    for _ in range(1, k):  # k-means++ seeding on a sample
        C = np.vstack([C, sample[rng.choice(len(sample), p=d / d.sum())]])
        d = np.minimum(d, ((sample - C[-1]) ** 2).sum(1))
    for _ in range(iters):  # Lloyd; an empty cluster keeps its centre
        lab = assign(X, C)
        sums = np.zeros_like(C); np.add.at(sums, lab, X)
        cnt = np.bincount(lab, minlength=k)
        C = np.where(cnt[:, None] > 0, sums / np.maximum(cnt, 1)[:, None], C)
    return C, assign(X, C)


def score_scene(job):
    from superglance.executor import pdm as pdms_batch
    tok, mc_path, cands = job
    out, nbad = pdms_batch.score_pairs(pdms_batch.load_cache(mc_path), cands)
    return tok, out, nbad


def main():
    inputs, cbdir, held, outdir = sys.argv[1:5]
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 48
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for line in open(inputs):
        r = json.loads(line)
        f = r["future_poses"]
        rows[r["token"]] = (r["subset"], np.asarray(json.loads(f) if isinstance(f, str) else f, float)[:8], r["metric_cache"])
    fit = np.stack([v[1] for v in rows.values() if v[0] == "fit"])  # n x 8 x 3
    sets = {}
    if cbdir != "-":   # the earlier 16 x 16 codebook, scored beside the k-means sets when its directory is given
        meta = json.load(open(Path(cbdir) / "codebooks.json"))
        L1, L2 = np.load(Path(cbdir) / "rvq_L1.npy"), np.load(Path(cbdir) / "rvq_L2.npy")
        V = (L1[:, None, :] + L2[None, :, :]).reshape(256, -1)
        sets["codebook 16x16"] = np.stack([V[:, :8], V[:, 8:16], V[:, 16:24] / meta["S_HEAD"]], -1)
    for k in (256, 1024, 4096):
        f = out / f"kmeans{k}.npy"
        if not f.exists():
            C, lab = kmeans(fit[:, :, :2].reshape(len(fit), -1), k)
            head = np.stack([fit[lab == j, :, 2].mean(0) if (lab == j).any() else np.zeros(8) for j in range(k)])
            np.save(f, np.vstack([np.concatenate([C.reshape(k, 8, 2), head[:, :, None]], -1), np.zeros((1, 8, 3))]))
        sets[f"k-means {k} + still"] = np.load(f)
    names = list(sets)
    allc = np.concatenate([sets[n] for n in names]); bounds = np.cumsum([0] + [len(sets[n]) for n in names])
    toks = [t.strip() for t in open(held) if t.strip()]
    f = out / "scores.npz"
    if not f.exists():
        res = {}
        with Pool(procs) as p:
            for i, (tok, sc, nbad) in enumerate(p.imap_unordered(score_scene, [(t, rows[t][2], allc) for t in toks], chunksize=1)):
                res[tok] = sc
                if i % 200 == 0:
                    print(f"scored {i + 1}/{len(toks)} scenes", flush=True)
        np.savez_compressed(f, tokens=np.array(toks), scores=np.stack([res[t] for t in toks]).astype(np.float32), bounds=bounds)
    S = np.load(f)["scores"].astype(float)  # scenes x candidates x [NC, DAC, EP, TTC, C, DDC, PDMS]
    gt = np.stack([rows[t][1][:, :2] for t in toks])
    print("| candidate set | count | mean distance to nearest candidate | best | nearest the human | old target's first choice | its 3 nearest (one step off) |")
    print("|---|---|---|---|---|---|---|")
    for n, a, b in zip(names, bounds[:-1], bounds[1:]):
        P = sets[n][:, :, :2]
        D = np.linalg.norm(P[:, None] - P[None], axis=-1).mean(-1)
        nn3 = np.argsort(D, 1)[:, 1:4]
        pd, ddc = S[:, a:b, 6], S[:, a:b, 5]
        ade = np.linalg.norm(P[None] - gt[:, None], axis=-1).mean(-1)  # scenes x candidates
        i = np.arange(len(toks))
        top = (pd * ddc - 0.1 * ade).argmax(1)
        print(f"| {n} | {b - a} | {np.sort(D, 1)[:, 1].mean():.2f} m | {100 * pd.max(1).mean():.2f} | {100 * pd[i, ade.argmin(1)].mean():.2f} | "
              f"{100 * pd[i, top].mean():.2f} | {100 * pd[i[:, None], nn3[top]].mean():.2f} |")


if __name__ == "__main__":
    main()
