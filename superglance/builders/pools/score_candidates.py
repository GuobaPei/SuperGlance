"""NAVSIM rewards and soft targets: every candidate trajectory (K x 8 x 3 ego-frame poses at 2 Hz) scored on every scene of an
input export with the stock NAVSIM v1.1 PDM scorer (superglance.executor.pdm.score_pairs), in shards of 1,000 scenes (a rerun skips
finished shards), then merged into
  scores.npz  tokens, subset, subset_names, cands, scores (scenes x K x [NC, DAC, EP, TTC, C, DDC, PDMS], float16),
              ade (scenes x K, mean waypoint distance to the logged trajectory, float16), nonfinite
  q_t01b01.npz / q_t003b002.npz  soft targets: tokens, subset, subset_names, ok, q (softmax((PDMS x DDC - beta ADE) / tau), float16),
              r (PDMS x DDC, float16), tau, beta, cands
usage (navsim env, NAVSIM metric caches readable):
  python -m superglance.builders.pools.score_candidates <cands.npy> <navtrain_inputs.jsonl> <out dir> [procs]"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SHARD = 1000
CANDS = None


def init(path):
    global CANDS
    CANDS = np.load(path).astype(np.float64)


def score(job):
    from superglance.executor import pdm as pb
    tok, mc = job
    s, nbad = pb.score_pairs(pb.load_cache(mc), CANDS)
    return tok, s.astype(np.float16), nbad


def main():
    cands_path, inputs, outdir = sys.argv[1:4]
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 48
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    cands = np.load(cands_path)
    assert cands.ndim == 3 and cands.shape[1:] == (8, 3), cands.shape
    rows = [json.loads(l) for l in open(inputs)]
    shards = [rows[s:s + SHARD] for s in range(0, len(rows), SHARD)]
    with Pool(procs, initializer=init, initargs=(cands_path,)) as pool:
        for i, sh in enumerate(shards):
            f = out / f"shard_{i:03d}.npz"
            if f.exists():
                continue
            res = {t: (s, b) for t, s, b in pool.imap_unordered(score, [(r["token"], r["metric_cache"]) for r in sh], chunksize=1)}
            toks = [r["token"] for r in sh]
            np.savez(str(f) + ".part.npz", tokens=np.array(toks), scores=np.stack([res[t][0] for t in toks]),
                     nonfinite=np.array([res[t][1] for t in toks]))
            Path(str(f) + ".part.npz").rename(f)
            print(f"shard {i + 1}/{len(shards)} done", flush=True)
    toks, S, nf = [], [], []
    for i in range(len(shards)):
        with np.load(out / f"shard_{i:03d}.npz") as z:
            toks += [str(t) for t in z["tokens"]]; S.append(z["scores"]); nf.append(z["nonfinite"])
    S = np.concatenate(S); nf = np.concatenate(nf)
    assert toks == [r["token"] for r in rows]
    gt = np.stack([np.asarray(json.loads(r["future_poses"]) if isinstance(r["future_poses"], str) else r["future_poses"], float)[:8, :2]
                   for r in rows])
    ade = np.concatenate([np.linalg.norm(cands[None, :, :, :2] - gt[s:s + 2048, None], axis=-1).mean(-1)
                          for s in range(0, len(gt), 2048)])
    subset_names = np.array(["fit", "heldout"])
    subset = np.array([0 if r["subset"] == "fit" else 1 for r in rows], np.int8)
    np.savez(out / "scores.part.npz", tokens=np.array(toks).astype("S16"), subset=subset, subset_names=subset_names, cands=cands,
             scores=S, ade=ade.astype(np.float16), nonfinite=nf)
    (out / "scores.part.npz").rename(out / "scores.npz")
    ok = np.isfinite(S).all((1, 2))
    for tau, beta, name in ((0.1, 0.1, "q_t01b01"), (0.03, 0.02, "q_t003b002")):
        Q, Rr, top, best = [], [], [], []
        for s in range(0, len(S), 4096):  # chunks: a float64 copy of all K = 4,097 scores would be ~24 GB
            Sc = S[s:s + 4096].astype(np.float64); r = Sc[:, :, 6] * Sc[:, :, 5]
            v = np.where(ok[s:s + 4096, None], r - beta * ade[s:s + 4096], 0.0)
            w = np.exp((v - v.max(1, keepdims=True)) / tau); q = w / w.sum(1, keepdims=True)
            Q.append(q.astype(np.float16)); Rr.append(r.astype(np.float16))
            top.append(Sc[np.arange(len(q)), q.argmax(1), 6]); best.append(Sc[:, :, 6].max(1))
        q = np.concatenate(Q)
        np.savez_compressed(out / f"{name}.part.npz", tokens=np.array(toks).astype("S16"), subset=subset, subset_names=subset_names,
                            ok=ok, q=q, r=np.concatenate(Rr), tau=tau, beta=beta, cands=cands)
        (out / f"{name}.part.npz").rename(out / f"{name}.npz")  # q_t003b002.npz last: the runner's "done" marker
        print(json.dumps({name: {"K": int(cands.shape[0]), "scenes": len(toks), "ok": int(ok.sum()),
                                 "mean_top_prob": round(float(q.astype(np.float64).max(1).mean()), 4),
                                 "best_pdms_mean": round(float(100 * np.concatenate(best).mean()), 2),
                                 "target_top_pdms_mean": round(float(100 * np.concatenate(top).mean()), 2),
                                 "nonfinite_proposals": int(nf.sum())}}), flush=True)


if __name__ == "__main__":
    main()
