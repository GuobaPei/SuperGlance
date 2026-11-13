"""PDMS soft targets of the navtrain rows: q = softmax((PDMS x DDC - 0.1 ADE) / 0.1) over the 5,121 candidates (ADE = mean
distance over the 8 poses at 0.5-4 s to the logged future), for the navtrain fit rows and the navtrain shard's 200 dev rows, from
superglance.builders.pools.score_candidates' scores.npz (every navtrain scene, one PDM scoring). -> <out.npz>: tokens (fit, then
dev), split (0 fit, 1 dev), q float32 [N, 5121], gold = argmax q. Prints q's expected and argmax PDMS x DDC over the fit rows, next
to the imitation target's argmax (the candidate nearest the logged future).
usage: python -m superglance.builders.shards.pdms_targets <scores.npz> <navtrain shard dev.jsonl | -> <cands.npy>
           <navtrain_inputs.jsonl> <out.npz>"""
import json, sys
import numpy as np

fit_p, dev_p, cands_p, inputs_p, out = sys.argv[1:6]
C = np.load(cands_p)[:, :, :2].astype(np.float64); K = len(C)
fut = {}
for l in open(inputs_p):
    r = json.loads(l)
    fut[r["token"]] = np.asarray(r["future_poses"], np.float64)[:8, :2]
tk = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def targets(toks, S, tau=0.1, beta=0.1):
    Q = np.empty((len(toks), K), np.float32); rep = {k: [] for k in ("exp_r", "top_r", "top_pdms", "top_p", "im_r", "im_pdms", "best_pdms")}
    for s in range(0, len(toks), 2048):
        Sc = S[s:s + 2048].astype(np.float64); assert np.isfinite(Sc).all()
        r = Sc[:, :, 6] * Sc[:, :, 5]
        ade = np.linalg.norm(C[None] - np.stack([fut[t] for t in toks[s:s + 2048]])[:, None], axis=-1).mean(-1)
        v = r - beta * ade; w = np.exp((v - v.max(1, keepdims=True)) / tau); q = w / w.sum(1, keepdims=True)
        Q[s:s + 2048] = q; i = np.arange(len(q)); g = q.argmax(1); n = ade.argmin(1)
        rep["exp_r"].append((q * r).sum(1)); rep["top_r"].append(r[i, g]); rep["top_pdms"].append(Sc[i, g, 6]); rep["top_p"].append(q[i, g])
        rep["im_r"].append(r[i, n]); rep["im_pdms"].append(Sc[i, n, 6]); rep["best_pdms"].append(Sc[:, :, 6].max(1))
    return Q, {k: round(100 * float(np.concatenate(v).mean()), 2) if k != "top_p" else round(float(np.concatenate(v).mean()), 4) for k, v in rep.items()}


z = np.load(fit_p); at = tk(z["tokens"]); names = tk(z["subset_names"]); sub = [names[int(s)] for s in z["subset"]]
fi = [i for i, s in enumerate(sub) if s == "fit"]; ft = [at[i] for i in fi]; Sf = z["scores"][fi]; assert Sf.shape == (len(ft), K, 7)
Qf, rf = targets(ft, Sf); dt, Qd, rd, dev_c4_maxdiff = [], np.zeros((0, K), np.float32), None, None
if dev_p != "-":   # "-": fit rows only
    ix = {t: i for i, t in enumerate(at)}; dt = [json.loads(l)["token"] for l in open(dev_p)]
    assert not set(ft) & set(dt)
    Qd, rd = targets(dt, z["scores"][[ix[t] for t in dt]])
Q = np.concatenate([Qf, Qd]); s = Q.astype(np.float64).sum(1)
np.savez(out, tokens=np.array(ft + dt), split=np.array([0] * len(ft) + [1] * len(dt), np.int8), q=Q, gold=Q.argmax(1).astype(np.int32),
         tau=0.1, beta=0.1, label_source="pdm_scored_5121_softmax_t0.1_b0.1")
print(json.dumps({"fit": len(ft), "dev": len(dt), "K": K, "q_sum_abs_err_max": float(np.abs(s - 1).max()), "dev_first4097_vs_cand4096_maxdiff": dev_c4_maxdiff,
                  "fit_report": rf, "dev_report": rd}))
