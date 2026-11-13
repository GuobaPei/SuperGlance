"""The plan from the Policy Head's logits, read three ways over the head's top 64 at temperature 0.2: argmax; mean = the
softmax-weighted mean of the 8 poses (the paper's readout); medoid = the candidate with the least expected ADE to the others.
  prep   features (superglance.eval.features) + adapter + candidate library -> <out.npz> (tok, arg, mean, med, p1)
  score  NAVSIM PDM scores of the three plans per scene, from each scene's metric cache (the scene export's metric_cache);
         run in the NAVSIM environment
usage: python -m superglance.eval.readout prep <name> <features dir> <adapter> <cands.npy> <out.npz>
       python -m superglance.eval.readout score <prep.npz> <scenes.jsonl> <out.npz>"""
import json, sys
import numpy as np
from superglance.eval.heads import load_h, head

TOPK, TAU = 64, 0.2


def prep():
    name, xd, ad, ca, out = sys.argv[2:7]; tok, _, h = load_h(xd); W, b = head(ad); C = np.load(ca).astype(np.float64); L = h @ W.T + b
    top = np.argsort(-L, 1)[:, :TOPK]; lt = np.take_along_axis(L, top, 1); q = np.exp((lt - lt[:, :1]) / TAU); q /= q.sum(1, keepdims=True)
    x = C[top]; mean = (q[..., None, None] * x).sum(1); med = np.empty((len(tok), 8, 3))
    for i in range(0, len(tok), 256):
        X, Q = x[i:i + 256], q[i:i + 256]; D = np.linalg.norm(X[:, :, None, :, :2] - X[:, None, :, :, :2], axis=-1).mean(-1)
        med[i:i + 256] = X[np.arange(len(X)), (D * Q[:, None]).sum(-1).argmin(1)]
    p = np.exp(L - L.max(1, keepdims=True)); p1 = 1 / p.sum(1)   # the top candidate's softmax probability
    print(f"{name}: {len(tok)} scenes, top-1 probability median {np.median(p1):.4f}; medoid equals argmax in {np.mean(np.abs(med - x[:, 0]).max((1, 2)) < 1e-9):.3f}")
    np.savez(out, tok=np.array(tok), arg=x[:, 0], mean=mean, med=med, p1=p1)


def _score(a):
    s, bad = PB.score_pairs(PB.load_cache(a[0]), a[1])
    return s, bad


def score():
    global PB
    from multiprocessing import Pool
    from superglance.executor import pdm as PB
    Z = np.load(sys.argv[2]); scn, out = sys.argv[3:5]; tok = [str(t) for t in Z["tok"]]; mc = {r["token"]: r["metric_cache"] for r in map(json.loads, open(scn))}
    arg, mean, med = Z["arg"], Z["mean"], Z["med"]
    with Pool(40) as p:
        res = p.map(_score, [(mc[t], np.stack([arg[i], mean[i], med[i]])) for i, t in enumerate(tok)], chunksize=8)
    A = np.stack([s for s, _ in res]); logs = np.array([mc[t].split("/")[-4] for t in tok]); u, g = np.unique(logs, return_inverse=True)
    print(f"nonfinite {sum(b for _, b in res)}")
    r = np.random.default_rng(0).integers(0, len(u), (2000, len(u)))
    for j, k in enumerate(("argmax", "mean", "medoid")):
        m = 100 * A[:, j].mean(0); d = 100 * (A[:, j, 6] - A[:, 0, 6]); s = np.bincount(g, d); c = np.bincount(g); bs = s[r].sum(1) / c[r].sum(1)
        print(f"{k}: NC {m[0]:.2f} DAC {m[1]:.2f} EP {m[2]:.2f} TTC {m[3]:.2f} C {m[4]:.2f} DDC {m[5]:.2f} PDMS {m[6]:.2f}; vs argmax {d.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]", flush=True)
    p1 = Z["p1"]; e = np.quantile(p1, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(e, p1, side="right") - 1, 0, 9); P = 100 * A[:, 0, 6]
    print("confidence deciles (top-1 probability range: mean PDMS, NC, DAC, TTC of the argmax):")
    for i in range(10):
        s = b == i; print(f"  {e[i]:.4f}-{e[i + 1]:.4f}: PDMS {P[s].mean():.2f}  NC {100 * A[s, 0, 0].mean():.2f}  DAC {100 * A[s, 0, 1].mean():.2f}  TTC {100 * A[s, 0, 3].mean():.2f}  EP {100 * A[s, 0, 2].mean():.2f}")
    rk = np.argsort(np.argsort(p1)); rp = np.argsort(np.argsort(P)); print(f"Spearman(top-1 probability, PDMS) {np.corrcoef(rk, rp)[0, 1]:+.3f}")
    np.savez(out, tok=Z["tok"], scores=A.astype(np.float32), p1=p1)


if __name__ == "__main__":
    {"prep": prep, "score": score}[sys.argv[1]]()
