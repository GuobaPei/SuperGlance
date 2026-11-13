"""WOD-E2E readouts written as feature directories that superglance.eval.wod_submit build reads unchanged. From the
features of superglance.eval.features (tok_0.txt, code_0.npy, h_0.npy), the adapter's Policy Head and the candidate set:
  <out>_mbr02/   tok_0.txt + code_0.npy: the top-64 candidate with the best expected proxy RFS, the top 64 as raters of
                 score 10 weighted by the tau 0.2 softmax;
  <out>_mean02/  tok_0.txt + code_0.npy (0..N-1) + cands.npy (per frame the tau 0.2 softmax mean of the top 64's 8 poses, one
                 "candidate" per frame): the paper's readout.
With rated rows (validation), also prints their official RFS beside the argmax's.
usage: python -m superglance.eval.wod_decode <features dir> <adapter dir> <cands.npy> <rows dir> <out prefix>"""
import glob, json, os, shutil, sys
import numpy as np
from superglance.builders.pools.wod_proxy import proxy, to20

TOPK, TAU = 64, 0.2


def main():
    xd, ad, ca, rows, out = sys.argv[1:6]
    from safetensors import safe_open
    with safe_open(f"{ad}/adapter_model.safetensors", "pt") as f:
        Wt, b = [f.get_tensor(f"base_model.model.choice_head.{k}").float().numpy() for k in ("weight", "bias")]
    tok = open(f"{xd}/tok_0.txt").read().split(); code = np.load(f"{xd}/code_0.npy"); h = np.load(f"{xd}/h_0.npy").astype(np.float32)
    R = {}
    for p in sorted(glob.glob(f"{rows}/meta/*.jsonl")):
        for l in open(p):
            r = json.loads(l); R[r["token"]] = r
    v = np.array([R[t]["speed"] for t in tok]); C = np.load(ca)[:, :, :2].astype(np.float64); assert Wt.shape[0] == len(C), (Wt.shape, C.shape)
    L = h @ Wt.T + b; assert (L.argmax(1) == code).all(); top = np.argsort(-L, 1)[:, :TOPK]; lt = np.take_along_axis(L, top, 1)
    q = np.exp((lt - lt[:, :1]) / TAU); q /= q.sum(1, keepdims=True); X = to20(C)
    mbr = np.empty(len(tok), np.int64)
    for i in range(len(tok)):
        x = X[top[i]]; M = np.stack([proxy(x, x[k], v[i]) for k in range(TOPK)], 1)   # M[j, k]: candidate j with candidate k as the only rater
        assert np.allclose(np.diag(M), 10)
        mbr[i] = top[i, (M * q[i, None]).sum(1).argmax()]
    mean = (q[..., None, None] * C[top]).sum(1)
    for name, c, cc in ((f"{out}_mbr02", mbr, None), (f"{out}_mean02", np.arange(len(tok)), mean)):
        os.makedirs(name); shutil.copy(f"{xd}/tok_0.txt", name); np.save(f"{name}/code_0.npy", c)
        if cc is not None:
            np.save(f"{name}/cands.npy", cc)
    print(f"{len(tok)} frames: mbr0.2 changes argmax on {(mbr != code).sum()}", flush=True)
    if R[tok[0]].get("prefs"):
        from superglance.builders.pools.rater_feedback_utils import get_rater_feedback_score
        S = [R[t] for t in tok]
        def real(x):
            return get_rater_feedback_score(x[:, None], np.ones((len(S), 1)), [[np.array(p["traj"]) for p in r["prefs"]] for r in S], [np.array([p["score"] for p in r["prefs"]]) for r in S],
                                            v, frequency=4, length_seconds=5)["rater_feedback_score"]
        a, m1, m2 = real(X[code]), real(X[mbr]), real(to20(mean))
        print(f"real RFS: argmax {a.mean():.3f} (floor {(a <= 4 + 1e-9).sum()}), mbr0.2 {m1.mean():.3f} ({(m1 <= 4 + 1e-9).sum()}), mean0.2 {m2.mean():.3f} ({(m2 <= 4 + 1e-9).sum()})")
        for lo, hi in ((0, .5), (.5, 2), (2, 6), (6, 10), (10, 40)):
            k = (v >= lo) & (v < hi); print(f"  {lo}-{hi} m/s n {k.sum():3d}: argmax {a[k].mean():.3f}, mbr0.2 {m1[k].mean():.3f}, mean0.2 {m2[k].mean():.3f}")


if __name__ == "__main__":
    main()
