"""The readout's number of candidates k and temperature on PAI-AV (appendix, Benchmark protocols): per features dir and k,
ADE at 3 s / 5 s of the temperature-weighted mean over the head's top k, densified by the cubic spline, paired against k = 64
and against the argmax with the spline, 95 % intervals over logs (10,000 resamples, seed 0). The appendix's k sweep runs
<tau> 0.2 on the validation rows; its temperature choice compares k = 64 at 1, 0.5, 0.2, 0.1, 0.05 and 0.02.
Checks (assert): k = 1 is the argmax candidate on every frame; k = 64 equals the plain top-64 mean on every frame.
usage: python -m superglance.eval.pai_topk <rows.jsonl> <out.json> <tau> <k,k,...,all> <name>=<features dir>=<cands.npy> ..."""
import json, sys
import numpy as np
from scipy.interpolate import CubicSpline
from superglance.eval.pai_ade import T8, T10, to10
from superglance.eval.heads import head

T9, LATE = np.concatenate([[0.0], T8]), T10 > 4 + 1e-9
E = np.zeros((8, 8, 2)); E[np.arange(8), np.arange(8), 0] = 1
ML = to10(E)[:, :, 0].T                                   # today's densification as a [50, 8] matrix


def spl(p8, v0):
    return CubicSpline(T9, np.concatenate([[0.0], p8]), bc_type=((1, v0), (2, 0.0)))(T10)


MS = np.stack([spl(np.eye(8)[k], 0.0) for k in range(8)], 1); MV = spl(np.zeros(8), 1.0)
MS[LATE], MV[LATE] = ML[LATE], 0.0


def lin(x8):
    return np.einsum("tk,...kd->...td", ML, x8)


def spline(x8, v0):
    return np.einsum("tk,...kd->...td", MS, x8) + MV[:, None] * v0[:, None, :]


p8 = np.cumsum(np.full((1, 8, 2), [5.0, 0.0]), 1)          # self-check: 10 m/s straight is exact under both
assert np.allclose(lin(p8), to10(p8)) and np.allclose(spline(p8, np.array([[10.0, 0.0]]))[0, :, 0], 10 * T10)


def ade(traj, F):
    d = np.linalg.norm(traj - F, axis=-1)
    return d[:, :30].mean(-1), d.mean(-1)


def boot(d, g, n=10000):   # mean of per-frame d, 95 % CI resampling logs
    keys = sorted(set(g)); ix = {k: i for i, k in enumerate(keys)}; s = np.zeros(len(keys)); c = np.zeros(len(keys))
    np.add.at(s, [ix[k] for k in g], d); np.add.at(c, [ix[k] for k in g], 1)
    r = np.random.default_rng(0).integers(0, len(keys), (n, len(keys)))
    m = s[r].sum(1) / c[r].sum(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def tau_mean(L, order, C, k, tau):   # the tau softmax-weighted mean of the top k (float32 weights, as the one-plan readout)
    top = order[:, :k]; lt = np.take_along_axis(L, top, 1)
    q = np.exp((lt - lt[:, :1]) / tau); q /= q.sum(1, keepdims=True)
    mean = np.empty((len(L), 8, 2))
    for i in range(0, len(L), 100):
        mean[i:i + 100] = (q[i:i + 100, :, None, None] * C[top[i:i + 100]]).sum(1)
    return mean, q[:, 0]


def main():
    R = [json.loads(l) for l in open(sys.argv[1])]; byt = {r["token"]: r for r in R}
    out, tau = {"rows": sys.argv[1], "tau": float(sys.argv[3]), "models": {}}, float(sys.argv[3])
    for arg in sys.argv[5:]:
        name, xd, ca = arg.split("=")
        C = np.load(ca)[:, :, :2].astype(np.float64); W, b = head(open(f"{xd}/adapter.txt").read().strip())
        assert W.shape[0] == len(C), (name, W.shape, C.shape)   # the head's width is the library's K
        ks = [len(C) if x == "all" else int(x) for x in sys.argv[4].split(",")]
        tok, code, h = [], [], []
        for k in range(64):
            try:
                t = open(f"{xd}/tok_{k}.txt").read().split()
            except FileNotFoundError:
                continue
            tok += t; code.append(np.load(f"{xd}/code_{k}.npy")); h.append(np.load(f"{xd}/h_{k}.npy"))
        code, h = np.concatenate(code), np.concatenate(h).astype(np.float32)
        S = [byt[t] for t in tok]; F = np.array([r["fut10"] for r in S]); v0 = np.array([r["ego_velocity"] for r in S], float)
        g = [r["log_name"] for r in S]
        L = h @ W.T + b; order = np.argsort(-L, 1)
        assert (order[:, 0] == code).all(), f"{name}: argmax from h = code on {(order[:, 0] == code).mean():.4f} of frames"
        # k = 64 by the plain top-64 mean's own lines
        top = np.argsort(-L, 1)[:, :64]; lt = np.take_along_axis(L, top, 1)
        q = np.exp((lt - lt[:, :1]) / tau); q /= q.sum(1, keepdims=True)
        ref64 = np.empty((len(S), 8, 2))
        for i in range(0, len(S), 100):
            ref64[i:i + 100] = (q[i:i + 100][..., None, None] * C[top[i:i + 100]]).sum(1)
        base = ade(spline(C[code], v0), F)   # P1: argmax + spline
        res = {k: tau_mean(L, order, C, k, tau) for k in sorted(set(ks + [1, 64]))}
        assert np.array_equal(res[64][0], ref64), f"{name}: k = 64 differs from the plain top-64 mean"
        assert np.array_equal(res[1][0], C[code]), f"{name}: k = 1 is not the argmax candidate"
        a = {k: ade(spline(m, v0), F) for k, (m, _) in res.items()}
        print(f"== {name}: n {len(S)}, logs {len(set(g))}, K {len(C)}, tau {tau}; P1 (argmax, spline) ADE@3 {base[0].mean():.4f} "
              f"@5 {base[1].mean():.4f}; checks: k = 1 = argmax on all frames, k = 64 = the plain top-64 mean on all frames")
        M = {"n": len(S), "logs": len(set(g)), "K": len(C), "extract": xd, "cands": ca,
             "P1": {"ade3": float(base[0].mean()), "ade5": float(base[1].mean())}, "k": {}}
        for k in ks:
            r = {"ade3": float(a[k][0].mean()), "ade5": float(a[k][1].mean()), "top1_weight_median": float(np.median(res[k][1]))}
            for ref, (b3, b5) in (("k64", a[64]), ("P1", base)):
                for lab, d in (("d3", a[k][0] - b3), ("d5", a[k][1] - b5)):
                    r[f"{lab}_vs_{ref}"] = [float(x) for x in boot(d, g)]
            M["k"][str(k)] = r
            print(f"  k {k:5d}  top-1 weight med {r['top1_weight_median']:.3f}  ADE@3 {r['ade3']:.4f} @5 {r['ade5']:.4f}  vs k 64: "
                  "d3 {:+.4f} [{:+.4f}, {:+.4f}]  d5 {:+.4f} [{:+.4f}, {:+.4f}]".format(*r["d3_vs_k64"], *r["d5_vs_k64"])
                  + "  vs P1: d3 {:+.4f} [{:+.4f}, {:+.4f}]".format(*r["d3_vs_P1"]))
        out["models"][name] = M
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
