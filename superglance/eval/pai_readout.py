"""Table 6's one-plan cells: PAI-AV ADE at 3 s / 5 s to six decimals of the top-64 mean at temperature 0.2, densified by the
cubic spline from the logged velocity (superglance.eval.pai_decode), beside the argmax densified linearly. A features dir holds
superglance.eval.features' output and adapter.txt (the adapter's path), as scripts/eval_pai.sh writes them.
usage: python -m superglance.eval.pai_readout <rows jsonl> <features dir>=<cands .npy> ..."""
import json, os, sys
import numpy as np
from superglance.eval.heads import load_h, head  # noqa: E402
from superglance.eval.pai_decode import lin, spline, ade  # noqa: E402
R = [json.loads(l) for l in open(sys.argv[1])]; tok = [r["token"] for r in R]
F = np.array([r["fut10"] for r in R]); v0 = np.array([r["ego_velocity"] for r in R], float)
for a in sys.argv[2:]:
    xr, cp = a.split("=")
    C = np.load(cp)[:, :, :2].astype(np.float64)
    ta, _, h = load_h(xr); ia = {t: i for i, t in enumerate(ta)}; Wt, b = head(open(f"{xr}/adapter.txt").read().strip())
    assert Wt.shape[0] == len(C)
    L = h[[ia[t] for t in tok]] @ Wt.T + b; top = np.argsort(-L, 1)[:, :64]; lt = np.take_along_axis(L, top, 1)
    q = np.exp((lt - lt[:, :1]) / 0.2); q /= q.sum(1, keepdims=True)
    am = ade(lin(C[top[:, 0]]), F); mn = ade(spline((q[..., None, None] * C[top]).sum(1), v0), F)
    print(f"{os.path.basename(xr)}: argmax {am[0].mean():.6f} / {am[1].mean():.6f}; mean {mn[0].mean():.6f} / {mn[1].mean():.6f}")
