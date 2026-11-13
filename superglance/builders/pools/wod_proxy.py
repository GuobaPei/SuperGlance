"""A proxy of the official WOD-E2E RFS for frames without raters: the logged future is the only rater trajectory, score 10.
Vectorized over the candidates, the same formula as rater_feedback_utils.get_rater_feedback_score; the __main__ check holds it to
the official function on rated val rows. Soft target for SFT rows: softmax(proxy / 0.5 - (ADE - min ADE) / 1 m).
usage (import): X = to20(C); s = proxy(X, f, v); t = target(s, ade)
check: python -m superglance.builders.pools.wod_proxy <rows dir> <cands.npy>"""
import glob, json, sys
import numpy as np

T8, T20 = np.arange(1, 9) * 0.5, np.arange(1, 21) * 0.25
IDX = np.array([11, 19]); LAT = np.array([1.0, 1.8]); LNG = LAT * 4


def to20(p8):   # [K, 8, 2] at 0.5-4 s -> [K, 20, 2] at 0.25-5 s; wod_rfs.to20, vectorized
    t = np.concatenate([[0.0], T8]); p = np.concatenate([np.zeros((len(p8), 1, 2)), p8], 1)
    i = np.clip(np.searchsorted(t, T20) - 1, 0, 7); w = (T20 - t[i]) / 0.5
    out = p[:, i] * (1 - w)[None, :, None] + p[:, i + 1] * w[None, :, None]
    late = T20 > 4; out[:, late] = p8[:, 7:8] + (p8[:, 7:8] - p8[:, 6:7]) / 0.5 * (T20[late] - 4)[None, :, None]
    return out


def proxy(X, f, v):   # X [K, 20, 2], f [20, 2], v m/s -> [K] scores in [~0, 10]; floor 4 outside the trust region as official
    d = np.diff(np.vstack([[0.0, 0.0], f]), axis=0); n = np.linalg.norm(d, axis=1)
    if n[0] == 0:
        d[0] = (1.0, 0.0)
    for t in range(1, 20):
        if n[t] == 0:
            d[t] = d[t - 1]
    u = d[IDX] / np.linalg.norm(d[IDX], axis=1, keepdims=True); e = X[:, IDX] - f[IDX]
    lng = np.abs((e * u).sum(-1)); lat = np.abs(e[..., 0] * -u[:, 1] + e[..., 1] * u[:, 0])
    sc = np.clip(0.5 + 0.5 * (v - 1.4) / 9.6, 0.5, 1.0); m = np.maximum(lng / (sc * LNG), lat / (sc * LAT))
    s = (10.0 * 0.1 ** np.maximum(m - 1, 0)).mean(1)
    return np.where((m <= 1).all(1), s, np.maximum(s, 4.0))


def target(s, ade, tp=0.5, ta=1.0):   # the soft SFT target over K candidates
    q = s / tp - (ade - ade.min()) / ta; q = np.exp(q - q.max())
    return q / q.sum()


if __name__ == "__main__":
    from superglance.builders.pools.rater_feedback_utils import get_rater_feedback_score
    R = [json.loads(l) for p in sorted(glob.glob(f"{sys.argv[1]}/meta/*.jsonl")) for l in open(p)]; R = [r for r in R if r["prefs"]][:200]
    X = to20(np.load(sys.argv[2])[:, :, :2].astype(np.float64)); K = len(X)
    for r in R:
        f = np.array(r["future"], float); a = proxy(X, f, r["speed"])
        b = get_rater_feedback_score(X[:, None], np.ones((K, 1)), [[f]] * K, [np.array([10.0])] * K, np.full(K, r["speed"]), frequency=4, length_seconds=5)["rater_feedback_score"]
        assert np.allclose(a, b), (r["token"], np.abs(a - b).max())
    print(f"proxy == official on {len(R)} frames x {K} candidates")
