"""PAI-AV densification of a 4-s plan (8 poses at 0.5 s) to the log's 10 Hz over 5 s. lin: linear from the origin through
the poses, 4-5 s at the last segment's velocity (superglance.eval.pai_ade.to10). spline: a cubic spline through the origin and
the 8 poses, clamped at the logged ego velocity at 0 s and natural at 4 s, 4-5 s as lin; the paper's PAI-AV plans use it
(supplement, Benchmark protocols). ade: ADE at 3 s and 5 s against the logged future. boot: the mean of per-frame values with
its 95 % interval over 10,000 resamples of the logs. Importing the module runs its self-check (10 m/s straight is exact under
both)."""
import numpy as np
from scipy.interpolate import CubicSpline
from superglance.eval.pai_ade import T8, T10, to10

T9, LATE, TOPK = np.concatenate([[0.0], T8]), T10 > 4 + 1e-9, 64
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
