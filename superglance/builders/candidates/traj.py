"""Rule-built 4-s trajectories in the library's format (8 poses at 0.5-4 s: x ahead, y left, heading;
ego frame, origin = the pose at 0 s). A trajectory = a speed profile s(t) along a planar path with curvature kappa(s):
  speed    constant acceleration acc from v0 (acc < 0 brakes; at v = 0 the car stops and holds the position and heading)
  path     straight | arc (kappa = A) | clo (clothoid entry: kappa = A s / S, S = s(4 s)) | ramp (kappa = A min(1, 2 s / S): clothoid
           entry over the first half, then the arc)
  heading  the path tangent (integrated with the positions, so it is consistent with the motion)
For curved shapes the amplitude A is solved by bisection so that y(4 s) equals a target lateral offset Y (|Y| < s(4) / 1.5)."""
import numpy as np

T = 0.5 * np.arange(1, 9)


def arclen(v0, acc, t=T):
    if acc < 0:
        tt = np.minimum(t, v0 / -acc)
        return v0 * tt + 0.5 * acc * tt ** 2
    return v0 * t + 0.5 * acc * t ** 2


def path(s, shape, A, S, ds=0.02):   # poses [len(s), 3] at the arc lengths s
    g = np.arange(0.0, max(float(np.max(s)), ds) + 2 * ds, ds)
    if shape == "straight" or A == 0:
        k = np.zeros_like(g)
    elif shape == "arc":
        k = np.full_like(g, A)
    elif shape == "clo":
        k = A * g / S
    elif shape == "ramp":
        k = A * np.minimum(1.0, 2 * g / S)
    else:
        raise ValueError(shape)
    th = np.concatenate([[0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * ds)])
    x = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(th[1:]) + np.cos(th[:-1])) * ds)])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(th[1:]) + np.sin(th[:-1])) * ds)])
    return np.stack([np.interp(s, g, x), np.interp(s, g, y), np.interp(s, g, th)], -1)


def traj(v0, acc=0.0, shape="straight", Y=None, A=None):
    """Y: target y(4 s) (solved), or A: the curvature amplitude directly (1/m)"""
    s = arclen(v0, acc); S = float(s[-1])
    if shape == "straight":
        return path(s, "straight", 0.0, S)
    if A is None:
        assert abs(Y) < S / 1.5, (v0, acc, shape, Y, S)
        lo, hi = 0.0, 3.0 / max(S, 1e-6)
        for _ in range(60):   # y(4 s) grows with A for these heading ranges
            mid = 0.5 * (lo + hi)
            if path(s, shape, mid, S)[-1, 1] < abs(Y):
                lo = mid
            else:
                hi = mid
        A = np.sign(Y) * 0.5 * (lo + hi)
    return path(s, shape, A, S)


def selftest():
    P = traj(10.0, -2.5)   # stops at 4 s after 20 m
    assert abs(P[-1, 0] - 20.0) < 1e-6 and np.allclose(P[:, 1:], 0)
    P = traj(12.0, -4.0)   # stops at 3 s after 18 m, holds
    assert np.allclose(P[-3:, 0], 18.0, atol=1e-6)
    for shape in ("arc", "clo", "ramp"):
        P = traj(12.0, 0.0, shape, Y=-9.0)
        assert abs(P[-1, 1] + 9.0) < 1e-3, (shape, P[-1])
        d = np.diff(np.concatenate([[[0, 0]], P[:, :2]]), axis=0); mot = np.arctan2(d[:, 1], d[:, 0])
        mid = 0.5 * (np.concatenate([[0], P[:-1, 2]]) + P[:, 2])   # chord direction ~ mean of the end headings
        assert np.abs(mot - mid).max() < 0.01, (shape, np.abs(mot - mid).max())
        assert abs(np.hypot(*d.T).sum() - 48.0) < 0.05   # arc length (chords are slightly shorter)
    P = traj(8.0, -3.0, "arc", A=0.02); assert P[-1, 1] > 0 and abs(P[-1, 2] - 0.02 * 64 / 6) < 1e-3


if __name__ == "__main__":
    selftest(); print("traj selftest ok")
