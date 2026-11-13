"""The candidate library's shared kinematics of 4-s trajectories (8 poses at 0.5-4 s, x ahead / y left / heading, ego frame;
the origin is the pose at 0 s) and the nearest-candidate distance.
  seg speeds   v_i = |P_i - P_(i-1)| / 0.5 at t = 0.25, 0.75, ..., 3.75 s (P_0 = origin)
  start speed  given (the logged ego speed) or extrapolated: max(0, 1.5 v_1 - 0.5 v_2) (exact for constant acceleration)
  end speed    v_8 (speed over 3.5-4 s); stopped = v_8 <= 0.5 m/s (the L2D stop window)
  brake        max over j >= 2 of (v0 - v_j) / t_j: the mean deceleration from now to any later segment (a brake to a stop at T
               gives ~v0 / T); dv = v0 - v_8
  stop point   arc length at 4 s for stopped trajectories
  lat, turn    |y(4 s)|, |heading(4 s)| (rad)
  ade          mean pose distance over the 8 (x, y) poses (the 'badly covered' definition: nearest candidate > 1 m)"""
import numpy as np

T_SEG = 0.25 + 0.5 * np.arange(8)


def seg_speeds(P):   # [N, 8, >=2] -> [N, 8]
    P = np.asarray(P, np.float64)[..., :2]
    Q = np.concatenate([np.zeros_like(P[:, :1]), P], 1)
    return np.linalg.norm(np.diff(Q, axis=1), axis=-1) / 0.5


def kin(P, v0=None):
    P = np.asarray(P, np.float64); v = seg_speeds(P)
    if v0 is None:
        v0 = np.maximum(0.0, 1.5 * v[:, 0] - 0.5 * v[:, 1])
    v0 = np.asarray(v0, np.float64)
    brake = ((v0[:, None] - v[:, 1:]) / T_SEG[None, 1:]).max(1)
    arc = (v * 0.5).sum(1)
    out = {"v0": v0, "v_end": v[:, 7], "v_mean": arc / 4.0, "dv": v0 - v[:, 7], "brake": brake, "stopped": v[:, 7] <= 0.5,
           "arc4": arc, "x4": P[:, 7, 0], "y4": P[:, 7, 1], "lat": np.abs(P[:, 7, 1]), "seg": v}
    if P.shape[-1] >= 3:
        out["turn"] = np.abs(P[:, 7, 2])
    else:   # heading from the last segment's direction
        d = P[:, 7, :2] - P[:, 6, :2]; out["turn"] = np.abs(np.arctan2(d[:, 1], d[:, 0]))
    return out


def ade_to(F, C, chunk=512):   # [N, 8, >=2] rows, [K, 8, >=2] candidates -> [N, K] float32 mean pose distance
    F = np.asarray(F, np.float32)[..., :2]; C = np.asarray(C, np.float32)[..., :2]
    return np.concatenate([np.linalg.norm(F[i:i + chunk, None] - C[None], axis=-1).mean(-1) for i in range(0, len(F), chunk)])


def nearest(F, C, chunk=256):   # -> (min ADE [N] float32, argmin [N] int32), memory ~ chunk x K x 8 x 2 floats
    F = np.asarray(F, np.float32)[..., :2]; C = np.asarray(C, np.float32)[..., :2]
    m = np.empty(len(F), np.float32); a = np.empty(len(F), np.int32)
    for i in range(0, len(F), chunk):
        d = np.linalg.norm(F[i:i + chunk, None] - C[None], axis=-1).mean(-1)
        a[i:i + chunk] = d.argmin(1); m[i:i + chunk] = d[np.arange(len(d)), a[i:i + chunk]]
    return m, a


def gap_category(k):
    """per row: the gap categories (four, defined here on the logged future + ego speed; overlapping) and a
    primary label in priority order hard brake > sharp turn > fast > mid-speed curve > other"""
    hb = k["brake"] > 1.5
    st = k["turn"] > np.deg2rad(45)
    fa = k["v0"] > 16
    mc = (k["v0"] >= 10) & (k["v0"] <= 16) & (k["lat"] >= 6)
    prim = np.where(hb, "hard_brake", np.where(st, "sharp_turn", np.where(fa, "fast_gt16", np.where(mc, "mid_curve", "other"))))
    return {"hard_brake": hb, "sharp_turn": st, "fast_gt16": fa, "mid_curve": mc}, prim


def selftest():
    t = 0.5 * np.arange(1, 9)
    v0, a = 10.0, 2.0   # constant deceleration, no stop within 4 s (v(4) = 2)
    P = np.stack([v0 * t - a * t ** 2 / 2, np.zeros(8), np.zeros(8)], -1)[None]
    k = kin(P)
    assert abs(k["v0"][0] - v0) < 1e-9 and abs(k["brake"][0] - a) < 1e-9 and not k["stopped"][0], k
    s = np.minimum(t, 3.0); P2 = np.stack([12 * s - 2 * s ** 2, np.zeros(8), np.zeros(8)], -1)[None]   # 12 m/s, 4 m/s^2: stops at 3 s
    k2 = kin(P2); assert k2["stopped"][0] and abs(k2["arc4"][0] - 18.0) < 1e-9 and abs(k2["v0"][0] - 12) < 1e-9, k2
    m, ix = nearest(P, np.concatenate([P2, P])); assert ix[0] == 1 and m[0] < 1e-6


if __name__ == "__main__":
    selftest(); print("kin selftest ok")
