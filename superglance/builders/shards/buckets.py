"""The PAI-AV rows' sampling bucket: "event" when the 4-s future stops, brakes hard or pulls away from standstill, else
"plain" (superglance.builders.shards.common.bucket)."""
import numpy as np



def aux_label(P):
    """[8, >=2] future poses at 0.5 s steps -> (stop, v1..v4, event parts)."""
    P = np.concatenate([np.zeros((1, 2)), np.asarray(P, np.float64)[:, :2]])
    d = np.hypot(*np.diff(P, axis=0).T); s = np.concatenate([[0.0], np.cumsum(d)]); u = d / 0.5
    slow = [k for k in range(8) if u[k] < 0.5 and s[k] <= 30.0]
    stop = 1 if slow else (0 if s[8] >= 30.0 else -1)
    v = [s[2 * k + 2] - s[2 * k] for k in range(4)]
    return stop, v, u


def event(stop, u, v0):
    dec = np.diff(np.concatenate([[v0], u[:4]])) / 0.5   # 0-2 s
    return stop == 1 or bool(dec.min() <= -1.5) or (v0 < 0.5 and u[-1] > 2.0)


