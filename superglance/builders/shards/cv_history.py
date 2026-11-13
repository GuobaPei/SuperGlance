"""cv_history: the past positions of a constant-speed ego on the path it drove (no acceleration cue), the training dropout's
replacement for the logged past (superglance.builders.shards.dropout). From the AlpaSim driver the dropout was written for."""
import numpy as np


def Rz(h):
    c, s = np.cos(h), np.sin(h)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def cv_history(times, P, yaws, t_now, speed, offsets_s=(1.5, 1.0, 0.5)):
    """Diagnostic history without acceleration cues: the same path the ego drove, but the points placed where a
    constant-speed ego would have been (arc length speed * dt back from now along the received poses; past the oldest
    pose, straight on along the last segment, or along -x when the ego has not moved)."""
    i_now = int(np.argmin(np.abs(times - t_now)))
    Rt = Rz(yaws[i_now]).T[:2, :2]
    uy = np.unwrap(yaws)
    back = [i for i in range(i_now, -1, -1)]
    xy = np.array([Rt @ (P[i, :2] - P[i_now, :2]) for i in back])
    hh = np.array([(uy[i] - uy[i_now] + np.pi) % (2 * np.pi) - np.pi for i in back])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]) if len(xy) > 1 else np.zeros(1)
    out = []
    for dt in offsets_s:
        d = speed * dt
        if s[-1] >= d and d > 0:
            x, y, h = (float(np.interp(d, s, xy[:, 0])), float(np.interp(d, s, xy[:, 1])), float(np.interp(d, s, hh)))
        else:
            u = xy[-1] - xy[-2] if len(xy) > 1 and s[-1] > 0.05 else np.array([-1.0, 0.0])
            u = u / max(np.hypot(*u), 1e-9)
            x, y = (xy[-1] + u * (d - s[-1])).tolist()
            h = float(hh[-1])
        out.append([x, y, h])
    return out + [[0.0, 0.0, 0.0]]
