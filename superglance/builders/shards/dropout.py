"""The training dropout of the PAI-AV and rollout rows: a row with crc32(token) % 2 == 1 gets acceleration [0, 0] and its past
positions re-placed by cv_history on its own 4 history poses at its speed (constant-speed points along the driven path).
Augmentation on training rows only; validation, dev and deployment keep the real values."""
import zlib

import numpy as np



def dropped(tok):
    return zlib.crc32(tok.encode()) % 2 == 1


def scene_of(r, img, cv_history, drop=True):
    """The policy scene of a navsim-style row; on dropout rows (train only: drop) acceleration 0 and the cv_history past."""
    v = r["ego_velocity"]; acc, past = r["ego_acceleration"], r["history_poses"]
    if drop and dropped(r["token"]):
        P = np.asarray(past, np.float64)
        acc = [0.0, 0.0]
        past = cv_history(np.array([-1.5e6, -1.0e6, -0.5e6, 0.0]), np.c_[P[:, :2], np.zeros(4)], P[:, 2], 0.0, float(np.hypot(*v[:2])))
    return {"token": r["token"], "image": img, "velocity": v, "acceleration": acc, "command": r["driving_command"], "past": past}


