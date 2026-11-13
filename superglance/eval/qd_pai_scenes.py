#!/usr/bin/env python3
"""Qwen-Drive-1.0 scene file for our PAI-AV open-loop rows (Table 6). It holds the same frames as our rows
(superglance.eval.pai_rows: 140 gold test clips x T0 + (-2..2) s = 700) in Qwen-Drive's scene format (its docs/data.md), so its
run_planning.py scores Qwen-Drive-1.0 on exactly our frames.
Per row (token <clip>_olXX, at t = 5.1 s + XX / 10 in the egomotion timebase, as superglance.eval.pai_rows):
- views (data.md): front wide, cross left and cross right, raw frames at t - 1.5 / 1.0 / 0.5 / 0 s. Each is the frame closest
  before that time (the devkit VideoReader's rule), saved as <clip>/<camera>/<frame index:08d>.jpg (data.md's layout).
- resized sizes, inferred from the 4 WOD demo scenes (the only released metadata):
  - history: smart_resize to the config's history_image_pixels (972x1079 -> 384x416 in the demo);
  - current: short side 720 (972x1079 -> 720x799).
- 10 Hz trajectory from the online egomotion through the devkit's EgomotionState interpolator (data.md: "via the official
  egomotion interpolator"), in the ego frame at t: history 16 poses over 1.5 s, future 50 over 5 s. Velocity and acceleration
  are world-frame in the egomotion; they are rotated into each pose's heading.
- driving command = our row's (superglance.eval.pai_inputs: heading change over 4 s beyond +-20 deg). Qwen-Drive's PAI-AV rule
  is unpublished, and this way both models see the same command.
- instruction text: qwen_drive.scene.DrivingScene.instruction() on that history, as for scenes built from scratch.
- ground truth: our rows' fut10 (offline egomotion) stays ours. This file's future_traj_10hz is the online one, Qwen-Drive's
  convention.
QD_TEST_FRONT_ONLY=1 puts the front camera in all three views, to test the chain without the cross cameras. Never score it.
usage (Qwen-Drive-1.0's environment with physical_ai_av): python -m superglance.eval.qd_pai_scenes <rows.jsonl>
  <pai root>[,<root>...] <out dir> [procs=16]
  Several roots: each clip is read from the first root holding its chunk's egomotion.
  -> <out dir>/scenes.jsonl and <out dir>/frames/<clip>/<camera>/<frame>.jpg"""
import io
import json
import math
import os
import sys
import zipfile
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from physical_ai_av.egomotion import EgomotionState
from physical_ai_av.video import SeekVideoReader
from qwen_drive.scene import CAMERA_VIEWS, CameraFrame, DrivingScene, smart_resize

CAMS = ("camera_front_wide_120fov", "camera_cross_left_120fov", "camera_cross_right_120fov")   # in CAMERA_VIEWS order
OFFS = (-1.5, -1.0, -0.5, 0.0)
T0 = 5.1
HIST_PIXELS, FACTOR = 174080, 32   # Qwen-Drive-1.0-4B config.json history_image_pixels; patch 16 x spatial merge 2
FRONT_ONLY = os.environ.get("QD_TEST_FRONT_ONLY") == "1"


def sizes(w, h, current):
    if current:
        s = 720 / min(w, h)
        return round(w * s), round(h * s)
    gh, gw = smart_resize(h, w, FACTOR, 4 * FACTOR ** 2, HIST_PIXELS)
    return gw, gh


def yaw_of(rotation):
    x, y, z, w = rotation.as_quat().T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def trajectory(interp, t_us):
    """History (16) and future (50) at 10 Hz around t_us: poses (x, y, heading) in the ego frame at t, plus velocity and
    acceleration (x forward, y left) in each pose's own heading."""
    tt = t_us + np.arange(-15, 51, dtype=np.int64) * 100_000
    s = interp(tt)
    P, yaw = s.pose.translation, yaw_of(s.pose.rotation)
    assert np.isfinite(P).all() and np.isfinite(s.velocity).all(), "outside the egomotion track"
    c, sn = math.cos(yaw[15]), math.sin(yaw[15])
    d = P[:, :2] - P[15, :2]
    pose = np.stack([c * d[:, 0] + sn * d[:, 1], -sn * d[:, 0] + c * d[:, 1], (yaw - yaw[15] + np.pi) % (2 * np.pi) - np.pi], 1)
    cy, sy = np.cos(yaw), np.sin(yaw)
    body = lambda v: np.stack([cy * v[:, 0] + sy * v[:, 1], -sy * v[:, 0] + cy * v[:, 1]], 1)
    return pose[:16], body(s.velocity)[:16], body(s.acceleration)[:16], pose[16:]


def one(job):
    clip, chunk, rows, root, out = job
    df = pd.read_parquet(io.BytesIO(zipfile.ZipFile(f"{root}/labels/egomotion/egomotion.chunk_{chunk:04d}.zip").read(f"{clip}.egomotion.parquet")))
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    interp = EgomotionState.from_egomotion_df(df).create_interpolator(df["timestamp"].to_numpy(np.int64))
    times = {r["token"]: int(round((T0 + r["t0_offset_s"]) * 1e6)) for r in rows}
    frames = {}   # (view, token, k) -> image item
    for view, cam in zip(CAMERA_VIEWS, CAMS):
        src = CAMS[0] if FRONT_ONLY else cam
        z = zipfile.ZipFile(f"{root}/camera/{src}/{src}.chunk_{chunk:04d}.zip")
        fts = pd.read_parquet(io.BytesIO(z.read(f"{clip}.{src}.timestamps.parquet")))["timestamp"].to_numpy(np.int64)
        want = {(tok, k): t + int(off * 1e6) for tok, t in times.items() for k, off in enumerate(OFFS)}
        idx = {key: int(np.searchsorted(fts, t, side="right")) - 1 for key, t in want.items()}
        for key, i in idx.items():
            assert i >= 0 and want[key] - fts[i] <= 100_000, f"{clip} {cam}: no frame within 0.1 s before {want[key]}"
        uniq = np.array(sorted(set(idx.values())), dtype=np.int64)
        reader = SeekVideoReader(io.BytesIO(z.read(f"{clip}.{src}.mp4")), timestamps=fts, thread_count=2)
        imgs = dict(zip(uniq.tolist(), reader.decode_images_from_frame_indices(uniq)))
        reader.close()
        os.makedirs(f"{out}/frames/{clip}/{cam}", exist_ok=True)
        for i, a in imgs.items():
            f = f"{out}/frames/{clip}/{cam}/{i:08d}.jpg"
            if not os.path.exists(f):
                Image.fromarray(a).save(f, quality=95)
        for (tok, k), i in idx.items():
            h, w = imgs[i].shape[:2]
            rw, rh = sizes(w, h, k == len(OFFS) - 1)
            frames[view, tok, k] = {"image": f"{clip}/{cam}/{i:08d}.jpg", "width": w, "height": h, "timestamp_idx": k,
                                    "timestamp_offset_sec": OFFS[k], "resized_width": rw, "resized_height": rh}
    recs = []
    for r in rows:
        tok = r["token"]
        hist, hv, ha, fut = trajectory(interp, times[tok])
        cmd = [int(x) for x in r["driving_command"]]
        nav = {0: 1, 1: 0, 2: 2}[cmd.index(1)]   # one-hot (left, straight, right, unknown) -> 0 straight / 1 left / 2 right
        scene = DrivingScene(views={v: [CameraFrame("unused")] * len(OFFS) for v in CAMERA_VIEWS}, history=hist, history_velocity=hv,
                             history_acceleration=ha, ego_velocity=hv[-1], ego_acceleration=ha[-1], driving_command=cmd, nav_command=nav)
        content = []
        for v in CAMERA_VIEWS:
            content.append({"text": v})
            for k in range(len(OFFS)):
                content += [{"text": f"frame: {k}"}, frames[v, tok, k]]
        content.append({"text": scene.instruction()})
        rnd = lambda a: np.round(np.asarray(a, np.float64), 6).tolist()
        recs.append({"type": "chatml", "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": []}],
                     "meta_info": {"dataset": "PhysicalAI", "task_name": "trajectory_prediction", "token": tok, "scene_token": clip,
                                   "cam_order": list(CAMS), "history_time_offsets": list(OFFS), "image_hz": 2.0, "traj_hz": 10.0,
                                   "t_us": times[tok], "front_only_test": FRONT_ONLY},
                     "trajectory": {"hist_traj_10hz": rnd(hist), "hist_vel_10hz": rnd(hv), "hist_acc_10hz": rnd(ha), "hist_valid_mask_10hz": [1] * 16,
                                    "future_traj_10hz": rnd(fut), "future_valid_mask_10hz": [1] * 50,
                                    "ego_status": {"in_global_frame": False, "ego_pose": [0.0, 0.0, 0.0], "ego_velocity": rnd(hv[-1]),
                                                   "ego_acceleration": rnd(ha[-1]), "driving_command": cmd}, "nav_command": nav},
                     "_gt_gap_m": float(np.linalg.norm(fut[:, :2] - np.asarray(r["fut10"]), axis=1).mean())})
    return recs


def main():
    rows_path, roots, out = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    rows = [json.loads(line) for line in open(rows_path)]
    chunk = pd.read_parquet(f"{roots[0]}/clip_index.parquet")["chunk"]
    root_of = lambda ch: next(r for r in roots if os.path.exists(f"{r}/labels/egomotion/egomotion.chunk_{ch:04d}.zip"))
    by = defaultdict(list)
    for r in rows:
        by[r["log_name"]].append(r)
    with Pool(procs) as pool:
        recs = sorted((x for rs in pool.imap_unordered(one, [(c, int(chunk[c]), rs, root_of(int(chunk[c])), out) for c, rs in by.items()]) for x in rs),
                      key=lambda x: x["meta_info"]["token"])
    gap = np.array([x.pop("_gt_gap_m") for x in recs])
    assert len(recs) == len(rows), (len(recs), len(rows))
    assert all(np.allclose(x["trajectory"]["hist_traj_10hz"][-1], 0) for x in recs)
    # online vs offline egomotion: the same drive, so the 5 s futures agree to centimetres; a timebase slip shows as metres
    assert np.median(gap) < 0.3, f"online vs offline future gap median {np.median(gap):.3f} m"
    with open(f"{out}/scenes.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in recs)
    print(f"{len(recs)} scenes from {len(by)} clips -> {out}/scenes.jsonl; online vs offline future gap (mean over 50 poses): "
          f"median {np.median(gap):.3f} m, p95 {np.percentile(gap, 95):.3f} m, max {gap.max():.3f} m" + ("; FRONT-ONLY TEST" if FRONT_ONLY else ""))


if __name__ == "__main__":
    main()
