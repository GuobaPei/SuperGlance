"""PAI-AV open-loop rows: our version of Qwen-Drive-1.0's 700-frame leakage-free subset (not released): the clips of the
public gold eval list of the alpamayo recipes (1005_7cam_gold_eval_metadb_public.parquet) in one PAI-AV split (default test:
140 clips), 5 frames per clip at T0 + (-2, -1, 0, 1, 2) s. T0 = 5.1 s into the clip is the official eval's fixed t0 (its
--t0_us 5_100_000; the parquet's event_t0s are in another timebase and unused), so the ol+00 rows alone are that protocol's
140 frames. Rows are pai_inputs' (the front wide frame re-projected into CAM_F0, history / velocity / acceleration / command)
plus fut10 = the logged ego positions at 0.1-5.0 s (10 Hz) in the ego frame at t, for ADE at 3 s / 5 s. A frame needs 1.9 s of
track before it, 5.0 s after it, and a video frame within 0.1 s.
Gold "-": every clip of the split whose chunk's front camera zip is in the root, not the gold list.
Out: <out>/pai_openloop_<split>.jsonl, <out>/pai_openloop_<split>_tokens.txt, <out>/img/<token>.jpg (token <clip>_ol<+-NN>).
usage: python -m superglance.eval.pai_rows <pai root> <gold parquet | -> <out dir> [split=test|val|all] [procs=16]"""
import io, json, math, os, sys, zipfile
from multiprocessing import Pool
import numpy as np, pandas as pd
from superglance.eval import pai_inputs as pi

T0, OFFS = 5.1, (-2.0, -1.0, 0.0, 1.0, 2.0)


def fut10(ts, P, yaw, t):   # logged positions at t + 0.1 .. 5.0 s in the ego frame at t
    uy = np.unwrap(yaw); x0, y0, h0 = np.interp(t, ts, P[:, 0]), np.interp(t, ts, P[:, 1]), np.interp(t, ts, uy)
    tt = t + 0.1 * np.arange(1, 51); dx, dy = np.interp(tt, ts, P[:, 0]) - x0, np.interp(tt, ts, P[:, 1]) - y0
    c, s = math.cos(h0), math.sin(h0)
    return np.stack([c * dx + s * dy, -s * dx + c * dy], 1)


def one(job):
    import av, cv2
    clip, chunk, root, out = job; t0 = T0
    try:
        ts, P, yaw, src = pi.ego_track(root, chunk, clip)
        cz = zipfile.ZipFile(f"{root}/camera/{pi.CAM}/{pi.CAM}.chunk_{chunk:04d}.zip")
        fts = pd.read_parquet(io.BytesIO(cz.read(f"{clip}.{pi.CAM}.timestamps.parquet")))["timestamp"].to_numpy(np.float64) / 1e6
        times = [t0 + d for d in OFFS if ts[0] + 1.9 <= t0 + d <= ts[-1] - 5.0]
        want = {int(np.argmin(np.abs(fts - t))): t for t in times}
        want = {i: t for i, t in want.items() if abs(fts[i] - t) <= 0.1}
        if not want:
            return [], f"skip {clip}: no frame with 1.9 s of track before and 5.0 s after"
        p = pi.ftheta_params(pi.calib_row(root, chunk, clip, pi.CAM, "camera_intrinsics"))
        ex = pi.calib_row(root, chunk, clip, pi.CAM, "sensor_extrinsics")
        mx, my = pi.remap_grid(p, pi.quat_R(ex["qx"], ex["qy"], ex["qz"], ex["qw"]))
        rows = []
        with av.open(io.BytesIO(cz.read(f"{clip}.{pi.CAM}.mp4"))) as c:
            st = c.streams.video[0]; st.thread_type = "AUTO"
            for i, fr in enumerate(c.decode(st)):
                if i in want:
                    t = want[i]; tok = f"{clip}_ol{int(round((t - t0) * 10)):+03d}"
                    f = f"{out}/img/{tok}.jpg"
                    cv2.imwrite(f, cv2.remap(fr.to_ndarray(format="bgr24"), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), [cv2.IMWRITE_JPEG_QUALITY, 92])
                    s = pi.ego_samples(ts, P, yaw, t)
                    rows.append(dict(token=tok, log_name=clip, map=None, split="pai", subset="pai_openloop", timestamp_us=int(round(fts[i] * 1e6)), cam_f0=f,
                                     cam_l0=None, cam_r0=None, cams_exist=[True, False, False], metric_cache=None, ego_source=src, t0_offset_s=round(t - t0, 3),
                                     **{k: (v if k == "driving_command" else np.round(np.asarray(v, np.float64), 5).tolist()) for k, v in s.items()},
                                     fut10=np.round(fut10(ts, P, yaw, t), 4).tolist()))
                if i > max(want):
                    break
        return rows, "ok"
    except Exception as e:
        return [], f"error {clip}: {repr(e)[:200]}"


def main():
    root, gold, out = sys.argv[1:4]; split = sys.argv[4] if len(sys.argv) > 4 else "test"; procs = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    idx = pd.read_parquet(f"{root}/clip_index.parquet"); idx = idx if split == "all" else idx[idx.split == split]
    if gold == "-":
        have = {c for c in set(idx.chunk) if os.path.exists(f"{root}/camera/{pi.CAM}/{pi.CAM}.chunk_{int(c):04d}.zip")}
        idx = idx[idx.chunk.isin(have)]
    else:
        idx = idx[idx.index.astype(str).isin(set(pd.read_parquet(gold)["key"]))]
    jobs = [(c, int(k), root, out) for c, k in zip(idx.index.astype(str), idx.chunk)]
    os.makedirs(f"{out}/img", exist_ok=True)
    rows, notes = [], []
    with Pool(procs) as pool:
        for r, note in pool.imap_unordered(one, jobs):
            rows += r; notes.append(note)
    rows.sort(key=lambda r: r["token"])
    with open(f"{out}/pai_openloop_{split}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    open(f"{out}/pai_openloop_{split}_tokens.txt", "w").write("\n".join(r["token"] for r in rows) + "\n")
    bad = [n for n in notes if n != "ok"]
    print(f"{split}: {len(jobs)} clips, {len(rows)} frames; not ok {len(bad)}: {bad[:5]}")


if __name__ == "__main__":
    main()
