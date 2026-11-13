"""PAI-AV human rows, step 1 (CPU only): every training clip's sample times, front frames and ego state, in the policy's scene-row
format (superglance.eval.pai_inputs' fields, plus ego_source, chunk, t_s).
--clips: clip_index.parquet's train split, less the closed-loop evaluation and rollout scenes. Lists (env; a clip may be in
  several): VAL441 (curated_val_441), DAG_ROWS (a glob of the rollout rounds' rows.jsonl: their held scenes, crc32(scene) % 5 == 0),
  TRA232, S916 (the 916 AlpaSim scenes, "clipgt-" stripped), REALSIM; HELD2 (the head refits' held-out PAI-AV clips, never in any
  training pool) goes to part "val", like navtrain's held-out logs. -> clips.tsv (clip, chunk, part), excluded.tsv (clip, split,
  reasons), clips.json (counts).
main, per clip (one job per chunk, so its calibration tables and zips are read once; inputs/<chunk>.jsonl is written last, and a
restart skips finished chunks):
- times: the window [lo, hi] = [start + 1.9 s, end - 4.4 s] of the ego track (an online track is also cut to the front camera's
  frames); t1, t2 = linspace(lo, hi, 4)[1:3] and a 2-s grid anchored on t1 (grid points within 1 s of t2 dropped): ~7 per 20-s clip;
- ego: the offline egomotion, else the online one (pai_inputs.ego_track; ego_source per row); fields from pai_inputs.ego_samples;
- frame: the front-wide frame nearest t within 0.1 s, decoded by the devkit's SeekVideoReader, re-projected to nuPlan's CAM_F0
  (pai_inputs.remap_grid), JPEG quality 92 at 1920 x 1080 -> <img dir>/pai/<clip>/<ms>.jpg, ms = round(1000 t), token <clip>_<ms:05d>.
usage (cv2, av, pandas, physical_ai_av):
  python -m superglance.builders.shards.pai_inputs --clips <pai root> <out dir>
  python -m superglance.builders.shards.pai_inputs <pai root> <out dir> <img dir> [procs=40]   (rows carry img_bytes, the JPEG's size)
  python -m superglance.builders.shards.pai_inputs --check-decode <pai root> <out dir> [clips=40]
  python -m superglance.builders.shards.pai_inputs --selftest"""
import glob, io, json, os, sys, time, zipfile, zlib
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np

from superglance.eval import pai_inputs as pi   # noqa: E402

CAM = pi.CAM


def times_of(lo, hi):
    t1, t2 = np.linspace(lo, hi, 4)[1:-1]
    g = [t1 + 2.0 * k for k in range(-int((t1 - lo) // 2), int((hi - t1) // 2) + 1)]
    return sorted([t for t in g if t == t1 or abs(t - t2) > 1.0] + [t2])


def clips(root, out):
    import pandas as pd
    t = pd.read_parquet(f"{root}/clip_index.parquet")
    ids = lambda f: {l.strip().replace("clipgt-", "") for l in open(f) if l.strip()}
    E = os.environ
    dag = [json.loads(l)["scene"] for f in sorted(glob.glob(E["DAG_ROWS"])) for l in open(f)]
    sets = {"val441": ids(E["VAL441"]),
            "held": {s.split("clipgt-", 1)[1] for s in set(dag) if zlib.crc32(s.encode()) % 5 == 0},
            "tra232": ids(E["TRA232"]), "s916": ids(E["S916"]),
            "realsim": ids(E["REALSIM"])}
    held2 = {json.loads(l)["log_name"] for l in open(E["HELD2"])}
    os.makedirs(out, exist_ok=True)
    rep = {"index": len(t), "split": t["split"].value_counts().to_dict(), "lists": {k: len(v) for k, v in sets.items()},
           "lists_in_index": {k: len(v & set(t.index)) for k, v in sets.items()}, "excluded_train": {k: 0 for k in sets}}
    keep, n_ex = [], 0
    with open(f"{out}/excluded.tsv", "w") as fe:
        for clip, chunk, split in zip(t.index.astype(str), t["chunk"].astype(int), t["split"]):
            why = [k for k, v in sets.items() if clip in v]
            if split != "train" or why:
                fe.write(f"{clip}\t{split}\t{','.join(why)}\n")
            if split == "train":
                for k in why:
                    rep["excluded_train"][k] += 1
                n_ex += bool(why)
                if not why:
                    keep.append((clip, chunk, "val" if clip in held2 else "train"))
    with open(f"{out}/clips.tsv", "w") as f:
        f.writelines(f"{c}\t{k}\t{p}\n" for c, k, p in keep)
    rep.update(excluded_train_clips=n_ex, held2=len(held2), kept={p: sum(x[2] == p for x in keep) for p in ("train", "val")},
               excluded_not_train=int((t["split"] != "train").sum()))
    json.dump(rep, open(f"{out}/clips.json", "w"), indent=1); print(json.dumps(rep))


def remap(p, R):   # pai_inputs.remap_grid (the run cached its CAM_F0 rays per process: same grid, same calls)
    return pi.remap_grid(p, R)


def clip_rows(root, chunk, clip, tsb, mp4, img):
    import cv2, pandas as pd
    from physical_ai_av.video import SeekVideoReader
    ts, P, yaw, src = pi.ego_track(root, chunk, clip)
    fts_us = pd.read_parquet(io.BytesIO(tsb))["timestamp"].to_numpy(np.int64)
    fts = fts_us.astype(np.float64) / 1e6
    lo, hi = ts[0] + 1.5 + 0.4, ts[-1] - 4.0 - 0.4   # pai_inputs.sample_times
    if src == "online":
        lo, hi = max(lo, fts[0] + 1.5 + 0.4), min(hi, fts[-1] - 4.0 - 0.4)
    if hi <= lo:
        return [], "short"
    p = pi.ftheta_params(pi.calib_row(root, chunk, clip, CAM, "camera_intrinsics"))
    assert list(p["resolution"]) == [pi.W, pi.H], p.get("resolution")
    ex = pi.calib_row(root, chunk, clip, CAM, "sensor_extrinsics")
    mx, my = remap(p, pi.quat_R(ex["qx"], ex["qy"], ex["qz"], ex["qw"]))
    want = {int(np.argmin(np.abs(fts - t))): t for t in times_of(lo, hi)}
    want = {i: t for i, t in want.items() if abs(fts[i] - t) <= 0.1}   # pai_inputs: a frame at t, never the clip's first / last instead
    if not want:
        return [], "noframe"
    idx = np.array(sorted(want), np.int64)
    rd = SeekVideoReader(io.BytesIO(mp4), timestamps=fts_us, thread_count=1)
    frames = rd.decode_images_from_frame_indices(idx); rd.close()
    os.makedirs(f"{img}/pai/{clip}", exist_ok=True)
    rows = []
    for i, a in zip(idx.tolist(), frames):
        t = want[i]; ms = int(round(t * 1000)); f = f"{img}/pai/{clip}/{ms:05d}.jpg"
        im = cv2.remap(cv2.cvtColor(a, cv2.COLOR_RGB2BGR), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        assert cv2.imwrite(f, im, [cv2.IMWRITE_JPEG_QUALITY, 92]), f
        nb = os.path.getsize(f)
        s = pi.ego_samples(ts, P, yaw, t)
        rows.append(dict(token=f"{clip}_{ms:05d}", log_name=clip, map=None, split="pai", subset="pai_train", timestamp_us=int(round(fts[i] * 1e6)),
                         cam_f0=f, cam_l0=None, cam_r0=None, cams_exist=[True, False, False],
                         **{k: (v if k == "driving_command" else np.round(np.asarray(v, np.float64), 5).tolist()) for k, v in s.items()}, metric_cache=None,
                         frame_offset_ms=round(1000 * (fts[i] - t), 1), ego_source=src, chunk=chunk, t_s=float(t), img_bytes=nb))
    return rows, "ok"


def chunk_job(job):
    chunk, cl, root, out, img = job
    dst = f"{out}/inputs/{chunk:04d}.jsonl"
    if os.path.exists(dst):
        return chunk, None
    import cv2
    cv2.setNumThreads(1)
    t0, st, rows, err = time.time(), {}, [], []
    cz = zipfile.ZipFile(f"{root}/camera/{CAM}/{CAM}.chunk_{chunk:04d}.zip")
    get = lambda c: (cz.read(f"{c}.{CAM}.timestamps.parquet"), cz.read(f"{c}.{CAM}.mp4"))
    rd = ThreadPoolExecutor(1); fut = [rd.submit(get, cl[0][0])]   # the next clip's video is read while this one decodes
    for j, (clip, part) in enumerate(cl):
        if j + 1 < len(cl):
            fut.append(rd.submit(get, cl[j + 1][0]))
        try:
            tsb, mp4 = fut[j].result(); fut[j] = None
            rs, k = clip_rows(root, chunk, clip, tsb, mp4, img)
        except Exception as ex:
            rs, k = [], "error"; err.append(f"{clip}\t{type(ex).__name__}: {str(ex)[:200]}\n")
        st[k] = st.get(k, 0) + 1
        rows += [dict(r, part=part) for r in rs]
    rd.shutdown()
    with open(dst + ".part", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    open(f"{out}/inputs/{chunk:04d}.err", "w").writelines(err)
    os.replace(dst + ".part", dst)
    return chunk, dict(st, rows=len(rows), s=round(time.time() - t0, 1))


def main(root, out, img, procs=40):
    by = {}
    for l in open(f"{out}/clips.tsv"):
        c, k, p = l.split()
        by.setdefault(int(k), []).append((c, p))
    os.makedirs(f"{out}/inputs", exist_ok=True)
    jobs = [(k, v, root, out, img) for k, v in sorted(by.items())]   # chunk order: pai_rows.py builds parts of finished chunks
    t0, tot = time.time(), {}
    with Pool(int(procs), maxtasksperchild=8) as pool:
        for n, (chunk, st) in enumerate(pool.imap_unordered(chunk_job, jobs), 1):
            for k, v in (st or {"skipped": 1}).items():
                tot[k] = tot.get(k, 0) + v
            if n % 20 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} chunks, {time.time() - t0:.0f} s: {json.dumps({k: round(v, 1) for k, v in tot.items()})}", flush=True)
    print(f"DONE {len(jobs)} chunks: {json.dumps(tot)}", flush=True)


def check_decode(root, out, n=40):
    """SeekVideoReader's frame i vs pai_inputs' sequential PyAV decode (frame i = the i-th decoded frame) on n clips' sample frames."""
    import av, cv2, pandas as pd
    from physical_ai_av.video import SeekVideoReader
    cl = [l.split() for l in open(f"{out}/clips.tsv")][::max(1, 150000 // n)][:n]
    m = []
    for clip, chunk, _ in cl:
        chunk = int(chunk); cz = zipfile.ZipFile(f"{root}/camera/{CAM}/{CAM}.chunk_{chunk:04d}.zip")
        fts_us = pd.read_parquet(io.BytesIO(cz.read(f"{clip}.{CAM}.timestamps.parquet")))["timestamp"].to_numpy(np.int64)
        data = cz.read(f"{clip}.{CAM}.mp4"); idx = np.unique(np.linspace(30, len(fts_us) - 30, 7).astype(np.int64))
        rd = SeekVideoReader(io.BytesIO(data), timestamps=fts_us, thread_count=1); a = rd.decode_images_from_frame_indices(idx); rd.close()
        b = {}
        with av.open(io.BytesIO(data)) as c:
            st = c.streams.video[0]; st.thread_type = "AUTO"
            for i, fr in enumerate(c.decode(st)):
                if i in idx:
                    b[i] = cv2.cvtColor(fr.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB)
                if i > idx.max():
                    break
        m += [float(np.abs(x.astype(np.int16) - b[i]).max()) for i, x in zip(idx.tolist(), a)]
    print(json.dumps({"clips": len(cl), "frames": len(m), "max_abs_diff_max": max(m), "identical_frames": sum(x == 0 for x in m)}))


def selftest():
    t = times_of(1.9, 15.6)   # a 20-s clip: t1 6.4667, t2 11.0333 kept; the grid point 10.4667 dropped
    assert len(t) == 7 and np.allclose(t, [2.4667, 4.4667, 6.4667, 8.4667, 11.0333, 12.4667, 14.4667], atol=1e-4), t
    assert np.linspace(1.9, 15.6, 4)[1] in t and np.linspace(1.9, 15.6, 4)[2] in t
    t = times_of(10.0, 11.2)   # a short window: only the old two
    assert len(t) == 2 and np.allclose(t, [10.4, 10.8]), t
    assert min(np.diff(times_of(0.0, 40.0))) > 1.0
    print("SELFTEST_OK")


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        selftest()
    elif sys.argv[1] == "--clips":
        clips(*sys.argv[2:4])
    elif sys.argv[1] == "--check-decode":
        check_decode(sys.argv[2], sys.argv[3], *map(int, sys.argv[4:5]))
    else:
        main(*sys.argv[1:5])
