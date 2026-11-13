"""PAI-AV past frames, made as the training rows' were: for each row's time t, the front wide frames nearest
t - 1.5 / 1.0 / 0.5 s (within 0.1 s), re-projected to nuPlan's CAM_F0 by pai_inputs.remap_grid as the current frames are, then
superglance.model.history.hist_jpeg (shrink to 174,080 px, JPEG q92). Frames come from one sequential PyAV decode per clip
(frame i = the i-th decoded frame, stopped after the last one needed). clip_hist writes them as tar members
pai/<clip>/<ms>_<k>.jpg (k = 0, 1, 2 for 1.5 / 1.0 / 0.5 s back; ms = the row's); superglance.eval.pai_history passes a writer
that stores them as files.
usage: python -m superglance.eval.pai_past --selftest"""
import io, sys, tarfile

import numpy as np

from superglance.model import history as mf_hist
from superglance.eval import pai_inputs as pi

CAM = pi.CAM


def want_frames(fts, rows):
    """{frame index: [(ms, k), ...]} for each row's three past times; rows missing any of the three are left out whole."""
    out = {}
    for r in rows:
        ms = int(r["token"].rsplit("_", 1)[1]); got = []
        for k, off in enumerate(mf_hist.OFFS):
            i = int(np.argmin(np.abs(fts - (r["t_s"] - off))))
            if abs(fts[i] - (r["t_s"] - off)) > 0.1:
                break
            got.append((i, ms, k))
        if len(got) == len(mf_hist.OFFS):
            for i, m, k in got:
                out.setdefault(i, []).append((m, k))
    return out


def decode(mp4, idx):
    """{i: RGB array} for the sorted frame indices idx, by one sequential decode (frame i = the i-th decoded frame)."""
    import av
    out, need = {}, set(idx)
    with av.open(io.BytesIO(mp4)) as c:
        st = c.streams.video[0]; st.thread_count = 1   # one core per process: ~17 ms CPU per HEVC 1080p frame, the pool fills the box
        for i, fr in enumerate(c.decode(st)):
            if i in need:
                out[i] = fr.to_ndarray(format="rgb24")
            if i >= idx[-1]:
                break
    return out


def clip_hist(root, chunk, clip, rows, tsb, mp4, tf):
    import cv2, pandas as pd
    from PIL import Image
    fts_us = pd.read_parquet(io.BytesIO(tsb))["timestamp"].to_numpy(np.int64)
    want = want_frames(fts_us.astype(np.float64) / 1e6, rows)
    if not want:
        return 0
    p = pi.ftheta_params(pi.calib_row(root, chunk, clip, CAM, "camera_intrinsics"))
    assert list(p["resolution"]) == [pi.W, pi.H], p.get("resolution")
    ex = pi.calib_row(root, chunk, clip, CAM, "sensor_extrinsics")
    mx, my = pi.remap_grid(p, pi.quat_R(ex["qx"], ex["qy"], ex["qz"], ex["qw"]))
    idx = sorted(want); fr = decode(mp4, idx)
    assert len(fr) == len(idx), (clip, len(fr), len(idx))
    n = 0
    for i in idx:   # RGB; the current frames are remapped from the BGR copy, the same pixels per channel
        a = fr[i]
        im = cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        b = mf_hist.hist_jpeg(Image.fromarray(im))
        for ms, k in want[i]:
            ti = tarfile.TarInfo(f"pai/{clip}/{ms:05d}_{k}.jpg"); ti.size = len(b); ti.mtime = 0
            tf.addfile(ti, io.BytesIO(b)); n += 1
    return n


def selftest():
    fts = np.arange(0, 20, 1 / 30)   # 30 fps
    w = want_frames(fts, [{"token": "c_06467", "t_s": 6.4667}, {"token": "c_00500", "t_s": 0.5}])   # the second lacks 1.5 / 1.0 s back
    assert sorted(m for v in w.values() for m, _ in v) == [6467] * 3, w
    assert sorted(k for v in w.values() for _, k in v) == [0, 1, 2]
    assert all(abs(fts[i] - (6.4667 - mf_hist.OFFS[k])) < 1 / 60 for i, v in w.items() for _, k in v)
    print("SELFTEST_OK")


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        selftest()
