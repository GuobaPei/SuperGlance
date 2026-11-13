"""Past frames for the PAI-AV open-loop rows (superglance.eval.pai_rows): for every row, the front wide frames nearest
t - 1.5 / 1.0 / 0.5 s (within 0.1 s), made as the training rows' (superglance.eval.pai_past.clip_hist: the CAM_F0
re-projection, then hist_jpeg). t = the row's timestamp_us, its own video frame's time in the clip's timestamps.parquet. Every
clip of the rows.
Out: <out>/pai/<clip>/<ms>_<k>.jpg and <out>/index.jsonl (image = the row's cam_f0, token, past = 3 paths oldest first or
null), the --index of superglance.eval.features.
usage: python -m superglance.eval.pai_history <pai root> <rows.jsonl> <out dir> [procs=8]"""
import json, os, sys, zipfile
from multiprocessing import Pool

from superglance.eval import pai_past as mph


class Files:   # clip_hist writes tar members; here each member is a file under the out dir
    def __init__(self, d):
        self.d = d

    def addfile(self, ti, f):
        p = f"{self.d}/{ti.name}"; os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as o:
            o.write(f.read())


def ms(r):
    return round(r["timestamp_us"] / 1000)


def clip_job(job):
    import cv2
    cv2.setNumThreads(1)
    root, chunk, clip, rows, out = job
    cz = zipfile.ZipFile(f"{root}/camera/{mph.CAM}/{mph.CAM}.chunk_{chunk:04d}.zip")
    shim = [{"token": f"{clip}_{ms(r)}", "t_s": r["timestamp_us"] / 1e6} for r in rows]
    return mph.clip_hist(root, chunk, clip, shim, cz.read(f"{clip}.{mph.CAM}.timestamps.parquet"), cz.read(f"{clip}.{mph.CAM}.mp4"), Files(out))


def main(root, rows_path, out, procs=8):
    import pandas as pd
    rows = [json.loads(l) for l in open(rows_path)]
    chunk = pd.read_parquet(f"{root}/clip_index.parquet")["chunk"]; chunk.index = chunk.index.astype(str)
    by = {}
    for r in rows:
        by.setdefault(r["log_name"], []).append(r)
    with Pool(int(procs)) as pool:
        frames = sum(pool.imap_unordered(clip_job, [(root, int(chunk[c]), c, rs, out) for c, rs in by.items()]))
    idx = []
    for r in rows:
        p = [f"{out}/pai/{r['log_name']}/{ms(r):05d}_{k}.jpg" for k in range(3)]
        idx.append({"image": r["cam_f0"], "token": r["token"], "past": p if all(map(os.path.exists, p)) else None})
    with open(f"{out}/index.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in idx)
    print(json.dumps({"rows": len(rows), "clips": len(by), "frames": frames, "rows_with_past": sum(x["past"] is not None for x in idx)}))


if __name__ == "__main__":
    main(*sys.argv[1:5])
