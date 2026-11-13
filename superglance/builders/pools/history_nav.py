"""Front-camera history frames of NAVSIM scenes (CPU only). For every scene token of the given input exports (navsim_export.py's
jsonl: token, log_name, split), the log's frames i-3, i-2, i-1 (1.5 / 1.0 / 0.5 s back at NAVSIM's 2 Hz; each within 0.1 s of its
offset) must have a CAM_F0 jpg on disk; then each such frame goes through superglance.model.history.hist_jpeg.
Out: <out>/<split>/<log>.tar (members nav/<split>/<log>/CAM_F0/<file>, one per distinct frame) and <out>/index.jsonl
(token, split, 3 relative paths oldest first, or the reason it has none). Rows reuse a frame shared by several tokens.
usage (OPENSCENE_DATA_ROOT set): python -m superglance.builders.pools.history_nav <out> <procs> <inputs.jsonl>..."""
import io, json, os, pickle, sys, tarfile, time
from multiprocessing import Pool

from superglance.model import history as mf_hist  # noqa: E402

B = os.environ["OPENSCENE_DATA_ROOT"]
SPL = {"trainval": (f"{B}/navsim_logs/trainval", f"{B}/sensor_blobs/trainval"),
       "test": (f"{B}/navsim_logs/test", f"{B}/sensor_blobs/test")}


def past(fr, i, S):
    """The data_paths of frames i-3, i-2, i-1 (oldest first), or the reason they are not all there."""
    out = []
    for k in (3, 2, 1):
        j = i - k
        if j < 0:
            return "log_start"
        c = fr[j]["cams"].get("CAM_F0")
        if abs((fr[i]["timestamp"] - fr[j]["timestamp"]) / 1e6 - 0.5 * k) > 0.1:
            return "gap"
        if not c or not os.path.exists(f"{S}/{c['data_path']}"):
            return "missing_jpg"
        out.append(c["data_path"])
    return out


def one(job):
    spl, log, toks, out = job
    from PIL import Image
    L, S = SPL[spl]
    fr = pickle.load(open(f"{L}/{log}.pkl", "rb"))
    idx = {f["token"]: i for i, f in enumerate(fr)}
    res, need = [], set()
    for t in toks:
        p = past(fr, idx[t], S) if t in idx else "token_not_in_log"
        res.append((t, p))
        if isinstance(p, list):
            need.update(p)
    if need:
        os.makedirs(f"{out}/{spl}", exist_ok=True)
        with tarfile.open(f"{out}/{spl}/{log}.tar.part", "w") as tf:
            for dp in sorted(need):
                with Image.open(f"{S}/{dp}") as im:
                    b = mf_hist.hist_jpeg(im)
                ti = tarfile.TarInfo(f"nav/{spl}/{dp}"); ti.size = len(b); ti.mtime = 0
                tf.addfile(ti, io.BytesIO(b))
        os.replace(f"{out}/{spl}/{log}.tar.part", f"{out}/{spl}/{log}.tar")
    return spl, [{"token": t, "split": spl, "past": [f"nav/{spl}/{x}" for x in p]} if isinstance(p, list) else
                 {"token": t, "split": spl, "none": p} for t, p in res], len(need)


def main(out, procs, *inputs):
    os.makedirs(out, exist_ok=True)
    jobs, seen = {}, set()
    for f in inputs:
        for l in open(f):
            r = json.loads(l)
            spl = "test" if r.get("split") == "navtest" else "trainval"
            if (spl, r["token"]) in seen:
                continue
            seen.add((spl, r["token"])); jobs.setdefault((spl, r["log_name"]), []).append(r["token"])
    t0, n, frames, why = time.time(), 0, 0, {}
    with Pool(int(procs)) as pool, open(f"{out}/index.jsonl.part", "w") as fo:
        for spl, res, k in pool.imap_unordered(one, [(s, lg, t, out) for (s, lg), t in sorted(jobs.items())]):
            n += 1; frames += k
            for r in res:
                fo.write(json.dumps(r) + "\n"); key = (r["split"], "all3" if "past" in r else r["none"]); why[key] = why.get(key, 0) + 1
            if n % 200 == 0:
                print(f"{n}/{len(jobs)} logs, {time.time() - t0:.0f} s", flush=True)
    os.replace(f"{out}/index.jsonl.part", f"{out}/index.jsonl")
    rep = {"logs": len(jobs), "tokens": len(seen), "frames": frames, "by_split": {f"{s}/{w}": c for (s, w), c in sorted(why.items())}}
    json.dump(rep, open(f"{out}/report.json", "w"), indent=1); print("DONE", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
