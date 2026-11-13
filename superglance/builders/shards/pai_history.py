"""PAI-AV past frames of the training rows: for every sample time t of pai_inputs' rows (<pai_in>/inputs/<chunk>.jsonl), the
front-wide frames nearest t - 1.5 / 1.0 / 0.5 s (within 0.1 s), made by superglance.eval.pai_past.clip_hist (CAM_F0 re-projection,
then hist_jpeg). Only clips with crc32(clip) % 2 == 0 (SEL). One tar per chunk, <out>/<chunk>.sel.tar, members
pai/<clip>/<ms>_<k>.jpg (k = 0, 1, 2 for 1.5 / 1.0 / 0.5 s back; ms = the row's), plus <chunk>.sel.json (rows, frames written,
rows with all three); a chunk is claimed by creating its .part exclusively, written, then renamed; restarts skip finished chunks.
The paper's run first wrote every clip of some chunks (<chunk>.tar), then the SEL clips of the rest; done() accepts both.
usage: python -m superglance.builders.shards.pai_history <pai root> <pai_in dir> <out dir> [procs=80] [chunks=a:b | rev]
       python -m superglance.builders.shards.pai_history --one <pai root> <pai_in dir> <out dir> [fwd|rev]   (one serial worker)"""
import json, os, sys, tarfile, time, zipfile, zlib
from multiprocessing import Pool

from superglance.model import history as mf_hist   # noqa: E402
from superglance.eval.pai_past import clip_hist   # noqa: E402
from superglance.eval import pai_inputs as pi   # noqa: E402

CAM = pi.CAM
SEL = lambda clip: zlib.crc32(clip.encode()) % 2 == 0


def done(out, chunk):
    full = f"{out}/{chunk:04d}.tar"
    return any(os.path.exists(f) and os.path.getsize(f) > 0 for f in (f"{out}/{chunk:04d}.sel.tar", full))


def chunk_job(job):
    chunk, root, pin, out = job
    dst = f"{out}/{chunk:04d}.sel.tar"
    if done(out, chunk):
        return chunk, None
    try:
        fd = os.open(dst + ".part", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:   # in flight in another worker (or a drained run's last chunk)
        return chunk, None
    import cv2
    cv2.setNumThreads(1)
    t0 = time.time(); by = {}
    for l in open(f"{pin}/inputs/{chunk:04d}.jsonl"):
        r = json.loads(l)
        if SEL(r["log_name"]):
            by.setdefault(r["log_name"], []).append(r)
    st = {"clips": len(by), "rows": sum(map(len, by.values())), "frames": 0, "errors": 0}
    cz = zipfile.ZipFile(f"{root}/camera/{CAM}/{CAM}.chunk_{chunk:04d}.zip")
    with os.fdopen(fd, "wb") as fo, tarfile.open(fileobj=fo, mode="w") as tf:
        for clip, rows in by.items():
            try:
                st["frames"] += clip_hist(root, chunk, clip, rows, cz.read(f"{clip}.{CAM}.timestamps.parquet"), cz.read(f"{clip}.{CAM}.mp4"), tf)
            except Exception as ex:
                st["errors"] += 1; st.setdefault("first_error", f"{clip}: {type(ex).__name__}: {str(ex)[:200]}")
    st["rows_all3"] = st["frames"] // len(mf_hist.OFFS); st["s"] = round(time.time() - t0, 1)
    json.dump(st, open(f"{out}/{chunk:04d}.sel.json", "w"))
    os.replace(dst + ".part", dst)
    return chunk, st


def main(root, pin, out, procs=80, sel=None):
    os.makedirs(out, exist_ok=True)
    chunks = sorted(int(f[:4]) for f in os.listdir(f"{pin}/inputs") if f.endswith(".jsonl"))
    if sel == "rev":   # a second machine from the other end (each skips finished chunks)
        chunks = chunks[::-1]
    elif sel:
        a, b = map(int, sel.split(":")); chunks = chunks[a:b]
    chunks = sorted(chunks, key=lambda c: os.path.exists(f"{out}/{c:04d}.tar.part"))   # the first run's in-flight chunks last
    t0, tot = time.time(), {}
    with Pool(int(procs), maxtasksperchild=8) as pool:
        for n, (chunk, st) in enumerate(pool.imap_unordered(chunk_job, [(c, root, pin, out) for c in chunks]), 1):
            for k, v in (st or {"skipped": 1}).items():
                if isinstance(v, (int, float)):
                    tot[k] = tot.get(k, 0) + v
            if n % 20 == 0 or n == len(chunks):
                print(f"{n}/{len(chunks)} chunks, {time.time() - t0:.0f} s: {json.dumps({k: round(v, 1) for k, v in tot.items()})}", flush=True)
    print(f"DONE {len(chunks)} chunks: {json.dumps(tot)}", flush=True)


def one(root, pin, out, order="fwd"):
    chunks = sorted(int(f[:4]) for f in os.listdir(f"{pin}/inputs") if f.endswith(".jsonl"))
    for c in chunks[::-1] if order == "rev" else chunks:
        chunk, st = chunk_job((c, root, pin, out))
        if st:
            print(time.strftime("%H:%M:%S", time.gmtime()), chunk, json.dumps(st), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(*sys.argv[2:7])
    else:
        main(*sys.argv[1:4], *sys.argv[4:6])
