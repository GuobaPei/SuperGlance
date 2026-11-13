"""Staging of a stage-3 pool with history (a rerun keeps what is there):
  rows      <mf>/pool_sftmf/{train,dev}.jsonl (history_rows.py; MF_POOL picks another pool dir) and the targets, copied into <dest>
  images    the image root <img> (navtrain, pai, pai_dagger, qa, l2d, wod), linked into <dest>; history: <mf>/hist_nav/trainval/*.tar
            and <mf>/hist_pai/*.tar (non-empty ones), extracted into <dest>/hist
STAGED is written only after every image any row names exists and is non-empty; otherwise the missing ones are counted per top
dir and it stops.
usage: python -m superglance.builders.pools.stage <mf dir> <img dir> <dest> [threads=48]"""
import json, os, shutil, subprocess, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def main(mf, img, dest, threads=48):
    os.makedirs(f"{dest}/hist", exist_ok=True); threads = int(threads)
    for f in ("train.jsonl", "dev.jsonl", "train_targets.npy", "dev_targets.npy"):
        s, d = os.path.realpath(f"{mf}/{os.environ.get('MF_POOL', 'pool_sftmf')}/{f}"), f"{dest}/{f}"   # MF_POOL: e.g. the cleaned pool
        if not (os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s)):
            shutil.copyfile(s, d + ".part"); os.replace(d + ".part", d)
    for top in ("navtrain", "pai", "pai_dagger", "qa", "l2d", "wod"):
        if not os.path.islink(f"{dest}/{top}"):
            os.symlink(f"{img}/{top}", f"{dest}/{top}")
    log("rows, targets and image links staged")
    tars = sorted(f"{mf}/hist_nav/trainval/{t}" for t in os.listdir(f"{mf}/hist_nav/trainval") if t.endswith(".tar"))
    tars += sorted(f"{mf}/hist_pai/{t}" for t in os.listdir(f"{mf}/hist_pai") if t.endswith(".tar") and os.path.getsize(f"{mf}/hist_pai/{t}") > 0)
    fill = sorted(f"{mf}/hist_nav_fill/trainval/{t}" for t in os.listdir(f"{mf}/hist_nav_fill/trainval") if t.endswith(".tar")) \
        if os.path.isdir(f"{mf}/hist_nav_fill/trainval") else []   # after hist_nav's, so --skip-old-files keeps its frames
    for ts in (tars, fill):
        with ThreadPoolExecutor(16) as ex:
            rc = list(ex.map(lambda t: subprocess.run(["tar", "-xf", t, "-C", f"{dest}/hist", "--skip-old-files"]).returncode, ts))
        assert not any(rc), sum(map(bool, rc))
    tars += fill
    log(f"{len(tars)} history tars extracted")
    need = set()
    for s in ("train", "dev"):
        for l in open(f"{dest}/{s}.jsonl"):
            need.update(json.loads(l)["images"])
    def ok(rel):
        try:
            return os.path.getsize(f"{dest}/{rel}") > 0
        except OSError:
            return False
    need = sorted(need)
    def missing():
        with ThreadPoolExecutor(threads) as ex:
            return Counter(r.split("/", 1)[0] for r, k in zip(need, ex.map(ok, need, chunksize=1024)) if not k)
    miss = missing()
    if miss:
        log(f"STOP: missing images per top dir {dict(miss)}"); sys.exit(1)
    open(f"{dest}/STAGED", "w").write(json.dumps({"images": len(need), "tars": len(tars)}) + "\n")
    log(f"STAGED: all {len(need)} images present")


if __name__ == "__main__":
    main(*sys.argv[1:5])
