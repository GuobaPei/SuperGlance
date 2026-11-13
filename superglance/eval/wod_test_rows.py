"""WOD-E2E test frames to submit -> rows, with superglance.builders.pools.wod_rows (the same CAM_F0 re-projection, prompt,
ego and nav fields). Test frames carry no future and no rater trajectories, so their rows have future [] and prefs [].
Selection: the frame names in the WOD-E2E release's test_sequence_frames_for_submission_as_list.txt. A pool of <workers>
processes, one shard per task; each shard's pages are dropped from the page cache after its read, and the stitch-map cache is
cleared after every frame (one frame per segment, so it never hits). Out as wod_rows: <out>/img/wod_<name>.jpg,
<out>/meta/<shard>.jsonl, then <out>/meta/<shard>.done ("N frames, K kept"); a shard with a .done is skipped. Ends with the
count check against the list.
usage: python -m superglance.eval.wod_test_rows <list.txt> <out> <workers> <shard> [<shard> ...]"""
import collections, glob, json, os, struct, sys, time
from multiprocessing import Pool
import cv2
from superglance.builders.pools import wod_rows as W

LIST, OUT = sys.argv[1], sys.argv[2]
WANT = {l.strip() for l in open(LIST) if l.strip()}


def records(path):   # wod_rows.records' framing (length, crc, data, crc), then the file leaves the page cache
    with open(path, "rb") as f:
        while len(h := f.read(12)) == 12:
            n = struct.unpack("<Q", h[:8])[0]; d = f.read(n); f.read(4); yield d
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def shard(sh):
    base = os.path.basename(sh); n = k = 0; t0 = time.time()
    if os.path.exists(f"{OUT}/meta/{base}.done"):
        return base, -1, -1, 0.0
    cv2.setNumThreads(2)
    with open(f"{OUT}/meta/{base}.jsonl.tmp", "w") as f:
        for d in records(sh):
            n += 1; x = W.e2e.E2EDFrame.FromString(d); name = x.frame.context.name
            if name not in WANT:
                continue
            img = f"{OUT}/img/wod_{name}.jpg"
            if not os.path.exists(img):
                ok, buf = cv2.imencode(".jpg", W.stitch(x.frame), [cv2.IMWRITE_JPEG_QUALITY, 92]); assert ok
                open(img + ".part", "wb").write(buf.tobytes()); os.replace(img + ".part", img); W._maps.clear()
            f.write(json.dumps(W.row(x, img)) + "\n"); k += 1
    os.replace(f"{OUT}/meta/{base}.jsonl.tmp", f"{OUT}/meta/{base}.jsonl")
    open(f"{OUT}/meta/{base}.done", "w").write(f"{n} frames, {k} kept\n")
    return base, n, k, time.time() - t0


def main():
    workers, shards = int(sys.argv[3]), sys.argv[4:]
    os.makedirs(f"{OUT}/img", exist_ok=True); os.makedirs(f"{OUT}/meta", exist_ok=True); t0 = time.time(); N = K = 0
    with Pool(workers) as pool:
        for i, (base, n, k, dt) in enumerate(pool.imap_unordered(shard, shards), 1):
            N += max(n, 0); K += max(k, 0)
            print(f"{time.time() - t0:6.0f} s  {i}/{len(shards)} {base} {n} frames {k} kept ({dt:.1f} s)", flush=True)
    c = collections.Counter(json.loads(l)["token"] for p in glob.glob(f"{OUT}/meta/*.jsonl") for l in open(p))
    dup = [t for t, m in c.items() if m > 1]; miss = WANT - set(c); extra = set(c) - WANT
    done = len(glob.glob(f"{OUT}/meta/*.done"))
    print(f"CHECK {len(shards)} shards ({done} done), {N} frames read this run; listed {len(WANT)}, found {len(set(c) & WANT)}, "
          f"duplicates {len(dup)}, missing {len(miss)}, not listed {len(extra)}; {time.time() - t0:.0f} s", flush=True)
    if dup or miss or extra:
        print("MISMATCH dup", dup[:5], "missing", sorted(miss)[:5], "extra", sorted(extra)[:5], flush=True); sys.exit(2)


if __name__ == "__main__":
    main()
