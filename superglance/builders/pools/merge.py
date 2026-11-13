"""Stage-3 pool merge: per-source shards (each: {train,dev}.jsonl + {train,dev}_targets.npy float32, compact rows from 0, image
paths relative to the shard dir, copies.tsv "<source>\t<relative path>") into the one pool dir the trainer reads (<dir>/{split}.jsonl,
<dir>/{split}_targets.npy, images[0] relative to <dir>). Compact rows are offset by the rows of the shards before; rows without
"compact" (QA rows) pass through unchanged. A shard may lack a split, or a targets file when none of its rows is compact (the QA
replay shard). No prompt may hold the road-users block, no image path may be absolute.
usage: python -m superglance.builders.pools.merge <out dir> <shard dir>..."""
import json, os, sys
import numpy as np

OUT, SH = sys.argv[1], sys.argv[2:]
os.makedirs(OUT)   # never overwrite
rep = {}
for split in ("train", "dev"):
    S = [s for s in SH if os.path.exists(f"{s}/{split}.jsonl")]
    T = {s: np.load(f"{s}/{split}_targets.npy", mmap_mode="r") for s in S if os.path.exists(f"{s}/{split}_targets.npy")}
    K = {t.shape[1] for t in T.values()}
    assert len(K) == 1 and all(t.dtype == np.float32 for t in T.values()), (split, K)
    M = np.lib.format.open_memmap(f"{OUT}/{split}_targets.npy.tmp", "w+", np.float32, (sum(map(len, T.values())), K.pop())); off = 0
    with open(f"{OUT}/{split}.jsonl", "w") as f:
        for s in S:
            t, n = T.get(s), 0
            for l in open(f"{s}/{split}.jsonl"):
                r = json.loads(l)
                assert "Road users ahead" not in r.get("question", "") and not any(p.startswith("/") for p in r.get("images", [])), (s, r.get("id"))
                if "compact" in r:
                    assert t is not None and r["compact"]["row"] < len(t); r["compact"]["row"] += off
                f.write(json.dumps(r) + "\n"); n += 1
            if t is not None:
                for i in range(0, len(t), 65536):
                    c = t[i:i + 65536]; M[off + i:off + i + len(c)] = c
                off += len(t)
            rep.setdefault(split, {})[os.path.basename(s.rstrip("/"))] = n
    M.flush(); del M; os.replace(f"{OUT}/{split}_targets.npy.tmp", f"{OUT}/{split}_targets.npy")
with open(f"{OUT}/copies.tsv", "w") as f:
    for s in SH:
        if os.path.exists(f"{s}/copies.tsv"):
            f.writelines(open(f"{s}/copies.tsv"))
json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1); print(json.dumps(rep))
