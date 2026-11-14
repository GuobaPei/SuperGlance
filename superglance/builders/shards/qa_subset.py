"""The QA replay's per-source row counts: a seeded subset of the stage-2 training QA with its images in one dir. Per source,
min(all, MULT x the expected draws) rows; expected draws = N_QA x the source's share of stage 2's own mixture weights (its run
config). Each row's image is copied to <out>/img (images keyed in an sqlite image store are extracted to files), and the row is
rewritten to point there. superglance.builders.pools.qa_replay reads its report.json ("per_source" ... "kept") and weights.json.
Out: <out>/train_qa.jsonl, <out>/img/, <out>/weights.json (stage 2's weights), <out>/report.json.
usage: python -m superglance.builders.shards.qa_subset <stage-2 data dir> <stage-2 run_config.json> <out dir> [n_qa=61440] [mult=3]"""
import json, math, os, random, shutil, sqlite3, sys, zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


def plan(counts, weights, n_qa, mult):   # source -> rows to keep
    W = sum(weights[s] for s in counts)
    return {s: min(n, math.ceil(mult * n_qa * weights[s] / W)) for s, n in counts.items()}


def main(D, cfg, out, n_qa=61440, mult=3):
    w = json.load(open(cfg))["mixture_sampling"]["source_weights"]
    rows = defaultdict(list)
    for l in open(f"{D}/train.jsonl"):
        r = json.loads(l); rows[r["source"]].append(r)
    assert set(rows) == set(w), (sorted(set(rows) ^ set(w)))
    keep = plan({s: len(v) for s, v in rows.items()}, w, n_qa, mult)
    os.makedirs(f"{out}/img")   # never overwrite
    rng = random.Random(20261023); pick, jobs = [], {}
    for s in sorted(rows):
        for r in rng.sample(rows[s], keep[s]):
            p = r["images"][0]; p = p if p.startswith("/") else f"{D}/{p}"
            if "image_store_key" in r:
                src = (os.path.realpath(p), str(r.pop("image_store_key"))); name = f"store_{src[1]}.img"
            else:
                src = os.path.realpath(p); name = f"{zlib.crc32(src.encode()):08x}_{os.path.basename(src)}"
            jobs[f"{out}/img/{name}"] = src; r["images"] = [f"{out}/img/{name}"]; pick.append(r)

    db = {}

    def put(item):
        dst, src = item
        if isinstance(src, tuple):
            con = db.setdefault(src[0], sqlite3.connect(f"file:{src[0]}?mode=ro", uri=True, check_same_thread=False))
            data = con.execute("SELECT data FROM images WHERE id = ?", (src[1],)).fetchone()[0]
            open(dst, "wb").write(data); return len(data)
        shutil.copyfile(src, dst); return os.path.getsize(dst)

    stores = [x for x in jobs.items() if isinstance(x[1], tuple)]; files = [x for x in jobs.items() if not isinstance(x[1], tuple)]
    nbytes = sum(map(put, stores))   # one thread for the sqlite reads (a shared connection), 16 for plain file copies
    with ThreadPoolExecutor(16) as ex:
        nbytes += sum(ex.map(put, files))
    with open(f"{out}/train_qa.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in pick)
    json.dump(w, open(f"{out}/weights.json", "w"), indent=1)
    rep = {"from": D, "n_qa": n_qa, "mult": mult, "rows": len(pick), "images": len(jobs), "store_images": len(stores), "gb": round(nbytes / 1e9, 2),
           "per_source": {s: {"all": len(rows[s]), "kept": keep[s]} for s in sorted(rows)}}
    json.dump(rep, open(f"{out}/report.json", "w"), indent=1); print(json.dumps(rep))


def selftest():
    k = plan({"a": 100, "b": 10, "c": 5}, {"a": 0.5, "b": 0.3, "c": 0.2}, 40, 3)
    assert k == {"a": 60, "b": 10, "c": 5}, k


if __name__ == "__main__":
    selftest()
    main(sys.argv[1], sys.argv[2], sys.argv[3], *map(int, sys.argv[4:6]))
