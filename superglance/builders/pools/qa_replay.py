"""Stage-3 QA replay shard: a seeded sample of the stage-2 training QA (random.Random(20261023), sources in sorted order): all
DriveLM (object, ego, box, relation) and spatialsense rows, SAT 60,000, tallyqa 20,000, GQA 20,000, every other source at the kept
count of the earlier QA subset (<qa subset>/report.json "per_source" ... "kept"; its weights.json holds the stage-2 mixture weights).
Rows are the stage-2 rows, unchanged but for the image: each image is copied to <O>/img/qa/<source>/<name> (SQLite-stored ones
extracted to files, image_store_key dropped) and the row points at qa/<source>/<name>, relative to the shard dir (<O>/qa/qa links
to img/qa). No compact targets: the QA rows keep their own options.
Out: <O>/qa/train.jsonl, train_images.tsv (abs src, rel, bytes), weights.json, report.json (images and answers per row, per source).
usage: python -m superglance.builders.pools.qa_replay <stage-2 data dir> <qa subset dir> <shard root O>"""
import json, os, random, shutil, sqlite3, sys, zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

O = sys.argv[3]
MORE = {"drivelm_object": None, "drivelm_ego": None, "drivelm_box": None, "drivelm_relation": None, "spatialsense": None,
        "SAT": 60000, "tallyqa": 20000, "GQA": 20000}   # None: every row


def main(D, Q):
    old = json.load(open(f"{Q}/report.json"))["per_source"]; w = json.load(open(f"{Q}/weights.json"))
    rows = defaultdict(list)
    for l in open(f"{D}/train.jsonl"):
        r = json.loads(l); rows[r["source"]].append(r)
    assert set(rows) == set(w) == set(old), sorted(set(rows) ^ set(w))
    keep = {s: min(len(v), old[s]["kept"] if s not in MORE else MORE[s] or len(v)) for s, v in rows.items()}
    out = f"{O}/qa"; os.makedirs(f"{O}/img/qa"); os.makedirs(out)   # never overwrite
    os.symlink("../img/qa", f"{out}/qa")
    rng = random.Random(20261023); pick, jobs = [], {}
    for s in sorted(rows):
        os.makedirs(f"{O}/img/qa/{s}")
        for r in rng.sample(rows[s], keep[s]):
            p = r["images"][0]; p = p if p.startswith("/") else f"{D}/{p}"
            if "image_store_key" in r:
                src = (os.path.realpath(p), str(r.pop("image_store_key"))); name = f"store_{src[1]}.img"
            else:
                src = os.path.realpath(p); name = f"{zlib.crc32(src.encode()):08x}_{os.path.basename(src)}"
            jobs[f"qa/{s}/{name}"] = src; r["images"] = [f"qa/{s}/{name}"]; pick.append(r)
    db = {}

    def put(item):
        rel, src = item; dst = f"{O}/img/{rel}"
        if isinstance(src, tuple):
            con = db.setdefault(src[0], sqlite3.connect(f"file:{src[0]}?mode=ro", uri=True, check_same_thread=False))
            open(dst, "wb").write(con.execute("SELECT data FROM images WHERE id = ?", (src[1],)).fetchone()[0])
        else:
            shutil.copyfile(src, dst)
        return rel, os.path.getsize(dst)

    stores = [x for x in jobs.items() if isinstance(x[1], tuple)]; files = [x for x in jobs.items() if not isinstance(x[1], tuple)]
    size = dict(map(put, stores))   # one thread for the sqlite reads (a shared connection), 16 for plain file copies
    with ThreadPoolExecutor(16) as ex:
        size.update(ex.map(put, files))
    with open(f"{out}/train.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in pick)
    with open(f"{out}/train_images.tsv", "w") as f:
        f.writelines(f"{O}/img/{p}\t{p}\t{size[p]}\n" for p in sorted(jobs))
    json.dump(w, open(f"{out}/weights.json", "w"), indent=1)
    rep = {"from": D, "seed": 20261023, "rows": len(pick), "images": len(jobs), "store_images": len(stores), "gb": round(sum(size.values()) / 1e9, 2),
           "per_source": {s: {"all": len(rows[s]), "kept": keep[s], "qa_2g_kept": old[s]["kept"]} for s in sorted(rows)}}
    rep.update(stats(pick))
    json.dump(rep, open(f"{out}/report.json", "w"), indent=1); print(json.dumps(rep))


def stats(rows):
    n, by = len(rows), defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    h = lambda rs, f: {str(k): v for k, v in sorted(Counter(f(r) for r in rs).items())}
    return {"images_per_row": h(rows, lambda r: len(r["images"])), "answers_per_row": h(rows, lambda r: len(r["answers"])),
            "per_source_rows": {s: len(v) for s, v in sorted(by.items())}, "per_source_share": {s: round(len(v) / n, 5) for s, v in sorted(by.items())},
            "per_source_images_per_row": {s: h(v, lambda r: len(r["images"])) for s, v in sorted(by.items())},
            "per_source_answers_per_row": {s: h(v, lambda r: len(r["answers"])) for s, v in sorted(by.items())}}


if __name__ == "__main__":
    main(*sys.argv[1:3])
