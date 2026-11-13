"""Rows of the refit's five feature sets (CPU): inputs, token lists and --index files (image -> 3 past frames oldest first,
or null) for superglance.eval.features, one set per extraction pass:
  nav      the navtrain fit scenes (--fit-tokens, 92,339: fit.py's split-0 rows) + held-out scenes (--heldout, 10,949) of the
           navtrain scene export; past frames by the supervised pool's NAVSIM rule (the history index, trainval split, then
           the fill index's) under --hist
  pai      the PAI-AV subset rows (--pai-subset, 48,236: fit.py's pilot/pai_subset rows), cam_f0 -> cam_f0_src; past frames by
           the pool's PAI-AV rule (clip crc32 % 2 == 0 and all three pai/<clip>/<ms>_<k>.jpg under --hist)
  navtest  12,146 navtest scenes; past frames by the history index's test split under --hist-test
  paival / pai700  the PAI-AV open-loop rows (superglance.eval.pai_rows) with their past-frame indexes
Every current and past frame must exist (exit 1 otherwise). Out: <out>/<set>_inputs.jsonl (nav and navtest: links to the scene
exports; the extraction keeps only the listed tokens), <set>_tokens.txt, <set>_index.jsonl, report.json.
usage: python -m superglance.refit.rows <out> --nav-inputs N --navtest-inputs T --hist-nav I [--hist-nav-fill F] --hist H
       --hist-test HT --fit-tokens FT --heldout HO --pai-subset P --pai-openloop O --hist-pai-ol HP"""
import argparse, json, os, zlib
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser(); A = ap.add_argument
A("out"); A("--nav-inputs", required=True); A("--navtest-inputs", required=True); A("--hist-nav", required=True); A("--hist-nav-fill")
A("--hist", required=True); A("--hist-test", required=True); A("--fit-tokens", required=True); A("--heldout", required=True)
A("--pai-subset", required=True); A("--pai-openloop", required=True); A("--hist-pai-ol", required=True)
a = ap.parse_args(); H, HT = a.hist, a.hist_test
SEL = lambda clip: zlib.crc32(clip.encode()) % 2 == 0   # mf_rows.SEL
out = a.out; os.makedirs(out, exist_ok=True)


def exists_all(paths):
    with ThreadPoolExecutor(32) as ex:
        ok = list(ex.map(lambda p: os.path.isfile(p) and os.path.getsize(p) > 0, paths))
    return [p for p, o in zip(paths, ok) if not o]


def write(name, rows, index, link=None):
    if link:
        if not os.path.lexists(f"{out}/{name}_inputs.jsonl"):
            os.symlink(link, f"{out}/{name}_inputs.jsonl")
    else:
        with open(f"{out}/{name}_inputs.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(f"{out}/{name}_tokens.txt", "w") as f:
        f.writelines(r["token"] + "\n" for r in rows)
    with open(f"{out}/{name}_index.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in index)


rep = {}

# nav: fit + held-out, history_rows.py's nav rule
nav = {}
for l in open(a.hist_nav):
    r = json.loads(l)
    if r["split"] == "trainval":
        nav[r["token"]] = r.get("past")
for l in (open(a.hist_nav_fill) if a.hist_nav_fill else []):
    r = json.loads(l)
    if r["split"] == "trainval" and r.get("past") and not nav.get(r["token"]):
        nav[r["token"]] = r["past"]
fit = [t.strip() for t in open(a.fit_tokens) if t.strip()]
held = [t.strip() for t in open(a.heldout) if t.strip()]
assert len(fit) == 92339 and len(held) == 10949 and not set(fit) & set(held), (len(fit), len(held))
scn = {}
for l in open(a.nav_inputs):
    r = json.loads(l); scn[r["token"]] = r
rows = [scn[t] for t in fit + held]
idx = [{"image": r["cam_f0"], "token": r["token"], "past": [f"{H}/{p}" for p in nav[r["token"]]] if nav.get(r["token"]) else None} for r in rows]
miss = exists_all([r["cam_f0"] for r in rows] + [p for x in idx if x["past"] for p in x["past"]])
rep["nav"] = {"rows": len(rows), "fit": len(fit), "held": len(held), "with_past": sum(x["past"] is not None for x in idx),
              "held_with_past": sum(x["past"] is not None for x in idx[len(fit):]), "missing_files": len(miss)}
print(json.dumps({"nav": rep["nav"]}), flush=True); assert not miss, miss[:3]
write("nav", rows, idx, link=os.path.abspath(a.nav_inputs))

# pai: fit.py's PAI subset rows, history_rows.py's PAI rule
rows = []
for l in open(a.pai_subset):
    r = json.loads(l); r["cam_f0_copy"] = r["cam_f0"]; r["cam_f0"] = r["cam_f0_src"]; rows.append(r)
idx = []
for r in rows:
    clip, ms = r["token"].rsplit("_", 1); p = [f"{H}/pai/{clip}/{ms}_{k}.jpg" for k in range(3)]
    idx.append({"image": r["cam_f0"], "token": r["token"], "past": p if SEL(clip) and all(map(os.path.isfile, p)) else None})
miss = exists_all([r["cam_f0"] for r in rows])
rep["pai"] = {"rows": len(rows), "with_past": sum(x["past"] is not None for x in idx), "missing_files": len(miss)}
print(json.dumps({"pai": rep["pai"]}), flush=True); assert not miss, miss[:3]
write("pai", rows, idx)

# navtest: the history index's test split (superglance.builders.pools.history_nav), frames under HT
past = {}
for l in open(a.hist_nav):
    r = json.loads(l)
    if r["split"] == "test":
        past[r["token"]] = r.get("past")
rows = [json.loads(l) for l in open(a.navtest_inputs)]
idx = [{"image": r["cam_f0"], "token": r["token"], "past": [f"{HT}/{q}" for q in past[r["token"]]] if past[r["token"]] else None} for r in rows]
miss = exists_all([r["cam_f0"] for r in rows] + [p for x in idx if x["past"] for p in x["past"]])
rep["navtest"] = {"rows": len(rows), "with_past": sum(x["past"] is not None for x in idx), "missing_files": len(miss)}
print(json.dumps({"navtest": rep["navtest"]}), flush=True); assert len(rows) == 12146 and not miss, (len(rows), miss[:3])
write("navtest", rows, idx, link=os.path.abspath(a.navtest_inputs))

# PAI val / 700: the open-loop rows (superglance.eval.pai_rows) and their past-frame indexes
for name, sp in (("paival", "val"), ("pai700", "test")):
    rows = [json.loads(l) for l in open(f"{a.pai_openloop}/pai_openloop_{sp}/pai_openloop_{sp}.jsonl")]
    toks = [t.strip() for t in open(f"{a.pai_openloop}/pai_openloop_{sp}/pai_openloop_{sp}_tokens.txt") if t.strip()]
    ix = {x["image"]: x for x in map(json.loads, open(f"{a.hist_pai_ol}/hist_pai_ol_{sp}/index.jsonl"))}
    idx = [{"image": r["cam_f0"], "token": r["token"], "past": ix[r["cam_f0"]]["past"]} for r in rows]
    miss = exists_all([r["cam_f0"] for r in rows] + [p for x in idx if x["past"] for p in x["past"]])
    rep[name] = {"rows": len(rows), "tokens": len(toks), "with_past": sum(x["past"] is not None for x in idx), "missing_files": len(miss)}
    print(json.dumps({name: rep[name]}), flush=True); assert set(toks) <= {r["token"] for r in rows} and not miss, miss[:3]
    write(name, [r for r in rows if r["token"] in set(toks)], [x for x in idx if x["token"] in set(toks)])
json.dump(rep, open(f"{out}/report.json", "w"), indent=1); print("ROWS_OK", json.dumps({k: v.get("rows") if isinstance(v, dict) else v for k, v in rep.items()}))
