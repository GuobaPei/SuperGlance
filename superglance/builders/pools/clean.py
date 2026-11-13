"""A stage-3 pool without any training row from an evaluation set. Drops, from <pool>/train.jsonl:
  PAI rows (pai_dagger: the clip of its scene from <e2e_inputs.npz> dag_tok -> dag_scene; pai_human and if_pai_*: the token's clip):
    clips of the PAI evaluation rows (PAI-AV 700, PAI val 399, the 644 gold list incl. its 325 train-split clips), the AlpaSim 916,
    the PAI-AV test split, the earlier closed-loop val suite (nurec_curated_val). PAI val-split clips outside the evaluations stay
    and are counted.
  NAVSIM rows (navtrain incl. the OpenScene extras, openscene, if_navtrain_*): navtest tokens, and rows from navtest logs.
  WOD rows (wod, if_wod_*): WOD-E2E val and test segments.
QA replay rows are not checked here. dev.jsonl is copied unchanged (dev rows are never trained on); its protected rows are counted.
Targets are untouched: rows index them by compact.row, so the clean pool links the same train_targets.npy / dev_targets.npy (hard
link, else symlink). Out: <out>/{train.jsonl, dev.jsonl, train_targets.npy, dev_targets.npy, [copies.tsv], report.json (+ "clean")};
the pool's per-shard "train" counts are recomputed when the rows follow its shard order. Re-reads the output and asserts 0
protected rows.
RL pools: RENUM=1 renumbers the kept rows' compact rows 0..n-1 and writes the row-indexed arrays (train_targets.npy,
train_rewards.npy) for the kept rows only; KEEP_WOD_VAL=1 keeps WOD-E2E val rows (the RL pool's WOD reward rows; WOD is then
reported on test). Images staged elsewhere are traced back through copies.tsv. The RL files (rewards state, nav_tokens.txt,
held_blocks.json, images_mode) are linked; nav_tokens.txt is rewritten without dropped navtrain tokens.
usage: [RENUM=1] [KEEP_WOD_VAL=1] python -m superglance.builders.pools.clean <pool dir> <out dir> <eval_keys.json> <e2e_inputs.npz>
       <WOD splits dir>"""
import json, os, shutil, sys
from collections import Counter

import numpy as np

P, O, K, Z, WS = sys.argv[1:6]
RENUM, KEEP_WOD_VAL = os.environ.get("RENUM") == "1", os.environ.get("KEEP_WOD_VAL") == "1"
k = json.load(open(K))
pai_eval = set().union(*map(set, k["pai_eval"].values()), k["gold_test"], k["gold_val"], k["gold_other"])
pai_drop = [("pai_eval", pai_eval), ("alpasim_916", set(k["alpasim_916"])), ("pai_test_split", set(k["pai_test"])),
            ("curated_val", set(k["curated_val"]))]
pai_val = set(k["pai_val"])
nav_tok, nav_log = set(k["navtest_tokens"]), set(k["navtest_logs"])
wod_files = ("test_segments.txt",) if KEEP_WOD_VAL else ("val_segments.txt", "test_segments.txt")
wod_drop = {s.strip() for f in wod_files for s in open(f"{WS}/{f}") if s.strip()}
assert len(wod_drop) == (0 if KEEP_WOD_VAL else 479) + 1505 and len(nav_log) == 136, (len(wod_drop), len(nav_log))
copies = {}   # staged image -> source path
if os.path.exists(f"{P}/copies.tsv"):
    for l in open(f"{P}/copies.tsv"):
        s, d = l.rstrip("\n").split("\t"); copies[d] = s
z = np.load(Z)
dag = {t: s.split("clipgt-", 1)[1] for t, s in zip(z["dag_tok"].tolist(), z["dag_scene"].tolist())}


def why(r):
    """(family, reason or None, note): reason = why the row must leave training; note = kept but counted (PAI val split)."""
    s, t = r["source"], r.get("token", "")
    if s == "pai_dagger" or s == "pai_human" or s.startswith("if_pai"):
        c = dag.get(t) if s == "pai_dagger" else t.rsplit("_", 1)[0]
        if c is None:   # dev's held-out pai_dagger rows have no scene in e2e_inputs.npz; a train row must have one (asserted below)
            return "pai", None, "pai_dagger_unmapped"
        for name, keys in pai_drop:
            if c in keys:
                return "pai", name, None
        return "pai", None, "pai_val_split" if c in pai_val else None
    if s in ("navtrain", "openscene") or s.startswith("if_navtrain"):
        parts = [p for im in r["images"] for p in copies.get(im, im).split("/")]   # every image (rows with history add past frames)
        if t in nav_tok:
            return "nav", "navtest_token", None
        if any(p in nav_log for p in parts):
            return "nav", "navtest_log", None
        return "nav", None, None if s.startswith("if_navtrain") or any(p[:4] == "2021" for p in parts) else "no_log"
    if s == "wod" or s.startswith("if_wod"):
        return "wod", ("wod_val_test" if t.rsplit("-", 1)[0] in wod_drop else None), None
    return "other", None, None


old, dnav = [], set()   # RENUM: the kept rows' compact rows in the source pool; navtrain tokens dropped


def scan(path, write=None):
    drop, note, fam, n = Counter(), Counter(), Counter(), 0
    out = open(write, "w") if write else None
    idx = []   # (row number, kept) for the shard recount
    for line in open(path):
        r = json.loads(line); f, reason, nt = why(r); n += 1; fam[f] += 1
        if nt:
            note[(r["source"] if not r["source"].startswith("if_") else "l2d", nt)] += 1
        if reason:
            drop[(r["source"] if not r["source"].startswith("if_") else "l2d:" + r["source"].split("_")[1], reason)] += 1
            if out and f == "nav":
                dnav.add(r["token"])
        elif out:
            if RENUM:
                old.append(r["compact"]["row"]); r["compact"]["row"] = len(old) - 1; line = json.dumps(r) + "\n"
            out.write(line)
        idx.append(reason is None)
    if out:
        out.close()
    return n, drop, note, fam, idx


def take(src, dst, rows):   # a row-indexed array -> the kept rows (RL-2's rewards: 391,964 x 5,121 float32, 8 GB), in chunks
    a = np.load(src, mmap_mode="r"); b = np.lib.format.open_memmap(dst + ".tmp", "w+", a.dtype, (len(rows),) + a.shape[1:])
    for i in range(0, len(rows), 20000):
        b[i:i + 20000] = a[rows[i:i + 20000]]
    b.flush(); del b; os.replace(dst + ".tmp", dst)


def link(src, dst):
    src = os.path.realpath(src)
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(src, dst)


os.makedirs(O, exist_ok=True)
n, drop, note, fam, keep = scan(f"{P}/train.jsonl", f"{O}/train.jsonl.tmp")
assert not note[("pai_dagger", "pai_dagger_unmapped")], "train pai_dagger rows without a scene"
os.replace(f"{O}/train.jsonl.tmp", f"{O}/train.jsonl")
dn, ddrop, _, _, _ = scan(f"{P}/dev.jsonl")
shutil.copyfile(f"{P}/dev.jsonl", f"{O}/dev.jsonl")
for f in ("train_targets.npy", "train_rewards.npy", "dev_targets.npy", "copies.tsv", "train_rewards.state", "nav_tokens.txt",
          "held_blocks.json", "images_mode"):
    if not os.path.exists(f"{P}/{f}") or os.path.exists(f"{O}/{f}"):
        continue
    if RENUM and f in ("train_targets.npy", "train_rewards.npy"):
        take(f"{P}/{f}", f"{O}/{f}", np.array(old))
    elif f == "nav_tokens.txt" and dnav:
        open(f"{O}/{f}", "w").writelines(l for l in open(f"{P}/{f}") if l.strip() not in dnav)
    else:
        link(f"{P}/{f}", f"{O}/{f}")
rep = json.load(open(f"{P}/report.json"))
tr = rep.get("train")
if isinstance(tr, dict) and all(isinstance(v, int) for v in tr.values()) and sum(tr.values()) == n:   # e2e pool: rows in shard order
    new, i = {}, 0
    for shard, m in tr.items():
        new[shard] = sum(keep[i:i + m]); i += m
    rep["train_before_clean"], rep["train"] = tr, new
m = sum(keep)
rep["clean"] = {"from": os.path.realpath(P), "rows_before": n, "rows_after": m, "dropped": sum(drop.values()),
                "dropped_by_source_reason": {f"{a} | {b}": v for (a, b), v in sorted(drop.items())},
                "kept_but_counted": {f"{a} | {b}": v for (a, b), v in sorted(note.items())},
                "rows_by_family": dict(fam), "dev_rows": dn, "dev_protected_rows_kept": {f"{a} | {b}": v for (a, b), v in sorted(ddrop.items())},
                "protected": "PAI eval clips (700 / val 399 / 644 incl. 325 train-split), AlpaSim 916, PAI test split, nurec_curated_val; "
                             f"navtest tokens and logs; WOD-E2E {'test' if KEEP_WOD_VAL else 'val + test'} segments",
                "renumbered": RENUM, "keys": os.path.realpath(K)}
json.dump(rep, open(f"{O}/report.json.tmp", "w"), indent=1); os.replace(f"{O}/report.json.tmp", f"{O}/report.json")
n2, drop2, _, _, _ = scan(f"{O}/train.jsonl")
assert n2 == m and not drop2, (n2, m, drop2)
if RENUM:   # rows 0..m-1, and the arrays hold the source rows
    assert [json.loads(l)["compact"]["row"] for l in open(f"{O}/train.jsonl")] == list(range(m))
    for f in ("train_targets.npy", "train_rewards.npy"):
        if os.path.exists(f"{P}/{f}"):
            a, b = np.load(f"{P}/{f}", mmap_mode="r"), np.load(f"{O}/{f}", mmap_mode="r"); assert b.shape[0] == m, (f, b.shape)
            for i in np.random.default_rng(0).choice(m, min(m, 2000), replace=False).tolist() + [0, m - 1]:
                assert np.array_equal(a[old[i]], b[i]), (f, i)
print(json.dumps(rep["clean"]))
