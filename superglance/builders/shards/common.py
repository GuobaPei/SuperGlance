"""Pieces every stage-3 source shard shares; what fixes a prompt or a target is imported, not copied:
  prompt(r, drop)  the policy prompt of a scene row (superglance.data.scene.c256_row) after the training dropout of
                   superglance.builders.shards.dropout (drop: training rows only; acceleration 0 and constant-speed past positions)
  strip(q)         an earlier question with its road-users block (motion_lines.HEAD ... before the instruction) cut out
  soft             superglance.builders.pools.targets.soft: softmax(-ADE / tau) over the K = 5,121 candidates (8 poses at 0.5-4 s)
  bucket(r)        the PAI-AV sampling bucket, "event" or "plain" (superglance.builders.shards.buckets on the 4-s future)
  write(...)       one split of a shard: <dir>/<split>.jsonl, <split>_targets.npy (float32 [N, K]; a row's compact.row = its line)
                   and <split>_images.tsv (absolute source, relative path, bytes; one line per image). Rows hold the relative path:
                   the trainer opens <dir of the jsonl>/<images[0]>, so <dir>/<top dir> links to <SHARDS>/img/<top dir>.
env: SHARDS (the shards' root; default ./shards), CANDS (the 5,121-candidate library)."""
import json, os

import numpy as np

O, K = os.environ.get("SHARDS", "shards"), 5121
CANDS = os.environ.get("CANDS", "")
from superglance.data.scene import c256_row   # noqa: E402
from superglance.builders.shards import motion_lines as ml   # noqa: E402
from superglance.builders.shards.cv_history import cv_history   # noqa: E402
from superglance.builders.shards.dropout import dropped, scene_of   # noqa: E402, F401 (dropped: imported from here)
from superglance.builders.shards.buckets import aux_label, event   # noqa: E402
from superglance.builders.pools.targets import soft   # noqa: E402


def cands():
    return np.load(CANDS)[:, :, :2].astype(np.float64)


def prompt(r, drop):
    return c256_row(scene_of(r, None, cv_history, drop=drop), k=K)["question"]


def strip(q):
    i = q.find(ml.HEAD); b = q[i:len(q) - len(ml.INSTR)]
    assert i > 0 and q.endswith(ml.INSTR) and (b == ml.HEAD + ": none. " or (b.startswith(ml.HEAD + ml.UNITS) and b.endswith(". "))), q[-300:]
    return q[:i] + ml.INSTR


def bucket(r):
    st, _, u = aux_label(r["future_poses"])
    return "event" if event(st, u, float(np.hypot(*r["ego_velocity"][:2]))) else "plain"


def row(src, tok, q, rel, **kw):   # the earlier pools' row fields; gold and compact set by write()
    return {"question": q, "no_options": True, "fixed_order": True, "gold_index": 0, "id": f"{src}_{tok}", "token": tok, "source": src,
            "sampling_task": src, "sampling_bucket": "all", "images": [rel], "correct_answer": "", **kw}


def write(d, split, rows, F, tau, imgs, C):
    """rows, their labels F [N, 8, 2] (float32 values) -> targets soft(F, C, tau); imgs: rel path -> (abs src, bytes)."""
    os.makedirs(d, exist_ok=True)
    T = np.lib.format.open_memmap(f"{d}/{split}_targets.npy", "w+", np.float32, (len(rows), K))
    for a in range(0, len(rows), 4096):
        T[a:a + 4096] = soft(np.asarray(F[a:a + 4096], np.float32).astype(np.float64), C, tau)
    for j, r in enumerate(rows):
        g = int(np.argmax(T[j])); r.update(gold_index=g, correct_answer=f"t{g:04d}", compact={"k": K, "row": j})
    T.flush()
    with open(f"{d}/{split}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(f"{d}/{split}_images.tsv", "w") as f:
        f.writelines(f"{imgs[p][0]}\t{p}\t{imgs[p][1]}\n" for p in sorted({r["images"][0] for r in rows}))
    s = np.asarray(T, np.float64).sum(1)
    return {"rows": len(rows), "images": len({r["images"][0] for r in rows}), "gb": round(sum(imgs[p][1] for p in {r["images"][0] for r in rows}) / 1e9, 3),
            "target_sum_abs_err_max": float(np.abs(s - 1).max()) if len(s) else 0.0}


def link(d, top):   # <shard>/<top> -> ../img/<top>, so the shard's relative image paths resolve before and after staging
    os.makedirs(f"{O}/img/{top}", exist_ok=True)
    if not os.path.islink(f"{d}/{top}"):
        os.symlink(f"../img/{top}", f"{d}/{top}")
