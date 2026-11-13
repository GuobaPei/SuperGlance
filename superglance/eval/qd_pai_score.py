#!/usr/bin/env python3
"""ADE of Qwen-Drive-1.0 predictions on our PAI-AV rows (Table 6), with its own metric code (qwen_drive.metrics
open_loop_metrics). ADE@3 s / 5 s are means over poses 1-30 / 1-50: averaged over the samples (its ADE), the best sample
(minADE, an oracle) and sample 0 alone (a single draw). Each is read against two ground truths:
- theirs: the scene file's future_traj_10hz (online egomotion through the devkit interpolator);
- ours: our rows' fut10 (offline egomotion), the ground truth our own model is scored against.
Subsets: all rows, and T0 (the _ol+00 rows, one per clip = the official eval's fixed t0).
Table 6 reads the "GT ours all" line: one plan = sample 0, best of six = minADE.
usage (Qwen-Drive-1.0's environment): python -m superglance.eval.qd_pai_score <predictions.jsonl> <our rows.jsonl>"""
import json
import sys

import numpy as np
from qwen_drive.metrics import open_loop_metrics

pred = [json.loads(line) for line in open(sys.argv[1]) if line.endswith("\n")]
ours = {r["token"]: np.c_[np.asarray(r["fut10"]), np.zeros(50)] for r in map(json.loads, open(sys.argv[2]))}
assert {p["token"] for p in pred} <= set(ours), "predictions for tokens outside our rows"
for gt in ("theirs", "ours"):
    for subset in ("all", "T0"):
        rows = [p for p in pred if subset == "all" or p["token"].endswith("_ol+00")]
        m = []
        for p in rows:
            C = np.asarray(p["trajectories"])
            G = np.asarray(p["future_trajectory"]) if gt == "theirs" else ours[p["token"]]
            a, s = open_loop_metrics(C, G), open_loop_metrics(C[:1], G)
            m.append([a["ADE_3s"], a["ADE_5s"], a["minADE_3s"], a["minADE_5s"], s["ADE_3s"], s["ADE_5s"]])
        v = np.mean(m, 0)
        print(f"GT {gt:6s} {subset:3s} n {len(rows):4d} | avg of {C.shape[0]}: ADE {v[0]:.3f} / {v[1]:.3f} | minADE {v[2]:.3f} / {v[3]:.3f} | "
              f"sample 0: {v[4]:.3f} / {v[5]:.3f}  (m, 3 s / 5 s)")
