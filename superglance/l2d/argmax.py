"""The argmax readout of a superglance.l2d.adapter run, from the same run: for each request, the library row of the traced
argmax through the adapter's own grid conversion, written in the SDK's export layout next to the SDK folder:
  RUN_ROOT/<mode>/argmax_from_trace/{rewrites.json, canonical.json}
The ledger, the trace and the SDK's rewrites.json align request by request (asserted). Also printed: how many requests' argmax
output equals the run's own output, and the distance between them at 4 s.
usage: python -m superglance.l2d.argmax RUN_ROOT MODE"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

R, mode = Path(sys.argv[1]), sys.argv[2]
from superglance.l2d import adapter as A  # noqa: E402

cfg = json.loads((R / "config" / f"model_{mode}.json").read_text())
D = R / mode / "official/navsim"; O = R / mode / "argmax_from_trace"
tr = [json.loads(l) for l in open(R / mode / "adapter_trace.jsonl")]
led = [json.loads(l) for l in open(D / "predictions.ledger.jsonl")]
w0 = json.loads((D / "rewrites.json").read_text())
assert len(tr) == len(led) == sum(len(v) for c in w0.values() for v in c.values()), (len(tr), len(led))
cands = np.load(cfg["cands"])
w, done, same, d4 = copy.deepcopy(w0), set(), 0, []
for t, g in zip(tr, led):
    m = g["meta"]; k = (m["condition"], m["pcu_id"], m["arm"])
    assert k not in done and t["instruction"] == g["instruction"], k
    e = w[k[0]][k[1]][k[2]]
    xy = A.to_grid(cands[t["argmax"]], e["timestamps"])
    own = np.asarray(w0[k[0]][k[1]][k[2]]["positions_xy"])
    same += bool(np.abs(own - np.asarray(xy)).max() <= 1e-6); d4.append(float(np.linalg.norm(own[-1] - np.asarray(xy[-1]))))
    e["positions_xy"] = xy; done.add(k)
O.mkdir(exist_ok=False)
(O / "rewrites.json").write_text(json.dumps(w, indent=2))
(O / "canonical.json").write_text(json.dumps(w["canonical"], indent=2))
print(json.dumps(dict(mode=mode, readout_of_run=cfg["readout"], requests=len(done), argmax_equals_run_output=same,
                      d4s_median_m=round(float(np.median(d4)), 3), d4s_max_m=round(max(d4), 3), out=str(O))))
