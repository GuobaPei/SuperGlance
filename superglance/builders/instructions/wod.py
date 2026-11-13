"""Instruction pairs on WOD-E2E train (contracts.py for the rules): B2 pace only, frames where the human kept going. Rows:
superglance.builders.pools.wod_rows --sel every:5 on the training shards (its meta dir). Contract = L2D's WOD SPEED_CHOICE
(score_p0_kinematic_proxy.speed_contract): normal reference = the human's late speed v (mean 4 Hz speed over 2.75-5 s), slow
reference = v / 2 (training frames have no rater witnesses), so slow <= 0.625 v and normal >= 0.875 v, path >= 2 m.
Candidates are scored as the adapter sends them to the WOD track (superglance.l2d.adapter.to_grid: 0.25 ... 5.0 s, constant velocity past 4 s).
Kept going = late speed >= 2 m/s (gap >= 1 m/s, the contract's own floor) and tail speed >= 2 m/s. Feasible = the kinematic check.
Out: <out>/work/wod/pairs.jsonl, one line per frame with its pair (or none).
usage: WOD_META=<every:5 meta dir> python -m superglance.builders.instructions.wod <out>"""
import glob, json, os, sys
import numpy as np

from superglance.builders.instructions import contracts as ifc   # noqa: E402

sys.path.insert(0, f"{ifc.L2D}/ws/language2drive.OpenloopTrack.WOD-E2E/scripts")
META = os.environ["WOD_META"]
GRID = [0.25 * i for i in range(1, 21)]


def contract(v):   # L2D speed_contract with references (v / 2, v)
    lo, hi = v / 2, v; gap = hi - lo
    return {"family": "SPEED_CHOICE", "slow_arm": 1, "normal_arm": 0, "slow_reference_late_speed_mps": lo, "normal_reference_late_speed_mps": hi,
            "slow_accept_max_mps": lo + 0.25 * gap, "normal_accept_min_mps": hi - 0.25 * gap, "minimum_path_m": 2.0}


def wfeats(P20):   # score_p0_kinematic_proxy.features, vectorized: [N, 20, 2] -> late speed, path, tail speed
    d = np.diff(P20, axis=1, prepend=np.zeros_like(P20[:, :1])); s = np.hypot(d[..., 0], d[..., 1]) * 4.0
    return s[:, 11:].sum(1) / 9.0, s.sum(1) / 4.0, s[:, -4:].sum(1) / 4.0


def main(out):
    from superglance.l2d.adapter import to_grid
    import score_p0_kinematic_proxy as p0
    C = np.load(ifc.CANDS).astype(np.float64); KIN = ifc.kin_ok(C)
    G = np.array([to_grid(c, GRID) for c in C]); late, path, _ = wfeats(G)
    files = sorted(glob.glob(f"{META}/*.jsonl")); assert len(files) == 263 and all(os.path.basename(f).startswith("training_") for f in files)
    os.makedirs(f"{out}/work/wod", exist_ok=True); n = kept = 0
    with open(f"{out}/work/wod/pairs.jsonl.tmp", "w") as fo:
        for f in files:
            for l in open(f):
                r = json.loads(l); n += 1; F20 = np.asarray(r["future"], np.float64)
                hl, hp, ht = (x[0] for x in wfeats(F20[None])); res = {"token": r["token"], "segment": r["segment"], "shard": os.path.basename(f),
                                                                        "v0": float(r["speed"]), "late": float(hl), "tail": float(ht), "pairs": []}
                if hl >= 2.0 and ht >= 2.0:
                    c = contract(hl); ok_n = (late >= c["normal_accept_min_mps"]) & (path >= 2.0); ok_s = (late <= c["slow_accept_max_mps"]) & (path >= 2.0)
                    E, S = ok_n & ~ok_s, ok_s & ~ok_n & KIN
                    hn, _, _ = p0.primitive(p0.features(F20.tolist()), "normal", c); hs, _, _ = p0.primitive(p0.features(F20.tolist()), "slow", c)
                    if hn and not hs and S.sum() >= 3 and E.any():
                        res["pairs"].append({"family": "pace", "h": "normal", "c": "slow", "contract_h": c, "contract_c": c, "nE": int(E.sum()),
                                             "nS": int(S.sum()), "E": ifc.pack(E), "S": ifc.pack(S)}); kept += 1
                fo.write(json.dumps(res) + "\n")
    os.replace(f"{out}/work/wod/pairs.jsonl.tmp", f"{out}/work/wod/pairs.jsonl")
    print(json.dumps({"frames": n, "pairs": kept}))


if __name__ == "__main__":
    main(sys.argv[1])
