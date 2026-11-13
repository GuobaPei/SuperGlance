"""The 5,273-candidate library: rule-built candidates appended to the 5,121 (rows 0-5,120 unchanged; new rows from 5,121).
Families (traj.py shapes; headings = path tangent):
  brake   (a) brake to a stop and hold / hard brake, no human data: start speed V_A (m/s) x constant deceleration DEC_A (m/s^2), straight,
              plus mild constant-curvature variants (V_AC x DEC_AC x +-KAP_AC 1/m). Rule grid, kept in grid order.
  curve   (b) medium-speed curves and ramps, both directions: start speed 10-16 m/s x acceleration x y(4 s) 6-15 m x {arc, clo, ramp}
  fast    (c) > 16 m/s gentle curves: start speed 16-30 m/s x acceleration x y(4 s) 2-15 m x {arc, clo, ramp}
(b) and (c) are dense rule pools; the grid points are chosen by greedy coverage on TRAINING rows only (train_futures.npz: navtrain fit +
clean PAI train-split futures), 9 of 10 clips (crc32 of the PAI clip / navtrain token % 10 != 0; the tenth is a check split):
each step adds the pool candidate that brings the most rows from > 1 m to <= 1 m of the library (gain >= G_MIN rows, family budget).
Every family obeys the dedup rule: a new candidate is dropped if its mean pose distance to any existing candidate or any already-kept
new one is < 0.3 m; within the families the kept ones are at least SPACING (0.5 m) apart.
Out: <out>/cands_v2_new.npy [n, 8, 3] (float64, the library's dtype), <out>/cands_k<K'>_v2.npy (old + new; rows 0-5,120 asserted equal
to the 5,121 file), <out>/cands_v2_meta.json (family and parameters per new row), <out>/gen_report.json.
usage: python -m superglance.builders.candidates.extend <cands_k4097_fast1024.npy> <train_futures.npz> <train_nn5121.npz> <out dir>"""
import json, os, sys, zlib

import numpy as np

from superglance.builders.candidates.kin import ade_to, kin  # noqa: E402
from superglance.builders.candidates.traj import traj  # noqa: E402

DEDUP, SPACING, COVER = 0.3, 0.5, 1.0
V_A = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22]
DEC_A = [2.5, 3.0, 3.5, 4.5, 6.0]
V_AC, DEC_AC, KAP_AC = [6, 8, 10, 12, 14], [3.0, 4.5], [0.015, -0.015, 0.03, -0.03]
POOL_B = dict(v0=np.arange(10.0, 16.01, 0.5), acc=[-2.25, -1.5, -0.75, 0.0, 0.5], Y=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15], shapes=("arc", "clo", "ramp"))
POOL_C = dict(v0=np.arange(16.0, 30.01, 1.0), acc=[-1.0, 0.0, 0.5], Y=[2, 3, 4, 5, 6, 8, 10, 12, 15], shapes=("arc", "clo", "ramp"))
BUDGET = {"curve": 150, "fast": 60}
G_MIN = {"curve": 15, "fast": 30}


def brake_family():
    out = []
    for v in V_A:
        for a in DEC_A:
            out.append(({"family": "brake", "v0": v, "acc": -a, "shape": "straight", "kappa": 0.0}, traj(v, -a)))
    for v in V_AC:
        for a in DEC_AC:
            for k in KAP_AC:
                out.append(({"family": "brake", "v0": v, "acc": -a, "shape": "arc", "kappa": k}, traj(v, -a, "arc", A=k)))
    return out


def pool(fam, P):
    out = []
    for v in P["v0"]:
        for a in P["acc"]:
            S = float(v * 4 + 0.5 * a * 16)
            for Y in P["Y"]:
                if Y >= S / 1.5:
                    continue
                for sh in P["shapes"]:
                    for sg in (1, -1):
                        out.append(({"family": fam, "v0": float(v), "acc": float(a), "shape": sh, "Y": float(sg * Y)}, traj(float(v), float(a), sh, Y=sg * Y)))
    return out


def dist_min(A, B):   # min over B of the mean pose distance, for each row of A
    return ade_to(A, B).min(1) if len(B) else np.full(len(A), np.inf, np.float32)


def keep_rule(items, old, kept):   # grid order: keep unless < DEDUP to old / kept or < SPACING to this family's kept
    fam_kept, rep = [], {"tried": len(items), "drop_near_old": 0, "drop_near_new": 0, "drop_spacing": 0}
    for meta, P in items:
        if dist_min(P[None], old)[0] < DEDUP:
            rep["drop_near_old"] += 1; continue
        allnew = [q for _, q in kept] + [q for _, q in fam_kept]
        if allnew and dist_min(P[None], np.stack(allnew))[0] < DEDUP:
            rep["drop_near_new"] += 1; continue
        if fam_kept and dist_min(P[None], np.stack([q for _, q in fam_kept]))[0] < SPACING:
            rep["drop_spacing"] += 1; continue
        fam_kept.append((meta, P))
    rep["kept"] = len(fam_kept)
    return fam_kept, rep


def greedy(items, old, kept, F, cur, fam):
    """cur: the rows' current nearest distance (old + kept). Adds pool candidates by coverage gain; returns kept items, new cur, report"""
    Pc = np.stack([P for _, P in items])
    d_old = dist_min(Pc, old)
    d_new = dist_min(Pc, np.stack([q for _, q in kept])) if kept else np.full(len(Pc), np.inf, np.float32)
    ok = (d_old >= DEDUP) & (d_new >= DEDUP)
    U = np.where(cur > COVER)[0]
    D = ade_to(F[U], Pc)                 # [nU, pool]
    hit = D <= COVER
    cu = cur[U].copy(); fam_kept, gains = [], []
    PP = ade_to(Pc, Pc)                  # pool-pool distances for the spacing rule
    for _ in range(BUDGET[fam]):
        open_ = cu > COVER
        g = (hit & open_[:, None]).sum(0); g[~ok] = -1
        j = int(np.argmax(g))
        if g[j] < G_MIN[fam]:
            break
        fam_kept.append(items[j]); gains.append(int(g[j]))
        cu = np.minimum(cu, D[:, j])
        ok &= PP[j] >= SPACING            # spacing within the family (>= DEDUP to every kept one follows)
    new_cur = cur.copy(); new_cur[U] = cu
    return fam_kept, new_cur, {"pool": len(items), "pool_valid": int(((d_old >= DEDUP) & (d_new >= DEDUP)).sum()), "rows_open": int(len(U)),
                               "kept": len(fam_kept), "gains": gains}


def main(cands, tf, nn, out):
    C = np.load(cands); assert C.shape == (5121, 8, 3)
    z = np.load(tf); F = z["F"].astype(np.float64); src = z["src"]; toks = [str(t) for t in z["tokens"]]
    key = [t.rsplit("_", 1)[0] if s == 1 else t for t, s in zip(toks, src)]
    sel = np.array([zlib.crc32(k.encode()) % 10 != 0 for k in key])
    cur0 = np.load(nn)["ade_min"].astype(np.float64)
    kept, rep = [], {"rules": {"dedup_m": DEDUP, "spacing_m": SPACING, "cover_m": COVER, "V_A": V_A, "DEC_A": DEC_A, "V_AC": V_AC, "DEC_AC": DEC_AC,
                               "KAP_AC": KAP_AC, "POOL_B": {k: (list(map(float, v)) if not isinstance(v, tuple) else list(v)) for k, v in POOL_B.items()},
                               "POOL_C": {k: (list(map(float, v)) if not isinstance(v, tuple) else list(v)) for k, v in POOL_C.items()},
                               "budget": BUDGET, "gain_min_rows": G_MIN, "select_rows": int(sel.sum()), "check_rows": int((~sel).sum())}}
    a, rep["brake"] = keep_rule(brake_family(), C, kept); kept += a
    cur = cur0[sel].copy()
    if a:
        cur = np.minimum(cur, dist_min(F[sel], np.stack([P for _, P in a])))
    for fam, P in (("curve", POOL_B), ("fast", POOL_C)):
        b, cur, rep[fam] = greedy(pool(fam, P), C, kept, F[sel], cur, fam); kept += b
    N = np.stack([P for _, P in kept]).astype(C.dtype); meta = [m for m, _ in kept]
    for i, (m, P) in enumerate(zip(meta, N)):   # final checks: spacing, heading consistency, kinematics
        m["index"] = 5121 + i
    dn = ade_to(N, N); np.fill_diagonal(dn, np.inf); do = ade_to(N, C).min(1)
    assert do.min() >= DEDUP and dn.min() >= DEDUP, (do.min(), dn.min())
    allc = np.concatenate([C, N]); assert np.array_equal(allc[:5121], C)
    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/cands_v2_new.npy", N); np.save(f"{out}/cands_k{len(allc)}_v2.npy", allc)
    json.dump(meta, open(f"{out}/cands_v2_meta.json", "w"), indent=0)
    fam = np.array([m["family"] for m in meta]); k = kin(N)
    rep["result"] = {"n_new": len(N), "K_prime": len(allc), "by_family": {f: int((fam == f).sum()) for f in ("brake", "curve", "fast")},
                     "nearest_old_m": {"min": round(float(do.min()), 3), "median": round(float(np.median(do)), 3)},
                     "nearest_new_m": {"min": round(float(dn.min()), 3), "median": round(float(np.median(dn.min(1))), 3)},
                     "within_family_nn_median_m": {f: round(float(np.median(np.where(fam[None] == f, dn, np.inf)[fam == f].min(1))), 3)
                                                   for f in ("brake", "curve", "fast") if (fam == f).sum() > 1},
                     "brake_stopped": int(k["stopped"][fam == "brake"].sum()), "brake_v0_max_stopped": float(k["v0"][(fam == "brake") & k["stopped"]].max())}
    json.dump(rep, open(f"{out}/gen_report.json", "w"), indent=1); print(json.dumps(rep["result"])); print({f: rep[f] for f in ("brake", "curve", "fast")})


if __name__ == "__main__":
    main(*sys.argv[1:5])
