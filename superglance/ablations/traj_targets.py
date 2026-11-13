"""The regression and text heads' targets: for every line of the imitation pool (the supervised pool built with the imitation navtrain
shard), the trajectory its softmax(-ADE / tau) target measured ADE against: 8 poses at 0.5-4 s, x / y / heading in the library's ego
frame. Sources, as their builders took them:
  navtrain (fit; heldout for dev) and if_navtrain_*   NAV_INPUTS (navtrain_inputs.jsonl) future_poses, tau 1 m
  navtrain with origin openscene_extra                 OSX_INPUTS (the OpenScene export's inputs.jsonl) future_poses, tau 1 m
  pai_human                                            PAI_INPUTS (the PAI-AV inputs dir, *.jsonl) future_poses, tau 0.5 m
  if_pai_*                                             PAI_FIT_ROWS (the PAI-AV fit rows of the instruction builder) future_poses, tau 0.5 m
  pai_dagger                                           E2E_INPUTS (e2e_inputs.npz) dag_E (train) / held_E (dev), the recovery label,
                                                       x / y only, tau 0.5 m
  wod and if_wod_pace                                  WOD_META (the WOD-E2E training rows' meta dir) future[1:16:2], x / y only, tau 1 m
L2D contrast rows (softmax(-ADE_perp) over S_c: no single trajectory) are dropped. Every other row is kept only if its stored target
equals softmax(-ADE / tau) to that trajectory on the target's own support (L2D hindsight targets are masked softmaxes), max |diff| <= TOL;
the check also counts gold_index == argmin ADE on the support. All recovered rows are checked, not a sample.
Out: <out>/{train,dev}_traj.npy (float32 [lines of the jsonl, 8, 3]; all NaN on QA and dropped lines; NaN heading where the source has
none) and report.json (per source: rows, kept, dropped by reason and share, heading missing, check statistics).
usage: POOL=<imitation pool> CANDS=<5,121 library> NAV_INPUTS=... OSX_INPUTS=... PAI_INPUTS=... PAI_FIT_ROWS=... E2E_INPUTS=... WOD_META=...
       python -m superglance.ablations.traj_targets <out dir>"""
import glob, json, os, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

POOL, CANDS, NAV, OSX, PAI, L2DPAI, E2E, WOD = (os.environ[k] for k in (
    "POOL", "CANDS", "NAV_INPUTS", "OSX_INPUTS", "PAI_INPUTS", "PAI_FIT_ROWS", "E2E_INPUTS", "WOD_META"))
TOL, B, NW = 2e-3, 256, 8


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def family(src):   # (reference map, tau in m) of a pool source; (None, None) for QA
    if src == "navtrain" or src.startswith("if_navtrain_"):
        return "nav", 1.0
    if src == "pai_human":
        return "pai", 0.5
    if src.startswith("if_pai_"):
        return "l2dpai", 0.5
    if src == "pai_dagger":
        return "dag", 0.5
    if src in ("wod", "if_wod_pace"):
        return "wod", 1.0
    return None, None


def poses3(path, key="future_poses"):   # {token: float32 [8, 3]} from a jsonl of input rows
    out = {}
    for line in open(path):
        r = json.loads(line); f = np.asarray(r[key], np.float32)
        assert f.shape == (8, 3), (path, r["token"], f.shape)
        out[r["token"]] = f
    return out


def wod_file(path):   # {token: [8, 3]}: future[1:16:2] (0.5-4 s of the 4 Hz logged future), heading NaN
    out = {}
    for line in open(path):
        r = json.loads(line); f = np.full((8, 3), np.nan, np.float32); f[:, :2] = np.asarray(r["future"], np.float32)[1:16:2]
        out[r["token"]] = f
    return out


def two(tok, E):   # npz labels [n, 8, 2] -> {token: [8, 3]}, heading NaN
    f = np.full((len(E), 8, 3), np.nan, np.float32); f[:, :, :2] = E
    return dict(zip(map(str, tok), f))


def check(job):   # sorted compact rows J of one split: per row max |stored q - softmax(-ADE / tau) on q's support|, gold == argmin, |support| == 1
    tpath, J, G, TAU, F = job
    C = np.load(CANDS)[:, :, :2].astype(np.float32); T = np.load(tpath, mmap_mode="r")
    d = np.empty(len(J)); ok = np.empty(len(J), bool); triv = np.empty(len(J), bool)
    for a in range(0, len(J), B):
        q = np.asarray(T[J[a:a + B]], np.float64); q /= q.sum(1, keepdims=True); S = q > 0
        ade = np.linalg.norm(C[None] - F[a:a + B, None, :, :2], axis=-1).mean(-1).astype(np.float64)
        z = np.where(S, -ade / TAU[a:a + B, None], -np.inf); z -= z.max(1, keepdims=True); e = np.exp(z); e /= e.sum(1, keepdims=True)
        d[a:a + B] = np.abs(e - q).max(1); ok[a:a + B] = np.where(S, ade, np.inf).argmin(1) == G[a:a + B]; triv[a:a + B] = S.sum(1) == 1
    return d, ok, triv


def main(out):
    os.makedirs(out, exist_ok=True)
    with ProcessPoolExecutor(NW) as ex:
        pai = {}
        for m in ex.map(poses3, sorted(glob.glob(f"{PAI}/*.jsonl"))):
            pai.update(m)
        wod = {}
        for m in ex.map(wod_file, sorted(glob.glob(f"{WOD}/*.jsonl"))):
            wod.update(m)
        nav, osx, l2dpai = ex.map(poses3, (NAV, OSX, L2DPAI))
    z = np.load(E2E)
    both = set(nav) & set(osx)   # openscene_export.py re-exports 200 navtrain fit tokens as its control rows
    assert len(both) <= 200 and all(np.array_equal(nav[t], osx[t]) for t in both), len(both)
    maps = {"nav": {**osx, **nav}, "pai": pai, "l2dpai": l2dpai, "wod": wod,
            "dag": {"train": two(z["dag_tok"], z["dag_E"]), "dev": two(z["held_tok"], z["held_E"])}}
    log("references:", {k: len(v) if k != "dag" else {s: len(x) for s, x in v.items()} for k, v in maps.items()})
    rep = {"pool": POOL, "library": CANDS, "tol_max_abs": TOL, "sources": {"navtrain": NAV, "openscene_extra": OSX, "pai_human": PAI,
           "if_pai": L2DPAI, "pai_dagger": E2E, "wod": WOD}}
    for split in ("train", "dev"):
        lines = 0; todo = []; cnt = defaultdict(Counter); src_of = []
        for i, line in enumerate(open(f"{POOL}/{split}.jsonl")):
            lines += 1; r = json.loads(line); s = r["source"]; f, tau = family(s)
            if f is None or "compact" not in r:
                continue
            cnt[s]["rows"] += 1
            if r.get("if_kind") == "contrast":
                cnt[s]["dropped_contrast"] += 1; continue
            F = (maps[f][split] if f == "dag" else maps[f]).get(r["token"])
            if F is None:
                cnt[s]["dropped_not_found"] += 1; continue
            todo.append((i, r["compact"]["row"], r["gold_index"], tau, F)); src_of.append(s)
        log(split, lines, "lines,", len(todo), "rows to check")
        o = np.argsort([t[1] for t in todo], kind="stable"); ch = [o[a:a + 16384] for a in range(0, len(o), 16384)]
        jobs = [(f"{POOL}/{split}_targets.npy", np.array([todo[k][1] for k in c]), np.array([todo[k][2] for k in c]),
                 np.array([todo[k][3] for k in c]), np.stack([todo[k][4] for k in c])) for c in ch]
        with ProcessPoolExecutor(NW) as ex:
            res = list(ex.map(check, jobs))
        d, ok, triv = (np.concatenate([r[x] for r in res]) for x in range(3)); idx = np.concatenate(ch)
        X = np.full((lines, 8, 3), np.nan, np.float32); stats = defaultdict(lambda: defaultdict(list))
        for k, dk, gk, tk in zip(idx, d, ok, triv):
            i, _, _, _, F = todo[k]; s = src_of[k]; st = stats[s]
            if dk <= TOL:
                X[i] = F; cnt[s]["kept"] += 1; cnt[s]["heading_missing"] += int(np.isnan(F[:, 2]).any())
                st["d"].append(dk); st["gold"].append(gk); st["trivial"].append(tk)
            else:
                cnt[s]["dropped_check"] += 1; st["fail_d"].append(dk)
        np.save(f"{out}/{split}_traj.part.npy", X); os.replace(f"{out}/{split}_traj.part.npy", f"{out}/{split}_traj.npy")
        R = {}
        for s in sorted(cnt):
            c = cnt[s]; st = stats[s]; dr = c["rows"] - c["kept"]
            R[s] = {**dict(c), "dropped": dr, "dropped_share": round(dr / c["rows"], 4),
                    "check": {"max_abs_max": float(np.max(st["d"])) if st["d"] else None,
                              "max_abs_p99": float(np.percentile(st["d"], 99)) if st["d"] else None,
                              "max_abs_median": float(np.median(st["d"])) if st["d"] else None,
                              "gold_is_argmin": round(float(np.mean(st["gold"])), 5) if st["gold"] else None,
                              "trivial_support": int(np.sum(st["trivial"])) if st["trivial"] else 0,
                              "failed_median_max_abs": float(np.median(st["fail_d"])) if st["fail_d"] else None}}
        tot = sum(c["rows"] for c in cnt.values()); kept = sum(c["kept"] for c in cnt.values())
        rep[split] = {"lines": lines, "trajectory_rows": tot, "kept": kept, "dropped_share": round(1 - kept / tot, 4), "by_source": R}
        log(split, json.dumps(rep[split]))
    json.dump(rep, open(f"{out}/report.json", "w"), indent=1)
    log("DONE", out)


if __name__ == "__main__":
    main(*sys.argv[1:2])
