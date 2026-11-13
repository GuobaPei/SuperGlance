"""Instruction pairs on navtrain (contracts.py for the rules). Fit scenes only, minus L2D's 24 log segments / 19 recordings.
Per scene:
  feasible   PDM NC = DAC = 1 (candidates 0-4,096 of superglance.builders.pools.score_candidates' table) and the kinematic check
  A1 lane    nuPlan lanes: the lane under the rear axle (polygon within 1 m, heading within 30 deg, as L2D's nuScenes builder) and
             its same-direction neighbours (adjacent_edges), each extended along its straightest successors (< 35 deg) far enough
             ahead; keep = footprint in the own corridor at 0.5-4.0 s, change = in the neighbour's at 3.5-4.0 s; both x(4 s) >= 5 m
  A2 junction the first lane end along the own corridor (within 60 m) with connectors of >= 2 turn types (geometric: left > 30 deg,
             right < -30 deg, straight < 15 deg); region = the connectors of a type + their exit lanes; footprint in it at 3.5-4.0 s
  A3, B1, B3 contracts.long_pairs; the lead for B3 = the nearest logged vehicle ahead (3-30 m) that stays within 2 m of the
             human's path
Out: <out>/work/nav/<log>.jsonl, one line per scene with its candidate pairs (masks packed).
env: NAV_INPUTS (navtrain_inputs.jsonl), NAV_SCORES (score_candidates' output dir: shard_NNN.npz), OPENSCENE_DATA_ROOT, NUPLAN_MAPS_ROOT.
usage (navsim env): python -m superglance.builders.instructions.navtrain pdm <out>
                    python -m superglance.builders.instructions.navtrain label <out> [procs] [log ...]"""
import json, math, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np
import shapely

from superglance.builders.instructions import contracts as ifc   # noqa: E402

INP, PDM = os.environ["NAV_INPUTS"], os.environ["NAV_SCORES"]
LOGS = f"{os.environ['OPENSCENE_DATA_ROOT']}/navsim_logs/trainval"
G = {}


def pdm_shard(i):   # (NC == 1) & (DAC == 1) for the 4,097 candidates, packed
    z = np.load(f"{PDM}/shard_{i:03d}.npz"); s = z["scores"][:, :ifc.KP]   # the table's first 4,097 candidates
    return i, [str(t) for t in z["tokens"]], np.packbits((s[:, :, 0] == 1) & (s[:, :, 1] == 1) & (z["nonfinite"][:, None] == 0), axis=1)


def pdm(out):
    toks = [json.loads(l)["token"] for l in open(INP)]; M = np.zeros((len(toks), 513), np.uint8)
    with Pool(12) as p:
        for i, t, m in p.imap_unordered(pdm_shard, range((len(toks) + 999) // 1000)):
            assert t == toks[i * 1000:i * 1000 + len(t)]; M[i * 1000:i * 1000 + len(t)] = m
    os.makedirs(f"{out}/work", exist_ok=True); np.save(f"{out}/work/nav_pdm_ok.npy", M)
    print(json.dumps({"scenes": len(toks), "feasible_mean": float(np.unpackbits(M, axis=1)[:, :ifc.KP].mean())}))


def maps(name):
    if name not in G:
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
        G[name] = get_maps_api(os.environ["NUPLAN_MAPS_ROOT"], "nuplan-maps-v1.0", name)
    return G[name]


def turn_of(c):
    p = c.baseline_path.discrete_path; d = math.degrees(ifc.rem(p[-1].heading - p[0].heading))
    return "turn_left" if 30 < d < 150 else "turn_right" if -150 < d < -30 else "straight" if abs(d) < 15 else None


def nxt(o):   # the outgoing edge that turns least over its own baseline (< 35 deg), or None
    best = None
    for e in o.outgoing_edges:
        p = e.baseline_path.discrete_path; d = abs(ifc.rem(p[-1].heading - p[0].heading))
        if d < math.radians(35) and (best is None or d < best[0]):
            best = (d, e)
    return best[1] if best else None


def chain(o, ahead):   # o and its straightest successors until `ahead` metres of baseline from o's start
    objs, n = [o], o.baseline_path.length
    while n < ahead:
        e = nxt(objs[-1])
        if e is None or e.id in {x.id for x in objs}:
            break
        objs.append(e); n += e.baseline_path.length
    return objs


def region(objs, a):   # union of the objects' polygons in the anchor frame a = (x, y, yaw): one ring, 2 cm simplified, cm coordinates
    c, s = math.cos(a[2]), math.sin(a[2])
    g = shapely.transform(shapely.union_all([o.polygon for o in objs]), lambda P: (P - [a[0], a[1]]) @ np.array([[c, -s], [s, c]]))
    if g.geom_type != "Polygon":   # map pieces that do not quite touch: close gaps up to 1 m
        g = g.buffer(0.5).buffer(-0.5)
    if g.geom_type != "Polygon":
        return None
    p = shapely.Polygon(np.round(np.asarray(shapely.Polygon(g.exterior).simplify(0.02).exterior.coords), 2))
    if not p.is_valid:   # rounding can pinch the ring: repair once, keep the largest piece's outline
        q = p.buffer(0); q = max(getattr(q, "geoms", [q]), key=lambda x: x.area)
        p = shapely.Polygon(np.round(np.asarray(q.exterior.coords), 2))
    return np.asarray(p.exterior.coords).tolist() if p.is_valid and not p.is_empty else None


def lane_c(poly, start, prog):
    return {"type": "lane_region", "polygon_xy": poly, "start_s": start, "end_s": 4.0, "min_progress_m": prog}


def junction_pairs(cur, dist, a, H, meta):   # A2 from lane `cur` (dist = metres from the ego to its end): the first lane end with
    for _ in range(4):                         # connectors of >= 2 turn types within 60 m; region = those connectors + exit lanes
        grp = {}
        for c in cur.outgoing_edges:
            t = turn_of(c)
            if t:
                grp.setdefault(t, []).append(c)
        if len(grp) >= 2 or dist > 60:
            break
        c = nxt(cur); cur = nxt(c) if c is not None else None
        if cur is None:
            return []
        dist += c.baseline_path.length + cur.baseline_path.length
    if len(grp) < 2 or dist > 60:
        return []
    reg = {}
    for t, cs in grp.items():
        objs = []
        for c in cs:
            objs.append(c)
            for e in c.outgoing_edges:
                objs += chain(e, 40.0)
        poly = region(objs, a)
        if poly is not None:
            reg[t] = lane_c(poly, 3.5, 0.0)
    meta["junction"] = sorted(reg); sat = [t for t, v in reg.items() if ifc.holds(v, H)[0]]
    if len(sat) != 1:
        return []
    h = sat[0]
    return [("junction", h, cc, reg[h], reg[cc]) for cc in (["straight"] if h != "straight" else ["turn_left", "turn_right"]) if cc in reg]


def map_pairs(m, a, F2, X, H, FE):   # A1 lane + A2 junction
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    pt = Point2D(a[0], a[1]); best = None; objs = m.get_proximal_map_objects(pt, 2.0, [L.LANE, L.LANE_CONNECTOR])
    for layer in (L.LANE, L.LANE_CONNECTOR):
        for o in objs[layer]:
            if o.polygon.distance(shapely.Point(a[0], a[1])) >= 1.0:
                continue
            p = o.baseline_path.get_nearest_pose_from_position(pt)
            if abs(ifc.rem(p.heading - a[2])) <= math.radians(30) and (best is None or math.hypot(p.x - a[0], p.y - a[1]) < best[0]):
                best = (math.hypot(p.x - a[0], p.y - a[1]), layer == L.LANE, o)
    if best is None:
        return [], {"lane": None}
    if not best[1]:   # inside a connector: only the junction family, from its entry lane, when the ego is within 15 m of its start
        cn = best[2]; meta = {"lane": "connector"}
        if cn.baseline_path.get_nearest_arc_length_from_position(pt) <= 15.0 and cn.incoming_edges:
            return junction_pairs(cn.incoming_edges[0], 0.0, a, H, meta), meta
        return [], meta
    L0 = best[2]; out = []; meta = {"lane": L0.id}
    need = max(80.0, float(np.cumsum(H["s"][0])[-1] * 0.5) + 40.0)
    own = region(chain(L0, L0.baseline_path.get_nearest_arc_length_from_position(pt) + need), a)
    side = {}
    for e in L0.adjacent_edges:
        if e is not None:
            q = e.baseline_path.get_nearest_pose_from_position(pt)
            y = -math.sin(a[2]) * (q.x - a[0]) + math.cos(a[2]) * (q.y - a[1])
            poly = region(chain(e, e.baseline_path.get_nearest_arc_length_from_position(pt) + need), a)
            if poly is not None and 2.0 < abs(y) < 5.5:
                side["left" if y > 0 else "right"] = poly
    meta["sides"] = sorted(side)
    if own is not None and side:
        keep, chg = lane_c(own, 0.5, 5.0), {k: lane_c(v, 3.5, 5.0) for k, v in side.items()}
        if ifc.holds(keep, H)[0]:
            out += [("lane", "keep", k, keep, v) for k, v in chg.items()]
        else:
            hs = [k for k, v in chg.items() if ifc.holds(v, H)[0]]
            if len(hs) == 1:
                out.append(("lane", hs[0], "keep", chg[hs[0]], keep))
    out += junction_pairs(L0, L0.baseline_path.length - L0.baseline_path.get_nearest_arc_length_from_position(pt), a, H, meta)
    return out, meta


def tracks(frames, i0):   # logged vehicles at the anchor frame -> {track: [9, 2] centres at 0, 0.5, ..., 4 s in the anchor frame}
    E0i = np.linalg.inv(frames[i0]["ego2global"]); out = {}
    for k in range(9):
        an = frames[i0 + k]["anns"]; Mk = E0i @ frames[i0 + k]["ego2global"]
        for b, n, t in zip(an["gt_boxes"], an["gt_names"], an["track_tokens"]):
            if n != "vehicle" or (k and t not in out):
                continue
            out.setdefault(t, np.full((9, 2), np.nan))[k] = (Mk @ np.r_[b[:3], 1.0])[:2]
    return out


def lead_of(F2, trk):   # nearest vehicle ahead (3-30 m) staying within 2 m of the human's path at 0-4 s -> its 8 future centres
    path, best = ifc.human_path(F2), None
    for P in trk.values():
        if np.isnan(P).any() or not 3.0 < P[0, 0] <= 30.0:
            continue
        if ifc.path_dist(P, path).max() < 2.0 and (best is None or P[0, 0] < best[0, 0]):
            best = P
    return None if best is None else best[1:]


def work(job):
    log, rows, out = job
    dst = f"{out}/work/nav/{log}.jsonl"
    if os.path.exists(dst):
        return log, None
    t0 = time.time(); C = np.load(ifc.CANDS); X = ifc.feats(C[:, :, :2]); KIN = ifc.kin_ok(C)
    M = np.load(f"{out}/work/nav_pdm_ok.npy", mmap_mode="r")
    frames = pickle.load(open(f"{LOGS}/{log}.pkl", "rb")); at = {f["token"]: i for i, f in enumerate(frames)}
    m = maps(rows[0][1]["map"]); res = []
    for i, r in rows:
        i0 = at[r["token"]]
        if i0 + 8 >= len(frames):
            continue
        E = frames[i0]["ego2global"]; a = (E[0, 3], E[1, 3], math.atan2(E[1, 0], E[0, 0]))
        F2 = np.asarray(r["future_poses"], np.float64)[:, :2]; H = ifc.feats(F2[None])
        rel = np.linalg.inv(E) @ np.stack([frames[i0 + k]["ego2global"][:, 3] for k in range(1, 9)], 1)
        FE = np.zeros(ifc.K, bool); FE[:ifc.KP] = np.unpackbits(M[i])[:ifc.KP].astype(bool); FE &= KIN
        lead = lead_of(F2, tracks(frames, i0))
        cand = ifc.long_pairs(F2, FE, X, H, lead)
        mp, meta = map_pairs(m, a, F2, X, H, FE)
        pairs = [p for p in (ifc.judge(*q, X, H, FE) for q in cand + mp) if p is not None]
        res.append({"token": r["token"], "row": i, "log": log, "v0": float(np.hypot(*r["ego_velocity"][:2])),
                    "fut_err": float(np.abs(rel[:2].T - F2).max()), "lead": lead is not None, "map": meta, "pairs": pairs})
    with open(dst + ".tmp", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in res)
    os.replace(dst + ".tmp", dst)
    return log, (len(rows), sum(len(x["pairs"]) for x in res), round(time.time() - t0, 1))


def label(out, procs=24, only=()):
    l2d = json.load(open(f"{ifc.L2D}/meta/l2d_navsim_logs.json")); bad_log, bad_rec = set(l2d["l2d_logs"]), set(l2d["l2d_recordings"])
    by, skip = {}, {"heldout": 0, "l2d": 0}
    for i, l in enumerate(open(INP)):
        r = json.loads(l)
        if r["subset"] != "fit":
            skip["heldout"] += 1; continue
        if r["log_name"] in bad_log or r["log_name"].rsplit("_", 2)[0] in bad_rec:
            skip["l2d"] += 1; continue
        by.setdefault(r["log_name"], []).append((i, r))
    os.makedirs(f"{out}/work/nav", exist_ok=True); print(json.dumps({"logs": len(by), "scenes": sum(map(len, by.values())), "skipped": skip}), flush=True)
    jobs = sorted(((k, v, out) for k, v in by.items() if not only or k in only), key=lambda j: (j[1][0][1]["map"], j[0]))
    with Pool(procs) as p:
        for n, (log, st) in enumerate(p.imap_unordered(work, jobs), 1):
            print(n, log, st, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "pdm":
        pdm(sys.argv[2])
    else:
        label(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 24, set(sys.argv[4:]))
