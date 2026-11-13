"""Instruction pairs on PAI-AV (contracts.py for the rules). Frames: the earlier pool's pai_human rows (images already staged)
whose clip is in the train split, not a RealSim scene and not one of the 916 AlpaSim clips. Per frame (rig frame at the frame time t_c):
  boxes      obstacle.offline tracks (every class), each sample moved from its rig frame to the anchor frame with the clip's
             egomotion (offline, else online; linear position, normalized-lerp quaternion); per track, the box at t_c + 0.5 k s by
             linear interpolation between samples <= 0.15 s away; tracks overlapping the ego footprint at t_c are dropped (counted)
  feasible   kinematic check and no overlap of the candidate footprint (own heading) with any box at 0.5 ... 4.0 s
  pairs      contracts.long_pairs (A3, B1, B3); the B3 lead = the nearest vehicle-class track ahead (3-30 m) within 2 m of the
             human's path at 0-4 s
Out: <out>/work/pai/<chunk>.jsonl, one line per frame.
env: PAI_ROOT (the PhysicalAI-AV root), PAI_FIT_ROWS (the PAI-AV fit rows), OLD_POOL (the earlier pool: train.jsonl), S916.
usage (navsim env): python -m superglance.builders.instructions.pai <out> [procs] [chunk ...]"""
import io, json, math, os, sys, time, zipfile
from multiprocessing import Pool
import numpy as np
import pandas as pd
import shapely

from superglance.builders.instructions import contracts as ifc   # noqa: E402

PA, FIT, POOL = os.environ["PAI_ROOT"], os.environ["PAI_FIT_ROWS"], os.environ["OLD_POOL"]
VEH = {"automobile", "heavy_truck", "trailer", "bus", "other_vehicle"}


def yaw_of(q):   # (x, y, z, w) -> heading
    return np.arctan2(2 * (q[..., 3] * q[..., 2] + q[..., 0] * q[..., 1]), 1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2))


def ego_at(eg, ts):   # egomotion (timestamp us, x, y, qx, qy, qz, qw) at times ts -> x, y, yaw
    t = eg[:, 0]; j = np.clip(np.searchsorted(t, ts) - 1, 0, len(t) - 2); a = np.clip((ts - t[j]) / (t[j + 1] - t[j]), 0, 1)[:, None]
    q0, q1 = eg[j, 3:7], eg[j + 1, 3:7]; q1 = np.where((q0 * q1).sum(1, keepdims=True) < 0, -q1, q1)
    xy = (1 - a) * eg[j, 1:3] + a * eg[j + 1, 1:3]
    return xy[:, 0], xy[:, 1], yaw_of((1 - a) * q0 + a * q1)


def rects(x, y, h, lo, wi, x0=0.0):   # [n, 5, 2] rectangles: centre (x, y), heading h, length lo, width wi; x0 = centre offset along h
    c, s = np.cos(h)[:, None], np.sin(h)[:, None]
    u = np.array([1, 1, -1, -1, 1]) * lo[:, None] / 2 + x0; v = np.array([-1, 1, 1, -1, -1]) * wi[:, None] / 2
    return np.stack([x[:, None] + c * u - s * v, y[:, None] + s * u + c * v], -1)


def clip_data(chunk, clip):
    def read(lab):
        return pd.read_parquet(io.BytesIO(zipfile.ZipFile(f"{PA}/labels/{lab}/{lab}.chunk_{chunk:04d}.zip").read(f"{clip}.{lab}.parquet")))
    eg = None
    for lab in ("egomotion.offline", "egomotion"):
        try:
            eg = read(lab).sort_values("timestamp").drop_duplicates("timestamp"); break
        except KeyError:
            pass
    eg = eg[["timestamp", "x", "y", "qx", "qy", "qz", "qw"]].to_numpy(np.float64)
    try:
        ob = read("obstacle.offline")
    except KeyError:
        ob = None
    return eg, ob


def frame(r, eg, ob, C, X, KIN):
    tc = float(r["timestamp_us"]); F2 = np.asarray(r["future_poses"], np.float64)[:, :2]; H = ifc.feats(F2[None])
    x0, y0, h0 = (v[0] for v in ego_at(eg, np.array([tc]))); c0, s0 = math.cos(h0), math.sin(h0)
    loc = lambda x, y: (c0 * (x - x0) + s0 * (y - y0), -s0 * (x - x0) + c0 * (y - y0))
    ex, ey, _ = ego_at(eg, tc + 0.5e6 * np.arange(1, 9)); fut_err = float(np.abs(np.stack(loc(ex, ey), 1) - F2).max())
    boxes = {k: [] for k in range(9)}; trk = {}; dropped = 0
    if ob is not None and len(ob):
        o = ob[(ob.timestamp_us >= tc - 0.3e6) & (ob.timestamp_us <= tc + 4.3e6)]
        ts = o.timestamp_us.to_numpy(np.float64); wx, wy, wh = ego_at(eg, ts)
        cx, cy = o.center_x.to_numpy(), o.center_y.to_numpy(); oh = yaw_of(o[["orientation_x", "orientation_y", "orientation_z", "orientation_w"]].to_numpy())
        gx, gy = wx + np.cos(wh) * cx - np.sin(wh) * cy, wy + np.sin(wh) * cx + np.cos(wh) * cy
        ax_, ay_ = loc(gx, gy); ah = ifc.rem(wh + oh - h0)
        df = pd.DataFrame({"tid": o.track_id.to_numpy(), "cls": o.label_class.to_numpy(), "t": ts, "x": ax_, "y": ay_, "h": ah,
                           "l": o.size_x.to_numpy(), "w": o.size_y.to_numpy()})
        foot0 = shapely.Polygon(ifc.FOOT)
        for tid, g in df.groupby("tid", sort=False):
            g = g.sort_values("t"); t, P = g.t.to_numpy(), g[["x", "y", "h", "l", "w"]].to_numpy()
            at = np.full((9, 5), np.nan)
            for k in range(9):
                tk = tc + 0.5e6 * k; j = np.searchsorted(t, tk)
                if 0 < j < len(t) and tk - t[j - 1] <= 0.15e6 and t[j] - tk <= 0.15e6:
                    a = (tk - t[j - 1]) / (t[j] - t[j - 1]); at[k] = (1 - a) * P[j - 1] + a * P[j]
                    at[k, 2] = P[j - 1, 2] + a * ifc.rem(P[j, 2] - P[j - 1, 2])
                elif j < len(t) and abs(t[j] - tk) <= 0.06e6:
                    at[k] = P[j]
                elif j > 0 and abs(tk - t[j - 1]) <= 0.06e6:
                    at[k] = P[j - 1]
            if not np.isnan(at[0, 0]) and shapely.Polygon(rects(at[:1, 0], at[:1, 1], at[:1, 2], at[:1, 3], at[:1, 4])[0]).intersects(foot0):
                dropped += 1; continue
            for k in range(9):
                if not np.isnan(at[k, 0]):
                    boxes[k].append(at[k])
            if g.cls.iloc[0] in VEH:
                trk[tid] = at[:, :2]
    coll = np.zeros(ifc.K, bool); hum = False
    for k in range(1, 9):
        if boxes[k]:
            b = np.array(boxes[k]); tree = shapely.STRtree(shapely.polygons(rects(b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4])))
            fp = shapely.polygons(rects(C[:, k - 1, 0], C[:, k - 1, 1], C[:, k - 1, 2], np.full(ifc.K, 4.8), np.full(ifc.K, 2.0), 1.0))
            coll[np.unique(tree.query(fp, predicate="intersects")[0])] = True
            hum |= len(tree.query(shapely.polygons(rects(F2[k - 1:k, 0], F2[k - 1:k, 1], H["h"][0, k - 1:k], np.array([4.8]), np.array([2.0]), 1.0)),
                                  predicate="intersects")[0]) > 0
    path = ifc.human_path(F2); lead = None
    for P in trk.values():
        if not np.isnan(P).any() and 3.0 < P[0, 0] <= 30.0 and ifc.path_dist(P, path).max() < 2.0 and (lead is None or P[0, 0] < lead[0, 0]):
            lead = P
    FE = KIN & ~coll
    pairs = [p for p in (ifc.judge(*q, X, H, FE) for q in ifc.long_pairs(F2, FE, X, H, None if lead is None else lead[1:])) if p is not None]
    return {"token": r["token"], "clip": r["log_name"], "v0": float(np.hypot(*r["ego_velocity"][:2])), "fut_err": fut_err, "n_box": len(boxes[1]),
            "dropped": dropped, "coll_frac": float(coll.mean()), "hum_coll": bool(hum), "lead": lead is not None, "pairs": pairs}


def work(job):
    chunk, rows, out = job
    dst = f"{out}/work/pai/{chunk:04d}.jsonl"
    if os.path.exists(dst):
        return chunk, None
    t0 = time.time(); C = np.load(ifc.CANDS).astype(np.float64); X = ifc.feats(C[:, :, :2]); KIN = ifc.kin_ok(C); res = []; err = 0
    by = {}
    for r in rows:
        by.setdefault(r["log_name"], []).append(r)
    for clip, rs in by.items():
        try:
            eg, ob = clip_data(chunk, clip)
        except Exception:
            err += len(rs); continue
        res += [frame(r, eg, ob, C, X, KIN) for r in rs]
    with open(dst + ".tmp", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in res)
    os.replace(dst + ".tmp", dst)
    return chunk, (len(rows), err, sum(len(x["pairs"]) for x in res), round(time.time() - t0, 1))


def main(out, procs=24, only=()):
    idx = pd.read_parquet(f"{PA}/clip_index.parquet"); split, chunk = dict(zip(idx.index, idx.split)), dict(zip(idx.index, idx.chunk))
    realsim = {l.strip() for l in open(f"{ifc.L2D}/meta/realsim_scene_uuids.txt") if l.strip()}
    p2601 = {l.strip().replace("clipgt-", "") for l in open(os.environ["S916"]) if l.strip()}
    pool = {json.loads(l)["token"] for l in open(f"{POOL}/train.jsonl") if '"source": "pai_human"' in l}
    by, skip = {}, {"not_in_pool": 0, "not_train": 0, "realsim": 0, "public_2601": 0}
    for l in open(FIT):
        r = json.loads(l); c = r["log_name"]
        why = ("not_in_pool" if r["token"] not in pool else "not_train" if split.get(c) != "train" else "realsim" if c in realsim
               else "public_2601" if c in p2601 else None)
        if why:
            skip[why] += 1; continue
        by.setdefault(int(chunk[c]), []).append(r)
    os.makedirs(f"{out}/work/pai", exist_ok=True)
    print(json.dumps({"chunks": len(by), "frames": sum(map(len, by.values())), "skipped": skip}), flush=True)
    jobs = [(c, rs, out) for c, rs in sorted(by.items()) if not only or c in only]
    with Pool(procs) as p:
        for n, (c, st) in enumerate(p.imap_unordered(work, jobs), 1):
            print(n, c, st, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 24, {int(x) for x in sys.argv[3:]})
