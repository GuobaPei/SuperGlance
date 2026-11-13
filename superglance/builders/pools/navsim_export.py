"""NAVSIM v1.1 scene export: for every scene of the navtrain or navtest scene filter, the official metric cache (navsim
MetricCacheProcessor: PDM-Closed trajectory, 51-frame 10 Hz observation, full-precision map polygons) and one input row (front /
left / right camera paths, ego velocity and acceleration, driving command, 4 history and 8 future poses, the metric cache path),
both from the same navsim Scene. These rows are what superglance.data.scene.load_scenes, the pool builders and the NAVSIM
evaluation read. Resumable: existing cache files are kept, never regenerated.
env: OPENSCENE_DATA_ROOT, NUPLAN_MAPS_ROOT, NAVSIM_DEVKIT_ROOT (the navsim package dir), D (output root; inputs/ and status/
inside), HELDOUT_TOKENS (navtrain: the held-out token list, one per line; those scenes get subset "heldout", the rest "fit").
usage (navsim env): python -m superglance.builders.pools.navsim_export --split navtrain|navtest [--nproc 40]"""
import os, re, time, json, pickle, argparse, warnings
import numpy as np
from pathlib import Path
from multiprocessing import Pool

B = os.environ["OPENSCENE_DATA_ROOT"]; D = os.environ["D"]
ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=["navtrain", "navtest"])
ap.add_argument("--nproc", type=int, default=40)
ap.add_argument("--chunk", type=int, default=40)
ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N tokens")
ap.add_argument("--tag", default="", help="suffix for output names (smoke tests)")
args = ap.parse_args()
SPLIT, TAG = args.split, args.split + args.tag
CFG = f"{os.environ['NAVSIM_DEVKIT_ROOT']}/planning/script/config/common/train_test_split/scene_filter/{SPLIT}.yaml"
if SPLIT == "navtrain":
    LOGS, BLOBS = B + "/navsim_logs/trainval", B + "/sensor_blobs/trainval"
else:
    LOGS, BLOBS = B + "/navsim_logs/test", B + "/sensor_blobs/test"
CACHE = f"{D}/metric_cache_v11/{SPLIT}"
CAMS = ["CAM_F0", "CAM_L0", "CAM_R0"]
MAPS = os.environ["NUPLAN_MAPS_ROOT"]
_G = {}


def parse_filter(path):
    """log_names / tokens lists of a navsim scene_filter yaml (regex: yaml would turn all-digit tokens into ints)."""
    sec, out = None, {"log_names": [], "tokens": []}
    for line in open(path):
        m = re.match(r"^(\w+):", line)
        if m:
            sec = m.group(1); continue
        m = re.match(r"^\s*-\s*['\"]?([^'\"\s]+)['\"]?\s*$", line)
        if m and sec in out:
            out[sec].append(m.group(1))
    return out


def scan_log(lg):
    frames = pickle.load(open(f"{LOGS}/{lg}.pkl", "rb"))
    return lg, [f["token"] for f in frames]


def work(task):
    tid, lg, toks, subs = task
    subset = dict(zip(toks, subs))
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig, Scene
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    if "proc" not in _G:
        from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
        _G["proc"] = MetricCacheProcessor(cache_path=CACHE, force_feature_computation=False)
    t0 = time.time()
    out = []
    try:
        sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True, max_scenes=None,
                         log_names=[lg], tokens=list(toks))
        sl = SceneLoader(data_path=Path(LOGS), sensor_blobs_path=None, scene_filter=sf, sensor_config=SensorConfig.build_no_sensors())
        sdicts = sl.scene_frames_dicts
    except Exception as ex:
        return tid, [dict(token=tk, log_name=lg, status="log_load_error", err=repr(ex)[:300]) for tk in toks], time.time() - t0
    for tk in toks:
        rec = dict(token=tk, log_name=lg)
        sd = sdicts.get(tk)
        if sd is None:
            rec["status"] = "missing_scene"; out.append(rec); continue
        try:
            assert sd[3]["token"] == tk and len(sd) == 14, (sd[3]["token"], len(sd))
            scene = Scene.from_scene_dict_list(sd, None, num_history_frames=4, num_future_frames=10,
                                               sensor_config=SensorConfig.build_no_sensors())
            ai = scene.get_agent_input()
            fut = np.asarray(scene.get_future_trajectory(8).poses, dtype=np.float64)
            hist = np.stack([np.asarray(e.ego_pose, dtype=np.float64) for e in ai.ego_statuses])
            vel = np.stack([np.asarray(e.ego_velocity, dtype=np.float64) for e in ai.ego_statuses])
            acc = np.stack([np.asarray(e.ego_acceleration, dtype=np.float64) for e in ai.ego_statuses])
            cmd = np.asarray(ai.ego_statuses[-1].driving_command, dtype=np.int64)
            cams = {}
            for c in CAMS:
                rel = sd[3]["cams"][c]["data_path"]
                cams[c] = (BLOBS + "/" + str(rel), os.path.exists(BLOBS + "/" + str(rel)))
            scen = NavSimScenario(scene, map_root=MAPS, map_version="nuplan-maps-v1.0")
            cf = f"{CACHE}/{scen.log_name}/{scen.scenario_type}/{scen.token}/metric_cache.pkl"
            if os.path.exists(cf):
                cstat = "existing"
            else:
                e = _G["proc"].compute_metric_cache(scen)
                cstat = "generated" if (e is not None and os.path.exists(cf)) else "cache_failed"
            rec.update(status="ok" if cstat != "cache_failed" else "cache_failed", cache_status=cstat, metric_cache=cf,
                       cache_bytes=os.path.getsize(cf) if os.path.exists(cf) else 0,
                       map=scene.scene_metadata.map_name, subset=subset.get(tk, SPLIT), timestamp_us=int(sd[3]["timestamp"]),
                       cams=cams, fut=fut, hist=hist, vel=vel, acc=acc, cmd=cmd)
        except Exception:
            import traceback
            rec.update(status="error", err=traceback.format_exc()[-600:])
        out.append(rec)
    return tid, out, time.time() - t0


def r(a, nd):
    return [round(float(x), nd) for x in np.ravel(a)]


def to_row(rec):
    return dict(
        token=rec["token"], log_name=rec["log_name"], map=rec["map"], split=SPLIT, subset=rec["subset"], timestamp_us=rec["timestamp_us"],
        cam_f0=rec["cams"]["CAM_F0"][0], cam_l0=rec["cams"]["CAM_L0"][0], cam_r0=rec["cams"]["CAM_R0"][0],
        cams_exist=[bool(rec["cams"][c][1]) for c in CAMS],
        ego_velocity=r(rec["vel"][-1], 4), ego_acceleration=r(rec["acc"][-1], 4), driving_command=[int(x) for x in rec["cmd"]],
        history_poses=[r(p, 5) for p in rec["hist"]], history_velocity=[r(v, 4) for v in rec["vel"]],
        history_acceleration=[r(a, 4) for a in rec["acc"]], future_poses=[r(p, 5) for p in rec["fut"]],
        metric_cache=rec["metric_cache"])


def main():
    t0 = time.time()
    os.makedirs(CACHE + "/metadata", exist_ok=True)
    flt = parse_filter(CFG)
    toks_cfg = flt["tokens"]
    assert all(re.fullmatch(r"[0-9a-f]{16}", t) for t in toks_cfg), "unexpected token format"
    tokset = set(toks_cfg)
    print(f"{SPLIT}: filter has {len(flt['log_names'])} logs, {len(toks_cfg)} tokens ({len(tokset)} unique)", flush=True)
    subset = {}
    if SPLIT == "navtrain":
        held = {l.strip() for l in open(os.environ["HELDOUT_TOKENS"]) if l.strip()}
        subset = {tk: ("heldout" if tk in held else "fit") for tk in toks_cfg}
    with Pool(min(args.nproc, len(flt["log_names"]))) as pool:
        per_log = {}
        for lg, ftoks in pool.imap_unordered(scan_log, flt["log_names"]):
            hit = [t for t in ftoks if t in tokset]
            if hit:
                per_log[lg] = hit
    mapped = set(t for v in per_log.values() for t in v)
    unmapped = sorted(tokset - mapped)
    print(f"mapped {len(mapped)} tokens to {len(per_log)} logs; unmapped {len(unmapped)} ({time.time()-t0:.0f}s)", flush=True)
    tasks = []
    for lg in sorted(per_log):
        tl = sorted(set(per_log[lg]))
        for s in range(0, len(tl), args.chunk):
            ch = tl[s:s + args.chunk]
            tasks.append((len(tasks), lg, ch, [subset.get(t, SPLIT) for t in ch]))
    if args.limit:
        keep, cnt = [], 0
        for t in tasks:
            if cnt >= args.limit:
                break
            keep.append(t); cnt += len(t[2])
        tasks = keep
    ntok = sum(len(t[2]) for t in tasks)
    print(f"{len(tasks)} tasks, {ntok} tokens, nproc {args.nproc}", flush=True)
    recs, st = [], {}
    last = time.time()
    with Pool(args.nproc) as pool:
        for i, (tid, out, dt) in enumerate(pool.imap_unordered(work, tasks)):
            recs += out
            for rc in out:
                k = rc.get("cache_status", rc["status"]); st[k] = st.get(k, 0) + 1
            if time.time() - last > 60 or i + 1 == len(tasks):
                last = time.time(); el = last - t0
                prog = dict(split=TAG, tasks_done=i + 1, tasks=len(tasks), scenes_done=len(recs), scenes=ntok, counts=st,
                            elapsed_s=round(el), rate_scenes_s=round(len(recs) / el, 2))
                json.dump(prog, open(f"{D}/status/{TAG}_gen_progress.json", "w"))
                print(f"[{el:.0f}s] {json.dumps(prog)}", flush=True)
    ok = sorted([rc for rc in recs if rc["status"] == "ok"], key=lambda x: x["token"])
    bad = [rc for rc in recs if rc["status"] != "ok"]
    with open(f"{D}/inputs/{TAG}_inputs.jsonl", "w") as f:
        for rc in ok:
            f.write(json.dumps(to_row(rc), separators=(",", ":")) + "\n")
    np.savez(f"{D}/inputs/{TAG}_inputs.npz", tokens=np.array([x["token"] for x in ok]), logs=np.array([x["log_name"] for x in ok]),
             maps=np.array([x["map"] for x in ok]), subset=np.array([x["subset"] for x in ok]),
             timestamp_us=np.array([x["timestamp_us"] for x in ok], dtype=np.int64),
             future_poses=np.stack([x["fut"] for x in ok]), history_poses=np.stack([x["hist"] for x in ok]),
             history_velocity=np.stack([x["vel"] for x in ok]), history_acceleration=np.stack([x["acc"] for x in ok]),
             driving_command=np.stack([x["cmd"] for x in ok]), cams_exist=np.array([[x["cams"][c][1] for c in CAMS] for x in ok]))
    if not args.tag:
        with open(f"{CACHE}/metadata/{SPLIT}_metric_cache_v11.csv", "w") as f:
            f.write("file_name\n")
            for rc in ok:
                f.write(rc["metric_cache"] + "\n")
    ce = np.array([[x["cams"][c][1] for c in CAMS] for x in ok]) if ok else np.zeros((0, 3), bool)
    summ = dict(split=TAG, filter_tokens=len(toks_cfg), unmapped_tokens=len(unmapped), unmapped_examples=unmapped[:20],
                scenes_attempted=len(recs), scenes_ok=len(ok), failures=len(bad), status_counts=st,
                failure_examples=[{k: v for k, v in b.items() if k in ("token", "log_name", "status", "err")} for b in bad[:20]],
                cams_missing={c: int((~ce[:, j]).sum()) for j, c in enumerate(CAMS)},
                cache_bytes_total=int(sum(x["cache_bytes"] for x in ok)),
                subset_counts={s: int(sum(1 for x in ok if x["subset"] == s)) for s in sorted(set(x["subset"] for x in ok))},
                wall_s=round(time.time() - t0, 1))
    json.dump(summ, open(f"{D}/status/{TAG}_gen_summary.json", "w"), indent=1)
    open(f"{D}/status/{TAG}_gen_DONE", "w").write(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    print("DONE", json.dumps({k: v for k, v in summ.items() if k != "failure_examples"}), flush=True)


if __name__ == "__main__":
    main()
