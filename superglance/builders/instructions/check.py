"""Instruction data checks, written into report.json["checks"]:
  sums        every target row sums to 1 (float64, |d| < 1e-5)
  contrast    each contrast row's mass lies in its S_c, and S_c lies in the exclusive set recomputed from the contracts
              (contracts.holds) and in the feasible set recomputed (navtrain PDM NC = DAC = 1 and k < 4,097; kinematic; first-step
              speed; PAI-AV collisions re-run on 300 sampled frames)
  verify      >= 250 sampled train pairs per source x family plus every dev pair: the SDK's own verifier (verify.py; WOD-E2E: the
              track's score_p0_kinematic_proxy on the adapter's 0.25-5 s grid) on each row's top candidate and on the human future
  exclusions  rows from navtrain held-out scenes, L2D logs / recordings, navtest logs, non-trainval logs; PAI-AV non-train, RealSim,
              916 AlpaSim clips; WOD-E2E non-training shards; units in both splits; wordings of the other split; strings equal or
              near (contracts.NEAR_DUP) to L2D's
  prompts     instruction rows = the frame's normal prompt with the command phrase set to "go straight" + " Driving instruction: "
              + text (rebuilt independently from the earlier pool's / WOD meta questions); plain rows = the normal prompt; the
              adapter's own observation parsing of the navtrain history; 0 rows with "Road users ahead"; command phrase per row kind
env: NAV_INPUTS, NAVTEST_INPUTS, OLD_POOL, PAI_ROOT, WOD_META, S916.
usage (navsim env): python -m superglance.builders.instructions.check <out>"""
import glob, json, os, random, re, sys, zlib
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np

from superglance.builders.instructions import contracts as ifc   # noqa: E402

NAV_IN, NAVTEST_IN, POOL, PA, WOD_META = (os.environ[k] for k in ("NAV_INPUTS", "NAVTEST_INPUTS", "OLD_POOL", "PAI_ROOT", "WOD_META"))
C = np.load(ifc.CANDS).astype(np.float64); C2 = C[:, :, :2]; X = ifc.feats(C2); KIN = ifc.kin_ok(C); S0 = X["s"][:, 0]
GRID = [0.25 * i for i in range(1, 21)]


G = {}


def tok_of(p):   # pair id = "<source>_<token>"
    return p["pair"][len(p["source"]) + 1:]


def excl(p):   # recomputed exclusive set of the contrast instruction
    if p["family"] == "pace":
        if "wod" not in G:
            from superglance.builders.instructions.wod import wfeats
            from superglance.l2d.adapter import to_grid
            G["wod"] = wfeats(np.array([to_grid(c, GRID) for c in C]))
        late, path, _ = G["wod"]; c = p["contract_c"]
        return (late <= c["slow_accept_max_mps"]) & (path >= 2.0) & ~((late >= c["normal_accept_min_mps"]) & (path >= 2.0))
    return ifc.holds(p["contract_c"], X) & ~ifc.holds(p["contract_h"], X)


def chk_contrast(job):
    p, q, pdm = job; S = ifc.unpack(p["S"]); E = excl(p); FE = KIN & (np.abs(S0 - p["v0"]) / 0.5 <= 3.5)
    if pdm is not None:
        FE &= pdm
    return int((q[~S] > 0).sum()), int((S & ~E).sum()), int((S & ~FE).sum()), float(q[~(E & FE)].sum())


def top_ok(p, P, kind):   # the SDK verifier on P (a candidate [8, 3], or the human future; WOD's is on the 0.25-5 s grid): own contract, not the other
    own, opp = (p["contract_h"], p["contract_c"]) if kind == "hindsight" else (p["contract_c"], p["contract_h"])
    if p["family"] == "pace":
        import score_p0_kinematic_proxy as p0
        from superglance.l2d.adapter import to_grid
        f = p0.features(P.tolist() if len(P) == 20 else to_grid(P, GRID)); names = ("normal", "slow") if kind == "hindsight" else ("slow", "normal")
        return p0.primitive(f, names[0], own)[0] and not p0.primitive(f, names[1], own)[0]
    return ifc.verify_one(P[:, :2], own) and not ifc.verify_one(P[:, :2], opp)


def main(out):
    rep = json.load(open(f"{out}/report.json")); ck = rep["checks"] = {}; sys.path.insert(0, f"{ifc.L2D}/ws/language2drive.OpenloopTrack.WOD-E2E/scripts")
    l2d = json.load(open(f"{ifc.L2D}/meta/l2d_navsim_logs.json")); L2D_S = ifc.l2d_strings(); L2D_N = {ifc.norm_text(s) for s in L2D_S}
    nav = {}
    for l in open(NAV_IN):
        x = json.loads(l); nav[x["token"]] = x
    navtest_logs = {json.loads(l)["log_name"] for l in open(NAVTEST_IN)}; nav_row = {t: i for i, t in enumerate(nav)}
    trainval = {f[:-4] for f in os.listdir(f"{os.environ['OPENSCENE_DATA_ROOT']}/navsim_logs/trainval")}   # OpenScene trainval logs (navtest's 136 logs are not among them)
    import pandas as pd
    idx = pd.read_parquet(f"{PA}/clip_index.parquet"); split = dict(zip(idx.index, idx.split))
    realsim = {l.strip() for l in open(f"{ifc.L2D}/meta/realsim_scene_uuids.txt") if l.strip()}
    p2601 = {l.strip().replace("clipgt-", "") for l in open(os.environ["S916"]) if l.strip()}
    pai_in = {}
    for l in open(os.environ["PAI_FIT_ROWS"]):
        x = json.loads(l); pai_in[x["token"]] = x
    wod_shard = {json.loads(l)["token"]: json.loads(l)["shard"] for l in open(f"{out}/work/wod/pairs.jsonl")}
    pool_q = {}
    for l in open(f"{POOL}/train.jsonl"):
        r = json.loads(l)
        if r["source"] in ("navtrain", "pai_human"):
            pool_q[r["token"]] = (r["question"], r.get("dropout", False))
    wod_q, wod_f = {}, {}
    for f in sorted(glob.glob(f"{WOD_META}/*.jsonl")):
        for l in open(f):
            x = json.loads(l); wod_q[x["token"]] = x["question"]; wod_f[x["token"]] = x["future"]
    M = np.load(f"{out}/work/nav_pdm_ok.npy", mmap_mode="r"); q_plain, q_if = ifc.prompts()
    from superglance.l2d import adapter as ad
    from superglance.data import scene as s3
    units = defaultdict(set)
    for split_ in ("train", "dev"):
        rows = [json.loads(l) for l in open(f"{out}/{split_}.jsonl")]; con = [json.loads(l) for l in open(f"{out}/{split_}_contracts.jsonl")]
        T = np.load(f"{out}/{split_}_targets.npy", mmap_mode="r"); c = ck[split_] = {}
        s = np.concatenate([np.asarray(T[i:i + 20000], np.float64).sum(1) for i in range(0, len(T), 20000)])
        c["sums"] = {"rows": len(s), "max_abs_err": float(np.abs(s - 1).max()), "bad": int((np.abs(s - 1) >= 1e-5).sum())}
        jobs = []
        for p in con:
            src = p["source"].split("_")[1]; pdm = None
            if src == "navtrain":
                pdm = np.zeros(ifc.K, bool); pdm[:ifc.KP] = np.unpackbits(M[nav_row[tok_of(p)]])[:ifc.KP].astype(bool)
            jobs.append((p, np.asarray(T[p["rows"][2]], np.float64), pdm))
        with Pool(32) as pool:
            res = pool.map(chk_contrast, jobs, chunksize=32)
        c["contrast"] = {"rows": len(res), "mass_outside_S_rows": sum(r[0] > 0 for r in res), "S_not_exclusive_rows": sum(r[1] > 0 for r in res),
                         "S_not_feasible_rows": sum(r[2] > 0 for r in res), "max_mass_on_infeasible_or_nonexclusive": max(r[3] for r in res)}
        # PAI collisions re-run on sampled frames
        from superglance.builders.instructions import pai as if_pai
        pc = [p for p in con if p["source"].startswith("if_pai")]; random.Random(1).shuffle(pc); bad = n = 0
        byc = defaultdict(list)
        for p in pc[:300]:
            byc[pai_in[tok_of(p)]["log_name"]].append(p)
        chunk = dict(zip(idx.index, idx.chunk))
        for clip, ps in byc.items():
            eg, ob = if_pai.clip_data(int(chunk[clip]), clip)
            for p in ps:
                fr = if_pai.frame(pai_in[tok_of(p)], eg, ob, C, X, KIN)
                q = np.asarray(T[p["rows"][2]], np.float64)   # the re-run S_c excludes every colliding candidate
                same = [x for x in fr["pairs"] if x["family"] == p["family"] and x["c"] == p["c"]]
                n += 1; bad += int(not same or (q[~(ifc.unpack(same[0]["S"]))] > 0).any())
        c["pai_collision_rerun"] = {"frames": n, "mass_outside_rerun_S": bad}
        # SDK verifier on top candidates and on the human future
        by = defaultdict(list)
        for p in con:
            by[p["source"]].append(p)
        ver = {}
        for src_, ps in sorted(by.items()):
            random.Random(2).shuffle(ps); ps = ps if split_ == "dev" else ps[:250]; a = Counter()
            for p in ps:
                for kind, r in (("hindsight", p["rows"][1]), ("contrast", p["rows"][2])):
                    a[kind] += int(top_ok(p, C[int(np.argmax(T[r]))], kind))
                src, tok = p["source"].split("_")[1], tok_of(p)
                Fh = np.asarray(wod_f[tok] if src == "wod" else (nav if src == "navtrain" else pai_in)[tok]["future_poses"], np.float64)
                a["human_h"] += int(top_ok(p, Fh, "hindsight")); a["n"] += 1
            ver[src_] = dict(a)
        c["verify_top1"] = ver
        # exclusions and wordings
        ex = Counter()
        for p in con:
            src, tok = p["source"].split("_")[1], tok_of(p); units[(src, p["unit"])].add(split_)
            if src == "navtrain":
                x = nav[tok]; ex["navtrain_heldout"] += x["subset"] != "fit"; ex["navtrain_l2d_log"] += x["log_name"] in l2d["l2d_logs"]
                ex["navtrain_l2d_recording"] += x["log_name"].rsplit("_", 2)[0] in l2d["l2d_recordings"]; ex["navtrain_navtest_log"] += x["log_name"] in navtest_logs
                ex["navtrain_log_not_trainval"] += x["log_name"] not in trainval
            elif src == "pai":
                cl = pai_in[tok]["log_name"]; ex["pai_not_train"] += split.get(cl) != "train"; ex["pai_realsim"] += cl in realsim; ex["pai_public_2601"] += cl in p2601
            else:
                ex["wod_not_training_shard"] += not wod_shard[tok].startswith("training_")
                ex["wod_held_segment_in_train"] += split_ == "train" and zlib.crc32(p["unit"].encode()) % 40 == 0
        strings = {r["if_instruction"] for r in rows if r["if_instruction"]}
        wid = Counter((r["if_wording"].split("/")[0], int(r["if_wording"].split("/")[1]) >= len(ifc.WORDS[r["if_wording"].split("/")[0]]) - ifc.NDEV)
                      for r in rows if r["if_wording"])
        ex["rows_with_other_split_wording"] = sum(n for (k, d), n in wid.items() if d != (split_ == "dev"))
        ex["strings_exact_l2d"] = sum(t in L2D_S for t in strings); ex["strings_normalized_l2d"] = sum(ifc.norm_text(t) in L2D_N for t in strings)
        sims = {t: max(ifc.similarity(t, u) for u in set(L2D_S)) for t in strings}
        ex["strings_near_dup_l2d"] = sum(v >= ifc.NEAR_DUP for v in sims.values()); c["wording"] = {"distinct_strings": len(strings), "max_similarity_to_l2d": max(sims.values())}
        c["exclusions"] = dict(ex)
        # prompts
        pr = Counter(); cmd = Counter()
        for r in rows:
            src, tok = r["source"].split("_")[1], r["token"]; q = r["question"]; pr["road_users"] += "Road users ahead" in q
            m = re.search(r"navigation command: ([a-z ]+)\. ", q); cmd[f"{r['if_kind'] if r['if_kind'] == 'plain' else 'instruction'}:{m.group(1) if m else None}"] += 1
            if src != "wod" and r["if_kind"] != "plain":   # the adapter's own path: L2D-style ego_history -> parse_observation -> build_prompt
                x = nav[tok] if src == "navtrain" else pai_in[tok]
                obs = {"cameras": {"CAM_F0": [{"image": None}]}, "ego_history": [{"x": p_[0], "y": p_[1], "velocity_xy": v_} for p_, v_ in zip(x["history_poses"], x["history_velocity"])]}
                _, sc = ad.parse_observation("navsim", obs); qa = ad.build_prompt(s3, sc, r["if_instruction"], dict(ad.DEFAULTS, prompt_mode="append"))
                pr[f"{src}_adapter_path_rows"] += 1; pr[f"{src}_adapter_path_identical"] += qa == q
                pr[f"{src}_adapter_path_identical_but_acceleration"] += re.sub(r"acceleration [+-][0-9.]+", "", qa) == re.sub(r"acceleration [+-][0-9.]+", "", q)
            if src == "wod":
                base = wod_q[tok]
            elif tok in pool_q and not pool_q[tok][1]:
                base = pool_q[tok][0]
                if "Road users ahead" in base:
                    base = base[:base.index("Road users ahead")] + base[base.index("Choose the trajectory"):]
            else:
                continue
            real = re.search(r"navigation command: ([a-z ]+)\. ", base).group(1)
            want = base if r["if_kind"] == "plain" else base.replace(f"navigation command: {real}. ", "navigation command: go straight. ") + " Driving instruction: " + r["if_instruction"]
            pr[f"{src}_rebuilt"] += 1; pr[f"{src}_identical"] += q == want
            if r["if_kind"] == "plain":
                pr["plain_real_command"] += m.group(1) == real
        c["prompts"] = dict(pr); c["command_phrase"] = dict(sorted(cmd.items()))
        if split_ == "train":   # three example pairs: top-3 candidates [k, mass, x(4 s), y(4 s), speed 3.5-4 s] of both instruction rows
            c["examples"] = []
            for fam in ("if_navtrain_lane", "if_pai_speed", "if_wod_pace"):
                p = next(p for p in con if p["source"] == fam); src, tok = p["source"].split("_")[1], tok_of(p)
                F = np.asarray(wod_f[tok], np.float64)[1:16:2] if src == "wod" else np.asarray((nav if src == "navtrain" else pai_in)[tok]["future_poses"], np.float64)
                e = {"pair": p["pair"], "h": p["h"], "c": p["c"], "v0": round(p["v0"], 2), "human": [round(F[-1, 0], 2), round(F[-1, 1], 2), round(float(np.hypot(*(F[-1, :2] - F[-2, :2]))) / 0.5, 2)]}
                for kind, i in (("hindsight", p["rows"][1]), ("contrast", p["rows"][2])):
                    t = np.asarray(T[i], np.float64)
                    e[kind] = {"instruction": rows[i]["if_instruction"], "top3": [[int(j), round(t[j], 3), round(C2[j, -1, 0], 2), round(C2[j, -1, 1], 2), round(X["s"][j, -1], 2)] for j in np.argsort(-t)[:3]]}
                c["examples"].append(e)
    ck["units_in_both_splits"] = sum(len(v) > 1 for v in units.values())
    json.dump(rep, open(f"{out}/report.json", "w"), indent=1); print(json.dumps(ck, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
