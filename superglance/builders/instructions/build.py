"""Instruction data build (contracts.py for the rules). One family per scene, the scarcest family first, round-robin over the
family's instruction pairs (sub-types) up to the quota (navtrain 60k scenes, PAI-AV 40k, WOD-E2E 20k; yield decides). Per scene:
  plain      the source's normal prompt (real command): softmax(-ADE / tau) over all candidates
  hindsight  the adapter's append-mode prompt (command "go straight") with h's wording: softmax(-ADE / tau) over {C_h, not C_c}
  contrast   the same prompt with c's wording: softmax(-ADE_perp / tau) over S_c (feasible; here also no first-step speed jump:
             |v_0.5 - v_now| / 0.5 s <= 3.5 m/s^2, trajectory_quality's own bound applied from the current speed)
Prompts: superglance.data.scene.scene_head from the frame's real ego state, history and command (no road-users block).
Dev: held-out units (navtrain logs and PAI-AV clips with crc32 % 50 == 0; WOD-E2E segments with crc32 % 40 == 0) with the dev-only
wordings, up to 100 pairs per source x family. Out: {train,dev}.jsonl, {train,dev}_targets.npy (float32), {train,dev}_contracts.jsonl
(both contracts of each pair), copies.tsv (src, dst), images_{split}.tsv, manifest.tsv (src, path under L2D_IMG_ROOT, bytes),
report.json (counts; check.py adds the checks).
env: NAV_INPUTS, PAI_FIT_ROWS, OLD_POOL, WOD_META, L2D_IMG_ROOT (where the rows' images are staged).
usage (navsim env): python -m superglance.builders.instructions.build <out>"""
import glob, json, os, random, sys, zlib
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np

from superglance.builders.instructions import contracts as ifc   # noqa: E402

NAV_IN, PAI_IN, POOL, WOD_META = (os.environ[k] for k in ("NAV_INPUTS", "PAI_FIT_ROWS", "OLD_POOL", "WOD_META"))
ST = os.environ["L2D_IMG_ROOT"].rstrip("/")
QUOTA = {"navtrain": {"lane": 15000, "junction": 9000, "speed": 18000, "stopgo": 12000, "follow": 6000},
         "pai": {"speed": 20000, "stopgo": 13333, "follow": 6667}, "wod": {"pace": 20000}}
DEV_PAIRS = 100
C = np.load(ifc.CANDS).astype(np.float64); C2 = C[:, :, :2]; S0 = ifc.feats(C2)["s"][:, 0]


def subtype(p):
    c = p["c"].split(":")
    return p["h"].split(":")[0] + ">" + (c[2] if len(c) == 3 else c[0])


def load(out):   # source -> [scene records]; a pair survives the first-step speed check with |S_c| >= 3
    def jump_ok(r, p):
        S = ifc.unpack(p["S"]) & (np.abs(S0 - r["v0"]) / 0.5 <= 3.5); p["S"], p["nS"] = ifc.pack(S), int(S.sum())
        return p["nS"] >= 3
    R = {"navtrain": [json.loads(l) for f in sorted(glob.glob(f"{out}/work/nav/*.jsonl")) for l in open(f)],
         "pai": [json.loads(l) for f in sorted(glob.glob(f"{out}/work/pai/*.jsonl")) for l in open(f)],
         "wod": [json.loads(l) for l in open(f"{out}/work/wod/pairs.jsonl")]}
    pool_dev = {json.loads(l)["token"] for l in open(f"{POOL}/dev.jsonl")}   # pool_full's own dev scenes stay out of IF train
    R = {src: [r for r in rs if r["token"] not in pool_dev] for src, rs in R.items()}; yld = defaultdict(lambda: [0, 0, 0, 0])
    for src, rs in R.items():
        for r in rs:
            r["src"], r["unit"] = src, r.get("log") or r.get("clip") or r["segment"]
            r["dev"] = zlib.crc32(r["unit"].encode()) % (40 if src == "wod" else 50) == 0
            keep = [p for p in r["pairs"] if jump_ok(r, dict(p))]
            for p in r["pairs"]:
                yld[f"{src}/{p['family']}/{subtype(p)}"][0] += 1
            for p in keep:
                yld[f"{src}/{p['family']}/{subtype(p)}"][1] += 1
            for f in {p["family"] for p in r["pairs"]}:
                yld[f"{src}/{f}/scenes"][2] += 1
            for f in {p["family"] for p in keep}:
                yld[f"{src}/{f}/scenes"][3] += 1
            r["pairs"] = [p for p in r["pairs"] if jump_ok(r, p)]
    yld = {k: ({"pairs_before": v[0], "pairs_after": v[1]} if not k.endswith("/scenes") else {"scenes_before": v[2], "scenes_after": v[3]})
           for k, v in sorted(yld.items())}
    return R, yld


def select(rs, quota, seed):   # {token: (scene, pair)}: scarcest family first, round-robin over sorted sub-types, seeded scene order
    order = sorted(rs, key=lambda r: r["token"]); random.Random(seed).shuffle(order); taken = {}
    avail = {f: sum(any(p["family"] == f for p in r["pairs"]) for r in order) for f in quota}
    for f in sorted(quota, key=lambda f: avail[f] / quota[f]):
        by = defaultdict(list)
        for r in order:
            for p in r["pairs"]:
                if p["family"] == f:
                    by[subtype(p)].append((r, p))
        qs = [by[k] for k in sorted(by)]; ptr = [0] * len(qs); n = 0
        while n < quota[f]:
            moved = False
            for i, q in enumerate(qs):
                while ptr[i] < len(q) and q[ptr[i]][0]["token"] in taken:
                    ptr[i] += 1
                if ptr[i] < len(q) and n < quota[f]:
                    taken[q[ptr[i]][0]["token"]] = q[ptr[i]]; ptr[i] += 1; n += 1; moved = True
            if not moved:
                break
    return taken, avail


def inputs(sel):   # token -> (scene for the prompt, F2 [8, 2], path points for ADE_perp, image dst, image src, real command text)
    want = {s: {t for t, (r, _) in sel.items() if r["src"] == s} for s in ("navtrain", "pai", "wod")}; out = {}
    for l in open(NAV_IN):
        x = json.loads(l)
        if x["token"] in want["navtrain"]:
            F2 = np.asarray(x["future_poses"], np.float64)[:, :2]
            out[x["token"]] = ({"velocity": x["ego_velocity"], "acceleration": x["ego_acceleration"], "command": x["driving_command"], "past": x["history_poses"]},
                               F2, F2, f"{ST}/img/nav/{os.path.basename(x['cam_f0'])}", x["cam_f0"])
    src = {}
    for l in open(f"{POOL}/copies.tsv"):
        a, b = l.rstrip("\n").split("\t")
        if "/img/pai/" in b:
            src[b] = a
    for l in open(PAI_IN):
        x = json.loads(l)
        if x["token"] in want["pai"]:
            F2 = np.asarray(x["future_poses"], np.float64)[:, :2]; dst = f"{ST}/img/pai/{x['token']}.jpg"
            out[x["token"]] = ({"velocity": x["ego_velocity"], "acceleration": x["ego_acceleration"], "command": x["driving_command"], "past": x["history_poses"]},
                               F2, F2, dst, src[dst])
    for f in sorted(glob.glob(f"{WOD_META}/*.jsonl")):
        for l in open(f):
            x = json.loads(l)
            if x["token"] in want["wod"]:
                F20 = np.asarray(x["future"], np.float64); ps = x["past16"]
                out[x["token"]] = ({"velocity": [x["speed"], 0.0], "acceleration": [x["accel"], 0.0], "command": x["command"], "past": [ps[-7], ps[-5], ps[-3], ps[-1]]},
                                   F20[1:16:2], F20, f"{ST}/img/wod/{os.path.basename(x['images'][0])}", x["images"][0])
    return out


def tgt(job):   # (plain, hindsight, contrast) float32 targets of one scene
    src, F2, Fp, p = job; tau = ifc.TAU[src]; a = ifc.ade(C2, F2)
    ap = ifc.ade_lon(C2, F2) if p["family"] in ifc.LATERAL else ifc.ade_lat(C2, ifc.human_path(Fp))
    return np.stack([ifc.softmin(a, tau), ifc.softmin(a, tau, ifc.unpack(p["E"])), ifc.softmin(ap, tau, ifc.unpack(p["S"]))]).astype(np.float32)


def write(out, split, sel, inp, q_plain, q_if):
    keys = sorted(sel, key=lambda t: (sel[t][0]["src"], sel[t][1]["family"], t)); rows, con, words = [], [], Counter()
    T = np.lib.format.open_memmap(f"{out}/{split}_targets.npy", "w+", np.float32, (3 * len(keys), ifc.K))
    with Pool(24) as pool:
        for i, (t, Q) in enumerate(zip(keys, pool.imap(tgt, [(sel[t][0]["src"], inp[t][1], inp[t][2], sel[t][1]) for t in keys], chunksize=64))):
            r, p = sel[t]; scene, _, _, dst, _ = inp[t]; src = f"if_{r['src']}_{p['family']}"; pid = f"{src}_{t}"; T[3 * i:3 * i + 3] = Q
            th, ih = ifc.wording(p["h"], p["family"], split, t + "/h"); tc, ic = ifc.wording(p["c"], p["family"], split, t + "/c"); words.update([ih, ic])
            for k, (kind, q, lab, bucket) in enumerate((("plain", q_plain(scene), "softmax_minus_ade", "plain"),
                                                         ("hindsight", q_if(scene, th), "softmax_minus_ade", subtype(p).split(">")[0]),
                                                         ("contrast", q_if(scene, tc), "softmax_minus_adeperp", subtype(p).split(">")[1]))):
                g = int(Q[k].argmax())
                rows.append({"question": q, "no_options": True, "fixed_order": True, "gold_index": g, "id": f"{pid}_{kind}", "token": t, "source": src,
                             "sampling_task": f"{src}/{kind}", "sampling_bucket": bucket, "images": [dst], "correct_answer": f"t{g:04d}",
                             "label_source": f"if_{lab}_{ifc.TAU[r['src']]:g}m", "compact": {"k": ifc.K, "row": 3 * i + k},
                             "if_pair": pid, "if_kind": kind, "if_label": None if kind == "plain" else p["h" if kind == "hindsight" else "c"],
                             "if_instruction": None if kind == "plain" else (th if kind == "hindsight" else tc),
                             "if_wording": None if kind == "plain" else (ih if kind == "hindsight" else ic)})
            con.append({"pair": pid, "source": src, "family": p["family"], "h": p["h"], "c": p["c"], "unit": r["unit"], "v0": r["v0"],
                        "nE": p["nE"], "nS": p["nS"], "S": p["S"], "contract_h": p["contract_h"], "contract_c": p["contract_c"], "rows": [3 * i, 3 * i + 1, 3 * i + 2]})
    T.flush(); s = np.asarray(T, np.float64).sum(1)
    with open(f"{out}/{split}.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in rows)
    with open(f"{out}/{split}_contracts.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in con)
    return rows, words, float(np.abs(s - 1).max())


def main(out):
    q_plain, q_if = ifc.prompts(); R, yld = load(out)
    rep = {"quota": QUOTA, "first_step_check": {"rule": "contrast feasibility also needs |v(0.5 s) - v(0)| / 0.5 s <= 3.5 m/s^2: v(0.5 s) = the "
           "candidate's first 0.5 s segment speed (verify.validate_trajectory), v(0) = the current speed |ego velocity|; i.e. |v(0.5 s) - v(0)| "
           "<= 1.75 m/s, trajectory_quality's own acceleration bound applied from the current state",
           "yield_before_after": yld}, "available_scenes": {}}
    wsz = {l.split("\t")[0]: int(l.split("\t")[1]) for l in open(f"{out}/wod_image_sizes.tsv")}   # WOD sources' sizes, taken where they live
    sel = {"train": {}, "dev": {}}
    for src, rs in R.items():
        for split in ("train", "dev"):
            q = QUOTA[src] if split == "train" else {f: DEV_PAIRS for f in QUOTA[src]}
            s, avail = select([r for r in rs if r["dev"] == (split == "dev")], q, 20261023)
            sel[split].update(s); rep["available_scenes"][f"{split}/{src}"] = avail
    for split in ("train", "dev"):
        inp = inputs(sel[split]); rows, words, err = write(out, split, sel[split], inp, q_plain, q_if)
        cnt = Counter((x["source"], x["if_kind"], x["sampling_bucket"]) for x in rows)
        rep[split] = {"rows": len(rows), "scenes": len(rows) // 3, "target_sum_abs_err_max": err,
                      "by_source_kind": {f"{a}/{b}": n for (a, b), n in sorted(Counter((x["source"], x["if_kind"]) for x in rows).items())},
                      "by_source_kind_bucket": {"/".join(k): n for k, n in sorted(cnt.items())}, "wordings_used": dict(sorted(words.items()))}
        imgs = sorted({(inp[t][3], inp[t][4]) for t in sel[split]}); have = {l.split("\t")[1].strip() for l in open(f"{POOL}/copies.tsv")}
        im = rep[split]["images"] = {}
        with open(f"{out}/images_{split}.tsv", "w") as f:   # dst, src, bytes, already on the training side (the earlier pool's images; WOD: all)
            for dst, src in imgs:
                n = os.path.getsize(src) if os.path.exists(src) else wsz.get(src, -1); on = "/img/wod/" in dst or dst in have
                f.write(f"{dst}\t{src}\t{n}\t{int(on)}\n"); x = im.setdefault(dst.split("/")[-2], {"n": 0, "gb": 0.0, "n_staged": 0, "gb_to_move": 0.0, "missing": 0})
                x["n"] += 1; x["gb"] += n / 1e9; x["n_staged"] += on; x["gb_to_move"] += 0 if on else n / 1e9; x["missing"] += n < 0
    with open(f"{out}/copies.tsv", "w") as f, open(f"{out}/manifest.tsv", "w") as g:   # copies: src, dst; manifest: src, rel, bytes
        for split in ("train", "dev"):
            for l in open(f"{out}/images_{split}.tsv"):
                dst, src, n, _ = l.rstrip("\n").split("\t"); f.write(f"{src}\t{dst}\n"); g.write(f"{src}\t{dst[len(ST) + 1:]}\t{n}\n")
    json.dump(rep, open(f"{out}/report.json", "w"), indent=1); print(json.dumps({k: rep[k] for k in ("train", "dev")})[:3000])


if __name__ == "__main__":
    main(sys.argv[1])
