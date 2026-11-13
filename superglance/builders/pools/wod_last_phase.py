"""WOD training rows at frames 135-165 of the 10 Hz training frames (the rated val and test frames sit at 147-150) for the stage-3
last phase (SFT continued). proxy = wod_proxy.proxy: the official RFS with the logged future as the only rater trajectory, score
10. ADE over the 8 poses at 0.5-4 s.
  rows <out>            training segments only (none of val / test / the SFT and RL pools' dev segments); per row the RL rewards
                        proxy / 10 and RL targets softmax(-(ADE - min) / 1 m) over K' (5,273), the SFT targets t0 (softmax(-(ADE -
                        min ADE) / 1 m)) / t1 (softmax(proxy / 0.5 - (ADE - min ADE) / 1 m)) over the first 5,121 candidates and the
                        golds (argmax). Checks: proxy == official on 20 rows; t0, gold and question equal the final SFT pool's where
                        the row is in it. Prints the val reads of t1 and the proxy (argmax / expected real RFS / mass).
  sftpool <t0|t1> <base targets.npy> <hist dir> <dest>   the final SFT pool with its WOD rows replaced; its targets are float32
                        [rows, 5,121]: t0 rows are the pool's own rows for those tokens (bit for bit), t1 rows the float16 t1.
                        The paper's last phase: t1.
env: WOD_ROWS, SFT_POOL, RL_DEV, CANDS, LAST_ROWS (= the rows dir), IMG_ROOT.
usage: python -m superglance.builders.pools.wod_last_phase rows|sftpool ..."""
import glob, json, os, shutil, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from superglance.builders.pools.wod_proxy import proxy, target, to20   # noqa: E402

ENV = os.environ.get
WR, SFTP, RLDEV, CA = ENV("WOD_ROWS"), ENV("SFT_POOL"), ENV("RL_DEV"), ENV("CANDS")
RD = ENV("LAST_ROWS"); K1 = 5121; IMG = ENV("IMG_ROOT")


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def soft_ade(ade):   # softmax(-(ADE - min) / 1 m) in float64, as e2e_pool.soft
    q = np.exp(-(ade - ade.min())); return q / q.sum()


def mm(path, shape, dtype):
    return np.lib.format.open_memmap(path + ".part", mode="w+", dtype=dtype, shape=shape)


def close(path, a):
    a.flush(); os.replace(path + ".part", path)


def copy_rows(dst, src, a=0, n=None, S=100000):   # dst[a:a + n] = src[:n] in chunks
    n = len(src) if n is None else n
    for i in range(0, n, S):
        dst[a + i:a + min(i + S, n)] = src[i:min(i + S, n)]


def link(t, f):
    if not os.path.lexists(f):
        os.symlink(t, f)
    assert os.path.realpath(f) == os.path.realpath(t), f


def rows(out):
    from superglance.builders.pools import wod_rfs as W
    from superglance.builders.pools.rater_feedback_utils import get_rater_feedback_score
    os.makedirs(out, exist_ok=True)
    C = np.load(CA)[:, :, :2].astype(np.float64); X = to20(C); K = len(C)
    sp = {k: {l.strip() for l in open(f"{WR}/splits/{k}_segments.txt") if l.strip()} for k in ("training", "val", "test")}
    dev = {json.loads(l)["token"].rsplit("-", 1)[0] for p in (f"{SFTP}/dev.jsonl", RLDEV) for l in open(p) if '"source": "wod"' in l}
    R, skip = [], Counter()
    for p in sorted(glob.glob(f"{WR}/train1/meta/*.jsonl")):
        for l in open(p):
            t = l[:300].split('"token": "', 1)[1].split('"', 1)[0]
            if not 135 <= int(t.rsplit("-", 1)[1]) <= 165:
                continue
            r = json.loads(l); s = r["segment"]
            why = "val" if s in sp["val"] else "test" if s in sp["test"] else "not training" if s not in sp["training"] else "dev" if s in dev else ""
            if why:
                skip[why] += 1; continue
            assert 135 <= r["frame"] <= 165 and r["token"] == f"{s}-{r['frame']}", r["token"]
            R.append(r)
    R.sort(key=lambda r: (r["segment"], r["frame"])); n = len(R)
    log(f"{n} rows, {len({r['segment'] for r in R})} segments, frames {min(r['frame'] for r in R)}-{max(r['frame'] for r in R)}; skipped {dict(skip)}")
    A = {k: mm(f"{out}/{k}.npy", (n, w), dt) for k, w, dt in (("rl_rewards", K, np.float32), ("rl_targets", K, np.float16),
                                                             ("t0", K1, np.float16), ("t1", K1, np.float16))}
    G = np.zeros((n, 3), np.int64)   # gold: t0, t1 (v1), RL (K', argmax of target(proxy, ADE): the proxy's best, ties to the nearer)
    for i, r in enumerate(R):
        f = np.asarray(r["future"], np.float64); s = proxy(X, f, float(r["speed"])); ade = np.linalg.norm(C - f[1:16:2][None], axis=-1).mean(-1)
        q0, q1 = soft_ade(ade[:K1]), target(s[:K1], ade[:K1])
        A["rl_rewards"][i] = s / 10; A["rl_targets"][i] = soft_ade(ade); A["t0"][i] = q0; A["t1"][i] = q1
        G[i] = q0.argmax(), q1.argmax(), target(s, ade).argmax()
        if i % 10000 == 0:
            log(f"{i} / {n}")
    for k, a in A.items():
        close(f"{out}/{k}.npy", a)
    np.save(f"{out}/gold.npy", G)
    rng = np.random.default_rng(0); chk = {}
    for i in rng.choice(n, 20, replace=False):   # proxy == the official function (the logged future as the one rater, score 10)
        r = R[i]; f = np.asarray(r["future"], np.float64)
        b = get_rater_feedback_score(X[:, None], np.ones((K, 1)), [[f]] * K, [np.array([10.0])] * K, np.full(K, r["speed"]), frequency=4, length_seconds=5)
        assert np.allclose(np.load(f"{out}/rl_rewards.npy", mmap_mode="r")[i] * 10, b["rater_feedback_score"], atol=1e-4), r["token"]
    chk["proxy_vs_official_rows"] = 20
    # t0 / gold / question against the final SFT pool where the row is in it
    pos = {r["token"]: i for i, r in enumerate(R)}; ST = np.load(f"{SFTP}/train_targets.npy", mmap_mode="r"); T0 = np.load(f"{out}/t0.npy", mmap_mode="r")
    m = d = dg = dq = 0
    for l in open(f"{SFTP}/train.jsonl"):
        if '"source": "wod"' not in l:
            continue
        x = json.loads(l); i = pos.get(x["token"])
        if i is None or x["source"] != "wod":
            continue
        m += 1; dq += x["question"] != R[i]["question"]; dg += x["gold_index"] != int(G[i, 0])
        if m <= 3000:
            a = np.asarray(ST[x["compact"]["row"]], np.float64); d = max(d, np.abs(a / a.sum() - np.asarray(T0[i], np.float64)).max())
    assert m > 0.9 * n and dq == 0 and d < 2e-3, (m, dq, d)
    chk.update(in_final_sft_pool=m, t0_max_abs_vs_pool=d, gold_differs=dg, question_differs=dq)
    log(f"vs the final SFT pool: {m} of {n} rows in it; t0 max abs {d:.1e} (3,000 rows); gold differs {dg}; question differs {dq}")
    # val reads of the targets, over v1 (the SFT head) and K' (the RL head): argmax real RFS, expected real RFS, mass in the raters' regions
    V = [r for r in W.rows(f"{WR}/val_rated") if r["prefs"]]; acc = {k: [] for k in ("t1_arg", "t1_exp", "t1_mass", "t0_arg", "rl_arg")}
    for r in V:
        o = get_rater_feedback_score(X[:, None], np.ones((K, 1)), [[np.array(p["traj"]) for p in r["prefs"]]] * K,
                                     [np.array([p["score"] for p in r["prefs"]])] * K, np.full(K, r["speed"]), frequency=4, length_seconds=5)
        real, ins = o["rater_feedback_score"], o["is_fully_within_trust_region"][:, 0]
        f = np.asarray(r["future"], np.float64); s = proxy(X, f, float(r["speed"])); ade = np.linalg.norm(C - f[1:16:2][None], axis=-1).mean(-1)
        q1 = target(s[:K1], ade[:K1]); acc["t1_arg"].append(real[q1.argmax()]); acc["t1_exp"].append((q1 * real[:K1]).sum())
        acc["t1_mass"].append(q1[ins[:K1]].sum()); acc["t0_arg"].append(real[ade[:K1].argmin()]); acc["rl_arg"].append(real[target(s, ade).argmax()])
    chk["val"] = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
    log(f"val (479): {chk['val']}")
    with open(f"{out}/rows.jsonl", "w") as o:
        for r in R:
            o.write(json.dumps({k: r[k] for k in ("token", "segment", "frame", "speed", "command", "question")} | {"src": r["images"][0]}) + "\n")
    msg = {"rows": n, "segments": len({r["segment"] for r in R}), "skipped": dict(skip), "checks": chk}
    json.dump(msg, open(f"{out}/report.json", "w"), indent=1); open(f"{out}/OK", "w").write(json.dumps(msg) + "\n"); log("OK", json.dumps(msg))


def new_rows():
    return [json.loads(l) for l in open(f"{RD}/rows.jsonl")], np.load(f"{RD}/gold.npy")


def sftpool(var, bt, hist, dest):
    M, G = new_rows(); T = np.load(f"{RD}/{var}.npy", mmap_mode="r"); g = G[:, 0 if var == "t0" else 1]
    B = np.load(bt, mmap_mode="r"); NT = len(B); assert B.shape[1] == K1 and os.path.getsize(bt) == os.path.getsize(f"{SFTP}/train_targets.npy"), (B.shape, B.dtype)
    os.makedirs(dest, exist_ok=True); lab = "softmax_minus_ade_1m" if var == "t0" else "softmax_proxy_0.5_minus_ade_1m"; nw = nk = 0
    pos = {m["token"]: i for i, m in enumerate(M)}; old = np.full(len(M), -1)   # the pool's own compact row of each new row's token
    with open(f"{dest}/train.jsonl.part", "w") as o:
        for l in open(f"{SFTP}/train.jsonl"):
            if '"source": "wod"' in l:
                x = json.loads(l)
                if x["source"] == "wod":
                    nw += 1; i = pos.get(x["token"])
                    if i is not None:
                        old[i] = x["compact"]["row"]
                    continue
            o.write(l); nk += 1
        for i, (m, k) in enumerate(zip(M, g)):
            o.write(json.dumps({"question": m["question"], "no_options": True, "fixed_order": True, "gold_index": int(k), "id": f"wod_{m['token']}",
                                "token": m["token"], "source": "wod", "sampling_task": "wod", "sampling_bucket": "all",
                                "images": [f"wod/{m['segment']}/{m['frame']}.jpg"], "correct_answer": f"t{int(k):04d}", "label_source": lab,
                                "compact": {"k": K1, "row": NT + i}}) + "\n")
    os.replace(f"{dest}/train.jsonl.part", f"{dest}/train.jsonl"); log(f"rows: kept {nk}, dropped {nw} old WOD rows, added {len(M)}")
    assert var == "t1" or (old >= 0).all(), int((old < 0).sum())
    new = (lambda a, b: np.asarray(B[np.sort(old[a:b])])[np.argsort(np.argsort(old[a:b]))]) if var == "t0" else (lambda a, b: np.asarray(T[a:b], B.dtype))
    o = mm(f"{dest}/train_targets.npy", (NT + len(M), K1), B.dtype); copy_rows(o, B)
    for a in range(0, len(M), 10000):
        o[NT + a:NT + min(a + 10000, len(M))] = new(a, min(a + 10000, len(M)))
    close(f"{dest}/train_targets.npy", o)
    o = np.load(f"{dest}/train_targets.npy", mmap_mode="r"); ix = np.sort(np.random.default_rng(0).choice(NT, 1000, replace=False))
    assert (np.asarray(o[ix]) == np.asarray(B[ix])).all() and all((np.asarray(o[NT + a:NT + min(a + 10000, len(M))]) == new(a, min(a + 10000, len(M)))).all() for a in range(0, len(M), 10000))
    if var == "t0":
        assert (np.abs(np.asarray(o[NT:NT + 2000], np.float64) - np.asarray(T[:2000], np.float64)).max() < 2e-3)
    log(f"targets written ({B.dtype}; new rows {'the pool' + chr(39) + 's own' if var == 't0' else 't1'})")
    for f in ("dev.jsonl", "dev_targets.npy"):
        shutil.copyfile(f"{SFTP}/{f}", f"{dest}/{f}")
    for top in ("navtrain", "pai", "pai_dagger", "qa", "l2d", "wod"):
        link(f"{IMG}/{top}", f"{dest}/{top}")
    link(hist, f"{dest}/hist")

    def put(m):   # the train1 jpg, where the SFT pool has it (its WOD-E2E shard's copies.tsv: train1/img -> wod/<seg>/<frame>.jpg)
        d = f"{IMG}/wod/{m['segment']}/{m['frame']}.jpg"
        if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(m["src"]):
            return 0
        os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copyfile(m["src"], d + ".part"); os.replace(d + ".part", d); return 1
    with ThreadPoolExecutor(32) as ex:
        nc = sum(ex.map(put, M))
    same = sum(open(m["src"], "rb").read() == open(f"{IMG}/wod/{m['segment']}/{m['frame']}.jpg", "rb").read() for m in M[::300])
    assert same == len(M[::300]), same
    need = sorted({x for s in ("train", "dev") for l in open(f"{dest}/{s}.jsonl") for x in json.loads(l)["images"]})

    def ok(rel):
        try:
            return os.path.getsize(f"{dest}/{rel}") > 0
        except OSError:
            return False
    with ThreadPoolExecutor(48) as ex:
        miss = Counter(r.split("/", 1)[0] for r, k in zip(need, ex.map(ok, need, chunksize=1024)) if not k)
    assert not miss, dict(miss)
    msg = {"var": var, "kept": nk, "dropped_wod": nw, "added_wod": len(M), "images": len(need), "copied_wod_images": nc, "targets_rows": NT + len(M), "targets_dtype": str(B.dtype)}
    open(f"{dest}/STAGED", "w").write(json.dumps(msg) + "\n"); log("STAGED", json.dumps(msg))


if __name__ == "__main__":
    {"rows": rows, "sftpool": sftpool}[sys.argv[1]](*sys.argv[2:])
