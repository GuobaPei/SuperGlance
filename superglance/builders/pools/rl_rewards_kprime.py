"""RL pool for the K' head: a copy of an RL pool (compact rows, train_rewards.npy [N, 5,121]) with the new candidates' reward columns
appended, so that no RL array stays 5,121 wide (the loader zero-pads a narrower reward or reference row without an error, and
reward 0 would be the best value on every PAI / DAgger row). Old columns are copied bytewise; the new columns follow each source's
rule:
  navtrain    PDMS x DDC - 10 x [NC < 1 | DAC < 1 | TTC < 1] on $W/score/navtrain_pdms_new_fit.npz (by token)
  pai_human   -0.2 x ade16 (ADE in m, float16), $W/rewards/pai_reward_new.npy (rows of pai_ade_new.npz's tokens)
  wod         RFS / 10, $W/rewards/wod_rfs_new.npz
  pai_dagger  -0.2 x ade16 to the shard label, $W/rewards/dag_<shard>_new.npz (DAG=<shard name>)
  any other source: STOP (no rule registered)
Check first: the same rules on 16 existing columns reproduce the pool's own rewards bit for bit on a sample of every source it holds
(inputs under CHECK). Out <out>: train_rewards.npy [N, K'] float32, train_targets.npy [N, K'] float16 uniform 1 / K' (the RL loss
ignores it; the loader needs one), train.jsonl with compact.k = K', dev_targets.npy = the pool's dev targets with K' - 5,121 zero
columns, train_rewards.state, report.json, links for the other files.
Optional REF=<train_ref.npy> DELTA=<d>: train_ref.npy [N, K'] = [ref, ref[:, nn] - DELTA] (nn = the head widening's nearest old rows):
the reference of the widened head before any fit of the new rows; NOT valid after a fit (then re-extract).
LIMIT=<n>: only the first n rows of each source (a dry run; never a training pool).
usage: [DAG=<shard>] [REF=... DELTA=4] [LIMIT=n] CANDS=<5,121 library> CHECK=<dir> python -m superglance.builders.pools.rl_rewards_kprime
       <pool dir> <W> <out dir>"""
import json, os, sys, zipfile

import numpy as np

P, W, O = sys.argv[1:4]; ENV = os.environ.get
E = ENV("CHECK")
LIMIT = int(ENV("LIMIT", "0")); DAG = ENV("DAG"); REF = ENV("REF"); DELTA = float(ENV("DELTA", "nan"))
tk = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def npz_mm(p, key):   # read-only memmap of an uncompressed np.savez member
    z = zipfile.ZipFile(p); i = z.getinfo(key + ".npy"); assert i.compress_type == 0, (p, key)
    with open(p, "rb") as f:
        f.seek(i.header_offset); h = f.read(30); f.seek(i.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little"))
        v = np.lib.format.read_magic(f); sh, fo, dt = (np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        off = f.tell()
    return np.memmap(p, dt, "r", off, sh, "F" if fo else "C")


def nav_v(S):   # e2e_rl1c_pool.nav_v
    S = S.astype(np.float64)
    return (S[..., 6] * S[..., 5] - 10.0 * ((S[..., 0] < 1) | (S[..., 1] < 1) | (S[..., 3] < 1))).astype(np.float32)


def ade16(F, C):   # e2e_rl1c_pool.ade16
    return np.concatenate([np.linalg.norm(C[None] - F[i:i + 256, None], axis=-1).mean(-1).astype(np.float16) for i in range(0, len(F), 256)])


N = np.load(f"{W}/data/cands_v2_new.npy"); n = len(N); K0 = 5121; Kp = K0 + n
C = np.load(ENV("CANDS")); C2 = C[:, :, :2].astype(np.float64)
rows = [json.loads(l) for l in open(f"{P}/train.jsonl")]
assert [r["compact"]["row"] for r in rows] == list(range(len(rows))) and all(r["compact"]["k"] == K0 for r in rows)
R0 = np.load(f"{P}/train_rewards.npy", mmap_mode="r"); assert R0.shape == (len(rows), K0), R0.shape
state = open(f"{P}/train_rewards.state").read().strip() if os.path.exists(f"{P}/train_rewards.state") else ""
by = {}
for j, r in enumerate(rows):
    by.setdefault(r["source"], []).append(j)
if LIMIT:
    by = {s: v[:LIMIT] for s, v in by.items()}
keep = sorted(j for v in by.values() for j in v)
unknown = set(by) - {"navtrain", "pai_human", "wod", "pai_dagger"}
assert not unknown, f"no reward rule for {unknown}"
new = {}   # source -> (token -> row of its new-column table, table), loaded for the sources the pool holds
if "navtrain" in by:
    z = np.load(f"{W}/score/navtrain_pdms_new_fit.npz"); new["navtrain"] = ({t: i for i, t in enumerate(tk(z["tokens"]))}, z["scores"], "pdm")
if "pai_human" in by:
    z = np.load(f"{W}/rewards/pai_ade_new.npz"); new["pai_human"] = ({t: i for i, t in enumerate(tk(z["tokens"]))}, np.load(f"{W}/rewards/pai_reward_new.npy", mmap_mode="r"), "v")
if "wod" in by:
    z = np.load(f"{W}/rewards/wod_rfs_new.npz"); new["wod"] = ({t: i for i, t in enumerate(tk(z["tokens"]))}, z["rfs"], "rfs")
if "pai_dagger" in by:
    assert DAG, "pai_dagger rows: set DAG=<shard name>"
    z = np.load(f"{W}/rewards/dag_{DAG}_new.npz"); new["pai_dagger"] = ({t: i for i, t in enumerate(tk(z["tokens"]))}, z["reward"], "v")


def new_cols(src, js):
    ix, T, kind = new[src]; t = [ix[rows[j]["token"]] for j in js]
    if kind == "pdm":
        return nav_v(T[t])
    if kind == "rfs":
        return (T[t].astype(np.float64) / 10).astype(np.float32)
    return np.asarray(T[t], np.float32)


# check: 16 existing columns by the same rules reproduce the pool's rewards bit for bit (a sample per source)
cols = np.sort(np.random.default_rng(20261027).choice(K0, 16, replace=False)); chk = {}
for src, js in by.items():
    js = js[:2000]; got = None
    if src == "navtrain" and state.startswith("final"):
        ft = {t: i for i, t in enumerate(l.strip() for l in open(f"{E}/rl2_draw/fit_tokens.txt") if l.strip())}
        A = npz_mm(f"{E}/rl2_draw/navtrain_pdms.npz", "scores"); got = nav_v(np.stack([np.asarray(A[ft[rows[j]["token"]]])[cols] for j in js]))
    elif src == "pai_human":
        zi = np.load(f"{E}/in/e2e_inputs.npz"); ii = {str(t): i for i, t in enumerate(zi["pai_tok"])}
        got = (-0.2 * ade16(zi["pai_F"][[ii[rows[j]["token"]] for j in js]].astype(np.float64), C2[cols]).astype(np.float64)).astype(np.float32)
    elif src == "wod":
        zw = np.load(f"{E}/rl_draw/wod_rfs.npz"); iw = {str(t): i for i, t in enumerate(zw["tokens"])}
        got = (zw["rfs"][[iw[rows[j]["token"]] for j in js]][:, cols].astype(np.float64) / 10).astype(np.float32)
    elif src == "pai_dagger":
        L = np.load(f"{E}/dagger/{DAG}/train_labels.npz"); il = {str(t): i for i, t in enumerate(L["token"])}
        got = (-0.2 * ade16(L["E"][[il[rows[j]["token"]] for j in js]].astype(np.float64), C2[cols]).astype(np.float64)).astype(np.float32)
    if got is None:
        chk[src] = "not checked (navtrain rewards of this pool are not final: " + state[:80] + ")"
        continue
    ref = np.stack([np.asarray(R0[j])[cols] for j in js]); ok = bool(np.array_equal(got, ref))
    chk[src] = {"rows": len(js), "equal": ok, "max_abs_diff": float(np.abs(got - ref).max())}
    assert ok, (src, chk[src])
print(json.dumps({"check": chk}), flush=True)
os.makedirs(O, exist_ok=False)
m = len(keep); pos = {j: i for i, j in enumerate(keep)}
R = np.lib.format.open_memmap(f"{O}/train_rewards.npy", "w+", np.float32, (m, Kp))
for src, js in by.items():
    for a in range(0, len(js), 2000):
        b = js[a:a + 2000]; ii = [pos[j] for j in b]
        R[ii, :K0] = np.asarray(R0[b]); R[ii, K0:] = new_cols(src, b)
R.flush()
assert all(np.isfinite(np.asarray(R[a:a + 20000])).all() for a in range(0, m, 20000))
T = np.lib.format.open_memmap(f"{O}/train_targets.npy", "w+", np.float16, (m, Kp))
for a in range(0, m, 20000):
    T[a:a + 20000] = np.float16(1 / Kp)
T.flush(); del T
with open(f"{O}/train.jsonl", "w") as f:
    for i, j in enumerate(keep):
        r = dict(rows[j]); r["compact"] = {"k": Kp, "row": i}; f.write(json.dumps(r) + "\n")
if os.path.exists(f"{P}/dev_targets.npy"):
    D0 = np.load(f"{P}/dev_targets.npy", mmap_mode="r"); assert D0.shape[1] == K0
    np.save(f"{O}/dev_targets.npy", np.concatenate([np.asarray(D0, np.float32), np.zeros((len(D0), n), np.float32)], 1).astype(D0.dtype))
    with open(f"{P}/dev.jsonl") as fi, open(f"{O}/dev.jsonl", "w") as fo:
        for l in fi:
            r = json.loads(l)
            if "compact" in r:
                r["compact"] = dict(r["compact"], k=Kp)
            fo.write(json.dumps(r) + "\n")
if REF:
    nn = np.array(json.load(open(ENV("EXT_JSON", f"{W}/data/extend_head_nn.json")))["nearest_old"]); assert len(nn) == n and DELTA == DELTA
    F0 = np.load(REF, mmap_mode="r"); assert F0.shape == (len(rows), K0), F0.shape
    G = np.lib.format.open_memmap(f"{O}/train_ref.npy", "w+", F0.dtype, (m, Kp))
    for a in range(0, m, 20000):
        b = keep[a:a + 20000]; x = np.asarray(F0[b]); G[a:a + len(b), :K0] = x; G[a:a + len(b), K0:] = (x[:, nn].astype(np.float32) - DELTA).astype(F0.dtype)
    G.flush(); open(f"{O}/train_ref.init", "w").write(f"extend_head.py init of {REF} with DELTA {DELTA} (valid only before any fit of the new rows)\n")
for f in ("copies.tsv", "held_blocks.json", "images_mode", "nav_tokens.txt"):
    if os.path.exists(f"{P}/{f}"):
        os.symlink(os.path.realpath(f"{P}/{f}"), f"{O}/{f}")
open(f"{O}/train_rewards.state", "w").write(f"final: {state[:200]} + {n} new columns (cand v2, K' {Kp}){' LIMIT ' + str(LIMIT) + ' (dry run)' if LIMIT else ''}\n")
rep = {"from": os.path.realpath(P), "K_old": K0, "K_new": Kp, "rows": m, "rows_by_source": {s: len(v) for s, v in by.items()}, "check": chk,
       "limit": LIMIT, "ref": REF, "delta": DELTA if REF else None,
       "new_best_share_by_source": {s: round(float(np.mean([np.asarray(R[pos[j]]).argmax() >= K0 for j in v[:5000]])), 4) for s, v in by.items()}}
json.dump(rep, open(f"{O}/report.json", "w"), indent=1); print(json.dumps(rep))
