"""The training rows the K' extension may be tuned on, and their nearest-candidate distance to the 5,121.
  nav  the NAVSIM training ("fit") rows of navtrain_inputs.jsonl (92,339): logged future (x, y, heading), ego speed
  pai  the PAI-AV human rows of the stage-3 pools (e2e_inputs.npz pai_tok / pai_F) after the evaluation-key rule of
       superglance.builders.pools.clean (eval_keys.json: PAI-AV evaluation clips, the AlpaSim 916, PAI-AV's test split) and restricted to
       PAI-AV's train split; heading and ego speed joined from the PAI-AV fit rows (fit_all_clean.jsonl, pai_F's source, checked equal)
No evaluation row is read here.
Out: <out>/train_futures.npz (src 0 nav / 1 pai, tokens, F [N, 8, 3] float32, v0, a0 [N] float32, cmd [N] int8) and
     <out>/train_nn5121.npz (ade_min [N] float32, argmin [N] int32 against the 5,121).
usage: python -m superglance.builders.candidates.train_futures <navtrain_inputs.jsonl> <e2e_inputs.npz> <fit_all_clean.jsonl>
       <eval_keys.json> <cands_k4097_fast1024.npy> <out dir>"""
import json, os, sys

import numpy as np

from superglance.builders.candidates.kin import nearest  # noqa: E402

nav_p, inp_p, fit_p, keys_p, cands_p, out = sys.argv[1:7]
k = json.load(open(keys_p))
drop = set().union(*map(set, k["pai_eval"].values()), k["gold_test"], k["gold_val"], k["gold_other"], k["alpasim_916"], k["pai_test"],
                   k["curated_val"])
not_train = set(k["pai_test"]) | set(k["pai_val"])
T, F, V0, A0, CMD, SRC = [], [], [], [], [], []
for l in open(nav_p):
    r = json.loads(l)
    if r.get("subset") != "fit":
        continue
    T.append(r["token"]); F.append(np.asarray(r["future_poses"], np.float32)[:8, :3]); V0.append(np.hypot(*r["ego_velocity"][:2]))
    A0.append(r["ego_acceleration"][0]); CMD.append(int(np.argmax(r["driving_command"]))); SRC.append(0)
n_nav = len(T); assert n_nav == 92339, n_nav
z = np.load(inp_p); ptok = [str(t) for t in z["pai_tok"]]; PF = z["pai_F"]
clip = lambda t: t.rsplit("_", 1)[0]
keep = [i for i, t in enumerate(ptok) if clip(t) not in drop and clip(t) not in not_train]
want = {ptok[i]: i for i in keep}; got = {}
for l in open(fit_p):
    r = json.loads(l)
    if r["token"] in want:
        got[r["token"]] = r
assert len(got) == len(want), (len(got), len(want))
bad = 0
for i in keep:
    t = ptok[i]; r = got[t]; f = np.asarray(r["future_poses"], np.float32)[:8, :3]
    bad += int(not np.array_equal(f[:, :2], PF[i]))
    T.append(t); F.append(f); V0.append(np.hypot(*r["ego_velocity"][:2])); A0.append(r["ego_acceleration"][0])
    CMD.append(int(np.argmax(r["driving_command"]))); SRC.append(1)
assert bad == 0, f"{bad} PAI futures differ from pai_F"
F = np.stack(F); C = np.load(cands_p); assert C.shape == (5121, 8, 3)
os.makedirs(out, exist_ok=True)
np.savez(f"{out}/train_futures.npz", src=np.array(SRC, np.int8), tokens=np.array(T), F=F, v0=np.array(V0, np.float32),
         a0=np.array(A0, np.float32), cmd=np.array(CMD, np.int8))
m, a = nearest(F, C)
np.savez(f"{out}/train_nn5121.npz", ade_min=m, argmin=a)
rep = {"nav_fit": n_nav, "pai_pool_rows": len(ptok), "pai_dropped_eval_rule": sum(clip(t) in drop for t in ptok),
       "pai_dropped_not_train_split": sum(clip(t) not in drop and clip(t) in not_train for t in ptok), "pai_kept": len(keep),
       "pai_clips_kept": len({clip(ptok[i]) for i in keep})}
for s, name in ((0, "nav"), (1, "pai")):
    x = m[np.array(SRC) == s]
    rep[f"{name}_nn5121"] = {"median": round(float(np.median(x)), 3), "gt1m_pct": round(100 * float((x > 1).mean()), 3), "gt1m": int((x > 1).sum())}
json.dump(rep, open(f"{out}/train_futures.json", "w"), indent=1); print(json.dumps(rep))
