"""The RL pool with 5,121 reward columns (the pool the RL ablations ran on, and the base of the K' pool), in the earlier e2e
pool's compact format for the exact_rl objective.
  navtrain   all 92,339 fit scenes in fit_tokens.txt order (the merged table's): the earlier e2e pool's row where it has one, else
             superglance.data.scene.c256_row on navtrain_inputs.jsonl (checked equal to that pool's question and image on every token
             both have); v = PDMS x DDC - 10 x [NC < 1 | DAC < 1 | TTC < 1]
  pai_human  every pai_human row of the earlier e2e pool; v = -0.2 x ADE (m; float16) of each candidate to the human's 4-s future
             (e2e_inputs.npz pai_F)
  wod        the 479 rated WOD-E2E val rows of the first RL pool; v = RFS / 10 (<rl_draw>/wod_rfs.npz)
Subcommands:
  rows      everything; the navtrain rewards are provisional (the 4,097-candidate table's columns 0-4,096; columns 4,097-5,120 the
            15K draw's where it has them, else the row's minimum over the 4,097), and train_rewards.state says so.
            Checks: the first RL pool's 15K PAI-AV rows get exactly -0.2 x <rl_draw>/pai_ade.npz.
  final     <navtrain_pdms.npz>: the merged table must have the fit tokens in order, shape [92,339, 5,121, 7], finite values, its
            15K draw rows equal to <rl_draw>/navtrain_pdms.npz and its columns 0-4,096 equal to the 4,097-candidate table's exactly;
            then the navtrain rewards in place, the gold indices, state final.
  refcheck  <train_ref.npy> <first RL pool> <its train_ref.npy>: finite, one row per compact row, and against the first RL pool's
            reference on its 30,479 rows: logit differences, top-1 agreement, KL.
copies.tsv: the source of every train and dev image (navtrain cam_f0; the earlier pool's other sources as they are; the rated WOD
rows under WOD_VAL_IMG, the 200 WOD dev images under WOD_DEV_IMG). Rows point at IMG_ROOT (where the images are staged).
usage: python -m superglance.builders.pools.rl_rows rows <earlier e2e pool> <navtrain_inputs.jsonl> <fit_tokens.txt> <e2e_inputs.npz>
           <cands.npy> <4,097-candidate scores.npz> <rl_draw> <first RL pool> <out>
       python -m superglance.builders.pools.rl_rows final <out> <navtrain_pdms.npz> <rl_draw>
       python -m superglance.builders.pools.rl_rows refcheck <out> <train_ref.npy> <first RL pool> <its train_ref.npy>"""
import json, os, random, sys
import numpy as np

K, CH = 5121, 2000
WODV, WODD = os.environ.get("WOD_VAL_IMG", ""), os.environ.get("WOD_DEV_IMG", "")
IMG_ROOT = os.environ.get("IMG_ROOT", "img").rstrip("/")
tok = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def nav_v(S):   # [n, K', 7] PDM metrics (NC, DAC, EP, TTC, C, DDC, PDMS) -> [n, K'] float32, RL-1's rule in float64
    S = S.astype(np.float64)
    return (S[..., 6] * S[..., 5] - 10.0 * ((S[..., 0] < 1) | (S[..., 1] < 1) | (S[..., 3] < 1))).astype(np.float32)


def ade16(F, C):   # [n, 8, 2], [K, 8, 2] -> [n, K] float16, as RL-1's pai_ade.npz
    return np.concatenate([np.linalg.norm(C[None] - F[i:i + 256, None], axis=-1).mean(-1).astype(np.float16) for i in range(0, len(F), 256)])


def c4096_rows(c4096, C, toks):   # cand4096's PDM metrics for toks, [n, 4097, 7]
    z = np.load(c4096); assert np.array_equal(z["cands"], C[:4097]), "cand4096's candidates are not the first 4,097"
    ix = {k: i for i, k in enumerate(tok(z["tokens"]))}
    return z["scores"][np.array([ix[k] for k in toks])]


def rows(pe, nav_inputs, fit_tokens, inputs, cands, c4096, rd, ra, out):
    from superglance.data.scene import c256_row, load_scenes
    os.makedirs(out)   # never overwrite
    C = np.load(cands); assert C.shape == (K, 8, 3), C.shape
    pool, pai = {}, []
    for l in open(f"{pe}/train.jsonl"):
        r = json.loads(l)
        if r["source"] == "navtrain":
            pool[r["token"]] = r
        elif r["source"] == "pai_human":
            pai.append(r)
    src_of = dict(reversed(l.rstrip("\n").split("\t")) for l in open(f"{pe}/copies.tsv"))   # staged path -> its source in the earlier e2e pool
    fit = [l.strip() for l in open(fit_tokens) if l.strip()]; scenes = load_scenes(nav_inputs); nav, cp, same = [], {}, 0
    for t in fit:
        s = scenes[t]; q = c256_row(s, k=K)["question"]; img = f"{IMG_ROOT}/nav/{os.path.basename(s['image'])}"
        cp[img] = s["image"]; p = pool.get(t)
        if p is not None:
            assert p["question"] == q and p["images"] == [img], t; same += 1; nav.append(dict(p))
        else:
            nav.append({"question": q, "no_options": True, "fixed_order": True, "id": f"navtrain_{t}", "token": t, "source": "navtrain",
                        "sampling_task": "navtrain", "sampling_bucket": "all", "images": [img]})
    for r in pai:
        cp[r["images"][0]] = src_of[r["images"][0]]
    miss = [a for a in random.Random(20261023).sample([cp[r["images"][0]] for r in pai], 2000) if not os.path.exists(a)]
    assert not miss, f"PAI sources missing: {miss[:3]}"
    wz = np.load(f"{rd}/wod_rfs.npz"); wt = tok(wz["tokens"])
    wr = {r["token"]: r for r in map(json.loads, open(f"{ra}/train.jsonl")) if r["source"] == "wod"}
    wod = [dict(wr[t]) for t in wt]
    for r in wod:
        cp[r["images"][0]] = f"{WODV}/{os.path.basename(r['images'][0])}"
    for r in map(json.loads, open(f"{pe}/dev.jsonl")):
        for i in r["images"]:
            cp[i] = f"{WODD}/{os.path.basename(i)}" if "/img/wod/" in i else src_of[i]
    n = [len(nav), len(pai), len(wod)]; N = sum(n)
    R = np.lib.format.open_memmap(f"{out}/train_rewards.npy", "w+", np.float32, (N, K))
    # navtrain, provisional: cand4096's columns; the fast columns from the 15K draw, else the row's minimum over the 4,097
    S4 = c4096_rows(c4096, C, fit); nz = np.load(f"{rd}/navtrain_pdms.npz"); i15 = {k: i for i, k in enumerate(tok(nz["tokens"]))}
    S15 = nz["scores"]
    for a in range(0, n[0], CH):
        b = min(a + CH, n[0]); v = nav_v(S4[a:b]); f = np.repeat(v.min(1, keepdims=True), K - 4097, 1)
        for j, t in enumerate(fit[a:b]):
            if t in i15:
                f[j] = nav_v(S15[i15[t]][None, 4097:])[0]
        R[a:b] = np.concatenate([v, f], 1)
    del S4
    # pai_human: -0.2 x ADE; RL-1's 15K rows must come out exactly as its table
    z = np.load(inputs); ix = {str(k): i for i, k in enumerate(z["pai_tok"])}; C2 = C[:, :, :2].astype(np.float64)
    A = ade16(z["pai_F"][[ix[r["token"]] for r in pai]].astype(np.float64), C2)
    pz = np.load(f"{rd}/pai_ade.npz"); ip = {r["token"]: j for j, r in enumerate(pai)}
    assert np.array_equal(A[[ip[t] for t in tok(pz["tokens"])]], pz["ade"]), "PAI ADE differs from RL-1's pai_ade.npz"
    for a in range(0, n[1], CH):
        b = min(a + CH, n[1]); R[n[0] + a:n[0] + b] = (-0.2 * A[a:b].astype(np.float64)).astype(np.float32)
    R[n[0] + n[1]:] = (wz["rfs"].astype(np.float64) / 10).astype(np.float32)
    R.flush()
    T = np.lib.format.open_memmap(f"{out}/train_targets.npy", "w+", np.float16, (N, K))   # the loader needs one; grpo ignores it
    for a in range(0, N, 20000):
        T[a:a + 20000] = np.float16(1 / K)
    T.flush(); del T
    with open(f"{out}/train.jsonl", "w") as fo:
        for j, r in enumerate(nav + pai + wod):
            g = int(np.argmax(R[j])); r.update(compact={"k": K, "row": j}, gold_index=g, correct_answer=f"t{g:04d}", label_source="rl_reward_c86g")
            r.pop("target_probs", None); fo.write(json.dumps(r) + "\n")
    need = {i for r in nav + pai + wod for i in r["images"]} | {i for r in map(json.loads, open(f"{pe}/dev.jsonl")) for i in r["images"]}
    assert need <= cp.keys(), sorted(need - cp.keys())[:5]
    with open(f"{out}/copies.tsv", "w") as fo:
        fo.writelines(f"{cp[i]}\t{i}\n" for i in sorted(need))
    for x in ("dev.jsonl", "dev_targets.npy", "held_blocks.json"):
        os.symlink(f"{pe}/{x}", f"{out}/{x}")
    open(f"{out}/nav_tokens.txt", "w").write("\n".join(fit) + "\n")
    open(f"{out}/train_rewards.state", "w").write("provisional: navtrain from cand4096 + the 15K draw + row minimum (smoke only)\n")
    V = [np.asarray(R[a:b]) for a, b in ((0, n[0]), (n[0], n[0] + n[1]), (n[0] + n[1], N))]
    rep = {"navtrain": n[0], "navtrain_from_pool": same, "navtrain_built": n[0] - same, "pai_human": n[1], "wod": n[2], "rows": N,
           "images": len(need), "v_mean": {s: round(float(v.mean()), 4) for s, v in zip(("navtrain_provisional", "pai", "wod"), V)},
           "pai_15k_equal_rl1": True}
    json.dump(rep, open(f"{out}/report.json", "w"), indent=1); print(json.dumps(rep))


def final(out, table, rd):
    fit = [l.strip() for l in open(f"{out}/nav_tokens.txt") if l.strip()]; n = len(fit)
    z = np.load(table); assert tok(z["tokens"]) == fit, "table tokens are not the fit order"
    S = z["scores"]; assert S.shape == (n, K, 7), S.shape
    for a in range(0, n, CH):
        assert np.isfinite(S[a:a + CH].astype(np.float32)).all(), f"non-finite at rows {a}+"
    nz = np.load(f"{rd}/navtrain_pdms.npz"); ix = {k: i for i, k in enumerate(fit)}
    assert np.array_equal(S[np.array([ix[k] for k in tok(nz["tokens"])])], nz["scores"]), "the 15K draw rows differ"
    R = np.load(f"{out}/train_rewards.npy", mmap_mode="r+"); v4 = 0
    for a in range(0, n, CH):   # R also holds the PAI and WOD rows after row n
        b = min(a + CH, n); v = nav_v(S[a:b]); v4 += int(np.array_equal(v[:, :4097], R[a:b, :4097])); R[a:b] = v
    assert v4 == -(-n // CH), "columns 0-4,096 differ from cand4096's"
    R.flush(); g = np.concatenate([np.argmax(R[a:min(a + CH, n)], 1) for a in range(0, n, CH)])
    with open(f"{out}/train.jsonl") as fi, open(f"{out}/train.jsonl.tmp", "w") as fo:
        for j, l in enumerate(fi):
            if j < n:
                r = json.loads(l); assert r["source"] == "navtrain" and r["token"] == fit[j]
                r.update(gold_index=int(g[j]), correct_answer=f"t{int(g[j]):04d}"); l = json.dumps(r) + "\n"
            fo.write(l)
    os.replace(f"{out}/train.jsonl.tmp", f"{out}/train.jsonl")
    V = np.asarray(R[:n]); P = S[..., 6].astype(np.float32)
    rep = json.load(open(f"{out}/report.json"))
    rep.update(navtrain_final={"table": table, "v_mean": round(float(V.mean()), 4), "v_best_mean": round(float(V.max(1).mean()), 4),
                               "penalized_share": round(float((V < 0).mean()), 4), "best_pdms_mean": round(float(P.max(1).mean()), 4),
                               "rows_15k_equal_draw": True, "cols_4097_equal_cand4096": True})
    rep["v_mean"].pop("navtrain_provisional", None); json.dump(rep, open(f"{out}/report.json", "w"), indent=1)
    open(f"{out}/train_rewards.state", "w").write(f"final: navtrain from {table}\n"); print(json.dumps(rep["navtrain_final"]))


def refcheck(out, ref, ra, ref_a):
    Q = np.load(ref, mmap_mode="r"); rc = {}
    for l in open(f"{out}/train.jsonl"):
        r = json.loads(l); rc[(r["source"], r["token"])] = r["compact"]["row"]
    assert Q.shape == (len(rc), K), Q.shape
    for a in range(0, len(Q), 20000):
        assert np.isfinite(np.asarray(Q[a:a + 20000], np.float32)).all(), f"non-finite at rows {a}+"
    old = [(r["source"], r["token"], r["compact"]["row"]) for r in map(json.loads, open(f"{ra}/train.jsonl")) if "compact" in r]
    Qa = np.load(ref_a, mmap_mode="r"); dmax, dsum, top, kl, m = 0.0, 0.0, 0, 0.0, 0
    for a in range(0, len(old), 1000):
        b = old[a:a + 1000]; x = np.asarray(Qa[[j for _, _, j in b]], np.float64); y = np.asarray(Q[[rc[(s, t)] for s, t, _ in b]], np.float64)
        d = np.abs(x - y); dmax = max(dmax, float(d.max())); dsum += float(d.mean(1).sum()); top += int((x.argmax(1) == y.argmax(1)).sum())
        lp = y - y.max(1, keepdims=True); lp -= np.log(np.exp(lp).sum(1, keepdims=True))
        lq = x - x.max(1, keepdims=True); lq -= np.log(np.exp(lq).sum(1, keepdims=True))
        kl += float((np.exp(lp) * (lp - lq)).sum()); m += len(b)
    rep = {"rows": len(Q), "rl1_rows_compared": m, "logit_absdiff_max": round(dmax, 4), "logit_absdiff_mean": round(dsum / m, 5),
           "top1_agree": round(top / m, 4), "kl_new_vs_rl1_mean": round(kl / m, 6)}
    print(json.dumps(rep))   # bf16 batches of other rows move logits a little; other prompts or images would move them a lot
    assert rep["top1_agree"] >= 0.95 and rep["kl_new_vs_rl1_mean"] <= 0.01, "the reference differs from RL-1's on its rows"


if __name__ == "__main__":
    {"rows": rows, "final": final, "refcheck": refcheck}[sys.argv[1]](*sys.argv[2:])
