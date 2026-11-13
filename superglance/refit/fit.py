"""Head refit, 5,121 -> 5,273 Policy Head rows: the 152 new rows are fit on frozen features (superglance.eval.features of
the supervised model, the last supervised phase); the old rows, their bias and the backbone stay frozen. CPU (torch) or one GPU.
  fit   rows = the navtrain scenes the supervised pool trains on (targets/sft_pdms_targets_k5273.npz: softmax((PDMS x DDC
        - 0.1 ADE) / 0.1) over the 5,273 candidates, split 0 = fit) and a PAI-AV subset (pilot/pai_subset.npz: every row with
        >= 0.1 target mass on the new rows plus a random 36,000 of the rest, weighted back to the population; target
        softmax(-ADE / 0.5 m); the check tenth, crc32(clip) % 10 == 0, is validation only)
        loss  pi_nav E_nav[CE] + pi_pai E_pai,w[CE] (pi 1/3, 2/3) + lam/2 ||W_new - W_init||^2, CE over all 5,273 logits;
              W_init = superglance.refit.extend_head's (nearest old row, bias - delta), the bias unpenalised; L-BFGS, full batch
        selection over the lams: those whose navtrain held-out PDMS (argmax, table read) is >= the 5,121 head's - 0.05; of those
              the lowest weighted PAI-check ADE@3 s; ties and the no-survivor case go to the largest lam.
              -> <out>/new_rows.npz (W, b, lam, delta, nearest_old), <out>/fit.json
  eval  table reads of a head's argmax against the 5,121 head on the same features (paired bootstrap 95 % CI): navtest PDMS
        (navtest/{old,new}/scores.npz, the PDM scores of every candidate), navtrain held-out PDMS, PAI-AV val / test ADE@3 s
        and 5 s (tables/pai_ade_k5273_{paival,pai700}.npz) -> the --out json. scripts/train_stage3.sh applies the stop rule.
Refit data directory (--data): data/cands_k5273_v2.npy, targets/, pilot/, tables/, navtest/inputs.jsonl, navtest/{old,new}/.
Features (--feat): nav, pai, navtest, paival, pai700 (superglance.refit.rows makes their rows).
usage: python -m superglance.refit.fit fit  --base A --data D --feat F --heldout T --out DIR [--lams 1e-1,1e-2,1e-3] [--device cpu]
       python -m superglance.refit.fit eval --base A --data D --feat F --heldout T --adapter A' --full --out J"""
import argparse, glob, json, os, sys, time, zlib
import numpy as np

KW, KB = "base_model.model.choice_head.weight", "base_model.model.choice_head.bias"; K = 5121
BASE = W_ = HF = HELD = None; PAI = {}   # set in __main__ from the command line


def head(ad):   # float32 W [K, d], b [K] of an adapter's choice head
    from safetensors import safe_open
    with safe_open(f"{ad}/adapter_model.safetensors", "pt") as f:
        return f.get_tensor(KW).float().numpy(), f.get_tensor(KB).float().numpy()


def tstr(a):   # token arrays: score_candidates.py stores S16 bytes, the others unicode
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def feats(dirs, want):   # float32 [len(want), d] in want's order, from superglance.eval.features dirs (h_<r>.npy float16, tok_<r>.txt)
    loc = {}
    for d in dirs:
        for tp in sorted(glob.glob(f"{d}/tok_*.txt")):
            r = tp.rsplit("_", 1)[1][:-4]
            for i, t in enumerate(x.strip() for x in open(tp) if x.strip()):
                loc.setdefault(t, (f"{d}/h_{r}.npy", i))
    miss = [t for t in want if t not in loc]
    assert not miss, f"{len(miss)} tokens without features, e.g. {miss[:3]}"
    files = {}
    for t in want:
        files.setdefault(loc[t][0], []).append(loc[t][1])
    pos = {t: j for j, t in enumerate(want)}; H = None
    for f, ii in files.items():
        m = np.load(f, mmap_mode="r"); H = np.empty((len(want), m.shape[1]), np.float32) if H is None else H
        ii = np.array(ii); o = np.argsort(ii); rows = np.asarray(m[ii[o]], np.float32)
        tk = [t for t in want if loc[t][0] == f]
        H[[pos[tk[k]] for k in o]] = rows
    return H


def soft(F, C, tau):   # e2e_pool.soft, verbatim
    out = np.empty((len(F), len(C)), np.float32)
    for i in range(0, len(F), 256):
        a = np.linalg.norm(C[None] - F[i:i + 256, None], axis=-1).mean(-1)
        q = np.exp(-(a - a.min(1, keepdims=True)) / tau); out[i:i + 256] = q / q.sum(1, keepdims=True)
    return out


def old_stats(H, Wo, bo, Q=None, chunk=4096):   # per row: lse, q.z over the old rows (Q given), max, argmax of the old logits
    import torch
    n = len(H); lse, qz, mx = np.empty(n), np.zeros(n), np.empty(n); am = np.empty(n, np.int64)
    Wt, bt = torch.from_numpy(Wo), torch.from_numpy(bo)
    for s in range(0, n, chunk):
        z = torch.from_numpy(H[s:s + chunk]) @ Wt.T + bt
        lse[s:s + chunk] = torch.logsumexp(z.double(), 1).numpy(); m = z.max(1); mx[s:s + chunk] = m.values.numpy(); am[s:s + chunk] = m.indices.numpy()
        if Q is not None:
            q = np.asarray(Q[s:s + chunk], np.float32)
            qz[s:s + chunk] = (torch.from_numpy(q[:, :K]).double() * z.double()).sum(1).numpy()
    return lse, qz, mx, am


def lookup_rows(tokens, table_tokens):
    ix = {t: i for i, t in enumerate(tstr(table_tokens))}
    return np.array([ix[t] for t in tokens])


# ----------------------------------------------------------------------------------------------------------------------------- fit
def fit(a):
    import torch
    torch.set_num_threads(int(os.environ.get("PILOT_THREADS", 32)))
    from superglance.refit.extend_head import nearest_old
    t0 = time.time(); os.makedirs(a.out, exist_ok=True); log = lambda *x: print(f"[{time.time() - t0:6.0f} s]", *x, flush=True)
    C = np.load(f"{W_}/data/cands_k5273_v2.npy"); C2 = C[:, :, :2].astype(np.float64); Kp = len(C)
    Wo, bo = head(BASE); assert Wo.shape[0] == K
    nn_ix, _ = nearest_old(C[:K], C[K:]); W0 = Wo[nn_ix].copy(); b0 = bo[nn_ix] - a.delta
    # navtrain fit rows and targets
    T = np.load(f"{W_}/targets/sft_pdms_targets_k5273.npz", mmap_mode="r")
    ttok = tstr(T["tokens"]); fit_rows = np.flatnonzero(np.asarray(T["split"]) == 0)
    navtok = [ttok[i] for i in fit_rows]; Hn = feats([HF], navtok); log(f"navtrain fit rows {len(navtok)} (features {Hn.shape})")
    Qn = np.asarray(T["q"][fit_rows], np.float32); assert Qn.shape[1] == Kp and np.allclose(Qn.sum(1), 1, atol=1e-4)
    # PAI rows
    z = np.load(f"{W_}/pilot/pai_subset.npz"); ptok = tstr(z["tokens"]); PF = z["F"].astype(np.float64)
    pw = z["weight"].astype(np.float64); pchk = z["check"]; Hp = feats([a.pai_feat], ptok)
    log(f"PAI rows {len(ptok)} (check {int(pchk.sum())}, {a.pai_feat})")
    Qp = soft(PF[:, :, :2], C2, 0.5)
    # held-out validation
    held = [t.strip() for t in open(HELD) if t.strip()]
    Ht = np.load(f"{W_}/tables/navtrain_pdms_k5273_held.npz", mmap_mode="r"); hrow = lookup_rows(held, Ht["tokens"])
    Hh = feats([HF], held); Sh = np.asarray(Ht["scores"][hrow][:, :, 6], np.float32)   # [n, K'] PDMS
    ade6 = lambda F, idx: np.linalg.norm(C2[idx][:, :6] - F[:, :6, :2], axis=-1).mean(-1)   # PAI check ADE@3 s (0.5-3 s poses)
    # training tensors: old-logit statistics once
    tr_p = ~pchk; Htr = np.concatenate([Hn, Hp[tr_p]]); Qtr = np.concatenate([Qn[:, K:], Qp[tr_p][:, K:]])
    ln, qn, _, _ = old_stats(Hn, Wo, bo, Q=Qn); lp, qp, _, _ = old_stats(Hp[tr_p], Wo, bo, Q=Qp[tr_p])
    om = np.concatenate([np.full(len(Hn), (1 / 3) / len(Hn)), (2 / 3) * pw[tr_p] / pw[tr_p].sum()])
    LSE, QZ = np.concatenate([ln, lp]), np.concatenate([qn, qp])
    vh = old_stats(Hh, Wo, bo); vp = old_stats(Hp[pchk], Wo, bo, Q=Qp[pchk])
    log(f"old-logit statistics done: train {len(Htr)}, held {len(Hh)}, PAI check {int(pchk.sum())}")
    dev = torch.device(a.device); tt = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=dev)
    Ht_, Q_, LSE_, QZ_, OM_ = tt(Htr), tt(Qtr), tt(LSE, torch.float64), tt(QZ, torch.float64), tt(om, torch.float64)
    W0_, b0_ = tt(W0), tt(b0)

    def ce(Wn, bn, H, Qnew, lse, qz):   # per-row CE over K' with the old part fixed
        zn = H @ Wn.T + bn
        return torch.logaddexp(lse, torch.logsumexp(zn.double(), 1)) - qz - (Qnew.double() * zn.double()).sum(1)

    def reads(Wn, bn):   # held PDMS and PAI-check ADE@3 s of the K' argmax, CE on the PAI check rows
        with torch.no_grad():
            out = {}
            for nm, H, st in (("held", Hh, vh), ("paichk", Hp[pchk], vp)):
                zn = (tt(H) @ Wn.T + bn).cpu().numpy(); new = zn.max(1) > st[2]; pick = np.where(new, K + zn.argmax(1), st[3])
                if nm == "held":
                    out.update(held_pdms=100 * float(Sh[np.arange(len(pick)), pick].mean()), held_new=int(new.sum()))
                else:
                    w = pw[pchk]; out.update(paichk_ade3=float((w * ade6(PF[pchk], pick)).sum() / w.sum()), paichk_new=int(new.sum()),
                                             paichk_ce=float((w * ce(Wn, bn, tt(H), tt(Qp[pchk][:, K:]), tt(st[0], torch.float64),
                                                                     tt(st[1], torch.float64)).cpu().numpy()).sum() / w.sum()))
            return out
    base = {"held_pdms": 100 * float(Sh[np.arange(len(held)), vh[3]].mean()), "paichk_ade3": float((pw[pchk] * ade6(PF[pchk], vp[3])).sum() / pw[pchk].sum())}
    init = reads(W0_, b0_); log(f"5,121 head: {base}; init (delta {a.delta}): {init}")
    res = {}
    for lam in a.lams:
        Wn = W0_.clone().requires_grad_(True); bn = b0_.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([Wn, bn], lr=1, max_iter=a.iters, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)
        def closure():
            opt.zero_grad(); L = (OM_ * ce(Wn, bn, Ht_, Q_, LSE_, QZ_)).sum() + lam / 2 * ((Wn - W0_) ** 2).sum().double()
            L.backward(); return L
        opt.step(closure); r = reads(Wn.detach(), bn.detach())
        r.update(lam=lam, loss=float(closure().detach()), dW=float((Wn - W0_).norm()), b_mean=float(bn.mean()), b_max=float(bn.max()), evals=opt.state[opt._params[0]]["func_evals"])
        res[lam] = (r, Wn.detach().cpu().numpy(), bn.detach().cpu().numpy()); log(f"lam {lam}: {r}")
    ok = [l for l in a.lams if res[l][0]["held_pdms"] >= base["held_pdms"] - 0.05]
    pickl = min(ok, key=lambda l: (round(res[l][0]["paichk_ade3"], 6), -l)) if ok else max(a.lams)
    r, Wb, bb = res[pickl]
    np.savez(f"{a.out}/new_rows.npz", W=Wb, b=bb, lam=pickl, delta=a.delta, nearest_old=nn_ix)
    rep = {"nav_rows": len(navtok), "pai_rows_train": int(tr_p.sum()), "pai_rows_check": int(pchk.sum()),
           "held_rows": len(held), "base_5121": base, "init": init, "grid": [res[l][0] for l in a.lams], "chosen_lam": pickl,
           "survivors": ok, "seconds": round(time.time() - t0)}
    json.dump(rep, open(f"{a.out}/fit.json", "w"), indent=1); log(f"chosen lam {pickl}: {r}")


# ---------------------------------------------------------------------------------------------------------------------------- eval
def boot(d, n=2000, seed=0):
    m = np.random.default_rng(seed).integers(0, len(d), (n, len(d))); s = d[m].mean(1)
    return [round(float(d.mean()), 4), round(float(np.quantile(s, 0.025)), 4), round(float(np.quantile(s, 0.975)), 4)]


def evaluate(a):
    from superglance.refit.kin import nearest
    Wk, bk = head(a.adapter); Wo, bo = head(BASE); Kp = len(Wk)
    C = np.load(f"{W_}/data/cands_k5273_v2.npy"); assert len(C) == Kp and np.array_equal(Wk[:K], Wo) and np.array_equal(bk[:K], bo)
    pick = lambda H: ((H @ Wk.T + bk).argmax(1), (H @ Wo.T + bo).argmax(1))
    out = {"adapter": os.path.realpath(a.adapter), "K": Kp}
    # navtest
    o_p, n_p = a.navtest_tables.split(",")
    zo, zn = np.load(o_p), np.load(n_p); tok = tstr(zo["tokens"])
    assert tok == tstr(zn["tokens"])
    have = set()
    for d in a.navtest_feat.split(","):
        for tp in glob.glob(f"{d}/tok_*.txt"):
            have |= {x.strip() for x in open(tp)}
    nt = [t for t in tok if t in have]
    if a.full:
        assert len(nt) == len(tok) == 12146, (len(nt), len(tok))
    rows = lookup_rows(nt, tok); S = np.concatenate([zo["scores"][rows], zn["scores"][rows]], 1).astype(np.float32)
    H = feats(a.navtest_feat.split(","), nt); pk, p0 = pick(H)
    inp = {r["token"]: r for r in map(json.loads, open(f"{W_}/navtest/inputs.jsonl"))}
    Fn = np.stack([np.asarray(json.loads(inp[t]["future_poses"]) if isinstance(inp[t]["future_poses"], str) else inp[t]["future_poses"], float)[:8, :3] for t in nt])
    bad = nearest(Fn, C[:K])[0] > 1.0; i = np.arange(len(nt))
    dec = {}   # the extraction's own argmax per navtest token
    for d in a.navtest_feat.split(","):
        for c in sorted(glob.glob(f"{d}/code_*.npy")):
            dec.update(zip(open(f"{d}/tok_{c.rsplit('_', 1)[1][:-4]}.txt").read().split(), np.load(c).tolist()))
    dc = np.array([dec[t] for t in nt])   # the decode's own picks: the table must give its 89.80 back
    out["navtest"] = {"n": len(nt), "full": len(nt) == 12146, "pdms_decode_code": round(100 * float(S[i, dc, 6].mean()), 3),
                      "h_argmax_eq_decode": round(float((p0 == dc).mean()), 4), "pdms_kprime": round(100 * float(S[i, pk, 6].mean()), 3),
                      "pdms_5121": round(100 * float(S[i, p0, 6].mean()), 3), "delta_ci": boot(100 * (S[i, pk, 6] - S[i, p0, 6])),
                      "metrics_kprime": np.round(100 * S[i, pk].mean(0), 2).tolist(), "metrics_5121": np.round(100 * S[i, p0].mean(0), 2).tolist(),
                      "new_picks": int((pk >= K).sum()), "badly_n": int(bad.sum()),
                      "badly_pdms": [round(100 * float(S[i[bad], pk[bad], 6].mean()), 2), round(100 * float(S[i[bad], p0[bad], 6].mean()), 2)] if bad.any() else None}
    # held-out
    held = [t.strip() for t in open(HELD) if t.strip()]
    Ht = np.load(f"{W_}/tables/navtrain_pdms_k5273_held.npz", mmap_mode="r"); hr = lookup_rows(held, Ht["tokens"])
    Sh = np.asarray(Ht["scores"][hr], np.float32); pk, p0 = pick(feats([HF], held)); i = np.arange(len(held))
    out["held"] = {"n": len(held), "pdms_kprime": round(100 * float(Sh[i, pk, 6].mean()), 3), "pdms_5121": round(100 * float(Sh[i, p0, 6].mean()), 3),
                   "delta_ci": boot(100 * (Sh[i, pk, 6] - Sh[i, p0, 6])), "new_picks": int((pk >= K).sum())}
    # PAI
    for nm, xd in PAI.items():
        z = np.load(f"{W_}/tables/pai_ade_k5273_{nm}.npz"); tk = tstr(z["tokens"])
        H = feats([xd], tk); pk, p0 = pick(H); i = np.arange(len(tk)); a3, a5, bad = z["ade3"], z["ade5"], z["badly"]
        code = dict(zip(open(f"{xd}/tok_0.txt").read().split(), np.load(f"{xd}/code_0.npy")))
        out[nm] = {"n": len(tk), "top1_kprime": [round(float(a3[i, pk].mean()), 4), round(float(a5[i, pk].mean()), 4)],
                   "top1_5121": [round(float(a3[i, p0].mean()), 4), round(float(a5[i, p0].mean()), 4)],
                   "top1_logged_code": round(float(a3[i, [int(code[t]) for t in tk]].mean()), 4),
                   "h_argmax_eq_code": round(float(np.mean(p0 == np.array([int(code[t]) for t in tk]))), 4),
                   "delta3_ci": boot(a3[i, pk] - a3[i, p0]), "new_picks": int((pk >= K).sum()), "badly_n": int(bad.sum()),
                   "badly_ade3": [round(float(a3[i[bad], pk[bad]].mean()), 4), round(float(a3[i[bad], p0[bad]].mean()), 4)],
                   "badly_delta3_ci": boot(a3[i[bad], pk[bad]] - a3[i[bad], p0[bad]])}
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["fit", "eval"]); ap.add_argument("--out", required=True)
    ap.add_argument("--base", required=True); ap.add_argument("--data", required=True); ap.add_argument("--feat", required=True)
    ap.add_argument("--heldout", required=True); ap.add_argument("--pai-feat", default=None); ap.add_argument("--lams", default="1e-1,1e-2,1e-3")
    ap.add_argument("--iters", type=int, default=300); ap.add_argument("--delta", type=float, default=4.0); ap.add_argument("--device", default="cpu")
    ap.add_argument("--adapter"); ap.add_argument("--navtest-feat", default=None)
    ap.add_argument("--navtest-tables", default=None); ap.add_argument("--full", action="store_true")
    a = ap.parse_args(); a.lams = [float(x) for x in a.lams.split(",")]
    BASE, W_, HF, HELD = a.base, a.data, f"{a.feat}/nav", a.heldout
    PAI = {"pai700": f"{a.feat}/pai700", "paival": f"{a.feat}/paival"}
    a.pai_feat, a.navtest_feat = a.pai_feat or f"{a.feat}/pai", a.navtest_feat or f"{a.feat}/navtest"
    a.navtest_tables = a.navtest_tables or f"{W_}/navtest/old/scores.npz,{W_}/navtest/new/scores.npz"
    fit(a) if a.mode == "fit" else evaluate(a)
