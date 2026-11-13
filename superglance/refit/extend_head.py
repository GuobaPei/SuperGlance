"""The Policy Head widened from 5,121 to K' rows: a copy of a LoRA adapter whose choice head (modules_to_save choice_head:
from the 5,121 rows to K' rows for the library cands_k<K'>_v2.npy. Rows 0-5,120 (weight and bias) are copied unchanged; each new row
5,121 + j starts as its nearest old candidate's row (mean pose distance over the 8 (x, y) poses) with the bias lowered by DELTA:
  logit_new_j(h) = logit_nn(j)(h) - DELTA   -> p_new_j = p_nn(j) x exp(-DELTA) before any training (DELTA >= ~4: the old argmax and
  ranking are kept on every input; superglance.refit.fit fits the new rows from there)
CPU only (torch on CPU, no model load). Writes <out>/adapter_model.safetensors (same keys and dtype; only the two choice_head tensors
change shape), <out>/adapter_config.json (copied: PEFT takes the head shape from the tensors; the model must be built with
choice_k = K'), <out>/extend_head.json (source, DELTA, K, K', nearest old row and distance per new row, the cands file) and a check:
on 64 random feature vectors the first 5,121 logits equal the source head's exactly and each new logit equals its nearest row's
minus DELTA up to the bias' rounding to the stored dtype (float32 math on the stored values).
Run it on the adapter the refit starts from (the last supervised phase's final checkpoint); never on a run's own directory.
FIT=<new_rows.npz> (superglance.refit.fit's W, b, nearest_old): the new rows are those fitted rows, rounded to the stored dtype,
instead of nearest - DELTA; the check is then old logits equal and every stored new value within the dtype's rounding of the fitted one.
usage: [FIT=new_rows.npz] python -m superglance.refit.extend_head <adapter dir> <old cands.npy> <K' cands.npy> <out dir> [DELTA=4.0]"""
import json, os, shutil, sys

import numpy as np
import torch
from safetensors.torch import load_file, save_file

KW, KB = "base_model.model.choice_head.weight", "base_model.model.choice_head.bias"


def nearest_old(old, new):   # [K, 8, >=2], [n, 8, >=2] -> nearest old index [n], mean pose distance [n]
    O = np.asarray(old, np.float64)[..., :2]; N = np.asarray(new, np.float64)[..., :2]
    d = np.linalg.norm(N[:, None] - O[None], axis=-1).mean(-1)
    return d.argmin(1), d.min(1)


def main(ad, old_p, kp_p, out, delta=4.0):
    delta = float(delta); old, allc = np.load(old_p), np.load(kp_p); K = len(old)
    assert allc.shape[1:] == (8, 3) and len(allc) > K and np.array_equal(allc[:K], old), "the K' file must start with the old library"
    st = load_file(f"{ad}/adapter_model.safetensors"); W, b = st[KW], st[KB]
    assert W.shape[0] == b.shape[0] == K, (W.shape, b.shape, K)
    nn_ix, nn_d = nearest_old(old, allc[K:])
    Wn = W[torch.as_tensor(nn_ix)].clone()
    bn = (b[torch.as_tensor(nn_ix)].float() - delta).to(b.dtype)
    fit = os.environ.get("FIT")
    if fit:   # the pilot's fitted rows (fitted from this very init: same nearest-row map)
        z = np.load(fit); Wf, bf = torch.from_numpy(z["W"]).float(), torch.from_numpy(z["b"]).float()
        assert Wf.shape == Wn.shape and bf.shape == bn.shape and np.array_equal(z["nearest_old"], nn_ix), (Wf.shape, Wn.shape)
        Wn, bn = Wf.to(W.dtype), bf.to(b.dtype)
        rnd = 2.0 ** -8 if W.dtype == torch.bfloat16 else 2.0 ** -11   # half an ulp, relative
        assert ((Wn.float() - Wf).abs() <= Wf.abs() * rnd).all() and ((bn.float() - bf).abs() <= bf.abs() * rnd).all()
    W2, b2 = torch.cat([W, Wn]).contiguous(), torch.cat([b, bn]).contiguous()
    # check: old logits unchanged, new = nearest - delta (in float32 on the stored values)
    g = torch.Generator().manual_seed(20261027); H = torch.randn(64, W.shape[1], generator=g)
    lo = H @ W.float().T + b.float(); l2 = H @ W2.float().T + b2.float()
    assert torch.equal(l2[:, :K], lo), "old logits changed"
    keep_argmax = bool((l2.argmax(1) == lo.argmax(1)).all())
    if fit:
        err = (l2[:, K:] - (H @ Wf.T + bf)).abs().max().item()   # reported: the dtype rounding's effect on the logits
    else:
        err = (l2[:, K:] - (lo[:, torch.as_tensor(nn_ix)] - delta)).abs().max().item()
        tol = float((b.float().abs().max() + delta) * 2 ** -7) + 1e-6   # only the bias' rounding to the stored dtype (bf16: 2^-8 relative)
        assert err <= tol, (err, tol)
    os.makedirs(out, exist_ok=False)
    st[KW], st[KB] = W2, b2
    save_file(st, f"{out}/adapter_model.safetensors", metadata={"format": "pt"})
    shutil.copy(f"{ad}/adapter_config.json", out)
    rep = {"from": os.path.realpath(ad), "cands": os.path.realpath(kp_p), "K_old": K, "K_new": len(allc), "delta": delta,
           "fit": os.path.realpath(fit) if fit else None,
           "dtype": str(W.dtype), "check_logit_err_max": err, "check_old_logits_equal": True, "check_argmax_kept_on_random_h": keep_argmax,
           "nearest_old": nn_ix.tolist(), "nearest_old_dist_m": np.round(nn_d, 3).tolist()}
    json.dump(rep, open(f"{out}/extend_head.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if not isinstance(v, list)}))


if __name__ == "__main__":
    main(*sys.argv[1:6])
