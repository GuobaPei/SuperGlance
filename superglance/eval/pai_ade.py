"""PAI-AV open-loop ADE at 3 s / 5 s (10 Hz) of superglance.eval.pai_rows rows. A pick is one of the 4 s candidates (8 poses
at 0.5 s): linear from the origin through the 8 poses for 0.1-4.0 s, then 4.1-5.0 s at the last segment's velocity (as
superglance.builders.pools.wod_rfs.to20), against fut10 (the logged ego positions). top1 = the head's argmax (code_<r>.npy of
superglance.eval.features). Six trajectories per frame, average ADE and min ADE over them (best of six): the head's six
highest-scoring candidates, from h_<r>.npy and the adapter's Policy Head when the features dir has adapter.txt (the adapter's
path). Reports all frames and the T0 frames alone (ol+00, the official eval's t0 = 5.1 s).
References: the oracle candidate (min ADE to the logged future over the 8 poses) and constant velocity from ego_velocity.
usage: python -m superglance.eval.pai_ade <rows.jsonl> <cands.npy> [<name>=<features dir> ...]"""
import json, os, sys
import numpy as np
from superglance.eval.heads import head

T8, T10 = np.arange(1, 9) * 0.5, np.arange(1, 51) * 0.1


def to10(p8):   # [..., 8, 2] at 0.5-4 s -> [..., 50, 2] at 0.1-5 s
    sh = p8.shape[:-2]; p8 = p8.reshape(-1, 8, 2)
    t = np.concatenate([[0.0], T8]); p = np.concatenate([np.zeros((len(p8), 1, 2)), p8], 1)
    out = np.stack([np.stack([np.interp(T10, t, q[:, k]) for k in (0, 1)], -1) for q in p])
    late = T10 > 4 + 1e-9; out[:, late] = p8[:, 7:8] + (p8[:, 7:8] - p8[:, 6:7]) / 0.5 * (T10[late] - 4)[None, :, None]
    return out.reshape(*sh, 50, 2)


def report(name, R, traj, six=None):   # traj [N, 50, 2]; six [N, 6, 50, 2]
    F = np.array([r["fut10"] for r in R]); d = np.linalg.norm(traj - F, axis=-1)
    d6 = np.linalg.norm(six - F[:, None], axis=-1) if six is not None else None
    for sub, m in (("all", np.ones(len(R), bool)), ("T0", np.array([r["token"].endswith("_ol+00") for r in R]))):
        s = f"{name:22s} {sub:3s} n {m.sum():4d}  top1 {d[m, :30].mean():.3f} / {d[m].mean():.3f}"
        if d6 is not None:
            a3, a5 = d6[m, :, :30].mean(-1), d6[m].mean(-1)
            s += f"  avg6 {a3.mean():.3f} / {a5.mean():.3f}  min6 {a3.min(1).mean():.3f} / {a5.min(1).mean():.3f}"
        print(s + "   (ADE@3 s / 5 s, m)")


def main():
    R = [json.loads(l) for l in open(sys.argv[1])]; C = np.load(sys.argv[2])[:, :, :2]
    F8 = np.array([r["fut10"] for r in R])[:, 4::5][:, :8]
    report("oracle candidate", R, to10(C[np.linalg.norm(C[None] - F8[:, None], axis=-1).mean(-1).argmin(1)]))
    report("constant velocity", R, np.array([r["ego_velocity"] for r in R])[:, None] * T10[None, :, None])
    for arg in sys.argv[3:]:
        name, xd = arg.split("=", 1); pick, top = {}, {}
        W = head(open(f"{xd}/adapter.txt").read().strip()) if os.path.exists(f"{xd}/adapter.txt") else None
        for k in range(64):
            if os.path.exists(f"{xd}/tok_{k}.txt"):
                tk = open(f"{xd}/tok_{k}.txt").read().split(); pick.update(zip(tk, np.load(f"{xd}/code_{k}.npy")))
                if W is not None:
                    top.update(zip(tk, np.argsort(-(np.load(f"{xd}/h_{k}.npy").astype(np.float32) @ W[0].T + W[1]), 1)[:, :6]))
        S = [r for r in R if r["token"] in pick]
        if top:
            agree = np.mean([top[r["token"]][0] == pick[r["token"]] for r in S])
            print(f"{name:22s} top-6 from h: argmax agrees with code on {agree:.3f} of frames (h is float16)")
        report(name, S, to10(C[[int(pick[r["token"]]) for r in S]]), to10(C[[top[r["token"]] for r in S]]) if top else None)


if __name__ == "__main__":
    p8 = np.cumsum(np.full((1, 8, 2), [5.0, 0.0]), 1)   # self-check: 10 m/s straight -> 1 m per 0.1 s, also past 4 s
    assert np.allclose(to10(p8)[0, :, 0], T10 * 10) and np.allclose(to10(p8)[0, :, 1], 0) and to10(np.stack([p8, p8], 1)).shape == (1, 2, 50, 2)
    main()
