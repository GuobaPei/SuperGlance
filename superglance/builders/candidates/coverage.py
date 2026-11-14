"""The appendix's coverage table (Coverage of NAVSIM scenes) on the 10,549 held-out NAVSIM training scenes (the 2,000 evaluation scenes of
the scene draw + the 8,549 unused ones). Per set, as superglance.builders.candidates.study: best = mean over scenes of the highest candidate
PDMS; nearest = mean PDMS of the candidate with the smallest mean (x, y) waypoint distance to the human trajectory (float64).
  prep <navtrain_inputs.jsonl> <held-out list .jsonl (field t)> <evaluation tokens> <out dir>
        the 10,549 rows of navtrain_inputs.jsonl (checked: held out, all 2,000 evaluation scenes in) -> held_inputs.jsonl;
        k-means 256 + standing still with study.py's code and seed -> kmeans256.npy (then score it with
        superglance.builders.pools.score_candidates into <out dir>/k257)
  table <out dir> <evaluation tokens> <1,025 scores.npz> <4,097 scores.npz> <navtrain_pdms_k5273_held.npz> <cands_k4097_fast1024.npy>
        the four sets on the 10,549 and on the 2,000
usage: python -m superglance.builders.candidates.coverage prep|table ..."""
import json, sys, zipfile
import numpy as np



def prep(inp, fc, ev, out):
    from superglance.builders.candidates.study import kmeans
    held = [json.loads(l)["t"] for l in open(fc)]; E = set(open(ev).read().split())
    rows = [json.loads(l) for l in open(inp)]; by = {r["token"]: r for r in rows}
    assert len(held) == len(set(held)) == 10549 and E <= set(held), (len(held), len(E & set(held)))
    assert all(by[t]["subset"] != "fit" for t in held)
    hs = set(held)
    with open(f"{out}/held_inputs.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows if r["token"] in hs)
    fut = lambda r: np.asarray(json.loads(r["future_poses"]) if isinstance(r["future_poses"], str) else r["future_poses"], float)[:8]
    fit = np.stack([fut(r) for r in rows if r["subset"] == "fit"])
    C, lab = kmeans(fit[:, :, :2].reshape(len(fit), -1), 256)
    head = np.stack([fit[lab == j, :, 2].mean(0) if (lab == j).any() else np.zeros(8) for j in range(256)])
    np.save(f"{out}/kmeans256.npy", np.vstack([np.concatenate([C.reshape(256, 8, 2), head[:, :, None]], -1), np.zeros((1, 8, 3))]))
    print(f"fit {len(fit)}, held {len(held)} (eval {len(E)} in); kmeans256.npy written", flush=True)


def npz_rows(path, key, idx):   # rows idx of a stored (uncompressed) array inside an .npz, without reading the rest
    z = zipfile.ZipFile(path); i = z.getinfo(f"{key}.npy"); assert i.compress_type == 0
    with z.open(i) as fh:
        v = np.lib.format.read_magic(fh); sh, fo, dt = np.lib.format._read_array_header(fh, v); hl = fh.tell()
    with open(path, "rb") as f:
        f.seek(i.header_offset); h = f.read(30); n, m = int.from_bytes(h[26:28], "little"), int.from_bytes(h[28:30], "little")
    a = np.memmap(path, dt, "r", i.header_offset + 30 + n + m + hl, sh, "F" if fo else "C")
    o = np.argsort(idx); r = np.empty((len(idx),) + sh[1:], dt); r[o] = a[np.asarray(idx)[o]]
    return r


def table(out, ev, s1024, s4096, k5273, cands):
    held = [json.loads(l)["token"] for l in open(f"{out}/held_inputs.jsonl")]; E = set(open(ev).read().split())
    fut = {}
    for l in open(f"{out}/held_inputs.jsonl"):
        r = json.loads(l); f = r["future_poses"]; fut[r["token"]] = np.asarray(json.loads(f) if isinstance(f, str) else f, float)[:8, :2]
    gt = np.stack([fut[t] for t in held]); ev_m = np.array([t in E for t in held])
    sets = []
    z = np.load(f"{out}/k257/scores.npz"); assert [x.decode() for x in z["tokens"]] == held
    sets.append(("k-means 256 + stationary", z["cands"], z["scores"]))
    for name, p in (("k-means 1,024 + stationary", s1024), ("k-means 4,096 + stationary", s4096)):
        with np.load(p) as z0:
            tk, cands = [x.decode() for x in z0["tokens"]], z0["cands"]
        pos = {t: i for i, t in enumerate(tk)}; sets.append((name, cands, npz_rows(p, "scores", [pos[t] for t in held])))
    with np.load(k5273) as z1:
        tk = [str(x) for x in z1["tokens"]]; pos = {t: i for i, t in enumerate(tk)}; S = z1["scores"]
        sets.append(("library v1 (4,097 + 1,024 fast)", None, S[[pos[t] for t in held]][:, :5121]))
    v1 = np.load(cands)
    print(f"| set | K | best, {len(held)} | nearest, {len(held)} | best, {ev_m.sum()} | nearest, {ev_m.sum()} |\n|---|---|---|---|---|---|")
    for name, cands, S in sets:
        cands = v1 if cands is None else cands; assert len(cands) == S.shape[1], (name, cands.shape, S.shape)
        pd = S[:, :, 6].astype(np.float64)
        ade = np.concatenate([np.linalg.norm(cands[None, :, :, :2] - gt[s:s + 1024, None], axis=-1).mean(-1) for s in range(0, len(gt), 1024)])
        near = pd[np.arange(len(pd)), ade.argmin(1)]; best = pd.max(1)
        assert np.isfinite(pd).all(), name
        print(f"| {name} | {len(cands)} | {100 * best.mean():.2f} | {100 * near.mean():.2f} | {100 * best[ev_m].mean():.2f} | {100 * near[ev_m].mean():.2f} |", flush=True)


if __name__ == "__main__":
    {"prep": prep, "table": table}[sys.argv[1]](*sys.argv[2:])
