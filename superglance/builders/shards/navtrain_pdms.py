"""navtrain shard with the PDMS soft targets (the paper's navtrain target): the navtrain shard with only its targets changed to
superglance.builders.shards.pdms_targets' q. train / dev rows: gold_index = argmax q, correct_answer = t<gold>, label_source =
pdm_scored_5121_softmax_t0.1_b0.1, and the row's target (compact.row) = q; every other field and the val split stay as they are.
Rows are re-serialized only after checking that json.dumps reproduces every original line byte for byte.
usage: python -m superglance.builders.shards.navtrain_pdms <navtrain shard dir> <q npz> <out dir>"""
import json, os, shutil, sys
import numpy as np

src, qp, out = sys.argv[1:4]
z = np.load(qp); tok = [str(t) for t in z["tokens"]]; split = z["split"]; Q = z["q"]; G = z["gold"]; LS = str(z["label_source"])
ix = {(int(s), t): i for i, (s, t) in enumerate(zip(split, tok))}
os.makedirs(out)   # never overwrite
rep = json.load(open(f"{src}/report.json")); rep["pdms"] = {"from_q": qp, "label_source": LS, "tau": float(z["tau"]), "beta": float(z["beta"])}
for sp, code in (("train", 0), ("dev", 1)):
    lines = open(f"{src}/{sp}.jsonl").read().splitlines(); T0 = np.load(f"{src}/{sp}_targets.npy", mmap_mode="r")
    T = np.lib.format.open_memmap(f"{out}/{sp}_targets.npy", "w+", np.float32, T0.shape)
    with open(f"{out}/{sp}.jsonl", "w") as f:
        for n, l in enumerate(lines):
            r = json.loads(l); assert json.dumps(r, ensure_ascii=False) == l, (sp, n)
            i = ix[(code, r["token"])]; k = r["compact"]["row"]
            r.update(gold_index=int(G[i]), correct_answer=f"t{int(G[i]):04d}", label_source=LS)
            T[k] = Q[i]; f.write(json.dumps(r, ensure_ascii=False) + "\n")
    T.flush()
    rows = [json.loads(l) for l in lines]; new = [json.loads(l) for l in open(f"{out}/{sp}.jsonl")]
    keys = {"gold_index", "correct_answer", "label_source"}
    assert all({k: v for k, v in a.items() if k not in keys} == {k: v for k, v in b.items() if k not in keys} for a, b in zip(rows, new))
    assert sorted(r["compact"]["row"] for r in rows) == list(range(len(rows))) == list(range(T0.shape[0]))
    s = np.asarray(T, np.float64).sum(1)
    rep["pdms"][sp] = {"rows": len(rows), "q_sum_abs_err_max": float(np.abs(s - 1).max()), "gold_changed": sum(a["gold_index"] != b["gold_index"] for a, b in zip(rows, new)),
                       "other_fields_equal": len(rows)}
for f in ("val.jsonl", "val_targets.npy"):
    shutil.copy2(f"{src}/{f}", f"{out}/{f}")
json.dump(rep, open(f"{out}/report.json", "w"), indent=1)
print(json.dumps(rep["pdms"]))
