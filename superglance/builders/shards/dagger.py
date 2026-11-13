"""Rollout-row shard (source pai_dagger): the earlier pool's closed-loop recovery rows (training rows of the 916 AlpaSim
scenes left out, target softmax(-ADE / 0.5 m) to the recovery label) with the road-users block removed.
  prompt  strip() of the earlier question; asserted equal, row for row, to prompt() of the row's source (the rollout rounds'
          rows.jsonl) with the dropout on training rows, i.e. what the PAI-AV human rows produce for the same fields;
  target  soft(label, 0.5 m) with the recovery labels of e2e_inputs.npz (dag_E train, held_E dev);
  images  pai_dagger/<rows tag>/<file>.jpg = <round>/rows/<tag>/frames/<file>.jpg (the earlier pool's copies.tsv).
Out: <SHARDS>/pai_dagger/{train,dev}.jsonl, *_targets.npy, *_images.tsv, report.json.
usage: python -m superglance.builders.shards.dagger <earlier pool dir> <e2e_inputs.npz> <rollout rounds dir>"""
import glob, json, os, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from superglance.builders.shards.common import O, cands, link, ml, prompt, strip, write   # noqa: E402


def main(old, inp, rounds):
    D = f"{O}/pai_dagger"; os.makedirs(D)   # never overwrite
    z = np.load(inp); C = cands(); s916 = set(map(str, z["s916"]))
    lab = {"train": dict(zip(map(str, z["dag_tok"]), z["dag_E"])), "dev": dict(zip(map(str, z["held_tok"]), z["held_E"]))}
    scene = dict(zip(map(str, z["dag_tok"]), map(str, z["dag_scene"])))
    src = {}
    for l in open(f"{old}/copies.tsv"):
        a, b = l.rstrip("\n").split("\t")
        if "/img/dag/" in b:
            src[os.path.basename(b)] = a
    want = {}
    for split in ("train", "dev"):
        want[split] = [json.loads(l) for l in open(f"{old}/{split}.jsonl") if '"source": "pai_dagger"' in l]
    toks = {x["token"] for v in want.values() for x in v}; drow = {}
    for f in sorted(glob.glob(f"{rounds}/*/rows/*/rows.jsonl")):   # the rollout rounds' source rows
        for l in open(f):
            r = json.loads(l)
            if r["token"] in toks:
                drow[r["token"]] = r
    assert toks <= set(drow), len(toks - set(drow))
    link(D, "pai_dagger"); dirs = {}
    rep = {"from": old, "labels": inp, "tau_m": 0.5, "target": "e2e_pool.soft, softmax(-ADE / 0.5 m) to the v4c recovery label"}
    for split, xs in want.items():
        keep = [x for x in xs if split != "train" or scene[x["token"]] not in s916]
        rows, F, bad = [], [], []
        for x in keep:
            q = x["question"]; s = strip(q)
            if s != prompt(drow[x["token"]], drop=split == "train"):
                bad.append(x["token"])
            p = src[os.path.basename(x["images"][0])]; tag = p.split("/")[-3]
            assert p.split("/")[-2] == "frames" and dirs.setdefault(tag, os.path.dirname(p)) == os.path.dirname(p), p
            r = dict(x, question=s, images=[f"pai_dagger/{tag}/{os.path.basename(p)}"], label_source="e2e_softmax_minus_ade_0p5m"); r.pop("compact")
            rows.append(r); F.append(lab[split][x["token"]])
        with ThreadPoolExecutor(32) as ex:
            ims = {r["images"][0]: src[os.path.basename(r["images"][0])] for r in rows}
            imgs = dict(zip(ims, zip(ims.values(), ex.map(os.path.getsize, ims.values()))))
        rep[split] = dict(write(D, split, rows, F, 0.5, imgs, C), old_rows=len(xs), dropped_public_2601=len(xs) - len(keep),
                          with_block=sum(ml.HEAD in x["question"] for x in keep), stripped_equal_new_format=len(keep) - len(bad),
                          dropout=sum(bool(r.get("dropout")) for r in rows), mismatch_tokens=bad[:20])
    for tag, d in dirs.items():
        if not os.path.islink(f"{O}/img/pai_dagger/{tag}"):
            os.symlink(d, f"{O}/img/pai_dagger/{tag}")
    json.dump(rep, open(f"{D}/report.json", "w"), indent=1); print(json.dumps(rep))
    assert all(not rep[s]["mismatch_tokens"] for s in ("train", "dev")), "a stripped prompt differs from the block-free format"


if __name__ == "__main__":
    main(*sys.argv[1:4])
