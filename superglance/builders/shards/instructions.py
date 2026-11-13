"""Instruction-row shard (the Language2Drive-style rows of superglance.builders.instructions) for merge / dev_draw: rows as they
come (train.jsonl, dev.jsonl; images staged at <L2D_IMG_ROOT>/<rel>, rel = manifest.tsv's img/<nav|pai|wod>/<file>), the image
path rewritten to l2d/<rel>. The instruction data's dev becomes the shard's val split, so dev_draw draws the 200 dev rows from it.
Every row must be compact with k 5121, carry no road-users block and point at a manifest image. Targets are symlinked.
usage: L2D_IMG_ROOT=<staged image root> python -m superglance.builders.shards.instructions <instruction data dir> <out shard dir>"""
import json, os, sys

D, OUT = sys.argv[1:3]; P = os.environ["L2D_IMG_ROOT"].rstrip("/") + "/"
rel = {l.split("\t")[1] for l in open(f"{D}/manifest.tsv")}
os.makedirs(OUT)   # never overwrite
rep = {}
for a, b in (("train", "train"), ("dev", "val")):
    n = 0
    with open(f"{OUT}/{b}.jsonl", "w") as f:
        for l in open(f"{D}/{a}.jsonl"):
            r = json.loads(l); p = r["images"][0]
            assert r["compact"]["k"] == 5121 and "Road users ahead" not in r["question"] and len(r["images"]) == 1, r["id"]
            assert p.startswith(P) and p[len(P):] in rel, (r["id"], p)
            r["images"] = ["l2d/" + p[len(P):]]; f.write(json.dumps(r) + "\n"); n += 1
    os.symlink(f"{D}/{a}_targets.npy", f"{OUT}/{b}_targets.npy"); rep[b] = n
json.dump(rep, open(f"{OUT}/report.json", "w")); print(rep)
