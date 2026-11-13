"""Checks of a staged stage-3 pool before training, written to <pool>/check.json: rows per source and split, the mixture sampler's
effective weight per source (from the config), rows whose prompt holds the road-users block (must be 0) and rows whose image is
missing under the pool dir (must be 0). Exit 1 if either is not 0.
usage: python -m superglance.builders.pools.check <pool dir> <config.json>"""
import json, os, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from superglance.data.mixture import MixtureSampler

P, cfg = sys.argv[1], json.load(open(sys.argv[2]))
out = {}
for split in ("train", "dev"):
    R = [json.loads(l) for l in open(f"{P}/{split}.jsonl")]
    with ThreadPoolExecutor(32) as ex:
        missing = sum(not x for x in ex.map(lambda r: os.path.exists(f"{P}/{r['images'][0]}"), R, chunksize=4096))
    out[split] = {"rows": len(R), "by_source": dict(Counter(r["source"] for r in R).most_common()),
                  "road_users_ahead": sum("Road users ahead" in r.get("question", "") for r in R), "missing_images": missing}
    if split == "train":
        s = MixtureSampler(R, cfg["mixture_sampling"], cfg["seed"]); w = Counter()
        for r, x in zip(R, s.weights.tolist()):
            w[r["source"]] += x
        out["effective_weights"] = {k: round(v, 6) for k, v in w.most_common()}
        out["max_expected_exposures_per_row"] = round(s.report["maximum_expected_exposures_per_qa"], 2)
json.dump(out, open(f"{P}/check.json", "w"), indent=1)
print(json.dumps(out))
sys.exit(1 if any(out[s]["road_users_ahead"] or out[s]["missing_images"] for s in ("train", "dev")) else 0)
