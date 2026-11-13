"""The protected evaluation keys for clean.py: PAI-AV clips by split (clip_index.parquet), the gold evaluation list (PAI-AV 700 = its
test clips, PAI val = its val clips), the AlpaSim 916 scenes and the earlier closed-loop val suite (nurec_curated_val), navtest
tokens and logs, and the clips of the PAI evaluation rows themselves (the 644 list holds 325 train-split clips, so those leave
training too).
usage: python -m superglance.builders.pools.eval_keys <out.json> <clip_index.parquet> <gold_eval_644.parquet> <alpasim_916.txt>
       <sim_suites_curated.csv> <navtest_inputs.jsonl> <PAI open-loop root: pai_openloop_{test,val,644}/>"""
import csv, json, sys

import pandas as pd

CI, GOLD, S916, SUITES, NAV, U = sys.argv[2:8]
split = pd.read_parquet(CI)["split"].astype(str).to_dict()
gold = pd.read_parquet(GOLD)["key"].astype(str).tolist()
clip = lambda s: s.split("clipgt-", 1)[1]
s916 = [clip(l.strip()) for l in open(S916) if l.strip()]
suite = {}
for r in csv.DictReader(open(SUITES)):
    suite.setdefault(r["test_suite_id"], set()).add(r["scene_id"])
nav = [json.loads(l) for l in open(NAV)]
ev = {n: sorted({json.loads(l)["log_name"] for l in open(f"{U}/{f}")}) for n, f in   # the clips of the eval rows themselves
      {"pai700": "pai_openloop_test/pai_openloop_test.jsonl", "paival": "pai_openloop_val/pai_openloop_val.jsonl",
       "paival_train": "pai_openloop_val/pai_openloop_train.jsonl", "pai644": "pai_openloop_644/pai_openloop_644.jsonl"}.items()}
k = {"pai_test": sorted(c for c, s in split.items() if s == "test"), "pai_val": sorted(c for c, s in split.items() if s == "val"),
     "pai_split_counts": pd.Series(split).value_counts().to_dict(),
     "gold_test": sorted(c for c in gold if split.get(c) == "test"), "gold_val": sorted(c for c in gold if split.get(c) == "val"),
     "gold_other": sorted(c for c in gold if split.get(c) not in ("test", "val")),
     "alpasim_916": sorted(s916), "curated_val": sorted(map(clip, suite["nurec_curated_val"])),
     "pai_eval": ev, "navtest_tokens": sorted({r["token"] for r in nav}), "navtest_logs": sorted({r["log_name"] for r in nav})}
assert [len(ev[n]) for n in ("pai700", "paival", "paival_train", "pai644")] == [140, 399, 325, 644], {n: len(v) for n, v in ev.items()}
assert len(s916) == len(set(s916)) == 916 and len(k["curated_val"]) == 441 and len(k["navtest_tokens"]) == 12146
json.dump(k, open(sys.argv[1], "w"))
print({x: len(v) if isinstance(v, list) else v for x, v in k.items() if x != "pai_eval"}, {n: len(v) for n, v in ev.items()},
      "pai644 = gold:", set(ev["pai644"]) == set(gold), "pai700 = gold test:", set(ev["pai700"]) == set(k["gold_test"]))
