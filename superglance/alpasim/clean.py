"""The cleaned summary of an AlpaSim run. Drops only the rows whose failure_reason is a full disk ("Errno 28", ENOSPC) and whose
rollout_id also has a scored row; a scene left with ENOSPC rows only is listed (rerun it before any comparison). Writes
aggregate/results-summary.clean.json next to the raw summary (nothing is removed) and prints raw vs cleaned, per rollout and per
scene (rows of one scene averaged). Without ENOSPC rows the cleaned summary equals the raw one.
usage: python -m superglance.alpasim.clean <run dir> [<run dir> ...]"""
import collections, json, sys


def stats(rows):
    by = collections.defaultdict(list)
    for x in rows:
        by[x["clipgt_id"]].append(x)
    sc = [sum(x["score"] for x in v) / len(v) for v in by.values()]
    ps = [sum(bool(x["passed"]) for x in v) / len(v) for v in by.values()]
    return (f"{len(rows)} rows / {len(by)} scenes; per row pass {sum(bool(x['passed']) for x in rows) / len(rows):.1%}"
            f" score {sum(x['score'] for x in rows) / len(rows):.4f}; per scene pass {sum(ps) / len(ps):.1%} score {sum(sc) / len(sc):.4f}")


for run in sys.argv[1:]:
    p = f"{run}/aggregate/results-summary.json"; d = json.load(open(p)); raw = d["rollouts"]
    nos = lambda x: "Errno 28" in str(x.get("failure_reason"))
    scored = {x.get("rollout_id") for x in raw if not nos(x)}
    drop = [x for x in raw if nos(x) and x.get("rollout_id") in scored]
    d["rollouts"] = [x for x in raw if x not in drop]
    only = sorted({x["clipgt_id"] for x in raw if nos(x)} - {x["clipgt_id"] for x in d["rollouts"] if not nos(x)})
    json.dump(d, open(p[:-5] + ".clean.json", "w"))
    print(f"{run.rsplit('/', 1)[-1]}: raw {stats(raw)}\n  cleaned {stats(d['rollouts'])}\n  dropped {len(drop)}:"
          f" {' '.join(x['clipgt_id'][7:15] for x in drop) or '-'}; ENOSPC-only scenes (rerun first): {' '.join(c[7:15] for c in only) or 'none'}")
