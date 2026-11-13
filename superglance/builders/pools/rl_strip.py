"""The RL pool without the road-users block: rl_rows' pool with the block cut out of every question that has one
(superglance.builders.shards.common.strip). Rows, order, images, targets and rewards stay the same (the arrays are links).
Checks: 0 questions with motion_lines.HEAD afterwards; every changed question q gives back q when its block is re-inserted into
strip(q); only PAI-like sources change. Also counts the distinct PAI-AV clips (token minus its frame suffix) of each file.
usage: python -m superglance.builders.pools.rl_strip <rl_rows pool> <out dir>"""
import json
import os
import shutil
import sys

src, out = sys.argv[1:3]
from superglance.builders.shards.common import ml, strip  # noqa: E402
os.makedirs(out)
rep = {"from": src, "strip": "superglance.builders.shards.common.strip"}
for name in ("train.jsonl", "dev.jsonl"):
    n, by, clips = 0, {}, set()
    with open(f"{src}/{name}") as f, open(f"{out}/{name}.tmp", "w") as g:
        for line in f:
            r = json.loads(line); q = r["question"]; n += 1
            if ml.HEAD in q:
                t, i = strip(q), q.find(ml.HEAD); b = q[i:len(q) - len(ml.INSTR)]
                assert ml.HEAD not in t and t[:-len(ml.INSTR)] + b + ml.INSTR == q and strip(t[:-len(ml.INSTR)] + b + ml.INSTR) == t, r["id"]
                r["question"] = t; by[r["source"]] = by.get(r["source"], 0) + 1
            if r["source"] == "pai_human":
                clips.add(r["token"].rsplit("_", 1)[0])
            g.write(json.dumps(r) + "\n")
    os.rename(f"{out}/{name}.tmp", f"{out}/{name}")
    assert set(by) <= {"pai_human", "pai_dagger"}, by
    assert sum(ml.HEAD in json.loads(x)["question"] for x in open(f"{out}/{name}")) == 0
    rep[name] = {"rows": n, "stripped": sum(by.values()), "stripped_by_source": by, "pai_human_clips": len(clips)}
for f in ("copies.tsv", "images_mode"):
    shutil.copyfile(f"{src}/{f}", f"{out}/{f}")
for f in ("train_targets.npy", "train_rewards.npy", "train_rewards.state", "nav_tokens.txt", "dev_targets.npy", "held_blocks.json"):
    p = f"{src}/{f}"; os.symlink(os.readlink(p) if os.path.islink(p) else p, f"{out}/{f}")   # the same link text as the source's
json.dump(rep, open(f"{out}/report.json", "w"), indent=1); print(json.dumps(rep))
