#!/usr/bin/env python3
"""Add GQA relation-flip minimal pairs: both objects present, relation false.

In the stage-1 pool, GQA relation-verify "no" answers are ~88% substituted or absent
objects and only ~12% "both objects present, relation false", so a model can learn
"both named objects visible -> yes". For each relVerify yes-question whose verified
relation (from the GQA program) is an exclusive spatial relation that appears exactly
once in the question, this writes the same question with the opposite relation and
answer "no" ("Is the cup to the left of the plate?" -> "... to the right of ..."). The
original question stays in the pool, so each flip forms a minimal pair with it.

Writes a new pool directory: train.jsonl = original rows + flips; dev.jsonl and the
image store are symlinked unchanged, so dev-based checkpoint selection stays comparable.
"""
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

FLIP = {"to the left of": "to the right of", "to the right of": "to the left of",
        "above": "below", "below": "above", "beneath": "above",
        "in front of": "behind", "behind": "in front of",
        "on top of": "under", "under": "on top of"}


def verified_relation(row):
    op = row["source_semantic"][-1]
    parts = op["argument"].split(",")
    return parts[1].strip() if op["operation"] == "verify rel" and len(parts) > 2 else None


def flip(row):
    if row.get("source_types", {}).get("detailed") != "relVerify" or row["correct_answer"] != "yes":
        return None
    relation = verified_relation(row)
    if relation not in FLIP:
        return None
    pattern = re.compile(rf"\b{re.escape(relation)}\b")
    if len(pattern.findall(row["question"])) != 1:
        return None
    new = dict(row)
    new["id"] = row["id"] + "_relflip"
    new["question"] = pattern.sub(FLIP[relation], row["question"])
    new["gold_index"] = row["answers"].index("no")
    new["source_gold_index"] = row["source_answers"].index("no")
    new["correct_answer"] = new["source_answer"] = new["sampling_bucket"] = "no"
    new["label_source"] = "rule_relation_flip"
    new["question_type"] = new["sampling_task"] = "gqa/verify/rel_flip"
    new["source_types"] = dict(row["source_types"], detailed="relVerifyFlip")
    new["flip_of"] = row["id"]
    new["flipped_relation"] = [relation, FLIP[relation]]
    return new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()  # never overwrite an existing pool
    counts, examples, flips = Counter(), [], []
    with (args.pool / "train.jsonl").open() as src, (args.output / "train.jsonl").open("w") as out:
        for line in src:
            out.write(line)
            new = flip(json.loads(line))
            if new:
                flips.append(new)
                counts[" -> ".join(new["flipped_relation"])] += 1
                if len(examples) < 20:
                    examples.append({"original": json.loads(line)["question"], "flipped": new["question"]})
        for new in flips:
            out.write(json.dumps(new, ensure_ascii=False) + "\n")
    for name in ("dev.jsonl", "image_store.sqlite", "images"):
        os.symlink(args.pool / name, args.output / name)
    report = {"source_pool": str(args.pool), "flips": len(flips), "by_relation": dict(counts.most_common()),
              "label_source": "rule_relation_flip (derived by rule from GQA relVerify yes-questions, not dataset GT)",
              "dev": "unchanged (symlink)", "examples": examples}
    (args.output / "flip_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("flips", "by_relation")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
