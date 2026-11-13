#!/usr/bin/env python3
"""Aggregate official 3DSRBench circular groups and additional flip diagnostics."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def grouped_accuracy(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    hits = [all(row["correct"] for row in group) for group in groups.values()]
    return {"n_groups": len(hits), "accuracy": sum(hits) / len(hits)}


def summarize(manifest, predictions):
    source = {row["id"]: row for row in manifest}
    assert len(source) == len(manifest)
    assert len({row["id"] for row in predictions}) == len(predictions)
    assert set(source) == {row["id"] for row in predictions}
    rows = []
    for prediction in predictions:
        row = dict(source[prediction["id"]])
        assert prediction["gold_index"] == row["gold_index"]
        assert len(prediction["probabilities"]) == len(row["answers"])
        row["correct"] = prediction["predicted_index"] == row["gold_index"]
        rows.append(row)

    # The official helper removes a trailing numeric rotation, keeping '-flip'.
    official_key = lambda row: row["source_qid"][:-2] if row["source_qid"][-2] == "-" else row["source_qid"]
    by_subtype, by_category = defaultdict(list), defaultdict(list)
    for row in rows:
        by_subtype[row["question_type"]].append(row)
        by_category[row["task"]].append(row)
    official = grouped_accuracy(rows, official_key)
    official["protocol"] = "All option rotations correct; original/flip are separate groups; matches official helper"
    official["by_category"] = {key: grouped_accuracy(group, official_key) for key, group in by_category.items()}
    official["by_subtype"] = {key: grouped_accuracy(group, official_key) for key, group in by_subtype.items()}
    flip_coverage = defaultdict(set)
    for row in rows:
        flip_coverage[row["base_question_id"]].add(row["flipped_image"])
    strict = grouped_accuracy(rows, lambda row: row["base_question_id"])
    strict["protocol"] = "All provided original/flip/rotation variants correct; includes questions with one image orientation"
    strict["n_both_image_orientations"] = sum(len(flags) == 2 for flags in flip_coverage.values())
    return {
        "benchmark": "3DSRBench-real", "n_scored_rows": len(rows),
        "official_circular": official,
        "strict_flip_circular": strict,
        "flip_pairs_per_rotation": grouped_accuracy(rows, lambda row: row["source_qid"].replace("-flip", "")),
        "row_micro_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "readout_protocol": "Native candidate-token logits; official grouping without generated-answer parsing",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines() if line]
    result = summarize(manifest, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
