#!/usr/bin/env python3
"""Report CV-Bench source weighting alongside micro and BLINK task results."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def cv_source_metrics(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["source"]].append(record)
    result = {source: {"n": len(rows), "accuracy": sum(
        row["gold_index"] == row["predicted_index"] for row in rows) / len(rows)}
        for source, rows in groups.items()}
    third, = (source for source in result if source not in ("ADE20K", "COCO"))   # the 3D tasks' source
    weighted = 0.5 * ((result["ADE20K"]["accuracy"] + result["COCO"]["accuracy"]) / 2 + result[third]["accuracy"])
    return result, weighted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory containing cvbench/ and blink_single_image/ evaluation results")
    args = parser.parse_args()
    cv = json.loads((args.output / "cvbench" / "metrics.json").read_text())
    blink = json.loads((args.output / "blink_single_image" / "metrics.json").read_text())
    with (args.output / "cvbench" / "predictions.jsonl").open() as stream:
        cv_records = [json.loads(line) for line in stream if line.strip()]
    by_source, weighted = cv_source_metrics(cv_records)
    result = {
        "model": cv["model"], "adapter": cv["adapter"],
        "readout": cv["readout"], "max_pixels": cv["max_pixels"],
        "cvbench": {"official_source_weighted_accuracy": weighted,
                    "formula": "0.5 * ((ADE20K_acc + COCO_acc) / 2 + 3D-source_acc)",
                    "micro": cv["all"], "by_source": by_source,
                    "by_task": cv["by_question_type"]},
        "blink_single_image": {"scope": "Four val tasks, 509 questions; not full BLINK",
                               "micro": blink["all"], "by_task": blink["by_question_type"]},
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
