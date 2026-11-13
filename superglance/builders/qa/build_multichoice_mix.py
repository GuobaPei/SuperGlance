#!/usr/bin/env python3
"""Combine independently prepared visual multiple-choice train/dev sources.

Writes ROOT/supplement_multichoice/prepared/{train,dev}.jsonl from its gqa/, count/ and ai2d/prepared/ (the steps above).
usage: python -m superglance.builders.qa.build_multichoice_mix --root ROOT/supplement_multichoice"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path


def image_group(row):
    if row.get("image_group"):
        return row["image_group"]
    if row.get("source_coco_id"):
        return f"coco_{int(row['source_coco_id'])}"
    return row["image_id"]


def summary(rows):
    return {
        "questions": len(rows), "image_groups": len({image_group(r) for r in rows}),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_task": dict(Counter(r["question_type"] for r in rows)),
        "candidate_counts": dict(Counter(len(r["answers"]) for r in rows)),
        "gold_positions_by_candidate_count": {
            str(k): dict(Counter(r["gold_index"] for r in rows if len(r["answers"]) == k))
            for k in sorted({len(r["answers"]) for r in rows})
        },
        "max_questions_per_image_group": max(Counter(image_group(r) for r in rows).values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    output = args.root / "prepared"
    output.mkdir(parents=True, exist_ok=True)
    source_dirs = {"gqa": args.root / "gqa", "count": args.root / "count",
                   "ai2d": args.root / "ai2d/prepared"}
    merged, counts, examples = {}, {}, []
    for split in ["train", "dev"]:
        combined = []
        counts[split] = {}
        for source, source_dir in source_dirs.items():
            path = source_dir / f"{split}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            for row in rows:
                row["images"] = [str((path.parent / image).resolve()) for image in row["images"]]
                assert len(row["images"]) == 1
                assert Path(row["images"][0]).is_file()
                assert 4 <= len(row["answers"]) <= 6
                assert 0 <= row["gold_index"] < len(row["answers"])
            counts[split][source] = len(rows)
            combined.extend(rows)
            if split == "train":
                for row in random.Random(args.seed).sample(rows, min(6, len(rows))):
                    examples.append(row)
        assert len({r["id"] for r in combined}) == len(combined)
        random.Random(args.seed + (split == "dev")).shuffle(combined)
        merged[split] = combined
    assert not ({image_group(r) for r in merged["train"]} & {image_group(r) for r in merged["dev"]})
    for split, rows in merged.items():
        with (output / f"{split}.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "examples.jsonl").open("w") as stream:
        for row in examples:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "version": "multichoice-data-v1", "status": "ready", "seed": args.seed,
        "source_manifests": counts,
        "train": summary(merged["train"]), "dev": summary(merged["dev"]),
        "validation": "One existing image per row, 4/5/6 candidates, valid GT indices, unique question IDs, and disjoint train/dev image groups.",
        "scope": "Additional training pool using original source training partitions only. It does not change SAT50k objective controls or fixed external benchmarks.",
        "binary_questions": "Original yes/no tasks remain semantically binary; they were not padded with artificial choices. This new pool adds multiclass questions.",
        "provenance": "GQA/Tally retain VG/COCO identity and prior split assignments. AI2D packaged image row IDs group all QAs from each image; original AllenAI image IDs are unavailable in the source package.",
        "per_source_reports": [str(p / "preparation_report.json") for p in source_dirs.values()],
    }
    (output / "preparation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
