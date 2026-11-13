#!/usr/bin/env python3
"""Prepare an image-grouped SAT-v2 pilot from already downloaded parquet files."""
import argparse
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def image_bytes(image, source_root, shard):
    if image.get("bytes") is not None:
        return image["bytes"]
    path = Path(image["path"])
    if path.is_absolute():
        return path.read_bytes()
    for parent in (source_root, shard.parent):
        if (parent / path).is_file():
            return (parent / path).read_bytes()
    raise FileNotFoundError(image["path"])


def parquet_files(source_root, split):
    return sorted(source_root.rglob(f"{split}-*.parquet"))


def convert_split(source_root, split, image_ids, image_payloads):
    rows, excluded = [], Counter()
    paths = parquet_files(source_root, split)
    for shard in paths:
        row_in_shard = 0
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=256):
            for raw in batch.to_pylist():
                source_index = row_in_shard
                row_in_shard += 1
                if len(raw["images"]) != 1:
                    excluded["not_single_image"] += 1
                    continue
                options = raw["answers"]
                if len(options) < 2 or len(options) > 26 or raw["correct_answer"] not in options:
                    excluded["unusable_candidates"] += 1
                    continue
                if len(set(options)) != len(options):
                    excluded["repeated_candidate_text"] += 1
                    continue
                payload = image_bytes(raw["images"][0], source_root, shard)
                if payload not in image_ids:
                    image_ids[payload] = f"image_{len(image_ids):06d}"
                    image_payloads[image_ids[payload]] = payload
                image_id = image_ids[payload]
                native_binary = {s.strip().lower() for s in options} == {"yes", "no"}
                row = {k: v for k, v in raw.items() if k != "images"}
                row.update({
                    "id": f"sat_v2/{split}/{shard.stem}/{source_index}",
                    "image_id": image_id,
                    "source": "array/SAT-v2", "source_split": split,
                    "source_file": str(shard.relative_to(source_root)),
                    "source_row": source_index,
                    "source_image_paths": [im.get("path") for im in raw["images"]],
                    "pattern": "Noul" if native_binary else "Choice",
                    "gold_index": options.index(raw["correct_answer"]),
                    "label_source": "dataset_gt", "conversion": "native_candidates",
                })
                if native_binary:
                    row["yes_index"] = next(i for i, s in enumerate(options) if s.strip().lower() == "yes")
                rows.append(row)
        print(f"Read {split}: {shard.name}", flush=True)
    return rows, {"files": len(paths), "included": len(rows), "excluded": dict(excluded)}


def grouped_subsets(rows, train_size, dev_size, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["image_id"]].append(row)
    rng = random.Random(seed)
    group_ids = list(groups)
    rng.shuffle(group_ids)
    dev, train, reserved = [], [], set()
    for group_id in group_ids:
        group_rows = list(groups[group_id])
        rng.shuffle(group_rows)
        if len(dev) < dev_size:
            dev.extend(group_rows[:dev_size - len(dev)])
            reserved.add(group_id)
        elif len(train) < train_size:
            train.extend(group_rows[:train_size - len(train)])
    if len(train) != train_size or len(dev) != dev_size:
        raise ValueError(f"Available grouped data yielded train={len(train)}, dev={len(dev)}; reduce requested sizes")
    rng.shuffle(train)
    rng.shuffle(dev)
    return train, dev, len(reserved)


def shuffle_candidate_order(row, rng):
    """Randomize answer position while keeping the original GT and order traceable."""
    result = dict(row)
    source_answers = list(row["answers"])
    permutation = list(range(len(source_answers)))
    rng.shuffle(permutation)
    result["source_answers"] = source_answers
    result["source_gold_index"] = row["gold_index"]
    result["candidate_permutation"] = permutation
    result["answers"] = [source_answers[index] for index in permutation]
    result["gold_index"] = permutation.index(row["gold_index"])
    if "yes_index" in row:
        result["source_yes_index"] = row["yes_index"]
        result["yes_index"] = permutation.index(row["yes_index"])
    result["conversion"] = "native_candidates_shuffled"
    return result


def describe(rows):
    gold_positions = defaultdict(Counter)
    for row in rows:
        gold_positions[len(row["answers"])][row["gold_index"]] += 1
    return {
        "qa": len(rows), "images": len({r["image_id"] for r in rows}),
        "patterns": dict(Counter(r["pattern"] for r in rows)),
        "question_types": dict(Counter(r["question_type"] for r in rows)),
        "candidate_counts": dict(Counter(len(r["answers"]) for r in rows)),
        "gold_positions_by_candidate_count": {
            count: {position: positions[position] for position in range(count)}
            for count, positions in sorted(gold_positions.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Downloaded HF repository or parquet directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--dev-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    args.source = args.source.resolve()
    image_ids, image_payloads = {}, {}
    val, val_log = convert_split(args.source, "val", image_ids, image_payloads)
    test, test_log = convert_split(args.source, "test", image_ids, image_payloads)
    rows, static_log = convert_split(args.source, "static", image_ids, image_payloads)
    heldout_images = {r["image_id"] for r in val + test}
    rows_for_train = [r for r in rows if r["image_id"] not in heldout_images]
    train, dev, dev_groups = grouped_subsets(rows_for_train, args.train_size, args.dev_size, args.seed)
    outputs = {"train": train, "dev": dev}
    if val:
        outputs["sat_val_single_image"] = val
    if test:
        outputs["sat_test_single_image"] = test
    shuffled_outputs, candidate_order_seeds = {}, {}
    for split_index, (name, subset) in enumerate(outputs.items()):
        candidate_order_seeds[name] = args.seed + 1000 + split_index
        rng = random.Random(candidate_order_seeds[name])
        shuffled_outputs[name] = [shuffle_candidate_order(row, rng) for row in subset]
    outputs = shuffled_outputs
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "images").mkdir(exist_ok=True)
    used_ids = sorted({r["image_id"] for subset in outputs.values() for r in subset})
    image_paths = {}
    for image_id in used_ids:
        payload = image_payloads[image_id]
        image_format = Image.open(io.BytesIO(payload)).format.lower()
        suffix = "jpg" if image_format == "jpeg" else image_format
        relative_path = f"images/{image_id}.{suffix}"
        (args.output / relative_path).write_bytes(payload)
        image_paths[image_id] = relative_path
    for name, subset in outputs.items():
        with (args.output / f"{name}.jsonl").open("w", encoding="utf-8") as out:
            for row in subset:
                row["images"] = [image_paths[row["image_id"]]]
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "dataset": "array/SAT-v2", "source": str(args.source), "seed": args.seed,
        "grouping": "exact encoded image bytes; source has no scene ID",
        "candidate_order": "fixed-seed shuffle per split; candidate_permutation[new_index] is the source index",
        "candidate_order_seeds": candidate_order_seeds,
        "requested_train": args.train_size, "requested_dev": args.dev_size,
        "source_static": static_log, "source_val": val_log, "source_test": test_log,
        "static_rows_excluded_for_val_or_test_image_overlap": len(rows) - len(rows_for_train),
        "dev_reserved_image_groups": dev_groups,
        "outputs": {name: describe(subset) for name, subset in outputs.items()},
        "evaluation_protocol": "single-image val/test subsets are adapted slices, not full official SAT-v2 splits",
        "score_examples": 0,
    }
    (args.output / "preparation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
