#!/usr/bin/env python3
"""Convert original TallyQA train into nearby-integer multiple-choice questions.

Writes ROOT/supplement_multichoice/count/{train,dev}.jsonl from ROOT/supplement_gqa/raw/tallyqa.zip, on the image split of
prepare_gqa_multichoice.py. usage: python -m superglance.builders.qa.prepare_count_multichoice --root ROOT"""
import argparse
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def select_counts(rows, target, seed, dominant_cap=0.30):
    rng = random.Random(seed)
    pools = defaultdict(list)
    for row in rows:
        pools[int(row["answer"])].append(row)
    for pool in pools.values():
        rng.shuffle(pool)
    selected, per_image, per_answer = [], Counter(), Counter()
    while any(pools.values()) and len(selected) < target:
        labels = sorted(pools)
        rng.shuffle(labels)
        for label in labels:
            if label in (1, 2) and per_answer[label] >= int(target * dominant_cap):
                pools[label].clear()
                continue
            while pools[label]:
                row = pools[label].pop()
                if per_image[row["imageId"]] >= 2:
                    continue
                selected.append(row)
                per_image[row["imageId"]] += 1
                per_answer[label] += 1
                break
            if len(selected) == target:
                break
    # If rare counts run out early, reduce dominant labels against the actual total.
    buckets = defaultdict(list)
    for row in selected:
        buckets[int(row["answer"])].append(row)
    changed = True
    while changed:
        changed = False
        for label in (1, 2):
            others = sum(len(v) for k, v in buckets.items() if k != label)
            allowed = int(dominant_cap * others / (1 - dominant_cap))
            if len(buckets[label]) > allowed:
                buckets[label] = buckets[label][:allowed]
                changed = True
    selected = [row for values in buckets.values() for row in values]
    rng.shuffle(selected)
    return selected


def convert(rows, mapping, output, seed):
    rng = random.Random(seed)
    sizes = ([4, 5, 6] * ((len(rows) + 2) // 3))[:len(rows)]
    rng.shuffle(sizes)
    result = []
    for row, k in zip(rows, sizes):
        answer = int(row["answer"])
        if answer < k:
            start = 0
        else:
            start = rng.randint(max(0, answer - k + 1), min(answer, 16 - k))
        canonical = [str(n) for n in range(start, start + k)]
        permutation = list(range(k))
        rng.shuffle(permutation)
        candidates = [canonical[i] for i in permutation]
        identifier = row["imageId"]
        info = mapping[identifier]
        result.append({
            "id": f"tallyqa_mc_train_{row['question_id']}",
            "image_id": f"vg_{identifier}",
            "images": [str(output / "images" / f"vg_{identifier}.jpg")],
            "question": row["question"], "answers": candidates,
            "gold_index": candidates.index(str(answer)), "pattern": "Choice",
            "question_type": "count", "source": "TallyQA",
            "source_split": "train", "source_id": row["question_id"],
            "source_image_id": identifier, "source_coco_id": info.get("coco_id"),
            "image_group": info["group"], "source_answer": answer,
            "source_data_source": row["data_source"],
            "source_issimple": row.get("issimple"),
            "source_answers": canonical, "candidate_permutation": permutation,
            "candidate_generation": "Contiguous integer window: [0,K-1] for GT<K, otherwise a uniformly selected legal window in 0..15; K assigned independently before conversion.",
        })
    return result


def summarize(rows):
    counts = Counter(len(row["answers"]) for row in rows)
    return {
        "questions": len(rows), "images": len({row["image_id"] for row in rows}),
        "candidate_counts": dict(counts),
        "answer_counts": dict(Counter(row["source_answer"] for row in rows)),
        "gold_positions_by_candidate_count": {
            str(k): dict(Counter(row["gold_index"] for row in rows if len(row["answers"]) == k))
            for k in sorted(counts)
        },
        "max_questions_per_image": max(Counter(row["image_id"] for row in rows).values(), default=0),
    }


def candidate_only_baselines(train, dev):
    global_counts = Counter(row["source_answer"] for row in train)
    conditional = defaultdict(Counter)
    for row in train:
        key = tuple(sorted(map(int, row["answers"])))
        conditional[key][row["source_answer"]] += 1

    def evaluate(rows):
        correct, seen_windows = Counter(), 0
        for row in rows:
            options = list(map(int, row["answers"]))
            key = tuple(sorted(options))
            distribution = conditional.get(key, global_counts)
            seen_windows += key in conditional
            predictions = {
                "minimum": min(options),
                "training_answer_frequency": max(options, key=lambda x: global_counts[x]),
                "training_candidate_set_frequency": max(options, key=lambda x: distribution[x]),
            }
            for name, prediction in predictions.items():
                correct[name] += prediction == row["source_answer"]
        return {
            "n": len(rows), "candidate_sets_seen_in_train": seen_windows,
            "accuracy": {name: value / len(rows) for name, value in correct.items()},
        }
    return {"fit": "All frequency estimates use only this selected training manifest; ties use first presented option.",
            "train": evaluate(train), "dev": evaluate(dev)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--dev-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    raw = args.root / "supplement_gqa/raw"
    output = args.root / "supplement_multichoice/count"
    (output / "images").mkdir(parents=True, exist_ok=True)
    split_file = args.root / "supplement_multichoice/image_split.json"
    mapping = json.loads(split_file.read_text())["images"]
    with zipfile.ZipFile(raw / "tallyqa.zip") as archive:
        source = json.loads(archive.read("train.json"))
    pools, seen = defaultdict(list), set()
    for row in source:
        if not row["image"].startswith(("VG_100K/", "VG_100K_2/")):
            continue
        identifier = Path(row["image"]).stem
        key = identifier, row["question"].strip().casefold()
        if identifier not in mapping or key in seen:
            continue
        seen.add(key)
        row["imageId"] = identifier
        pools[mapping[identifier]["split"]].append(row)
    selected = {name: select_counts(pools[name], size, args.seed + offset)
                for name, size, offset in [("train", args.train_size, 101), ("dev", args.dev_size, 102)]}
    records = {name: convert(rows, mapping, output, args.seed + offset)
               for (name, rows), offset in zip(selected.items(), [201, 202])}
    needed = {row["source_image_id"] for rows in records.values() for row in rows}
    exported = set()
    for shard in sorted((raw / "train_balanced_images").glob("train-*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                identifier = str(row["id"])
                if identifier not in needed or identifier in exported:
                    continue
                with Image.open(io.BytesIO(row["image"]["bytes"])) as image:
                    image.convert("RGB").save(output / "images" / f"vg_{identifier}.jpg", quality=95)
                exported.add(identifier)
    assert exported == needed
    for name, rows in records.items():
        with (output / f"{name}.jsonl").open("w") as stream:
            for row in rows:
                assert str(row["source_answer"]) == row["answers"][row["gold_index"]]
                assert len(set(row["answers"])) == len(row["answers"])
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    assert not ({r["image_group"] for r in records["train"]} & {r["image_group"] for r in records["dev"]})
    report = {
        "version": "tallyqa-nearby-multichoice-v1", "status": "ready",
        "source": "TallyQA (manoja328/tallyqa)", "source_split": "train only",
        "image_split": str(split_file), "seed": args.seed,
        "available": {name: {"questions": len(rows),
                              "images": len({r["imageId"] for r in rows}),
                              "answer_counts": dict(Counter(r["answer"] for r in rows))}
                      for name, rows in pools.items()},
        "sampling": "Round-robin by actual GT, no duplicate questions, at most two count questions per image, counts 1 and 2 each at most 30% of the selected split. Scarce answers are not repeated to fill a quota.",
        "requested": {"train": args.train_size, "dev": args.dev_size},
        "train": summarize(records["train"]), "dev": summarize(records["dev"]),
        "exported_images": len(exported),
        "candidate_only_baselines": candidate_only_baselines(records["train"], records["dev"]),
        "comparison_note": "This is a changed question pool and candidate construction, so its dev accuracy is not comparable to original open-ended TallyQA or the old 2k count pool as a model improvement.",
    }
    (output / "preparation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["train", "dev", "candidate_only_baselines"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
