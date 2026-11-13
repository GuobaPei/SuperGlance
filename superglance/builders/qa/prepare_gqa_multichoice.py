#!/usr/bin/env python3
"""Build GQA query multiple choice and a shared, backwards-compatible image split.

Writes ROOT/supplement_multichoice/image_split.json (every GQA image's train / dev group; adapt_stage1.py keeps it) and
ROOT/supplement_multichoice/gqa/{train,dev}.jsonl (12,000 + 1,000 typed GQA queries as multiple choice). Reads
ROOT/supplement_gqa (prepare_supplement.py, GQA image shards 0-3: download_more_gqa.py) and the CV-Bench / 3DSRBench
manifests (prepare_external_eval.py, prepare_3dsr_eval.py).
usage: python -m superglance.builders.qa.prepare_gqa_multichoice --root ROOT"""
import argparse
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from superglance.builders.qa.prepare_supplement import excluded_coco_ids


SEED = 20260918
DOMAINS = {
    "color": ["black", "white", "gray", "red", "green", "blue", "brown", "yellow", "orange", "pink", "purple"],
    "material": ["wood", "metal", "plastic", "glass", "brick", "concrete", "leather", "porcelain"],
    "name/furniture": ["chair", "table", "desk", "bed", "sofa", "cabinet", "shelf", "bench", "stool"],
    "name/animal": ["dog", "cat", "horse", "cow", "sheep", "elephant", "giraffe", "zebra", "bear", "goat", "duck"],
    "name/vehicle": ["car", "bus", "truck", "train", "van", "carriage", "tractor", "trailer"],
    "name/device": ["laptop", "keyboard", "cell phone", "television", "monitor", "remote control", "camera", "computer mouse", "speaker", "printer"],
    "name/vegetable": ["broccoli", "tomato", "pepper", "lettuce", "onion", "carrot", "potato", "olive", "spinach", "basil", "cucumber", "asparagus", "cabbage", "corn", "pumpkin"],
    "name/fruit": ["banana", "apple", "orange", "watermelon", "pineapple", "strawberry", "grape", "blueberry", "pear", "grapefruit", "pomegranate", "kiwi", "mango", "cherry"],
    "name/food": ["sandwich", "egg", "salad", "rice", "cookie", "toast", "soup", "cereal", "pasta", "burrito", "yogurt", "pancake"],
    "name/appliance": ["refrigerator", "stove", "oven", "microwave", "dishwasher", "blender", "coffee maker", "toaster", "washing machine"],
}
# Avoid simultaneous alternatives whose names can overlap in ordinary usage.
RELATED_FAMILIES = [frozenset(["table", "desk"]), frozenset(["television", "monitor"]),
                    frozenset(["stove", "oven", "microwave"])]
TASK_WEIGHTS = {"color": 25, "material": 10, "name/furniture": 15,
                "name/animal": 12, "name/vehicle": 8, "name/device": 10,
                "name/vegetable": 8, "name/fruit": 4, "name/food": 4, "name/appliance": 4}


def group_name(identifier, metadata):
    coco = metadata[identifier]["coco_id"]
    return f"coco_{coco}" if coco else f"vg_{identifier}"


def make_image_split(root, output, shards):
    raw = root / "supplement_gqa/raw"
    with zipfile.ZipFile(raw / "image_data.json.zip") as archive:
        metadata = {str(row["image_id"]): row for row in json.loads(archive.read("image_data.json"))}
    exclusions, exclusion_counts = excluded_coco_ids([
        root / "eval_external/prepared/cvbench.jsonl",
        root / "eval_3dsr/prepared/3dsr_real_circular.jsonl",
    ])
    image_ids = {}
    for index in shards:
        path = raw / "train_balanced_images" / f"train-{index:05d}-of-00021.parquet"
        image_ids[index] = set(str(x) for x in pq.read_table(path, columns=["id"])["id"].to_pylist())
    all_ids = set().union(*image_ids.values())
    missing = all_ids - metadata.keys()
    overlap = {i for i in all_ids & metadata.keys() if metadata[i]["coco_id"] in exclusions}
    available = all_ids - missing - overlap

    original_ids = (image_ids[0] | image_ids[1]) & available
    original_groups = sorted({group_name(i, metadata) for i in original_ids})
    random.Random(SEED).shuffle(original_groups)
    original_dev = set(original_groups[:len(original_groups) // 10])
    group_split = {g: "dev" if g in original_dev else "train" for g in original_groups}
    new_groups = sorted({group_name(i, metadata) for i in available} - group_split.keys())
    random.Random(SEED + 20).shuffle(new_groups)
    new_dev = set(new_groups[:len(new_groups) // 10])
    group_split.update({g: "dev" if g in new_dev else "train" for g in new_groups})
    mapping = {i: {"group": group_name(i, metadata),
                   "split": group_split[group_name(i, metadata)],
                   "coco_id": metadata[i]["coco_id"]}
               for i in sorted(available, key=int)}
    old_counts = Counter()
    for split, filename in [("train", "supplement_train10k.jsonl"), ("dev", "supplement_dev1k.jsonl")]:
        with (root / "supplement_gqa/prepared_mixed" / filename).open() as stream:
            for line in stream:
                row = json.loads(line)
                identifier = row["source_image_id"]
                assert mapping[identifier]["split"] == split, (identifier, split)
                old_counts[split] += 1
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "seed_original": SEED, "seed_new_groups": SEED + 20,
        "image_shards": shards, "downloaded_images": len(all_ids),
        "original_available_images": len(original_ids), "available_images": len(mapping),
        "by_split": dict(Counter(row["split"] for row in mapping.values())),
        "original_groups": len(original_groups), "new_groups": len(new_groups),
        "removed_known_external_overlap_images": len(overlap),
        "excluded_missing_vg_metadata_images": len(missing),
        "known_coco_ids_by_manifest": exclusion_counts,
        "old_prepared_rows_verified": dict(old_counts),
        "policy": "Reproduce original shard 0/1 group split exactly; new images sharing old COCO groups inherit that split, and new groups independently reserve 10% dev. Only official GQA train images and questions.",
    }
    pending = output / "image_split.tmp"
    pending.write_text(json.dumps({"images": mapping, "report": report}, indent=2) + "\n")
    pending.replace(output / "image_split.json")
    (output / "image_split_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return mapping, metadata


def task_for(row):
    if row["types"]["structural"] != "query":
        return None
    last = row["semantic"][-1]
    if last["operation"] != "query":
        return None
    argument = last["argument"]
    if argument in ("color", "material"):
        task = argument
        if row["groups"]["global"] != argument:
            return None
    elif argument == "name":
        task = "name/" + (row["groups"]["global"] or "untyped")
    else:
        return None
    return task if row["answer"] in DOMAINS.get(task, []) else None


def sample_queries(rows, n, seed, max_per_image=3):
    rng = random.Random(seed)
    pools = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pools[row["task"]][row["answer"]].append(row)
    for answers in pools.values():
        for pool in answers.values():
            rng.shuffle(pool)
    selected, image_counts = [], Counter()

    def take(task, quota):
        added = 0
        while any(pools[task].values()) and added < quota:
            labels = sorted(pools[task])
            rng.shuffle(labels)
            for label in labels:
                while pools[task][label]:
                    row = pools[task][label].pop()
                    if image_counts[row["imageId"]] < max_per_image:
                        selected.append(row)
                        image_counts[row["imageId"]] += 1
                        added += 1
                        break
                if added == quota:
                    break
        return added

    for task in sorted(TASK_WEIGHTS, key=lambda t: sum(map(len, pools[t].values()))):
        take(task, n * TASK_WEIGHTS[task] // 100)
    while len(selected) < n:
        previous = len(selected)
        for task in sorted(TASK_WEIGHTS):
            take(task, min(1, n - len(selected)))
        if len(selected) == previous:
            break
    if len(selected) < n:
        raise ValueError(f"Available distinct typed queries support {len(selected)} rows, requested {n}")
    rng.shuffle(selected)
    return selected


def choose_candidates(row, count, rng):
    correct = row["answer"]
    candidates = [correct]
    alternatives = list(DOMAINS[row["task"]])
    rng.shuffle(alternatives)
    for answer in alternatives:
        if answer in candidates:
            continue
        if any(answer in family and family.intersection(candidates) for family in RELATED_FAMILIES):
            continue
        candidates.append(answer)
        if len(candidates) == count:
            break
    assert len(candidates) == count
    canonical = sorted(candidates)
    permutation = list(range(count))
    rng.shuffle(permutation)
    return canonical, permutation


def materialize_images(root, output, selected_ids, shards):
    output.mkdir(parents=True, exist_ok=True)
    paths, sizes = {}, {}
    for identifier in selected_ids:
        existing = root / "supplement_gqa/prepared_mixed/images" / f"vg_{identifier}.jpg"
        if existing.is_file():
            with Image.open(existing) as image:
                image.load()
                sizes[identifier] = list(image.size)
            paths[identifier] = str(existing)
    missing = selected_ids - paths.keys()
    for shard in shards:
        if not missing:
            break
        path = root / "supplement_gqa/raw/train_balanced_images" / f"train-{shard:05d}-of-00021.parquet"
        table = pq.read_table(path, filters=[("id", "in", sorted(missing))])
        for row in table.to_pylist():
            identifier = str(row["id"])
            payload = row["image"]["bytes"]
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                extension = {"JPEG": ".jpg", "PNG": ".png"}[image.format]
                sizes[identifier] = list(image.size)
            target = output / f"vg_{identifier}{extension}"
            target.write_bytes(payload)
            paths[identifier] = str(target)
        missing = selected_ids - paths.keys()
    assert not missing
    return paths, sizes


def convert(rows, split, mapping, metadata, image_paths, image_sizes, seed):
    rng = random.Random(seed)
    counts = [4 + i % 3 for i in range(len(rows))]
    rng.shuffle(counts)
    result = []
    for row, count in zip(rows, counts):
        canonical, permutation = choose_candidates(row, count, rng)
        answers = [canonical[i] for i in permutation]
        identifier = row["imageId"]
        result.append({
            "id": f"gqa_query_{row['id']}", "image_id": f"vg_{identifier}",
            "images": [image_paths[identifier]], "image_size": image_sizes[identifier],
            "question": row["question"], "answers": answers,
            "gold_index": answers.index(row["answer"]), "pattern": "Choice",
            "question_type": row["task"], "source": "GQA",
            "source_repo": "lmms-lab-encoder/GQA", "source_split": "train_balanced",
            "source_id": row["id"], "source_image_id": identifier,
            "source_coco_id": metadata[identifier]["coco_id"],
            "source_image_url": metadata[identifier]["url"], "source_answer": row["answer"],
            "source_full_answer": row["fullAnswer"], "source_types": row["types"],
            "source_groups": row["groups"], "source_semantic": row["semantic"],
            "source_semantic_str": row["semanticStr"], "image_group": mapping[identifier]["group"],
            "assigned_split": split, "source_answers": canonical,
            "candidate_permutation": permutation, "candidate_domain": DOMAINS[row["task"]],
            "candidate_generation": "Original query GT unchanged; distractors from the same declared attribute or official object category; related name families cannot co-occur.",
        })
    return result


def write_rows(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"manifest": str(path), "questions": len(rows),
            "images": len({row["image_id"] for row in rows}),
            "by_task": dict(Counter(row["question_type"] for row in rows)),
            "candidate_counts": dict(Counter(len(row["answers"]) for row in rows)),
            "gold_positions_by_candidate_count": {str(k): dict(Counter(row["gold_index"] for row in rows if len(row["answers"]) == k)) for k in [4, 5, 6]},
            "gt_by_task": {task: dict(Counter(row["source_answer"] for row in rows if row["question_type"] == task)) for task in DOMAINS},
            "max_questions_per_image": max(Counter(row["image_id"] for row in rows).values()),
            "questions_with_relate_operation": sum(any(step["operation"].startswith("relate") for step in row["source_semantic"]) for row in rows)}


def prepare_queries(root, output, mapping, metadata, shards, train_size, dev_size):
    path = root / "supplement_gqa/raw/train_balanced_instructions/train-00000-of-00001.parquet"
    columns = ["id", "imageId", "question", "answer", "fullAnswer", "types", "groups", "semantic", "semanticStr"]
    table = pq.read_table(path, columns=columns, filters=[("imageId", "in", sorted(mapping))])
    inventory, filtered, seen = Counter(), [], set()
    duplicate_programs = 0
    for row in table.to_pylist():
        last = row["semantic"][-1]
        inventory[f"{row['types']['structural']}/{last['operation']}"] += 1
        task = task_for(row)
        if task is None:
            continue
        identity = row["imageId"], row["semanticStr"], row["answer"]
        if identity in seen:
            duplicate_programs += 1
            continue
        seen.add(identity)
        row["task"] = task
        filtered.append(row)
    train_pool = [row for row in filtered if mapping[row["imageId"]]["split"] == "train"]
    dev_pool = [row for row in filtered if mapping[row["imageId"]]["split"] == "dev"]
    unused_domain_values = {}
    for task, domain in DOMAINS.items():
        observed = {row["answer"] for row in train_pool if row["task"] == task}
        unused_domain_values[task] = [answer for answer in domain if answer not in observed]
        DOMAINS[task] = [answer for answer in domain if answer in observed]
    dev_pool = [row for row in dev_pool if row["answer"] in DOMAINS[row["task"]]]
    print("Typed unique query pools:", json.dumps({"train": dict(Counter(row["task"] for row in train_pool)), "dev": dict(Counter(row["task"] for row in dev_pool))}), flush=True)
    train_selected = sample_queries(train_pool, train_size, SEED + 30)
    dev_selected = sample_queries(dev_pool, dev_size, SEED + 31)
    image_paths, image_sizes = materialize_images(root, output / "images", {row["imageId"] for row in train_selected + dev_selected}, shards)
    train = convert(train_selected, "train", mapping, metadata, image_paths, image_sizes, SEED + 32)
    dev = convert(dev_selected, "dev", mapping, metadata, image_paths, image_sizes, SEED + 33)
    train_report = write_rows(output / "train.jsonl", train)
    dev_report = write_rows(output / "dev.jsonl", dev)
    prior = Counter((row["question_type"], row["source_answer"]) for row in train)
    for rows, report in [(train, train_report), (dev, dev_report)]:
        report["candidate_task_prior_accuracy"] = sum(max(range(len(row["answers"])), key=lambda i: prior[row["question_type"], row["answers"][i]]) == row["gold_index"] for row in rows) / len(rows)
        report["uniform_candidate_accuracy"] = sum(1 / len(row["answers"]) for row in rows) / len(rows)
    sample = []
    for task in DOMAINS:
        sample.extend([row for row in train if row["question_type"] == task][:3])
    write_rows(output / "samples.jsonl", sample)
    assert not ({row["image_group"] for row in train} & {row["image_group"] for row in dev})
    report = {
        "status": "ready", "dataset": "GQA official train query converted to typed multiple choice",
        "train": train_report, "dev": dev_report,
        "eligible_pool": {"train": dict(Counter(row["task"] for row in train_pool)), "dev": dict(Counter(row["task"] for row in dev_pool))},
        "input_task_inventory": dict(inventory), "same_image_program_gt_duplicates_removed": duplicate_programs,
        "domains": DOMAINS, "related_families_excluded_from_cooccurrence": [sorted(x) for x in RELATED_FAMILIES],
        "curated_values_absent_from_train_pool_removed": unused_domain_values,
        "gt_policy": "Exact original answer string preserved; only canonical closed vocabulary labels accepted. Plural forms, synonyms outside the curated vocabulary, binary attributes, untyped object queries and overlapping person labels are excluded rather than relabelled.",
        "candidate_policy": "K=4/5/6 assigned by an independent seeded shuffle, approximately equal counts; all options from one semantic domain and observed as GT in the train pool; GT included, selected related name families kept apart, source order and final permutation retained.",
        "selection": "Target task mixture with round-robin over GT values within each task; at most 3 GQA queries per image; identical image/program/GT paraphrases deduplicated. Scarce tasks give remaining capacity to available tasks without duplicates.",
        "split": "Uses the shared image_split.json; reproduces the old shard0/1 assignment, groups new images by VG/COCO ID, and removes known CV/3DSR source image overlaps.",
        "image_split_manifest": str(output.parent / "image_split.json"),
        "exported_or_reused_images": len(image_paths), "all_selected_images_decoded": True,
        "validation": "GT belongs to every distinct candidate set; all source questions and answers come from train_balanced; image groups do not cross train/dev; image files decoded successfully.",
        "limitations": "These are constructed choice questions, not the original open-ended GQA benchmark. Source GT is retained but the correctness of every visual annotation is not re-labelled. BLINK lacks source image IDs for exhaustive overlap exclusion. Internal candidate/task-prior accuracy is reported to expose non-visual shortcuts; visual transfer must be measured on fixed external evaluations.",
    }
    (output / "preparation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"train": train_report, "dev": dev_report, "images": len(image_paths)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--dev-size", type=int, default=1000)
    args = parser.parse_args()
    output = args.root / "supplement_multichoice"
    mapping, metadata = make_image_split(args.root, output, args.shards)
    if args.split_only:
        return
    prepare_queries(args.root, output / "gqa", mapping, metadata, args.shards, args.train_size, args.dev_size)


if __name__ == "__main__":
    main()
