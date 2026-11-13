#!/usr/bin/env python3
"""Prepare a native yes/no GQA train supplement, with images and held-out dev."""
import argparse
import io
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import requests
from PIL import Image
from huggingface_hub.constants import ENDPOINT


GQA_REPO = "lmms-lab-encoder/GQA"
IMAGE_SHARDS = [f"train_balanced_images/train-{i:05d}-of-00021.parquet" for i in range(2)]
QUESTION_FILE = "train_balanced_instructions/train-00000-of-00001.parquet"
TASK_WEIGHTS = {
    "verify/rel": 40, "verify/attr": 20, "logical/obj": 20,
    "logical/attr": 5, "compare/attr": 8, "verify/obj": 5, "verify/global": 2,
}


def download(endpoint, repo, filename, target):
    download_url(f"{endpoint}/datasets/{repo}/resolve/main/{filename}", target)


def download_url(url, target):
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with partial.open("wb") as stream:
            for chunk in response.iter_content(4 * 1024 * 1024):
                stream.write(chunk)
    partial.replace(target)
    print(f"Downloaded {target.name}: {target.stat().st_size} bytes", flush=True)


def excluded_coco_ids(manifests):
    excluded, by_benchmark = set(), {}
    for path in manifests:
        ids = set()
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if "COCO" not in row.get("source", "").upper():
                    continue
                original = row.get("source_image_url", row.get("source_filename", ""))
                match = re.search(r"(?:/|_)(\d{12})\.(?:jpg|png)$", original)
                if match:
                    ids.add(int(match.group(1)))
        excluded.update(ids)
        by_benchmark[path.name] = len(ids)
    return excluded, by_benchmark


def sample_rows(rows, n, seed, max_per_image, initial_counts=None):
    """Fixed task mix, balanced true labels, and at most three QAs per image."""
    rng = random.Random(seed)
    pools = defaultdict(list)
    for row in rows:
        pools[(row["task"], row["answer"])].append(row)
    for pool in pools.values():
        rng.shuffle(pool)
    selected, per_image = [], Counter(initial_counts or {})
    # Reserve rare tasks first so the per-image cap cannot crowd them out.
    for task in sorted(TASK_WEIGHTS, key=lambda t: len(pools[(t, "yes")]) + len(pools[(t, "no")])):
        quota = n * TASK_WEIGHTS[task] // 100
        assert quota % 2 == 0
        for answer in ("yes", "no"):
            added = 0
            for row in pools[(task, answer)]:
                if per_image[row["imageId"]] >= max_per_image:
                    continue
                selected.append(row)
                per_image[row["imageId"]] += 1
                added += 1
                if added == quota // 2:
                    break
            if added != quota // 2:
                raise ValueError(f"Not enough {task}/{answer}: {added} of {quota // 2}")
    assert len(selected) == n
    rng.shuffle(selected)
    return selected


def sample_counts(rows, n, seed):
    rng = random.Random(seed)
    pools = defaultdict(list)
    for row in rows:
        pools[row["answer"]].append(row)
    for pool in pools.values():
        rng.shuffle(pool)
    result, counts = [], Counter()
    # Round-robin over real counts. Sparse labels run out naturally; no repeats.
    while any(pools.values()):
        labels = sorted(pools)
        rng.shuffle(labels)
        for label in labels:
            while pools[label]:
                row = pools[label].pop()
                if counts[row["imageId"]] < 2:
                    result.append(row)
                    counts[row["imageId"]] += 1
                    break
            if len(result) == n:
                rng.shuffle(result)
                return result
    raise ValueError(f"Only {len(result)} unique native counting questions, requested {n}")


def convert_counts(rows, metadata, answer_domain, seed):
    rng = random.Random(seed)
    candidate_counts = [5] * (len(rows) // 2) + [6] * (len(rows) - len(rows) // 2)
    rng.shuffle(candidate_counts)
    records = []
    for row, count in zip(rows, candidate_counts):
        correct = row["answer"]
        canonical = sorted([correct] + rng.sample([x for x in answer_domain if x != correct], count - 1))
        permutation = list(range(count))
        rng.shuffle(permutation)
        answers = [str(canonical[i]) for i in permutation]
        identifier = row["imageId"]
        original = metadata[identifier]
        records.append({
            "id": f"tallyqa_train_{row['question_id']}", "image_id": f"vg_{identifier}",
            "images": [f"images/vg_{identifier}.jpg"], "question": row["question"],
            "answers": answers, "gold_index": answers.index(str(correct)),
            "pattern": "Choice", "question_type": "count",
            "source": "TallyQA", "source_repo": "TallyQA (official release)",
            "source_split": "train", "source_id": row["question_id"],
            "source_image_id": identifier, "source_coco_id": original["coco_id"],
            "source_image_url": original["url"], "source_answer": correct,
            "source_data_source": row["data_source"], "source_issimple": row.get("issimple"),
            "candidate_generation": "Uniform without replacement from official train integer answer domain; includes GT",
            "source_answers": [str(x) for x in canonical], "candidate_permutation": permutation,
        })
    return records


def record_report(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"manifest": str(path), "questions": len(rows),
            "images": len({row["image_id"] for row in rows}),
            "by_source": dict(Counter(row["source"] for row in rows)),
            "by_task": dict(Counter(row["question_type"] for row in rows)),
            "by_pattern": dict(Counter(row["pattern"] for row in rows)),
            "candidate_counts": dict(Counter(len(row["answers"]) for row in rows)),
            "gold_positions_by_candidate_count": {str(k): dict(Counter(row["gold_index"] for row in rows if len(row["answers"]) == k)) for k in sorted({len(row["answers"]) for row in rows})},
            "count_gt_by_candidate_count": {str(k): dict(Counter(row["source_answer"] for row in rows if row["source"] == "TallyQA" and len(row["answers"]) == k)) for k in [5, 6]},
            "max_questions_per_image": max(Counter(row["image_id"] for row in rows).values())}


def write_manifest(output, name, rows, metadata, seed):
    rng, converted = random.Random(seed), []
    for row in rows:
        answers = ["yes", "no"]
        rng.shuffle(answers)
        identifier = str(row["imageId"])
        original = metadata[identifier]
        record = {
            "id": f"gqa_train_{row['id']}", "image_id": f"vg_{identifier}",
            "images": [f"images/vg_{identifier}.jpg"],
            "question": row["question"], "answers": answers,
            "gold_index": answers.index(row["answer"]), "pattern": "Noul",
            "yes_index": answers.index("yes"), "question_type": row["task"],
            "source": "GQA", "source_repo": GQA_REPO,
            "source_split": "train_balanced", "source_id": row["id"],
            "source_image_id": identifier, "source_coco_id": original["coco_id"],
            "source_image_url": original["url"], "source_answer": row["answer"],
            "source_full_answer": row["fullAnswer"], "source_types": row["types"],
            "source_semantic": row["semantic"], "source_semantic_str": row["semanticStr"],
            "source_is_balanced": row["isBalanced"],
        }
        converted.append(record)
    path = output / f"{name}.jsonl"
    with path.open("w") as stream:
        for record in converted:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return converted, {
        "manifest": str(path), "questions": len(converted),
        "images": len({row["image_id"] for row in converted}),
        "by_task": dict(Counter(row["question_type"] for row in converted)),
        "source_answers": dict(Counter(row["source_answer"] for row in converted)),
        "gold_positions": dict(Counter(row["gold_index"] for row in converted)),
        "pattern": "Noul", "max_questions_per_image": max(Counter(row["image_id"] for row in converted).values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, action="append", required=True)
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--dev-size", type=int, default=1000)
    parser.add_argument("--count-train-size", type=int, default=0)
    parser.add_argument("--count-dev-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--tallyqa-url", help="tallyqa.zip of the TallyQA release (read with --count-train-size)")
    args = parser.parse_args()
    raw = args.root / "raw"
    output = args.root / ("prepared_mixed" if args.count_train_size else "prepared")
    (output / "images").mkdir(parents=True, exist_ok=True)
    downloads = [(GQA_REPO, filename, raw / filename) for filename in [QUESTION_FILE, *IMAGE_SHARDS, "README.md"]]
    downloads.append(("jn12/VisualGenome", "image_data.json.zip", raw / "image_data.json.zip"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda item: download(args.endpoint, *item), downloads))
    if args.count_train_size:
        download_url(args.tallyqa_url, raw / "tallyqa.zip")

    with zipfile.ZipFile(raw / "image_data.json.zip") as archive:
        metadata = {str(row["image_id"]): row for row in json.loads(archive.read("image_data.json"))}
    excluded, exclusion_sources = excluded_coco_ids(args.eval_manifest)
    downloaded_ids = set()
    for filename in IMAGE_SHARDS:
        downloaded_ids.update(str(x) for x in pq.read_table(raw / filename, columns=["id"])["id"].to_pylist())
    missing_metadata = downloaded_ids - metadata.keys()
    overlap_ids = {i for i in downloaded_ids & metadata.keys() if metadata[i]["coco_id"] in excluded}
    available = downloaded_ids - missing_metadata - overlap_ids
    candidates, seen_questions = [], set()
    counters = Counter()
    counters["all_train_questions_read"] = pq.ParquetFile(raw / QUESTION_FILE).metadata.num_rows
    native_rows = pq.read_table(raw / QUESTION_FILE, filters=[
        ("imageId", "in", sorted(available)), ("answer", "in", ["yes", "no"])
    ])
    for batch in native_rows.to_batches(max_chunksize=8192):
        for row in batch.to_pylist():
            key = row["imageId"], row["question"].strip().casefold()
            if key in seen_questions:
                counters["duplicate_image_question_removed"] += 1
                continue
            seen_questions.add(key)
            row["task"] = row["types"]["structural"] + "/" + row["types"]["semantic"]
            if row["task"] in TASK_WEIGHTS:
                candidates.append(row)
    tally_rows, count_train, count_dev, answer_counts = [], [], [], Counter()
    if args.count_train_size:
        with zipfile.ZipFile(raw / "tallyqa.zip") as archive:
            # The archive also contains test.json; only official train is read.
            tally_train = json.loads(archive.read("train.json"))
        answer_counts.update(row["answer"] for row in tally_train)
        tally_seen = set()
        for row in tally_train:
            if not row["image"].startswith(("VG_100K/", "VG_100K_2/")):
                continue
            image_id = Path(row["image"]).stem
            key = image_id, row["question"].strip().casefold()
            if image_id not in available or key in tally_seen:
                continue
            tally_seen.add(key)
            row["imageId"] = image_id
            tally_rows.append(row)
        print(f"Matched TallyQA train: {len(tally_rows)} native counting questions", flush=True)
    image_groups = {i: f"coco_{metadata[i]['coco_id']}" if metadata[i]["coco_id"] else f"vg_{i}"
                    for i in available}
    groups = sorted(set(image_groups.values()))
    random.Random(args.seed).shuffle(groups)
    dev_groups = set(groups[:len(groups) // 10])
    train_pool = [row for row in candidates if image_groups[row["imageId"]] not in dev_groups]
    dev_pool = [row for row in candidates if image_groups[row["imageId"]] in dev_groups]
    if args.count_train_size:
        count_train = sample_counts([row for row in tally_rows if image_groups[row["imageId"]] not in dev_groups], args.count_train_size, args.seed + 5)
        count_dev = sample_counts([row for row in tally_rows if image_groups[row["imageId"]] in dev_groups], args.count_dev_size, args.seed + 6)
    train = sample_rows(train_pool, args.train_size, args.seed + 1, 3, Counter(row["imageId"] for row in count_train))
    dev = sample_rows(dev_pool, args.dev_size, args.seed + 2, 3, Counter(row["imageId"] for row in count_dev))
    selected_ids = {row["imageId"] for row in train + dev + count_train + count_dev}
    image_sizes = {}
    for filename in IMAGE_SHARDS:
        for batch in pq.ParquetFile(raw / filename).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                identifier = str(row["id"])
                if identifier not in selected_ids:
                    continue
                payload = row["image"]["bytes"]
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    assert image.format == "JPEG"
                    image_sizes[identifier] = list(image.size)
                (output / "images" / f"vg_{identifier}.jpg").write_bytes(payload)
    assert image_sizes.keys() == selected_ids
    train_records, train_report = write_manifest(output, f"gqa_train{args.train_size}", train, metadata, args.seed + 3)
    dev_records, dev_report = write_manifest(output, f"gqa_dev{args.dev_size}", dev, metadata, args.seed + 4)
    if args.count_train_size:
        train_records.extend(convert_counts(count_train, metadata, sorted(answer_counts), args.seed + 7))
        dev_records.extend(convert_counts(count_dev, metadata, sorted(answer_counts), args.seed + 8))
        random.Random(args.seed + 9).shuffle(train_records)
        random.Random(args.seed + 10).shuffle(dev_records)
        train_report = record_report(output / "supplement_train10k.jsonl", train_records)
        dev_report = record_report(output / "supplement_dev1k.jsonl", dev_records)
    assert not ({row["image_id"] for row in train_records} & {row["image_id"] for row in dev_records})
    assert not ({row["source_coco_id"] for row in train_records if row["source_coco_id"]} &
                {row["source_coco_id"] for row in dev_records if row["source_coco_id"]})
    report = {
        "dataset": "GQA balanced train + TallyQA native counting" if args.count_train_size else "GQA balanced train, native yes/no supplement", "status": "ready",
        "download_endpoint": args.endpoint, "repo": GQA_REPO,
        "original_dataset": "https://cs.stanford.edu/people/dorarad/gqa/",
        "download_files": [{"repo": repo, "file": file, "bytes": target.stat().st_size} for repo, file, target in downloads],
        "download_bytes": sum(target.stat().st_size for _, _, target in downloads),
        "image_shards": IMAGE_SHARDS, "downloaded_images": len(downloaded_ids),
        "available_native_yes_no_questions": len(candidates), "scan_counts": dict(counters),
        "selection": "First two released image shards. Within GQA: seeded 40% relation, 20% attribute, 20% object logic and 20% other native yes/no tasks, with equal yes/no labels per task. At most 3 questions per image across the mixture.",
        "sampling_limitation": "This is a bounded pilot from two image shards, not a uniform sample of all GQA images.",
        "image_split": "10% of original train image groups reserved before sampling dev; shared COCO IDs grouped together.",
        "source_split": "Only train_balanced; no original validation/test/benchmark questions used.",
        "label_policy": "Original GQA yes/no GT and semantic programs retained. Candidate order seeded and labels remapped. No teacher or invented distractors.",
        "seed": args.seed, "train": train_report, "dev": dev_report,
        "exported_images": len(image_sizes), "all_exported_images_decoded": True,
        "excluded_missing_vg_metadata_images": len(missing_metadata),
        "external_overlap_exclusion": {
            "mapping": "Visual Genome image_data.json official metadata mirror jn12/VisualGenome",
            "known_coco_ids_by_manifest": exclusion_sources, "known_coco_id_union": len(excluded),
            "removed_downloaded_images": len(overlap_ids), "remaining_known_overlap": 0,
            "limitations": "BLINK fixed subset provides no source image IDs in the prepared metadata. ID-based exclusion cannot certify absence of unreported BLINK overlap or cross-dataset crops/re-encodings. SAT uses rendered synthetic scenes; this supplement uses VG natural images.",
        },
        "spar_status": "Official jasonzhango/SPAR-7M currently distributes annotations only. No raw ScanNet/ScanNet++/Matterport3D/Structured3D images are bundled, so it is deferred until those image sources are available.",
    }
    if args.count_train_size:
        report.update({
            "source_split": "GQA train_balanced + TallyQA train only; test.json inside the TallyQA archive was not read.",
            "label_policy": "Original native yes/no and integer count GT retained. Count distractors are mutually exclusive integer alternatives; no teacher.",
            "tallyqa": {
                "source_archive": args.tallyqa_url,
                "archive_bytes": (raw / "tallyqa.zip").stat().st_size,
                "official_train_questions": len(tally_train),
                "official_train_answer_distribution": dict(sorted(answer_counts.items())),
                "matched_train_questions_after_image_and_overlap_filter": len(tally_rows),
                "matched_answer_distribution": dict(sorted(Counter(row["answer"] for row in tally_rows).items())),
                "train_selected_answer_distribution": dict(sorted(Counter(row["answer"] for row in count_train).items())),
                "dev_selected_answer_distribution": dict(sorted(Counter(row["answer"] for row in count_dev).items())),
                "count_sampling": "Round-robin over real integer GT values with seeded random within each value; scarce values are exhausted without duplicates and remaining capacity goes to available values. At most two count questions per image.",
                "candidate_domain": sorted(answer_counts),
                "candidate_policy": "5 or 6 candidates allocated independently of GT by a seeded shuffle, half of each. Sample 4/5 distinct false integers uniformly from the official train answer domain 0..15; add GT and shuffle. Store original ascending candidates and permutation.",
                "caveat": "Constructed multi-choice train conversion; its accuracy is not the original open-ended TallyQA benchmark metric.",
            },
            "joint_image_split": "GQA and TallyQA use one shared image-group split and one max-3-questions-per-image cap; common VG and COCO IDs cannot cross train/dev.",
            "gqa_count_check": "No integer answer, count semantic operation, or How many question was found for the downloaded GQA image IDs.",
        })
    (output / "preparation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
