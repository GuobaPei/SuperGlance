#!/usr/bin/env python3
"""Prepare native AI2D training MCQs from the trusted Cauldron train package.

usage: python -m superglance.builders.qa.prepare_ai2d_multichoice --source ROOT/supplement_multichoice/ai2d/raw \
         --output ROOT/supplement_multichoice/ai2d/prepared --download
(--download fetches The Cauldron's ai2d train parquet files into --source; adapt_stage1.py reads them there too)"""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import random
import re

from PIL import Image
import pyarrow.parquet as pq
import requests
from huggingface_hub.constants import ENDPOINT


REPO = "HuggingFaceM4/the_cauldron"
SOURCE_CARD = f"{ENDPOINT}/datasets/{REPO}"
CHOICE_RE = re.compile(r"^([A-Z])\. (.*?)(?=\n[A-Z]\. |\Z)", re.M | re.S)
ORDER_REFERENCE_RE = re.compile(
    r"\b(?:all|none|neither)\b.*\babove\b|\bboth\s+[a-d]\b|"
    r"\b[a-d]\s+(?:and|or|&)\s+[a-d]\b|\boption\s+[a-d]\b", re.I
)


def download_train(raw_dir, endpoint):
    raw_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"{endpoint}/api/datasets/{REPO}", timeout=30)
    response.raise_for_status()
    names = sorted(item["rfilename"] for item in response.json()["siblings"]
                   if item["rfilename"].startswith("ai2d/train-")
                   and item["rfilename"].endswith(".parquet"))
    for index, name in enumerate(names):
        target = raw_dir / f"train_{index}.parquet"
        if target.exists():
            continue
        response = requests.get(f"{endpoint}/datasets/{REPO}/resolve/main/{name}",
                                stream=True, timeout=(30, 120))
        response.raise_for_status()
        with target.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
        print(f"Downloaded {target.name}: {target.stat().st_size:,} bytes", flush=True)


def parse_turn(turn):
    question, choices = turn["user"].removeprefix("Question: ").split("\nChoices:\n", 1)
    choices = choices.removesuffix("\nAnswer with the letter.")
    parsed = CHOICE_RE.findall(choices)
    labels = [label for label, _ in parsed]
    answers = [answer for _, answer in parsed]
    answer_letter = turn["assistant"].removeprefix("Answer: ").strip()
    assert labels == list("ABCD") and answer_letter in labels
    return question, answers, labels.index(answer_letter)


def split_stats(rows):
    return {
        "questions": len(rows),
        "images": len({row["image_id"] for row in rows}),
        "candidate_counts": dict(Counter(str(row["candidate_count"]) for row in rows)),
        "gold_index_counts": dict(Counter(str(row["gold_index"]) for row in rows)),
        "question_types": dict(Counter(row["question_type"] for row in rows)),
        "order_preserved_for_option_references": sum(
            row["candidate_order_policy"] == "preserve_option_references" for row in rows
        ),
    }


def write_rows(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    if args.download:
        download_train(args.source, args.endpoint.rstrip("/"))
    files = sorted(args.source.glob("train_*.parquet"))
    assert files, "No AI2D train parquet files found."
    source_rows = [row for path in files for row in pq.read_table(path).to_pylist()]
    assert all(len(row["images"]) == 1 for row in source_rows)

    # Cauldron already groups the QA pairs for each image. Preserve that grouping.
    group_ids = list(range(len(source_rows)))
    random.Random(args.seed).shuffle(group_ids)
    dev_ids = set(group_ids[:round(len(group_ids) * args.dev_fraction)])
    output = args.output
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = {"train": [], "dev": []}
    source_image_bytes = set()
    image_formats = Counter()
    image_records = {}
    skipped_duplicate_candidates = []
    for image_index, source_row in enumerate(source_rows):
        image = source_row["images"][0]
        image_bytes = image["bytes"]
        source_image_bytes.add(image_bytes)
        with Image.open(BytesIO(image_bytes)) as decoded:
            decoded.load()
            size = list(decoded.size)
            fmt = decoded.format.lower()
        image_formats[fmt] += 1
        image_id = f"ai2d_train_image_{image_index:04d}"
        image_file = f"images/{image_id}.{fmt}"
        image_records[image_id] = (image_file, image_bytes)
        split = "dev" if image_index in dev_ids else "train"
        for question_index, turn in enumerate(source_row["texts"]):
            assert turn["source"] == "AI2D"
            question, answers, source_gold = parse_turn(turn)
            question_id = f"{image_id}_q{question_index:02d}"
            if len({answer.strip().casefold() for answer in answers}) != len(answers):
                skipped_duplicate_candidates.append(question_id)
                continue
            preserve_order = any(ORDER_REFERENCE_RE.search(answer) for answer in answers)
            permutation = list(range(len(answers)))
            if not preserve_order:
                random.Random(f"{args.seed}:{question_id}").shuffle(permutation)
            gold_index = permutation.index(source_gold)
            rows_by_split[split].append({
                "id": question_id,
                "image_id": image_id,
                "images": [image_file],
                "image_size": size,
                "question": question,
                "answers": [answers[index] for index in permutation],
                "gold_index": gold_index,
                "candidate_count": len(answers),
                "pattern": "Choice",
                "question_type": "diagram_mcq",
                "question_type_origin": "project label; official task subtype not packaged",
                "source": "AI2D",
                "source_repo": REPO,
                "source_config": "ai2d",
                "source_split": "train",
                "split": split,
                "source_image_row": image_index,
                "source_question_in_image": question_index,
                "source_image_id": None,
                "source_answers": answers,
                "source_gold_index": source_gold,
                "source_user": turn["user"],
                "source_assistant": turn["assistant"],
                "candidate_permutation": permutation,
                "candidate_order_policy": "preserve_option_references" if preserve_order else "seeded_shuffle",
            })
    # The package is image-grouped; verify that identical source images cannot cross splits.
    assert len(source_image_bytes) == len(source_rows)
    available_train = len(rows_by_split["train"])
    if args.train_size and available_train > args.train_size:
        rows_by_split["train"] = random.Random(args.seed + 1).sample(
            rows_by_split["train"], args.train_size
        )
    for rows in rows_by_split.values():
        rows.sort(key=lambda row: (row["source_image_row"], row["source_question_in_image"]))
        for row in rows:
            assert row["answers"][row["gold_index"]] == row["source_answers"][row["source_gold_index"]]
    selected_ids = {row["image_id"] for rows in rows_by_split.values() for row in rows}
    for image_id in sorted(selected_ids):
        image_file, image_bytes = image_records[image_id]
        (output / image_file).write_bytes(image_bytes)
    for split, rows in rows_by_split.items():
        write_rows(output / f"{split}.jsonl", rows)
    write_rows(output / "examples.jsonl", rows_by_split["train"][:3] + rows_by_split["dev"][:3])

    train_ids = {row["image_id"] for row in rows_by_split["train"]}
    eval_ids = {row["image_id"] for row in rows_by_split["dev"]}
    assert not train_ids & eval_ids
    report = {
        "source_repo": REPO,
        "source_config": "ai2d",
        "source_split": "train",
        "source_card": SOURCE_CARD,
        "source_split_evidence": "The Cauldron card states it contains only training splits, with extra decontamination against MMMU, MathVista and MMBench.",
        "native_multichoice": True,
        "source_image_groups": len(source_rows),
        "source_qa_pairs": sum(len(row["texts"]) for row in source_rows),
        "skipped_duplicate_candidate_questions": len(skipped_duplicate_candidates),
        "skipped_duplicate_candidate_ids": skipped_duplicate_candidates,
        "source_download_bytes": sum(path.stat().st_size for path in files),
        "source_images_decoded": len(source_rows),
        "source_unique_image_bytes": len(source_image_bytes),
        "source_image_formats": dict(image_formats),
        "dev_group_fraction": args.dev_fraction,
        "dev_image_groups": len(dev_ids),
        "train_qa_available_after_dev_split": available_train,
        "train_qa_not_selected": available_train - len(rows_by_split["train"]),
        "seed": args.seed,
        "splits": {split: split_stats(rows) for split, rows in rows_by_split.items()},
        "exported_images": len(selected_ids),
        "exported_image_bytes": sum(len(image_records[key][1]) for key in selected_ids),
        "train_dev_image_overlap": 0,
        "candidate_policy": "Seeded permutation; original order retained when options refer to other option letters or all/none of the above. Original texts and permutation saved.",
        "external_evaluation_overlap": {
            "fixed_benchmarks": ["CV-Bench", "BLINK single-image", "3DSRBench-real"],
            "original_image_id_available": False,
            "id_based_exclusion": "unavailable: Cauldron provides image bytes but no original AllenAI image ID, path, or source URL",
            "excluded_by_source_id": 0,
            "limitation": "Cannot certify cross-dataset image separation by source metadata. No benchmark QA or official AI2D test data was used.",
        },
        "outputs": {split: str(output / f"{split}.jsonl") for split in rows_by_split},
        "examples": str(output / "examples.jsonl"),
    }
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
