#!/usr/bin/env python3
"""Download static external benchmarks and preserve their original options."""
import argparse
import io
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import requests
from PIL import Image
from huggingface_hub.constants import ENDPOINT


BLINK_TASKS = {"Spatial_Relation": 143, "Relative_Depth": 124,
               "Object_Localization": 122, "Counting": 120}
CV_FILES = ("test_2d.parquet", "test_3d.parquet")


def download(repo, filename, target, endpoint):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return
    url = f"{endpoint}/datasets/{repo}/resolve/main/{filename}"
    partial = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with partial.open("wb") as output:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                output.write(chunk)
    partial.replace(target)
    print(f"Downloaded {repo}/{filename}: {target.stat().st_size} bytes", flush=True)


def image_file(image_value, output, identifier):
    payload = image_value["bytes"]
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        extension = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}[image.format]
        size = list(image.size)
    relative = Path("images") / f"{identifier}.{extension}"
    (output / relative).write_bytes(payload)
    return relative.as_posix(), size


def decision(row, image_value, output, identifier, benchmark, split, question_type):
    answers = list(row["choices"])
    answer_letter = row["answer"].strip().removeprefix("(").removesuffix(")")
    gold = ord(answer_letter) - ord("A")
    assert len(answer_letter) == 1 and 0 <= gold < len(answers)
    assert len(answers) >= 2 and len(set(answers)) == len(answers)
    relative, image_size = image_file(image_value, output, identifier)
    record = {
        "id": identifier, "image_id": identifier, "images": [relative],
        "image_size": image_size, "question": row["question"], "answers": answers,
        "gold_index": gold, "pattern": "Choice", "question_type": question_type,
        "source_benchmark": benchmark, "source_split": split, "source_id": row["idx"],
        "source_answers": answers, "source_answer": row["answer"],
        "source_prompt": row["prompt"], "source": row.get("source", benchmark),
        "task": row.get("task", row.get("sub_task", question_type)),
    }
    lower = [answer.strip().lower() for answer in answers]
    if set(lower) == {"yes", "no"}:
        record.update(pattern="Noul", yes_index=lower.index("yes"))
    for key in ("type", "filename", "source_dataset", "source_filename"):
        if key in row:
            record[key] = row[key]
    return record


def write_manifest(path, rows):
    assert len({row["id"] for row in rows}) == len(rows)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")
    return {"manifest": str(path), "n": len(rows),
            "by_task": dict(Counter(row["question_type"] for row in rows)),
            "by_source": dict(Counter(row["source"] for row in rows)),
            "by_pattern": dict(Counter(row["pattern"] for row in rows)),
            "candidate_counts": dict(Counter(len(row["answers"]) for row in rows)),
            "gold_positions": dict(Counter(row["gold_index"] for row in rows))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--endpoint", default=ENDPOINT)
    args = parser.parse_args()
    raw, output = args.root / "raw", args.root / "prepared"
    (output / "images").mkdir(parents=True, exist_ok=True)
    for filename in (*CV_FILES, "README.md"):
        download("nyu-visionx/CV-Bench", filename, raw / "cvbench" / filename, args.endpoint)
    for task in BLINK_TASKS:
        filename = f"{task}/val-00000-of-00001.parquet"
        download("BLINK-Benchmark/BLINK", filename, raw / "blink" / filename, args.endpoint)
    download("BLINK-Benchmark/BLINK", "README.md", raw / "blink" / "README.md", args.endpoint)

    cv_rows = []
    for filename in CV_FILES:
        for row in pq.read_table(raw / "cvbench" / filename).to_pylist():
            identifier = f"cvbench_{row['type']}_{row['idx']}"
            cv_rows.append(decision(row, row["image"], output, identifier,
                                    "CV-Bench", "test", f"{row['type']}/{row['task']}"))
    assert len(cv_rows) == 2638
    assert len({row["source"] for row in cv_rows}) == 3 and {"ADE20K", "COCO"} < {row["source"] for row in cv_rows}

    blink_rows = []
    for task, expected in BLINK_TASKS.items():
        rows = pq.read_table(raw / "blink" / task / "val-00000-of-00001.parquet").to_pylist()
        assert len(rows) == expected
        for row in rows:
            images = [value for key, value in row.items() if key.startswith("image_") and value is not None]
            assert len(images) == 1, f"Expected one image: {row['idx']}"
            blink_rows.append(decision(row, images[0], output, f"blink_{row['idx']}",
                                       "BLINK", "val", task))

    report = {
        "purpose": "Fixed external evaluation only; never used for training or model selection",
        "download_endpoint": args.endpoint,
        "option_order": "Unchanged from original benchmark; source answers and labels retained",
        "readout_protocol": "Native A/B/C/... candidate-token logits; no generated-answer parsing",
        "cvbench": write_manifest(output / "cvbench.jsonl", cv_rows),
        "blink_single_image": write_manifest(output / "blink_single_image.jsonl", blink_rows),
        "checks": ["All images decode", "Exactly one image per example", "Unique example IDs",
                   "Ground-truth letters map to original candidate positions", "Expected split counts"],
    }
    (args.root / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
