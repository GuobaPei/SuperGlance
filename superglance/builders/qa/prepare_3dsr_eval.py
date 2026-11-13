#!/usr/bin/env python3
"""Export the official 3DSRBench real-image circular TSV without changing options."""
import argparse
import base64
import csv
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image
from huggingface_hub.constants import ENDPOINT

from superglance.builders.qa.prepare_external_eval import download


REPO = "ccvl/3DSRBench"
TSV_NAME = "3dsrbench_v1_vlmevalkit_circular.tsv"
OFFICIAL_SCRIPT = "compute_3drbench_results_circular.py"
CATEGORY_MAPPING = {
    "height_higher": "height",
    "location_above": "location",
    "location_closer_to_camera": "location",
    "location_next_to": "location",
    "orientation_in_front_of": "orientation",
    "orientation_on_the_left": "orientation",
    "orientation_viewpoint": "orientation",
    "multi_object_closer_to": "multi_object",
    "multi_object_facing": "multi_object",
    "multi_object_viewpoint_towards_object": "multi_object",
    "multi_object_parallel": "multi_object",
    "multi_object_same_direction": "multi_object",
}


def variant_info(qid):
    match = re.fullmatch(r"([A-Z0-9]{8})(-flip)?(?:-([1-3]))?", qid)
    assert match is not None, qid
    base, flipped, rotation = match.groups()
    return base, bool(flipped), int(rotation or 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--endpoint", default=ENDPOINT)
    args = parser.parse_args()
    raw, prepared = args.root / "raw", args.root / "prepared"
    (prepared / "images").mkdir(parents=True, exist_ok=True)
    for filename in (TSV_NAME, OFFICIAL_SCRIPT, "3dsrbench_v1.csv", "README.md"):
        download(REPO, filename, raw / filename, args.endpoint)
    with (raw / "3dsrbench_v1.csv").open() as stream:
        raw_annotations = list(csv.DictReader(stream))
    assert len(raw_annotations) == 2100

    csv.field_size_limit(10_000_000)
    records, images, circular_groups = [], {}, defaultdict(list)
    with (raw / TSV_NAME).open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            assert row["image_source"] == "MS-COCO"
            qid = row["qid"]
            base, flipped, rotation = variant_info(qid)
            answers = [row[key] for key in "ABCD" if row[key]]
            gold = ord(row["answer"]) - ord("A")
            assert len(answers) in (2, 4) and len(set(answers)) == len(answers)
            assert 0 <= gold < len(answers)
            image_name = Path(urlparse(row["image_url"]).path).stem
            image_id = f"coco_{image_name}_{'flip' if flipped else 'original'}"
            payload = base64.b64decode(row["image"], validate=True)
            if image_id in images:
                assert images[image_id][2] == payload, image_id
                relative, image_size, _ = images[image_id]
            else:
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    image_size = list(image.size)
                    extension = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}[image.format]
                relative = f"images/{image_id}.{extension}"
                (prepared / relative).write_bytes(payload)
                images[image_id] = (relative, image_size, payload)
            circular_key = base + ("-flip" if flipped else "")
            record = {
                "id": f"3dsr_{row['index']}", "image_id": image_id,
                "images": [relative], "image_size": image_size,
                "question": row["question"], "answers": answers, "gold_index": gold,
                "pattern": "Choice", "question_type": row["category"],
                "task": CATEGORY_MAPPING[row["category"]],
                "source": "MS-COCO", "source_benchmark": "3DSRBench-real",
                "source_split": "official_circular", "source_id": int(row["index"]),
                "source_qid": qid, "source_image_url": row["image_url"],
                "source_answers": answers, "source_answer": row["answer"],
                "base_question_id": base, "flipped_image": flipped,
                "circular_order": rotation, "circular_original_id": circular_key,
            }
            records.append(record)
            circular_groups[circular_key].append(record)

    assert len({row["id"] for row in records}) == len(records)
    assert len({row["source_qid"] for row in records}) == len(records)
    for group in circular_groups.values():
        original = next(row for row in group if row["circular_order"] == 0)
        assert {row["circular_order"] for row in group} == set(range(len(original["answers"])))
        for row in group:
            assert row["question"] == original["question"]
            assert row["image_id"] == original["image_id"]
            assert sorted(row["answers"]) == sorted(original["answers"])
            assert row["answers"][row["gold_index"]] == original["answers"][original["gold_index"]]

    manifest = prepared / "3dsr_real_circular.jsonl"
    with manifest.open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")
    flip_coverage = defaultdict(set)
    for row in records:
        flip_coverage[row["base_question_id"]].add(row["flipped_image"])
    report = {
        "purpose": "Fixed external evaluation only; never training or model selection",
        "dataset": REPO, "source_file": TSV_NAME, "download_endpoint": args.endpoint,
        "source_raw_annotations": len(raw_annotations), "manifest": str(manifest),
        "scope_note": "The author's processed TSV expands 2,100 raw annotations into 2,625 base prompted questions and preserves its supplied image-flip/option-rotation variants; no custom augmentation",
        "n_scored_rows": len(records), "n_exported_images": len(images),
        "n_source_coco_images": len({row["source_image_url"] for row in records}),
        "n_base_questions": len({row["base_question_id"] for row in records}),
        "n_circular_groups": len(circular_groups),
        "flip_coverage": {"both_original_and_flip": sum(len(flags) == 2 for flags in flip_coverage.values()),
                          "one_orientation_only": sum(len(flags) == 1 for flags in flip_coverage.values())},
        "by_subtype_rows": dict(Counter(row["question_type"] for row in records)),
        "by_category_rows": dict(Counter(row["task"] for row in records)),
        "candidate_counts": dict(Counter(len(row["answers"]) for row in records)),
        "rows_per_base_question": dict(Counter(Counter(row["base_question_id"] for row in records).values())),
        "official_metric": "All circular orders must be correct within qid after removing final -1/-2/-3; original and -flip are separate groups; ordinary mean across these groups",
        "additional_metric": "Strict flip+circular accuracy requires every provided variant of each base question to be correct; some official questions have only one image orientation",
        "readout_protocol": "Native A/B/C/D candidate-token logits; no generated-answer judge",
        "checks": ["All decoded images are valid", "Exactly one image per row",
                   "Original option order and GT preserved", "Complete circular groups",
                   "Answer meaning unchanged under option rotation",
                   "Flipped image payloads stored separately from original images"],
        "evidence": {
            "official_card": "ccvl/3DSRBench dataset card",
            "official_aggregation": f"{REPO}: {OFFICIAL_SCRIPT}",
            "community_aggregation": "lmms-eval: lmms_eval/tasks/3dsrbench/utils.py",
        },
    }
    (args.root / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
