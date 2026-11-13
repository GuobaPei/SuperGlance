#!/usr/bin/env python3
"""Evaluate every cyclic option rotation using unchanged candidate-token readout.

One JSONL output line is one original QA with K view records. Argmax is taken
in letter space BEFORE mapping to original answer identity, preserving ties.
No rotation voting or probability averaging changes any prediction.
Out: <output>/rotation_predictions.jsonl and rotation_metrics.json (all_rotations_correct_rate: right in every cyclic order).
usage: python -m superglance.eval.qa_rotations --model BASE [--adapter ADAPTER] --manifest M.jsonl --output DIR [--batch-size 8]
"""
import argparse
import json
import string
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(manifest, limit=None):
    """Select eligible candidate decisions; limit applies after filtering."""
    rows, excluded = [], Counter()
    scanned = 0
    with Path(manifest).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            scanned += 1
            if row.get("pattern", "Choice") not in ("Choice", "Noul"):
                excluded["unsupported_pattern"] += 1
                continue
            if row.get("candidate_order_policy") == "preserve_option_references":
                excluded["preserve_option_references"] += 1
                continue
            images = row.get("images")
            if images is not None and len(images) != 1:
                excluded["not_single_image"] += 1
                continue
            image = row.get("image_file") or (images[0] if images else None)
            if not image:
                excluded["no_image"] += 1
                continue
            count = len(row["answers"])
            if not 2 <= count <= 26 or not 0 <= row["gold_index"] < count:
                raise ValueError(f"Invalid candidates or GT at manifest line {line_number}")
            image = Path(image)
            if not image.is_absolute():
                image = Path(manifest).resolve().parent / image
            row["image_file"] = str(image)
            row["manifest_line"] = line_number
            row["evaluation_qa_index"] = len(rows)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("Manifest has no eligible single-image candidate decisions")
    return rows, {"scanned_rows": scanned, "selected_qa": len(rows),
                  "excluded_rows": dict(excluded), "limit_after_filter": limit}


def rotated_row(row, shift):
    """order[new letter index] = original answer index; GT follows its answer."""
    count = len(row["answers"])
    order = [(i + shift) % count for i in range(count)]
    result = dict(row, answers=[row["answers"][i] for i in order],
                  gold_index=order.index(row["gold_index"]))
    if "yes_index" in row:
        result["yes_index"] = order.index(row["yes_index"])
    return result, order


def view_record(row, shift, logits):
    """Store the actual letter-space winner, then align by answer identity."""
    import torch

    view, order = rotated_row(row, shift)
    values = torch.as_tensor(logits).detach().float().cpu()[:len(order)]
    probabilities = values.softmax(-1).tolist()
    letter_index = int(values.argmax())
    content_index = order[letter_index]
    aligned = [probabilities[order.index(i)] for i in range(len(order))]
    result = {
        "shift": shift, "order": order, "gold_letter_index": view["gold_index"],
        "gold_letter": string.ascii_uppercase[view["gold_index"]],
        "predicted_letter_index": letter_index,
        "predicted_letter": string.ascii_uppercase[letter_index],
        "predicted_content_index": content_index,
        "predicted_answer": row["answers"][content_index],
        "correct": content_index == row["gold_index"],
        "candidate_logits": values.tolist(),
        "probabilities_by_letter": probabilities,
        "probabilities_by_content": aligned,
        "exact_top_logit_tie": bool((values == values.max()).sum() > 1),
    }
    if "yes_index" in view:
        result.update(yes_letter_index=view["yes_index"], p_yes=probabilities[view["yes_index"]])
    return result


def complete_record(row, views):
    # Retain the original question, candidate texts, provenance and split fields.
    result = dict(row)
    original = views[0]["correct"]
    changed = views[1:]
    result.update(
        candidate_count=len(row["answers"]), views=views,
        original_correct=original,
        all_rotations_correct=all(view["correct"] for view in views),
        content_consistent=len({view["predicted_content_index"] for view in views}) == 1,
        rotation_accuracy=sum(view["correct"] for view in views) / len(views),
        nonzero_rotation_accuracy=sum(view["correct"] for view in changed) / len(changed),
        correct_to_wrong_count=sum(original and not view["correct"] for view in changed),
        wrong_to_correct_count=sum(not original and view["correct"] for view in changed),
    )
    return result


def score_all_rotations(rows, processor, model, batch_size):
    """Same QA batch membership at every shift; native independent readout."""
    import torch

    from superglance.data.qa import DecisionCollator, candidate_token_ids, native_candidate_logits
    from superglance.model.prompt import validate_answer_boundary

    max_options = max(len(row["answers"]) for row in rows)
    validate_answer_boundary(processor, max_options)
    token_ids = candidate_token_ids(processor, max_options)
    collator = DecisionCollator(processor)
    by_count = defaultdict(list)
    for row in rows:
        by_count[len(row["answers"])].append(row)
    views = {row["evaluation_qa_index"]: [] for row in rows}
    with torch.inference_mode():
        for count, subset in sorted(by_count.items()):
            for shift in range(count):
                for start in range(0, len(subset), batch_size):
                    originals = subset[start:start + batch_size]
                    batch = collator([rotated_row(row, shift)[0] for row in originals])
                    batch.pop("gold_index")
                    batch.pop("candidate_count")
                    batch = {key: value.to(model.device) for key, value in batch.items()}
                    logits = native_candidate_logits(model, batch, token_ids)
                    for row, values in zip(originals, logits):
                        views[row["evaluation_qa_index"]].append(view_record(row, shift, values))
                print(f"K={count} shift={shift}: {len(subset)} QA complete", flush=True)
    return [complete_record(row, views[row["evaluation_qa_index"]]) for row in rows]


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def aggregate(records):
    n = len(records)
    all_views = [view for row in records for view in row["views"]]
    original_correct = sum(row["original_correct"] for row in records)
    correct_pairs = sum(len(row["views"]) - 1 for row in records if row["original_correct"])
    wrong_pairs = sum(len(row["views"]) - 1 for row in records if not row["original_correct"])
    ctw = sum(row["correct_to_wrong_count"] for row in records)
    wtc = sum(row["wrong_to_correct_count"] for row in records)
    result = {
        "n_qa": n, "n_views": len(all_views),
        "original_accuracy": original_correct / n,
        "all_rotations_accuracy_qa_mean": sum(row["rotation_accuracy"] for row in records) / n,
        "nonzero_rotations_accuracy_qa_mean": sum(row["nonzero_rotation_accuracy"] for row in records) / n,
        "all_rotations_accuracy_view_mean": sum(view["correct"] for view in all_views) / len(all_views),
        "all_rotations_correct_rate": sum(row["all_rotations_correct"] for row in records) / n,
        "content_consistency_rate": sum(row["content_consistent"] for row in records) / n,
        "exact_top_logit_tie_view_rate": sum(view["exact_top_logit_tie"] for view in all_views) / len(all_views),
        "correct_to_wrong": {"view_pairs": ctw, "eligible_view_pairs": correct_pairs,
                             "conditional_pair_rate": fraction(ctw, correct_pairs),
                             "qa_with_any_flip": sum(row["correct_to_wrong_count"] > 0 for row in records),
                             "eligible_qa": original_correct},
        "wrong_to_correct": {"view_pairs": wtc, "eligible_view_pairs": wrong_pairs,
                             "conditional_pair_rate": fraction(wtc, wrong_pairs),
                             "qa_with_any_flip": sum(row["wrong_to_correct_count"] > 0 for row in records),
                             "eligible_qa": n - original_correct},
    }
    result["all_rotations_minus_original"] = result["all_rotations_accuracy_qa_mean"] - result["original_accuracy"]
    result["nonzero_rotations_minus_original"] = result["nonzero_rotations_accuracy_qa_mean"] - result["original_accuracy"]
    return result


def summarize(records):
    by_count = defaultdict(list)
    for row in records:
        by_count[row["candidate_count"]].append(row)
    result = {"all": aggregate(records), "by_k": {}}
    for count, subset in sorted(by_count.items()):
        part = aggregate(subset)
        by_shift = {}
        for shift in range(count):
            views = [row["views"][shift] for row in subset]
            by_shift[str(shift)] = {
                "n": len(views), "accuracy": sum(view["correct"] for view in views) / len(views),
                "predicted_letter_counts": dict(Counter(view["predicted_letter"] for view in views)),
                "gold_letter_counts": dict(Counter(view["gold_letter"] for view in views)),
            }
        worst = min(value["accuracy"] for value in by_shift.values())
        worst_nonzero = min(by_shift[str(shift)]["accuracy"] for shift in range(1, count))
        part.update(by_shift=by_shift, worst_shift_accuracy=worst,
                    worst_shifts=[int(key) for key, value in by_shift.items() if value["accuracy"] == worst],
                    worst_nonzero_shift_accuracy=worst_nonzero,
                    worst_nonzero_minus_original=worst_nonzero - part["original_accuracy"])
        result["by_k"][str(count)] = part
    # Provenance remains on every QA; these summaries localize changed sources.
    for field in ("source", "source_benchmark", "question_type", "task"):
        grouped = defaultdict(list)
        for row in records:
            if field in row:
                grouped[str(row[field])].append(row)
        if grouped:
            result[f"by_{field}"] = {key: aggregate(subset) for key, subset in sorted(grouped.items())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base Qwen3-VL checkpoint")
    parser.add_argument("--adapter", help="Optional LoRA adapter; base remains explicit")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    rows, selection = read_rows(args.manifest, args.limit)
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    # Qwen3.5: Auto class resolves Qwen3_5ForConditionalGeneration.
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.to(args.device).eval()
    records = score_all_rotations(rows, processor, model, args.batch_size)
    report = {
        "model": args.model, "adapter": args.adapter, "manifest": str(args.manifest.resolve()),
        "selection": selection,
        "protocol": {"rotation": "left cyclic shifts 0..K-1; K views, not all K! permutations",
                     "readout": "native_candidate_logits; original letter argmax then content mapping",
                     "prediction_aggregation": "none; each view is an independent decision",
                     "overall_weight": "primary means weight original QA equally; view-weighted mean also reported",
                     "batch_membership": "group by K; same QA batch at every shift",
                     "precision": args.dtype, "device": args.device, "batch_size": args.batch_size,
                     "min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
        **summarize(records),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "rotation_predictions.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output / "rotation_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"all": report["all"], "by_k": report["by_k"]}, indent=2))


if __name__ == "__main__":
    main()
