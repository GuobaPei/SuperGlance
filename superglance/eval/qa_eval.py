#!/usr/bin/env python3
"""Single-GPU option-letter evaluation of one model on one manifest; saves every probability for later analysis.
Out: <output>/predictions.jsonl and metrics.json (accuracy, NLL, Brier, ECE); --circular also scores the binary questions with
their options reversed (circular_predictions.jsonl, metrics.json circular_binary).
usage: python -m superglance.eval.qa_eval --model BASE [--adapter ADAPTER] --manifest M.jsonl --output DIR [--batch-size 8]
       [--max-pixels 262144] [--circular]"""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText

from superglance.data.qa import DecisionCollator, DecisionDataset, candidate_token_ids, native_candidate_logits
from superglance.model.prompt import validate_answer_boundary


def metrics(records):
    n = len(records)
    correct, nll, brier = 0, 0., 0.
    bins = [[] for _ in range(15)]
    for row in records:
        p, y = row["probabilities"], row["gold_index"]
        hit = int(row["predicted_index"] == y)
        correct += hit
        nll -= math.log(max(p[y], 1e-12))
        brier += sum((value - int(i == y)) ** 2 for i, value in enumerate(p))
        confidence = max(p)
        bins[min(int(confidence * 15), 14)].append((confidence, hit))
    ece = sum(abs(sum(c - hit for c, hit in bucket)) for bucket in bins) / n
    return {"n": n, "accuracy": correct / n, "nll": nll / n, "brier_multiclass_sum": brier / n, "ece_15_equal_width": ece}


def score_rows(rows, processor, model, token_ids, batch_size):
    """Score each row once, retaining candidate probabilities and source labels."""
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=DecisionCollator(processor))
    records = []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            counts = batch.pop("candidate_count").tolist()
            batch.pop("gold_index")
            batch = {key: value.to(model.device) for key, value in batch.items()}
            logits = native_candidate_logits(model, batch, token_ids)
            for i, count in enumerate(counts):
                row = rows[offset + i]
                p = logits[i, :count].softmax(-1).cpu().tolist()
                record = {key: row[key] for key in ("id", "image_id", "pattern", "question_type", "gold_index")}
                record.update(probabilities=p, predicted_index=max(range(count), key=p.__getitem__))
                for key in ("source", "source_split", "source_id", "source_benchmark", "task", "circular_order", "circular_original_id"):
                    if key in row:
                        record[key] = row[key]
                if "yes_index" in row:
                    record["p_yes"] = p[row["yes_index"]]
                records.append(record)
            offset += len(counts)
            print(f"Evaluated {offset}/{len(rows)}", flush=True)
    return records


def reverse_binary_rows(rows):
    """Reverse only binary options and remap both GT and semantic yes index."""
    reversed_rows = []
    for row in rows:
        if len(row["answers"]) != 2:
            continue
        reversed_row = dict(row)
        reversed_row.update(answers=list(reversed(row["answers"])),
                            gold_index=1 - row["gold_index"],
                            circular_order="reversed", circular_original_id=row["id"])
        if "yes_index" in row:
            reversed_row["yes_index"] = 1 - row["yes_index"]
        reversed_rows.append(reversed_row)
    return reversed_rows


def circular_records(rows, original_records, processor, model, token_ids, batch_size):
    """Reuse original predictions; only reversed binary rows need another pass."""
    reversed_rows = reverse_binary_rows(rows)
    reversed_records = score_rows(reversed_rows, processor, model, token_ids, batch_size)
    originals = {row["id"]: row for row in original_records}
    records = []
    for reversed_record in reversed_records:
        original = dict(originals[reversed_record["id"]], circular_order="original",
                        circular_original_id=reversed_record["id"])
        records.extend([original, reversed_record])
    return records


def checkpoint_paths(model_path, adapter_path=None):
    """Explicit base+adapter paths work even after moving an experiment."""
    if adapter_path is None and (Path(model_path) / "adapter_config.json").is_file():
        from peft import PeftConfig
        adapter_path = model_path
        model_path = PeftConfig.from_pretrained(adapter_path).base_model_name_or_path
    return model_path, adapter_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base checkpoint; a saved adapter directory is also accepted")
    parser.add_argument("--adapter", help="Explicit trained LoRA directory; --model supplies its base checkpoint")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--circular", action="store_true", help="Also score original/reversed binary options")
    args = parser.parse_args()
    base_path, adapter_path = checkpoint_paths(args.model, args.adapter)
    processor = AutoProcessor.from_pretrained(base_path, min_pixels=65536, max_pixels=args.max_pixels)
    # q35 port: Auto class resolves Qwen3_5ForConditionalGeneration (and Qwen3VL for the env check).
    model = AutoModelForImageTextToText.from_pretrained(base_path, dtype=torch.bfloat16, attn_implementation="sdpa")
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model = model.to("cuda").eval()
    data = DecisionDataset(args.manifest, args.limit)
    max_options = max(len(row["answers"]) for row in data.rows)
    validate_answer_boundary(processor, max_options)
    ids = candidate_token_ids(processor, max_options)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [data[index] for index in range(len(data))]
    torch.cuda.synchronize()
    started = time.perf_counter()
    records = score_rows(rows, processor, model, ids, args.batch_size)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    with (args.output / "predictions.jsonl").open("w") as out:
        for record in records:
            out.write(json.dumps(record) + "\n")
    grouped, types = defaultdict(list), defaultdict(list)
    for row in records:
        grouped[row["pattern"]].append(row)
        types[row["question_type"]].append(row)
    result = {"model": base_path, "adapter": adapter_path, "manifest": str(args.manifest),
              "readout": "native_candidate_logits", "max_pixels": args.max_pixels,
              "inference_timing": {"seconds": elapsed, "questions_per_second": len(records) / elapsed,
                                   "batch_size": args.batch_size, "gpu": torch.cuda.get_device_name(),
                                   "scope": "main pass including image I/O and preprocessing; model loading and circular pass excluded; no warmup excluded"},
              "all": metrics(records), "by_pattern": {key: metrics(rows) for key, rows in grouped.items()},
              "by_question_type": {key: metrics(rows) for key, rows in types.items()}}
    if args.circular:
        circular = circular_records(rows, records, processor, model, ids, args.batch_size)
        circular_by_order = defaultdict(list)
        for row in circular:
            circular_by_order[row["circular_order"]].append(row)
        result["circular_binary"] = {
            "protocol": "ordinary mean over original and reversed option rows; gold and yes index remapped",
            "n_original_questions": len(circular) // 2,
            "n_scored_rows": len(circular),
            "all": metrics(circular),
            "by_order": {key: metrics(rows) for key, rows in circular_by_order.items()},
        }
        with (args.output / "circular_predictions.jsonl").open("w") as stream:
            for row in circular:
                stream.write(json.dumps(row) + "\n")
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
