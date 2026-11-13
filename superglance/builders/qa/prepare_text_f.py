#!/usr/bin/env python3
"""Text-only data for control F: train sources, held-out retention evals, and the F pool.

Every text row carries the same small blank image so text and image rows share one code path
(an image-free micro-batch would skip the vision tower on some ranks). Steps:
  evals     write retention-eval manifests (never trained on): MMLU, MMLU-Pro, ARC-C/E, HellaSwag,
            WinoGrande, BoolQ, TruthfulQA-MC1
  train     convert train sources (choice + yes/no), derive "is this answer correct?" yes/no items,
            drop items whose question overlaps any retention eval (exact or >=50% of 10-word shingles)
  assemble  F pool = E pool rows + text rows; dev and image store symlinked unchanged
"""
import argparse
import ast
import gzip
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

BLANK = "blank.png"


def read(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    if path.name.endswith(".jsonl.gz"):
        return [json.loads(l) for l in gzip.open(path, "rt")]
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in open(path)]
    return json.load(open(path))


def seq(x):
    """Parquet list columns arrive as numpy arrays; some datasets store lists as strings."""
    if isinstance(x, str):
        try:
            return list(ast.literal_eval(x))
        except (ValueError, SyntaxError):
            return [x]
    return list(x)


REFERS_TO_MISSING_IMAGE = re.compile(r"\b(shown below|shown in the (image|figure|picture|photo)|(given|following|above|below) (image|figure|picture|photo|diagram|x-ray|radiograph|ecg)|in the (image|figure|picture|photo|diagram))\b", re.I)


def item(source, idx, question, options, gold, pattern="Choice"):
    options = [str(o).strip() for o in options]
    if source.startswith("text_") and REFERS_TO_MISSING_IMAGE.search(str(question)):
        return None  # text rows carry a blank image; questions about a missing picture would teach guessing
    if not question or len(options) < 2 or not 0 <= gold < len(options) or len(set(options)) != len(options) or any(not o for o in options):
        return None
    return {"source": source, "idx": str(idx), "question": str(question).strip(), "options": options, "gold": int(gold), "pattern": pattern}


# ---------- retention evals ----------
def evals(raw):
    out = {}
    out["mmlu"] = [item("mmlu", r["subject"] + str(i), r["question"], seq(r["choices"]), int(r["answer"]))
                   for i, r in enumerate(read(raw / "cais/mmlu/all/test-00000-of-00001.parquet"))]
    out["mmlu_pro"] = [item("mmlu_pro", r["question_id"], r["question"], seq(r["options"]), int(r["answer_index"]))
                       for r in read(raw / "TIGER-Lab/MMLU-Pro/data/test-00000-of-00001.parquet")]
    for name in ("ARC-Challenge", "ARC-Easy"):
        rows = read(raw / f"allenai/ai2_arc/{name}/test-00000-of-00001.parquet")
        out[name.lower().replace("-", "_")] = [item(name, r["id"], r["question"], seq(r["choices"]["text"]),
                                                    seq(r["choices"]["label"]).index(r["answerKey"])) for r in rows]
    out["hellaswag"] = [item("hellaswag", r["ind"], r["ctx"] + "\nWhich ending is the most plausible continuation?",
                             seq(r["endings"]), int(r["label"])) for r in read(raw / "Rowan/hellaswag/data/validation-00000-of-00001.parquet")]
    out["winogrande"] = [item("winogrande", i, r["sentence"] + "\nWhich option correctly fills the blank (_)?",
                              [r["option1"], r["option2"]], int(r["answer"]) - 1)
                         for i, r in enumerate(read(raw / "allenai/winogrande/winogrande_xl/validation-00000-of-00001.parquet"))]
    out["boolq"] = [item("boolq", i, r["passage"] + "\nQuestion: " + r["question"].rstrip("?") + "?", ["yes", "no"],
                         0 if str(r["answer"]).lower() == "true" else 1, "Noul")
                    for i, r in enumerate(read(raw / "google/boolq/data/validation-00000-of-00001.parquet"))]
    out["truthfulqa_mc1"] = [item("truthfulqa", i, r["question"], seq(r["mc1_targets"]["choices"]),
                                  seq(r["mc1_targets"]["labels"]).index(1))
                             for i, r in enumerate(read(raw / "truthfulqa/truthful_qa/multiple_choice/validation-00000-of-00001.parquet"))]
    return {k: [x for x in v if x] for k, v in out.items()}


# ---------- train sources ----------
def train_sources(raw):
    s = {}
    s["text_csqa"] = [item("text_csqa", r["id"], r["question"], seq(r["choices"]["text"]), seq(r["choices"]["label"]).index(r["answerKey"]))
                      for r in read(raw / "tau/commonsense_qa/data/train-00000-of-00001.parquet")]
    s["text_siqa"] = [item("text_siqa", i, r["context"] + "\n" + r["question"], [r["answerA"], r["answerB"], r["answerC"]], int(r["label"]) - 1)
                      for i, r in enumerate(read(raw / "allenai/social_i_qa/default/train-00000.parquet"))]
    s["text_cosmosqa"] = [item("text_cosmosqa", r["id"], r["context"] + "\n" + r["question"], [r[f"answer{k}"] for k in range(4)], int(r["label"]))
                          for r in read(raw / "allenai/cosmos_qa/default/train-00000.parquet")]
    s["text_piqa"] = [item("text_piqa", i, "Goal: " + r["goal"] + "\nWhich solution achieves the goal?", [r["sol1"], r["sol2"]], int(r["label"]))
                      for i, r in enumerate(read(raw / "ybisk/piqa/plain_text/train-00000.parquet"))]
    s["text_medmcqa"] = [item("text_medmcqa", r["id"], r["question"], [r["opa"], r["opb"], r["opc"], r["opd"]], int(r["cop"]))
                         for r in read(raw / "openlifescienceai/medmcqa/data/train-00000-of-00001.parquet") if r["choice_type"] == "single"]
    s["text_medqa"] = [item("text_medqa", i, r["question"], [r["options"][k] for k in "ABCD"], "ABCD".index(r["answer_idx"]))
                       for i, r in enumerate(read(raw / "GBaker/MedQA-USMLE-4-options/phrases_no_exclude_train.jsonl"))]
    s["text_qasc"] = [item("text_qasc", r["id"], r["question"], seq(r["choices"]["text"]), seq(r["choices"]["label"]).index(r["answerKey"]))
                      for r in read(raw / "allenai/qasc/data/train-00000-of-00001.parquet")]
    s["text_obqa"] = [item("text_obqa", r["id"], r["question_stem"], seq(r["choices"]["text"]), seq(r["choices"]["label"]).index(r["answerKey"]))
                      for r in read(raw / "allenai/openbookqa/main/train-00000-of-00001.parquet")]
    s["text_aqua"] = [item("text_aqua", i, r["question"], [re.sub(r"^[A-E]\)\s*", "", o) for o in seq(r["options"])], "ABCDE".index(r["correct"]))
                      for i, r in enumerate(read(raw / "deepmind/aqua_rat/raw/train-00000-of-00001.parquet"))]
    labels = ["entailment", "neutral", "contradiction"]
    s["text_wanli"] = [item("text_wanli", r["id"], f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}\nHow does the hypothesis relate to the premise?",
                            labels, labels.index(r["gold"])) for r in read(raw / "alisawuffles/WANLI/train.jsonl")]
    s["text_art"] = [item("text_art", i, f"Observation 1: {r['observation_1']}\nObservation 2: {r['observation_2']}\nWhich hypothesis best explains what happened in between?",
                          [r["hypothesis_1"], r["hypothesis_2"]], int(r["label"]) - 1)
                     for i, r in enumerate(read(raw / "allenai/art/anli/train-00000-of-00001.parquet"))]
    s["text_strategyqa"] = [item("text_strategyqa", r["qid"], r["question"], ["yes", "no"], 0 if r["answer"] else 1, "Noul")
                            for r in read(raw / "tasksource/strategy-qa/strategyQA_train.json")]
    s["text_ruletaker"] = [item("text_ruletaker", i, r["context"] + "\nBased only on the facts and rules above, is this statement true?\n" + r["question"],
                                ["yes", "no"], 0 if r["label"] == "entailment" else 1, "Noul")
                           for i, r in enumerate(read(raw / "tasksource/ruletaker/default/train-00000.parquet"))]
    return {k: [x for x in v if x] for k, v in s.items()}


def derive_yesno(sources, n, rng):
    """'Is this answer correct?' from the choice buckets, half with the gold option and half with a wrong one."""
    pool = [x for k in ("text_csqa", "text_siqa", "text_cosmosqa", "text_piqa", "text_medmcqa", "text_medqa", "text_qasc", "text_obqa", "text_aqua")
            for x in sources[k]]
    out = []
    for j, x in enumerate(rng.sample(pool, min(n, len(pool)))):
        correct = j % 2 == 0
        shown = x["options"][x["gold"]] if correct else rng.choice([o for i, o in enumerate(x["options"]) if i != x["gold"]])
        out.append(item("text_yesno", f"{x['source']}:{x['idx']}", f"{x['question']}\nProposed answer: {shown}\nIs the proposed answer correct?",
                        ["yes", "no"], 0 if correct else 1, "Noul"))
    return out


# ---------- decontamination ----------
def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(text).lower())).strip()


def shingles(text, n=10):
    words = norm(text).split()
    return {hash(" ".join(words[i:i + n])) for i in range(max(0, len(words) - n + 1))}


def decontaminate(items, eval_items):
    exact = {norm(x["question"]) for x in eval_items}
    grams = set().union(*(shingles(x["question"]) for x in eval_items))
    kept, dropped = [], 0
    for x in items:
        sh = shingles(x["question"])
        if norm(x["question"]) in exact or (sh and len(sh & grams) / len(sh) >= 0.5):
            dropped += 1
        else:
            kept.append(x)
    return kept, dropped


# ---------- rows ----------
def to_row(x, rng, image):
    perm = list(range(len(x["options"])))
    rng.shuffle(perm)
    shown = [x["options"][p] for p in perm]
    row = {"id": f"{x['source']}_{x['idx']}", "question": x["question"], "answers": shown, "gold_index": perm.index(x["gold"]),
           "source": x["source"], "source_split": "train", "source_answers": x["options"], "source_gold_index": x["gold"],
           "candidate_permutation": perm, "correct_answer": x["options"][x["gold"]], "label_source": "dataset_gt",
           "pattern": x["pattern"], "question_type": x["source"], "sampling_task": x["source"],
           "sampling_bucket": x["options"][x["gold"]] if x["pattern"] == "Noul" else "choice",
           "images": [image], "image_id": "text_blank", "image_group": f"text_{x['source']}_{x['idx']}", "pool_split": "train"}
    if x["pattern"] == "Noul":
        row["yes_index"] = shown.index("yes")
    return row


def write_jsonl(path, rows):
    with open(path, "w") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["evals", "train", "assemble"])
    p.add_argument("--root", type=Path, required=True, help="text_f directory (raw/ inside)")
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--derived-yesno", type=int, default=40000)
    p.add_argument("--cap", type=int, default=40000, help="max rows per train source; draws are set by source weights")
    p.add_argument("--base-pool", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    rng, raw = random.Random(args.seed), args.root / "raw"
    if not (args.root / BLANK).exists():
        Image.new("RGB", (64, 64), (128, 128, 128)).save(args.root / BLANK)
    if args.step == "evals":
        out = args.root / "retention_evals"
        out.mkdir(exist_ok=True)
        os.path.lexists(out / BLANK) or os.symlink(args.root / BLANK, out / BLANK)
        report = {}
        for name, items in evals(raw).items():
            rows = [dict(to_row(x, rng, BLANK), source_split="eval", question_type=name) for x in items]
            write_jsonl(out / f"{name}.jsonl", rows)
            report[name] = len(rows)
        print(json.dumps(report))
    elif args.step == "train":
        eval_items = [x for v in evals(raw).values() for x in v]
        sources = train_sources(raw)
        sources["text_yesno"] = derive_yesno(sources, args.derived_yesno, rng)
        rows, report = [], {}
        for name, items in sources.items():
            kept, dropped = decontaminate(items, eval_items)
            converted_kept = len(kept)
            if len(kept) > args.cap:
                kept = rng.sample(kept, args.cap)
            rows += [to_row(x, rng, "external_text/" + BLANK) for x in kept]
            report[name] = {"converted": len(items), "dropped_overlap": dropped, "after_decontam": converted_kept, "kept": len(kept)}
        write_jsonl(args.root / "text_train.jsonl", rows)
        (args.root / "text_train_report.json").write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report))
    else:
        args.output.mkdir()  # never overwrite an existing pool
        with open(args.output / "train.jsonl", "w") as dst:
            for path in (args.base_pool / "train.jsonl", args.root / "text_train.jsonl"):
                with open(path) as src:
                    for line in src:
                        dst.write(line)
        for name in ("dev.jsonl", "image_store.sqlite", "images", "external"):
            os.symlink(os.path.realpath(args.base_pool / name), args.output / name)
        os.symlink(args.root, args.output / "external_text")
        counts = Counter(json.loads(l)["source"] for l in open(args.root / "text_train.jsonl"))
        (args.output / "text_report.json").write_text(json.dumps({"base_pool": str(args.base_pool), "text_rows": dict(counts)}, indent=1) + "\n")
        print(json.dumps(dict(counts)))


if __name__ == "__main__":
    main()
