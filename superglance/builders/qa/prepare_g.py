#!/usr/bin/env python3
"""Data for step g (the f pool + two targeted fixes): "directly above / underneath" pairs and "none of the other options".

1. Vertical relations. The f-pool model still answers "yes" to "Is X directly above Y?" whenever X is higher in the image
   (3DSR location_above: 837 "yes" against 600). From upright OpenImages images, objects whose class occurs
   once (body parts and clothing-in-general excluded):
     yes     stacked pairs (no vertical overlap, gap <= 8% of the height, horizontal overlap >= 50% of the
             narrower box) whose classes physically rest on one another (SUPPORT: cup on saucer, helmet of a
             rider over a motorcycle, plant in a pot, upper over lower garment)
     offset  X higher than Y but horizontally apart by >= 5% of the width -> no  (val/test images)
     swap    a "yes" pair asked with the objects swapped, same wording -> no
   Labels from stacked 2D boxes alone, or with monocular depth, were rejected after visual checks (~60%
   correct: stacked often means "behind", e.g. a window over a table); SUPPORT pairs avoid that.
2. Hedge answers. TruthfulQA errors concentrate on questions whose correct answer is a hedge or negation.
   From text multiple-choice rows (>= 3 content options; WANLI/aNLI/PIQA excluded) half get the gold option
   replaced by "None of the other options is correct" (now the answer), half get it as an extra distractor.
Steps:
  select    scan annotations -> vertical_g/candidates.jsonl (images with a SUPPORT pair, or an offset pair)
  fetch     download missing images into openimages/images (shared with E)
  dedup     perceptual-hash candidate images against the eval images -> vertical_g/dedup.json
  generate  vertical_g/train.jsonl (50/50 yes/no)
  nota      vertical_g/text_nota.jsonl from the base pool's text rows
  assemble  new pool = --base-pool + both files (dev, images, image store, external links unchanged)
"""
import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from superglance.builders.qa import prepare_external_e as ext

PREFIXES = ["", "Consider the real-world 3D locations of the objects. ", "In the real world, "]
ABOVE = ["Is the {a} directly above the {b}?", "Is the {a} located directly over the {b}?"]
BELOW = ["Is the {b} directly underneath the {a}?", "Is the {b} directly below the {a}?"]
EXCLUDED = ("clothing", "footwear", "fashion accessory")  # plus every "human ..." body part
SUPPORT = [  # (upper classes, lower classes): the upper object rests on, or is carried right over, the lower one
    ({"coffee", "tea", "coffee cup", "mug", "cup"}, {"saucer", "plate"}),
    ({"helmet", "bicycle helmet", "football helmet"}, {"motorcycle", "bicycle", "horse", "skateboard", "scooter", "snowboard", "surfboard"}),
    ({"person", "man", "woman", "boy", "girl"}, {"horse", "bicycle", "motorcycle", "skateboard", "surfboard", "snowboard", "elephant", "camel", "scooter"}),
    ({"shirt", "jacket", "coat"}, {"jeans", "trousers", "shorts", "skirt"}),
    ({"houseplant", "plant", "flower", "rose"}, {"flowerpot", "vase"}),
]  # objects on furniture were dropped: 2 of 3 wrong in a 16-pair check (cup on a shelf behind a desk, book on a bookcase behind drawers)
SUPPORT_CLASSES = set().union(*(u | l for u, l in SUPPORT))
NOTA = "None of the other options is correct"
NOTA_SKIP = ("text_wanli", "text_art", "text_piqa")  # fixed label sets or two options
HEDGES = ("none of", "all of the above", "not enough information", "cannot be determined")  # rows that already have one


def supported(upper, lower):
    return any(upper in u and lower in l for u, l in SUPPORT)


def boxes(root):
    """Filtered boxes per upright image; train images keep only SUPPORT classes (enough for yes/swap rows)."""
    names = {r[0]: r[1].lower() for r in csv.reader(open(root / "class-descriptions-boxable.csv"))}
    missing = sorted(SUPPORT_CLASSES - set(names.values()))
    splits = [s for s in ("validation", "test", "train") if (root / f"{s}-annotations-bbox.csv").exists()]
    upright, split_of = set(), {}
    for split in splits:
        for r in csv.DictReader(open(root / f"{split}-images-with-rotation.csv")):
            if r["Rotation"] in ("", "0", "0.0"):
                upright.add(r["ImageID"])
                split_of[r["ImageID"]] = split
    out = defaultdict(list)
    for split in splits:
        for r in csv.DictReader(open(root / f"{split}-annotations-bbox.csv")):
            name = names.get(r["LabelName"], "")
            if split == "train" and name not in SUPPORT_CLASSES:
                continue
            if (r["ImageID"] in upright and r["IsGroupOf"] == "0" and r["IsDepiction"] == "0" and r["IsInside"] == "0"
                    and name and not name.startswith("human ") and name not in EXCLUDED):
                b = [float(r[k]) for k in ("XMin", "YMin", "XMax", "YMax")]
                if (b[2] - b[0]) * (b[3] - b[1]) >= 0.005:
                    out[r["ImageID"]].append((name, b))
    return out, split_of, missing


def pairs(objs):
    """(kind, upper, lower, upper box, lower box) for unique-class pairs where one object is higher."""
    count = Counter(n for n, _ in objs)
    objs = [(n, b) for n, b in objs if n and count[n] == 1]
    for a_name, a in objs:
        for b_name, b in objs:
            if a_name == b_name or a[3] > b[1] + 0.02:  # a must end (bottom) at or above b's top
                continue
            overlap = min(a[2], b[2]) - max(a[0], b[0])
            if b[1] - a[3] <= 0.08 and overlap >= 0.5 * min(a[2] - a[0], b[2] - b[0]):
                yield "stacked", a_name, b_name, a, b
            elif -overlap >= 0.05:
                yield "offset", a_name, b_name, a, b


def question(rng, above, upper, lower):
    q = rng.choice(ABOVE if above else BELOW).format(a=upper, b=lower)
    prefix = rng.choice(PREFIXES)
    return prefix + (q[0].lower() + q[1:] if prefix.endswith(", ") else q)


def cmd_select(args):
    rng = random.Random(args.seed)
    by_image, split_of, missing = boxes(args.root / "openimages")
    support, offset_only = [], []
    for image_id in sorted(by_image):
        found = defaultdict(list)
        for kind, upper, lower, a, b in pairs(by_image[image_id]):
            if kind == "offset" or supported(upper, lower):
                found[kind].append([upper, lower, a, b])
        row = {"image_id": image_id, "split": split_of[image_id], "stacked": found["stacked"], "offset": found["offset"]}
        if found["stacked"]:
            support.append(row)
        elif found["offset"] and split_of[image_id] != "train":
            offset_only.append(row)
    rng.shuffle(support)
    rng.shuffle(offset_only)
    rows = support[:args.max_images] + offset_only[:int(0.6 * min(len(support), args.max_images))]
    out = args.root / "vertical_g"
    out.mkdir(exist_ok=True)
    ext.jsonl(out / "candidates.jsonl", rows)
    print(json.dumps({"support_images": len(support), "offset_only_images": len(offset_only), "kept": len(rows),
                      "missing_class_names": missing,
                      "top_support_pairs": Counter((p[0], p[1]) for r in support for p in r["stacked"]).most_common(20)}))


def cmd_fetch(args):
    ext.cache_dns()
    images = args.root / "openimages/images"
    jobs = [(ext.S3.format(split=r["split"], id=r["image_id"]), images / f"{r['image_id']}.jpg")
            for r in ext.read_jsonl(args.root / "vertical_g/candidates.jsonl")]
    with ThreadPoolExecutor(args.threads) as pool:
        results = list(pool.map(lambda j: ext.fetch_one(*j), jobs))
    print(json.dumps({"jobs": len(jobs), "status": Counter(r if r in ("ok", "exists") else "failed" for r in results)}))


def cmd_dedup(args):
    images = args.root / "openimages/images"
    paths = [images / f"{r['image_id']}.jpg" for r in ext.read_jsonl(args.root / "vertical_g/candidates.jsonl")]
    paths = [p for p in paths if p.exists()]
    evals = [p for d in args.eval_dirs for p in Path(d).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    with ThreadPoolExecutor(args.threads) as pool:
        ref = list(pool.map(ext.phash, evals))
        new = list(pool.map(ext.phash, paths))
    dist = ext.hamming_min(new, ref)
    drop = sorted(p.stem for p, d in zip(paths, dist) if d <= args.threshold)
    (args.root / "vertical_g/dedup.json").write_text(json.dumps({"threshold": args.threshold, "checked": len(paths),
                                                                  "eval_images": len(evals), "drop": drop}) + "\n")
    print(json.dumps({"checked": len(paths), "eval_images": len(evals), "dropped": len(drop)}))


def cmd_generate(args):
    rng = random.Random(args.seed)
    out = args.root / "vertical_g"
    drops = set(json.load(open(out / "dedup.json"))["drop"])
    yes, offset, swap = [], [], []
    for c in ext.read_jsonl(out / "candidates.jsonl"):
        if c["image_id"] in drops or not (args.root / "openimages/images" / f"{c['image_id']}.jpg").exists():
            continue
        if c["stacked"]:
            upper, lower = rng.choice(c["stacked"])[:2]
            above = rng.random() < 0.5
            yes.append((c, question(rng, above, upper, lower)))
            swap.append((c, question(rng, above, lower, upper)))
        elif c["offset"]:
            upper, lower = rng.choice(c["offset"])[:2]
            offset.append((c, question(rng, rng.random() < 0.5, upper, lower)))
    rng.shuffle(offset)
    rng.shuffle(swap)
    n_offset = min(len(offset), int(0.6 * len(yes)))
    chosen = {"yes": yes, "offset": offset[:n_offset], "swap": swap[:len(yes) - n_offset]}
    rows = [ext.make_row(f"oi_vertical_{c['image_id']}_{kind}", q, ["yes", "no"], 0 if kind == "yes" else 1,
                         "openimages_vertical", f"vertical/{kind}", f"external/openimages/images/{c['image_id']}.jpg",
                         rng, "rule_2d_vertical_support", source_image_id=c["image_id"], source_image_split=c["split"])
            for kind, items in chosen.items() for c, q in items]
    ext.jsonl(out / "train.jsonl", rows)
    report = {"rows": len(rows), "by_kind": {k: len(v) for k, v in chosen.items()},
              "examples": {k: [q for _, q in v[:5]] for k, v in chosen.items()},
              "label_source": "rule_2d_vertical_support (OpenImages boxes, SUPPORT class pairs)"}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(report, ensure_ascii=False))


def nota_row(r, rng, kind):
    answers, gold = list(r["source_answers"]), r["source_gold_index"]
    if kind == "nota_correct":  # gold removed: the hedge option is the answer
        answers = [a for i, a in enumerate(answers) if i != gold] + [NOTA]
        gold = len(answers) - 1
    else:  # gold kept: the hedge option is one more distractor
        answers = answers + [NOTA]
    perm = list(range(len(answers)))
    rng.shuffle(perm)
    return dict(r, id=f"{r['id']}_{kind}", answers=[answers[p] for p in perm], gold_index=perm.index(gold),
                source="text_nota", source_answers=answers, source_gold_index=gold, candidate_permutation=perm,
                correct_answer=answers[gold], label_source="rule_none_of_the_other_options", question_type=f"text_nota/{kind}",
                sampling_task=f"text_nota/{kind}", sampling_bucket="choice", image_group=f"text_{r['id']}_{kind}", nota_of=r["id"])


def cmd_nota(args):
    rng = random.Random(args.seed)
    rows = [r for r in ext.read_jsonl(args.base_pool / "train.jsonl")
            if r["source"].startswith("text_") and r["source"] not in NOTA_SKIP and r.get("pattern") == "Choice"
            and len(r["answers"]) >= 3 and not any(h in a.lower() for a in r["answers"] for h in HEDGES)]
    rng.shuffle(rows)
    out = [nota_row(r, rng, "nota_correct" if n % 2 == 0 else "nota_distractor") for n, r in enumerate(rows[:args.nota])]
    ext.jsonl(args.root / "vertical_g/text_nota.jsonl", out)
    print(json.dumps({"eligible": len(rows), "written": len(out), "by_source": dict(Counter(r["question_type"] for r in rows[:args.nota])),
                      "example": {k: out[0][k] for k in ("question", "answers", "correct_answer")}}, ensure_ascii=False))


def cmd_assemble(args):
    out = args.output
    out.mkdir()  # never overwrite an existing pool
    added = [args.root / "vertical_g/train.jsonl", args.root / "vertical_g/text_nota.jsonl"]
    with open(out / "train.jsonl", "w") as dst:
        for src in [args.base_pool / "train.jsonl"] + added:
            with open(src) as f:
                shutil.copyfileobj(f, dst)
    for entry in args.base_pool.iterdir():
        if entry.suffix != ".json" and entry.name != "train.jsonl":
            os.symlink(os.path.realpath(entry), out / entry.name)
    print(json.dumps({"base_pool": str(args.base_pool), "added": {p.name: sum(1 for _ in open(p)) for p in added}}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["select", "fetch", "dedup", "generate", "nota", "assemble"])
    p.add_argument("--root", type=Path, required=True, help="E external data directory (openimages/, dedup.json)")
    p.add_argument("--max-images", type=int, default=15000, help="cap on images with a SUPPORT pair")
    p.add_argument("--threads", type=int, default=48)
    p.add_argument("--eval-dirs", nargs="*", default=[])
    p.add_argument("--threshold", type=int, default=10)
    p.add_argument("--nota", type=int, default=40000, help="text rows with the hedge option (half correct, half distractor)")
    p.add_argument("--base-pool", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--seed", type=int, default=20260926)
    args = p.parse_args()
    globals()[f"cmd_{args.step}"](args)


if __name__ == "__main__":
    main()
