#!/usr/bin/env python3
"""External data for control E: box-choice localization and SpatialSense relations.

All sources are non-COCO. Steps (run in order, each writes under --root):
  openimages   select BLINK-style box-choice items from OpenImages V5 val/test boxes
  fetch        download selected OpenImages images from the public S3 bucket, resized
  render       draw the two candidate boxes (labels A/B) on each box-choice image
  spatialsense convert SpatialSense Flickr train+valid relations to yes/no questions
  dedup        perceptual-hash every new image against all eval images, record drops
  assemble     write the E pool: flip pool rows + new rows, dev and image store symlinked
"""
import argparse
import csv
import io
import json
import os
import random
import re
import tarfile
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MAX_SIDE = 1024
S3 = "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg"


def jsonl(path, rows):
    with open(path, "w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


# ---------- OpenImages box choice ----------
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def jitter(box, rng):
    """Distractor near the GT box: gaussian noise, or a loose / tight rescale around the center."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    kind = rng.choices(["noise", "loose", "tight"], [0.7, 0.15, 0.15])[0]
    if kind == "noise":
        s = rng.uniform(0.1, 0.3)
        new = [x1 + rng.gauss(0, s * w), y1 + rng.gauss(0, s * h), x2 + rng.gauss(0, s * w), y2 + rng.gauss(0, s * h)]
    else:
        f = rng.uniform(1.2, 1.6) if kind == "loose" else rng.uniform(0.5, 0.8)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        new = [cx - f * w / 2, cy - f * h / 2, cx + f * w / 2, cy + f * h / 2]
    new = [min(max(v, 0.0), 1.0) for v in new]
    new = [min(new[0], new[2]), min(new[1], new[3]), max(new[0], new[2]), max(new[1], new[3])]
    return new, kind


def cmd_openimages(args):
    rng = random.Random(args.seed)
    root = args.root / "openimages"
    names = {r[0]: r[1] for r in csv.reader(open(root / "class-descriptions-boxable.csv"))}
    upright = set()
    for split in ("validation", "test"):
        for r in csv.DictReader(open(root / f"{split}-images-with-rotation.csv")):
            if r["Rotation"] in ("", "0", "0.0"):
                upright.add(r["ImageID"])
    boxes = defaultdict(list)
    for split in ("validation", "test"):
        for r in csv.DictReader(open(root / f"{split}-annotations-bbox.csv")):
            if r["ImageID"] in upright:
                boxes[(split, r["ImageID"])].append(r)
    keys = sorted(boxes)
    rng.shuffle(keys)
    have = {p.stem for p in (root / "images").glob("*.jpg")}
    keys.sort(key=lambda k: k[1] not in have)  # stable: reuse already-downloaded images first
    items, need = [], {"tight": args.tight, "easy": args.easy}
    for split, image_id in keys:
        if not any(need.values()):
            break
        rows = boxes[(split, image_id)]
        box = lambda r: [float(r["XMin"]), float(r["YMin"]), float(r["XMax"]), float(r["YMax"])]
        label_count = Counter(r["LabelName"] for r in rows)
        # "the {name}" must be unambiguous: the target class appears exactly once in the image
        ok = [r for r in rows if label_count[r["LabelName"]] == 1 and r["IsGroupOf"] == "0" and r["IsDepiction"] == "0"
              and r["IsInside"] == "0" and not names[r["LabelName"]].startswith("Human ")
              and 0.01 <= (float(r["XMax"]) - float(r["XMin"])) * (float(r["YMax"]) - float(r["YMin"])) <= 0.5]
        if not ok:
            continue
        target = rng.choice(ok)
        gt = box(target)
        same = [box(r) for r in rows if r["LabelName"] == target["LabelName"] and r is not target]
        want = "easy" if need["easy"] and rng.random() < args.easy / (args.tight + args.easy) else "tight"
        if not need[want]:
            want = "tight" if need["tight"] else "easy"
        distractor = kind = None
        if want == "tight":
            for _ in range(30):
                cand, k = jitter(gt, rng)
                if 0.3 <= iou(cand, gt) <= 0.8 and all(iou(cand, o) < 0.3 for o in same) \
                        and cand[2] - cand[0] > 0.02 and cand[3] - cand[1] > 0.02:
                    distractor, kind = cand, k
                    break
        else:
            others = [box(r) for r in rows if r["LabelName"] != target["LabelName"] and r["IsGroupOf"] == "0"
                      and iou(box(r), gt) < 0.3 and (float(r["XMax"]) - float(r["XMin"])) * (float(r["YMax"]) - float(r["YMin"])) >= 0.005
                      and all(iou(box(r), o) < 0.3 for o in same)]
            if others:
                distractor, kind = rng.choice(others), "other_object"
        if distractor is None:
            continue
        need[want] -= 1
        items.append({"item_id": f"oibox_{image_id}", "image_id": image_id, "split": split, "name": names[target["LabelName"]].lower(),
                      "gt": gt, "distractor": distractor, "kind": kind, "difficulty": want, "iou": round(iou(gt, distractor), 3),
                      "gold": rng.choice("AB"), "colors": rng.sample(range(len(PALETTE)), 2), "corner": rng.choices(["tl", "tr"], [0.8, 0.2])[0],
                      "template": rng.randrange(len(BOX_QUESTIONS)), "move_gt_label": rng.random() < 0.5})
    jsonl(root / "selected.jsonl", items)
    print(json.dumps({"selected": len(items), "by_kind": Counter(i["kind"] for i in items), "gold": Counter(i["gold"] for i in items)}))


PALETTE = [(230, 25, 25), (20, 200, 20), (30, 90, 255), (255, 150, 0), (220, 0, 220), (0, 190, 200), (250, 230, 0)]
BOX_QUESTIONS = [  # BLINK wording is one of five, about 1/5 of items
    "Which bounding box more accurately localizes and encloses the {name}?",
    "Which box fits the {name} more tightly?",
    "Which of the two boxes correctly outlines the {name}?",
    "Which labeled box best matches the full extent of the {name}?",
    "Which box covers the whole {name} with the least extra background?",
]


def font(size):
    for path in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def draw_item(image, item):
    image = image.convert("RGB")
    W, H = image.size
    d = ImageDraw.Draw(image)
    width = max(2, round(0.004 * max(W, H)))
    side = int(min(max(0.06 * min(W, H), 16), 40))
    f = font(int(side * 0.8))
    letters = {"gt": item["gold"], "distractor": "B" if item["gold"] == "A" else "A"}
    px = {w: [round(v * s) for v, s in zip(item[w], (W, H, W, H))] for w in ("gt", "distractor")}
    labels = place_labels(item, px, side, H)
    for which, color in zip(("gt", "distractor"), (PALETTE[c] for c in item["colors"])):
        d.rectangle(px[which], outline=color, width=width)
        d.rectangle(labels[which], fill=color)
        text_color = (0, 0, 0) if sum(color) > 450 else (255, 255, 255)
        lx, ly = labels[which][:2]
        d.text((lx + side / 2, ly + side / 2), letters[which], fill=text_color, font=f, anchor="mm")
    return image


def place_labels(item, px, side, H):
    labels = {w: label_rect(px[w], item["corner"], side, H) for w in px}
    if overlaps(labels["gt"], labels["distractor"]):
        # move one label, chosen per item at selection, so a displaced label never marks the wrong box
        mover = "gt" if item.get("move_gt_label", False) else "distractor"
        fixed = labels["distractor" if mover == "gt" else "gt"]
        other = "tr" if item["corner"] == "tl" else "tl"
        for candidate in (label_rect(px[mover], other, side, H), label_rect(px[mover], item["corner"], side, H, outside=True),
                          label_rect(px[mover], other, side, H, outside=True)):
            if not overlaps(candidate, fixed):
                labels[mover] = candidate
                break
    return labels


def label_rect(box, corner, side, H, outside=False):
    x1, y1, x2, y2 = box
    lx = x1 if corner == "tl" else max(x1, x2 - side)
    if outside:  # just above the box, or just below it when there is no room above
        ly = y1 - side if y1 - side >= 0 else min(y2, H - side)
    else:
        ly = y1
    return [lx, ly, lx + side, ly + side]


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ---------- download ----------
def cache_dns():
    """The host resolver drops lookups under dozens of threads; resolve each host once."""
    import functools
    import socket
    resolve = functools.lru_cache(maxsize=64)(socket.getaddrinfo)  # failures raise, so they are not cached
    socket.getaddrinfo = lambda *a, **k: resolve(*a, **k)


def fetch_one(url, dest):
    if dest.exists():
        return "exists"
    import time
    for attempt in range(6):
        if attempt:
            time.sleep(2 ** attempt)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                image = Image.open(io.BytesIO(r.read())).convert("RGB")
            image.thumbnail((MAX_SIDE, MAX_SIDE))
            tmp = dest.with_suffix(".part")
            image.save(tmp, "JPEG", quality=90)
            os.replace(tmp, dest)
            return "ok"
        except Exception as error:  # noqa: BLE001 - network/decoding failures are retried then recorded
            last = repr(error)
    return last


def cmd_fetch(args):
    cache_dns()
    jobs =[(S3.format(split=i["split"], id=i["image_id"]), args.root / "openimages/images" / f"{i['image_id']}.jpg")
            for i in read_jsonl(args.root / "openimages/selected.jsonl")]
    for d in ("openimages/images",):
        (args.root / d).mkdir(exist_ok=True)
    with ThreadPoolExecutor(args.threads) as pool:
        results = list(pool.map(lambda j: fetch_one(*j), jobs))
    failed = [(j[0], r) for j, r in zip(jobs, results) if r not in ("ok", "exists")]
    jsonl(args.root / "fetch_failures.jsonl", [{"url": u, "error": e} for u, e in failed])
    print(json.dumps({"jobs": len(jobs), "status": Counter(r if r in ("ok", "exists") else "failed" for r in results)}))


def cmd_render(args):
    out = args.root / "openimages/box_images"
    out.mkdir(exist_ok=True)
    n = 0
    for item in read_jsonl(args.root / "openimages/selected.jsonl"):
        src = args.root / "openimages/images" / f"{item['image_id']}.jpg"
        if src.exists():
            with Image.open(src) as image:
                draw_item(image, item).save(out / f"{item['item_id']}.jpg", "JPEG", quality=92)
            n += 1
    print(json.dumps({"rendered": n}))


# ---------- SpatialSense ----------
def cmd_spatialsense(args):
    root = args.root / "spatialsense"
    wanted, rows = {}, {"train": [], "heldout_test": []}
    for im in json.load(open(root / "annotations.json")):
        if "staticflickr" not in im["url"]:
            continue  # NYU images: SUN RGB-D (CV-Bench 3D) contains NYU
        base = im["url"].rsplit("/", 1)[-1]
        wanted[base] = im
        for a in im["annotations"]:
            rows["train" if im["split"] in ("train", "valid") else "heldout_test"].append(
                {"ann_id": a["_id"], "image": base, "subject": a["subject"]["name"], "predicate": a["predicate"],
                 "object": a["object"]["name"], "label": bool(a["label"]), "split": im["split"]})
    out = root / "images"
    out.mkdir(exist_ok=True)
    with tarfile.open(root / "images.tar.gz") as tar:
        for member in tar:
            base = member.name.rsplit("/", 1)[-1]
            if member.isfile() and base in wanted and not (out / base).exists():
                image = Image.open(tar.extractfile(member)).convert("RGB")
                image.thumbnail((MAX_SIDE, MAX_SIDE))
                image.save(out / base, "JPEG", quality=90)
    report = {}
    for name, items in rows.items():
        kept = [r for r in items if (out / r["image"]).exists()]
        jsonl(root / f"{name}.jsonl", kept)
        report[name] = {"relations": len(items), "with_image": len(kept), "true": sum(r["label"] for r in kept)}
    print(json.dumps(report))


# ---------- perceptual-hash dedup ----------
_N = 32
DCT = np.array([[np.sqrt((1 if k == 0 else 2) / _N) * np.cos(np.pi * (2 * n + 1) * k / (2 * _N)) for n in range(_N)] for k in range(_N)])


def phash(path):
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((_N, _N), Image.LANCZOS), dtype=np.float64)
    low = (DCT @ gray @ DCT.T)[:8, :8].ravel()[1:]
    bits = low > np.median(low)
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming_min(new, ref):
    """Smallest Hamming distance from each new hash to any reference hash."""
    ref = np.array(ref, dtype=np.uint64)
    out = []
    for chunk in np.array_split(np.array(new, dtype=np.uint64), max(1, len(new) // 2000)):
        x = (chunk[:, None] ^ ref[None, :]).view(np.uint8).reshape(len(chunk), len(ref), 8)
        out.append(np.unpackbits(x, axis=2).sum(axis=2).min(axis=1))
    return np.concatenate(out)


def cmd_dedup(args):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    evals = [p for d in args.eval_dirs for p in Path(d).rglob("*") if p.suffix.lower() in exts]
    new = [p for d in ("openimages/images", "spatialsense/images")
           for p in (args.root / d).glob("*.jpg")]
    with ThreadPoolExecutor(args.threads) as pool:
        ref = list(pool.map(phash, evals))
        hashes = list(pool.map(phash, new))
    dist = hamming_min(hashes, ref)
    drop = sorted(str(p.relative_to(args.root)) for p, d in zip(new, dist) if d <= args.threshold)
    report = {"eval_images": len(evals), "new_images": len(new), "threshold": args.threshold, "dropped": len(drop),
              "distance_histogram_0_12": np.bincount(np.minimum(dist, 12), minlength=13).tolist(), "drop": drop}
    (args.root / "dedup.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "drop"}))


# ---------- assemble the E pool ----------
def make_row(row_id, question, answers, gold, source, task, image, rng, label_source, **extra):
    perm = list(range(len(answers)))
    rng.shuffle(perm)
    shown = [answers[p] for p in perm]
    row = {"id": row_id, "question": question, "answers": shown, "gold_index": perm.index(gold), "source": source,
           "source_split": "train", "source_answers": answers, "source_gold_index": gold, "candidate_permutation": perm,
           "correct_answer": answers[gold], "label_source": label_source,
           "pattern": "Noul" if answers == ["yes", "no"] else "Choice", "question_type": task, "sampling_task": task,
           "sampling_bucket": answers[gold].lower(), "images": [image], "image_id": f"ext_{Path(image).stem}",
           "image_group": f"ext_{Path(image).stem}", "pool_split": "train", **extra}
    if row["pattern"] == "Noul":
        row["yes_index"] = shown.index("yes")
    return row


def cmd_assemble(args):
    rng = random.Random(args.seed)
    root, out = args.root, args.output
    drops = set(json.load(open(root / "dedup.json"))["drop"])
    rows, counts = [], Counter()
    for it in read_jsonl(root / "openimages/selected.jsonl"):
        if f"openimages/images/{it['image_id']}.jpg" in drops or not (root / "openimages/box_images" / f"{it['item_id']}.jpg").exists():
            continue
        rows.append(make_row(it["item_id"], BOX_QUESTIONS[it["template"]].format(name=it["name"]), ["Box A", "Box B"],
                             "AB".index(it["gold"]), "openimages_box", f"box/{it['difficulty']}",
                             f"external/openimages/box_images/{it['item_id']}.jpg", rng, "rule_box_choice",
                             box_kind=it["kind"], box_iou=it["iou"], source_image_id=it["image_id"], source_image_split=it["split"]))
    for it in read_jsonl(root / "spatialsense/train.jsonl"):
        rel = f"spatialsense/images/{it['image']}"
        if rel in drops:
            continue
        rows.append(make_row(f"spatialsense_{it['ann_id']}", f"Is the {it['subject']} {it['predicate']} the {it['object']}?",
                             ["yes", "no"], 0 if it["label"] else 1, "spatialsense", f"spatialsense/{it['predicate']}",
                             "external/" + rel, rng, "dataset_gt"))
    counts.update(r["source"] for r in rows)
    out.mkdir()  # never overwrite an existing pool
    with open(out / "train.jsonl", "w") as dst:
        with open(args.base_pool / "train.jsonl") as src:
            for line in src:
                dst.write(line)
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name in ("dev.jsonl", "image_store.sqlite", "images"):
        os.symlink(os.path.realpath(args.base_pool / name), out / name)
    os.symlink(root, out / "external")
    report = {"base_pool": str(args.base_pool), "external_root": str(root), "added": dict(counts),
              "added_by_task": dict(Counter(r["sampling_task"] for r in rows)),
              "gold_balance": dict(Counter(f'{r["source"]}|{r["correct_answer"]}' for r in rows)),
              "dedup_dropped_images": len(drops), "dev": "unchanged (symlink to the stage-1 dev)"}
    (out / "external_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["openimages", "fetch", "render", "spatialsense", "dedup", "assemble"])
    p.add_argument("--root", type=Path, required=True, help="external data directory")
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--tight", type=int, default=15000)
    p.add_argument("--easy", type=int, default=3000)
    p.add_argument("--threads", type=int, default=48)
    p.add_argument("--eval-dirs", nargs="*", default=[])
    p.add_argument("--threshold", type=int, default=10)
    p.add_argument("--base-pool", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    globals()[f"cmd_{args.step}"](args)


if __name__ == "__main__":
    main()
