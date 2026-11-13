#!/usr/bin/env python3
"""Stage-2 driving items from DriveLM-nuScenes v1.1: box-choice localization and spatial-relation yes/no.

Uses each key frame's key_object_infos ("<c1,CAM_FRONT,x,y>" -> Category, Status, Visual_description, 2d_bbox).
  inspect   print the JSON structure (run first: the schema is taken from the DriveLM docs, not yet verified here)
  generate  box-choice items (GT box vs jittered box, same drawing as control E) and relation minimal pairs
            (left/right in the image; closer/farther by box bottom on flat ground), then perceptual-hash dedup
            against eval images (CV-Bench 3D contains nuScenes frames)
  qa        DriveLM's closed-form questions as choice rows: per-object questions on the object's camera image with
            the object in a red box, ego questions on the front image; same eval-image dedup
  assemble  stage-2 pool = base (stage-1) pool + DriveLM train rows; DriveLM dev/test come from held-out scenes
Box/relation labels come from 2D boxes by rule (rule_2d_bbox); qa labels are DriveLM's own answers (dataset_gt).
"""
import argparse
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageDraw

from superglance.builders.qa import prepare_external_e as ext

TAG = re.compile(r"<(c\d+),(CAM_[A-Z_]+),([\d.]+),([\d.]+)>")
GROUND = ("vehicle", "pedestrian", "cyclist", "bicycle", "motorcycle", "car", "truck", "bus")
BOX_QUESTIONS = [
    "Which bounding box more accurately localizes the {d}?",
    "Which box covers the whole {d} with the least extra background?",  # "fits more tightly" also reads true for a shrunken box
    "Which of the two boxes correctly outlines the {d}?",
]
SPLIT_SEED = 20260926  # scene-level dev/test split shared by every step
CAMERA = {"CAM_FRONT": "front", "CAM_FRONT_LEFT": "front-left", "CAM_FRONT_RIGHT": "front-right",
          "CAM_BACK": "rear", "CAM_BACK_LEFT": "rear-left", "CAM_BACK_RIGHT": "rear-right"}
ANY_TAG = re.compile(r"<(c\d+),\s*(CAM_[A-Z_]+),\s*([\d.]+),\s*([\d.]+)>")
OBJ = "the object in the red box"
YES_NO = (["yes", "no"], {"Yes.": "yes", "No.": "no"})
FUTURE = ["Keep going straight", "Stay stationary", "Turn left", "Turn right", "Brake to a stop"]
SIGN = ["Traffic light", "Traffic sign", "Traffic cone", "Zebra crossing", "Road marking"]
# DriveLM closed-form template -> (task, question, options, answer map); rare paraphrased answers outside the map are skipped
OBJECT_QA = {
    "Is <obj> a traffic sign or a road barrier?": ("sign_or_barrier", "Is {o} a traffic sign or a road barrier?", *YES_NO),
    "Would <obj> be in the moving direction of the ego vehicle?": ("in_ego_path", "Would {o} be in the moving direction of the ego vehicle?", *YES_NO),
    "Is <obj> an object that the ego vehicle should consider in the current scene?": ("consider", "Should the ego vehicle consider {o} in the current scene?", *YES_NO),
    "Is it necessary for the ego vehicle to take <obj> into account?": ("consider", "Is it necessary for the ego vehicle to take {o} into account?", *YES_NO),
    "What is the observed status of object <obj>?": ("status", "What is the observed status of {o}?", ["Moving", "Stationary"],
                                                     {"Moving.": "Moving", "Stationary.": "Stationary"}),
    "What is the moving status of object <obj>?": ("motion", "What is the moving status of {o}?", ["Going ahead", "Turning left", "Turning right"],
                                                   {"Going ahead.": "Going ahead", "Turn left.": "Turning left", "Turn right.": "Turning right"}),
    "What is the future state of <obj>?": ("future", "What is the future state of {o}?", FUTURE,
                                           {"Keep going straight.": FUTURE[0], "Stationary.": FUTURE[1], "Keep stationary.": FUTURE[1],
                                            "Turn left.": FUTURE[2], "Turn right.": FUTURE[3], "Brake gently to a stop.": FUTURE[4],
                                            "Brake gently to come to a stop.": FUTURE[4], "Brake gently to stop.": FUTURE[4],
                                            "Brakes gently to a stop.": FUTURE[4]}),
    "What kind of traffic sign is <obj>?": ("sign_kind", "What kind of traffic element is {o}?", SIGN,
                                            {"Traffic light.": SIGN[0], "Traffic lights.": SIGN[0], "Traffic sign.": SIGN[1], "Road sign.": SIGN[1],
                                             "Traffic cone.": SIGN[2], "Cone barrel.": SIGN[2], "Zebra crossing.": SIGN[3], "Crosswalk.": SIGN[3],
                                             "Road marking.": SIGN[4], "Traffic marking.": SIGN[4], "Traffic markings.": SIGN[4],
                                             "Ground marking.": SIGN[4]}),
}
DESCRIBE = "What is the visual description of <obj>?"
EGO_QA = {  # frame-level questions, asked on the front camera image
    "What is the target action of the ego vehicle?": (
        "target_action", "What is the target action of the ego vehicle?", ["Go straight", "Stay stationary", "Turn left", "Turn right"],
        {"Go straight.": "Go straight", "Stationary.": "Stay stationary", "Keep stationary.": "Stay stationary",
         "Turn left.": "Turn left", "Turn right.": "Turn right"}),
    "What is the traffic signal that the ego vehicle should pay attention to?": (
        "signal", "What traffic signal should the ego vehicle pay attention to?", ["None", "Red light", "Green light", "Yellow light", "Crosswalk"],
        {"None.": "None", "Red light.": "Red light", "Green light.": "Green light", "Yellow light.": "Yellow light", "Crosswalk.": "Crosswalk"}),
    "Is there any traffic element in the front view?": (
        "traffic_element", "Is there any traffic element in the front view?", ["yes", "no"],
        {"Yes, there are traffic elements in the front view.": "yes", "Yes, there are some traffic elements in the front view.": "yes",
         "No, there are no traffic elements in the front view.": "no"}),
}
BEHAVIOR_Q = "Predict the behavior of the ego vehicle."
BEHAVIOR = re.compile(r"The ego vehicle is (.+?)\. The ego vehicle is (.+?)\.$")
STEERING = {"going straight": "Going straight", "slightly steering to the left": "Slightly steering left",
            "slightly steering to the right": "Slightly steering right", "steering to the left": "Steering left",
            "steering to the right": "Steering right"}
SPEED = {"not moving": "Not moving", "driving slowly": "Driving slowly", "driving with normal speed": "Driving at normal speed",
         "driving fast": "Driving fast", "driving very fast": "Driving very fast"}
# describe distractors must share no word with the answer after folding near-synonyms ("dark car" vs "black sedan")
SYNONYM = {"dark": "black", "grey": "gray", "silver": "gray", "sedan": "car", "suv": "car", "hatchback": "car", "vehicle": "car",
           "pedestrian": "person", "man": "person", "woman": "person", "people": "person"}
STOP = {"a", "an", "the", "in", "with", "and", "of", "on", "is", "wearing"}


def objects(frame):
    """Key objects with a usable 2D box, grouped by camera."""
    by_cam = {}
    for tag, info in (frame.get("key_object_infos") or {}).items():
        m, box = TAG.match(tag), info.get("2d_bbox")
        desc = re.sub(r"^(a|an|the)\s+", "", str(info.get("Visual_description") or "").strip().rstrip(".").lower())
        if not m or not box or len(box) != 4 or not desc:
            continue
        by_cam.setdefault(m.group(2), []).append({"id": m.group(1), "category": str(info.get("Category", "")).lower(),
                                                  "desc": desc, "box": [float(v) for v in box]})
    return by_cam


def relation_pairs(objs, width, height, rng):
    """(question, yes) minimal pairs from 2D boxes: each true relation plus its word-flipped false twin."""
    out = []
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            if a["desc"] == b["desc"]:
                continue  # descriptions must tell the two objects apart
            ax, bx = (a["box"][0] + a["box"][2]) / 2, (b["box"][0] + b["box"][2]) / 2
            x_overlap = min(a["box"][2], b["box"][2]) - max(a["box"][0], b["box"][0])
            if abs(ax - bx) >= 0.1 * width and x_overlap <= 0:
                left, right = (a, b) if ax < bx else (b, a)
                out.append((f"In this image, is the {left['desc']} to the left of the {right['desc']}?", True, "left_right"))
                out.append((f"In this image, is the {left['desc']} to the right of the {right['desc']}?", False, "left_right"))
            if all(any(g in o["category"] for g in GROUND) for o in (a, b)) and abs(a["box"][3] - b["box"][3]) >= 0.05 * height:
                near, far = (a, b) if a["box"][3] > b["box"][3] else (b, a)
                out.append((f"Is the {near['desc']} closer to the ego vehicle than the {far['desc']}?", True, "closer_farther"))
                out.append((f"Is the {near['desc']} farther from the ego vehicle than the {far['desc']}?", False, "closer_farther"))
    return out


def frames(data):
    for scene_token, scene in data.items():
        for frame_token, frame in (scene.get("key_frames") or {}).items():
            yield scene_token, frame_token, frame


def scene_split(data):
    """10% of scenes dev, 10% test, the rest train; fixed seed so every step agrees."""
    order = sorted(data)
    random.Random(SPLIT_SEED).shuffle(order)
    n = len(order) // 10
    return {s: "dev" if i < n else "test" if i < 2 * n else "train" for i, s in enumerate(order)}


def words(desc):
    return {SYNONYM.get(w, w) for w in re.findall(r"[a-z]+", desc.lower())} - STOP


def distractors(gold, local, pool, rng, k=3):
    """k descriptions sharing no word with gold: other objects of the same frame first, then random ones."""
    out, g = [], words(gold)
    for d in local + [rng.choice(pool) for _ in range(100)]:
        if len(out) == k:
            break
        if d not in out and not words(d) & g:
            out.append(d)
    return out if len(out) == k else None


def render_marked(job):
    src, box, dst = job
    with Image.open(src) as im:
        im = im.convert("RGB")
        width = im.width
        im.thumbnail((ext.MAX_SIDE, ext.MAX_SIDE))
        s = im.width / width
        ImageDraw.Draw(im).rectangle([v * s for v in box], outline=(255, 0, 0), width=3)
        im.save(dst, "JPEG", quality=92)


def cmd_inspect(args):
    data = json.load(open(args.qa))
    scene_token, frame_token, frame = next(frames(data))
    print("scenes:", len(data), "| scene keys:", list(data[scene_token]))
    print("frame keys:", list(frame))
    print("key_object_infos sample:", json.dumps(dict(list((frame.get("key_object_infos") or {}).items())[:2]), ensure_ascii=False)[:600])
    print("image_paths:", json.dumps(frame.get("image_paths"), ensure_ascii=False)[:400])
    qa = frame.get("QA") or {}
    print("QA categories:", {k: len(v) for k, v in qa.items()})
    n_obj = Counter(len(v) for _, _, f in frames(data) for v in objects(f).values())
    print("objects-with-box per camera image (count: images):", dict(sorted(n_obj.items())))


def cmd_generate(args):
    rng = random.Random(args.seed)
    data = json.load(open(args.qa))
    by_name = {p.name: p for p in args.images.rglob("*.jpg")}
    out = args.output
    (out / "images").mkdir(parents=True)  # never overwrite an earlier generation
    box_items, rel_items, missing = [], [], 0
    for scene_token, frame_token, frame in frames(data):
        for cam, objs in objects(frame).items():
            src = by_name.get(Path(str((frame.get("image_paths") or {}).get(cam, ""))).name)
            if src is None:
                missing += 1
                continue
            with Image.open(src) as im:
                W, H = im.size
            for q, yes, kind in relation_pairs(objs, W, H, rng):
                rel_items.append({"image": str(src), "question": q, "yes": yes, "kind": kind, "scene": scene_token, "frame": frame_token, "cam": cam})
            for o in objs:
                gt = [o["box"][0] / W, o["box"][1] / H, o["box"][2] / W, o["box"][3] / H]
                area = (gt[2] - gt[0]) * (gt[3] - gt[1])
                if area < 0.005:
                    continue
                others = [[p["box"][0] / W, p["box"][1] / H, p["box"][2] / W, p["box"][3] / H] for p in objs if p is not o]
                for _ in range(30):
                    cand, k = ext.jitter(gt, rng)
                    if 0.3 <= ext.iou(cand, gt) <= 0.8 and all(ext.iou(cand, p) < 0.3 for p in others):
                        box_items.append({"image": str(src), "desc": o["desc"], "gt": gt, "distractor": cand, "kind": k,
                                          "gold": rng.choice("AB"), "colors": rng.sample(range(len(ext.PALETTE)), 2), "corner": "tl",
                                          "move_gt_label": rng.random() < 0.5, "template": rng.randrange(len(BOX_QUESTIONS)),
                                          "scene": scene_token, "frame": frame_token, "cam": cam, "obj": o["id"]})
                        break
    # perceptual-hash dedup of source images against the eval images
    sources = sorted({i["image"] for i in box_items + rel_items})
    ref = [ext.phash(p) for d in args.eval_dirs for p in Path(d).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    dist = ext.hamming_min([ext.phash(p) for p in sources], ref) if ref else [99] * len(sources)
    dropped = {s for s, d in zip(sources, dist) if d <= args.threshold}
    rows = []
    for n, it in enumerate(i for i in box_items if i["image"] not in dropped):
        with Image.open(it["image"]) as im:
            im = im.convert("RGB")
            im.thumbnail((ext.MAX_SIDE, ext.MAX_SIDE))
            ext.draw_item(im, it).save(out / "images" / f"box_{n:06d}.jpg", "JPEG", quality=92)
        rows.append(ext.make_row(f"drivelm_box_{n:06d}", BOX_QUESTIONS[it["template"]].format(d=it["desc"]), ["Box A", "Box B"],
                                 "AB".index(it["gold"]), "drivelm_box", f"drivelm_box/{it['kind']}", f"images/box_{n:06d}.jpg",
                                 rng, "rule_2d_bbox", scene=it["scene"], frame=it["frame"], cam=it["cam"]))
    for n, it in enumerate(i for i in rel_items if i["image"] not in dropped):
        rows.append(ext.make_row(f"drivelm_rel_{n:06d}", it["question"], ["yes", "no"], 0 if it["yes"] else 1, "drivelm_relation",
                                 f"drivelm_relation/{it['kind']}", it["image"], rng, "rule_2d_bbox",
                                 scene=it["scene"], frame=it["frame"], cam=it["cam"]))
    with open(out / "drivelm_s2.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"box_items": sum(r["source"] == "drivelm_box" for r in rows), "relation_items": sum(r["source"] == "drivelm_relation" for r in rows),
              "by_task": dict(Counter(r["sampling_task"] for r in rows)), "camera_images_without_file": missing,
              "dedup_dropped_images": len(dropped), "label_source": "rule_2d_bbox (DriveLM key_object_infos boxes)",
              "note": "relation images are absolute paths to the extracted DriveLM images; box images are rendered under images/"}
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


def cmd_qa(args):
    rng = random.Random(args.seed)
    data = json.load(open(args.qa))
    split = scene_split(data)
    by_name = {p.name: p for p in args.images.rglob("*.jpg")}
    out = args.output
    (out / "images").mkdir(parents=True)  # never overwrite an earlier generation
    descs = sorted({o["desc"] for _, _, f in frames(data) for objs in objects(f).values() for o in objs})
    items, skipped = [], Counter()
    for scene_token, frame_token, frame in frames(data):
        paths = frame.get("image_paths") or {}
        objs = {o["id"]: dict(o, cam=cam) for cam, group in objects(frame).items() for o in group}
        base = {"scene": scene_token, "frame": frame_token, "drivelm_split": split[scene_token]}
        for qa in (qa for group in (frame.get("QA") or {}).values() for qa in group):
            q, a = qa["Q"].strip(), qa["A"].strip()
            tags = ANY_TAG.findall(q)
            template = ANY_TAG.sub("<obj>", q)
            if template in EGO_QA or template == BEHAVIOR_Q:
                src = by_name.get(Path(str(paths.get("CAM_FRONT", ""))).name)
                if src is None:
                    skipped["ego/no_front_image"] += 1
                    continue
                if template in EGO_QA:
                    task, text, options, amap = EGO_QA[template]
                    found = [(task, text, options, amap.get(a))]
                else:
                    m = BEHAVIOR.match(a)
                    found = [("steering", "Predict the steering of the ego vehicle.", list(STEERING.values()), m and STEERING.get(m.group(1))),
                             ("speed", "Predict the speed of the ego vehicle.", list(SPEED.values()), m and SPEED.get(m.group(2)))]
                for task, text, options, answer in found:
                    if answer is None:
                        skipped[f"ego/{task}/unmapped_answer"] += 1
                        continue
                    items.append(dict(base, source="drivelm_ego", task=task, question="This image is from the ego vehicle's front camera. " + text,
                                      answers=options, gold=options.index(answer), src=str(src), cam="CAM_FRONT", render=None))
                continue
            if len(tags) != 1 or (template not in OBJECT_QA and template != DESCRIBE):
                continue  # open-ended or multi-object question
            cid, cam = tags[0][:2]
            o, src = objs.get(cid), by_name.get(Path(str(paths.get(cam, ""))).name)
            if o is None or o["cam"] != cam or src is None:
                skipped["object/no_box_or_image"] += 1
                continue
            head = f"This image is from the ego vehicle's {CAMERA[cam]} camera. "
            common = dict(base, source="drivelm_object", src=str(src), cam=cam,
                          render=(str(src), o["box"], str(out / "images" / f"obj_{frame_token}_{cid}.jpg")))
            if template == DESCRIBE:
                local = [p["desc"] for p in objs.values() if p is not o]
                rng.shuffle(local)
                wrong = distractors(o["desc"], local, descs, rng)
                if wrong is None:
                    skipped["object/describe/no_distractors"] += 1
                    continue
                items.append(dict(common, task="describe", question=head + f"Which description matches {OBJ}?",
                                  answers=[d[0].upper() + d[1:] for d in [o["desc"]] + wrong], gold=0))
                continue
            task, text, options, amap = OBJECT_QA[template]
            answer = amap.get(a)
            if answer is None:
                skipped[f"object/{task}/unmapped_answer"] += 1
                continue
            items.append(dict(common, task=task, question=head + text.format(o=OBJ), answers=options, gold=options.index(answer)))
    # perceptual-hash dedup of source camera images against the eval images
    sources = sorted({i["src"] for i in items})
    ref_paths = [p for d in args.eval_dirs for p in Path(d).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    with Pool(args.workers) as pool:
        ref = pool.map(ext.phash, ref_paths, chunksize=64)
        dist = ext.hamming_min(pool.map(ext.phash, sources, chunksize=64), ref) if ref else [99] * len(sources)
        dropped = {s for s, d in zip(sources, dist) if d <= args.threshold}
        items = [i for i in items if i["src"] not in dropped]
        pool.map(render_marked, list({i["render"][2]: i["render"] for i in items if i["render"]}.values()), chunksize=32)
    rows = [ext.make_row(f"drivelm_qa_{n:06d}", i["question"], i["answers"], i["gold"], i["source"], f"{i['source']}/{i['task']}",
                         i["render"][2] if i["render"] else i["src"], rng,
                         "dataset_gt_rule_distractors" if i["task"] == "describe" else "dataset_gt",
                         scene=i["scene"], frame=i["frame"], cam=i["cam"], drivelm_split=i["drivelm_split"])
            for n, i in enumerate(items)]
    with open(out / "drivelm_qa.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"rows": len(rows), "by_task": dict(sorted(Counter(r["sampling_task"] for r in rows).items())),
              "by_split": dict(Counter(r["drivelm_split"] for r in rows)),
              "gold_balance": dict(sorted(Counter(f'{r["sampling_task"]}|{r["correct_answer"]}' for r in rows).items())),
              "skipped": dict(skipped), "dedup_dropped_images": len(dropped), "marked_images": len({r["images"][0] for r in rows if "/images/obj_" in r["images"][0]})}
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


def cmd_assemble(args):
    rng = random.Random(args.seed)
    root, out = args.root, args.output
    split = scene_split(json.load(open(args.qa)))
    rows = []
    for r in ext.read_jsonl(root / "s2_items/drivelm_s2.jsonl"):
        image = r["images"][0]
        rows.append(dict(r, images=[image if image.startswith("/") else str(root / "s2_items" / image)], drivelm_split=split[r["scene"]]))
    rows += list(ext.read_jsonl(root / "qa_items/drivelm_qa.jsonl"))
    parts = defaultdict(list)
    for r in rows:
        parts[r["drivelm_split"]].append(r)

    def capped(part, n):  # at most n rows per task, so eval time stays bounded and small tasks keep weight
        by_task = defaultdict(list)
        for r in part:
            by_task[r["sampling_task"]].append(r)
        return [r for task in sorted(by_task) for r in rng.sample(by_task[task], min(n, len(by_task[task])))]

    dev, test = capped(parts["dev"], args.dev_cap), capped(parts["test"], args.test_cap)
    out.mkdir()  # never overwrite an existing pool
    for name, added in (("train.jsonl", parts["train"]), ("dev.jsonl", dev)):
        with open(out / name, "w") as dst:
            with open(args.base_pool / name) as src:
                shutil.copyfileobj(src, dst)
            for r in added:
                dst.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "drivelm_test.jsonl", "w") as f:
        for r in test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for entry in args.base_pool.iterdir():  # images, image store, external data links of the base pool
        if entry.suffix not in (".jsonl", ".json"):
            os.symlink(os.path.realpath(entry), out / entry.name)
    report = {"base_pool": str(args.base_pool), "drivelm_root": str(root),
              "train_added": dict(sorted(Counter(r["sampling_task"] for r in parts["train"]).items())),
              "dev_added": dict(sorted(Counter(r["sampling_task"] for r in dev).items())),
              "test": dict(sorted(Counter(r["sampling_task"] for r in test).items())),
              "scenes": dict(Counter(split.values())), "dev": "base dev + DriveLM dev (held-out scenes)",
              "test_manifest": "drivelm_test.jsonl (held-out scenes, disjoint from dev)"}
    (out / "stage2_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["inspect", "generate", "qa", "assemble"])
    p.add_argument("--qa", type=Path, required=True, help="v1_1_train_nus.json")
    p.add_argument("--images", type=Path, help="directory with the extracted drivelm_nus_imgs_train.zip")
    p.add_argument("--output", type=Path)
    p.add_argument("--eval-dirs", nargs="*", default=[])
    p.add_argument("--threshold", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--root", type=Path, help="DriveLM directory with s2_items/ and qa_items/ (assemble)")
    p.add_argument("--base-pool", type=Path, help="stage-1 pool to extend (assemble)")
    p.add_argument("--dev-cap", type=int, default=150, help="DriveLM dev rows per task")
    p.add_argument("--test-cap", type=int, default=400, help="DriveLM test rows per task")
    args = p.parse_args()
    globals()[f"cmd_{args.step}"](args)


if __name__ == "__main__":
    main()
