#!/usr/bin/env python3
"""Headline numbers of the QA benchmarks from eval_qa output dirs (sat/, external/, 3dsr/, drivelm_test/, text/), with the
definitions of Table 2: CV-Bench source-weighted (0.5 x (the mean of the two 2D sources + the 3D source)), BLINK's four tasks,
3DSRBench official circular, DriveLM test right in every cyclic order (and in the stored order), the text benchmarks' accuracy;
SAT when present. Prints one column per dir, in percent.
usage: python -m superglance.eval.qa_table <eval dir> [<eval dir> ...]"""
import json
import sys
from pathlib import Path

TEXT = [("mmlu", "MMLU"), ("mmlu_pro", "MMLU-Pro"), ("arc_challenge", "ARC-C"), ("arc_easy", "ARC-E"),
        ("hellaswag", "HellaSwag"), ("winogrande", "WinoGrande"), ("boolq", "BoolQ"), ("truthfulqa_mc1", "TruthfulQA-MC1")]
CV_TASKS = [("2D/Count", "2D count"), ("2D/Relation", "2D relation"), ("3D/Depth", "3D depth"), ("3D/Distance", "3D distance")]
BLINK_TASKS = [("Object_Localization", "object localization"), ("Spatial_Relation", "spatial relation"),
               ("Counting", "counting"), ("Relative_Depth", "relative depth")]


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def task_acc(by_task, key):
    for name, value in (by_task or {}).items():
        if name.lower() == key.lower():
            return value["accuracy"]
    return None


def headline(p):
    out = {}
    m = load(p["sat"] / "dev/metrics.json")
    if m:
        out["SAT dev (2,000)"] = m["all"]["accuracy"]
    m = load(p["sat"] / "sat_test_single_image/metrics.json")
    if m:
        out["SAT test single-image, circular (104 x 2)"] = m["circular_binary"]["all"]["accuracy"]
    e = load(p["external"] / "summary.json")
    if e:
        cv, bl = e["cvbench"], e["blink_single_image"]
        out["CV-Bench weighted (2,638)"] = cv["official_source_weighted_accuracy"]
        for key, label in CV_TASKS:
            out[f"CV-Bench {label}"] = task_acc(cv["by_task"], key)
        out["BLINK 4-task subset (509)"] = bl["micro"]["accuracy"]
        for key, label in BLINK_TASKS:
            out[f"BLINK {label}"] = task_acc(bl["by_task"], key)
    t = load(p["3dsr"] / "3dsr_summary.json")
    if t:
        out["3DSRBench official circular"] = t["official_circular"]["accuracy"]
        out["3DSRBench row micro (11,686 rows)"] = t["row_micro_accuracy"]
        for cat, v in sorted(t["official_circular"]["by_category"].items()):
            out[f"3DSRBench circular: {cat}"] = v["accuracy"]
    d = load(p["drivelm"] / "rotation_metrics.json")
    if d:
        out["DriveLM test, original order (6,571)"] = d["all"]["original_accuracy"]
        out["DriveLM test, all cyclic orders correct"] = d["all"]["all_rotations_correct_rate"]
        out["DriveLM test, content consistency"] = d["all"]["content_consistency_rate"]
    for key, label in TEXT:
        m = load(Path(p["text"]) / key / "metrics.json")
        if m:
            out[f"Text {label}"] = m["all"]["accuracy"]
    return out


def main():
    """The headline numbers of each eval dir, one column per dir (eval_qa's layout)."""
    dirs = [Path(d) for d in sys.argv[1:]]
    nine = lambda r: {"sat": r / "sat", "external": r / "external", "3dsr": r / "3dsr", "drivelm": r / "drivelm_test", "text": r / "text"}
    table = [headline(nine(d)) for d in dirs]
    keys = list(dict.fromkeys(k for t in table for k in t))
    fmt = lambda x: "—" if x is None else f"{100 * x:.2f}"
    print("| Benchmark | " + " | ".join(d.name for d in dirs) + " |")
    print("|---|" + "---:|" * len(dirs))
    for k in keys:
        print(f"| {k} | " + " | ".join(fmt(t.get(k)) for t in table) + " |")


if __name__ == "__main__":
    main()
