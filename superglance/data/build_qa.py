"""Builds a stage-1 or stage-2 trainer data directory: the released rows from the QA dataset (its Hugging Face layout), and
the rows we used as published from a rebuild of the pool from the original datasets.

  stage<N>/train/<source>.jsonl   the released rows of a source, each with `row`: its line in the paper's train.jsonl
  counts.json                     per source: rows (trained on), uploaded (released), held (why the others are not released)
  images/text/blank.png, build/build_images.py (+ build/README.md)

A row is released when we generated its question or options. The questions used as published, with the source's own
question and options, are not: all of SAT, Social IQa, PIQA and AI2D, and GQA's and ChartQA's own yes/no questions. They
come from a rebuild of the pool with scripts/build_qa.sh (all steps up to g for stage 1, s2 for stage 2). The whole pool is
rebuilt because the train/dev split, the de-duplication and every line number depend on all sources; option shifts and
mixture draws are keyed by line number.

train.jsonl takes line i from the dataset when a released row has row = i, else line i of the rebuilt pool's train.jsonl.
Checks: every source of counts.json has `uploaded` released rows and `rows` rebuilt rows; each released row's id and source
equal the rebuilt pool's at its line, so the rebuild has the paper's order. A rebuilt row keeps its image (a path is made
absolute against the rebuilt pool, as the trainer reads it); a released row's image comes from build/build_images.py.
The development split is not released: without dev.jsonl the trainer runs no evaluation and keeps the last step, the
checkpoint both of the paper's QA runs selected.

  python -m superglance.data.build_qa --hf <dataset snapshot> --stage 1 --rebuilt <ROOT>/stage1_adapted_g_20260926 --out data/stage1
  python -m superglance.data.build_qa --hf <dataset snapshot> --stage 2 --rebuilt <ROOT>/stage2_drivelm_20260925 --out data/stage2
  python <dataset snapshot>/build/build_images.py <dataset snapshot> data/stage1 --stage stage1 ...   # the released rows' images
Without --rebuilt nothing is written while a source is not released in full; --allow-incomplete then writes the released
rows alone (not the paper's file: a smoke test).
"""
import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

READ = ("source", "question", "answers", "gold_index", "sampling_task", "sampling_bucket")   # the fields the trainer reads


def jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", type=Path, required=True, help="local snapshot of the QA dataset")
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--rebuilt", type=Path, help="the pool directory scripts/build_qa.sh rebuilt (its train.jsonl)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-incomplete", action="store_true", help="without --rebuilt, write the released rows alone")
    a = ap.parse_args()
    counts = json.loads((a.hf / "counts.json").read_text())[f"stage{a.stage}"]
    released = {}
    for source, c in counts.items():
        path = a.hf / f"stage{a.stage}" / "train" / f"{source}.jsonl"
        got = jsonl(path) if path.exists() else []
        assert len(got) == c["uploaded"] and all(r["source"] == source for r in got), (source, len(got), c)
        for r in got:
            assert r["row"] not in released, ("two released rows on line", r["row"])
            released[r["row"]] = r
    total = sum(c["rows"] for c in counts.values())
    assert all(0 <= i < total for i in released), "a released row's line is past the stage's row count"
    held = {s: c["rows"] - c["uploaded"] for s, c in counts.items() if c["rows"] > c["uploaded"]}
    if a.rebuilt is None:
        for s, n in held.items():
            print(f"not released: {s}, {n} of {counts[s]['rows']} rows ({counts[s]['held']})")
        if held and not a.allow_incomplete:
            raise SystemExit("rebuild the pool with scripts/build_qa.sh and pass it as --rebuilt "
                             "(README, 'Stages 1 and 2: the QA dataset'); nothing written")
        rows, n_rebuilt, differ = [released[i] for i in sorted(released)], Counter(), 0
    else:
        rebuilt = jsonl(a.rebuilt / "train.jsonl")
        per = Counter(r["source"] for r in rebuilt)
        want = Counter({s: c["rows"] for s, c in counts.items()})
        assert per == want, {s: (per[s], want[s]) for s in set(per) | set(want) if per[s] != want[s]}
        rows, n_rebuilt, differ = [], Counter(), 0
        for i, r in enumerate(rebuilt):
            mine = released.get(i)
            if mine is not None:   # the rebuild must have the paper's order: same row id and source on this line
                assert (r["id"], r["source"]) == (mine["id"], mine["source"]), (f"line {i}", r["id"], mine["id"])
                differ += any(r.get(k) != mine[k] for k in READ)
                rows.append(mine)
                continue
            img = r["images"][0]
            if not os.path.isabs(img):
                r["images"] = [str(a.rebuilt.absolute() / img)] + r["images"][1:]
            n_rebuilt[r["source"]] += 1
            rows.append(r)
        print(f"{differ} released rows differ from the rebuild in a field the trainer reads (the released rows are used)")
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "train.jsonl").open("w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    blank = a.hf / "images" / "text" / "blank.png"
    if blank.exists():
        (a.out / "images" / "text").mkdir(parents=True, exist_ok=True)
        shutil.copy2(blank, a.out / "images" / "text" / "blank.png")
    for s, c in counts.items():
        print(f"{s:20s} {c['rows']:>8,} rows: {c['uploaded']:>8,} released, {n_rebuilt[s]:>8,} rebuilt")
    print(f"stage {a.stage}: {len(rows):,} training rows of {total:,} -> {a.out}\n"
          f"next: python {a.hf / 'build' / 'build_images.py'} {a.hf} {a.out} --stage stage{a.stage} (sources: build/README.md)")


if __name__ == "__main__":
    main()
