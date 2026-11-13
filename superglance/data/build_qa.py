"""Builds a stage-1 or stage-2 trainer data directory from the released QA dataset (its Hugging Face layout):

  stage<N>/train/<source>.jsonl   one file per source with its released rows (none for a source held in full)
  counts.json                     per source: rows (the paper's), uploaded (released), held (why rows were held, or null)
  images/text/blank.png, build/build_images.py (+ build/README.md)

train.jsonl = the stage's source files concatenated and sorted by `row`. It is the paper's training file row for row only
when every source is complete: option shifts and mixture draws are keyed by line number, so a held source must be rebuilt
(superglance/builders/qa, see the README) and its file put back as stage<N>/train/<source>.jsonl with every row; dropping
it changes the other rows' draws. Without --allow-incomplete nothing is written while a source is incomplete. The images
come from each source's public release: run build/build_images.py <out> afterwards (build/README.md). The development
split is not released: without dev.jsonl the trainer runs no evaluation and keeps the last step, the checkpoint both of
the paper's QA runs selected.

  python -m superglance.data.build_qa --hf <dataset snapshot> --stage 1 --out <dir> [--allow-incomplete]
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", type=Path, required=True, help="local snapshot of the QA dataset")
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-incomplete", action="store_true", help="write train.jsonl even if a source misses rows "
                    "(not the paper's file: a smoke test only)")
    a = ap.parse_args()
    counts = json.loads((a.hf / "counts.json").read_text())[f"stage{a.stage}"]
    rows, short = [], {}
    for source, c in sorted(counts.items()):
        path = a.hf / f"stage{a.stage}" / "train" / f"{source}.jsonl"
        got = [json.loads(line) for line in path.open() if line.strip()] if path.exists() else []
        assert len(got) in (c["uploaded"], c["rows"]), (source, len(got), c)   # as released, or complete after a rebuild
        if len(got) < c["rows"]:
            short[source] = (c["rows"] - len(got), c.get("held"))
        rows += got
    for source, (n, why) in short.items():
        print(f"incomplete: {source} misses {n} of {counts[source]['rows']} rows ({why})")
    if short and not a.allow_incomplete:
        raise SystemExit(f"{len(short)} sources are incomplete: rebuild them (README, 'Stages 1 and 2: the QA dataset') "
                         f"and put the full files under {a.hf / f'stage{a.stage}' / 'train'}; nothing written")
    rows.sort(key=lambda r: r["row"])
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "train.jsonl").open("w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    blank = a.hf / "images" / "text" / "blank.png"
    if blank.exists():
        (a.out / "images" / "text").mkdir(parents=True, exist_ok=True)
        shutil.copy2(blank, a.out / "images" / "text" / "blank.png")
    print(f"stage {a.stage}: {len(rows)} training rows from {len(counts) - len(short)} complete and {len(short)} incomplete "
          f"sources -> {a.out}\nnext: python {a.hf / 'build' / 'build_images.py'} {a.out}")


if __name__ == "__main__":
    main()
