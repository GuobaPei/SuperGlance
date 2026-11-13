"""Builds a stage-1 or stage-2 trainer data directory from the released QA dataset (its Hugging Face layout):

  stage<N>/train/<source>.jsonl   one file per source (a held source has none; counts.json says why)
  counts.json, images/text/blank.png, build/build_images.py (+ build/README.md)

train.jsonl = the stage's source files concatenated and sorted by `row`, which with every source present is the
paper's training file row for row (its image paths point into <out>/images/). The images come from each source's public
release: run build/build_images.py <out> afterwards, as the dataset's build/README.md describes. The development split
is not released: without dev.jsonl the trainer runs no evaluation and keeps the last step, the checkpoint both of the
paper's QA runs selected.

  python -m superglance.data.build_qa --hf <dataset snapshot> --stage 1 --out <dir>
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
    a = ap.parse_args()
    counts = json.loads((a.hf / "counts.json").read_text())[f"stage{a.stage}"]
    rows = []
    for source, c in sorted(counts.items()):
        path = a.hf / f"stage{a.stage}" / "train" / f"{source}.jsonl"
        if c.get("held") or not path.exists():
            print(f"held: {source} ({c.get('held') or 'no file'}); rebuild it as build/README.md says, or the stage's "
                  f"mixture (configs/) stops on the missing source")
            continue
        got = [json.loads(line) for line in path.open() if line.strip()]
        assert len(got) == c["uploaded"], (source, len(got), c["uploaded"])
        rows += got
    rows.sort(key=lambda r: r["row"])
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "train.jsonl").open("w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    blank = a.hf / "images" / "text" / "blank.png"
    if blank.exists():
        (a.out / "images" / "text").mkdir(parents=True, exist_ok=True)
        shutil.copy2(blank, a.out / "images" / "text" / "blank.png")
    print(f"stage {a.stage}: {len(rows)} training rows from {len(counts)} sources -> {a.out}\n"
          f"next: python {a.hf / 'build' / 'build_images.py'} {a.out}")


if __name__ == "__main__":
    main()
