#!/usr/bin/env python3
"""Append two image shards for the expanded multiple-choice data pool: GQA's train_balanced_images shards 2 and 3 (0 and
1 come with prepare_supplement.py). usage: python -m superglance.builders.qa.download_more_gqa ROOT"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from superglance.builders.qa.prepare_supplement import ENDPOINT, GQA_REPO, download


if __name__ == "__main__":
    raw = Path(sys.argv[1]) / "supplement_gqa/raw"
    files = [f"train_balanced_images/train-{i:05d}-of-00021.parquet" for i in (2, 3)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda name: download(ENDPOINT, GQA_REPO, name, raw / name), files))
    print("Two additional GQA image shards are ready.", flush=True)
