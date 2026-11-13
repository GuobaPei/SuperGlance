#!/usr/bin/env python3
"""Download the SAT-v2 static pilot and official held-out parquet files."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.constants import ENDPOINT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="Raw SAT-v2 directory on the data disk")
    parser.add_argument("--endpoint", default=ENDPOINT, help="Hugging Face endpoint (HF_ENDPOINT, else the Hub)")
    args = parser.parse_args()
    target = snapshot_download(
        repo_id="array/SAT-v2",
        repo_type="dataset",
        endpoint=args.endpoint,
        local_dir=args.output,
        allow_patterns=[
            "data/static-*.parquet",
            "data/val-*.parquet",
            "data/test-*.parquet",
            "README.md",
        ],
    )
    print(f"SAT-v2 static/val/test downloaded to {target}")


if __name__ == "__main__":
    main()
