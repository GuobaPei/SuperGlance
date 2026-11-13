#!/usr/bin/env python3
"""Row-file reading and writing of the SAT-derived seed manifests (prepare_perspective_pool.py)."""
import json


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        row['images'] = [str((path.parent / image).resolve()) for image in row['images']]
    return rows


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
