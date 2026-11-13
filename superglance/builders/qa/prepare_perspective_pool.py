#!/usr/bin/env python3
"""Use native single-image left/right perspective GT from SAT's train split."""
import argparse
from collections import Counter, defaultdict
import io
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq
from PIL import Image
from superglance.builders.qa.prepare_sat import shuffle_candidate_order
from superglance.builders.qa.rows_io import read_rows, write_rows


def turn_direction(row):
    match = re.search(r'(?:turned|turn|facing).*?\b(left|right)\b', row['question'].lower())
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--static', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-size', type=int, default=768)
    parser.add_argument('--dev-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=20260921)
    args = parser.parse_args()
    protected_paths = set()
    for split in ['dev', 'sat_val_single_image', 'sat_test_single_image']:
        protected_paths.update(image for row in read_rows(args.static / f'{split}.jsonl')
                               for image in row['images'])
    protected_bytes = {Path(path).read_bytes() for path in protected_paths}
    static_train_paths = {image for row in read_rows(args.static / 'train.jsonl')
                          for image in row['images']}
    static_train_bytes = {Path(path).read_bytes() for path in static_train_paths}
    image_ids, payloads, rows = {}, {}, []
    scanned, rejected = Counter(), Counter()
    for path in sorted(args.source.glob('train-*.parquet')):
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for raw in batch.to_pylist():
                row_number = offset
                offset += 1
                scanned[f"{raw['question_type']}/{len(raw['images'])}_images"] += 1
                if raw['question_type'] != 'perspective' or len(raw['images']) != 1:
                    continue
                if {answer.strip().lower() for answer in raw['answers']} != {'left', 'right'}:
                    rejected['other_perspective_subtype'] += 1
                    continue
                payload = raw['images'][0]['bytes']
                if payload in protected_bytes:
                    rejected['existing_dev_val_test_image'] += 1
                    continue
                if payload not in image_ids:
                    identity = f'perspective_image_{len(image_ids):06d}'
                    image_ids[payload] = identity
                    payloads[identity] = payload
                row = {key: value for key, value in raw.items() if key != 'images'}
                row.update(id=f'sat_v2/train/{path.stem}/{row_number}',
                           image_id=image_ids[payload], source='array/SAT-v2', source_split='train',
                           source_file=f'data/{path.name}', source_row=row_number,
                           pattern='Choice', gold_index=raw['answers'].index(raw['correct_answer']),
                           label_source='dataset_gt', conversion='native_candidates',
                           repair_family='perspective_left_right')
                rows.append(row)
        print('Scanned', path.name, flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row['image_id']].append(row)
    identities = list(groups)
    rng = random.Random(args.seed)
    rng.shuffle(identities)
    dev, reserved = [], set()
    for identity in identities:
        if len(dev) >= args.dev_size:
            break
        if payloads[identity] in static_train_bytes:
            continue
        group = list(groups[identity])
        rng.shuffle(group)
        dev.extend(group[:args.dev_size - len(dev)])
        reserved.add(identity)
    candidates = [row for row in rows if row['image_id'] not in reserved]
    assert len(dev) == args.dev_size, f'Only {len(dev)} eligible held-out perspective questions'
    # Four cells prevent both the same-direction and opposite-direction shortcut.
    # Rare real source QA are repeated deliberately; unique counts are reported.
    cells = defaultdict(list)
    for row in candidates:
        direction = turn_direction(row)
        assert direction in ('left', 'right'), row['question']
        cells[(direction, row['correct_answer'].lower())].append(row)
    assert len(cells) == 4 and args.train_size % 4 == 0
    per_cell = args.train_size // 4
    train = []
    for key in sorted(cells):
        pool = list(cells[key])
        rng.shuffle(pool)
        selected = pool[:per_cell]
        selected.extend(rng.choices(pool, k=per_cell - len(selected)))
        train.extend(selected)
    rng.shuffle(train)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'images').mkdir(exist_ok=True)
    paths = {}
    for identity in {row['image_id'] for row in train + dev}:
        payload = payloads[identity]
        extension = Image.open(io.BytesIO(payload)).format.lower()
        image_path = (args.output / 'images' / f'{identity}.{extension}').resolve()
        image_path.write_bytes(payload)
        paths[identity] = str(image_path)
    result = {}
    for index, (name, selected) in enumerate([('train', train), ('dev', dev)]):
        order_rng = random.Random(args.seed + 100 + index)
        selected = [shuffle_candidate_order(row, order_rng) for row in selected]
        for row in selected:
            row['images'] = [paths[row['image_id']]]
        write_rows(args.output / f'{name}.jsonl', selected)
        result[name] = {'qa': len(selected), 'unique_qa': len({r['id'] for r in selected}),
                        'images': len({r['image_id'] for r in selected}),
                        'gold_answers': dict(Counter(r['correct_answer'] for r in selected)),
                        'gold_positions': dict(Counter(r['gold_index'] for r in selected)),
                        'turn_and_gold': dict(Counter(f'{turn_direction(r)}/{r["correct_answer"].lower()}'
                                                       for r in selected))}
    report = {'source': str(args.source), 'seed': args.seed, 'scanned': dict(scanned),
              'excluded': dict(rejected), 'eligible_left_right_qa': len(rows),
              'eligible_images': len(groups), 'splits': result,
              'train_available_by_turn_and_gold': {f'{key[0]}/{key[1]}': len(pool)
                                                   for key, pool in cells.items()},
              'sampling': 'Equal exposure to the four turn-direction/GT cells. Rare real training QA '
              'are oversampled, with duplicate IDs retained; this is not 768 unique new examples.',
              'split_policy': 'Group by exact encoded image bytes. Exclude existing static dev and SAT val/test '
              'images. New perspective dev also excludes all existing static training images. '
              'Source has no scene IDs; differently rendered views of the same scene cannot be excluded by this check.',
              'supervision': 'Unmodified official train GT; no turn-word inversion or benchmark labels.'}
    (args.output / 'preparation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
