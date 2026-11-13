#!/usr/bin/env python3
"""Check actual adapted GT, two-order mappings, split separation and mixture sampling."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
from pathlib import Path

from accelerate.data_loader import prepare_data_loader
import torch
from torch.utils.data import DataLoader

from superglance.data.mixture import MixtureSampler
from superglance.data.qa_pairs import permute_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    groups, ids, rows = {}, set(), []
    sources, tasks, checked_images, count = Counter(), Counter(), set(), 0
    image_keys = {}
    known_answers = defaultdict(set)
    for split in ['train', 'dev']:
        groups[split] = set()
        with (args.data / f'{split}.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                assert row['id'] not in ids
                ids.add(row['id'])
                groups[split].add(row['image_group'])
                answers, gold = row['answers'], row['gold_index']
                assert answers[gold] == row['correct_answer']
                known_answers[(row['image_group'], row['question'].casefold())].add(row['correct_answer'].casefold())
                assert row['source_answers'][row['source_gold_index']] == row['correct_answer']
                assert [row['source_answers'][i] for i in row['candidate_permutation']] == answers
                assert len({a.casefold() for a in answers}) == len(answers)
                for shift in range(1, len(answers)):
                    order = list(range(shift, len(answers))) + list(range(shift))
                    view = permute_row(row, order)
                    assert view['answers'][view['gold_index']] == row['correct_answer']
                    if 'yes_index' in view:
                        assert view['answers'][view['yes_index']].lower() == 'yes'
                path = Path(row['images'][0])
                checked_images.add(str(path))
                if 'image_store_key' in row:
                    image_keys.setdefault(str(path), set()).add(row['image_store_key'])
                count += 1
                if split == 'train':
                    rows.append(row)
                    sources[row['source']] += 1
                    tasks[row['sampling_task']] += 1
    assert groups['train'].isdisjoint(groups['dev'])
    for split in ['train', 'dev']:
        for line in (args.data / f'{split}.jsonl').open():
            row = json.loads(line)
            golds = known_answers[(row['image_group'], row['question'].casefold())]
            assert len(golds & {a.casefold() for a in row['answers']}) == 1
    with ThreadPoolExecutor(max_workers=16) as pool:
        assert all(pool.map(lambda path: Path(path).is_file(), checked_images))
    for path, keys in image_keys.items():
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as store:
            stored = {r[0] for r in store.execute('SELECT id FROM images')}
            assert keys <= stored
    config = {'source_weights': {s: 1 for s in sources}, 'epoch_samples': 20000, 'answer_balance_power': 0.5}
    sampler = MixtureSampler(rows, config, seed=20260921)
    first = list(sampler)
    assert list(sampler) == first
    sampler.set_epoch(1)
    assert list(sampler) != first
    sampler.set_epoch(0)
    sampled = Counter(rows[i]['source'] for i in first)
    target = len(first) / len(sources)
    assert all(abs(n - target) < 0.1 * target for n in sampled.values())
    # Accelerate must shard the common stream once, not independently resample ranks.
    shards = []
    for rank in range(2):
        loader = DataLoader(range(len(rows)), batch_size=4, sampler=MixtureSampler(rows, config, 20260921))
        prepared = prepare_data_loader(loader, device=torch.device('cpu'), num_processes=2,
                                       process_index=rank, split_batches=False, put_on_device=False)
        shards.append([batch.tolist() for batch in prepared])
    reconstructed = [i for pair in zip(*shards) for batch in pair for i in batch]
    assert reconstructed == first
    report = {'status': 'passed', 'qa_checked': count, 'image_paths_checked': len(checked_images),
              'train_sources': dict(sources), 'train_tasks': dict(tasks),
              'split_groups_disjoint': True, 'all_cyclic_gt_mappings_checked': True,
              'no_multiple_known_correct_candidates': True,
              'stored_image_keys_checked': sum(map(len, image_keys.values())),
              'sampler_draws': dict(sampled), 'sampler_epoch_reproducible': True,
              'two_rank_accelerate_sharding_matches_single_stream': True,
              'sampler_config_for_check_only': config,
              'scope': 'GT mapping and structural checks; candidate semantics also require sample review.'}
    (args.data / 'validation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
