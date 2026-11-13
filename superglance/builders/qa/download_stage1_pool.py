#!/usr/bin/env python3
"""Download the expanded training sources and inventory their native records."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time

import pyarrow.parquet as pq
import requests
from huggingface_hub.constants import ENDPOINT


def listing(endpoint, repo, directory):
    response = requests.get(
        f'{endpoint}/api/datasets/{repo}/tree/main/{directory}',
        params={'limit': 1000}, timeout=60)
    response.raise_for_status()
    return sorted((item for item in response.json()
                   if item['type'] == 'file' and item['path'].endswith('.parquet')),
                  key=lambda item: item['path'])


def inventory(path, family):
    parquet = pq.ParquetFile(path)
    result = {'records': parquet.metadata.num_rows, 'columns': parquet.schema_arrow.names}
    if family in {'ai2d', 'tallyqa', 'chartqa', 'textvqa'}:
        questions, sources = 0, Counter()
        for batch in parquet.iter_batches(batch_size=512, columns=['texts']):
            for row in batch.to_pylist():
                questions += len(row['texts'])
                sources.update(turn.get('source', 'unknown') for turn in row['texts'])
        result.update(qa=questions, turn_sources=dict(sources))
    elif family == 'sat':
        tasks = Counter()
        for batch in parquet.iter_batches(batch_size=4096, columns=['question_type']):
            tasks.update(row['question_type'] for row in batch.to_pylist())
        result['question_types'] = dict(tasks)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--endpoint', default=ENDPOINT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = args.root
    output = root / 'stage1_data_expansion_20260921'
    output.mkdir(parents=True, exist_ok=True)
    tasks = []

    def append(family, repo, item, target, split, ordinal):
        tasks.append({'family': family, 'repo': repo, 'source_split': split,
                      'source_shard_ordinal': ordinal, 'expected_bytes': item['size'],
                      'path': str(target), '_source_path': item['path']})

    repo = 'array/SAT-v2'
    for ordinal, item in enumerate(listing(args.endpoint, repo, 'data')):
        name = Path(item['path']).name
        if name.startswith('static-'):
            append('sat', repo, item, root / 'raw/data' / name, 'static', ordinal)
        elif name.startswith('train-'):
            append('sat', repo, item, root / 'raw_perspective_train' / name, 'train', ordinal)

    repo = 'lmms-lab-encoder/GQA'
    for split in ['train_balanced_images', 'train_balanced_instructions']:
        for ordinal, item in enumerate(listing(args.endpoint, repo, split)):
            append('gqa', repo, item, root / 'supplement_gqa/raw' / item['path'], split, ordinal)

    repo = 'HuggingFaceM4/the_cauldron'
    for family in ['ai2d', 'tallyqa', 'chartqa', 'textvqa']:
        for ordinal, item in enumerate(listing(args.endpoint, repo, family)):
            if not Path(item['path']).name.startswith('train-'):
                continue
            target = (root / 'supplement_multichoice/ai2d/raw' / f'train_{ordinal}.parquet'
                      if family == 'ai2d' else output / 'raw' / family / f'train_{ordinal:03d}.parquet')
            append(family, repo, item, target, 'train', ordinal)

    def public(task):
        return {key: value for key, value in task.items() if not key.startswith('_')}

    plan = [public(task) for task in tasks]
    (output / 'download_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    total = sum(task['expected_bytes'] for task in tasks)
    pending = sum(task['expected_bytes'] for task in tasks if not Path(task['path']).exists())
    print(f'Plan: {len(tasks)} files, {total / 1e9:.2f} GB total, {pending / 1e9:.2f} GB missing', flush=True)

    def download(task):
        target = Path(task['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        reused = target.exists() and target.stat().st_size == task['expected_bytes']
        if not reused:
            partial = target.with_suffix('.part')
            url = f"{args.endpoint}/datasets/{task['repo']}/resolve/main/{task['_source_path']}"
            result = subprocess.run(['curl', '-L', '--fail', '--silent', '--show-error',
                                     '--retry', '4', '--retry-delay', '3', '--connect-timeout', '30',
                                     '--max-time', '1800', '-C', '-', '-o', str(partial), url],
                                    capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"download failed for {task['family']} shard {task['source_shard_ordinal']}: curl {result.returncode}")
            if partial.stat().st_size != task['expected_bytes']:
                raise RuntimeError(f"Incomplete download: {target.name}")
            partial.replace(target)
        row = public(task)
        row.update(status='ready', reused=reused, **inventory(target, task['family']))
        return row

    completed, errors = [], []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending_jobs = {pool.submit(download, task): task for task in tasks}
        for future in as_completed(pending_jobs):
            task = pending_jobs[future]
            try:
                row = future.result()
                completed.append(row)
                print(f"Ready {len(completed)}/{len(tasks)}: {row['family']} {row['source_split']} shard {row['source_shard_ordinal']}, {row['records']} records", flush=True)
            except Exception as error:
                errors.append({'family': task['family'], 'path': task['path'], 'error': str(error)})
                print(f"Failed: {task['family']} shard {task['source_shard_ordinal']}", flush=True)
            report = {'status': 'running', 'ready_files': completed, 'errors': errors,
                      'planned_files': len(tasks), 'elapsed_seconds': round(time.time() - started)}
            (output / 'download_report.json').write_text(json.dumps(report, indent=2) + '\n')
    report['status'] = 'complete' if not errors else 'incomplete'
    (output / 'download_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"Finished: {report['status']}; ready={len(completed)}, errors={len(errors)}", flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
