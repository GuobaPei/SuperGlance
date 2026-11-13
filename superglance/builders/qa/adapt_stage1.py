#!/usr/bin/env python3
"""Adapt complete source pools without imposing a fixed training mixture."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, ROUND_CEILING
import io
import json
from pathlib import Path
import random
import re
import sqlite3
import zipfile

from PIL import Image
import pyarrow.parquet as pq


SEED = 20260918
REFERENCE = re.compile(r'\b(?:all|none|neither)\b.*\babove\b|\bboth\s+[a-d]\b|\b[a-d]\s+(?:and|or|&)\s+[a-d]\b|\boption\s+[a-d]\b', re.I)
CHOICES = re.compile(r'^([A-Z])\. (.*?)(?=\n[A-Z]\. |\Z)', re.M | re.S)
DOMAINS = {
    'color': 'black white gray red green blue brown yellow orange pink purple'.split(),
    'material': 'wood metal plastic glass brick concrete leather porcelain'.split(),
    'name/furniture': ['chair', 'table', 'bed', 'sofa', 'cabinet', 'shelf', 'bench', 'stool'],
    'name/animal': 'dog cat horse cow sheep elephant giraffe zebra bear goat duck'.split(),
    'name/vehicle': 'car bus truck train van carriage tractor trailer'.split(),
    'name/fruit': 'banana apple orange watermelon pineapple strawberry grape blueberry pear grapefruit pomegranate kiwi mango cherry'.split(),
    'name/vegetable': 'broccoli tomato pepper lettuce onion carrot potato olive spinach basil cucumber asparagus cabbage corn pumpkin'.split(),
}


def json_rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def write_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def short_answer(text):
    return str(text).strip().removesuffix('.').strip()


def numeric_candidates(answer, rng, count=False, question=''):
    value_text = answer.replace(',', '')
    match = re.fullmatch(r'([-+]?\d+(?:\.\d+)?)\s*(%)?', value_text)
    if not match:
        return None
    value = Decimal(match.group(1))
    if count and (value != value.to_integral_value() or value < 0):
        return None
    year = bool(re.search(r'\b(?:which|what)(?: is the)? year\b', question, re.I))
    if count or year:
        start = max(0, int(value) - rng.randrange(5))
        return [str(i) for i in range(start, start + 5)], str(int(value))
    unit = match.group(2) or ''
    precision = Decimal(1).scaleb(value.as_tuple().exponent)
    step = max(precision, (abs(value) * Decimal('0.15')).quantize(precision, rounding=ROUND_CEILING))
    # Vary the GT offset; candidates have the same numeric precision and unit.
    max_offset = min(4, int(value // step)) if value >= 0 else 4
    offset = rng.randrange(max_offset + 1)
    values = [value + (i - offset) * step for i in range(5)]
    return [format(v, 'f') + unit for v in values], format(value, 'f') + unit


def filter_known_answer_conflicts(output):
    known = defaultdict(set)
    for split in ['train', 'dev']:
        for row in json_rows(output / f'{split}.jsonl'):
            known[(row['image_group'], row['question'].casefold())].add(row['correct_answer'].casefold())
    counts, removed, tasks, buckets = Counter(), Counter(), defaultdict(Counter), defaultdict(Counter)
    examples = defaultdict(list)
    with (output / 'deferred.jsonl').open('a') as deferred:
        for split in ['train', 'dev']:
            path = output / f'{split}.jsonl'
            pending = path.with_suffix('.filtered.tmp')
            with pending.open('w') as stream:
                for row in json_rows(path):
                    valid = known[(row['image_group'], row['question'].casefold())]
                    if len(valid & {a.casefold() for a in row['answers']}) > 1:
                        removed[row['source']] += 1
                        write_row(deferred, {'source': row['source'], 'id': row['id'],
                                            'reason': 'multiple_known_correct_candidates', 'known_answers': sorted(valid)})
                        continue
                    write_row(stream, row)
                    counts[split] += 1
                    tasks[split][row['sampling_task']] += 1
                    buckets[f"{split}/{row['sampling_task']}"][row['sampling_bucket']] += 1
                    if len(examples[row['sampling_task']]) < 3:
                        examples[row['sampling_task']].append(row)
            pending.replace(path)
    path = output / 'adaptation_report.json'
    report = json.loads(path.read_text())
    cumulative_removed = Counter(report.get('known_answer_candidate_conflicts_removed', {})) + removed
    report.update(rows=dict(counts), by_task={k: dict(v) for k, v in tasks.items()},
                  by_task_bucket={k: dict(v) for k, v in buckets.items()},
                  known_answer_candidate_conflicts_removed=dict(cumulative_removed))
    report.setdefault('multiple_gt_image_question_groups', sum(len(v) > 1 for v in known.values()))
    for source, count in removed.items():
        report['deferred'][source + '/multiple_known_correct_candidates'] = cumulative_removed[source]
    path.write_text(json.dumps(report, indent=2) + '\n')
    (output / 'examples.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False) + '\n')
    print('Final after known-answer checks:', dict(counts), 'removed', dict(removed), flush=True)


class ImageRegistry:
    def __init__(self, output):
        self.output = output
        self.payloads, self.aliases, self.paths = {}, {}, {}
        self.parents, self.roles, self.files = [], [], []
        self.pool = ThreadPoolExecutor(max_workers=16)
        self.writes = []
        self.store_path = output / 'image_store.sqlite'
        self.store = sqlite3.connect(self.store_path)
        self.store.execute('CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, data BLOB NOT NULL)')

    @staticmethod
    def validate_image(payload):
        with Image.open(io.BytesIO(payload)) as image:
            image.load()

    def finish(self):
        self.pool.shutdown(wait=True)
        for future in self.writes:
            future.result()
        self.store.commit()
        self.store.close()

    def root(self, index):
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def merge(self, first, second):
        first, second = self.root(first), self.root(second)
        if first != second:
            self.parents[second] = first
            self.roles[first].update(self.roles[second])
        return first

    def register(self, payload, role=None, aliases=(), existing=None):
        index = self.payloads.get(payload)
        if index is None:
            index = len(self.parents)
            self.parents.append(index)
            self.roles.append(set())
            target = Path(existing) if existing else self.store_path
            if existing is None:
                self.store.execute('INSERT OR REPLACE INTO images VALUES (?, ?)', (index, payload))
                self.writes.append(self.pool.submit(self.validate_image, payload))
            self.files.append(str(target))
            self.payloads[payload] = index
        for alias in aliases:
            if alias in self.aliases:
                self.merge(index, self.aliases[alias])
            self.aliases[alias] = index
        if role:
            self.roles[self.root(index)].add(role)
        return index

    def existing(self, path, role, aliases=()):
        path = path.resolve()
        if str(path) not in self.paths:
            self.paths[str(path)] = self.register(path.read_bytes(), role, aliases, path)
        else:
            index = self.paths[str(path)]
            self.roles[self.root(index)].add(role)
            for alias in aliases:
                if alias in self.aliases:
                    self.merge(index, self.aliases[alias])
                self.aliases[alias] = index


class Adapter:
    def __init__(self, root, output):
        self.root, self.output = root, output
        (output / 'images').mkdir(parents=True, exist_ok=True)
        self.registry = ImageRegistry(output)
        self.stats, self.deferred = Counter(), Counter()
        self.spool = (output / 'candidates.jsonl').open('w')
        self.pending = (output / 'deferred.jsonl').open('w')
        self.seed_manifests = []
        with zipfile.ZipFile(root / 'supplement_gqa/raw/image_data.json.zip') as archive:
            self.vg = {str(r['image_id']): r for r in json.loads(archive.read('image_data.json'))}

    def aliases(self, vg=None, coco=None):
        result = []
        if vg:
            vg = str(vg)
            result.append('vg:' + vg)
            coco = self.vg.get(vg, {}).get('coco_id') or coco
        if coco:
            result.append('coco:' + str(int(coco)))
        return result

    def seed(self):
        root = self.root
        entries = [(root / 'prepared_v2' / f'{split}.jsonl', split) for split in ['train', 'dev']]
        entries += [(root / 'prepared_v2' / f'sat_{split}_single_image.jsonl', 'excluded')
                    for split in ['val', 'test']]
        entries += [(root / 'perspective_repair_data_20260921' / f'{split}.jsonl', split) for split in ['train', 'dev']]
        for directory in ['supplement_multichoice/prepared', 'supplement_multichoice/gqa',
                          'supplement_multichoice/count', 'supplement_multichoice/ai2d/prepared']:
            entries += [(root / directory / f'{split}.jsonl', split) for split in ['train', 'dev']]
        entries += [(p, 'excluded') for directory in ['eval_external/prepared', 'eval_3dsr/prepared']
                    for p in sorted((root / directory).glob('*.jsonl')) if 'example' not in p.name]
        for path, role in entries:
            if not path.exists():
                continue
            count = 0
            for row in json_rows(path):
                aliases = self.aliases(row.get('source_image_id') if row.get('source') in ['GQA', 'TallyQA'] else None,
                                       row.get('source_coco_id'))
                original = row.get('source_image_url', row.get('source_filename', ''))
                match = re.search(r'(?:/|_)(\d{12})\.(?:jpg|png)$', original)
                if match and ('coco' in original.lower() or 'COCO' in row.get('source', '').upper()):
                    aliases += self.aliases(coco=match.group(1))
                for filename in row.get('images', []):
                    self.registry.existing(path.parent / filename, role, aliases)
                count += 1
            self.seed_manifests.append({'path': str(path), 'role': role, 'rows': count})
        # Preserve every previously assigned GQA image group, not only sampled QA.
        old = json.loads((root / 'supplement_multichoice/image_split.json').read_text())['images']
        self.old_gqa = old
        print('Registered existing splits:', len(self.registry.files), 'image records', flush=True)

    def defer(self, source, identifier, reason, reference=None):
        self.deferred[f'{source}/{reason}'] += 1
        write_row(self.pending, {'source': source, 'id': identifier, 'reason': reason, 'reference': reference})

    def emit(self, image, identifier, source, question, answers, gold, task, bucket=None, **provenance):
        answers = [str(a).strip() for a in answers]
        if len(answers) < 2 or len(answers) > 6 or len({a.casefold() for a in answers}) != len(answers):
            return self.defer(source, identifier, 'invalid_or_duplicate_candidates', provenance)
        if gold not in answers:
            return self.defer(source, identifier, 'gt_not_in_candidates', provenance)
        if any(REFERENCE.search(a) for a in answers):
            return self.defer(source, identifier, 'option_references', provenance)
        permutation = list(range(len(answers)))
        random.Random(f'{SEED}:{identifier}').shuffle(permutation)
        shuffled = [answers[i] for i in permutation]
        binary = {a.casefold() for a in answers} == {'yes', 'no'}
        row = {'id': identifier, '_image': image, 'question': question.strip(),
               'answers': shuffled, 'gold_index': shuffled.index(gold),
               'source': source, 'source_split': 'train', 'source_answers': answers,
               'source_gold_index': answers.index(gold), 'candidate_permutation': permutation,
               'correct_answer': gold, 'label_source': 'dataset_gt',
               'pattern': 'Noul' if binary else 'Choice', 'question_type': task,
               'sampling_task': task, 'sampling_bucket': str(bucket if bucket is not None else gold).casefold(),
               **provenance}
        if binary:
            row['yes_index'] = next(i for i, a in enumerate(shuffled) if a.casefold() == 'yes')
        write_row(self.spool, row)
        self.stats[source] += 1

    def sat(self):
        # Official train already contains the static pool in full.
        for path in sorted((self.root / 'raw_perspective_train').glob('train-*.parquet')):
            row_number = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    identifier = f'sat_v2/train/{path.stem}/{row_number}'
                    row_number += 1
                    if len(row['images']) != 1:
                        self.defer('SAT', identifier, 'multi_image'); continue
                    image = self.registry.register(row['images'][0]['bytes'])
                    gold, answers, question = str(row['correct_answer']), row['answers'], row['question']
                    kind = row['question_type']
                    task = 'sat/' + kind
                    bucket = gold
                    if re.search(r'\bhow many\b', question, re.I):
                        task = 'sat/count'
                    elif kind == 'perspective':
                        task = 'sat/perspective/' + '|'.join(sorted(answers))
                        direction = re.search(r'(?:turned|turn|facing).*?\b(left|right)\b', question.lower())
                        bucket = f'{direction.group(1)}->{gold}' if direction else gold
                    self.emit(image, identifier, 'SAT', question, answers, gold, task, bucket,
                              source_file=str(path), source_row=row_number - 1, source_question_type=kind)
            print('Adapted SAT', path.name, flush=True)

    def gqa(self):
        images = {}
        for path in sorted((self.root / 'supplement_gqa/raw/train_balanced_images').glob('*.parquet')):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    identifier = str(row['id'])
                    role = self.old_gqa.get(identifier, {}).get('split')
                    images[identifier] = self.registry.register(row['image']['bytes'], role, self.aliases(vg=identifier))
            print('Registered GQA image shard', path.name, flush=True)
        path = self.root / 'supplement_gqa/raw/train_balanced_instructions/train-00000-of-00001.parquet'
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                identifier = 'gqa_' + str(row['id'])
                image_id, gold = str(row['imageId']), row['answer']
                if image_id not in images:
                    self.defer('GQA', identifier, 'missing_image'); continue
                structural, semantic = row['types']['structural'], row['types']['semantic']
                task = f'gqa/{structural}/{semantic}'
                if gold in ['yes', 'no']:
                    answers = ['yes', 'no']
                else:
                    last = row['semantic'][-1]
                    domain = last['argument'] if last['operation'] == 'query' else ''
                    if domain == 'name':
                        domain = 'name/' + str(row['groups']['global'])
                    if structural != 'query' or gold not in DOMAINS.get(domain, []):
                        self.defer('GQA', identifier, 'open_answer_without_reliable_candidates'); continue
                    rng = random.Random(identifier)
                    answers = [gold] + rng.sample([a for a in DOMAINS[domain] if a != gold], 3)
                    task = 'gqa/query/' + domain
                self.emit(images[image_id], identifier, 'GQA', row['question'], answers, gold, task,
                          source_id=str(row['id']), source_image_id=image_id,
                          source_answer=gold, source_semantic=row['semantic'], source_types=row['types'])
        print('Adapted GQA', self.stats['GQA'], flush=True)

    def tally_index(self):
        with zipfile.ZipFile(self.root / 'supplement_gqa/raw/tallyqa.zip') as archive:
            rows = json.loads(archive.read('train.json'))
        index = defaultdict(set)
        for row in rows:
            index[(row['question'].strip(), str(row['answer']))].add(row['image'])
        return index

    def cauldron(self, family):
        directory = (self.root / 'supplement_multichoice/ai2d/raw' if family == 'ai2d'
                     else self.root / 'stage1_data_expansion_20260921/raw' / family)
        tally = self.tally_index() if family == 'tallyqa' else None
        for file_number, path in enumerate(sorted(directory.glob('train_*.parquet'))):
            row_number = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    group_id = f'{family}_{file_number}_{row_number}'
                    row_number += 1
                    if len(row['images']) != 1:
                        self.defer(family, group_id, 'multi_image'); continue
                    aliases = []
                    if tally is not None:
                        matches = None
                        for turn in row['texts']:
                            key = (turn['user'].split('\n')[0].strip(), short_answer(turn['assistant']))
                            possibilities = tally.get(key, set())
                            matches = possibilities if matches is None else matches & possibilities
                        if not matches or len(matches) != 1:
                            self.defer(family, group_id, 'ambiguous_original_image_identity'); continue
                        original = next(iter(matches))
                        if original.startswith('VG_'):
                            aliases = self.aliases(vg=Path(original).stem)
                        else:
                            match = re.search(r'(\d{12})\.', original)
                            aliases = self.aliases(coco=match.group(1)) if match else []
                    image = self.registry.register(row['images'][0]['bytes'], aliases=aliases)
                    for turn_number, turn in enumerate(row['texts']):
                        identifier = f'{group_id}_{turn_number}'
                        question = turn['user'].split('\n')[0].removeprefix('Question: ').strip()
                        gold = short_answer(turn['assistant'])
                        rng = random.Random(identifier)
                        task = family
                        if family == 'ai2d':
                            if '\nChoices:\n' not in turn['user']:
                                self.defer(family, identifier, 'unparsed_choices'); continue
                            raw_choices = turn['user'].split('\nChoices:\n', 1)[1].removesuffix('\nAnswer with the letter.')
                            parsed = CHOICES.findall(raw_choices)
                            labels, answers = [p[0] for p in parsed], [p[1] for p in parsed]
                            label = gold.removeprefix('Answer: ').strip()
                            if label not in labels:
                                self.defer(family, identifier, 'unparsed_answer'); continue
                            gold = answers[labels.index(label)]
                        elif gold.casefold() in ['yes', 'no']:
                            answers, gold = ['yes', 'no'], gold.lower()
                            task += '/verify'
                        else:
                            count = family == 'tallyqa' or (family == 'textvqa' and re.search(r'^how many\b', question, re.I))
                            result = numeric_candidates(gold, rng, count=bool(count), question=question)
                            if result is None or (family == 'textvqa' and not count):
                                self.defer(family, identifier, 'open_answer_without_reliable_candidates',
                                           {'source_file': str(path), 'source_row': row_number - 1, 'turn': turn_number}); continue
                            answers, gold = result
                            task += '/count' if count else '/numeric'
                        self.emit(image, identifier, family, question, answers, gold, task,
                                  source_file=str(path), source_row=row_number - 1, source_turn=turn_number,
                                  source_answer=turn['assistant'])
            print('Adapted', family, file_number, self.stats[family], flush=True)

    def finalize(self):
        self.spool.close(); self.pending.close()
        self.registry.finish()
        registry, rng = self.registry, random.Random(SEED)
        splits, role_conflicts = {}, 0
        for index in range(len(registry.parents)):
            group = registry.root(index)
            if group in splits:
                continue
            roles = registry.roles[group]
            role_conflicts += 'train' in roles and 'dev' in roles
            splits[group] = ('excluded' if 'excluded' in roles else 'dev' if 'dev' in roles
                             else 'train' if 'train' in roles else 'dev' if rng.random() < 0.05 else 'train')
        seen, seen_ids = set(), set()
        counts, tasks, buckets = Counter(), defaultdict(Counter), defaultdict(Counter)
        examples = defaultdict(list)
        streams = {s: (self.output / f'{s}.jsonl').open('w') for s in ['train', 'dev']}
        duplicate, excluded = 0, 0
        for row in json_rows(self.output / 'candidates.jsonl'):
            image = row.pop('_image')
            group = registry.root(image)
            split = splits[group]
            if split == 'excluded':
                excluded += 1; continue
            key = (group, row['question'].casefold(), row['correct_answer'].casefold())
            if key in seen:
                duplicate += 1; continue
            seen.add(key)
            assert row['id'] not in seen_ids
            seen_ids.add(row['id'])
            row.update(images=[registry.files[image]], image_id=f'pool_image_{image}',
                       image_group=f'pool_group_{group}', pool_split=split)
            if registry.files[image] == str(registry.store_path):
                row['image_store_key'] = image
            write_row(streams[split], row)
            counts[split] += 1
            tasks[split][row['sampling_task']] += 1
            buckets[f"{split}/{row['sampling_task']}"][row['sampling_bucket']] += 1
            if len(examples[row['sampling_task']]) < 3:
                examples[row['sampling_task']].append(row)
        for stream in streams.values():
            stream.close()
        report = {'status': 'complete', 'rows': dict(counts), 'by_task': {k: dict(v) for k, v in tasks.items()},
                  'by_task_bucket': {k: dict(v) for k, v in buckets.items()}, 'source_candidates': dict(self.stats),
                  'deferred': dict(self.deferred), 'duplicate_qa_removed': duplicate,
                  'external_image_overlap_removed_qa': excluded, 'existing_train_dev_group_conflicts': role_conflicts,
                  'registered_image_records': len(registry.files), 'registered_groups': len(splits),
                  'split_seed_manifests': self.seed_manifests,
                  'split_policy': 'Existing dev wins over train; external eval excluded; existing train stays train; new image groups reserve 5% dev.',
                  'image_identity': 'Exact encoded bytes plus known VG/COCO identities; no near-duplicate scene detection.',
                  'image_storage': 'Original bytes in SQLite keyed by integer ID; existing image files reused.',
                  'sampling': 'No balancing or source quotas applied here; all eligible unique QA retained.'}
        (self.output / 'adaptation_report.json').write_text(json.dumps(report, indent=2) + '\n')
        (self.output / 'examples.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False) + '\n')
        print('FINAL', dict(counts), 'duplicates', duplicate, 'external excluded', excluded, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    adapter = Adapter(args.root, args.output)
    adapter.seed()
    adapter.sat()
    adapter.gqa()
    for family in ['ai2d', 'tallyqa', 'chartqa', 'textvqa']:
        adapter.cauldron(family)
    adapter.finalize()
    filter_known_answer_conflicts(args.output)


if __name__ == '__main__':
    main()
