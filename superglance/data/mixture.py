"""Epoch-seeded mixture sampling, before Accelerate shards the shared index stream."""
from collections import Counter, defaultdict

import torch
from torch.utils.data import Sampler


class MixtureSampler(Sampler):
    def __init__(self, rows, config, seed):
        self.seed, self.epoch = seed, 0
        self.num_samples = int(config.get('epoch_samples', len(rows)))
        sources = config['source_weights']
        power = float(config.get('answer_balance_power', 0.5))
        task_power = float(config.get('task_balance_power', 0.5))
        if not 0 <= power <= 1:
            raise ValueError('answer_balance_power must be between 0 and 1')
        sizes = Counter((r['source'], r['sampling_task'], r['sampling_bucket']) for r in rows)
        tasks, buckets = defaultdict(set), defaultdict(dict)
        for (source, task, bucket), size in sizes.items():
            tasks[source].add(task)
            buckets[(source, task)][bucket] = size
        missing = {s for s, w in sources.items() if w > 0} - tasks.keys()
        if missing:
            raise ValueError(f'Configured sources missing from training pool: {sorted(missing)}')
        normalizers = {key: sum(size ** (1 - power) for size in pool.values())
                       for key, pool in buckets.items()}
        task_masses = {key: config.get('task_weights', {}).get(key[1], sum(pool.values()) ** (1 - task_power))
                       for key, pool in buckets.items()}
        source_norms = {source: sum(task_masses[(source, task)] for task in source_tasks)
                        for source, source_tasks in tasks.items()}
        weights = []
        for row in rows:
            source, task, bucket = row['source'], row['sampling_task'], row['sampling_bucket']
            size = sizes[(source, task, bucket)]
            weights.append(sources.get(source, 0) * task_masses[(source, task)] / source_norms[source] * size ** (-power)
                           / normalizers[(source, task)])
        self.weights = torch.tensor(weights, dtype=torch.double)
        if self.weights.sum() <= 0:
            raise ValueError('At least one source must have a positive weight')
        self.weights /= self.weights.sum()
        self.report = {
            'epoch_samples': self.num_samples, 'replacement': True,
            'source_weights': sources, 'task_balance_power': task_power,
            'task_weights': config.get('task_weights', {}),
            'within_source': 'task_size ** (1 - task_balance_power), with optional task_weights overrides',
            'answer_balance_power': power,
            'bucket_probability': 'proportional to bucket_size ** (1 - answer_balance_power)',
            'task_counts': {s: len(t) for s, t in tasks.items()},
            'maximum_expected_exposures_per_qa': float(self.weights.max() * self.num_samples),
            'distributed': 'Identical seeded stream on all ranks, sharded once by Accelerate.',
            'loss_weighting': 'none; each original QA contributes the mean of its two views',
        }

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True,
                                      generator=generator).tolist())

    def __len__(self):
        return self.num_samples
