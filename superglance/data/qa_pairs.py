"""Two views of each QA, with probabilities aligned by answer identity."""
import random
from collections import Counter

import torch

from superglance.data.qa import DecisionCollator, DecisionDataset


def permute_row(row, permutation):
    """permutation[new position] = old position; GT follows its content."""
    result = dict(row)
    inverse = [permutation.index(i) for i in range(len(permutation))]
    result["answers"] = [row["answers"][i] for i in permutation]
    result["gold_index"] = inverse[row["gold_index"]]
    if "yes_index" in row:
        result["yes_index"] = inverse[row["yes_index"]]
    if "target_probs" in row:  # soft targets (stage 3) follow their options like the gold index
        result["target_probs"] = [row["target_probs"][i] for i in permutation]
    if "option_rewards" in row:  # per-option rewards (stage-3 RL) likewise
        result["option_rewards"] = [row["option_rewards"][i] for i in permutation]
    result["canonical_index"] = inverse
    # Keep the prepared manifest's provenance valid after a second permutation.
    if "candidate_permutation" in row:
        result["candidate_permutation"] = [row["candidate_permutation"][i] for i in permutation]
    return result


class PermutationDataset(DecisionDataset):
    """One dataset item is one original QA, not two independent examples.

    Fixed seed gives reproducible pairs across train/resume. A nonzero cyclic
    shift moves every option. With K>2 this samples two orders, not all K!.
    repeat is the matched-exposure control with two identical input orders.
    """
    def __init__(self, manifest, limit=None, pair_mode="swap", seed=20260920, pair_rounds=1):
        super().__init__(manifest)
        # "both A and B" / "all of the above" change meaning when moved.
        # Apply the same pool restriction to repeat and swap controls.
        excluded = [row for row in self.rows
                    if row.get("candidate_order_policy") == "preserve_option_references"]
        self.rows = [row for row in self.rows
                     if row.get("candidate_order_policy") != "preserve_option_references"]
        self.pool_report = {
            "input_qa": len(self.rows) + len(excluded),
            "excluded_option_references": len(excluded),
            "excluded_by_source": dict(Counter(row.get("source", "unknown") for row in excluded)),
            "eligible_qa": len(self.rows),
        }
        if limit is not None:
            self.rows = self.rows[:limit]
        if pair_mode not in ("swap", "repeat", "coverage"):
            raise ValueError("pair_mode must be swap, repeat or coverage")
        self.pair_mode, self.seed = pair_mode, seed
        self.pair_rounds = pair_rounds
        if pair_mode == "coverage":
            counts = {len(row["answers"]) for row in self.rows}
            if any((2 * pair_rounds) % count for count in counts):
                raise ValueError("Coverage needs equal full cycles: 2 * pair_rounds must be divisible by every K")

    def __len__(self):
        return len(self.rows) * self.pair_rounds

    def __getitem__(self, index):
        row_index, round_index = divmod(index, self.pair_rounds)
        row = super().__getitem__(row_index)
        count = len(row["answers"])
        identity = list(range(count))
        shift = random.Random(self.seed + row_index).randrange(1, count)
        permutation = identity[shift:] + identity[:shift]
        if self.pair_mode == "repeat" or row.get("fixed_order"):  # fixed-order rows (stage 3 codebook) are never shuffled
            permutation = identity
        if self.pair_mode == "coverage":
            shifts = ((2 * round_index) % count, (2 * round_index + 1) % count)
            orders = [identity[s:] + identity[:s] for s in shifts]
            return {"views": [permute_row(row, order) for order in orders],
                    "labels": [row["gold_index"], count]}
        return {"views": [permute_row(row, identity), permute_row(row, permutation)],
                "labels": [row["gold_index"], count]}


class PermutationCollator:
    def __init__(self, processor, max_options):
        self.base = DecisionCollator(processor)
        self.max_options = max_options

    def __call__(self, items):
        rows = [view for item in items for view in item["views"]]
        batch = self.base(rows)
        batch["view_gold"] = batch.pop("gold_index")
        batch["labels"] = torch.tensor([item["labels"] for item in items])
        batch["canonical_index"] = torch.tensor([
            row["canonical_index"] + list(range(len(row["answers"]), self.max_options))
            for row in rows
        ])
        # Target distribution per view: the row's soft targets, else one-hot on the gold option.
        target = torch.zeros(len(rows), self.max_options)
        for i, row in enumerate(rows):
            if "target_probs" in row:
                target[i, :len(row["target_probs"])] = torch.tensor(row["target_probs"], dtype=torch.float)
            else:
                target[i, row["gold_index"]] = 1.0
        batch["view_target"] = target
        batch["soft_rows"] = torch.tensor([float("target_probs" in row) for row in rows])
        reward = torch.zeros(len(rows), self.max_options)
        for i, row in enumerate(rows):
            if "option_rewards" in row:
                reward[i, :len(row["option_rewards"])] = torch.tensor(row["option_rewards"], dtype=torch.float)
        batch["view_reward"] = reward
        batch["reward_rows"] = torch.tensor([float("option_rewards" in row) for row in rows])
        return batch
