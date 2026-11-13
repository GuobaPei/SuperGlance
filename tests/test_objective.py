"""CPU self-check of the training objectives on toy batches (about 10 s; needs torch and Pillow, which qa_pairs imports):

1. QA rows, stages 1-2: permutation_loss called as superglance/train/qa.py calls it, on two option orders built by
   permute_row (identity and a cyclic shift s), for 2 to 9 options and every shift. The paired-draw estimate of the
   gradient must match the exact gradient of the expected reward R = 2 p_y p'_y' - sum p^2 sum p'^2 (relative error
   below 5 %), with random logits and with a model that favours the gold answer. For the latter it must also be far
   from the gradient of the same reward with the shifted view's gold at y + s instead of y - s, which is what the
   estimator rewarded before the fix whenever 2s != 0 (mod k).
2. The soft-target branch of joint_brier_pg (a target over answer contents) against its exact gradient.
3. The KL term and the stage-3 RL loss: exact_rl_loss, reached through permutation_loss as the RL config reaches it,
   has the gradient of -(sum_k p_k r_k - beta KL(p || p_ref)), averaged over rows.

  python tests/test_objective.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from superglance.data.qa_pairs import permute_row  # noqa: E402
from superglance.objectives.permutation import candidate_kl, exact_rl_loss, joint_brier_pg, permutation_loss  # noqa: E402

DRAWS, MAX_K = 400_000, 9


def toy_batch(seed, confident=False):
    """Two views per question (identity, cyclic shift s) for k = 2..9 and every s, padded like PermutationCollator;
    confident: the logits favour the gold answer in both orders."""
    gen = torch.Generator().manual_seed(seed)
    cases, rows = [], []
    for k in range(2, MAX_K + 1):
        for s in range(1, k):
            g = int(torch.randint(k, (1,), generator=gen))
            row = {"answers": [f"option {c}" for c in range(k)], "gold_index": g}
            ident = list(range(k))
            views = permute_row(row, ident), permute_row(row, ident[s:] + ident[:s])
            assert views[1]["answers"][views[1]["gold_index"]] == row["answers"][g]
            cases.append((k, s, g))
            rows += views
    n = len(rows)
    logits = 1.5 * torch.randn(n, MAX_K, generator=gen)
    counts = torch.tensor([len(r["answers"]) for r in rows])
    gold = torch.tensor([r["gold_index"] for r in rows])
    canon = torch.tensor([r["canonical_index"] + list(range(len(r["answers"]), MAX_K)) for r in rows])
    onehot = torch.zeros(n, MAX_K)
    onehot[torch.arange(n), gold] = 1.0
    if confident:
        logits = logits / 1.5 + 2.5 * onehot
    return cases, logits, counts, gold, canon, onehot


def exact_grad(logits, counts, reward):
    """Gradient of -sum_pairs R / rows, the expectation of joint_brier_pg's loss; reward(i, p0, p1) -> R of pair i."""
    z = logits.clone().requires_grad_(True)
    total = 0
    for i in range(0, len(z), 2):
        k = int(counts[i])
        total = total + reward(i, z[i, :k].softmax(-1), z[i + 1, :k].softmax(-1))
    (-total / len(z)).backward()
    return z.grad


def estimated_grad(fn, logits):
    z = logits.clone().requires_grad_(True)
    fn(z).backward()
    return z.grad


def rel(a, b):
    return float((a - b).norm() / b.norm())


def check_qa_pairs(confident):
    cases, logits, counts, gold, canon, onehot = toy_batch(0, confident)
    zeros = torch.zeros(len(logits))
    torch.manual_seed(1)
    est = estimated_grad(lambda z: permutation_loss(   # the keyword arguments of train/qa.py's compute_loss
        z, gold, counts, canon, consistency_weight=0.0, samples=DRAWS, pair_reward="joint", view_target=onehot,
        soft_rows=zeros, soft_objective="brier", view_reward=torch.zeros_like(onehot), reward_rows=zeros)[0], logits)
    right = exact_grad(logits, counts, lambda i, p0, p1: 2 * p0[gold[i]] * p1[gold[i + 1]] - (p0 ** 2).sum() * (p1 ** 2).sum())
    # the shifted view's gold read at position y + s (content y + 2s) instead of y - s: the target before the fix
    wrong_pos = {2 * j: (g + s) % k for j, (k, s, g) in enumerate(cases)}
    wrong = exact_grad(logits, counts, lambda i, p0, p1: 2 * p0[gold[i]] * p1[wrong_pos[i]] - (p0 ** 2).sum() * (p1 ** 2).sum())
    worst, closest_wrong = 0.0, float("inf")
    for j, (k, s, g) in enumerate(cases):
        e, a, w = (x[2 * j:2 * j + 2, :k] for x in (est, right, wrong))
        worst = max(worst, rel(e, a))
        if (2 * s) % k:   # the cases the fix changed
            closest_wrong = min(closest_wrong, rel(e, w))
        assert rel(e, a) < 0.05, f"k={k} s={s} gold={g}: relative gradient error {rel(e, a):.3f}"
    line = f"QA pairs, {'gold-favouring' if confident else 'random'} logits, k 2-9, every shift ({len(cases)} questions): " \
           f"worst error {worst:.3f}"
    if confident:
        assert closest_wrong > 0.3, closest_wrong
        n_fixed = sum(1 for k, s, _ in cases if (2 * s) % k)
        line += f"; the {n_fixed} with 2s != 0 (mod k) differ from the pre-fix target's gradient by >= {closest_wrong:.2f}"
    print(line)


def check_soft_target():
    cases, logits, counts, gold, canon, _ = toy_batch(2)
    torch.manual_seed(3)
    q_content, target = {}, torch.zeros(len(logits), MAX_K)
    for j, (k, s, g) in enumerate(cases):
        q = torch.distributions.Dirichlet(torch.ones(k)).sample(); q_content[2 * j] = q
        for v in (0, 1):   # PermutationCollator's per-position target: position canonical_index[c] holds content c
            target[2 * j + v, canon[2 * j + v, :k]] = q
    torch.manual_seed(4)
    est = estimated_grad(lambda z: joint_brier_pg(z, gold, counts, samples=DRAWS, view_target=target, canonical_index=canon), logits)

    def reward(i, p0, p1):
        k = len(p0)
        by_content = [p0[canon[i, :k]], p1[canon[i + 1, :k]]]
        return 2 * (q_content[i] * by_content[0] * by_content[1]).sum() - (p0 ** 2).sum() * (p1 ** 2).sum()
    right = exact_grad(logits, counts, reward)
    worst = max(rel(est[2 * j:2 * j + 2, :k], right[2 * j:2 * j + 2, :k]) for j, (k, _, _) in enumerate(cases))
    assert worst < 0.05, worst
    print(f"soft targets over contents, k 2-9, every shift: worst error {worst:.3f}")


def check_kl_and_rl():
    gen = torch.Generator().manual_seed(5)
    n, k, beta = 6, 7, 0.3
    logits, ref = torch.randn(n, k, generator=gen), torch.randn(n, k, generator=gen)
    rewards, counts = torch.rand(n, k, generator=gen) - 0.5, torch.full((n,), k)
    kl = candidate_kl(logits, ref, counts)
    p, logp, logq = logits.softmax(-1), logits.log_softmax(-1), ref.log_softmax(-1)
    assert torch.allclose(kl, (p * (logp - logq)).sum(-1).mean())
    ones, canon = torch.ones(n), torch.arange(k).repeat(n, 1)
    torch.manual_seed(6)
    via_route = estimated_grad(lambda z: permutation_loss(   # the stage-3 RL config's route: every row a reward row
        z, torch.zeros(n, dtype=torch.long), counts, canon, consistency_weight=0.0, samples=32, pair_reward="independent",
        view_target=torch.full((n, k), 1.0 / k), soft_rows=ones, soft_objective="exact_rl", view_reward=rewards, reward_rows=ones,
        view_ref=ref, rl_kl=beta)[0], logits)
    direct = estimated_grad(lambda z: exact_rl_loss(z, rewards, ref, counts, 32, beta), logits)
    z = logits.clone().requires_grad_(True)
    pz = z.softmax(-1)
    objective = (pz * rewards).sum(-1) - beta * (pz * (z.log_softmax(-1) - logq)).sum(-1)
    (-objective.mean()).backward()
    assert torch.allclose(via_route, direct, atol=1e-7) and torch.allclose(direct, z.grad, atol=1e-6)
    adv = rewards - (p * rewards).sum(-1, keepdim=True)   # the gradient's policy term per logit: p_k (r_k - E_p r)
    kl_grad = p * ((logp - logq) - (p * (logp - logq)).sum(-1, keepdim=True))
    assert torch.allclose(direct, -(p * adv - beta * kl_grad) / n, atol=1e-6)
    print("KL term and exact RL loss: gradient = -(p (r - E_p r) - beta grad KL) / rows")


if __name__ == "__main__":
    check_qa_pairs(confident=False)
    check_qa_pairs(confident=True)
    check_soft_target()
    check_kl_and_rl()
    print("OBJECTIVE_OK")
