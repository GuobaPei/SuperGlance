"""Stage objectives on the candidate logits of each row. The two views (option orders) of a QA stay adjacent in
every local batch.

QA rows (stages 1-3): joint_brier_pg, the paired-draw Brier policy gradient on the joint answer of a question's two
option orders, plus content_js (answer-content Jensen-Shannon; weight 0 in every stage). Policy rows (stage 3):
soft_log_score toward the supervised target (soft_objective "log"), or exact_rl_loss, the exact expected reward with a
KL term to the supervised policy (soft_objective "exact_rl", RL; "grpo" is its earlier name).
"""
import math

import torch


def aligned_probabilities(logits, canonical_index, counts):
    aligned = logits.float().gather(1, canonical_index)
    valid = torch.arange(aligned.shape[-1], device=aligned.device)[None] < counts[:, None]
    return aligned.masked_fill(~valid, -torch.inf).softmax(-1).reshape(-1, 2, aligned.shape[-1])


def content_js(logits, canonical_index, counts):
    aligned = logits.float().gather(1, canonical_index).reshape(-1, 2, logits.shape[-1])
    terms = []
    for pair, count in zip(aligned, counts[::2]):
        logp = pair[:, :int(count)].log_softmax(-1)
        logm = torch.logaddexp(logp[0], logp[1]) - math.log(2)
        terms.append((logp.exp() * (logp - logm)).sum(-1).mean())
    return torch.stack(terms)


def candidate_kl(logits, base_logits, counts):
    """KL(p_model || p_base) over each row's valid candidates, averaged over rows.

    Slice rather than mask with -inf: masked -inf terms give 0 * NaN gradients.
    """
    terms = []
    for z, b, k in zip(logits, base_logits, counts):
        logp = z[:int(k)].float().log_softmax(-1)
        logq = b[:int(k)].float().log_softmax(-1)
        terms.append((logp.exp() * (logp - logq)).sum())
    return torch.stack(terms).mean()


def joint_brier_pg(logits, view_gold, counts, samples=32, view_target=None, canonical_index=None):
    """Paired-draw Brier PG on the joint answer of each QA's two adjacent views.

    A draw scores as correct only when both views pick their gold option, so a QA
    answered right in one order and wrong in the other counts as wrong. Expected
    gradient is that of the joint Brier score (sum P^2 - 2 P(gold, gold) + 1),
    averaged per row like candidate_loss.
    """
    if samples < 2:
        raise ValueError("paired reward needs at least two independent draws")
    collision_weight = 2 / (samples * (samples - 1))
    losses = []
    for i in range(0, len(logits), 2):
        k = int(counts[i])
        logp = [logits[i + v][:k].float().log_softmax(-1) for v in (0, 1)]
        p = [x.detach().exp() for x in logp]
        actions = [torch.multinomial(q, samples, replacement=True) for q in p]
        gold = [int(view_gold[i]), int(view_gold[i + 1])]
        joint = actions[0] * k + actions[1]
        collisions = torch.bincount(joint, minlength=k * k).to(p[0].dtype)[joint] - 1
        p_joint = p[0][actions[0]] * p[1][actions[1]]
        if view_target is None:
            hit = ((actions[0] == gold[0]) & (actions[1] == gold[1])).to(p[0].dtype)
            expected = p[0][gold[0]] * p[1][gold[1]]
        else:
            # Joint target: q(c) on pairs that pick the same content c in both orders, 0 elsewhere.
            # canonical_index is content -> position (permute_row's inverse), so a sampled position's content is
            # canon.argsort()[a], and per-content values are gathers by canon.
            canon = [canonical_index[i + v][:k] for v in (0, 1)]
            content = [c.argsort() for c in canon]
            q = view_target[i][:k].to(p[0].dtype)[canon[0]]
            c0, c1 = content[0][actions[0]], content[1][actions[1]]
            hit = q[c0] * (c0 == c1).to(p[0].dtype)
            by_content = [p[v][canon[v]] for v in (0, 1)]
            expected = (by_content[0] * by_content[1] * q).sum()
        local = 2 / samples * hit - collision_weight * collisions
        control = 2 / samples * expected - collision_weight * (p_joint.sum() - p_joint)
        losses.append(-((local - control).detach() * (logp[0][actions[0]] + logp[1][actions[1]])).sum())
    return torch.stack(losses).sum() / len(logits)


def soft_log_score(logits, targets, counts):
    """Cross-entropy against a target distribution (the log score, a proper scoring rule), mean over rows.
    Used for the policy rows' soft targets over K candidates, where the Brier gradient carries a factor p_k ~ 1/K and
    barely trains."""
    terms = [-(targets[i][:int(k)] * logits[i][:int(k)].float().log_softmax(-1)).sum() for i, k in enumerate(counts)]
    return torch.stack(terms).mean()


def exact_rl_loss(logits, rewards, ref, counts, samples, kl):
    """Stage-3 RL (soft_objective "exact_rl"; "grpo" is its earlier name): per row, -sum_k p_k (r_k - sum_j p_j r_j) log p_k
    with p held fixed, the exact expectation over all k candidates (no draws; the advantage is not normalized per row), plus
    kl * KL(p || softmax(ref)), ref = the supervised policy's head logits. `samples` is not used."""
    terms = []
    for i, k in enumerate(counts):
        logp = logits[i][:int(k)].float().log_softmax(-1)
        p = logp.detach().exp(); r = rewards[i][:int(k)].float()   # exact over all k candidates, no draws
        mu = (p * r).sum(); adv = r - mu   # no per-row std
        logq = ref[i][:int(k)].float().log_softmax(-1)
        terms.append(-(p * adv * logp).sum() + kl * (logp.exp() * (logp - logq)).sum())
    return torch.stack(terms).mean()


def permutation_loss(logits, view_gold, counts, canonical_index,
                     consistency_weight=0.1, samples=32, pair_reward="independent", view_target=None, soft_rows=None,
                     soft_objective="brier", view_reward=None, reward_rows=None, view_ref=None, rl_kl=0.0):
    if soft_objective in ("log", "exact_rl", "grpo") and soft_rows is not None and bool(soft_rows.any()):
        soft = soft_rows.bool()
        pairs_soft = soft[::2]  # both views of a QA share the flag
        rew = (reward_rows.bool() if (soft_objective in ("exact_rl", "grpo") and reward_rows is not None) else torch.zeros_like(soft)) & soft
        parts, weights = [], []
        if bool((soft & ~rew).any()):
            parts.append(soft_log_score(logits[soft & ~rew], view_target[soft & ~rew], counts[soft & ~rew]))
            weights.append(int((soft & ~rew)[::2].sum()))
        if bool(rew.any()):
            parts.append(exact_rl_loss(logits[rew], view_reward[rew], view_ref[rew], counts[rew], samples, rl_kl))
            weights.append(int(rew[::2].sum()))
        if not bool(pairs_soft.all()):
            hard = ~soft
            sub = lambda t: t[hard]
            parts.append(permutation_loss(sub(logits), sub(view_gold), sub(counts), sub(canonical_index), 0.0, samples,
                                          pair_reward, sub(view_target))[0])
            weights.append(int((~pairs_soft).sum()))
        rlcd = sum(w * x for w, x in zip(weights, parts)) / sum(weights)
    elif pair_reward == "joint":
        # Hard-gold rows take the gold path (their view_target is only the one-hot); soft rows the content-aligned branch.
        hard = soft_rows is None or not bool(soft_rows.bool().any())
        rlcd = joint_brier_pg(logits, view_gold, counts, samples=samples, view_target=None if hard else view_target,
                              canonical_index=canonical_index)
    else:
        raise ValueError(f"QA rows need pair_reward 'joint', got {pair_reward}")
    js = content_js(logits, canonical_index, counts).mean()
    return rlcd + consistency_weight * js, {"objective": rlcd.detach(), "content_js": js.detach()}
