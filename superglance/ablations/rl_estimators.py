"""The RL step's estimators of the supplement's RL step table beside the paper's exact_rl (superglance.objectives.permutation.exact_rl_loss).
  sampled_normalized  `samples` draws a ~ p per row (with replacement, detached), r = rewards[a], advantage A = (r - mean r) / (std r +
                      1e-6) within the draws, loss = -mean(A log p[a]) + kl * KL(p || softmax(ref)); with the two views of each row and
                      reward_samples 32, the table's "2 x 32"
  exact_normalized    the exact expectation over all of a row's candidates under the detached p: A = (r - E_p r) / (sd_p r + 1e-6),
                      loss = -sum_k p_k A_k log p_k + kl * KL(p || p_SFT); the sampled estimator's infinite-draw limit
  exact               exact_rl as released: A = r - E_p r, no per-row std
The chosen estimator replaces exact_rl_loss (permutation_loss calls it by name for soft_objective "exact_rl"), then the stage-3 trainer
runs with its own arguments; config, rewards, KL reference, mixture and steps are the RL run's.
usage: torchrun ... -m superglance.ablations.rl_estimators <sampled_normalized|exact_normalized|exact> <superglance.train.policy arguments>
       python -m superglance.ablations.rl_estimators --selftest"""
import sys

import torch

from superglance.objectives import permutation


def sampled_normalized_loss(logits, rewards, ref, counts, samples, kl):
    """Group-relative policy gradient on `samples` draws per row plus kl * exact KL(p || softmax(ref)) (GRPO's normalization)."""
    terms = []
    for i, k in enumerate(counts):
        logp = logits[i][:int(k)].float().log_softmax(-1)
        a = torch.multinomial(logp.detach().exp(), samples, replacement=True)
        r = rewards[i][:int(k)].float()[a]
        adv = (r - r.mean()) / (r.std() + 1e-6)
        logq = ref[i][:int(k)].float().log_softmax(-1)
        terms.append(-(adv * logp[a]).mean() + kl * (logp.exp() * (logp - logq)).sum())
    return torch.stack(terms).mean()


def exact_normalized_loss(logits, rewards, ref, counts, samples, kl):
    """The sampled estimator's exact expectation over all k candidates under the detached p, normalized per row."""
    terms = []
    for i, k in enumerate(counts):
        logp = logits[i][:int(k)].float().log_softmax(-1)
        p = logp.detach().exp(); r = rewards[i][:int(k)].float()   # exact over all k candidates, no draws
        mu = (p * r).sum(); adv = (r - mu) / ((p * (r - mu) ** 2).sum().sqrt() + 1e-6)
        logq = ref[i][:int(k)].float().log_softmax(-1)
        terms.append(-(p * adv * logp).sum() + kl * (logp.exp() * (logp - logq)).sum())
    return torch.stack(terms).mean()


ESTIMATORS = {"exact": None, "exact_normalized": exact_normalized_loss, "sampled_normalized": sampled_normalized_loss}


def selftest():
    g = torch.Generator().manual_seed(0)
    logits, rewards, ref = (torch.randn(1, 7, generator=g) for _ in range(3))
    counts = torch.tensor([7])
    p = logits[0].softmax(-1); sd = (p * (rewards[0] - (p * rewards[0]).sum()) ** 2).sum().sqrt()
    raw = permutation.exact_rl_loss(logits, rewards, ref, counts, 32, 0.0)
    assert torch.allclose(exact_normalized_loss(logits, rewards, ref, counts, 32, 0.0), raw / (sd + 1e-6), atol=1e-6)
    assert torch.isfinite(sampled_normalized_loss(logits, rewards, ref, counts, 32, 0.3))
    print("rl_estimators selftest ok")


def main():
    if sys.argv[1:2] == ["--selftest"]:
        return selftest()
    name = sys.argv.pop(1)
    assert name in ESTIMATORS, (name, sorted(ESTIMATORS))
    if ESTIMATORS[name] is not None:
        permutation.exact_rl_loss = ESTIMATORS[name]
    from superglance.train import policy   # after the patch; the selftest needs no trainer stack
    policy.main()


if __name__ == "__main__":
    main()
