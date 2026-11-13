"""Paired comparison of AlpaSim runs against the first: on the scenes every run has, per run the pass rate, mean scene score and
at-fault km per incident; per later run the mean score difference with a bootstrap 95 % interval over scenes (2,000 draws, seed
20261012), the scenes passed on one side only with an exact McNemar p, and km per at-fault incident on those common scenes.
km per incident = total dist_traveled_m / (at-fault collisions + offroads), offroad_or_collision_at_fault (checked against the aggregate's avg_dist_between_incidents_at_fault).
usage: python -m superglance.alpasim.pair <results-summary.json | run dir> <...> [...]"""
import collections, json, math, os, random, sys


def load(p):
    p = os.path.join(p, "aggregate", "results-summary.json") if os.path.isdir(p) else p
    d = json.load(open(p)); by = collections.defaultdict(list)
    for r in d["rollouts"]:
        by[r["clipgt_id"]].append(r)
    return d, by


def km(by, keys):
    rs = [r["metrics"] for k in keys for r in by[k] if "offroad_or_collision_at_fault" in r["metrics"]]   # failed rollouts carry no metrics
    n = sum(m["offroad_or_collision_at_fault"] for m in rs)
    return sum(m["dist_traveled_m"] for m in rs) / 1000 / n if n else float("inf"), int(n)


def mcnemar(b, c):   # exact two-sided
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def boot(d, n=2000, seed=20261012):   # percentile bootstrap of the mean
    rng = random.Random(seed)
    m = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n) - 1]


runs = sys.argv[1:]; L = [load(p) for p in runs]
for p, (d, by) in zip(runs, L):   # the km definition against the aggregate, all scenes
    print(f"{p}: km {km(by, list(by))[0]:.3f} (aggregate {d['metrics_results'][0]['avg_dist_between_incidents_at_fault']:.3f})")
common = sorted(set.intersection(*(set(by) for _, by in L)))
S = [{k: (sum(r["score"] for r in by[k]) / len(by[k]), sum(bool(r["passed"]) for r in by[k]) / len(by[k])) for k in common} for _, by in L]
print(f"common scenes {len(common)}")
for p, s, (_, by) in zip(runs, S, L):
    k_, n_ = km(by, common)
    print(f"{p}: passed {sum(v[1] for v in s.values()) / len(common):.1%}, score {sum(v[0] for v in s.values()) / len(common):.4f}, km {k_:.3f} ({n_} incidents)")
for p, s in zip(runs[1:], S[1:]):
    dd = [s[k][0] - S[0][k][0] for k in common]; lo, hi = boot(dd)
    b = sum(s[k][1] > S[0][k][1] for k in common); c = sum(s[k][1] < S[0][k][1] for k in common)
    print(f"{p} - {runs[0]}: score {sum(dd) / len(dd):+.4f} [{lo:+.4f}, {hi:+.4f}]; passed only here {b}, only in the first {c}, McNemar p {mcnemar(b, c):.2g}")
