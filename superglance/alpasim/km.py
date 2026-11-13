"""km per at-fault incident of one AlpaSim run with a bootstrap 95 % interval over scenes, its pass rate and left_corridor_laterally
share; against each reference run, on their common scenes, the km difference and ratio with paired intervals.
km = total dist_traveled_m / (at-fault collisions + offroads) over the rollouts with metrics (as superglance/alpasim/pair.py);
2,000 draws of the scenes with replacement, seed 20261012.
usage: python -m superglance.alpasim.km <run summary.json> [<reference summary.json> ...]"""
import collections, json, random, sys


def load(p):   # scene -> (km driven, at-fault incidents), and the rollouts
    by = collections.defaultdict(list); R = json.load(open(p))["rollouts"]
    for r in R:
        by[r["clipgt_id"]].append(r["metrics"])
    return {k: (sum(m["dist_traveled_m"] for m in v if "offroad_or_collision_at_fault" in m) / 1000,
                sum(m["offroad_or_collision_at_fault"] for m in v if "offroad_or_collision_at_fault" in m)) for k, v in by.items()}, R


def km(S, ks):
    i = sum(S[k][1] for k in ks)
    return sum(S[k][0] for k in ks) / i if i else float("inf")


def ci(f, ks, n=2000, seed=20261012):
    rng = random.Random(seed)
    m = sorted(f([rng.choice(ks) for _ in ks]) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n) - 1]


A, R = load(sys.argv[1]); ks = sorted(A); lo, hi = ci(lambda s: km(A, s), ks)
lc = sum(r.get("failure_reason") == "left_corridor_laterally" for r in R); nf = sum(not r["passed"] for r in R)
print(f"{sys.argv[1]}: n {len(R)}, pass {100 * (len(R) - nf) / len(R):.1f} %, km {km(A, ks):.3f} [{lo:.3f}, {hi:.3f}] "
      f"({int(sum(A[k][1] for k in ks))} at-fault incidents), left_corridor {100 * lc / len(R):.1f} % of rollouts ({lc} of {nf} failures)")
for p in sys.argv[2:]:
    B, _ = load(p); c = sorted(set(A) & set(B)); a, b = km(A, c), km(B, c)
    d = ci(lambda s: km(A, s) - km(B, s), c); q = ci(lambda s: km(A, s) / km(B, s), c)
    print(f"  vs {p}: {len(c)} common scenes, km {a:.3f} vs {b:.3f}: difference {a - b:+.3f} [{d[0]:+.3f}, {d[1]:+.3f}], "
          f"ratio {a / b:.2f} [{q[0]:.2f}, {q[1]:.2f}]")
