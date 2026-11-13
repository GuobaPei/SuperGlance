# The AlpaSim table's columns from a cleaned summary (python -m superglance.alpasim.clean): collision all / at fault, off-road and
# progress (AlpaSim's metrics_results, in %), km per incident all / at fault, the mean score (the rollouts' "score", unscored
# rollouts as 0, and metrics_results' own) and the pass rate. usage: python -m superglance.alpasim.table <summary.json> [...]
import json, sys

for p in sys.argv[1:]:
    d = json.load(open(p)); m = d["metrics_results"][0]; R = d["rollouts"]
    ms = sum(x.get("score") or 0 for x in R) / len(R)
    print(f"{p.split('/aggregate/')[0].rsplit('/', 1)[-1]:40s} n {len(R)} | collision {100 * m['collision_any']:.1f} / {100 * m['collision_at_fault']:.1f} | "
          f"off-road {100 * m['offroad']:.1f} | progress {100 * m['progress']:.1f} | km {m['avg_dist_between_incidents']:.3f} / "
          f"{m['avg_dist_between_incidents_at_fault']:.3f} | mean score {ms:.4f} (metrics_results {m.get('score', float('nan')):.4f}) | "
          f"pass {100 * sum(bool(x['passed']) for x in R) / len(R):.1f}")
