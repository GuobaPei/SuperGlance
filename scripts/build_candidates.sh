#!/usr/bin/env bash
# The candidate library (superglance.builders.candidates) and the supplement's coverage table, in the order the paper's were made:
#   kmeans    k-means 4,096 on the navtrain fit scenes' 4-s futures + one standing-still trajectory -> $OUT/cands_kmeans4096.npy (4,097)
#   fast      + 1,024 high-speed candidates (k-means over the PAI-AV fit rows at >= 13 m/s) -> $OUT/cands_k4097_fast1024.npy (5,121)
#   kprime    the 152 rule-built candidates of K' (braking, medium-speed curves and ramps) -> $OUT/kp/cands_k5273_v2.npy (5,273)
#   coverage  the coverage table's sets on the 10,549 held-out scenes: k-means 256 and 1,024 (+ standing still) and the 4,097, each
#             scored with superglance.builders.pools.score_candidates; then the table (the K' column reads K_HELD, the 5,273
#             candidates' scores on the same scenes)
#   NAV_INPUTS=<navtrain_inputs.jsonl> PAI_FIT_ROWS=<PAI-AV fit rows> E2E_INPUTS=<e2e_inputs.npz> EVAL_KEYS=<eval_keys.json>
#   HELD=<held-out list .jsonl> EVAL_TOKENS=<the 2,000 evaluation tokens> [K_HELD=<scores.npz>] [OUT=runs/candidates] [PROCS=48]
#   bash scripts/build_candidates.sh [kmeans] [fast] [kprime] [coverage]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-runs/candidates}; PROCS=${PROCS:-48}; mkdir -p "$OUT"
c() { python -m "superglance.builders.candidates.$1" "${@:2}"; }
for step in ${@:-kmeans fast kprime coverage}; do
  case $step in
    kmeans) c kmeans "${NAV_INPUTS:?}" 4096 "$OUT/cands_kmeans4096.npy" ;;
    fast) c fast "${PAI_FIT_ROWS:?}" "$OUT/cands_kmeans4096.npy" 1024 "$OUT/cands_k4097_fast1024.npy" ;;
    kprime) c train_futures "$NAV_INPUTS" "${E2E_INPUTS:?}" "$PAI_FIT_ROWS" "${EVAL_KEYS:?}" "$OUT/cands_k4097_fast1024.npy" "$OUT/kp"
            c extend "$OUT/cands_k4097_fast1024.npy" "$OUT/kp/train_futures.npz" "$OUT/kp/train_nn5121.npz" "$OUT/kp" ;;
    coverage) V=$OUT/coverage
              c coverage prep "$NAV_INPUTS" "${HELD:?held-out list}" "${EVAL_TOKENS:?evaluation tokens}" "$V"
              c kmeans "$NAV_INPUTS" 1024 "$V/kmeans1024.npy"
              python - "$OUT/cands_k4097_fast1024.npy" "$V/k4097.npy" <<'PY'
import sys, numpy as np
np.save(sys.argv[2], np.load(sys.argv[1])[:4097])   # rows 0-4,096 of the library: k-means 4,096 + standing still
PY
              for k in 257:kmeans256 1025:kmeans1024 4097:k4097; do
                python -m superglance.builders.pools.score_candidates "$V/${k#*:}.npy" "$V/held_inputs.jsonl" "$V/k${k%%:*}" "$PROCS"
              done
              c coverage table "$V" "$EVAL_TOKENS" "$V/k1025/scores.npz" "$V/k4097/scores.npz" "${K_HELD:?scores of the 5,273 candidates on the held-out scenes}" \
                "$OUT/cands_k4097_fast1024.npy" ;;
    *) echo "unknown step $step (kmeans, fast, kprime, coverage)" >&2; exit 2 ;;
  esac
done
