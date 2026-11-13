#!/usr/bin/env bash
# Language2Drive, NAVSIM track (Table 7): scores prediction packages with the benchmark's current scorer
# (scorer/score_corrected.py with its sidecars), the equal-weight mean over the four wordings: SemSISR / SemCFPS / SCS / PCS.
# A package holds one model's two runs on the 38 PCUs x 4 wordings, as the Language2Drive SDK writes them:
#   instructed/navsim/{canonical,rewrites}.json              the instruction in the prompt     -> Table 7's model rows
#   control_no_instruction/navsim/{canonical,rewrites}.json  the same scenes, no instruction  -> "no instruction"
#   argmax_from_same_run/{instructed,control_no_instruction}/{canonical,rewrites}.json   the head's argmax of the same runs
# Table 7: "SuperGlance" and "SuperGlance, no instruction" are the final model's mean and mean_noinstr; "no instruction rows"
# is the mean of the supervised model trained without instruction rows. scripts/l2d_run.sh runs the SDK on a model and
# writes its package.
#   L2D_SCORER=<the benchmark's scoring folder: scorer/score_corrected.py, its sidecars in private/, its packages in site/>
#   [OUT=runs/l2d] bash scripts/eval_l2d.sh <name>=<package dir> [...]
set -euo pipefail
: "${L2D_SCORER:?Language2Drive scoring folder}"
[ $# -gt 0 ] || { echo "usage: L2D_SCORER=<dir> bash scripts/eval_l2d.sh <name>=<package dir> ..." >&2; exit 2; }
OUT=${OUT:-runs/l2d}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
score() {   # name, dir with canonical.json + rewrites.json
  [ -s "$2/canonical.json" ] || return 0
  (cd "$L2D_SCORER" && PYTHONPATH=site PYTHONDONTWRITEBYTECODE=1 python3 scorer/score_corrected.py --sidecars private \
    --predictions "$2/canonical.json" --rewrite-predictions "$2/rewrites.json" --output "$OUT/$1.json") > "$OUT/$1.log" 2>&1 \
    || { echo "$1: the scorer failed, see $OUT/$1.log" >&2; return 1; }
  python3 -c 'import json, sys; x = json.load(open(sys.argv[2]))["equal_weight_rewrites"]
print(f"{sys.argv[1]:28s}", " / ".join(f"{x[k]:6.3f}" for k in ("sem_sisr", "sem_cfps", "scs", "pcs")))' "$1" "$OUT/$1.json"
}
echo "$(printf '%-28s' '') SemSISR / SemCFPS / SCS / PCS"
for a in "$@"; do
  n=${a%%=*}; d=$(cd "${a#*=}" && pwd)
  score "${n}_mean" "$d/instructed/navsim"
  score "${n}_mean_noinstr" "$d/control_no_instruction/navsim"
  score "${n}_argmax" "$d/argmax_from_same_run/instructed"
  score "${n}_argmax_noinstr" "$d/argmax_from_same_run/control_no_instruction"
done
