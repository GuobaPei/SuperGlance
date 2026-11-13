#!/usr/bin/env bash
# Language2Drive, NAVSIM track: one adapter's two SDK runs on all 38 PCUs x 4 wordings (304 requests each) and their package,
# which scripts/eval_l2d.sh scores (Table 7). The steps of the paper's runs:
#   1  configs (superglance.l2d.configs): the instructed run (append), the control (none, no instruction in the prompt)
#   2  the SDK's adapter-check on the append config
#   3  both runs on one GPU at once: the SDK's model service (l2d_benchmark.cli serve) with superglance/l2d/adapter.py, then the
#      SDK's own runner (language2drive evaluate --tracks navsim) into <mode>/official
#   4  output checks: 304 scheduled, 0 failed, canonical.json = rewrites.json's canonical, 38 PCUs per wording
#   5  the argmax readout of both runs from the adapter traces (superglance.l2d.argmax)
#   6  the package (superglance.l2d.package): RUN/package/l2d_navsim_<tag>_<yyyymmdd>/ and its .zip
#   BASE_MODEL=<stage-2 merged model> ADAPTER=<stage-3 adapter> CANDS=<its candidate set> TAG=<name>
#   L2D_EVAL=<the SDK workspace's .l2d/evaluation.json> L2D_CLI=<the SDK environment's language2drive executable>
#   SDK_PKG=<dir with the SDK's l2d_benchmark package, added to PYTHONPATH of the model service>
#   [PY=<python with torch, transformers, peft>] [GPU=0] [PORTS="18294 18295"] [RUN=runs/l2d_runs/<tag>] bash scripts/l2d_run.sh
# The model service runs in the training environment (PY); the SDK's runner in its own environment, with
# OPENBLAS_CORETYPE=Haswell as the paper's runs (README, Installation).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BASE_MODEL:?stage-2 merged model}" "${ADAPTER:?adapter}" "${CANDS:?candidate set}" "${TAG:?run name}"
: "${L2D_EVAL:?SDK evaluation.json}" "${L2D_CLI:?language2drive executable}" "${SDK_PKG:?SDK package dir}"
PY=${PY:-python}; GPU=${GPU:-0}; P=(${PORTS:-18294 18295}); ROOT=$(pwd)
R=${RUN:-runs/l2d_runs/$TAG}; [ ! -e "$R" ] || { echo "$R exists" >&2; exit 1; }
mkdir -p "$R/checks"; R=$(cd "$R" && pwd)
export BASE_MODEL L2D_EVAL
"$PY" -m superglance.l2d.configs "$R" "$ADAPTER" "$CANDS" "$TAG" "${P[0]}" "${P[1]}" | tee "$R/checks/configs.json"
( cd /tmp && PYTHONPATH=$ROOT "$L2D_CLI" adapter-check --adapter "$ROOT/superglance/l2d/adapter.py" \
    --config "$R/config/model_append.json" ) > "$R/checks/adapter_check.json"
serve() {   # mode port: the model service, then the SDK's runner against it
  local d=$R/$1; mkdir -p "$d"
  CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONPATH=$ROOT:$SDK_PKG \
    "$PY" -m l2d_benchmark.cli serve --adapter "$ROOT/superglance/l2d/adapter.py" --config "$R/config/model_$1.json" \
    --port "$2" > "$d/serve.log" 2>&1 < /dev/null &
  local s=$!
  for _ in $(seq 1 300); do curl -s -m 2 "http://127.0.0.1:$2/health" > "$d/health.json" 2> /dev/null && break; sleep 2; done
  local rc=0
  ( cd /tmp && OPENBLAS_CORETYPE=Haswell "$L2D_CLI" evaluate --config "$R/config/evaluation_$1.json" --tracks navsim \
      --output "$d/official" ) > "$d/evaluate.out" 2>&1 || rc=$?
  kill "$s" 2> /dev/null || true; return $rc
}
serve append "${P[0]}" & a=$!; serve none "${P[1]}" & n=$!
wait $a || { echo "the instructed run failed: $R/append" >&2; exit 1; }
wait $n || { echo "the control run failed: $R/none" >&2; exit 1; }
for m in append none; do
  "$PY" - "$R/$m/official/navsim" <<'PY'
import json, sys
d = sys.argv[1]; i = json.load(open(f"{d}/inference.json")); c = json.load(open(f"{d}/canonical.json")); w = json.load(open(f"{d}/rewrites.json"))
ok = (i["scheduled_requests"] == 304 and i["failed_requests"] == 0 and c == w["canonical"]
      and sorted(w) == ["canonical", "paraphrase_1", "paraphrase_2", "paraphrase_3"] and all(len(v) == 38 for v in w.values()))
print(d, "scheduled", i["scheduled_requests"], "failed", i["failed_requests"], "OK" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY
  "$PY" -m superglance.l2d.argmax "$R" "$m" | tee "$R/checks/argmax_$m.json"
done
"$PY" -m superglance.l2d.package "$R" "$TAG"
