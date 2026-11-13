#!/usr/bin/env bash
# Question answering (Table 2): option-letter accuracy of one model with the non-thinking chat prefix, max_pixels 262,144 and
# batch 8 (4, then 2 on CUDA out of memory); a step with OUT/.done_<step> is skipped. Four lanes over GPUS (lanes on the same
# GPU run one after another):
#   0  DriveLM test (our scene split), every cyclic option order: superglance.eval.qa_rotations
#   1  3DSRBench, official circular groups (qa_eval + qa_summary_3dsr); CV-Bench, source-weighted, and BLINK's four tasks
#      (qa_eval + qa_summary_external)
#   2  MMLU, HellaSwag, ARC-Challenge                                     (qa_eval)
#   3  MMLU-Pro, BoolQ, WinoGrande, TruthfulQA (MC1)                      (qa_eval)
# Then superglance.eval.qa_table prints the headline numbers of REF (another run's OUT, if given) and OUT side by side.
# Table 2's columns: the base model (MODEL = Qwen3.5-9B, no ADAPTER); stage 1 (its adapter, MODEL = Qwen3.5-9B); stage 2 (its
# adapter, MODEL = the merged stage-1 model); stage 3 (the final RL adapter, MODEL = the merged stage-2 model).
#   MODEL=<the adapter's base> OUT=<output dir> ROOT=<data root of scripts/build_qa.sh> [ADAPTER=<adapter>] [GPUS="0 1 2 3"]
#   [REF=<another OUT>] bash scripts/eval_qa.sh
set -uo pipefail
cd "$(dirname "$0")/.."
: "${MODEL:?base model}" "${OUT:?output dir}" "${ROOT:?QA data root}"
AD=${ADAPTER:-}; PY=${PY:-python}; GPUS=(${GPUS:-0 1 2 3}); MP=262144
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT/logs"

step() {   # name cmd...: the command reads its batch size from BS
  local name=$1; shift
  [ -f "$OUT/.done_$name" ] && return 0
  for BS in 8 4 2; do
    "$@" > "$OUT/logs/$name.bs$BS.log" 2>&1 && { touch "$OUT/.done_$name"; return 0; }
    grep -q "OutOfMemoryError\|CUDA out of memory" "$OUT/logs/$name.bs$BS.log" || break
  done
  echo "FAILED $name: $OUT/logs/$name.bs$BS.log" | tee -a "$OUT/FAILURES"; return 1
}
ev() { "$PY" -m superglance.eval.qa_eval --model "$MODEL" --adapter "$AD" --batch-size "$BS" --max-pixels $MP "$@"; }
drivelm() { "$PY" -m superglance.eval.qa_rotations --model "$MODEL" --adapter "$AD" --batch-size "$BS" --max-pixels $MP \
              --manifest "$ROOT/stage2_drivelm_20260925/drivelm_test.jsonl" --output "$OUT/drivelm_test"; }
sr() {
  local T=$ROOT/eval_3dsr/prepared/3dsr_real_circular.jsonl
  ev --manifest "$T" --output "$OUT/3dsr" && "$PY" -m superglance.eval.qa_summary_3dsr --manifest "$T" \
    --predictions "$OUT/3dsr/predictions.jsonl" --output "$OUT/3dsr/3dsr_summary.json"
}
external() {
  local b
  for b in cvbench blink_single_image; do
    ev --manifest "$ROOT/eval_external/prepared/$b.jsonl" --output "$OUT/external/$b" || return 1
  done
  "$PY" -m superglance.eval.qa_summary_external --output "$OUT/external"
}
text() { local t; for t in "$@"; do step text_$t ev --manifest "$ROOT/text_f_20260925/retention_evals/$t.jsonl" --output "$OUT/text/$t"; done; }
lane0() { step drivelm drivelm; }
lane1() { step 3dsr sr; step external external; }
lane2() { text mmlu hellaswag arc_challenge; }
lane3() { text mmlu_pro boolq winogrande truthfulqa_mc1; }

n=${#GPUS[@]}; pids=()
for ((g = 0; g < n && g < 4; g++)); do
  (export CUDA_VISIBLE_DEVICES=${GPUS[g]}; for l in 0 1 2 3; do [ $((l % n)) = $g ] && lane$l; done) > "$OUT/logs/gpu$g.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
"$PY" -m superglance.eval.qa_table ${REF:+"$REF"} "$OUT"
[ ! -s "$OUT/FAILURES" ]
