#!/usr/bin/env bash
# PAI-AV open loop (Table 6): ADE at 3 s / 5 s on our version of Qwen-Drive-1.0's 700-frame leakage-free subset, the 140 test
# clips of the public gold eval list with 5 frames each at 5.1 s + (-2 ... 2) s into the clip. Each step is skipped when its
# output exists:
#   rows      superglance.eval.pai_rows: the front wide frame re-projected to CAM_F0, ego history / velocity / command, the
#             logged future at 10 Hz
#   past      superglance.eval.pai_history: each row's frames 1.5 / 1.0 / 0.5 s back, made as the training rows' (HIST=1)
#   features  superglance.eval.features with that index; K = the candidate set's size
#   ade       superglance.eval.pai_ade: the head's argmax (top1) and its six most probable candidates (avg6 / min6) beside the
#             oracle candidate and constant velocity, over all frames and over the T0 frames
# Table 6's best-of-six cells are min6 over all frames. Its one-plan cells read the top-64 mean (tau 0.2) densified by a cubic
# spline from the logged velocity, a step whose script is not part of this release; top1 here is the argmax, densified linearly.
#   PAI_ROOT=<PhysicalAI-AV root: clip_index.parquet, camera/, labels/, calibration/>
#   GOLD=<1005_7cam_gold_eval_metadb_public.parquet, or - for every clip of SPLIT> BASE=<stage-2 merged base> ADAPTER=<adapter>
#   CANDS=<candidate set .npy> [SPLIT=test] [OUT=runs/pai] [NAME=superglance] [HIST=1] [GPU=0] [PROCS=16] bash scripts/eval_pai.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PAI_ROOT:?PhysicalAI-AV root}" "${GOLD:?gold eval parquet, or -}" "${BASE:?stage-2 merged base}" "${ADAPTER:?adapter}"
: "${CANDS:?candidate set}"
SPLIT=${SPLIT:-test}; OUT=${OUT:-runs/pai}; NAME=${NAME:-superglance}; PY=${PY:-python}
R=$OUT/pai_openloop_$SPLIT.jsonl; P=$OUT/past_$SPLIT; X=$OUT/features_${NAME}_$SPLIT
mkdir -p "$OUT"
[ -s "$R" ] || OMP_NUM_THREADS=2 "$PY" -m superglance.eval.pai_rows "$PAI_ROOT" "$GOLD" "$OUT" "$SPLIT" "${PROCS:-16}"
echo "rows: $(wc -l < "$R")"
idx=()
if [ "${HIST:-1}" = 1 ]; then
  [ -s "$P/index.jsonl" ] || OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -m superglance.eval.pai_history \
    "$PAI_ROOT" "$R" "$P" 8
  idx=(--index "$P/index.jsonl")
fi
K=$("$PY" -c 'import numpy, sys; print(len(numpy.load(sys.argv[1], mmap_mode="r")))' "$CANDS")
[ -s "$X/code_0.npy" ] || CUDA_VISIBLE_DEVICES=${GPU:-0} "$PY" -m superglance.eval.features --inputs "$R" \
  --tokens "$OUT/pai_openloop_${SPLIT}_tokens.txt" --model "$BASE" --adapter "$ADAPTER" --out "$X" --k "$K" --rank 0 --world 1 \
  ${idx[@]+"${idx[@]}"}
echo "$ADAPTER" > "$X/adapter.txt"   # pai_ade reads the Policy Head for the six most probable candidates from here
"$PY" -m superglance.eval.pai_ade "$R" "$CANDS" "$NAME=$X" | tee "$OUT/ade_${NAME}_$SPLIT.txt"
