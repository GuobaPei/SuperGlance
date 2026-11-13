#!/usr/bin/env bash
# Stage 1, general QA: a fresh rank-32 LoRA on Qwen3.5-9B, joint Brier policy gradient on two option orders, KL 0.1 to the
# base model; then the adapter is merged into the base (CPU).
#   BASE=<Qwen3.5-9B> DATA=<stage-1 questions: train.jsonl, dev.jsonl> [OUT=runs/stage1] bash scripts/train_stage1.sh
# 8 GPUs x micro-batch 4 x accumulation 2 = 64 questions (128 views) per step, 7,813 steps, DeepSpeed ZeRO-2.
# Extra arguments go to the trainer (e.g. --resume-from-checkpoint). If memory is tight, set gradient_checkpointing to
# true in the config: it trades compute for memory, the math is the same.
# Without dev.jsonl (the released data has none) the trainer runs no evaluation and final/ is the last step, the
# checkpoint the paper's run selected.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BASE:?Qwen3.5-9B directory}" "${DATA:?stage-1 question directory}"
OUT=${OUT:-runs/stage1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
torchrun --standalone --nproc_per_node=8 -m superglance.train.qa --config configs/stage1_general_qa.json \
  --model "$BASE" --data "$DATA" --output "$OUT/train" --pair-mode swap --pair-rounds 1 "$@"
CUDA_VISIBLE_DEVICES="" python -m superglance.model.merge --model "$BASE" --adapter "$OUT/train/final" --output "$OUT/merged"
