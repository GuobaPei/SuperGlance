#!/usr/bin/env bash
# Stage 1, general QA: a fresh rank-32 LoRA on Qwen3.5-9B, joint Brier policy gradient on two option orders, KL 0.1 to the
# base model; then the adapter is merged into the base (CPU).
#   BASE=<Qwen3.5-9B> DATA=<stage-1 questions: train.jsonl, dev.jsonl> [OUT=runs/stage1] bash scripts/train_stage1.sh
# 8 GPUs x micro-batch 4 x accumulation 2 = 64 questions (128 views) per step, 7,813 steps, DeepSpeed ZeRO-2.
# Extra arguments go to the trainer (e.g. --resume-from-checkpoint). If memory is short, set gradient_checkpointing to true
# in the config: the paper's run resumed that way after running out of memory at step 4,056 (it changes no update).
# The merged adapter is the last step's (checkpoint-7813), which the paper's run selected; dev.jsonl does not
# change it (see superglance/data/build_qa.py).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BASE:?Qwen3.5-9B directory}" "${DATA:?stage-1 question directory}"
OUT=${OUT:-runs/stage1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
torchrun --standalone --nproc_per_node=8 -m superglance.train.qa --config configs/stage1_general_qa.json \
  --model "$BASE" --data "$DATA" --output "$OUT/train" --pair-mode swap --pair-rounds 1 "$@"
CUDA_VISIBLE_DEVICES="" python -m superglance.model.merge --model "$BASE" --adapter "$OUT/train/checkpoint-7813" --output "$OUT/merged"
