#!/usr/bin/env bash
# Stage 2, driving QA: a fresh LoRA on the merged stage-1 model (the KL anchor, adapter disabled), DriveLM questions with
# stage-1 questions in the batches; then merged into the stage-1 model. The merged stage-2 model is stage 3's base.
#   STAGE1=<runs/stage1/merged> DATA=<stage-2 questions: train.jsonl, dev.jsonl> [OUT=runs/stage2] bash scripts/train_stage2.sh
# 8 GPUs x 4 x 2 = 64 questions per step, 3,000 steps, DeepSpeed ZeRO-2.
# The merged adapter is the last step's (checkpoint-3000), which the paper's run selected; dev.jsonl does not
# change it (see superglance/data/build_qa.py).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA:?stage-2 question directory}"
STAGE1=${STAGE1:-runs/stage1/merged}; OUT=${OUT:-runs/stage2}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
torchrun --standalone --nproc_per_node=8 -m superglance.train.qa --config configs/stage2_driving_qa.json \
  --model "$STAGE1" --data "$DATA" --output "$OUT/train" --pair-mode swap --pair-rounds 1 "$@"
CUDA_VISIBLE_DEVICES="" python -m superglance.model.merge --model "$STAGE1" --adapter "$OUT/train/checkpoint-3000" --output "$OUT/merged"
