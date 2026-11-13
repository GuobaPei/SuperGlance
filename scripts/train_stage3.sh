#!/usr/bin/env bash
# Stage 3, policy training on the merged stage-2 model (the KL anchor of the QA replay rows), in four steps:
#   sft        5,121 candidates: soft targets (log score) + QA replay, 20,900 steps x 128 rows   (configs/stage3_sft.json)
#   last       the last supervised phase, WOD-E2E weight 0.4, 2,500 steps x 128 from sft's last checkpoint
#                                                                                     (configs/stage3_sft_last_phase.json)
#   reference  the supervised policy's head logits on the RL pool -> $POOL_RL/train_ref.npy (8 GPUs, one shard each)
#   rl         exact expected reward + 0.3 KL to the supervised policy, 2,000 steps x 64 rows      (configs/stage3_rl.json)
# Between last and reference the Policy Head is refit from 5,121 to 5,273 rows on the supervised model's features (see
# README, "Head refit"); REFIT names that adapter.
#   STAGE2=<runs/stage2/merged> POOL_SFT=<pool> POOL_LAST=<pool> POOL_RL=<pool> REFIT=<5,273-row adapter> [OUT=runs/stage3] \
#     bash scripts/train_stage3.sh [sft] [last] [reference] [rl]      (no step names: all four in order)
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE2=${STAGE2:-runs/stage2/merged}; OUT=${OUT:-runs/stage3}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
policy() {   # <config> <pool> <run dir> [trainer arguments ...]
  local cfg=$1 pool=$2 run=$3; shift 3
  torchrun --standalone --nproc_per_node=8 -m superglance.train.policy --config "$cfg" --model "$STAGE2" --data "$pool" \
    --output "$run/train" --pair-mode swap --pair-rounds 1 "$@"
}
for step in ${@:-sft last reference rl}; do
  case $step in
    sft) policy configs/stage3_sft.json "${POOL_SFT:?supervised pool}" "$OUT/sft" ;;
    last) policy configs/stage3_sft_last_phase.json "${POOL_LAST:?last-phase pool}" "$OUT/last" \
            --init-adapter "$OUT/sft/train/checkpoint-20900" ;;
    reference)
      mkdir -p "$OUT/reference"
      for r in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$r python -m superglance.train.rl_reference extract "${POOL_RL:?RL pool}" configs/stage3_rl.json \
          "$STAGE2" "${REFIT:?5,273-row adapter}" "$OUT/reference" $r 8 > "$OUT/reference/extract_$r.log" 2>&1 &
      done
      wait
      python -m superglance.train.rl_reference merge "$POOL_RL" "$OUT/reference" 8 ;;
    rl) policy configs/stage3_rl.json "${POOL_RL:?RL pool}" "$OUT/rl" --init-adapter "${REFIT:?5,273-row adapter}" ;;
    *) echo "unknown step $step (sft, last, reference, rl)" >&2; exit 2 ;;
  esac
done
