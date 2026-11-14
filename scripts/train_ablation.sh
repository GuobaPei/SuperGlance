#!/usr/bin/env bash
# The ablation rows (Table 8, Table 7's "no instruction rows", and the appendix's One change at a time, RL step and AlpaSim
# results by training recipe): each is the stage-3 step of scripts/train_stage3.sh with one switch (a config of
# configs/ablations/, a pool, the start model or the trainer module). Supervised rows run 20,900 steps x 128 rows before the
# last phase; RL rows 2,000 steps x 64 from a supervised row's checkpoint-20900, with the 5,121-candidate head. One row per call:
#   reward_history       the supervised model before its last phase (Policy Head, reward target; "No change")   stage3_sft.json, POOL_SFT
#   imitation_history    Imitation target                      stage3_sft.json, POOL_IMIT (navtrain shard with imitation targets)
#   reward_current       No history frames (RL table: "None")  sft_current_frame.json, POOL_SFT_CF (the pool before its history step)
#   imitation_current    Imitation, current frame              sft_current_frame.json, POOL_IMIT_CF
#   regression           Linear regression                     sft_regression_text.json, POOL_IMIT, superglance.ablations.regression
#   text                 Text, decoded                          sft_regression_text.json, POOL_IMIT, superglance.ablations.text
#   no_rollout_rows / no_wod_rows / no_pai_rows / no_question_data / no_instruction_rows
#                        configs/ablations/sft_<row>.json, POOL_SFT (no_question_data also starts from BASE: no question data anywhere)
#   no_stages12          No stages 1 and 2                     stage3_sft.json, POOL_SFT, from BASE instead of the stage-2 model
#   rl_sampled / rl_exact_normalized / rl_exact   RL from reward_current: rl_k5121.json, POOL_RL_CF, superglance.ablations.rl_estimators
#   rl_exact_beta01      Exact, beta = 0.1                      rl_k5121_beta01.json, POOL_RL_CF
#   rl_imitation         From the imitation target              rl_k5121.json, POOL_RL_IMIT, from imitation_history
#   rl_original_set      History frames + exact RL, original set   rl_k5121.json, POOL_RL_HIST, from reward_history
# RL rows first extract their KL reference (the start checkpoint's head logits on the pool, 8 GPUs) unless <pool>/train_ref.npy exists.
#   STAGE2=<stage-2 merged model> BASE=<Qwen3.5-9B> POOL_SFT=... [POOL_IMIT=... POOL_SFT_CF=... POOL_IMIT_CF=... TRAJ_DIR=...
#   POOL_RL_CF=... POOL_RL_IMIT=... POOL_RL_HIST=...] [OUT=runs/ablations] bash scripts/train_ablation.sh <row>
# Evaluate a row's adapter ($OUT/<row>/train/final, or checkpoint-2000 for RL rows) as the main model: scripts/eval_navsim.sh,
# eval_pai.sh, eval_wod.sh, eval_alpasim.sh with CANDS = the 5,121 library and HIST=0 for the current-frame rows; "No PDM simulator"
# is reward_history under eval_alpasim.sh with EXEC=plan; "no instruction" is eval_l2d.sh's control run.
set -euo pipefail
cd "$(dirname "$0")/.."
row=${1:?row name}; OUT=${OUT:-runs/ablations}; STAGE2=${STAGE2:-runs/stage2/merged}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
A=configs/ablations
train() {   # <module> <config> <pool> <model> [trainer arguments ...]
  local mod=$1 cfg=$2 pool=$3 model=$4; shift 4
  torchrun --standalone --nproc_per_node=8 -m $mod --config "$cfg" --model "$model" --data "$pool" \
    --output "$OUT/$row/train" --pair-mode swap --pair-rounds 1 "$@"
}
rl() {   # <estimator> <config> <pool> <start adapter>
  local est=$1 cfg=$2 pool=$3 init=$4 r
  if [ ! -e "$pool/train_ref.npy" ]; then
    mkdir -p "$OUT/$row/reference"
    for r in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$r python -m superglance.train.rl_reference extract "$pool" "$cfg" "$STAGE2" "$init" "$OUT/$row/reference" $r 8 \
        > "$OUT/$row/reference/extract_$r.log" 2>&1 &
    done
    for r in $(jobs -p); do wait "$r" || { echo "a reference shard failed: $OUT/$row/reference" >&2; exit 1; }; done
    python -m superglance.train.rl_reference merge "$pool" "$OUT/$row/reference" 8
  fi
  train "superglance.ablations.rl_estimators $est" "$cfg" "$pool" "$STAGE2" --init-adapter "$init"
}
P=superglance.train.policy; S=configs/stage3_sft.json; CK=train/checkpoint-20900
case $row in
  reward_history)    train $P $S "${POOL_SFT:?}" "$STAGE2" ;;
  imitation_history) train $P $S "${POOL_IMIT:?imitation pool}" "$STAGE2" ;;
  reward_current)    train $P $A/sft_current_frame.json "${POOL_SFT_CF:?current-frame pool}" "$STAGE2" ;;
  imitation_current) train $P $A/sft_current_frame.json "${POOL_IMIT_CF:?current-frame imitation pool}" "$STAGE2" ;;
  regression|text)   : "${TRAJ_DIR:?superglance.ablations.traj_targets output}"; export TRAJ_DIR
                     train superglance.ablations.$row $A/sft_regression_text.json "${POOL_IMIT:?imitation pool}" "$STAGE2" ;;
  no_rollout_rows|no_wod_rows|no_pai_rows|no_instruction_rows) train $P $A/sft_$row.json "${POOL_SFT:?}" "$STAGE2" ;;
  no_question_data)  train $P $A/sft_no_question_data.json "${POOL_SFT:?}" "${BASE:?Qwen3.5-9B}" ;;
  no_stages12)       train $P $S "${POOL_SFT:?}" "${BASE:?Qwen3.5-9B}" ;;
  rl_sampled)          rl sampled_normalized $A/rl_k5121.json "${POOL_RL_CF:?}" "$OUT/reward_current/$CK" ;;
  rl_exact_normalized) rl exact_normalized $A/rl_k5121.json "${POOL_RL_CF:?}" "$OUT/reward_current/$CK" ;;
  rl_exact)            rl exact $A/rl_k5121.json "${POOL_RL_CF:?}" "$OUT/reward_current/$CK" ;;
  rl_exact_beta01)     rl exact $A/rl_k5121_beta01.json "${POOL_RL_CF:?}" "$OUT/reward_current/$CK" ;;
  rl_imitation)        rl exact $A/rl_k5121.json "${POOL_RL_IMIT:?}" "$OUT/imitation_history/$CK" ;;
  rl_original_set)     rl exact $A/rl_k5121.json "${POOL_RL_HIST:?}" "$OUT/reward_history/$CK" ;;
  *) echo "unknown row $row" >&2; exit 2 ;;
esac
