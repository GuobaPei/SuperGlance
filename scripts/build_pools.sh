#!/usr/bin/env bash
# Stage-3 pools (SFT, the S1 last phase, and the K' reward columns of the RL pool) in the order the paper's pools were built. Their
# NAVSIM, PAI-AV and WOD-E2E rows cannot be redistributed, so they are rebuilt from the source datasets. Steps:
#   export   NAVSIM v1.1 navtrain / navtest scenes + metric caches -> $NAV/inputs/{navtrain,navtest}_inputs.jsonl (navsim env)
#   score    the 5,121 candidates on every navtrain scene with the NAVSIM PDM scorer -> $OUT/score (scores.npz; q_t01b01.npz, the
#            navtrain PDMS soft targets the navtrain shard uses)
#   wod      WOD-E2E rows: every training frame -> $WOD_ROWS/train1, rated val frames -> $WOD_ROWS/val_rated; the WOD shard
#   qa       the QA replay shard from the stage-2 pool
#   dev      200 dev rows each (seed 20261023) from the PAI-AV human shards' and the Language2Drive shard's val splits
#   merge    the shards, in the order of the paper's pool -> $OUT/pool_sft, then the image root $IMG from its copies.tsv
#   clean    evaluation keys; every training row of an evaluation set dropped -> $OUT/pool_sft_clean
#   history  front-camera history: frames ($MF/hist_nav), rows ($MF/pool_sftmf), clean again as a check (0 rows drop)
#   stage    rows, targets, images and history -> $SFT_POOL (the SFT pool), checked against $SFT_CONFIG
#   last     WOD rows at training frames 135-165 -> $OUT/wod_last; $SFT_POOL with its WOD rows replaced, target t1 =
#            softmax(proxy RFS / 0.5 - (ADE - min ADE) / 1 m) -> $S1_POOL
#   rl       the RL pool ($RL_POOL, 5,121 reward columns) with the K' candidates' reward columns appended -> $RL_OUT
# Inputs made by builders that are not in this release (README, "Stage 3: policy pools"): the shard dirs NAV_SHARD (navtrain, PDMS soft
# targets), DAGGER_SHARD, PAI_SHARDS (the 16 PAI-AV human parts), L2D_SHARD (Language2Drive), OS_SHARD (OpenScene frames outside
# navtrain); QA_SUBSET (report.json and weights.json of the earlier QA subset), E2E_INPUTS (e2e_inputs.npz), $MF/hist_pai (PAI-AV
# history tars), RL_POOL and RL_DEV (its dev.jsonl), KP_W (the K' candidates' tables), CHECK (the reward check's inputs).
# Other inputs: CANDS (5,121 library), CANDS_KP (K' library, 5,273), OPENSCENE_DATA_ROOT, NUPLAN_MAPS_ROOT, NAVSIM_DEVKIT_ROOT,
# HELDOUT_TOKENS, OS_INPUTS (the OpenScene rows' inputs.jsonl, for their history), STAGE2 (stage-2 pool dir), WOD_TRAIN_SHARDS and
# WOD_VAL_SHARDS (WOD-E2E training / val shards), $WOD_ROWS/splits/{training,val,test}_segments.txt, PAI_CLIP_INDEX, PAI_GOLD,
# ALPASIM_916, SIM_SUITES, PAI_OPENLOOP (eval_keys.py's arguments), SFT_CONFIG (the stage-3 training config).
#   OUT=<dir> <inputs> bash scripts/build_pools.sh [step ...]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OUT:?output root}"
NAV=${NAV:-$OUT/navsim}; MF=${MF:-$OUT/mf}; IMG=${IMG:-$OUT/img}; WOD_ROWS=${WOD_ROWS:-$OUT/wod_rows}; PROCS=${PROCS:-48}
SFT_POOL=${SFT_POOL:-$OUT/sft_pool}; S1_POOL=${S1_POOL:-$OUT/s1_pool}; RL_OUT=${RL_OUT:-$OUT/rl_pool_kp}
q() { python -m "superglance.builders.pools.$1" "${@:2}"; }
drop_eval() { q clean "$1" "$2" "$OUT/eval_keys.json" "$E2E_INPUTS" "$WOD_ROWS/splits"; }
for step in ${@:-export score wod qa dev merge clean history stage last rl}; do
  case $step in
    export) mkdir -p "$NAV/inputs" "$NAV/status"
            for s in navtrain navtest; do D=$NAV q navsim_export --split $s; done ;;
    score) q score_candidates "$CANDS" "$NAV/inputs/navtrain_inputs.jsonl" "$OUT/score" "$PROCS" ;;
    wod) q wod_rows --sel every:1 --out "$WOD_ROWS/train1" $WOD_TRAIN_SHARDS
         q wod_rows --sel rated --out "$WOD_ROWS/val_rated" $WOD_VAL_SHARDS
         q wod_shard "$WOD_ROWS/train1" "$CANDS" "$OUT/shards/wod" ;;
    qa) q qa_replay "$STAGE2" "$QA_SUBSET" "$OUT/shards" ;;
    dev) q dev_draw "$OUT/shards/dev" 200 20261023 $PAI_SHARDS; q dev_draw "$OUT/shards/devl2d" 200 20261023 "$L2D_SHARD" ;;
    merge) q merge "$OUT/pool_sft" "$NAV_SHARD" "$DAGGER_SHARD" "$OUT/shards/wod" "$OUT/shards/qa" $PAI_SHARDS "$OUT/shards/dev" \
             "$L2D_SHARD" "$OUT/shards/devl2d" "$OS_SHARD"
           mkdir -p "$IMG"; ln -sfn "$OUT/shards/img/qa" "$IMG/qa"
           # ponytail: one cp per image (millions); run the copy with xargs -P if it is the bottleneck
           while IFS=$'\t' read -r s r; do [ -e "$IMG/$r" ] || { mkdir -p "$IMG/${r%/*}"; cp "$s" "$IMG/$r"; }; done < "$OUT/pool_sft/copies.tsv" ;;
    clean) q eval_keys "$OUT/eval_keys.json" "$PAI_CLIP_INDEX" "$PAI_GOLD" "$ALPASIM_916" "$SIM_SUITES" \
             "$NAV/inputs/navtest_inputs.jsonl" "$PAI_OPENLOOP"
           drop_eval "$OUT/pool_sft" "$OUT/pool_sft_clean" ;;
    history) q history_nav "$MF/hist_nav" "$PROCS" "$NAV/inputs/navtrain_inputs.jsonl" ${OS_INPUTS:-}
             q history_rows "$OUT/pool_sft_clean" "$MF/hist_nav" "$MF/hist_pai" "$MF/pool_sftmf"
             drop_eval "$MF/pool_sftmf" "$MF/pool_sftmf_clean" ;;
    stage) MF_POOL=pool_sftmf_clean q stage "$MF" "$IMG" "$SFT_POOL" 16
           q check "$SFT_POOL" "$SFT_CONFIG" ;;
    last) export WOD_ROWS SFT_POOL RL_DEV="$RL_DEV" IMG_ROOT="$IMG" LAST_ROWS="$OUT/wod_last"
          CANDS=$CANDS_KP q wod_last_phase rows "$OUT/wod_last"
          CANDS=$CANDS_KP q wod_last_phase sftpool t1 "$SFT_POOL/train_targets.npy" "$SFT_POOL/hist" "$S1_POOL" ;;
    rl) CANDS=$CANDS CHECK=$CHECK q rl_rewards_kprime "$RL_POOL" "$KP_W" "$RL_OUT" ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done
