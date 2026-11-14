#!/usr/bin/env bash
# Stage-3 pools (SFT, the last phase, and the K' reward columns of the RL pool) in the order the paper's pools were built. Their
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
#            softmax(proxy RFS / 0.5 - (ADE - min ADE) / 1 m) -> $LAST_POOL
#   rl       the RL pool ($RL_POOL, 5,121 reward columns) with the K' candidates' reward columns appended -> $RL_OUT
# Source shards (superglance.builders.shards, step "shards", under $SHARDS): navtrain (imitation targets) and navtrain_pdms (PDMS soft
# targets from score's table: NAV_SHARD, the paper's), navtrain_os (OpenScene frames outside navtrain: OS_SHARD), pai_human/pNN
# (PAI_SHARDS), pai_dagger (rollout rows: DAGGER_SHARD), l2d (instruction rows, superglance.builders.instructions: L2D_SHARD),
# qa_subset (QA_SUBSET), e2e_inputs.npz (E2E_INPUTS). Step "paihist" makes $MF/hist_pai, step "rlpool" the 5,121-reward RL pool
# ($RL5121, the RL ablations' and the K' pool's base: RL_POOL), step "stage_cf" the current-frame pool of the ablations.
# Earlier pools and run outputs these builders read (README, "Stage 3: policy pools"): OLD_POOL (the earlier pool: navtrain dev
# tokens, rollout rows and their images, PAI-AV rows for the instruction data), ROUNDS (the rollout rounds' rows), RR_META (the
# recovery labels' meta files), OLD_FIT (the earlier PAI-AV fit rows), the RL pool's inputs (RL_E2E_POOL, RL_DRAW, RL_FIRST,
# C4096, NAV_PDMS, FIT_TOKENS), and the scene lists VAL441, TRA232, S916, REALSIM, HELD2, DAG_ROWS.
# Other inputs: CANDS (5,121 library), CANDS_KP (K' library, 5,273), OPENSCENE_DATA_ROOT, NUPLAN_MAPS_ROOT, NAVSIM_DEVKIT_ROOT,
# HELDOUT_TOKENS, PAI_ROOT, L2D_ROOT, STAGE2 (stage-2 pool dir), STAGE2_CONFIG (its run config), WOD_TRAIN_SHARDS and WOD_VAL_SHARDS
# (WOD-E2E training / val shards), $WOD_ROWS/splits/{training,val,test}_segments.txt, PAI_CLIP_INDEX, PAI_GOLD, ALPASIM_916,
# SIM_SUITES, PAI_OPENLOOP (eval_keys.py's arguments), SFT_CONFIG (the stage-3 training config), RL_DEV, KP_W, CHECK.
# NAV_TARGET=imitation builds the imitation pools of the ablations (navtrain shard with softmax(-ADE / 1 m) targets).
#   OUT=<dir> <inputs> bash scripts/build_pools.sh [step ...]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OUT:?output root}"
NAV=${NAV:-$OUT/navsim}; MF=${MF:-$OUT/mf}; IMG=${IMG:-$OUT/img}; WOD_ROWS=${WOD_ROWS:-$OUT/wod_rows}; PROCS=${PROCS:-48}
SFT_POOL=${SFT_POOL:-$OUT/sft_pool}; LAST_POOL=${LAST_POOL:-$OUT/last_pool}; RL_OUT=${RL_OUT:-$OUT/rl_pool_kp}
SHARDS=${SHARDS:-$OUT/shards}; export SHARDS CANDS; L2D_DATA=${L2D_DATA:-$OUT/l2d_data}; RL5121=${RL5121:-$OUT/rl_pool}
[ "${NAV_TARGET:-pdms}" = imitation ] && NAV_SHARD=${NAV_SHARD:-$SHARDS/navtrain} || NAV_SHARD=${NAV_SHARD:-$SHARDS/navtrain_pdms}
DAGGER_SHARD=${DAGGER_SHARD:-$SHARDS/pai_dagger}; L2D_SHARD=${L2D_SHARD:-$SHARDS/l2d}; OS_SHARD=${OS_SHARD:-$SHARDS/navtrain_os}
QA_SUBSET=${QA_SUBSET:-$SHARDS/qa_subset}; E2E_INPUTS=${E2E_INPUTS:-$SHARDS/e2e_inputs.npz}; RL_POOL=${RL_POOL:-$RL5121/mf}
PAI_SHARDS=${PAI_SHARDS:-$(printf "$SHARDS/pai_human/p%02d " $(seq 0 15))}
sd() { python -m "superglance.builders.shards.$1" "${@:2}"; }
ins() { python -m "superglance.builders.instructions.$1" "${@:2}"; }
copies() { for d in "$@"; do cut -f1,2 "$d"/*_images.tsv | sort -u > "$d/copies.tsv"; done; }   # merge's image list per shard
q() { python -m "superglance.builders.pools.$1" "${@:2}"; }
drop_eval() { q clean "$1" "$2" "$OUT/eval_keys.json" "$E2E_INPUTS" "$WOD_ROWS/splits"; }
for step in ${@:-export score shards wod qa dev merge clean paihist history stage last rlpool rl}; do
  case $step in
    shards) # every input row, image and label file here comes from the datasets and the earlier pools listed above
      sd navtrain "$NAV/inputs/navtrain_inputs.jsonl" "${OLD_POOL:?earlier pool}"
      sd pdms_targets "$OUT/score/scores.npz" "$SHARDS/navtrain/dev.jsonl" "$CANDS" "$NAV/inputs/navtrain_inputs.jsonl" "$OUT/score/q_nav.npz"
      sd navtrain_pdms "$SHARDS/navtrain" "$OUT/score/q_nav.npz" "$SHARDS/navtrain_pdms"
      D=$NAV NAV_INPUTS=$NAV/inputs/navtrain_inputs.jsonl sd openscene_export --split osx
      NAV_INPUTS=$NAV/inputs/navtrain_inputs.jsonl sd openscene "$NAV/inputs/osx_inputs.jsonl"
      sd pai_inputs --clips "$PAI_ROOT" "$SHARDS/pai_in"; sd pai_inputs "$PAI_ROOT" "$SHARDS/pai_in" "$SHARDS/img" "$PROCS"
      OLD_ROWS=${OLD_ROWS:-$OLD_POOL/train.jsonl} OLD_FIT=${OLD_FIT:?earlier PAI-AV fit rows} sd pai_rows
      RR_META=${RR_META:?recovery-label meta files} PAI_FIT_ROWS=$OLD_FIT sd e2e_inputs "$E2E_INPUTS"
      sd dagger "$OLD_POOL" "$E2E_INPUTS" "${ROUNDS:?rollout rounds}"
      export NAV_INPUTS=$NAV/inputs/navtrain_inputs.jsonl NAVTEST_INPUTS=$NAV/inputs/navtest_inputs.jsonl NAV_SCORES=$OUT/score \
             PAI_FIT_ROWS=$OLD_FIT OLD_POOL WOD_META=$WOD_ROWS/train5/meta L2D_IMG_ROOT=${L2D_IMG_ROOT:-$L2D_DATA/staged}
      q wod_rows --sel every:5 --out "$WOD_ROWS/train5" $WOD_TRAIN_SHARDS
      ins navtrain pdm "$L2D_DATA"; ins navtrain label "$L2D_DATA" 24; ins pai "$L2D_DATA" 20; ins wod "$L2D_DATA"
      ins build "$L2D_DATA"; ins check "$L2D_DATA"
      sd instructions "$L2D_DATA" "$L2D_SHARD"
      awk -F'\t' '{print $1 "\tl2d/" $2}' "$L2D_DATA/manifest.tsv" > "$L2D_SHARD/copies.tsv"
      sd qa_subset "$STAGE2" "${STAGE2_CONFIG:?stage-2 run config}" "$QA_SUBSET"
      copies "$SHARDS/navtrain_pdms" "$SHARDS/navtrain" "$SHARDS/navtrain_os" "$DAGGER_SHARD" $PAI_SHARDS ;;
    paihist) python -m superglance.builders.shards.pai_history "$PAI_ROOT" "$SHARDS/pai_in" "$MF/hist_pai" "$PROCS" ;;
    rlpool) # the 5,121-reward RL pool: rows and rewards, then without the road-users block, then with history frames
      q rl_rows rows "${RL_E2E_POOL:?earlier e2e pool}" "$NAV/inputs/navtrain_inputs.jsonl" "${FIT_TOKENS:?fit tokens}" "$E2E_INPUTS" \
        "$CANDS" "${C4096:?4,097-candidate scores}" "${RL_DRAW:?rl_draw dir}" "${RL_FIRST:?first RL pool}" "$RL5121/rows"
      q rl_rows final "$RL5121/rows" "${NAV_PDMS:?merged navtrain table}" "$RL_DRAW"
      q rl_strip "$RL5121/rows" "$RL5121/sf"
      q history_rows "$RL5121/sf" "$MF/hist_nav" "$MF/hist_pai" "$RL5121/mf" ;;
    stage_cf) MF_POOL=$(realpath --relative-to="$MF" "$OUT/pool_sft_clean") q stage "$MF" "$IMG" "${SFT_POOL_CF:-$OUT/sft_pool_cf}" 16 ;;
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
           # one cp per image (millions); run the copy with xargs -P if it is the bottleneck
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
          CANDS=$CANDS_KP q wod_last_phase sftpool t1 "$SFT_POOL/train_targets.npy" "$SFT_POOL/hist" "$LAST_POOL" ;;
    rl) CANDS=$CANDS CHECK=$CHECK q rl_rewards_kprime "$RL_POOL" "$KP_W" "$RL_OUT" ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done
