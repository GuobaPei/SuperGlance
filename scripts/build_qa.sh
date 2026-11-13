#!/usr/bin/env bash
# Stage 1-2 question pools from their public sources, in the order the paper's pools were built (released on the Hub; this
# rebuilds them). Each step writes under ROOT; a step's output is the next one's input:
#   sat         SAT-v2 static / val / test -> prepared_v2 (50,000 train, 2,000 dev; val / test single-image sets: SAT dev)
#   evals       CV-Bench + BLINK (eval_external) and 3DSRBench (eval_3dsr): Table 1 benchmarks, and images kept out of training
#   supplement  GQA questions + 2 image shards, Visual Genome metadata, tallyqa.zip -> supplement_gqa/raw (read by adapt)
#   sources     SAT train, GQA, The Cauldron's AI2D / TallyQA / ChartQA / TextVQA -> raw dirs under ROOT
#   perspective SAT train left / right perspective rows -> perspective_repair_data_20260921 (a split seed of adapt)
#   adapt       every source -> stage1_adapted_20260921 (+ check_stage1_pool)
#   flips       GQA relation-flip minimal pairs -> stage1_adapted_flip_20260925
#   e           OpenImages box choice + SpatialSense relations -> stage1_adapted_e_20260925
#   f           13 text sources (blank image) + retention evals -> stage1_adapted_f_20260925
#   g           OpenImages "directly above" pairs + "none of the other options" text rows -> stage1_adapted_g_20260926 (stage 1)
#   s2          DriveLM-nuScenes v1.1 box / relation / QA rows on the f pool -> stage2_drivelm_20260925 (stage 2)
#   ROOT=<data root> [TALLYQA_URL=<tallyqa.zip of the TallyQA release>] [DRIVELM_QA=<v1_1_train_nus.json>]
#     [DRIVELM_IMAGES=<extracted DriveLM nuScenes images>] [EVAL_DIRS="<eval image dirs>"] bash scripts/build_qa.sh [step ...]
# Inputs a step expects already on disk: e: ROOT/external_e_20260925/openimages (OpenImages V5 boxable val / test boxes,
# class-descriptions-boxable.csv) and .../spatialsense (annotations.json, images); f: ROOT/text_f_20260925/raw/<dataset id>/...
# (the Hub datasets prepare_text_f.py names); adapt: ROOT/supplement_multichoice (the earlier supplements' split seeds and
# image_split.json; their builder is not part of this release).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ROOT:?data root}"
EVAL_DIRS=${EVAL_DIRS:-"$ROOT/eval_external/prepared/images $ROOT/eval_3dsr/prepared/images"}
EX=$ROOT/external_e_20260925; TX=$ROOT/text_f_20260925; DL=$ROOT/drivelm_20260925
q() { python -m "superglance.builders.qa.$1" "${@:2}"; }
for step in ${@:-sat evals supplement sources perspective adapt flips e f g s2}; do
  case $step in
    sat) q download_sat --output "$ROOT/raw"
         q prepare_sat --source "$ROOT/raw" --output "$ROOT/prepared_v2" --train-size 50000 --dev-size 2000 ;;
    evals) q prepare_external_eval --root "$ROOT/eval_external"; q prepare_3dsr_eval --root "$ROOT/eval_3dsr" ;;
    supplement) q prepare_supplement --root "$ROOT/supplement_gqa" --eval-manifest "$ROOT/eval_external/prepared/cvbench.jsonl" \
                  --eval-manifest "$ROOT/eval_3dsr/prepared/3dsr_real_circular.jsonl" --train-size 8000 --dev-size 800 \
                  --count-train-size 2000 --count-dev-size 200 --tallyqa-url "${TALLYQA_URL:?tallyqa.zip of the TallyQA release}" ;;
    sources) q download_stage1_pool --root "$ROOT" ;;
    perspective) q prepare_perspective_pool --source "$ROOT/raw_perspective_train" --static "$ROOT/prepared_v2" \
                   --output "$ROOT/perspective_repair_data_20260921" ;;
    adapt) [ -s "$ROOT/supplement_multichoice/image_split.json" ] || { echo "adapt needs $ROOT/supplement_multichoice (see above)" >&2; exit 1; }
           q adapt_stage1 --root "$ROOT" --output "$ROOT/stage1_adapted_20260921"
           q check_stage1_pool --data "$ROOT/stage1_adapted_20260921" ;;
    flips) q prepare_relation_flips --pool "$ROOT/stage1_adapted_20260921" --output "$ROOT/stage1_adapted_flip_20260925" ;;
    e) for s in openimages fetch render spatialsense; do q prepare_external_e $s --root "$EX"; done
       q prepare_external_e dedup --root "$EX" --eval-dirs $EVAL_DIRS
       q prepare_external_e assemble --root "$EX" --base-pool "$ROOT/stage1_adapted_flip_20260925" --output "$ROOT/stage1_adapted_e_20260925" ;;
    f) q prepare_text_f evals --root "$TX"; q prepare_text_f train --root "$TX"
       q prepare_text_f assemble --root "$TX" --base-pool "$ROOT/stage1_adapted_e_20260925" --output "$ROOT/stage1_adapted_f_20260925" ;;
    g) for s in select fetch; do q prepare_g $s --root "$EX"; done
       q prepare_g dedup --root "$EX" --eval-dirs $EVAL_DIRS
       q prepare_g generate --root "$EX"; q prepare_g nota --root "$EX" --base-pool "$ROOT/stage1_adapted_f_20260925"
       q prepare_g assemble --root "$EX" --base-pool "$ROOT/stage1_adapted_f_20260925" --output "$ROOT/stage1_adapted_g_20260926" ;;
    s2) : "${DRIVELM_QA:?v1_1_train_nus.json}" "${DRIVELM_IMAGES:?DriveLM nuScenes images}"
        q prepare_drivelm_s2 generate --qa "$DRIVELM_QA" --images "$DRIVELM_IMAGES" --output "$DL/s2_items" --eval-dirs $EVAL_DIRS
        q prepare_drivelm_s2 qa --qa "$DRIVELM_QA" --images "$DRIVELM_IMAGES" --output "$DL/qa_items" --eval-dirs $EVAL_DIRS
        q prepare_drivelm_s2 assemble --qa "$DRIVELM_QA" --root "$DL" --base-pool "$ROOT/stage1_adapted_f_20260925" \
          --output "$ROOT/stage2_drivelm_20260925" ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done
