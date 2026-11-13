#!/usr/bin/env bash
# WOD-E2E test submission (Table 5): the challenge's server scores RFS and ADE; this writes and checks the packages. Each step
# is skipped when its output exists:
#   rows      superglance.eval.wod_test_rows: the frames of the submission list, from the test split's TFRecord shards
#   inputs    superglance.builders.pools.wod_rfs inputs: scene rows + token list
#   features  superglance.eval.features on the current frame alone (the test split has no past frames); K = the candidate set's size
#   readouts  superglance.eval.wod_decode: mean02 (the paper's: the tau 0.2 softmax mean of the head's top 64) and mbr02
#   packages  superglance.eval.wod_submit build + check, for mean02, mbr02 and the argmax (the features dir itself)
# Then fill the metadata of the package you submit with the printed command and your own values, and upload its .tar.gz on
# the challenge's submission page. Nothing is uploaded here.
#   WOD_TEST=<dir of the test split's TFRecord shards> WOD_LIST=<test_sequence_frames_for_submission_as_list.txt>
#   BASE=<stage-2 merged base> ADAPTER=<adapter> CANDS=<candidate set .npy> [WOD_PROTOS=<dir of the generated protos>]
#   [OUT=runs/wod] [NAME=superglance] [WORKERS=8] [GPU=0] bash scripts/eval_wod.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${WOD_TEST:?test TFRecord dir}" "${WOD_LIST:?submission frame list}" "${BASE:?stage-2 merged base}" "${ADAPTER:?adapter}"
: "${CANDS:?candidate set}"
OUT=${OUT:-runs/wod}; NAME=${NAME:-superglance}; PY=${PY:-python}; RW=$OUT/rows; X=$OUT/features
mkdir -p "$OUT"
"$PY" -m superglance.eval.wod_test_rows "$WOD_LIST" "$RW" "${WORKERS:-8}" "$WOD_TEST"/*   # skips shards with a .done
[ -s "$RW/wod_test_inputs.jsonl" ] || "$PY" -m superglance.builders.pools.wod_rfs inputs "$RW" "$RW/wod_test"
K=$("$PY" -c 'import numpy, sys; print(len(numpy.load(sys.argv[1], mmap_mode="r")))' "$CANDS")
[ -s "$X/code_0.npy" ] || CUDA_VISIBLE_DEVICES=${GPU:-0} "$PY" -m superglance.eval.features --inputs "$RW/wod_test_inputs.jsonl" \
  --tokens "$RW/wod_test_tokens.txt" --model "$BASE" --adapter "$ADAPTER" --out "$X" --k "$K" --rank 0 --world 1
[ -d "$OUT/${NAME}_mean02" ] || "$PY" -m superglance.eval.wod_decode "$X" "$ADAPTER" "$CANDS" "$RW" "$OUT/$NAME"
for v in mean02 mbr02 argmax; do
  case $v in
    mean02) XD=$OUT/${NAME}_mean02; C=$XD/cands.npy ;;
    mbr02) XD=$OUT/${NAME}_mbr02; C=$CANDS ;;
    argmax) XD=$X; C=$CANDS ;;
  esac
  [ -s "$OUT/sub/wodtest_$NAME$v.tar.gz" ] || "$PY" -m superglance.eval.wod_submit build "$NAME$v" "$XD" "$C" "$OUT/sub" "$WOD_LIST"
  "$PY" -m superglance.eval.wod_submit check "$OUT/sub/wodtest_$NAME$v" "$WOD_LIST" "$RW" > "$OUT/check_$NAME$v.txt"
  echo "$v: $(grep '^predictions' "$OUT/check_$NAME$v.txt")"
done
cat << EOF
To submit the paper's readout, fill its package with your own values, then upload $OUT/sub/wodtest_${NAME}mean02.tar.gz:
  $PY -m superglance.eval.wod_submit fill $OUT/sub/wodtest_${NAME}mean02 --account-name <account> --method-name <method name> \\
    --authors <author> [<author> ...] --affiliation <affiliation> --description "<description>" --method-link <link> \\
    --uses-public-model-pretraining true --public-model-names <base model> --num-model-parameters <N>B
EOF
