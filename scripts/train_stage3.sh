#!/usr/bin/env bash
# Stage 3, policy training on the merged stage-2 model (the KL anchor of the QA replay rows), in five steps:
#   sft        5,121 candidates: soft targets (log score) + QA replay, 20,900 steps x 128 rows   (configs/stage3_sft.json)
#   last       the last supervised phase, WOD-E2E weight 0.4, 2,500 steps x 128 from sft's last checkpoint
#                                                                                     (configs/stage3_sft_last_phase.json)
#   refit      the Policy Head widened from 5,121 to 5,273 rows on last's features (README, "Head refit"): features of the
#              five sets, L-BFGS fit of the 152 new rows, the widened adapter, and the stop rule (navtest PDMS at most 0.1
#              below the 5,121 head's, PAI-AV val ADE@3 s at most 0.005 m above it) -> $REFIT
#   reference  the supervised policy's head logits on the RL pool -> $POOL_RL/train_ref.npy (8 GPUs, one shard each)
#   rl         exact expected reward + 0.3 KL to the supervised policy, 2,000 steps x 64 rows      (configs/stage3_rl.json)
#   STAGE2=<runs/stage2/merged> POOL_SFT=<pool> POOL_LAST=<pool> POOL_RL=<pool> CANDS=<5,121 library> \
#     REFIT_ROWS=<superglance.refit.rows output> REFIT_DATA=<refit data dir> HELDOUT=<navtrain held-out tokens> \
#     [OUT=runs/stage3] bash scripts/train_stage3.sh [sft] [last] [refit] [reference] [rl]   (no step names: all five in order)
# REFIT defaults to the refit step's output; set it to start reference and rl from another 5,273-row adapter.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE2=${STAGE2:-runs/stage2/merged}; OUT=${OUT:-runs/stage3}; REFIT=${REFIT:-$OUT/refit/adapter_k5273}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
policy() {   # <config> <pool> <run dir> [trainer arguments ...]
  local cfg=$1 pool=$2 run=$3; shift 3
  torchrun --standalone --nproc_per_node=8 -m superglance.train.policy --config "$cfg" --model "$STAGE2" --data "$pool" \
    --output "$run/train" --pair-mode swap --pair-rounds 1 "$@"
}
waitall() {   # every background job must succeed
  local p; for p in $(jobs -p); do wait "$p" || { echo "a shard failed ($1): see its log" >&2; exit 1; }; done
}
features() {   # <gpu> <set> <rank> <world>: the 5,121-row head's features of one refit set (as the paper's run: one GPU each)
  local o=$OUT/refit/feat/$2; mkdir -p "$o"
  CUDA_VISIBLE_DEVICES=$1 python -m superglance.eval.features --inputs "$REFIT_ROWS/$2_inputs.jsonl" --tokens "$REFIT_ROWS/$2_tokens.txt" \
    --index "$REFIT_ROWS/$2_index.jsonl" --model "$STAGE2" --adapter "$OUT/last/train/final" --out "$o" --rank $3 --world $4 \
    --k 5121 > "$o/$2_$3.log" 2>&1
}
for step in ${@:-sft last refit reference rl}; do
  case $step in
    sft) policy configs/stage3_sft.json "${POOL_SFT:?supervised pool}" "$OUT/sft" ;;
    last) policy configs/stage3_sft_last_phase.json "${POOL_LAST:?last-phase pool}" "$OUT/last" \
            --init-adapter "$OUT/sft/train/checkpoint-20900" ;;
    refit)
      : "${REFIT_ROWS:?refit rows}" "${REFIT_DATA:?refit data}" "${HELDOUT:?held-out tokens}" "${CANDS:?5,121 library}"
      R=$OUT/refit; AD=$OUT/last/train/final; mkdir -p "$R"
      for g in 0 1 2 3 4 5 6 7; do
        ( for s in nav pai navtest; do features $g $s $g 8; done
          if [ $g = 0 ]; then features 0 paival 0 1; fi; if [ $g = 1 ]; then features 1 pai700 0 1; fi ) &
      done
      waitall features
      python - "$R/feat" "$REFIT_ROWS" <<'PY'   # every token has features; per set the prompts read with past frames = the index's
import glob, json, re, sys
import numpy as np
F, R = sys.argv[1:3]; res = {}
for s in ("nav", "pai", "navtest", "paival", "pai700"):
    got = {}
    for f in sorted(glob.glob(f"{F}/{s}/code_*.npy")):
        r = f.rsplit("_", 1)[1][:-4]; t = open(f"{F}/{s}/tok_{r}.txt").read().split()
        assert len(t) == len(np.load(f)) == len(np.load(f"{F}/{s}/h_{r}.npy", mmap_mode="r")); got.update(zip(t, np.load(f).tolist()))
    want = open(f"{R}/{s}_tokens.txt").read().split(); miss = [t for t in want if t not in got]
    ip = sum(1 for x in map(json.loads, open(f"{R}/{s}_index.jsonl")) if x["past"])
    w = [re.findall(r"features: (\d+) prompts with past frames, (\d+) without", open(f).read()) for f in glob.glob(f"{F}/{s}/{s}_*.log")]
    assert all(len(x) == 1 for x in w), (s, "exit line missing")
    a, b = sum(int(x[0][0]) for x in w), sum(int(x[0][1]) for x in w)
    res[s] = {"tokens": len(want), "features": len(got), "missing": len(miss), "read_with_past": a, "read_without": b, "index_with_past": ip}
    assert not miss and a == ip and a + b == len(want), res[s]
print(json.dumps(res))
PY
      python -m superglance.refit.fit fit --base "$AD" --data "$REFIT_DATA" --feat "$R/feat" --heldout "$HELDOUT" --out "$R/fit"
      [ -e "$REFIT" ] || FIT=$R/fit/new_rows.npz python -m superglance.refit.extend_head "$AD" "$CANDS" \
        "$REFIT_DATA/data/cands_k5273_v2.npy" "$REFIT" 4.0
      python -m superglance.refit.fit eval --base "$AD" --data "$REFIT_DATA" --feat "$R/feat" --heldout "$HELDOUT" \
        --adapter "$REFIT" --full --out "$R/eval_k5273.json"
      python - "$R/eval_k5273.json" <<'PY'   # the stop rule of the paper's chain
import json, sys
e = json.load(open(sys.argv[1])); n, v = e["navtest"], e["paival"]
dn, dv = n["pdms_kprime"] - n["pdms_5121"], v["top1_kprime"][0] - v["top1_5121"][0]
print(("RULE_OK" if dn >= -0.1 and dv <= 0.005 else "RULE_FAIL"), json.dumps({"navtest_pdms": [n["pdms_kprime"], n["pdms_5121"], round(dn, 4)],
      "paival_top1_ade3": [v["top1_kprime"][0], v["top1_5121"][0], round(dv, 4)]}))
sys.exit(0 if dn >= -0.1 and dv <= 0.005 else 1)
PY
      ;;
    reference)
      mkdir -p "$OUT/reference"
      for r in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$r python -m superglance.train.rl_reference extract "${POOL_RL:?RL pool}" configs/stage3_rl.json \
          "$STAGE2" "$REFIT" "$OUT/reference" $r 8 > "$OUT/reference/extract_$r.log" 2>&1 &
      done
      waitall reference
      python -m superglance.train.rl_reference merge "$POOL_RL" "$OUT/reference" 8 ;;
    rl) policy configs/stage3_rl.json "${POOL_RL:?RL pool}" "$OUT/rl" --init-adapter "$REFIT" ;;
    *) echo "unknown step $step (sft, last, refit, reference, rl)" >&2; exit 2 ;;
  esac
done
