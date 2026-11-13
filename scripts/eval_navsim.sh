#!/usr/bin/env bash
# NAVSIM navtest (12,146 scenes), open loop: the Policy Head's features with the front-camera history, the top-64 mean readout
# (temperature 0.2) and the NAVSIM v1.1 PDM scorer (PDM simulator: LQR tracking + kinematic bicycle) on every scene's metric cache.
# The "mean" line is the paper's readout; argmax and medoid are printed beside it.
#   BASE=<stage-2 merged model> ADAPTER=<stage-3 adapter> CANDS=<library: 5,273 for RL, 5,121 for the supervised phases>
#   NAVTEST=<navtest scene export (superglance.builders.pools.navsim_export)> HIST=<history_nav output dir>
#   NAVSIM_PY=<python of the NAVSIM environment> [OUT=runs/eval/navtest] [GPUS="0 1 2 3 4 5 6 7"] bash scripts/eval_navsim.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BASE:?stage-2 merged model}" "${ADAPTER:?adapter}" "${CANDS:?candidate library}" "${NAVTEST:?navtest scene export}"
: "${HIST:?history_nav output}" "${NAVSIM_PY:?NAVSIM python}"
OUT=${OUT:-runs/eval/navtest}; GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); mkdir -p "$OUT/frames" "$OUT/feat"
K=$(python -c 'import numpy, sys; print(len(numpy.load(sys.argv[1], mmap_mode="r")))' "$CANDS")
if [ ! -s "$OUT/index.jsonl" ]; then   # past frames of the test split, unpacked once
  for t in "$HIST"/test/*.tar; do tar -xf "$t" -C "$OUT/frames"; done
  python - "$HIST/index.jsonl" "$NAVTEST" "$OUT/frames" > "$OUT/index.part" <<'PY'
import json, os, sys
past = {r["token"]: r.get("past") for r in map(json.loads, open(sys.argv[1])) if r["split"] == "test"}
for r in map(json.loads, open(sys.argv[2])):
    p = past[r["token"]] and [f"{sys.argv[3]}/{q}" for q in past[r["token"]]]
    assert not p or all(map(os.path.exists, p)), r["token"]
    print(json.dumps({"image": r["cam_f0"], "token": r["token"], "past": p}))
PY
  mv "$OUT/index.part" "$OUT/index.jsonl"
fi
python -c 'import json, sys; print("\n".join(json.loads(l)["token"] for l in open(sys.argv[1])))' "$NAVTEST" > "$OUT/tokens.txt"
for i in "${!GPUS[@]}"; do
  [ -s "$OUT/feat/code_$i.npy" ] || CUDA_VISIBLE_DEVICES=${GPUS[$i]} python -m superglance.eval.features --inputs "$NAVTEST" \
    --tokens "$OUT/tokens.txt" --index "$OUT/index.jsonl" --model "$BASE" --adapter "$ADAPTER" --out "$OUT/feat" \
    --rank $i --world ${#GPUS[@]} --k "$K" > "$OUT/feat/features_$i.log" 2>&1 &
done
for p in $(jobs -p); do wait "$p" || { echo "a feature shard failed: see $OUT/feat/features_*.log" >&2; exit 1; }; done
python -m superglance.eval.readout prep navtest "$OUT/feat" "$ADAPTER" "$CANDS" "$OUT/read.npz"
"$NAVSIM_PY" -m superglance.eval.readout score "$OUT/read.npz" "$NAVTEST" "$OUT/scores.npz" | tee "$OUT/scores.txt"
