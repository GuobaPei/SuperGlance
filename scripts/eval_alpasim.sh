#!/usr/bin/env bash
# AlpaSim closed loop on the 916 scenarios of PAI-AV-NuRec 26.01 (AlpaSim's public_2601 suite), set up as the paper's runs:
# four drivers (superglance.alpasim.driver, one GPU each, ports BP+10 / +30 / +50 / +70) with --rectify-f0 --cmd-rule ego
# --hist --exec navsim; AlpaSim's wizard with +e2e_challenge=dev, topology 4gpu_ext4, runtime.nr_workers=4 and the planner
# delay; video off; then the summaries: superglance.alpasim.clean, .table, .km, .collisions (at-fault collisions by kind,
# in AlpaSim's environment) and, against a reference run, .pair.
#   BASE=<stage-2 merged base> ADAPTER=<adapter> CANDS=<candidate set .npy> ALPASIM=<AlpaSim checkout>
#   DATA=<AlpaSim data folder with nre-artifacts/> NAVSIM_PYTHON=<python of the NAVSIM environment>
#   NRE_RUN=<runs a command inside AlpaSim's NRE renderer image on one GPU: scripts/alpasim_services.sh>
#   [ALPASIM_RUN="setsid -w bash scripts/alpasim_services.sh"] [NAVSIM_DEVKIT=<navsim checkout>] [DELAY_US=0] [EXEC=navsim]
#   [HIST=1] [OUT=runs/alpasim] [BP=27700] [DRIVER_GPUS="0 1 2 3"] [SIM_GPUS="4 5 6 7"] [REF=<results-summary.clean.json>]
#   bash scripts/eval_alpasim.sh
# DELAY_US is AlpaSim's runtime.simulation_config.planner_delay_us (0: none). EXEC=plan hands AlpaSim the plan's 8 poses
# instead of the PDM simulator's 40; HIST=0 for a model trained without past frames.
# ALPASIM_RUN is called with the run dir as its argument and GPUS (= SIM_GPUS), NRE_D (= DATA/nre-artifacts) and A (= ALPASIM)
# in the environment; it starts the generated config's services (renderer, physics, controller, runtime) and returns when the
# rollouts are done. The default is the paper's runner, scripts/alpasim_services.sh.
# All 916 scenarios run as one run. The paper's runs split them into four quarters and merged the summaries; with a fixed
# configuration AlpaSim gives the same metrics per scenario either way.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BASE:?stage-2 merged base}" "${ADAPTER:?adapter}" "${CANDS:?candidate set}" "${ALPASIM:?AlpaSim checkout}"
: "${DATA:?AlpaSim data folder}" "${NAVSIM_PYTHON:?NAVSIM python}"
ALPASIM_RUN=${ALPASIM_RUN:-setsid -w bash scripts/alpasim_services.sh}
OUT=${OUT:-runs/alpasim}; BP=${BP:-27700}; DELAY_US=${DELAY_US:-0}; EXEC=${EXEC:-navsim}; HIST=${HIST:-1}
DG=(${DRIVER_GPUS:-0 1 2 3}); SG=${SIM_GPUS:-4 5 6 7}; PY=${PY:-python}; APY=${ALPASIM_PY:-$ALPASIM/.venv/bin/python}
FR=${FRAMES:-/dev/shm/superglance_driver}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd); R=$OUT/run
[ ! -e "$R" ] || { echo "$R exists" >&2; exit 1; }
[ ${#DG[@]} = 4 ] || { echo "DRIVER_GPUS: four GPUs, one driver each" >&2; exit 1; }
ids=$(grep '^public_2601,' "$ALPASIM/data/scenes/sim_suites.csv" | cut -d, -f2 | LC_ALL=C sort | paste -sd, -)
[ "$(echo "$ids" | tr , '\n' | grep -c .)" = 916 ] || { echo "public_2601 does not list 916 scenarios" >&2; exit 1; }

flags=(--rectify-f0 --cmd-rule ego --exec "$EXEC" --navsim-python "$NAVSIM_PYTHON")
[ "$HIST" = 1 ] && flags+=(--hist)
[ -n "${NAVSIM_DEVKIT:-}" ] && flags+=(--navsim-repo "$NAVSIM_DEVKIT")
pids=(); drivers=
for k in 0 1 2 3; do
  p=$((BP + 10 + 20 * k))
  CUDA_VISIBLE_DEVICES=${DG[k]} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 "$PY" -m superglance.alpasim.driver \
    --port $p --model "$BASE" --adapter "$ADAPTER" --cands "$CANDS" --frames "${FR}_$p" "${flags[@]}" > "$OUT/driver_$p.log" 2>&1 < /dev/null &
  pids+=($!); drivers=${drivers:+$drivers,}localhost:$p
done
trap 'kill ${pids[*]} 2> /dev/null || true' EXIT
for k in 0 1 2 3; do
  p=$((BP + 10 + 20 * k))
  until "$PY" -c "import socket, sys; sys.exit(socket.socket().connect_ex(('127.0.0.1', $p)))"; do
    kill -0 ${pids[k]} 2> /dev/null || { echo "the driver on port $p exited: $OUT/driver_$p.log" >&2; exit 1; }
    sleep 5
  done
done
echo "drivers up: $drivers"

xw=(runtime.nr_workers=4)
[ "$DELAY_US" != 0 ] && xw+=(runtime.simulation_config.planner_delay_us=$DELAY_US)
if [ "$(nvidia-smi -L | wc -l)" -lt 8 ]; then   # the wizard's GPU lists, as the paper's runs set them on machines with fewer GPUs
  xw+=("services.driver.gpus=[$(IFS=,; echo "${DG[*]}")]" "services.renderer.gpus=[${SG// /,}]" "services.physics.gpus=[${SG// /,}]")
fi
( cd "$ALPASIM" && "$APY" -m alpasim_wizard +e2e_challenge=dev +nurec_scenes=curated_val scenes.test_suite_id=null "scenes.scene_ids=[$ids]" \
    "${xw[@]}" topology=4gpu_ext4 "wizard.external_services.driver=[$drivers]" wizard.run_method=NONE "wizard.log_dir=$R" \
    "defines.filesystem=$DATA" wizard.baseport=$BP ) > "$OUT/wizard.log" 2>&1
"$APY" - "$R/eval-config.yaml" <<'PY'
import sys, yaml
f = sys.argv[1]; c = yaml.safe_load(open(f)); c["video"]["render_video"] = False
yaml.safe_dump(c, open(f, "w"), sort_keys=False)
PY
echo "run: $(grep -c 'scene_id:' "$R/generated-user-config-0.yaml") scenarios, planner_delay_us $DELAY_US, $R"
GPUS="$SG" NRE_D="$DATA/nre-artifacts" A="$ALPASIM" $ALPASIM_RUN "$R" > "$OUT/run.log" 2>&1
for k in 0 1 2 3; do p=$((BP + 10 + 20 * k)); cp "${FR}_$p/decisions.jsonl" "$OUT/decisions_$p.jsonl"; done

"$PY" -m superglance.alpasim.clean "$R"
S=$R/aggregate/results-summary.clean.json
"$PY" -m superglance.alpasim.table "$S"
"$PY" -m superglance.alpasim.km "$S" ${REF:-}
"$APY" -m superglance.alpasim.collisions "$S" "$R/rollouts/{c}/*/rollout.asl" "$OUT/collisions.jsonl"
[ -z "${REF:-}" ] || "$PY" -m superglance.alpasim.pair "$REF" "$S"
