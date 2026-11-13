#!/usr/bin/env bash
# AlpaSim's services for one run dir made by its wizard (scripts/eval_alpasim.sh: +e2e_challenge=dev, topology=4gpu_ext4,
# wizard.run_method=NONE, our four drivers in wizard.external_services.driver), as the paper's runs started them: every renderer
# and physics endpoint of its generated-network-config.yaml on the GPUs of $GPUS (endpoint i on GPU i of the list), one
# controller, then one runtime (one aggregate). Renderer: NRE 26.04 with the preset's command minus --enable-harmonizer (the
# challenge's preset evaluates the unharmonized renderer); only the front wide camera is rendered (the one our driver reads;
# same spec as in 6cam_1080); CAMS (space-separated logical ids) keeps other cameras of the 6-camera set. HOOD=<rig id in
# ego-hoods> masks with another rig's hood (default: the generated config's).
#   A=<AlpaSim checkout with its .venv> NRE_D=<the data folder's nre-artifacts/> GPUS="4 5 6 7"
#   NRE_RUN=<command that runs its arguments inside AlpaSim's NRE renderer image on GPU <index>, NRE_DATA bound at the image's
#            data mount point; called as: $NRE_RUN <gpu index> <command ...>, with NRE_DATA and NRE_COMPAT=1 in its environment>
#   [WARP_CACHE=<physics kernel cache dir>] [VAU=<prefix for every Python process>] setsid bash scripts/alpasim_services.sh <run dir>
# The generated config's data_dir lies under the renderer image's data mount point; the host side reaches the data through a
# link at that mount point, so the physics servers read the same paths. The paper's machines ran the renderer image without
# docker (a user namespace and chroot); that runner is not part of this release. Started with setsid, the script stops this
# run's services when the runtime exits.
set -uo pipefail
exec 9>&-
R=${1:?run dir}; A=${A:?AlpaSim checkout}; D=${NRE_D:?the data folder's nre-artifacts}; G=(${GPUS:-0 2 4 6})
: "${NRE_RUN:?command that runs the renderer image on one GPU}"; WC=${WARP_CACHE:-$HOME/.cache/warp}
log() { echo "[pai4 $(date -u +%FT%TZ)] $*"; }
ports() { python3 -c "import yaml; c = yaml.safe_load(open('$R/generated-network-config.yaml')); print(' '.join(e['address'].rsplit(':', 1)[1] for e in c['$1']['endpoints']))"; }
up() { python3 -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(s.connect_ex(('127.0.0.1', $1)))"; }
RP=($(ports renderer)); PP=($(ports physics)); CP=$(ports controller); DP=($(ports driver))
SC=$(grep -o 'scenesets/[0-9a-f]*' $R/generated-user-config-0.yaml | head -1)
[ ${#RP[@]} -ge 1 ] && [ ${#RP[@]} -le ${#G[@]} ] && [ ${#PP[@]} -le ${#G[@]} ] && [ -n "$CP" ] && [ -n "$SC" ] || { log "STOP: ports / GPUs / sceneset"; exit 1; }
M=$(grep -o "data_dir: [^ ]*/$SC" $R/generated-user-config-0.yaml | head -1); M=${M#data_dir: }; M=${M%/$SC}
[ -n "$M" ] || { log "STOP: no data_dir in the generated config"; exit 1; }
[ "$(readlink $M)" = "$D" ] || ln -sfn $D $M   # M: the renderer image's data mount point, the prefix of the config's data_dir
${VAU:-} python3 - $R/generated-user-config-0.yaml <<'PY'
import os, sys, yaml
f = sys.argv[1]; c = yaml.safe_load(open(f)); cams = c["simulation_config"]["cameras"]
want = os.environ.get("CAMS", "camera_front_wide_120fov").split()
keep = [x for x in cams if x["logical_id"] in want]
assert len(keep) == len(want), cams
if len(cams) > 1:
    open(f + ".6cam", "w").write(open(f).read())
    c["simulation_config"]["cameras"] = keep
    yaml.safe_dump(c, open(f, "w"), sort_keys=False)
print("cameras:", keep, "; scenes:", len(c["scenes"]))
PY
# HOOD=<rig id in ego-hoods> masks with another rig's hood; default: the generated config's (hyperion_8_1)
[ -n "${HOOD:-}" ] && ${VAU:-} python3 - $R/generated-user-config-0.yaml "$HOOD" <<'PY'
import sys, yaml
f, h = sys.argv[1], sys.argv[2]; c = yaml.safe_load(open(f)); sc = c["simulation_config"]
assert "ego_mask_rig_config_id" in sc, sorted(sc)
sc["ego_mask_rig_config_id"] = h; yaml.safe_dump(c, open(f, "w"), sort_keys=False); print("hood:", h)
PY
mkdir -p $R/txt-logs $R/controller $WC
start_r() { local i=$1 p=${RP[$1]}; NRE_DATA=$D NRE_COMPAT=1 nohup $NRE_RUN ${G[$i]} env HOME=/tmp XDG_CACHE_HOME=/tmp/.cache \
    OMP_NUM_THREADS=1 /app/internal/scripts/pycena/runtime/pycena_nrm_full serve-grpc --port=$p --host=0.0.0.0 "--artifact-glob=$M/$SC/**/*.usdz" \
    --egocar-hood-dir=$M/ego-hoods --no-enable-nrend --download-cache-dir /tmp/nre-cache-dir --cache-size=3 --max-workers=4 \
    --enable-editing-actors > $R/txt-logs/renderer_$i.log 2>&1 < /dev/null 9>&- & echo $! > $R/txt-logs/renderer_$i.pid; }
for i in "${!RP[@]}"; do up ${RP[$i]} || start_r $i; done
retry_r() {   # a renderer that died at start-up gets up to two restarts, a physics server one
  local i n; for i in "${!RP[@]}"; do up ${RP[$i]} && continue; n=$(ls $R/txt-logs/renderer_${i}_failed*.log 2> /dev/null | wc -l); [ $n -ge 2 ] && continue
    kill -0 $(cat $R/txt-logs/renderer_$i.pid 2>/dev/null) 2>/dev/null && continue   # only a renderer whose process is gone
    grep -qE "readobject called with exception|Unknown compute capability|Traceback" $R/txt-logs/renderer_$i.log 2>/dev/null || continue
    mv $R/txt-logs/renderer_$i.log $R/txt-logs/renderer_${i}_failed$((n + 1)).log; start_r $i
    log "renderer $i (port ${RP[$i]}) died at start-up: restart $((n + 1)) of 2"; done
  for i in "${!PP[@]}"; do up ${PP[$i]} && continue; [ -e $R/txt-logs/physics_$i.retried ] && continue   # physics servers too
    kill -0 $(cat $R/txt-logs/physics_$i.pid 2>/dev/null) 2>/dev/null && continue   # only a server whose process is gone
    grep -qE "Traceback|RuntimeError" $R/txt-logs/physics_$i.log 2>/dev/null || continue
    touch $R/txt-logs/physics_$i.retried; mv $R/txt-logs/physics_$i.log $R/txt-logs/physics_${i}_failed1.log; start_p $i
    log "physics $i (port ${PP[$i]}) died at start-up: restarted once"; done; }
cd $A
start_p() { local i=$1 p=${PP[$1]}; CUDA_VISIBLE_DEVICES=${G[$i]} WARP_CACHE_PATH=$WC nohup ${VAU:-} .venv/bin/physics_server --host=0.0.0.0 --port=$p \
    "--artifact-glob=$M/$SC/**/*.usdz" --use-ground-mesh=true --cache-size=4 --log-level=INFO > $R/txt-logs/physics_$i.log 2>&1 < /dev/null 9>&- & echo $! > $R/txt-logs/physics_$i.pid; }
for i in "${!PP[@]}"; do up ${PP[$i]} || start_p $i; done
up $CP || { nohup ${VAU:-} .venv/bin/python -m alpasim_controller.server --port=$CP --log_dir=$R/controller --log-level=INFO \
  --config=$R/controller-config.yaml > $R/txt-logs/controller.log 2>&1 < /dev/null 9>&- & }
ALL=("${RP[@]}" "${PP[@]}" $CP "${DP[@]}")
for t in $(seq 1 240); do ok=1; for p in "${ALL[@]}"; do up $p || { ok=0; break; }; done; [ $ok = 1 ] && break; [ $((t % 12)) = 0 ] && retry_r; sleep 5; done
for p in "${ALL[@]}"; do up $p || { log "STOP: nothing on port $p"; exit 1; }; done
log "services up (renderers ${RP[*]}, physics ${PP[*]}, controller $CP, drivers ${DP[*]}); runtime starts"
cd $R && ${VAU:-} $A/.venv/bin/python -m alpasim_runtime.simulate --user-config=$R/generated-user-config-0.yaml --network-config=$R/generated-network-config.yaml \
  --log-dir=$R --log-level=INFO --array-job-dir=$R --eval-config=$R/eval-config.yaml > $R/txt-logs/runtime.log 2>&1
log "runtime exit $?"
# stop this run's services: every process of this script's session but the script (started with setsid; the renderer's
# workers do not carry its command line). The drivers have their own sessions.
S=$(ps -o sid= -p $$ | tr -d ' ')
[ "$S" = "$$" ] || { log "not started with setsid: services left running"; exit 0; }
ps -eo pid=,sid= | awk -v s=$S -v me=$$ '$2 == s && $1 != me { print $1 }' > $R/txt-logs/stop_pids.txt
xargs -r kill < $R/txt-logs/stop_pids.txt
log "stopped $(wc -l < $R/txt-logs/stop_pids.txt) service processes"
