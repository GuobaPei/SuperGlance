#!/bin/bash
# Qwen-Drive-1.0-4B on our PAI-AV rows (Table 6, its two rows). Steps, each skipped when its output exists:
#   1. the scene file (superglance.eval.qd_pai_scenes) from the rows of scripts/eval_pai.sh;
#   2. Qwen-Drive-1.0's scripts/run_planning.py per planner:mode, sharded over the given GPUs (6 samples, sdpa);
#   3. superglance.eval.qd_pai_score on each: Table 6 = its "GT ours all" line, one plan = sample 0, best of six = minADE.
# run_planning.py resumes a partial shard by itself. Paths are relative to the repository root.
#   QD_PY=<python of the Qwen-Drive-1.0 environment, with physical_ai_av> QD_REPO=<Qwen-Drive-1.0 checkout>
#   QD_MODEL=<Qwen-Drive-1.0-4B, with planner-rl/ and planner-sft/> PAI_ROOT=<PhysicalAI-AV root of the rows' clips, cross
#   cameras included> bash scripts/eval_pai_qwen_drive.sh <rows.jsonl> <out dir> "<gpus>" [planner:mode,...]
#   default runs: sft:reasoning_planning,rl:reasoning_planning (Table 6's SFT and RL rows); rl:direct_planning also runs
set -uo pipefail
ROWS=$1; O=$2; G=($3); RUNS=${4:-sft:reasoning_planning,rl:reasoning_planning}
P=${QD_PY:?python of the Qwen-Drive-1.0 environment}; Q=${QD_REPO:?Qwen-Drive-1.0 checkout}; M=${QD_MODEL:?Qwen-Drive-1.0-4B}
ROOT=${PAI_ROOT:?PhysicalAI-AV root}
cd "$(dirname "$0")/.." || exit 1
L=$O/run.log; mkdir -p $O
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
[ -s $O/scenes.jsonl ] || OMP_NUM_THREADS=2 $P -m superglance.eval.qd_pai_scenes $ROWS $ROOT $O 16 >> $L 2>&1 || { log "STOP: scenes"; exit 1; }
log "scenes: $(wc -l < $O/scenes.jsonl)"
for r in ${RUNS//,/ }; do
  pl=${r%%:*}; mode=${r#*:}; n=pred_${pl}_${mode%%_*}
  if [ ! -s $O/$n.jsonl ]; then
    for i in "${!G[@]}"; do
      CUDA_VISIBLE_DEVICES=${G[$i]} HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 $P $Q/scripts/run_planning.py --model $M --planner $M/planner-$pl \
        --scenes $O/scenes.jsonl --image-root $O/frames --output $O/$n.s$i.jsonl --mode $mode --num-samples 6 \
        --attn-implementation sdpa --num-workers 4 --num-shards ${#G[@]} --shard-index $i > $O/$n.s$i.log 2>&1 &
    done
    wait
    cat $O/$n.s*.jsonl > $O/$n.jsonl.tmp
    [ "$(wc -l < $O/$n.jsonl.tmp)" = "$(wc -l < $O/scenes.jsonl)" ] && mv $O/$n.jsonl.tmp $O/$n.jsonl \
      || { log "STOP: $n has $(wc -l < $O/$n.jsonl.tmp) of $(wc -l < $O/scenes.jsonl) predictions"; continue; }
  fi
  log "$n: $(grep -h "^wrote" $O/$n.s*.log | tr '\n' ' ' | cut -c1-240)"
  $P -m superglance.eval.qd_pai_score $O/$n.jsonl $ROWS > $O/$n.score.txt 2>&1; cat $O/$n.score.txt >> $L
done
log END
