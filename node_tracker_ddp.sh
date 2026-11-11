#!/usr/bin/env bash
# (11-10) Data-parallel tracker / teacher training (tracker/train_tracker.py) on several GPUs of one node: one process per GPU, each seeing only
# its own GPU; rsl_rl averages gradients, KL and the observation normaliser over the ranks (NCCL); rank 0 logs and checkpoints.
#   GPUS="0 3 4" ./node_tracker_ddp.sh tracker/configs/kin_d02_s5_g3.json        logs: <config out>/train_rank<i>.log
# Set WARP_CACHE_PATH for every run. env_steps / max_env_steps in the config count all GPUs (one iteration = steps_per_env x num_envs x GPUs).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); CFG=$(realpath -e "$1"); read -r -a G <<< "${GPUS:-0}"; n=${#G[@]}
OUT=$(python3 -c "import json, sys, pathlib; c = json.load(open(sys.argv[1])); print((pathlib.Path(sys.argv[1]).parent/c['out']).resolve())" "$CFG")
mkdir -p "$OUT"; cd "$HERE/tracker"; PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES=${G[$i]} RANK=$i WORLD_SIZE=$n MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT \
    setsid nohup ../.venv-rl-head/bin/python -I train_tracker.py "$CFG" > "$OUT/train_rank$i.log" 2>&1 < /dev/null &
  echo "rank $i on GPU ${G[$i]}: pid $!"
done
