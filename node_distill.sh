#!/usr/bin/env bash
# (11-10) Data-parallel DAgger distillation (rl/distill.py) on several GPUs of one node: one process per GPU, each seeing only its own GPU;
# gradients / normaliser statistics averaged over the ranks (NCCL; DISTILL_BACKEND=gloo to fall back), rank 0 logs and checkpoints.
#   GPUS="5 6 7" ./node_distill.sh rl/configs/student_kin_tc_s5_depth.json        logs: <config out>/train_rank<i>.log
# Set WARP_CACHE_PATH for every run (Warp kernel cache off a full home disk). Iteration counts in the config (max_iters, beta_iters) count
# data-parallel iterations: one iteration = steps_per_iter x num_envs x the number of GPUs env steps.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); CFG=$(realpath -e "$1"); read -r -a G <<< "${GPUS:-0}"; n=${#G[@]}
OUT=$(python3 -c "import json, sys, pathlib; c = json.load(open(sys.argv[1])); print((pathlib.Path(sys.argv[1]).parent/c['out']).resolve())" "$CFG")
mkdir -p "$OUT"; cd "$HERE/rl"; PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES=${G[$i]} RANK=$i WORLD_SIZE=$n MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT \
    setsid nohup ../.venv-rl-head/bin/python -I distill.py "$CFG" > "$OUT/train_rank$i.log" 2>&1 < /dev/null &
  echo "rank $i on GPU ${G[$i]}: pid $!"
done
