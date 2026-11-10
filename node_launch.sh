#!/usr/bin/env bash
# Launch (or resume: train.py continues from the run's latest.pt) one teacher run in the background, detached from the shell.
#   GPU=0 ./node_launch.sh [rl/configs/rr_T1k_tc_s5.json]      log: rl/runs/<config name>/train.log, progress: progress.jsonl
# Set WARP_CACHE_PATH (and keep it set for every run) to keep the Warp kernel cache off a full home disk.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CFG=$(realpath -e "${1:-$HERE/rl/configs/rr_T1k_tc_s5.json}"); NAME=$(basename "$CFG" .json)
cd "$HERE/rl"; mkdir -p "runs/$NAME"
CUDA_VISIBLE_DEVICES=${GPU:-0} setsid nohup ../.venv-rl-head/bin/python -I train.py "$CFG" > "runs/$NAME/train.log" 2>&1 < /dev/null &
echo "pid $! on GPU ${GPU:-0}: $HERE/rl/runs/$NAME/train.log (checkpoints go where the config's \"out\" points)"
