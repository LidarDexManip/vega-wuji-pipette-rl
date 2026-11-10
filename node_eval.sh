#!/usr/bin/env bash
# Evaluate one finished run on the node, one GPU per job, all in parallel (11-09):
#   eval_rr.py from start (64 worlds) / training starts (--repeat 4) / held-out hole pairs (x4) / held-out placements (x4),
#   eval_contact_stability.py (64 from start), reward_breakdown.py (512 x 300 training-like steps).
#   GPUS="0 3 4 5 6 7" ./node_eval.sh RUN_NAME TRAIN_CONFIG HOLDOUT_CONFIG HOLDPLACE_CONFIG [CKPT]
# CKPT defaults to rl/runs/RUN_NAME/final.pt; results: rl/runs/local_eval/RUN_NAME/{fromstart,train_starts,holdout,holdplace,contact,rb}.json
# (+ .log, rc lists the exit codes, ALLDONE at the end). Configs are paths relative to rl/ (eval only reads the env settings from them).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE/rl"
RUN=$1; CFG=$2; HO=$3; HP=$4; CK=${5:-runs/$RUN/final.pt}
read -r -a G <<< "${GPUS:-0 3 4 5 6 7}"; [ ${#G[@]} -ge 6 ] || { echo "need 6 GPUs in GPUS"; exit 1; }
O=runs/local_eval/$RUN; mkdir -p "$O"; rm -f "$O/rc"; P=../.venv-rl-head/bin/python
job() { local g=$1 name=$2; shift 2; ( CUDA_VISIBLE_DEVICES=$g "$P" -I "$@" > "$O/$name.log" 2>&1; echo "$name rc $?" >> "$O/rc" ) & }
job "${G[0]}" fromstart    eval_rr.py "$CFG" --ckpt "$CK" --from-start 64 --out "$O/fromstart.json"
job "${G[1]}" train_starts eval_rr.py "$CFG" --ckpt "$CK" --repeat 4 --out "$O/train_starts.json"
job "${G[2]}" holdout      eval_rr.py "$HO" --ckpt "$CK" --repeat 4 --out "$O/holdout.json"
job "${G[3]}" holdplace    eval_rr.py "$HP" --ckpt "$CK" --repeat 4 --out "$O/holdplace.json"
job "${G[4]}" contact      eval_contact_stability.py "$CFG" --ckpt "$CK" --out "$O/contact.json"
job "${G[5]}" rb           reward_breakdown.py "$CFG" --ckpt "$CK" --out "$O/rb.json"
wait; echo ALLDONE >> "$O/rc"; cat "$O/rc"
