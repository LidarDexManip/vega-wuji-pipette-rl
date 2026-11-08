#!/usr/bin/env bash
# Run a command in its own process group; kill the group if available memory drops below
# GUARD_MIN_MB (default 4000). The LIDAR scene compiles at ~2.5 GB per process, and an
# OOM on this 30 GB machine takes the desktop session down with it.
set -uo pipefail
min=${GUARD_MIN_MB:-4000}
setsid "$@" &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if (( avail < min )); then
    echo "guarded.sh: MemAvailable ${avail} MB < ${min} MB; killing process group ${pid}" >&2
    kill -TERM -- "-$pid"; sleep 5; kill -KILL -- "-$pid" 2>/dev/null
    wait "$pid"; exit 137
  fi
  sleep 2
done
wait "$pid"
