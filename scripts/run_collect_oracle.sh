#!/usr/bin/env bash
# Multi-GPU orchestrator for scripts/collect_oracle.py.
# Shard assignment depends on num_shards, so resume a run with the same shard count.
#
# Usage:
#   scripts/run_collect_oracle.sh <config.json> <datasets-comma-list> <output-dir> [num-workers-per-gpu] [collect-mode] [context-ids-file] [num-gpus]
#
# collect-mode: full (default) | br_only | rep_only.
# br_only and rep_only skip trajectory capture.
# context-ids-file: optional whitelist (one `dataset::qid` per line);
# only these contexts are collected.
# Pass the whitelist as argument 6, or as --context-ids-file <path>.
#
# Example:
#   scripts/run_collect_oracle.sh config/runtime.json gsm8k artifacts/oracle

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [[ -n "${SELFMORPH_PYTHON:-}" ]]; then
  PYTHON="$SELFMORPH_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="python"
fi

CONFIG="${1:?config path required}"
DATASETS="${2:?comma-separated dataset list required}"
OUTPUT_DIR="${3:?output dir required}"
WORKERS_PER_GPU="${4:-1}"
COLLECT_MODE="${5:-full}"
CONTEXT_IDS_FILE="${6:-}"
NUM_GPUS_RAW="${7:-}"

# A trailing --context-ids-file flag occupies the slot that would otherwise be num-gpus.
HAS_CONTEXT_FLAG=0
for arg in "${@:6}"; do
  if [ "$arg" = "--context-ids-file" ]; then
    HAS_CONTEXT_FLAG=1
  fi
done
if [ "$HAS_CONTEXT_FLAG" = "1" ]; then
  prev=""
  for arg in "${@:6}"; do
    if [ "$prev" = "--context-ids-file" ]; then
      CONTEXT_IDS_FILE="$arg"
    fi
    prev="$arg"
  done
  NUM_GPUS=1
else
  NUM_GPUS="${NUM_GPUS_RAW:-1}"
fi
NUM_SHARDS=$((NUM_GPUS * WORKERS_PER_GPU))

if pgrep -f "scripts/collect_oracle.py" > /dev/null; then
  echo "[ABORT] scripts/collect_oracle.py is already running. Inspect with:" >&2
  pgrep -af "scripts/collect_oracle.py" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "Starting $NUM_SHARDS shards across $NUM_GPUS GPUs ($WORKERS_PER_GPU workers/GPU), collect-mode=$COLLECT_MODE${CONTEXT_IDS_FILE:+, context-ids-file=$CONTEXT_IDS_FILE}..."

EXTRA_ARGS=()
if [[ -n "$CONTEXT_IDS_FILE" ]]; then
  EXTRA_ARGS+=(--context-ids-file "$CONTEXT_IDS_FILE")
fi

PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  GPU=$((i % NUM_GPUS))
  nohup "$PYTHON" "$(dirname "$0")/collect_oracle.py" \
    --config "$CONFIG" --datasets "$DATASETS" --output-dir "$OUTPUT_DIR" \
    --shard-id "$i" --num-shards "$NUM_SHARDS" --device "cuda:${GPU}" \
    --collect-mode "$COLLECT_MODE" "${EXTRA_ARGS[@]}" \
    > "${OUTPUT_DIR}/shard${i}of${NUM_SHARDS}.log" 2>&1 &
  PIDS+=($!)
done

sleep 5
echo "Launched PIDs: ${PIDS[*]}"
echo "Verify with: ps -eo pid,ppid,stat,etime,cmd | grep collect_oracle.py | grep -v grep"

wait "${PIDS[@]}"
echo "All $NUM_SHARDS shards finished."
