#!/usr/bin/env bash
# Screen every span on math_16 and eq_16.
# Resume a run with the same shard count; the assignment depends on it.
#
# Usage:
#   scripts/run_scan_span_math_eq.sh <config.json> <output-dir> [num-workers-per-gpu] [max-span-width] [model-path] [num-gpus]
#
# Example:
#   scripts/run_scan_span_math_eq.sh config/runtime.json artifacts/scan_span_math_eq
#   scripts/run_scan_span_math_eq.sh config/runtime.json artifacts/scan_span_math_eq "" 10

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [[ -n "${SELFMORPH_PYTHON:-}" ]]; then
  PYTHON="$SELFMORPH_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="python"
fi

# Atomic mutex: guards against two orchestrator invocations racing each
# other (e.g. a retried launch command) -- a plain `pgrep` pre-check has
# a TOCTOU race where both invocations pass the check before either has
# forked any shard. flock is atomic: the loser exits before forking.
LOCK_FILE="/tmp/selfmorph_scan_span_math_eq.lock"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
  echo "[ABORT] Another scripts/run_scan_span_math_eq.sh orchestrator holds the lock ($LOCK_FILE). Refusing to start a duplicate." >&2
  exit 1
fi

CONFIG="${1:?config path required}"
OUTPUT_DIR="${2:?output dir required}"
WORKERS_PER_GPU="${3:-1}"
MAX_SPAN_WIDTH="${4:-}"
MODEL_PATH="${5:-}"
NUM_GPUS="${6:-1}"
NUM_SHARDS=$((NUM_GPUS * WORKERS_PER_GPU))

mkdir -p "$OUTPUT_DIR"
echo "Starting $NUM_SHARDS shards across $NUM_GPUS GPUs ($WORKERS_PER_GPU workers/GPU)..."

EXTRA_ARGS=()
if [ -n "$MAX_SPAN_WIDTH" ]; then
  EXTRA_ARGS+=(--max-span-width "$MAX_SPAN_WIDTH")
  echo "Capping span width (j-i) at $MAX_SPAN_WIDTH for this run."
fi
if [ -n "$MODEL_PATH" ]; then
  EXTRA_ARGS+=(--model-path "$MODEL_PATH")
  echo "Scanning base model: $MODEL_PATH"
fi

PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  GPU=$((i % NUM_GPUS))
  nohup "$PYTHON" "$(dirname "$0")/scan_span_math_eq.py" \
    --config "$CONFIG" --output-dir "$OUTPUT_DIR" \
    --shard-id "$i" --num-shards "$NUM_SHARDS" --device "cuda:${GPU}" \
    "${EXTRA_ARGS[@]}" \
    > "${OUTPUT_DIR}/shard${i}of${NUM_SHARDS}.log" 2>&1 &
  PIDS+=($!)
done

sleep 5
echo "Launched PIDs: ${PIDS[*]}"
echo "Verify with: ps -eo pid,ppid,stat,etime,cmd | grep scan_span_math_eq.py | grep -v grep"
