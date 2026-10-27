#!/usr/bin/env bash
# Re-score the screened REPEAT and BRANCH spans on math_120 and eq_140.
#
# Usage:
#   scripts/run_scan_span_math120_eq140.sh <config.json> <output-dir> [num-workers-per-gpu] [model-path] [rep-spans-csv] [br-spans-csv]
#
# Example:
#   scripts/run_scan_span_math120_eq140.sh config/runtime.json artifacts/scan_span_math120_eq140

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [[ -n "${SELFMORPH_PYTHON:-}" ]]; then
  PYTHON="$SELFMORPH_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="python"
fi

LOCK_FILE="/tmp/selfmorph_scan_span_math120_eq140.lock"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
  echo "[ABORT] Another scripts/run_scan_span_math120_eq140.sh orchestrator holds the lock ($LOCK_FILE). Refusing to start a duplicate." >&2
  exit 1
fi

CONFIG="${1:?config path required}"
OUTPUT_DIR="${2:?output dir required}"
WORKERS_PER_GPU="${3:-1}"
MODEL_PATH="${4:-}"
REP_SPANS_CSV="${5:-}"
BR_SPANS_CSV="${6:-}"
NUM_GPUS="${7:-1}"
NUM_SHARDS=$((NUM_GPUS * WORKERS_PER_GPU))

mkdir -p "$OUTPUT_DIR"
echo "Starting $NUM_SHARDS shards across $NUM_GPUS GPUs ($WORKERS_PER_GPU workers/GPU)..."

EXTRA_ARGS=()
if [ -n "$MODEL_PATH" ]; then
  EXTRA_ARGS+=(--model-path "$MODEL_PATH")
  echo "Scanning base model: $MODEL_PATH"
fi
if [ -n "$REP_SPANS_CSV" ]; then
  EXTRA_ARGS+=(--rep-spans-csv "$REP_SPANS_CSV" --rep-top-k 30)
  echo "REPEAT candidates: top-30 from $REP_SPANS_CSV"
fi
if [ -n "$BR_SPANS_CSV" ]; then
  EXTRA_ARGS+=(--br-spans-csv "$BR_SPANS_CSV" --br-top-k 30)
  echo "BRANCH candidates: top-30 from $BR_SPANS_CSV"
fi

PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  GPU=$((i % NUM_GPUS))
  nohup "$PYTHON" "$(dirname "$0")/scan_span_math120_eq140.py" \
    --config "$CONFIG" --output-dir "$OUTPUT_DIR" \
    --shard-id "$i" --num-shards "$NUM_SHARDS" --device "cuda:${GPU}" \
    "${EXTRA_ARGS[@]}" \
    > "${OUTPUT_DIR}/shard${i}of${NUM_SHARDS}.log" 2>&1 &
  PIDS+=($!)
done

sleep 5
echo "Launched PIDs: ${PIDS[*]}"
echo "Verify with: ps -eo pid,ppid,stat,etime,cmd | grep scan_span_math120_eq140.py | grep -v grep"
