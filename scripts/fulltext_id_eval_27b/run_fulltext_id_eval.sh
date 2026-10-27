#!/usr/bin/env bash
# Completed-response evaluation. GPU count and workers per GPU are optional arguments.
#
# Usage:
#   scripts/fulltext_id_eval_27b/run_fulltext_id_eval.sh <config> <model-path> \
#     <selector-checkpoint> <train-dev.pt> <out-dir> [num-gpus] [workers-per-gpu] [max-new]

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
if [[ -n "${SELFMORPH_PYTHON:-}" ]]; then
  PYTHON="$SELFMORPH_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="python"
fi

CONFIG="${1:?config required}"
MODEL_PATH="${2:?model path required}"
SELECTOR_CKPT="${3:?selector checkpoint required}"
TRAIN_DEV_PT="${4:?train_dev.pt required}"
OUT_DIR="${5:?out dir required}"
NUM_GPUS="${6:-1}"
WORKERS_PER_GPU="${7:-1}"
MAX_NEW="${8:-1024}"
NUM_SHARDS=$((NUM_GPUS * WORKERS_PER_GPU))

DIR="$(dirname "$0")"
if pgrep -f "fulltext_id_eval_27b/run_fulltext_id_eval.py" > /dev/null; then
  echo "[ABORT] run_fulltext_id_eval.py already running" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "Starting $NUM_SHARDS shards ($NUM_GPUS GPUs x $WORKERS_PER_GPU workers), max_new=$MAX_NEW"
PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  GPU=$((i % NUM_GPUS))
  nohup "$PYTHON" "$DIR/run_fulltext_id_eval.py" \
    --config "$CONFIG" --model-path "$MODEL_PATH" \
    --selector-checkpoint "$SELECTOR_CKPT" --train-dev-pt "$TRAIN_DEV_PT" \
    --out-dir "$OUT_DIR" \
    --shard-id "$i" --num-shards "$NUM_SHARDS" --device "cuda:${GPU}" --max-new "$MAX_NEW" \
    > "${OUT_DIR}/shard${i}of${NUM_SHARDS}.log" 2>&1 &
  PIDS+=($!)
done
sleep 5
echo "Launched PIDs: ${PIDS[*]}"
wait "${PIDS[@]}"
echo "All $NUM_SHARDS shards finished."
