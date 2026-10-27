#!/usr/bin/env bash
# Multi-GPU orchestrator for unseen-benchmark evaluation.
#
# Usage:
#   scripts/run_baseline_selfmorph_ood_eval.sh <config.json> <model-path> \
#     <selector-checkpoint.pt> <train-dev.pt> <manifest.json> <output-dir> \
#     [num-gpus] [workers-per-gpu]
#
# Example (Qwen3.5-27B):
#   scripts/run_baseline_selfmorph_ood_eval.sh config/runtime.json \
#     /path/to/Qwen3.5-27B \
#     artifacts/checkpoints/qwen35_27b/best_checkpoint.pt \
#     artifacts/training_data/qwen35_27b/train_dev.pt \
#     resources/split_manifests/ood_manifest.json \
#     artifacts/ood_eval/qwen35_27b

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
MODEL_PATH="${2:?model path required}"
SELECTOR_CKPT="${3:?selector checkpoint path required}"
TRAIN_DEV_PT="${4:?train_dev.pt path required}"
MANIFEST="${5:?ood sample manifest path required}"
OUTPUT_DIR="${6:?output dir required}"
NUM_GPUS="${7:-1}"
WORKERS_PER_GPU="${8:-1}"
NUM_SHARDS=$((NUM_GPUS * WORKERS_PER_GPU))

if pgrep -f "scripts/run_baseline_selfmorph_ood_eval.py" > /dev/null; then
  echo "[ABORT] scripts/run_baseline_selfmorph_ood_eval.py is already running. Inspect with:" >&2
  pgrep -af "scripts/run_baseline_selfmorph_ood_eval.py" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "Starting $NUM_SHARDS shards across $NUM_GPUS GPUs ($WORKERS_PER_GPU workers/GPU), model=$MODEL_PATH ..."

PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  GPU=$((i % NUM_GPUS))
  nohup "$PYTHON" "$(dirname "$0")/run_baseline_selfmorph_ood_eval.py" \
    --config "$CONFIG" --model-path "$MODEL_PATH" \
    --selector-checkpoint "$SELECTOR_CKPT" --train-dev-pt "$TRAIN_DEV_PT" \
    --manifest "$MANIFEST" --output-dir "$OUTPUT_DIR" \
    --shard-id "$i" --num-shards "$NUM_SHARDS" --device "cuda:${GPU}" \
    > "${OUTPUT_DIR}/shard${i}of${NUM_SHARDS}.log" 2>&1 &
  PIDS+=($!)
done

sleep 5
echo "Launched PIDs: ${PIDS[*]}"
echo "Verify with: ps -eo pid,ppid,stat,etime,cmd | grep run_baseline_selfmorph_ood_eval.py | grep -v grep"

wait "${PIDS[@]}"
echo "All $NUM_SHARDS shards finished."
