#!/usr/bin/env bash
#
# Landseer CPU end-to-end run: MNIST -> in_noop (LeNet-5) -> clean evaluator.
#
# Everything runs on CPU in Docker. No GPU, no NVIDIA runtime, no GHCR access,
# no MySQL/MinIO/backend/worker -- this exercises the three container stages
# directly so you can confirm the CPU path works.
#
# Usage (from the repo root):
#   ./helper_scripts/run_mnist_lenet_cpu.sh
#
# Options via environment:
#   RUN_DIR=/path/to/dir   where artifacts land        (default: ./cpu_run_mnist_lenet)
#   EPOCHS=3               training epochs             (default: 3)
#   BATCH_SIZE=128         training batch size         (default: 128)
#   MODEL_CONFIG=path      model config to use         (default: configs/model/config_model_lenet.py)
#   SKIP_BUILD=1           reuse already-built images
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="${RUN_DIR:-$REPO_ROOT/cpu_run_mnist_lenet}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/config_model_lenet.py}"

IMG_DATA="landseer/dataset_mnist:cpu"
IMG_TRAIN="landseer/in_noop_lenet:cpu"
IMG_EVAL="landseer/eval_clean:cpu"

DATA_DIR="$RUN_DIR/dataset"
CONFIG_DIR="$RUN_DIR/config"
TRAIN_OUT="$RUN_DIR/trained"
EVAL_WS="$RUN_DIR/eval_workspace"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: the Docker daemon is not reachable. Start Docker Desktop and retry." >&2
  exit 1
fi

if [ ! -f "$MODEL_CONFIG" ]; then
  echo "ERROR: model config not found: $MODEL_CONFIG" >&2
  exit 1
fi

mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$TRAIN_OUT" "$EVAL_WS/input" "$EVAL_WS/output"

# The tool and evaluator contracts both expect the file to be named config_model.py.
cp "$MODEL_CONFIG" "$CONFIG_DIR/config_model.py"

if [ -z "${SKIP_BUILD:-}" ]; then
  say "Building images (CPU-only, native architecture)"
  docker build -t "$IMG_DATA"  -f containers/datasets/mnist/Dockerfile        containers/datasets/mnist
  docker build -t "$IMG_TRAIN" -f containers/tools/in_noop_lenet/Dockerfile   containers/tools/in_noop_lenet
  docker build -t "$IMG_EVAL"  -f containers/evals/clean/Dockerfile.cpu containers/evals
else
  say "SKIP_BUILD set - reusing existing images"
fi

say "Stage 1/3: MNIST dataset preparation"
docker run --rm \
  -v "$DATA_DIR:/output" \
  "$IMG_DATA" --output /output

say "Stage 2/3: in_noop during-training baseline (LeNet-5, ${EPOCHS} epochs)"
docker run --rm \
  -e EPOCHS="$EPOCHS" -e BATCH_SIZE="$BATCH_SIZE" \
  -v "$DATA_DIR:/data:ro" \
  -v "$CONFIG_DIR:/config:ro" \
  -v "$TRAIN_OUT:/output" \
  "$IMG_TRAIN"

say "Stage 3/3: clean evaluator"
# The evaluator reads $WORKSPACE/input and writes $WORKSPACE/output.
cp "$TRAIN_OUT"/model.pt "$TRAIN_OUT"/*.npy "$TRAIN_OUT"/config_model.py "$EVAL_WS/input/"
docker run --rm \
  -e WORKSPACE=/workspace \
  -v "$EVAL_WS:/workspace" \
  "$IMG_EVAL"

say "Results"
RESULTS="$EVAL_WS/output/evaluation_results.json"
if [ -f "$RESULTS" ]; then
  cat "$RESULTS"
  echo
  echo "Artifacts under: $RUN_DIR"
else
  echo "ERROR: expected results at $RESULTS but it was not written." >&2
  exit 1
fi
