#!/usr/bin/env bash
set -euo pipefail

# Start Landseer backend, workers, and frontend in separate tmux windows.
#
# Usage examples:
#   ./helper_scripts/start_landseer_tmux.sh
#   SESSION_NAME=landseer-dev WORKERS=2 ./helper_scripts/start_landseer_tmux.sh
#   BACKEND_CONFIG=configs/pipeline/trades.yaml WORKERS=4 ./helper_scripts/start_landseer_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SESSION_NAME="${SESSION_NAME:-landseer}"
WORKERS="${WORKERS:-1}"
WORKER_GPU_START="${WORKER_GPU_START:-0}"

BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:${BACKEND_PORT}}"
BACKEND_CONFIG="${BACKEND_CONFIG:-}"
TOOLS_CONFIG="${TOOLS_CONFIG:-configs/tools.yaml}"

FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

# Preferred shared env from current workspace setup.
LANDSEER_ENV="${LANDSEER_ENV:-/share/landseer/workspace-ayushi/.shared/envs/landseer-dev-restructure}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Install tmux and retry."
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

PYTHON_CMD=""
if [[ -x "$LANDSEER_ENV/bin/python" ]]; then
  PYTHON_CMD="$LANDSEER_ENV/bin/python"
elif command -v uv >/dev/null 2>&1; then
  # Fallback to uv if the shared virtual environment is not available.
  PYTHON_CMD="uv run python"
else
  echo "No Python launcher found. Expected $LANDSEER_ENV/bin/python or uv in PATH."
  exit 1
fi

backend_cmd="cd '$PROJECT_ROOT' && export PYTHONPATH='$PROJECT_ROOT':\${PYTHONPATH:-} && $PYTHON_CMD -m src.backend.cli --host '$BACKEND_HOST' --port '$BACKEND_PORT' --tools-config '$TOOLS_CONFIG'"
if [[ -n "$BACKEND_CONFIG" ]]; then
  backend_cmd+=" --config '$BACKEND_CONFIG'"
fi

frontend_cmd="cd '$PROJECT_ROOT' && export PYTHONPATH='$PROJECT_ROOT':\${PYTHONPATH:-} && $PYTHON_CMD -m src.frontend.cli dev -- --host '$FRONTEND_HOST' --port '$FRONTEND_PORT'"

# Create session and backend window.
tmux new-session -d -s "$SESSION_NAME" -n backend "$backend_cmd"

# Create worker windows.
for i in $(seq 0 $((WORKERS - 1))); do
  gpu=$((WORKER_GPU_START + i))
  worker_id="tmux-worker-${gpu}"
  worker_cmd="cd '$PROJECT_ROOT' && export PYTHONPATH='$PROJECT_ROOT':\${PYTHONPATH:-} && $PYTHON_CMD -m src.worker.cli --backend-url '$BACKEND_URL' --worker-id '$worker_id' --gpu '$gpu'"
  tmux new-window -t "$SESSION_NAME" -n "worker-${gpu}" "$worker_cmd"
done

# Create frontend window.
tmux new-window -t "$SESSION_NAME" -n frontend "$frontend_cmd"

# Optional monitor window for quick status checks.
monitor_cmd="cd '$PROJECT_ROOT' && export PYTHONPATH='$PROJECT_ROOT':\${PYTHONPATH:-} && $PYTHON_CMD helper_scripts/monitor_workers.py --backend-url '$BACKEND_URL'"
tmux new-window -t "$SESSION_NAME" -n monitor "$monitor_cmd"

# Focus backend window first.
tmux select-window -t "$SESSION_NAME:backend"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Windows: backend, worker-*, frontend, monitor"
