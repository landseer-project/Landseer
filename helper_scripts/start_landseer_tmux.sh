#!/usr/bin/env bash
set -euo pipefail

# Start Landseer backend, workers, and frontend in separate tmux windows.
#
# Uses uv (not Poetry). Prefer this or the Cursor skill script:
#   bash .cursor/skills/landseer-tmux-run/scripts/landseer-tmux.sh
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

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Install tmux and retry."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install: https://docs.astral.sh/uv/"
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

cd "$PROJECT_ROOT"
uv sync --all-groups

backend_cmd="cd '$PROJECT_ROOT' && uv run landseer-backend --host '$BACKEND_HOST' --port '$BACKEND_PORT' --tools-config '$TOOLS_CONFIG'"
if [[ -n "$BACKEND_CONFIG" ]]; then
  backend_cmd+=" --config '$BACKEND_CONFIG'"
fi

frontend_cmd="cd '$PROJECT_ROOT' && uv run landseer-frontend dev -- --host '$FRONTEND_HOST' --port '$FRONTEND_PORT'"

# Create session and backend window.
tmux new-session -d -s "$SESSION_NAME" -n backend "$backend_cmd"

# Create worker windows.
for i in $(seq 0 $((WORKERS - 1))); do
  gpu=$((WORKER_GPU_START + i))
  worker_id="tmux-worker-${gpu}"
  worker_cmd="cd '$PROJECT_ROOT' && uv run landseer-worker --backend-url '$BACKEND_URL' --worker-id '$worker_id' --gpu '$gpu'"
  tmux new-window -t "$SESSION_NAME" -n "worker-${gpu}" "$worker_cmd"
done

# Create frontend window.
tmux new-window -t "$SESSION_NAME" -n frontend "$frontend_cmd"

# Optional monitor window for quick status checks.
monitor_cmd="cd '$PROJECT_ROOT' && uv run python helper_scripts/monitor_workers.py --backend-url '$BACKEND_URL'"
tmux new-window -t "$SESSION_NAME" -n monitor "$monitor_cmd"

# Focus backend window first.
tmux select-window -t "$SESSION_NAME:backend"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
