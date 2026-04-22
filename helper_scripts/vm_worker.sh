#!/usr/bin/env bash
set -euo pipefail

# Simple VM worker manager for Landseer remote workers.
# Usage:
#   ./helper_scripts/vm_worker.sh start
#   ./helper_scripts/vm_worker.sh stop
#   ./helper_scripts/vm_worker.sh restart
#   ./helper_scripts/vm_worker.sh status
#   ./helper_scripts/vm_worker.sh logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${WORKER_ENV_FILE:-$SCRIPT_DIR/vm_worker.env}"
PID_FILE="${WORKER_PID_FILE:-$SCRIPT_DIR/.vm_worker.pid}"
LOG_FILE="${WORKER_LOG_FILE:-$SCRIPT_DIR/vm_worker.log}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE"
  echo "Copy helper_scripts/vm_worker.env.example to helper_scripts/vm_worker.env and edit values."
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

required_vars=(
  LANDSEER_BACKEND_URL
  LANDSEER_PIPELINE_KEYS
  MINIO_ENDPOINT
  MINIO_ACCESS_KEY
  MINIO_SECRET_KEY
  MINIO_BUCKET
)

for v in "${required_vars[@]}"; do
  if [[ -z "${!v:-}" ]]; then
    echo "Required variable missing in $ENV_FILE: $v"
    exit 1
  fi
done

WORKER_GPU="${WORKER_GPU:-0}"
WORKER_RUNTIME="${WORKER_RUNTIME:-docker}"
WORKER_CACHE_DIR="${WORKER_CACHE_DIR:-/tmp/landseer_cache}"
WORKER_WORKSPACE="${WORKER_WORKSPACE:-/tmp/landseer_worker}"
WORKER_ID="${WORKER_ID:-vm-worker-$(hostname)}"
WORKER_POLL_INTERVAL="${WORKER_POLL_INTERVAL:-5.0}"
WORKER_HEARTBEAT_INTERVAL="${WORKER_HEARTBEAT_INTERVAL:-30.0}"
WORKER_EXTRA_ARGS="${WORKER_EXTRA_ARGS:-}"
MINIO_SECURE="${MINIO_SECURE:-false}"

check_connectivity() {
  echo "Checking backend: ${LANDSEER_BACKEND_URL}/health"
  curl -fsS "${LANDSEER_BACKEND_URL}/health" >/dev/null

  echo "Checking MinIO: http://${MINIO_ENDPOINT}/minio/health/live"
  curl -fsS "http://${MINIO_ENDPOINT}/minio/health/live" >/dev/null
}

is_running() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" >/dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

start_worker() {
  if is_running; then
    echo "Worker already running (pid: $(cat "$PID_FILE"))."
    return 0
  fi

  check_connectivity

  mkdir -p "$(dirname "$LOG_FILE")" "$WORKER_CACHE_DIR" "$WORKER_WORKSPACE"

  (
    cd "$PROJECT_ROOT"

    export LANDSEER_PIPELINE_KEYS
    export MINIO_ENDPOINT
    export MINIO_ACCESS_KEY
    export MINIO_SECRET_KEY
    export MINIO_BUCKET
    export MINIO_SECURE

    exec poetry run landseer-worker \
      --backend-url "$LANDSEER_BACKEND_URL" \
      --worker-id "$WORKER_ID" \
      --workspace "$WORKER_WORKSPACE" \
      --cache-dir "$WORKER_CACHE_DIR" \
      --gpu "$WORKER_GPU" \
      --runtime "$WORKER_RUNTIME" \
      --poll-interval "$WORKER_POLL_INTERVAL" \
      --heartbeat-interval "$WORKER_HEARTBEAT_INTERVAL" \
      --minio-endpoint "$MINIO_ENDPOINT" \
      $WORKER_EXTRA_ARGS
  ) >>"$LOG_FILE" 2>&1 &

  echo $! >"$PID_FILE"
  echo "Worker started (pid: $(cat "$PID_FILE")). Logs: $LOG_FILE"
}

stop_worker() {
  if ! is_running; then
    echo "Worker is not running."
    rm -f "$PID_FILE"
    return 0
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  kill "$pid" >/dev/null 2>&1 || true

  for _ in {1..20}; do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      rm -f "$PID_FILE"
      echo "Worker stopped."
      return 0
    fi
    sleep 0.5
  done

  echo "Worker did not stop gracefully; sending SIGKILL."
  kill -9 "$pid" >/dev/null 2>&1 || true
  rm -f "$PID_FILE"
  echo "Worker force-stopped."
}

status_worker() {
  if is_running; then
    echo "Worker running (pid: $(cat "$PID_FILE"))."
  else
    echo "Worker not running."
  fi
}

show_logs() {
  if [[ -f "$LOG_FILE" ]]; then
    tail -n 100 "$LOG_FILE"
  else
    echo "No log file yet: $LOG_FILE"
  fi
}

cmd="${1:-}"
case "$cmd" in
  start)
    start_worker
    ;;
  stop)
    stop_worker
    ;;
  restart)
    stop_worker
    start_worker
    ;;
  status)
    status_worker
    ;;
  logs)
    show_logs
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs}"
    exit 1
    ;;
esac
