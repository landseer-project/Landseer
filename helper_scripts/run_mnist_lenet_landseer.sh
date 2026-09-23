#!/usr/bin/env bash
#
# Run the MNIST + LeNet-5 pipeline through Landseer on CPU.
#
# Starts the backend and one CPU worker, waits for every task to finish,
# prints the results, then shuts both down. No GPU, no MySQL (uses sqlite),
# no MinIO, no ghcr login needed - all the images are built locally.
#
# Usage, from the repo root:
#   ./helper_scripts/run_mnist_lenet_landseer.sh
#
# Options via environment:
#   PORT=8001        if 8000 is taken
#   RUN_DIR=/path    where the db / workspace / cache go (default ./cpu_landseer_run)
#   KEEP_DB=1        keep the previous run dir. NOTE: this does not resume an
#                    interrupted run - the backend re-syncs task state on
#                    startup, so completed tasks get re-queued and re-run.
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-8000}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/cpu_landseer_run}"
CONFIG="configs/pipeline/mnist_lenet.yaml"

# these two matter: without them you get a 6s stall and error spam per process
# trying to reach a MinIO server that isn't there
export LANDSEER_USE_MINIO=false
export LANDSEER_HARDLINK_MACRO=true
export LANDSEER_DB_TYPE=sqlite
export LANDSEER_DB_PATH="$RUN_DIR/landseer.db"
export PYTHONPATH=./src:.

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker isn't running. Start Docker Desktop (open -a Docker) and retry." >&2
  exit 1
fi

# Refuse to start if something is already on this port. Otherwise the health
# check below just succeeds against whatever backend is already there, we never
# actually start our own, and wiping RUN_DIR below pulls the database out from
# under the running one. That looks like "1 failed, 13 pending, nothing running".
if lsof -ti:"$PORT" >/dev/null 2>&1; then
  echo "ERROR: something is already listening on port $PORT." >&2
  echo "  Either stop it:  pkill -f 'landseer-backend|landseer-worker'" >&2
  echo "  or use another:  PORT=8001 $0" >&2
  exit 1
fi

# Same problem from the other side: a worker left over from an earlier run will
# fight this one over the same workspace.
if pgrep -f "landseer-worker" >/dev/null 2>&1; then
  echo "ERROR: a landseer-worker is already running. Stop it first:" >&2
  echo "  pkill -f 'landseer-backend|landseer-worker'" >&2
  exit 1
fi

# a leftover db still has all the tasks marked completed, so the worker would
# sit there with nothing to do and you'd think it was broken
if [ -z "${KEEP_DB:-}" ]; then
  rm -rf "$RUN_DIR"
fi
mkdir -p "$RUN_DIR/data" "$RUN_DIR/ws" "$RUN_DIR/cache"

cleanup() { kill "${WORKER_PID:-}" "${BACKEND_PID:-}" 2>/dev/null; }
# on ctrl-c we have to actually exit, not just clean up - otherwise the trap
# runs, the loop below carries on, and the script looks unkillable
on_interrupt() { echo; echo "interrupted, shutting down"; cleanup; exit 130; }
trap cleanup EXIT
trap on_interrupt INT TERM

say "Starting backend on port $PORT"
.venv/bin/landseer-backend --host 127.0.0.1 --port "$PORT" \
  --config "$CONFIG" --data-dir "$RUN_DIR/data" > "$RUN_DIR/backend.log" 2>&1 &
BACKEND_PID=$!

until curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "ERROR: backend died on startup. Last lines:" >&2
    tail -20 "$RUN_DIR/backend.log" >&2
    exit 1
  fi
  sleep 2
done
echo "backend up: $(curl -s http://127.0.0.1:$PORT/health)"

say "Starting CPU worker (no --gpu means cpu)"
.venv/bin/landseer-worker --backend-url "http://127.0.0.1:$PORT" \
  --workspace "$RUN_DIR/ws" --cache-dir "$RUN_DIR/cache" > "$RUN_DIR/worker.log" 2>&1 &
WORKER_PID=$!

say "Waiting for tasks (training LeNet-5 takes about 15s, whole run a couple of minutes)"
STALL=0
# 240*5s = 20min was too short: a 100-epoch DP run takes ~24min on its own.
# 2160*5s = 3h.
for _ in $(seq 1 2160); do
  sleep 5
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "backend exited unexpectedly - see $RUN_DIR/backend.log"; break
  fi
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "worker exited unexpectedly - see $RUN_DIR/worker.log"; break
  fi
  P=$(curl -s "http://127.0.0.1:$PORT/progress" 2>/dev/null)
  [ -z "$P" ] && { echo "  no response from backend"; continue; }
  echo "  $P"
  echo "$P" | grep -q '"is_complete": *true' && break
  # nothing running and nothing left to become ready means a task failed and
  # its dependents can never start. stop instead of spinning for 20 minutes.
  if echo "$P" | grep -q '"running": *0' && echo "$P" | grep -qv '"failed": *0'; then
    STALL=$((STALL+1))
    if [ $STALL -ge 3 ]; then
      echo
      echo "STALLED: a task failed and everything downstream depends on it."
      echo "Failed tasks:"
      .venv/bin/python - <<PY2
import sqlite3
c = sqlite3.connect("$RUN_DIR/landseer.db")
for r in c.execute("SELECT id,tool_name,error_message FROM tasks WHERE status='FAILED'"):
    print(f"  {r[0]} {r[1]}: {str(r[2])[:160]}")
PY2
      break
    fi
  else
    STALL=0
  fi
done

say "Task states"
.venv/bin/python - <<PY
import sqlite3
c = sqlite3.connect("$RUN_DIR/landseer.db")
for r in c.execute("SELECT id, tool_name, status, execution_time_ms FROM tasks ORDER BY rowid"):
    print(f"  {str(r[0]):9s} {str(r[1]):20s} {str(r[2]):10s} {r[3]}ms")
PY

say "Results"
CSV=$(ls -t results/*/*/metrics_summary.csv 2>/dev/null | head -1)
if [ -n "$CSV" ]; then
  echo "$CSV"
  .venv/bin/python - <<PY
import csv
r = list(csv.DictReader(open("$CSV")))[-1]
print("  combination_success =", r.get("combination_success"))
print("  --- metrics ---")
for k, v in r.items():
    if "." in k and not k.endswith(".status") and v not in ("-1", "-1.0", ""):
        print(f"    {k:40s} {v}")
PY
else
  echo "no metrics_summary.csv found - check $RUN_DIR/worker.log"
fi

say "Done (backend and worker are being shut down)"
