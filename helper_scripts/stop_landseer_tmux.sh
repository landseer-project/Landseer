#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-landseer}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed."
  exit 1
fi

if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "No tmux session found: $SESSION_NAME"
  exit 0
fi

tmux kill-session -t "$SESSION_NAME"
echo "Stopped tmux session: $SESSION_NAME"
