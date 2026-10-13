#!/usr/bin/env bash
# Smoke-start backend with a test pipeline config (uv only; no Poetry).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run landseer-backend --config configs/pipeline/test_config_1.yaml
