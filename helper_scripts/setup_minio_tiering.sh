#!/usr/bin/env bash
# Start local MinIO (hot) and tier hot objects under artifacts/ to GCS (cold).
# Secrets from .env.db (gitignored).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f "${REPO_ROOT}/.env.db" ]]; then
  echo "Missing ${REPO_ROOT}/.env.db (copy from .env.example)" >&2
  exit 1
fi
set -a
# shellcheck source=/dev/null
source "${REPO_ROOT}/.env.db"
set +a

ROOT_USER="${ROOT_USER:-${MINIO_ACCESS_KEY:?}}"
ROOT_PASSWORD="${ROOT_PASSWORD:-${MINIO_SECRET_KEY:?}}"
HOT_NAME="${HOT_CONTAINER:-minio-hot}"
HOT_PORT="${HOT_API_PORT:-9000}"
HOT_CONSOLE="${HOT_CONSOLE_PORT:-9001}"
HOT_BUCKET="${MINIO_BUCKET:-landseer-artifacts}"
HOT_DATA="${HOT_BIND_DIR:-/data/landseer/minio-hot}"
# Object keys workers write (see MinioStore.get_artifact_key)
HOT_PREFIX="${HOT_PREFIX:-artifacts/}"
TIER_NAME="${TIER_NAME:-GCS-COLD}"
REMOTE_BUCKET="${REMOTE_BUCKET:?set REMOTE_BUCKET in .env.db}"
# Unique prefix inside the GCS bucket (avoids "tier already in use")
REMOTE_PREFIX="${REMOTE_PREFIX:-landseer/artifacts/}"
TRANSITION_DAYS=0

GCS_CREDS="${REMOTE_GCS_CREDENTIALS_FILE:?set REMOTE_GCS_CREDENTIALS_FILE in .env.db}"
[[ "$GCS_CREDS" == /* ]] || GCS_CREDS="${REPO_ROOT}/${GCS_CREDS}"
[[ -f "$GCS_CREDS" ]] || { echo "GCS credentials not found: $GCS_CREDS" >&2; exit 1; }

command -v docker >/dev/null || { echo "docker required" >&2; exit 1; }

echo "Starting ${HOT_NAME} on :${HOT_PORT} (data: ${HOT_DATA})"
mkdir -p "$HOT_DATA"
docker rm -f "$HOT_NAME" >/dev/null 2>&1 || true
docker run -d --name "$HOT_NAME" \
  -p "${HOT_PORT}:9000" -p "${HOT_CONSOLE}:9001" \
  -e MINIO_ROOT_USER="$ROOT_USER" \
  -e MINIO_ROOT_PASSWORD="$ROOT_PASSWORD" \
  -v "${HOT_DATA}:/data" \
  minio/minio server /data --console-address ":9001" >/dev/null

HOT_URL="http://127.0.0.1:${HOT_PORT}"
echo "Waiting for ${HOT_URL}..."
for _ in $(seq 1 30); do
  if docker run --rm --network host minio/mc \
    alias set probe "$HOT_URL" "$ROOT_USER" "$ROOT_PASSWORD" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo "Configuring bucket + GCS tier (hot ${HOT_PREFIX} -> gs://${REMOTE_BUCKET}/${REMOTE_PREFIX})..."
docker run --rm --network host \
  -v "${GCS_CREDS}:/tmp/gcs.json:ro" \
  --entrypoint /bin/sh minio/mc -c "
    set -e
    mc alias set hot '${HOT_URL}' '${ROOT_USER}' '${ROOT_PASSWORD}' >/dev/null
    mc mb -p 'hot/${HOT_BUCKET}' >/dev/null || true
    mc ilm tier add gcs hot '${TIER_NAME}' \
      --credentials-file /tmp/gcs.json \
      --bucket '${REMOTE_BUCKET}' \
      --prefix '${REMOTE_PREFIX}' 2>/dev/null \
      || echo 'Tier ${TIER_NAME} already present'
    mc ilm rule add \
      --prefix '${HOT_PREFIX}' \
      --transition-days '${TRANSITION_DAYS}' \
      --transition-tier '${TIER_NAME}' \
      'hot/${HOT_BUCKET}' >/dev/null 2>&1 \
      || echo 'Transition rule already present'
    mc ilm tier ls hot
    mc ilm rule ls 'hot/${HOT_BUCKET}'
  "

echo "Done. MinIO API: ${HOT_URL}  console: http://127.0.0.1:${HOT_CONSOLE}"
echo "Workers upload to MinIO; objects under ${HOT_PREFIX} tier to GCS (${TRANSITION_DAYS}d)."
