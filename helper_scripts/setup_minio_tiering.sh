#!/usr/bin/env bash
set -euo pipefail

# Landseer MinIO lifecycle tiering bootstrap.
#
# Default behavior:
# - Starts a local HOT MinIO instance (tmpfs-backed by default)
# - Uses a remote GCS tier as the offload target
# - Creates buckets and configures ILM transition rules for artifacts/
#
# Optional behavior:
# - Skip container startup and only apply ILM rules
# - Use a local cold MinIO tier instead of GCS offload
#
# Usage:
#   bash helper_scripts/setup_minio_tiering.sh
#
#   MODE=remote TIER_TYPE=s3 REMOTE_ENDPOINT=https://s3.amazonaws.com \
#   REMOTE_ACCESS_KEY=... REMOTE_SECRET_KEY=... REMOTE_BUCKET=... \
#   bash helper_scripts/setup_minio_tiering.sh

########################################
# Configurable variables
########################################
MODE="${MODE:-remote}"                    # local | remote
RECREATE="${RECREATE:-1}"                  # 1 recreate containers, 0 reuse
START_CONTAINERS="${START_CONTAINERS:-1}"  # 1 start containers, 0 only configure ILM

ROOT_USER="${ROOT_USER:-minioadmin}"
ROOT_PASSWORD="${ROOT_PASSWORD:-minioadmin}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOT_CONTAINER="${HOT_CONTAINER:-minio-hot}"
HOT_API_PORT="${HOT_API_PORT:-9000}"
HOT_CONSOLE_PORT="${HOT_CONSOLE_PORT:-9001}"
HOT_BUCKET="${HOT_BUCKET:-landseer-artifacts}"
HOT_DATA_MODE="${HOT_DATA_MODE:-tmpfs}"    # tmpfs | bind
HOT_TMPFS_SIZE="${HOT_TMPFS_SIZE:-107374182400}"  # 100 GiB
HOT_BIND_DIR="${HOT_BIND_DIR:-/data/landseer/lanseer-minio-hot}"

COLD_CONTAINER="${COLD_CONTAINER:-minio-cold}"
COLD_API_PORT="${COLD_API_PORT:-9002}"
COLD_CONSOLE_PORT="${COLD_CONSOLE_PORT:-9003}"
COLD_BUCKET="${COLD_BUCKET:-landseer-artifacts-cold}"
COLD_BIND_DIR="${COLD_BIND_DIR:-/data/landseer/lanseer-minio-cold}"

TIER_NAME="${TIER_NAME:-COLD-TIER}"
TIER_PREFIX="${TIER_PREFIX:-artifacts/}"
TRANSITION_PREFIX="${TRANSITION_PREFIX:-artifacts/}"
TRANSITION_DAYS="${TRANSITION_DAYS:-1}"
EXPIRE_DAYS="${EXPIRE_DAYS:-}"            # optional, e.g. 60

# For MODE=remote
TIER_TYPE="${TIER_TYPE:-gcs}"              # s3 | gcs | minio | azure
REMOTE_ENDPOINT="${REMOTE_ENDPOINT:-}"
REMOTE_REGION="${REMOTE_REGION:-us-east-1}"
REMOTE_ACCESS_KEY="${REMOTE_ACCESS_KEY:-}"
REMOTE_SECRET_KEY="${REMOTE_SECRET_KEY:-}"
REMOTE_BUCKET="${REMOTE_BUCKET:-landseer-minio-cache}"
REMOTE_PREFIX="${REMOTE_PREFIX:-artifacts/}"
REMOTE_STORAGE_CLASS="${REMOTE_STORAGE_CLASS:-}"
REMOTE_GCS_CREDENTIALS_FILE="${REMOTE_GCS_CREDENTIALS_FILE:-}"

########################################
# Helpers
########################################

info() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }
err()  { echo "[ERROR] $*" >&2; }

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    err "Required command not found: $1"
    exit 1
  fi
}

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -Fxq "$1"
}

remove_container_if_needed() {
  local name="$1"
  if container_exists "$name"; then
    if [[ "$RECREATE" == "1" ]]; then
      info "Removing existing container: $name"
      docker rm -f "$name" >/dev/null
    else
      info "Reusing existing container: $name"
    fi
  fi
}

start_hot_container() {
  if container_exists "$HOT_CONTAINER" && [[ "$RECREATE" != "1" ]]; then
    info "HOT container already exists and RECREATE=0: $HOT_CONTAINER"
    return
  fi

  remove_container_if_needed "$HOT_CONTAINER"

  if [[ "$HOT_DATA_MODE" == "tmpfs" ]]; then
    info "Starting HOT MinIO on tmpfs (${HOT_TMPFS_SIZE} bytes)"
    docker run -d --name "$HOT_CONTAINER" \
      -p "${HOT_API_PORT}:9000" -p "${HOT_CONSOLE_PORT}:9001" \
      -e MINIO_ROOT_USER="$ROOT_USER" \
      -e MINIO_ROOT_PASSWORD="$ROOT_PASSWORD" \
      --mount "type=tmpfs,destination=/data,tmpfs-size=${HOT_TMPFS_SIZE}" \
      minio/minio server /data --console-address ":9001" >/dev/null
  else
    info "Starting HOT MinIO on bind mount: ${HOT_BIND_DIR}"
    mkdir -p "$HOT_BIND_DIR"
    docker run -d --name "$HOT_CONTAINER" \
      -p "${HOT_API_PORT}:9000" -p "${HOT_CONSOLE_PORT}:9001" \
      -e MINIO_ROOT_USER="$ROOT_USER" \
      -e MINIO_ROOT_PASSWORD="$ROOT_PASSWORD" \
      -v "${HOT_BIND_DIR}:/data" \
      minio/minio server /data --console-address ":9001" >/dev/null
  fi
}

start_cold_container() {
  if [[ "$MODE" != "local" ]]; then
    return
  fi

  if container_exists "$COLD_CONTAINER" && [[ "$RECREATE" != "1" ]]; then
    info "COLD container already exists and RECREATE=0: $COLD_CONTAINER"
    return
  fi

  remove_container_if_needed "$COLD_CONTAINER"

  info "Starting COLD MinIO on bind mount: ${COLD_BIND_DIR}"
  mkdir -p "$COLD_BIND_DIR"
  docker run -d --name "$COLD_CONTAINER" \
    -p "${COLD_API_PORT}:9000" -p "${COLD_CONSOLE_PORT}:9001" \
    -e MINIO_ROOT_USER="$ROOT_USER" \
    -e MINIO_ROOT_PASSWORD="$ROOT_PASSWORD" \
    -v "${COLD_BIND_DIR}:/data" \
    minio/minio server /data --console-address ":9001" >/dev/null
}

wait_for_minio() {
  local endpoint="$1"
  local attempts=60
  local i=1

  info "Waiting for MinIO endpoint: ${endpoint}"
  while (( i <= attempts )); do
    if docker run --rm --network host minio/mc alias set hotcheck "${endpoint}" "$ROOT_USER" "$ROOT_PASSWORD" >/dev/null 2>&1; then
      info "Endpoint is ready: ${endpoint}"
      return 0
    fi
    sleep 2
    ((i++))
  done

  err "Timed out waiting for endpoint: ${endpoint}"
  return 1
}

resolve_gcs_credentials_file() {
  local cred_path="${REMOTE_GCS_CREDENTIALS_FILE:-}"
  if [[ -z "$cred_path" ]]; then
    cred_path="${SCRIPT_DIR}/landseer-494115-4f746115c9cf-gcs-bucket-key.json"
  fi
  if [[ ! -f "$cred_path" ]]; then
    err "For MODE=remote and TIER_TYPE=gcs, set REMOTE_GCS_CREDENTIALS_FILE and REMOTE_BUCKET"
    exit 1
  fi

  if [[ "$cred_path" != /* ]]; then
    cred_path="$(pwd)/$cred_path"
  fi

  if [[ ! -f "$cred_path" ]]; then
    err "GCS credentials file not found: $cred_path"
    exit 1
  fi

  echo "$cred_path"
}

configure_ilm_and_tiering() {
  local hot_url="http://127.0.0.1:${HOT_API_PORT}"
  local cold_url_host="http://127.0.0.1:${COLD_API_PORT}"
  local cold_container_ip=""
  local cold_url_tier=""

  if [[ "$MODE" == "local" ]]; then
    cold_container_ip="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$COLD_CONTAINER" 2>/dev/null || true)"
    if [[ -z "$cold_container_ip" ]]; then
      err "Unable to resolve IP for local cold container: $COLD_CONTAINER"
      exit 1
    fi
    cold_url_tier="http://${cold_container_ip}:9000"
  fi

  local mc_script
  mc_script=$(mktemp)
  cat > "$mc_script" <<'EOS'
set -e
mc alias set hot "${HOT_URL}" "${ROOT_USER}" "${ROOT_PASSWORD}" >/dev/null
mc mb -p "hot/${HOT_BUCKET}" >/dev/null || true

# Remove old tier with same name if present (best-effort)
mc ilm tier rm "hot/${TIER_NAME}" >/dev/null 2>&1 || true

if [ "${MODE}" = "local" ]; then
  mc alias set cold "${COLD_URL_HOST}" "${ROOT_USER}" "${ROOT_PASSWORD}" >/dev/null
  mc mb -p "cold/${COLD_BUCKET}" >/dev/null || true

  mc ilm tier add minio hot "${TIER_NAME}" \
    --endpoint "${COLD_URL_TIER}" \
    --access-key "${ROOT_USER}" \
    --secret-key "${ROOT_PASSWORD}" \
    --bucket "${COLD_BUCKET}" \
    --prefix "${TIER_PREFIX}"
else
  case "${TIER_TYPE}" in
    s3|minio)
      mc ilm tier add "${TIER_TYPE}" hot "${TIER_NAME}" \
        --endpoint "${REMOTE_ENDPOINT}" \
        --access-key "${REMOTE_ACCESS_KEY}" \
        --secret-key "${REMOTE_SECRET_KEY}" \
        --bucket "${REMOTE_BUCKET}" \
        --prefix "${REMOTE_PREFIX}" \
        --region "${REMOTE_REGION}" \
        ${REMOTE_STORAGE_CLASS:+--storage-class "${REMOTE_STORAGE_CLASS}"}
      ;;
    gcs)
      mc ilm tier add gcs hot "${TIER_NAME}" \
        --credentials-file "${REMOTE_GCS_CREDENTIALS_FILE}" \
        --bucket "${REMOTE_BUCKET}" \
        --prefix "${REMOTE_PREFIX}"
      ;;
    *)
      echo "Unsupported TIER_TYPE: ${TIER_TYPE}" >&2
      exit 2
      ;;
  esac
fi

# Add lifecycle transition rule for artifacts/
mc ilm rule add \
  --prefix "${TRANSITION_PREFIX}" \
  --transition-days "${TRANSITION_DAYS}" \
  --transition-tier "${TIER_NAME}" \
  "hot/${HOT_BUCKET}" >/dev/null

# Optional expiry rule
if [ -n "${EXPIRE_DAYS}" ]; then
  mc ilm rule add \
    --prefix "${TRANSITION_PREFIX}" \
    --expire-days "${EXPIRE_DAYS}" \
    "hot/${HOT_BUCKET}" >/dev/null
fi

mc ilm tier ls hot
mc ilm rule ls "hot/${HOT_BUCKET}"
EOS

  local -a docker_args=(
    docker run --rm --network host
    -e MODE="$MODE"
    -e ROOT_USER="$ROOT_USER"
    -e ROOT_PASSWORD="$ROOT_PASSWORD"
    -e HOT_URL="$hot_url"
    -e COLD_URL_HOST="$cold_url_host"
    -e COLD_URL_TIER="$cold_url_tier"
    -e HOT_BUCKET="$HOT_BUCKET"
    -e COLD_BUCKET="$COLD_BUCKET"
    -e TIER_NAME="$TIER_NAME"
    -e TIER_PREFIX="$TIER_PREFIX"
    -e TRANSITION_PREFIX="$TRANSITION_PREFIX"
    -e TRANSITION_DAYS="$TRANSITION_DAYS"
    -e EXPIRE_DAYS="$EXPIRE_DAYS"
    -e TIER_TYPE="$TIER_TYPE"
    -e REMOTE_ENDPOINT="$REMOTE_ENDPOINT"
    -e REMOTE_REGION="$REMOTE_REGION"
    -e REMOTE_ACCESS_KEY="$REMOTE_ACCESS_KEY"
    -e REMOTE_SECRET_KEY="$REMOTE_SECRET_KEY"
    -e REMOTE_BUCKET="$REMOTE_BUCKET"
    -e REMOTE_PREFIX="$REMOTE_PREFIX"
    -e REMOTE_STORAGE_CLASS="$REMOTE_STORAGE_CLASS"
    -e REMOTE_GCS_CREDENTIALS_FILE="/tmp/gcs-creds.json"
    -v "$mc_script:/tmp/setup.sh:ro"
  )

  if [[ "$MODE" == "remote" && "$TIER_TYPE" == "gcs" ]]; then
    local gcs_cred_path
    gcs_cred_path="$(resolve_gcs_credentials_file)"
    docker_args+=( -v "${gcs_cred_path}:/tmp/gcs-creds.json:ro" )
  fi

  docker_args+=( --entrypoint /bin/sh minio/mc /tmp/setup.sh )
  "${docker_args[@]}"

  rm -f "$mc_script"
}

validate_remote_mode_inputs() {
  if [[ "$MODE" != "remote" ]]; then
    return
  fi

  case "$TIER_TYPE" in
    s3|minio)
      if [[ -z "$REMOTE_ENDPOINT" || -z "$REMOTE_ACCESS_KEY" || -z "$REMOTE_SECRET_KEY" || -z "$REMOTE_BUCKET" ]]; then
        err "For MODE=remote and TIER_TYPE=${TIER_TYPE}, set REMOTE_ENDPOINT, REMOTE_ACCESS_KEY, REMOTE_SECRET_KEY, REMOTE_BUCKET"
        exit 1
      fi
      ;;
    gcs)
      if [[ -z "$REMOTE_BUCKET" ]]; then
        err "For MODE=remote and TIER_TYPE=gcs, set REMOTE_BUCKET"
        exit 1
      fi
      resolve_gcs_credentials_file >/dev/null
      ;;
    *)
      err "Unsupported TIER_TYPE: ${TIER_TYPE}"
      exit 1
      ;;
  esac
}

main() {
  need_cmd docker
  validate_remote_mode_inputs

  if [[ "$START_CONTAINERS" == "1" ]]; then
    start_hot_container
    start_cold_container
  fi

  wait_for_minio "http://127.0.0.1:${HOT_API_PORT}"
  if [[ "$MODE" == "local" ]]; then
    wait_for_minio "http://127.0.0.1:${COLD_API_PORT}"
  fi

  configure_ilm_and_tiering

  info "Done."
  info "HOT endpoint: http://127.0.0.1:${HOT_API_PORT}"
  info "HOT console : http://127.0.0.1:${HOT_CONSOLE_PORT}"
  if [[ "$MODE" == "local" ]]; then
    info "COLD endpoint: http://127.0.0.1:${COLD_API_PORT}"
    info "COLD console : http://127.0.0.1:${COLD_CONSOLE_PORT}"
  fi
}

main "$@"
