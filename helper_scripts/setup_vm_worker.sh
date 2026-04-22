#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a mostly blank Linux VM for running Landseer worker.
# Supports apt, dnf, and pacman based distros.
#
# Example:
#   ./helper_scripts/setup_vm_worker.sh \
#     --repo-url https://github.com/landseer-project/Landseer.git \
#     --branch dev/restructure \
#     --install-dir "$HOME/Landseer" \
#     --host-ip 100.125.174.120 \
#     --pipeline-key "REPLACE_ME" \
#     --minio-access-key "REPLACE_ME" \
#     --minio-secret-key "REPLACE_ME" \
#     --worker-id "gcp-worker-1" \
#     --gpu 0


log() { echo "[setup-vm-worker] $*"; }
fail() { echo "[setup-vm-worker] ERROR: $*" >&2; exit 1; }

require_arg() {
  local name="$1"
  local value="${2:-}"
  [[ -n "$value" ]] || fail "Missing required argument: $name"
}

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  if ! command -v sudo >/dev/null 2>&1; then
    fail "Run as root or install sudo first."
  fi
  SUDO="sudo"
else
  SUDO=""
fi

REPO_URL=""
BRANCH="dev/restructure"
INSTALL_DIR="$HOME/Landseer"
HOST_IP=""
PIPELINE_KEY=""
MINIO_ACCESS_KEY=""
MINIO_SECRET_KEY=""
MINIO_BUCKET="landseer-artifacts"
WORKER_ID="gcp-$(hostname)"
WORKER_GPU="0"
WORKER_RUNTIME="docker"
SKIP_TAILSCALE="false"
TAILSCALE_HOSTNAME="$(hostname)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-url) REPO_URL="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --install-dir) INSTALL_DIR="$2"; shift 2 ;;
    --host-ip) HOST_IP="$2"; shift 2 ;;
    --pipeline-key) PIPELINE_KEY="$2"; shift 2 ;;
    --minio-access-key) MINIO_ACCESS_KEY="$2"; shift 2 ;;
    --minio-secret-key) MINIO_SECRET_KEY="$2"; shift 2 ;;
    --minio-bucket) MINIO_BUCKET="$2"; shift 2 ;;
    --worker-id) WORKER_ID="$2"; shift 2 ;;
    --gpu) WORKER_GPU="$2"; shift 2 ;;
    --runtime) WORKER_RUNTIME="$2"; shift 2 ;;
    --skip-tailscale) SKIP_TAILSCALE="true"; shift ;;
    --tailscale-hostname) TAILSCALE_HOSTNAME="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,60p' "$0"
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

require_arg "--repo-url" "$REPO_URL"
require_arg "--host-ip" "$HOST_IP"
require_arg "--pipeline-key" "$PIPELINE_KEY"
require_arg "--minio-access-key" "$MINIO_ACCESS_KEY"
require_arg "--minio-secret-key" "$MINIO_SECRET_KEY"

install_packages() {
  if command -v apt-get >/dev/null 2>&1; then
    log "Detected apt-based distro; installing dependencies."
    $SUDO apt-get update -y
    $SUDO apt-get install -y \
      git curl ca-certificates python3 python3-pip python3-venv \
      build-essential pkg-config docker.io jq
    $SUDO systemctl enable --now docker
  elif command -v dnf >/dev/null 2>&1; then
    log "Detected dnf-based distro; installing dependencies."
    $SUDO dnf install -y \
      git curl ca-certificates python3 python3-pip gcc gcc-c++ make \
      pkgconf-pkg-config docker jq
    $SUDO systemctl enable --now docker
  elif command -v pacman >/dev/null 2>&1; then
    log "Detected pacman-based distro; installing dependencies."
    $SUDO pacman -Sy --noconfirm \
      git curl ca-certificates python python-pip base-devel docker jq
    $SUDO systemctl enable --now docker
  else
    fail "Unsupported package manager. Install git/curl/python3/pip/docker manually."
  fi
}

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    log "uv already installed."
    return
  fi
  log "Installing uv."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  if ! grep -q 'HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
  fi
}

install_tailscale() {
  if [[ "$SKIP_TAILSCALE" == "true" ]]; then
    log "Skipping Tailscale installation (--skip-tailscale)."
    return
  fi
  if command -v tailscale >/dev/null 2>&1; then
    log "Tailscale already installed."
  else
    log "Installing Tailscale."
    curl -fsSL https://tailscale.com/install.sh | sh
  fi

  log "Starting Tailscale."
  $SUDO systemctl enable --now tailscaled
  log "Run this once manually to join tailnet:"
  log "  sudo tailscale up --ssh --hostname ${TAILSCALE_HOSTNAME}"
}

clone_or_update_repo() {
  if [[ -d "$INSTALL_DIR/.git" ]]; then
    log "Repo already exists at $INSTALL_DIR; fetching updates."
    git -C "$INSTALL_DIR" fetch --all --prune
  else
    log "Cloning repo to $INSTALL_DIR (branch: $BRANCH)."
    git clone "$REPO_URL" "$INSTALL_DIR"
  fi

  log "Checking out branch: $BRANCH"
  git -C "$INSTALL_DIR" checkout "$BRANCH"
  git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH"
}

configure_worker_env() {
  local env_file="$INSTALL_DIR/helper_scripts/vm_worker.env"
  local example_file="$INSTALL_DIR/helper_scripts/vm_worker.env.example"
  local script_file="$INSTALL_DIR/helper_scripts/vm_worker.sh"

  [[ -f "$example_file" ]] || fail "Missing $example_file in cloned repo."
  [[ -f "$script_file" ]] || fail "Missing $script_file in cloned repo."

  cp "$example_file" "$env_file"
  chmod +x "$script_file"

  sed -i "s|^LANDSEER_HOST_IP=.*|LANDSEER_HOST_IP=${HOST_IP}|" "$env_file"
  sed -i "s|^LANDSEER_PIPELINE_KEYS=.*|LANDSEER_PIPELINE_KEYS=\"${PIPELINE_KEY}\"|" "$env_file"
  sed -i "s|^MINIO_ACCESS_KEY=.*|MINIO_ACCESS_KEY=\"${MINIO_ACCESS_KEY}\"|" "$env_file"
  sed -i "s|^MINIO_SECRET_KEY=.*|MINIO_SECRET_KEY=\"${MINIO_SECRET_KEY}\"|" "$env_file"
  sed -i "s|^MINIO_BUCKET=.*|MINIO_BUCKET=\"${MINIO_BUCKET}\"|" "$env_file"
  sed -i "s|^WORKER_ID=.*|WORKER_ID=\"${WORKER_ID}\"|" "$env_file"
  sed -i "s|^WORKER_GPU=.*|WORKER_GPU=\"${WORKER_GPU}\"|" "$env_file"
  sed -i "s|^WORKER_RUNTIME=.*|WORKER_RUNTIME=\"${WORKER_RUNTIME}\"|" "$env_file"
}

install_python_deps() {
  log "Installing project dependencies with uv."
  export PATH="$HOME/.local/bin:$PATH"
  uv sync --project "$INSTALL_DIR" --all-groups
}

main() {
  install_packages
  install_uv
  install_tailscale
  clone_or_update_repo
  install_python_deps
  configure_worker_env

  log "Bootstrap complete."
  log "Branch in use: ${BRANCH}"
  log "Next steps on VM:"
  log "  1) Ensure tailnet join done: sudo tailscale up --ssh --hostname ${TAILSCALE_HOSTNAME}"
  log "  2) Start worker: cd ${INSTALL_DIR} && ./helper_scripts/vm_worker.sh start"
  log "  3) Restart after backend reboot: ./helper_scripts/vm_worker.sh restart"
  log "  4) View logs: ./helper_scripts/vm_worker.sh logs"
}

main
