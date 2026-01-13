#!/usr/bin/env bash
# b200_setup_remote.sh - Set up a fresh B200 box for InvarLock validation.
# Usage:
#   scp scripts/b200_setup_remote.sh root@<host>:/root/
#   ssh root@<host> 'chmod +x /root/b200_setup_remote.sh && /root/b200_setup_remote.sh'
#
# After setup, run:
#   ssh root@<host> 'cd /root/invarlock-public && . .venv/bin/activate && \
#     INVARLOCK_ALLOW_NETWORK=1 SKIP_FLASH_ATTN=true NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     MODEL_2="" MODEL_3="" MODEL_4="" MODEL_5="" MODEL_6="" MODEL_7="" MODEL_8="" \
#     OUTPUT_DIR=/root/invarlock-public/invarlock_validation_b200_7b_4gpus_$(date +%Y%m%d_%H%M%S) \
#     nohup ./scripts/b200_validation_suite.sh > /root/invarlock-public/b200_7b_4gpus_latest.log 2>&1 < /dev/null &'

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
fi

REPO_DIR="${REPO_DIR:-/root/invarlock-public}"
REPO_URL="${REPO_URL:-https://github.com/invarlock/invarlock.git}"
BRANCH="${BRANCH:-staging/next}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu128}"

export HF_HOME="${HF_HOME:-${REPO_DIR}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

install_system_deps() {
  log "Installing system dependencies (apt)..."
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git make jq tmux curl wget \
    build-essential ninja-build \
    python3.12 python3.12-venv python3.12-dev python3-pip
}

clone_or_update_repo() {
  if [[ ! -d "${REPO_DIR}/.git" ]]; then
    log "Cloning ${REPO_URL} to ${REPO_DIR}"
    git clone "${REPO_URL}" "${REPO_DIR}"
  fi

  cd "${REPO_DIR}"
  git fetch origin
  git checkout "${BRANCH}"
  git pull --ff-only origin "${BRANCH}"
}

setup_venv() {
  log "Creating/refreshing venv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  . "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
}

install_torch_b200() {
  log "Installing PyTorch nightly (cu128) for B200"
  # shellcheck disable=SC1091
  . "${VENV_DIR}/bin/activate"
  python -m pip install --pre --index-url "${TORCH_INDEX_URL}" torch --upgrade --force-reinstall
  python - <<'PY'
import torch
print("torch", torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available in torch")
print("cuda", torch.version.cuda)
print("gpus", torch.cuda.device_count())
print("gpu0", torch.cuda.get_device_name(0))
print("cc0", torch.cuda.get_device_capability(0))
PY
}

install_invarlock_stack() {
  log "Installing InvarLock + eval stack"
  # shellcheck disable=SC1091
  . "${VENV_DIR}/bin/activate"
  cd "${REPO_DIR}"

  python -m pip install -e ".[hf]"
  python -m pip install lm_eval accelerate sentencepiece protobuf safetensors
}

post_setup() {
  log "Ensuring validation scripts are executable"
  chmod +x "${REPO_DIR}/scripts/b200_validation_suite.sh"
}

main() {
  install_system_deps
  clone_or_update_repo
  setup_venv
  install_torch_b200
  install_invarlock_stack
  post_setup

  log "Setup complete."
  log "Run the validation suite with the ssh command in this script header."
}

main "$@"
