#!/usr/bin/env bash
# setup_remote.sh - Set up a GPU box for proof pack runs.
# Usage:
#   scp scripts/proof_packs/lib/setup_remote.sh root@<host>:/root/
#   ssh root@<host> 'chmod +x /root/setup_remote.sh && /root/setup_remote.sh'
#
# After setup, run:
#   ssh root@<host> 'cd /root/invarlock-public && . .venv/bin/activate && \
#     ./scripts/proof_packs/run_pack.sh --suite subset --net 1'

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
fi

REPO_DIR="${REPO_DIR:-/root/invarlock-public}"
REPO_URL="${REPO_URL:-https://github.com/invarlock/invarlock.git}"
BRANCH="${BRANCH:-staging/next}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch}"
PACK_SKIP_TORCH_CHECK="${PACK_SKIP_TORCH_CHECK:-0}"

export HF_HOME="${HF_HOME:-${REPO_DIR}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

pack_run_cmd() {
    env "$@"
}

pack_activate_venv() {
    # shellcheck disable=SC1091
    . "${VENV_DIR}/bin/activate"
}

install_system_deps() {
    log "Installing system dependencies (apt)..."
    pack_run_cmd apt-get update
    pack_run_cmd DEBIAN_FRONTEND=noninteractive apt-get install -y \
        git make jq tmux curl wget \
        build-essential ninja-build \
        python3.12 python3.12-venv python3.12-dev python3-pip
}

clone_or_update_repo() {
    if [[ ! -d "${REPO_DIR}/.git" ]]; then
        log "Cloning ${REPO_URL} to ${REPO_DIR}"
        pack_run_cmd git clone "${REPO_URL}" "${REPO_DIR}"
    fi

    cd "${REPO_DIR}"
    pack_run_cmd git fetch origin
    pack_run_cmd git checkout "${BRANCH}"
    pack_run_cmd git pull --ff-only origin "${BRANCH}"
}

setup_venv() {
    log "Creating/refreshing venv at ${VENV_DIR}"
    pack_run_cmd "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    pack_activate_venv
    pack_run_cmd python -m pip install --upgrade pip setuptools wheel
}

install_torch() {
    log "Installing PyTorch (${TORCH_PACKAGES})"
    pack_activate_venv

    local -a packages
    read -r -a packages <<< "${TORCH_PACKAGES}"

    local -a cmd=(python -m pip install --upgrade --force-reinstall)
    if [[ -n "${TORCH_INDEX_URL}" ]]; then
        cmd+=(--index-url "${TORCH_INDEX_URL}")
    fi
    cmd+=("${packages[@]}")

    pack_run_cmd "${cmd[@]}"

    if [[ "${PACK_SKIP_TORCH_CHECK}" != "1" ]]; then
        python "${REPO_DIR}/scripts/proof_packs/python/torch_env_check.py"

        if command -v nvidia-smi >/dev/null 2>&1; then
            local gpu_name=""
            gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
            if [[ "${gpu_name}" == *"B200"* ]]; then
                python "${REPO_DIR}/scripts/proof_packs/python/torch_sm100_warning.py" || true
            fi
        fi
    fi
}

install_invarlock_stack() {
    log "Installing InvarLock + proof pack dependencies"
    pack_activate_venv
    cd "${REPO_DIR}"

    pack_run_cmd python -m pip install -e ".[hf]"
    pack_run_cmd python -m pip install accelerate sentencepiece protobuf safetensors
}

post_setup() {
    log "Ensuring proof pack scripts are executable"
    pack_run_cmd chmod +x \
        "${REPO_DIR}/scripts/proof_packs/run_suite.sh" \
        "${REPO_DIR}/scripts/proof_packs/run_pack.sh" \
        "${REPO_DIR}/scripts/proof_packs/verify_pack.sh"
}

main() {
    install_system_deps
    clone_or_update_repo
    setup_venv
    install_torch
    install_invarlock_stack
    post_setup

    log "Setup complete. Run ./scripts/proof_packs/run_pack.sh --suite subset --net 1"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
