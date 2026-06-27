#!/usr/bin/env bash
# setup_remote.sh - Set up a GPU box for evidence pack runs.
# Usage:
#   scp scripts/evidence_packs/lib/core/setup_remote.sh root@<host>:/root/
#   ssh root@<host> 'chmod +x /root/setup_remote.sh && /root/setup_remote.sh'
#
# After setup, run:
#   ssh root@<host> 'cd /root/invarlock-public && . .venv/bin/activate && \
#     INVARLOCK_ALLOW_REMOTE_CODE=1 ./scripts/evidence_packs/run_pack.sh --suite subset --net 1'
#
# If you later validate a different clean worktree on the same host, either
# reinstall that worktree into its own venv or export PYTHONPATH=src so the
# evidence-pack commands use the intended checkout instead of an older editable
# install.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
fi

REPO_DIR="${REPO_DIR:-/root/invarlock-public}"
CANONICAL_REPO_ALIAS="${CANONICAL_REPO_ALIAS:-/root/invarlock-public}"
REPO_URL="${REPO_URL:-https://github.com/invarlock/invarlock.git}"
BRANCH="${BRANCH:-staging/next}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch}"
PACK_SKIP_TORCH_CHECK="${PACK_SKIP_TORCH_CHECK:-0}"
PACK_SKIP_RUNTIME_IMAGE_BUILD="${PACK_SKIP_RUNTIME_IMAGE_BUILD:-0}"
PACK_RUNTIME_IMAGE_FLAVOR="${PACK_RUNTIME_IMAGE_FLAVOR:-default}"

export HF_HOME="${HF_HOME:-${REPO_DIR}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

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

pack_evidence_pack_requirement_path() {
    local requirement_name="$1"
    echo "${REPO_DIR}/requirements/evidence-packs/${requirement_name}.txt"
}

pack_install_pinned_requirement() {
    local requirement_name="$1"
    shift
    local requirement_path
    requirement_path="$(pack_evidence_pack_requirement_path "${requirement_name}")"
    if [[ ! -f "${requirement_path}" ]]; then
        echo "ERROR: Missing pinned evidence-pack requirement file: ${requirement_path}" >&2
        return 1
    fi
    pack_run_cmd python -m pip install --require-hashes -r "${requirement_path}" "$@"
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

ensure_repo_alias() {
    if [[ -z "${CANONICAL_REPO_ALIAS:-}" || "${CANONICAL_REPO_ALIAS}" == "${REPO_DIR}" ]]; then
        return 0
    fi
    pack_run_cmd mkdir -p "$(dirname "${CANONICAL_REPO_ALIAS}")"
    pack_run_cmd ln -sfn "${REPO_DIR}" "${CANONICAL_REPO_ALIAS}"
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
        python "${REPO_DIR}/scripts/evidence_packs/python/runtime_tools.py" torch-env

        if command -v nvidia-smi >/dev/null 2>&1; then
            local gpu_name=""
            gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
            if [[ "${gpu_name}" == *"B200"* ]]; then
                python "${REPO_DIR}/scripts/evidence_packs/python/runtime_tools.py" torch-sm100-warning || true
            fi
        fi
    fi
}

install_invarlock_stack() {
    log "Installing InvarLock + evidence pack dependencies"
    pack_activate_venv
    cd "${REPO_DIR}"

    pack_run_cmd python -m pip install -e ".[hf]"
    pack_install_pinned_requirement "huggingface_hub"
    pack_install_pinned_requirement "accelerate" --no-deps
    pack_install_pinned_requirement "pyyaml"
    pack_install_pinned_requirement "protobuf"
    pack_install_pinned_requirement "sentencepiece"
}

pack_truthy_flag() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

pack_container_engine() {
    if command -v docker >/dev/null 2>&1; then
        echo "docker"
        return 0
    fi
    if command -v podman >/dev/null 2>&1; then
        echo "podman"
        return 0
    fi
    return 1
}

pack_runtime_image_flavor() {
    case "${PACK_RUNTIME_IMAGE_FLAVOR:-default}" in
        default|cuda|base)
            echo "default"
            ;;
        quant|cuda-quant)
            echo "quant"
            ;;
        *)
            echo "ERROR: unsupported PACK_RUNTIME_IMAGE_FLAVOR=${PACK_RUNTIME_IMAGE_FLAVOR}" >&2
            return 1
            ;;
    esac
}

pack_runtime_image_ref() {
    local flavor
    flavor="$(pack_runtime_image_flavor)" || return 1
    if [[ "${flavor}" == "quant" ]]; then
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            echo "ERROR: PACK_RUNTIME_IMAGE_FLAVOR=quant requires a CUDA host" >&2
            return 1
        fi
        echo "invarlock-runtime:cuda-quant"
        return 0
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "invarlock-runtime:cuda-local"
        return 0
    fi
    echo "invarlock-runtime:local"
}

pack_runtime_image_target() {
    local flavor
    flavor="$(pack_runtime_image_flavor)" || return 1
    if [[ "${flavor}" == "quant" ]]; then
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            echo "ERROR: PACK_RUNTIME_IMAGE_FLAVOR=quant requires a CUDA host" >&2
            return 1
        fi
        echo "runtime-image-cuda-quant"
        return 0
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "runtime-image-cuda"
        return 0
    fi
    echo "runtime-image"
}

ensure_runtime_image() {
    if [[ "${PACK_SKIP_RUNTIME_IMAGE_BUILD}" == "1" ]]; then
        log "Skipping local runtime image build (PACK_SKIP_RUNTIME_IMAGE_BUILD=1)"
        return 0
    fi
    if pack_truthy_flag "${INVARLOCK_ALLOW_HOST_EXECUTION:-0}"; then
        log "Skipping local runtime image build because host execution is enabled"
        return 0
    fi
    if [[ -n "${INVARLOCK_RUNTIME_IMAGE:-}" ]]; then
        log "Skipping local runtime image build because INVARLOCK_RUNTIME_IMAGE is already set (${INVARLOCK_RUNTIME_IMAGE})"
        return 0
    fi

    local engine=""
    if ! engine="$(pack_container_engine)"; then
        log "Skipping local runtime image build because no OCI container engine is installed"
        return 0
    fi

    local image_ref=""
    local target=""
    image_ref="$(pack_runtime_image_ref)" || return 1
    target="$(pack_runtime_image_target)" || return 1

    if "${engine}" image inspect "${image_ref}" >/dev/null 2>&1; then
        log "Local runtime image already present: ${image_ref}"
        export INVARLOCK_RUNTIME_IMAGE="${image_ref}"
        return 0
    fi

    log "Building local runtime image ${image_ref} via make ${target}"
    pack_activate_venv
    cd "${REPO_DIR}"
    pack_run_cmd make "${target}"
    export INVARLOCK_RUNTIME_IMAGE="${image_ref}"
}

verify_remote_stack() {
    log "Running evidence-pack remote smoke check"
    pack_activate_venv
    local -a smoke_args=(
        "${REPO_DIR}/scripts/evidence_packs/python/runtime_tools.py"
        remote-setup-smoke
    )
    if [[ "$(pack_runtime_image_flavor)" == "quant" ]]; then
        smoke_args+=(--module bitsandbytes --module gptqmodel --module hqq --module torchao)
    fi
    smoke_args+=(--repo-root "${REPO_DIR}")
    pack_run_cmd python "${smoke_args[@]}"
}

post_setup() {
    log "Ensuring evidence pack scripts are executable"
    pack_run_cmd chmod +x \
        "${REPO_DIR}/scripts/evidence_packs/run_suite.sh" \
        "${REPO_DIR}/scripts/evidence_packs/run_pack.sh" \
        "${REPO_DIR}/scripts/evidence_packs/verify_pack.sh" \
        "${REPO_DIR}/scripts/evidence_packs/run_mini_pack_gate.sh"
}

main() {
    install_system_deps
    clone_or_update_repo
    ensure_repo_alias
    setup_venv
    install_torch
    install_invarlock_stack
    ensure_runtime_image
    post_setup
    verify_remote_stack

    if [[ "${CANONICAL_REPO_ALIAS}" != "${REPO_DIR}" ]]; then
        log "Stable repo alias: ${CANONICAL_REPO_ALIAS} -> ${REPO_DIR}"
    fi
    log "Setup complete. Run INVARLOCK_ALLOW_REMOTE_CODE=1 ${REPO_DIR}/scripts/evidence_packs/run_pack.sh --suite subset --net 1"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
