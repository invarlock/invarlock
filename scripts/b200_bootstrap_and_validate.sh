#!/usr/bin/env bash
# b200_bootstrap_and_validate.sh - Complete B200 box setup and validation runner
# Version: 1.0.0
# Usage: scp this to B200 box, then: chmod +x b200_bootstrap_and_validate.sh && ./b200_bootstrap_and_validate.sh
#
# This script handles:
# 1. System dependencies installation
# 2. Python 3.12 venv setup with B200-compatible PyTorch (sm_100)
# 3. InvarLock and dependencies installation
# 4. Environment validation
# 5. Running the B200 validation suite with optimized settings

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
fi

# ============ CONFIGURATION ============
# Modify these as needed

WORK_DIR="${WORK_DIR:-$HOME/tests}"
VENV_DIR="${VENV_DIR:-${WORK_DIR}/.venv}"
LOG_FILE="${LOG_FILE:-${WORK_DIR}/setup.log}"

# PyTorch nightly with CUDA 12.8 support (includes sm_100 for B200)
TORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/cu128"

# HuggingFace caches (models + datasets). Defaulting to /root/.cache often fails
# on GPU nodes with small / partitions; keep caches under WORK_DIR by default.
export HF_HOME="${HF_HOME:-${WORK_DIR}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

# Ensure log directory exists before first log call (execution only)
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    mkdir -p "$(dirname "${LOG_FILE}")"
fi

# Validation script settings (optimized for lock contention reduction)
# GPU selection is auto-detected at runtime unless explicitly set:
# - If `CUDA_VISIBLE_DEVICES` is set, it is treated as the explicit physical GPU list.
# - Otherwise, the script detects all GPUs via nvidia-smi and uses them.
NUM_GPUS="${NUM_GPUS:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export GPU_REQUIRE_IDLE="${GPU_REQUIRE_IDLE:-false}"          # Reduces nvidia-smi calls
export SKIP_FLASH_ATTN="${SKIP_FLASH_ATTN:-true}"             # Skip flash_attn install
export WORKER_IDLE_SLEEP="${WORKER_IDLE_SLEEP:-10}"           # Reduce lock contention
export WORKER_HEARTBEAT_INTERVAL="${WORKER_HEARTBEAT_INTERVAL:-60}"
export GPU_MIN_FREE_GB="${GPU_MIN_FREE_GB:-10}"               # Memory threshold

# ============ HELPER FUNCTIONS ============

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}"
    echo "${msg}" >> "${LOG_FILE}"
}

log_section() {
    echo ""
    log "============================================"
    log "$*"
    log "============================================"
}

if ! declare -F _bootstrap_is_root >/dev/null 2>&1; then
    :
    _bootstrap_is_root() { [[ ${EUID} -eq 0 ]]; }
fi

check_root() {
    if _bootstrap_is_root; then
        log "WARNING: Running as root. Consider running as regular user."
    fi
}

_bootstrap_run_pkg() {
    if _bootstrap_is_root; then
        "$@"
        return $?
    fi
    if command -v sudo >/dev/null 2>&1; then
        sudo "$@"
        return $?
    fi
    log "ERROR: sudo is required to install system dependencies"
    return 1
}

# ============ PHASE 1: SYSTEM DEPENDENCIES ============

install_system_deps() {
    log_section "PHASE 1: Installing System Dependencies"

    if command -v apt-get &>/dev/null; then
        log "Detected Debian/Ubuntu system"
        if ! _bootstrap_run_pkg apt-get update || ! _bootstrap_run_pkg apt-get install -y \
            tmux \
            jq \
            python3.12 \
            python3.12-venv \
            python3.12-dev \
            build-essential \
            ninja-build \
            git \
            curl \
            wget; then
            return 1
        fi
    elif command -v yum &>/dev/null; then
        log "Detected RHEL/CentOS system"
        if ! _bootstrap_run_pkg yum install -y \
            tmux \
            jq \
            python3.12 \
            python3.12-devel \
            gcc \
            gcc-c++ \
            make \
            ninja-build \
            git \
            curl \
            wget; then
            return 1
        fi
    else
        log "ERROR: Unknown package manager. Please install dependencies manually."
        log "Required: tmux, jq, python3.12, python3.12-venv/devel, build-essential, ninja-build"
        return 1
    fi

    log "System dependencies installed successfully"
}

# ============ PHASE 2: PYTHON ENVIRONMENT ============

setup_python_venv() {
    log_section "PHASE 2: Setting up Python Virtual Environment"

    mkdir -p "${WORK_DIR}"
    cd "${WORK_DIR}"

    # Create venv if it doesn't exist
    if [[ ! -d "${VENV_DIR}" ]]; then
        log "Creating Python 3.12 virtual environment at ${VENV_DIR}"
        python3.12 -m venv "${VENV_DIR}"
    else
        log "Virtual environment already exists at ${VENV_DIR}"
    fi

    # Activate venv
    log "Activating virtual environment"
    source "${VENV_DIR}/bin/activate"

    # Upgrade pip
    log "Upgrading pip, setuptools, wheel"
    python -m pip install --upgrade pip setuptools wheel

    log "Python environment ready: $(python --version)"
}

# ============ PHASE 3: PYTORCH WITH B200 SUPPORT ============

install_pytorch_b200() {
    log_section "PHASE 3: Installing PyTorch with B200 (sm_100) Support"

    source "${VENV_DIR}/bin/activate"

    log "Installing PyTorch nightly with CUDA 12.8 (sm_100 support)"
    python -m pip install --pre \
        --index-url "${TORCH_INDEX_URL}" \
        torch \
        --upgrade \
        --force-reinstall

    # Verify installation
    log "Verifying PyTorch B200 compatibility..."
    python - <<'VERIFY_TORCH'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device 0: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

    arch_list = torch.cuda.get_arch_list()
    print(f"Supported architectures: {arch_list}")

    if 'sm_100' in arch_list or any('100' in a for a in arch_list):
        print("✓ B200 (sm_100) support confirmed!")
    else:
        print("⚠ WARNING: sm_100 not found in arch list. B200 may not be fully supported.")
        print("  This might still work with forward compatibility.")
else:
    print("ERROR: CUDA not available!")
    sys.exit(1)
VERIFY_TORCH

    log "PyTorch installation complete"
}

# ============ PHASE 4: INVARLOCK AND DEPENDENCIES ============

install_invarlock_deps() {
    log_section "PHASE 4: Installing InvarLock and Dependencies"

    source "${VENV_DIR}/bin/activate"

    # Core dependencies
    log "Installing core dependencies..."
    python -m pip install \
        invarlock \
        "transformers>=4.57.3" \
        lm_eval \
        pyyaml \
        protobuf \
        sentencepiece \
        datasets \
        accelerate \
        safetensors

    # Prefer local repo install when available (ensures local fixes are used)
    local invarlock_src="${INVARLOCK_SRC:-${WORK_DIR}}"
    if [[ -f "${invarlock_src}/pyproject.toml" && -d "${invarlock_src}/src/invarlock" ]]; then
        log "Detected local InvarLock repo at ${invarlock_src}, installing editable"
        python -m pip install -e "${invarlock_src}"
    fi

    # Optional: Install from requirements.txt if present (skip torch/flash_attn)
    local req_file="${WORK_DIR}/requirements.txt"
    if [[ -f "${req_file}" ]]; then
        log "Found requirements.txt, installing (excluding torch/flash_attn pins)..."
        # grep -v returns exit code 1 when it filters out all lines; don't treat that as fatal.
        local tmp_req=""
        tmp_req="$(mktemp "${TMPDIR:-/tmp}/req_filtered.XXXXXX" 2>/dev/null || printf '%s' "/tmp/req_filtered.$$")"
        grep -vE '^(torch|torchvision|torchaudio|flash_attn)==' "${req_file}" > "${tmp_req}" || true
        python -m pip install -r "${tmp_req}" || true
        rm -f "${tmp_req}"
    fi

    # Verify invarlock
    log "Verifying invarlock installation..."
    python -c "import invarlock; print(f'InvarLock version: {invarlock.__version__}')" || {
        log "WARNING: Could not import invarlock. It may be installed as CLI only."
    }

    # Check CLI
    if command -v invarlock &>/dev/null; then
        log "InvarLock CLI available: $(which invarlock)"
    else
        log "InvarLock CLI not in PATH, checking if installed..."
        python -m invarlock --help >/dev/null 2>&1 && log "InvarLock available via 'python -m invarlock'"
    fi

    log "Dependencies installation complete"

    # Apply runtime patches for known issues
    patch_invarlock_metrics_gather
}

# ============ PATCH: METRICS GATHER CLAMP (DEVICE-SIDE ASSERT GUARD) ============

patch_invarlock_metrics_gather() {
    log "Checking for metrics gather clamp fix..."

    # This patch prevents CUDA "device-side assert triggered" errors that can occur
    # when token IDs exceed the model's logits vocab dimension during perplexity
    # computation (gather). Those asserts poison the CUDA context for the process.

    local patch_rc=0
    python - <<'PATCH_METRICS' || patch_rc=$?
import importlib.util
import sys
from pathlib import Path

def _resolve_metrics_path():
    try:
        import invarlock.eval.metrics as m
        return Path(m.__file__)
    except Exception:
        spec = importlib.util.find_spec("invarlock.eval.metrics")
        if spec and spec.origin:
            return Path(spec.origin)
    return None

path = _resolve_metrics_path()
if not path:
    print("InvarLock not installed, skipping metrics patch")
    sys.exit(0)

try:
    text = path.read_text()
except Exception as e:
    print(f"Could not read {path}: {e}")
    sys.exit(0)

# Fix accidental literal "\n" sequences from older patch logic.
if "\\n        tgt = shift_labels.clamp(" in text or "shift_logits.size(-1)\\n" in text:
    text = text.replace("shift_logits.size(-1)\\n", "shift_logits.size(-1)\n")
    text = text.replace("\\n        tgt = shift_labels.clamp(", "\n        tgt = shift_labels.clamp(")
    try:
        path.write_text(text)
        print(f"Metrics gather de-escaped newline fix applied: {path}")
    except Exception as e:
        print(f"Could not write patch to {path}: {e}")
        print("You may need to run this with appropriate permissions or apply the patch manually.")
        sys.exit(1)
    # Re-read for patched checks below.
    text = path.read_text()

# Skip if a fixed implementation is already present (either upstream or patched).
if "clamp(min=0, max=vocab_size - 1)" in text or "_sanitize_token_ids_for_model" in text:
    print(f"Metrics gather already patched: {path}")
    sys.exit(0)

needle = "tgt = shift_labels.clamp_min(0).unsqueeze(-1)"
if needle not in text:
    print(f"Could not find gather clamp point in {path}, may need manual patching")
    sys.exit(0)

replacement = "vocab_size = shift_logits.size(-1)\n        tgt = shift_labels.clamp(min=0, max=vocab_size - 1).unsqueeze(-1)"
text = text.replace(needle, replacement)

try:
    path.write_text(text)
    print(f"Metrics gather patched successfully: {path}")
except Exception as e:
    print(f"Could not write patch to {path}: {e}")
    print("You may need to run this with appropriate permissions or apply the patch manually.")
    sys.exit(1)
PATCH_METRICS

    if [[ ${patch_rc} -eq 0 ]]; then
        log "Metrics gather patch check complete"
    else
        log "WARNING: Metrics gather patch could not be applied. Manual patching may be required."
    fi
}

# ============ GPU AUTO-DETECTION ============

configure_gpu_env() {
    # Requires: nvidia-smi available
    local source="nvidia-smi"
    local raw_list="${CUDA_VISIBLE_DEVICES:-}"
    local -a candidates=()

    if [[ -n "${raw_list}" ]]; then
        source="CUDA_VISIBLE_DEVICES"
        IFS=',' read -ra candidates <<< "${raw_list}"
    else
        while IFS= read -r id; do
            [[ -n "${id}" ]] && candidates+=("${id}")
        done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    fi

    local -a cleaned=()
    local id
    # Bash 3.2 + `set -u` treats `"${arr[@]}"` as unbound when empty; guard it.
    if [[ ${#candidates[@]} -gt 0 ]]; then
        for id in "${candidates[@]}"; do
            id=$(echo "${id}" | tr -d ' ')
            [[ -z "${id}" ]] && continue
            if ! [[ "${id}" =~ ^[0-9]+$ ]]; then
                log "ERROR: Non-numeric GPU id in ${source}: '${id}'"
                exit 1
            fi
            if ! nvidia-smi -i "${id}" &>/dev/null; then
                log "ERROR: GPU id '${id}' from ${source} is not valid on this host"
                exit 1
            fi
            cleaned+=("${id}")
        done
    fi

    if [[ ${#cleaned[@]} -eq 0 ]]; then
        log "ERROR: No GPUs detected (source=${source})"
        exit 1
    fi

    local requested="${NUM_GPUS:-}"
    if [[ -z "${requested}" ]]; then
        requested="${#cleaned[@]}"
    fi
    if ! [[ "${requested}" =~ ^[0-9]+$ ]]; then
        requested="${#cleaned[@]}"
    fi
    if [[ ${requested} -lt 1 ]]; then
        requested=1
    fi
    if [[ ${requested} -gt ${#cleaned[@]} ]]; then
        log "WARNING: NUM_GPUS=${requested} > available ${#cleaned[@]} from ${source}; clamping"
        requested=${#cleaned[@]}
    fi

    cleaned=("${cleaned[@]:0:${requested}}")
    NUM_GPUS="${#cleaned[@]}"
    CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${cleaned[*]}")

    export NUM_GPUS CUDA_VISIBLE_DEVICES
    export GPU_ID_LIST="${CUDA_VISIBLE_DEVICES}"  # Used by the B200 scheduler for GPU enumeration

    log "GPU pool configured from ${source}: NUM_GPUS=${NUM_GPUS}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
}

# ============ PHASE 5: ENVIRONMENT VALIDATION ============

validate_environment() {
    log_section "PHASE 5: Validating Environment"

    source "${VENV_DIR}/bin/activate"

    # Check GPU availability
    log "Checking GPU availability..."
    nvidia-smi || {
        log "ERROR: nvidia-smi not available. Is NVIDIA driver installed?"
        exit 1
    }

    # Check all expected GPUs
    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    log "Found ${gpu_count} GPUs"

    # Configure NUM_GPUS / CUDA_VISIBLE_DEVICES defaults from detected GPUs
    configure_gpu_env
    if [[ ${gpu_count} -lt ${NUM_GPUS} ]]; then
        log "WARNING: NUM_GPUS=${NUM_GPUS} > physical GPUs=${gpu_count} (unexpected after auto-detect)"
    fi

    # Check GPU memory
    log "GPU Memory Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

    # Check validation script exists
    local validation_script="${WORK_DIR}/b200_validation_suite.sh"
    if [[ -f "${validation_script}" ]]; then
        log "Validation script found: ${validation_script}"
    else
        log "WARNING: Validation script not found at ${validation_script}"
        log "Please copy b200_validation_suite.sh to ${WORK_DIR}/"
    fi

    # Check lib directory
    local lib_dir="${WORK_DIR}/lib"
    if [[ -d "${lib_dir}" ]]; then
        log "Library directory found: ${lib_dir}"
        ls -la "${lib_dir}/"*.sh 2>/dev/null | head -10 || true
    else
        log "WARNING: lib/ directory not found at ${lib_dir}"
        log "Please copy the lib/ directory to ${WORK_DIR}/"
    fi

    log "Environment validation complete"
}

# ============ PHASE 6: RUN VALIDATION ============

run_validation() {
    log_section "PHASE 6: Running B200 Validation Suite"

    source "${VENV_DIR}/bin/activate"
    cd "${WORK_DIR}"

    patch_invarlock_metrics_gather
    if ! python - <<'PY'
import invarlock.eval.metrics as m
from pathlib import Path

text = Path(m.__file__).read_text()
if "clamp(min=0, max=vocab_size - 1)" not in text and "_sanitize_token_ids_for_model" not in text:
    raise SystemExit("metrics gather patch NOT applied")
print("metrics gather patch verified")
PY
    then
        log "ERROR: metrics gather patch verification failed"
        exit 1
    fi

    local validation_script="${WORK_DIR}/b200_validation_suite.sh"

    if [[ ! -f "${validation_script}" ]]; then
        log "ERROR: Validation script not found: ${validation_script}"
        log "Please copy the validation script and lib/ directory first."
        exit 1
    fi

    if [[ ! -x "${validation_script}" ]]; then
        log "Making validation script executable..."
        chmod +x "${validation_script}"
    fi

    # Ensure HuggingFace caches are on a large filesystem (models + datasets).
    mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

    log "Starting validation with optimized settings:"
    log "  NUM_GPUS=${NUM_GPUS}"
    log "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    log "  GPU_REQUIRE_IDLE=${GPU_REQUIRE_IDLE}"
    log "  SKIP_FLASH_ATTN=${SKIP_FLASH_ATTN}"
    log "  WORKER_IDLE_SLEEP=${WORKER_IDLE_SLEEP}"
    log "  WORKER_HEARTBEAT_INTERVAL=${WORKER_HEARTBEAT_INTERVAL}"
    log "  HF_HOME=${HF_HOME}"
    log "  HF_HUB_CACHE=${HF_HUB_CACHE}"
    log "  HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
    log "  TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"

    # Run with tee to capture output
    local run_log="${WORK_DIR}/run_b200_$(date +%Y%m%d_%H%M%S).log"
    log "Output will be logged to: ${run_log}"

    # Export all settings
    export NUM_GPUS CUDA_VISIBLE_DEVICES GPU_REQUIRE_IDLE SKIP_FLASH_ATTN
    export WORKER_IDLE_SLEEP WORKER_HEARTBEAT_INTERVAL GPU_MIN_FREE_GB

    # Run the validation (allow failure so we can still run post-run diagnostics)
    set +e
    "${validation_script}" 2>&1 | tee "${run_log}"
    local exit_code=${PIPESTATUS[0]}
    set -e

    if [[ ${exit_code} -eq 0 ]]; then
        log "Validation completed successfully!"
    else
        log "Validation completed with exit code: ${exit_code}"
    fi

    return ${exit_code}
}

# ============ PHASE 7: POST-RUN DIAGNOSTICS ============

post_run_diagnostics() {
    log_section "PHASE 7: Post-Run Diagnostics"

    source "${VENV_DIR}/bin/activate"
    cd "${WORK_DIR}"

    # Find latest output directory
    local latest_output
    latest_output="$(ls -td invarlock_validation_b200_* 2>/dev/null | head -1 || true)"

    if [[ -z "${latest_output}" ]]; then
        log "No output directory found"
        return 0
    fi

    log "Analyzing output directory: ${latest_output}"

    # Check worker PIDs for uniqueness
    log "Worker PID Analysis:"
    if [[ -d "${latest_output}/workers" ]]; then
        log "  PID files:"
        for f in "${latest_output}/workers"/gpu_*.pid; do
            [[ -f "${f}" ]] && echo "    $(basename ${f}): $(cat ${f})"
        done

        local unique_pids
        unique_pids="$(cat "${latest_output}/workers"/gpu_*.pid 2>/dev/null | sort -u | wc -l | tr -d ' ' || echo "0")"
        log "  Unique PIDs: ${unique_pids}"
    fi

    # Check for lock errors
    log "Lock Error Summary:"
    local lock_errors
    lock_errors=$(grep -c "Failed to acquire" "${latest_output}/logs/"*.log 2>/dev/null || echo "0")
    log "  Total lock acquisition failures: ${lock_errors}"

    # Check task completion
    log "Task Completion Summary:"
    if [[ -d "${latest_output}/queue" ]]; then
        local completed
        completed="$(ls "${latest_output}/queue/completed/"*.task 2>/dev/null | wc -l | tr -d ' ' || echo "0")"
        local failed
        failed="$(ls "${latest_output}/queue/failed/"*.task 2>/dev/null | wc -l | tr -d ' ' || echo "0")"
        local pending
        pending="$(ls "${latest_output}/queue/pending/"*.task 2>/dev/null | wc -l | tr -d ' ' || echo "0")"
        local ready
        ready="$(ls "${latest_output}/queue/ready/"*.task 2>/dev/null | wc -l | tr -d ' ' || echo "0")"
        log "  Completed: ${completed}"
        log "  Failed: ${failed}"
        log "  Pending: ${pending}"
        log "  Ready: ${ready}"
    fi

    # Check for OOM errors
    log "OOM Error Check:"
    local oom_count
    oom_count=$(grep -c "CUDA out of memory" "${latest_output}/logs/"*.log 2>/dev/null || echo "0")
    log "  OOM occurrences: ${oom_count}"

    log "Diagnostics complete"
}

# ============ MAIN EXECUTION ============

main() {
    local mode="${1:-full}"
    local overall_rc=0

    mkdir -p "${WORK_DIR}"
    cd "${WORK_DIR}"

    # Initialize log file
    echo "=== B200 Setup and Run Log ===" > "${LOG_FILE}"
    echo "Started: $(date)" >> "${LOG_FILE}"
    echo "Mode: ${mode}" >> "${LOG_FILE}"

    check_root

    case "${mode}" in
        "setup-only")
            # Just setup, don't run validation
            install_system_deps
            setup_python_venv
            install_pytorch_b200
            install_invarlock_deps
            validate_environment
            log_section "SETUP COMPLETE"
            log "To run validation, execute: ./b200_bootstrap_and_validate.sh run-only"
            ;;
        "run-only")
            # Just run validation (assumes setup done)
            validate_environment
            local validation_rc=0
            run_validation || validation_rc=$?
            post_run_diagnostics || true
            overall_rc=${validation_rc}
            ;;
        "diagnostics")
            # Just run diagnostics
            post_run_diagnostics
            ;;
        "full"|*)
            # Full setup and run
            install_system_deps
            setup_python_venv
            install_pytorch_b200
            install_invarlock_deps
            validate_environment
            local validation_rc=0
            run_validation || validation_rc=$?
            post_run_diagnostics || true
            overall_rc=${validation_rc}
            ;;
    esac

    log_section "ALL PHASES COMPLETE"
    log "Log file: ${LOG_FILE}"
    return ${overall_rc}
}

# ============ SCRIPT ENTRY POINT ============

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    # Show usage if --help
    if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
        cat << USAGE
B200 Setup and Validation Runner

Usage: $0 [mode]

Modes:
  full         - Complete setup and validation run (default)
  setup-only   - Install dependencies and setup environment only
  run-only     - Run validation only (assumes setup complete)
  diagnostics  - Run post-validation diagnostics only

Environment Variables:
  WORK_DIR                    - Working directory (default: ~/tests)
  NUM_GPUS                    - Number of GPUs to use (default: auto-detect all)
  CUDA_VISIBLE_DEVICES        - GPU IDs to use (default: auto-detect all)
  GPU_REQUIRE_IDLE            - Require GPUs to be idle (default: false)
  SKIP_FLASH_ATTN             - Skip flash_attn installation (default: true)
  WORKER_IDLE_SLEEP           - Worker idle sleep seconds (default: 10)
  WORKER_HEARTBEAT_INTERVAL   - Heartbeat interval seconds (default: 60)
  GPU_MIN_FREE_GB             - Minimum free GPU memory in GB (default: 10)
  INVARLOCK_SRC               - Local InvarLock repo path for editable install (default: WORK_DIR)

Examples:
  # Full setup and run
  ./b200_bootstrap_and_validate.sh

  # Setup only (useful for tmux sessions)
  ./b200_bootstrap_and_validate.sh setup-only

  # Run validation after setup
  ./b200_bootstrap_and_validate.sh run-only

  # Check results
  ./b200_bootstrap_and_validate.sh diagnostics
USAGE
        exit 0
    fi

    # Run main
    main "${1:-full}"
fi
