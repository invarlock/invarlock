#!/usr/bin/env bash
# validation_runtime.sh - Logging, GPU, disk, dependency, and model setup helpers
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Usage: sourced by validation_suite.sh

log() {
    # Thread-safe logging using flock for parallel processes
    {
        flock -w 5 200 2>/dev/null || true  # Wait up to 5s for lock, continue anyway
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
    } 200>"${LOG_LOCK}"
}

log_section() {
    echo "" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "$*" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"
}

error_exit() {
    # Output to stderr to avoid polluting stdout (important for functions returning values via echo)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >> "${LOG_FILE}"
    exit 1
}

sanitize_model_name() {
    local model_id="$1"
    echo "${model_id}" \
        | tr '[:upper:]' '[:lower:]' \
        | sed 's#/#__#g' \
        | tr ' ' '_' \
        | tr -cd '[:alnum:]_-'
}

# ============ GPU SELECTION / TOPOLOGY ============
# Stable pool of physical GPU indices used by this run.
# - If CUDA_VISIBLE_DEVICES is set, it is treated as an explicit physical GPU list.
# - Otherwise, we detect all GPUs via nvidia-smi and use them.
#
# NOTE: Workers/tasks will override CUDA_VISIBLE_DEVICES per-task (single- or multi-GPU),
# so we keep the pool in GPU_ID_LIST for scheduler enumeration.
GPU_ID_LIST="${GPU_ID_LIST:-}"

# Print newline-separated GPU IDs for this run.
# Defaults to 0..NUM_GPUS-1 when GPU_ID_LIST isn't set yet.
list_run_gpu_ids() {
    if [[ -n "${GPU_ID_LIST:-}" ]]; then
        echo "${GPU_ID_LIST}" | tr -d ' ' | tr ',' '\n' | sed '/^$/d'
    else
        local total="${NUM_GPUS:-8}"
        if ! [[ "${total}" =~ ^[0-9]+$ ]]; then
            total=8
        fi
        [[ ${total} -lt 1 ]] && total=1
        seq 0 $((total - 1))
    fi
}

configure_gpu_pool() {
    # Identify candidate GPU IDs
    local source="nvidia-smi"
    local raw_list=""
    local -a candidates=()

    # Prefer CUDA_VISIBLE_DEVICES if set (Slurm/Ray commonly set this).
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        source="CUDA_VISIBLE_DEVICES"
        raw_list="${CUDA_VISIBLE_DEVICES}"
    elif [[ -n "${GPU_ID_LIST:-}" ]]; then
        # Fallback: allow callers to set GPU_ID_LIST directly.
        source="GPU_ID_LIST"
        raw_list="${GPU_ID_LIST}"
    fi

    if [[ -n "${raw_list}" ]]; then
        IFS=',' read -ra candidates <<< "${raw_list}"
    else
        while IFS= read -r id; do
            [[ -n "${id}" ]] && candidates+=("${id}")
        done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    fi

    # Sanitize and validate IDs
    local -a cleaned=()
    local id
    for id in "${candidates[@]}"; do
        id=$(echo "${id}" | tr -d ' ')
        [[ -z "${id}" ]] && continue
        if ! [[ "${id}" =~ ^[0-9]+$ ]]; then
            error_exit "Non-numeric GPU id in ${source}: '${id}'. Set CUDA_VISIBLE_DEVICES to numeric indices."
        fi
        if ! nvidia-smi -i "${id}" &>/dev/null; then
            error_exit "GPU id '${id}' from ${source} is not valid on this host."
        fi
        cleaned+=("${id}")
    done

    if [[ ${#cleaned[@]} -eq 0 ]]; then
        error_exit "No usable GPU ids found (${source})."
    fi

    # Determine how many GPUs to use.
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

    local -a selected=("${cleaned[@]:0:${requested}}")
    GPU_ID_LIST=$(IFS=','; echo "${selected[*]}")
    export GPU_ID_LIST
    export NUM_GPUS="${#selected[@]}"

    # Normalize CUDA_VISIBLE_DEVICES so torch + subprocesses see the same pool.
    export CUDA_VISIBLE_DEVICES="${GPU_ID_LIST}"

    log "GPU pool configured from ${source}: NUM_GPUS=${NUM_GPUS}, GPU_ID_LIST=${GPU_ID_LIST}"
}

# ============ DISK PRESSURE HARDENING ============
# Abort early under low disk to avoid half-written artifacts and cascading failures.
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-200}"

format_gb_as_tb() {
    local gb="$1"
    if [[ -z "${gb}" || ! "${gb}" =~ ^[0-9]+$ ]]; then
        echo ""
        return 0
    fi
    awk -v gb="${gb}" 'BEGIN { printf "%.1f", gb / 1024.0 }'
}

get_free_disk_gb() {
    local path="$1"
    local free_disk
    free_disk=$(df -BG "${path}" 2>/dev/null | awk 'NR==2 {gsub(/G/,""); print $4}')
    [[ -z "${free_disk}" || ! "${free_disk}" =~ ^[0-9]+$ ]] && return 1
    echo "${free_disk}"
}

estimate_model_weights_gb() {
    local model_id="$1"
    [[ -z "${model_id}" ]] && return 1
    if [[ -d "${model_id}" ]]; then
        return 1  # Unknown for local paths without a profile.
    fi
    local lower
    lower="$(echo "${model_id}" | tr '[:upper:]' '[:lower:]')"

    # Special-case MoE naming.
    if [[ "${lower}" == *"mixtral"* || "${lower}" == *"8x7b"* ]]; then
        echo 90
        return 0
    fi

    case "${lower}" in
        *"72b"*)
            echo 144
            ;;
        *"70b"*)
            echo 140
            ;;
        *"34b"*)
            echo 68
            ;;
        *"32b"*)
            echo 64
            ;;
        *"14b"*)
            echo 28
            ;;
        *"13b"*)
            echo 26
            ;;
        *"7b"*)
            echo 14
            ;;
        *)
            return 1
            ;;
    esac
}

estimate_planned_model_storage_gb() {
    local -a models=()
    if command -v mapfile >/dev/null 2>&1; then
        mapfile -t models < <(pack_model_list)
    else
        while IFS= read -r model; do
            [[ -n "${model}" ]] || continue
            models+=("${model}")
        done < <(pack_model_list)
    fi

    local edit_counts=""
    edit_counts="$(pack_count_edit_scenarios)"
    local edits_clean=0
    local edits_stress=0
    IFS='|' read -r edits_clean edits_stress _ <<< "${edit_counts}"
    if ! [[ "${edits_clean}" =~ ^[0-9]+$ ]]; then
        edits_clean=0
    fi
    if ! [[ "${edits_stress}" =~ ^[0-9]+$ ]]; then
        edits_stress=0
    fi
    local edits_total=$((edits_clean + edits_stress))
    local errors_total=0
    if [[ "${RUN_ERROR_INJECTION}" == "true" ]]; then
        # Derive count from scenarios.json when available to keep disk estimates aligned with the suite.
        local pack_root
        pack_root="$(cd "${_PACK_VALIDATION_LIB_DIR}/.." && pwd)"
        local scenarios_file="${pack_root}/scenarios.json"
        if [[ -f "${scenarios_file}" ]]; then
            errors_total="$(_pack_validation_state count-generation-kind "${scenarios_file}" --kind error 2>/dev/null || true)"
        fi
        if [[ -z "${errors_total}" || ! "${errors_total}" =~ ^[0-9]+$ ]]; then
            errors_total=9  # fallback for legacy scenario sets
        fi
    fi

    # When cleanup is enabled, edited/error models are removed after evaluation; estimate peak usage instead of total copies.
    if [[ "${PACK_CLEANUP_MODELS:-1}" != "0" ]]; then
        # In cleanup mode, default is per-edit unless explicitly forced to batch edits.
        local use_batch_edits="${PACK_USE_BATCH_EDITS:-}"
        local edits_peak=0
        local clean_runs="${CLEAN_EDIT_RUNS:-0}"
        local stress_runs="${STRESS_EDIT_RUNS:-0}"
        if ! [[ "${clean_runs}" =~ ^-?[0-9]+$ ]]; then
            clean_runs=0
        fi
        if ! [[ "${stress_runs}" =~ ^-?[0-9]+$ ]]; then
            stress_runs=0
        fi
        if [[ ${clean_runs} -gt 0 || ${stress_runs} -gt 0 ]] && [[ ${edits_total} -gt 0 ]]; then
            case "${use_batch_edits}" in
                1|true|yes|on)
                    edits_peak="${edits_total}"
                    ;;
                *)
                    edits_peak=1
                    ;;
            esac
        fi
        local errors_peak=0
        if [[ ${errors_total} -gt 0 ]]; then
            errors_peak=1
        fi
        edits_total="${edits_peak}"
        errors_total="${errors_peak}"
    fi

    local baseline_mode="${PACK_BASELINE_STORAGE_MODE:-snapshot_symlink}"
    local baseline_copy=1
    if [[ "${baseline_mode}" == "snapshot_symlink" ]]; then
        baseline_copy=0  # baseline dir is a symlink tree backed by HF hub cache blobs
    fi

    local hub_cache_on_output_fs=1
    if [[ -n "${HF_HUB_CACHE:-}" ]]; then
        local out_dev=""
        local hub_dev=""
        out_dev=$(df -P "${OUTPUT_DIR}" 2>/dev/null | awk 'NR==2 {print $1}' || true)
        hub_dev=$(df -P "${HF_HUB_CACHE}" 2>/dev/null | awk 'NR==2 {print $1}' || true)
        if [[ -n "${out_dev}" && -n "${hub_dev}" && "${out_dev}" != "${hub_dev}" ]]; then
            hub_cache_on_output_fs=0
        fi
    fi

    local total_gb=0
    local unknown=0
    local model_id
    for model_id in "${models[@]}"; do
        [[ -n "${model_id}" ]] || continue

        local w_gb=""
        w_gb="$(estimate_model_weights_gb "${model_id}" 2>/dev/null || true)"
        if [[ -z "${w_gb}" || ! "${w_gb}" =~ ^[0-9]+$ ]]; then
            unknown=$((unknown + 1))
            continue
        fi

        # Storage copies:
        # - 1× HF hub cache download (when model_id is remote)
        # - 1× baseline saved under OUTPUT_DIR (unless snapshot_symlink mode)
        # - N× edits (currently saved as full bf16 copies)
        # - M× error models (also full copies) when enabled
        local hub_copy=1
        [[ -d "${model_id}" ]] && hub_copy=0
        [[ ${hub_cache_on_output_fs} -eq 0 ]] && hub_copy=0
        local effective_baseline_copy=${baseline_copy}
        [[ -d "${model_id}" ]] && effective_baseline_copy=0
        local copies=$((hub_copy + effective_baseline_copy + edits_total + errors_total))

        total_gb=$((total_gb + (w_gb * copies)))
    done

    [[ ${unknown} -gt 0 ]] && return 1
    echo "${total_gb}"
}

disk_preflight() {
    [[ "${PACK_SKIP_DISK_PREFLIGHT:-0}" == "1" ]] && return 0

    local free_gb=""
    free_gb=$(get_free_disk_gb "${OUTPUT_DIR}" 2>/dev/null || echo "")
    if [[ -z "${free_gb}" ]]; then
        if [[ "${PACK_RELEASE_REVIEW:-0}" == "1" ]]; then
            error_exit "Release-review disk preflight could not determine free disk for OUTPUT_DIR=${OUTPUT_DIR}."
        fi
        return 0
    fi

    local planned_gb=""
    planned_gb=$(estimate_planned_model_storage_gb 2>/dev/null || echo "")
    if [[ -z "${planned_gb}" ]]; then
        if [[ "${PACK_RELEASE_REVIEW:-0}" == "1" ]]; then
            error_exit "Release-review disk preflight could not estimate planned model storage."
        fi
        return 0
    fi

    local min_free="${MIN_FREE_DISK_GB:-200}"
    if ! [[ "${min_free}" =~ ^[0-9]+$ ]]; then
        min_free=200
    fi

    local required_gb=$((planned_gb + min_free))
    if [[ ${free_gb} -ge ${required_gb} ]]; then
        return 0
    fi

    log_section "ABORTING: DISK PREFLIGHT"
    log "ERROR: Estimated storage for this configuration: ~${planned_gb}GB (~$(format_gb_as_tb "${planned_gb}")TB) for model weights alone."
    log "       Free disk on output filesystem: ${free_gb}GB (~$(format_gb_as_tb "${free_gb}")TB)."
    log "       This suite saves full bf16 copies of edits (+ error models if enabled)."
    log "       Baseline storage mode: ${PACK_BASELINE_STORAGE_MODE:-snapshot_symlink}."
    log "       snapshot_symlink now builds a cache-backed symlink tree into HF cache; it still needs one full model copy in HF cache when that cache shares the output filesystem."
    log "       Fix: mount a larger volume and set OUTPUT_DIR, or run the subset suite, or set RUN_ERROR_INJECTION=false."
    log "       Override (not recommended): PACK_SKIP_DISK_PREFLIGHT=1"

    if [[ "${PACK_RELEASE_REVIEW:-0}" == "1" ]]; then
        error_exit "Insufficient disk for release-review run (need >= ${required_gb}GB incl MIN_FREE_DISK_GB=${min_free})."
    fi

    # Resume mode may already have artifacts; allow non-release runs to proceed
    # if explicitly resuming.
    if [[ "${RESUME_FLAG:-false}" == "true" ]]; then
        log "WARNING: --resume mode enabled; continuing despite preflight estimate."
        return 0
    fi

    error_exit "Insufficient disk for planned run (need >= ${required_gb}GB incl MIN_FREE_DISK_GB=${min_free})."
}

write_disk_pressure_state() {
    local free_gb="$1"
    local min_gb="$2"
    mkdir -p "${OUTPUT_DIR}/state" 2>/dev/null || true
    _pack_validation_state write-disk-pressure \
        --path "${OUTPUT_DIR}/state/disk_pressure.json" \
        --free-gb "${free_gb}" \
        --min-gb "${min_gb}" \
        --output-dir "${OUTPUT_DIR}"
}

handle_disk_pressure() {
    local free_gb="$1"
    local min_gb="$2"

    log_section "ABORTING: DISK PRESSURE"
    log "ERROR: Low disk space in output filesystem: ${free_gb}GB free (< ${min_gb}GB)."
    log "       Free disk space and resume with: OUTPUT_DIR=${OUTPUT_DIR} $0 --resume"
    log "       (Override threshold: MIN_FREE_DISK_GB=0 to disable, or set a smaller value)"

    write_disk_pressure_state "${free_gb}" "${min_gb}"

    # Stop workers and aggressively stop running tasks so they don't keep writing.
    if type signal_shutdown &>/dev/null; then
        signal_shutdown "${OUTPUT_DIR}"
    else
        touch "${OUTPUT_DIR}/workers/SHUTDOWN"
    fi

    # Kill task process groups and move running tasks back to pending for resume.
    # Guard for early failures before the queue is initialized.
    if [[ -n "${QUEUE_DIR:-}" && -d "${QUEUE_DIR}" ]]; then
        local gpu_id
        for gpu_id in $(list_run_gpu_ids); do
            reclaim_orphaned_tasks "${gpu_id}" >> "${LOG_FILE}" 2>&1 || true
        done
    fi

    error_exit "Aborted due to disk pressure (free ${free_gb}GB < ${min_gb}GB)."
}

# ============ GPU ENVIRONMENT SETUP ============
setup_pack_environment() {
    log_section "PHASE 0: GPU ENVIRONMENT SETUP"

    local env_report
    local repo_root
    repo_root="$(_pack_validation_repo_root)"
    env_report=$(python3 "${repo_root}/scripts/evidence_packs/python/runtime_tools.py" env-report)
    local setup_rc=$?
    if [[ ${setup_rc} -ne 0 ]]; then
        printf '%s\n' "${env_report}"
        return ${setup_rc}
    fi
    printf '%s\n' "${env_report}"

    PACK_GPU_NAME=$(printf '%s\n' "${env_report}" | sed -n 's/^\[PACK_GPU_NAME=//p' | sed 's/\]$//' | tail -1)
    PACK_GPU_MEM_GB=$(printf '%s\n' "${env_report}" | sed -n 's/^\[PACK_GPU_MEM_GB=//p' | sed 's/\]$//' | tail -1)
    PACK_GPU_COUNT=$(printf '%s\n' "${env_report}" | sed -n 's/^\[PACK_GPU_COUNT=//p' | sed 's/\]$//' | tail -1)
    if [[ "${env_report}" == *"[FP8_NATIVE_SUPPORT=true]"* ]]; then
        export FP8_NATIVE_SUPPORT="true"
    else
        export FP8_NATIVE_SUPPORT="false"
    fi
    if [[ -n "${PACK_GPU_MEM_GB}" && -z "${GPU_MEMORY_GB}" ]]; then
        GPU_MEMORY_GB="${PACK_GPU_MEM_GB}"
    fi
    export PACK_GPU_NAME PACK_GPU_MEM_GB PACK_GPU_COUNT GPU_MEMORY_GB
    log "GPU Environment Setup: Complete (FP8_NATIVE_SUPPORT=${FP8_NATIVE_SUPPORT})"
}

# ============ DEPENDENCY CHECK ============
pack_evidence_pack_requirement_path() {
    local requirement_name="$1"
    local repo_root
    repo_root="$(_pack_validation_repo_root)"
    echo "${repo_root}/requirements/evidence-packs/${requirement_name}.txt"
}

pack_install_pinned_requirement() {
    local requirement_name="$1"
    shift
    local requirement_path
    requirement_path="$(pack_evidence_pack_requirement_path "${requirement_name}")"
    if [[ ! -f "${requirement_path}" ]]; then
        log "ERROR: pinned requirement file missing: ${requirement_path}"
        return 1
    fi
    python3 -m pip install --require-hashes -r "${requirement_path}" "$@"
}

pack_configure_pinned_cuda_nvcc() {
    local cuda_home
    cuda_home="$(
        python3 - <<'PY'
from __future__ import annotations

import site
import sysconfig
from pathlib import Path

roots: list[Path] = []
for raw in site.getsitepackages():
    roots.append(Path(raw))
purelib = sysconfig.get_paths().get("purelib")
if purelib:
    roots.append(Path(purelib))
for root in roots:
    candidates = [root / "nvidia" / "cuda_nvcc"]
    candidates.extend(sorted((root / "nvidia").glob("cu*")))
    for candidate in candidates:
        if (candidate / "bin" / "nvcc").is_file():
            print(candidate)
            raise SystemExit(0)
PY
    )"
    if [[ -z "${cuda_home}" ]]; then
        log "WARNING: pinned cuda-nvcc installed but nvcc was not found, using existing CUDA toolkit"
        return 1
    fi
    export CUDA_HOME="${cuda_home}"
    export CUDA_PATH="${cuda_home}"
    export PATH="${cuda_home}/bin:${PATH}"
    log "CUDA nvcc: using pinned compiler from CUDA nvcc package"
}

pack_prepare_flash_attn_build_toolchain() {
    if [[ "${PACK_NET}" != "1" ]]; then
        return 0
    fi
    if [[ "${1:-}" != "true" ]]; then
        return 0
    fi

    local cuda_nvcc_requirement
    cuda_nvcc_requirement="$(pack_evidence_pack_requirement_path "cuda-nvcc")"
    if [[ ! -f "${cuda_nvcc_requirement}" ]]; then
        log "WARNING: pinned cuda-nvcc requirement file missing, using existing CUDA toolkit"
        return 0
    fi

    log "Installing CUDA nvcc for flash-attn build..."
    if pack_install_pinned_requirement "cuda-nvcc" --no-deps >> "${LOG_FILE}" 2>&1; then
        pack_configure_pinned_cuda_nvcc || true
    else
        log "WARNING: cuda-nvcc install failed, using existing CUDA toolkit"
    fi
}

pack_try_install_flash_attn() {
    local flash_attn_requirement="$1"
    local -a pip_args=(
        python3 -m pip install
        --require-hashes
        -r "${flash_attn_requirement}"
        --no-deps
    )
    case "${PACK_FLASH_ATTN_ALLOW_SOURCE_BUILD:-0}" in
        1|true|TRUE|yes|YES|on|ON)
            pip_args+=(--no-build-isolation)
            ;;
        *)
            pip_args+=(--only-binary=:all:)
            ;;
    esac
    local old_opts="$-"
    set +e
    timeout 600 "${pip_args[@]}" >> "${LOG_FILE}" 2>&1
    local rc=$?
    case "${old_opts}" in
        *e*) set -e ;;
        *) set +e ;;
    esac
    return "${rc}"
}

check_dependencies() {
    log_section "PHASE 0: DEPENDENCY CHECK"

    local missing=()
    local pip_available="true"

    # Check Python
    command -v python3 >/dev/null 2>&1 || missing+=("python3")

    if [[ "${PACK_NET}" == "1" ]]; then
        if ! python3 -m pip --version >/dev/null 2>&1; then
            pip_available="false"
            # Try to bootstrap pip when available (common on Debian/Ubuntu images).
            if python3 -m ensurepip --upgrade >/dev/null 2>&1; then
                if python3 -m pip --version >/dev/null 2>&1; then
                    pip_available="true"
                fi
            fi
        fi
        if [[ "${pip_available}" != "true" ]]; then
            missing+=("pip")
            log "ERROR: python3 -m pip is not available."
            log "       Install python3-pip (or use a virtualenv) before running evidence packs with --net 1."
        fi
    fi

    # Check PyTorch with CUDA
    python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null || missing+=("torch+cuda")

    # Check transformers
    python3 -c "import transformers; print(f'Transformers {transformers.__version__}')" 2>/dev/null || missing+=("transformers")

    # Check huggingface_hub (required by --net 1 preflight/downloads)
    if [[ "${PACK_NET}" == "1" ]]; then
        if ! python3 -c "import huggingface_hub" 2>/dev/null; then
            log "Installing huggingface_hub..."
            if [[ "${pip_available}" != "true" ]]; then
                missing+=("huggingface_hub")
            elif ! pack_install_pinned_requirement "huggingface_hub"; then
                missing+=("huggingface_hub")
            fi
        fi
    fi

    # Check accelerate (required by transformers for device_map="auto")
    if ! python3 -c "import accelerate" 2>/dev/null; then
        if [[ "${PACK_NET}" == "1" ]]; then
            log "Installing accelerate..."
            if [[ "${pip_available}" == "true" ]]; then
                if pack_install_pinned_requirement "accelerate" --no-deps \
                    && python3 -c "import accelerate" 2>/dev/null; then
                    :
                else
                    missing+=("accelerate")
                fi
            else
                missing+=("accelerate")
            fi
        else
            missing+=("accelerate")
        fi
    fi

    # Check for flash-attn
    if python3 -c "import flash_attn; print('Flash Attention OK')" 2>/dev/null; then
        export FLASH_ATTENTION_AVAILABLE="true"
        log "Flash Attention 2: Available"
    else
        if [[ "${SKIP_FLASH_ATTN:-false}" == "true" ]]; then
            export FLASH_ATTENTION_AVAILABLE="false"
            log "Flash Attention 2: Skipped (SKIP_FLASH_ATTN=true)"
        else
            # Check if Python development headers are available (required for flash-attn build)
            local has_python_dev="false"
            if python3 -c "import sysconfig; exit(0 if sysconfig.get_config_var('INCLUDEPY') else 1)" 2>/dev/null; then
                local python_include=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))")
                if [[ -f "${python_include}/Python.h" ]]; then
                    has_python_dev="true"
                fi
            fi

            if [[ "${has_python_dev}" != "true" ]]; then
                export FLASH_ATTENTION_AVAILABLE="false"
                log "WARNING: Python development headers not found (Python.h missing)"
                log "         To enable flash-attn, install: apt-get install python3-dev  (or python3.X-dev)"
                log "         Or set SKIP_FLASH_ATTN=true to suppress this warning"
                log "         Continuing with eager attention (may be slower)"
            else
                if [[ "${PACK_NET}" != "1" ]]; then
                    export FLASH_ATTENTION_AVAILABLE="false"
                    log "Flash Attention 2: Not found (offline), using eager attention"
                else
                    log "Flash Attention 2: Not found, attempting install..."
                    local flash_attn_requirement
                    flash_attn_requirement="$(pack_evidence_pack_requirement_path "flash-attn")"
                    pack_prepare_flash_attn_build_toolchain "${pip_available}"
                    # Use timeout to prevent hanging on slow builds
                    if [[ "${pip_available}" == "true" ]] && [[ -f "${flash_attn_requirement}" ]] && pack_try_install_flash_attn "${flash_attn_requirement}"; then
                        # Verify it actually imported
                        if python3 -c "import flash_attn" 2>/dev/null; then
                            export FLASH_ATTENTION_AVAILABLE="true"
                            log "Flash Attention 2: Installed successfully"
                        else
                            export FLASH_ATTENTION_AVAILABLE="false"
                            log "WARNING: flash-attn installed but import failed, using eager attention"
                        fi
                    else
                        export FLASH_ATTENTION_AVAILABLE="false"
                        if [[ ! -f "${flash_attn_requirement}" ]]; then
                            log "WARNING: pinned flash-attn requirement file missing, using eager attention"
                        else
                            log "WARNING: flash-attn install failed (build error), using eager attention"
                        fi
                        log "         This is OK - script will work without flash attention, just slower."
                    fi
                fi
            fi
        fi
    fi

    # Check PyYAML
    if ! python3 -c "import yaml" 2>/dev/null; then
        if [[ "${PACK_NET}" == "1" ]]; then
            log "Installing pyyaml..."
            if [[ "${pip_available}" == "true" ]]; then
                pack_install_pinned_requirement "pyyaml" || missing+=("pyyaml")
            else
                missing+=("pyyaml")
            fi
        else
            missing+=("pyyaml")
        fi
    fi

    # Check protobuf (required by many HuggingFace models)
    if ! python3 -c "import google.protobuf" 2>/dev/null; then
        if [[ "${PACK_NET}" == "1" ]]; then
            log "Installing protobuf..."
            if [[ "${pip_available}" == "true" ]]; then
                pack_install_pinned_requirement "protobuf" || missing+=("protobuf")
            else
                missing+=("protobuf")
            fi
        else
            missing+=("protobuf")
        fi
    fi

    # Check sentencepiece (required by many tokenizers)
    if ! python3 -c "import sentencepiece" 2>/dev/null; then
        if [[ "${PACK_NET}" == "1" ]]; then
            log "Installing sentencepiece..."
            if [[ "${pip_available}" == "true" ]]; then
                pack_install_pinned_requirement "sentencepiece" || missing+=("sentencepiece")
            else
                missing+=("sentencepiece")
            fi
        else
            missing+=("sentencepiece")
        fi
    fi

    # Check InvarLock (Python module and CLI)
    python3 -c "import invarlock" 2>/dev/null || missing+=("invarlock")
    command -v invarlock >/dev/null 2>&1 || missing+=("invarlock-cli")

    # Check shell utilities used by the suite
    command -v jq >/dev/null 2>&1 || missing+=("jq")
    command -v nvidia-smi >/dev/null 2>&1 || missing+=("nvidia-smi")
    command -v flock >/dev/null 2>&1 || missing+=("flock")
    command -v timeout >/dev/null 2>&1 || missing+=("timeout")

    if [[ ${#missing[@]} -gt 0 ]]; then
        error_exit "Missing dependencies: ${missing[*]}"
    fi

    log "All dependencies satisfied"
}

# ============ MODEL SETUP WITH EVIDENCE PACK OPTIMIZATIONS ============
setup_model() {
    local model_id="$1"
    local gpu_id="${2:-0}"
    local model_name
    model_name=$(sanitize_model_name "${model_id}")
    local basename_name
    basename_name=$(basename "${model_id}" | tr '[:upper:]' '[:lower:]' | tr '/' '_')
    local model_dir="${OUTPUT_DIR}/models/${model_name}"
    local basename_dir="${OUTPUT_DIR}/models/${basename_name}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setting up model (GPU ${gpu_id}): ${model_id}" >> "${LOG_FILE}"

    # Check if local path
    if [[ -d "${model_id}" ]]; then
        echo "${model_id}"
        return 0
    fi

    # Check if already downloaded (prefer sanitized path, but honor basename fallback)
    if [[ -d "${model_dir}/baseline" ]]; then
        echo "${model_dir}/baseline"
        return 0
    fi
    if [[ -d "${basename_dir}/baseline" ]]; then
        echo "${basename_dir}/baseline"
        return 0
    fi

    local revision=""
    revision=$(pack_model_revision "${model_id}" || true)
    if [[ -z "${revision}" ]]; then
        if [[ "${PACK_NET}" == "1" ]]; then
            error_exit "Missing pinned revision for ${model_id}; run preflight (--net 1)."
        else
            error_exit "Offline mode requires model revisions. Run with --net 1 to preflight."
        fi
    fi

    if [[ "${PACK_NET}" != "1" ]]; then
        echo "ERROR: Offline mode requested and baseline not cached for ${model_id}." >&2
        echo "       Run with --net 1 to populate the cache." >&2
        return 1
    fi

    # Download with evidence pack optimizations
    mkdir -p "${model_dir}"

    local success_marker="${model_dir}/.download_success"
    rm -f "${success_marker}"

    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    {
        PACK_MODEL_REVISION="${revision}" CUDA_VISIBLE_DEVICES="${cuda_devices}" \
            python3 "${_PACK_VALIDATION_PY_DIR}/task_tools.py" download-baseline \
                --model-id "${model_id}" \
                --output-dir "${model_dir}/baseline" \
                --success-marker "${success_marker}"
    } 2>&1 | tee -a "${LOG_FILE}" >&2

	    if [[ ! -f "${success_marker}" ]]; then
	        # Output error to stderr (not stdout) and return empty string
	        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to download model: ${model_id}" >&2
	        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to download model: ${model_id}" >> "${LOG_FILE}"
	        # The Python downloader creates the baseline directory before downloading.
	        # If the download fails, remove the incomplete baseline dir so future runs
	        # don't treat it as a cached success.
	        rm -rf "${model_dir}/baseline" 2>/dev/null || true
	        echo ""  # Return empty string so caller can detect failure
	        return 1
	    fi
    rm -f "${success_marker}"

    echo "${model_dir}/baseline"
}
export -f setup_model

# ============ ESTIMATE MODEL SIZE FOR BATCH OPTIMIZATION ============
estimate_model_params() {
    local model_path="$1"
    _pack_validation_state estimate-model-params "${model_path}" 2>/dev/null || echo "7"
}
export -f estimate_model_params

# ============ MODEL-SIZE-AWARE INVARLOCK CONFIGURATION ============
# Returns: seq_len:stride:preview_n:final_n:eval_batch
# Based on model size and available GPU memory budget
get_model_invarlock_config() {
    local model_size="$1"  # 7, 13, 30, 40, 70, moe

    # WikiText-2 has ~1174 samples; defaults assume high-memory GPUs.
    # Format: seq_len:stride:preview_n:final_n:eval_batch
    case "${model_size}" in
        "7")
            # 7B models: ~14GB, can use longer sequences and more windows
            # Use shorter sequences for throughput: WT-2 samples are often short, so
            # longer seq_len mostly pads and wastes compute.
            echo "512:512:64:64:96"
            ;;
        "13")
            # 13-14B models: use shorter WT-2 windows by default because
            # long sequences tend to be heavily padded and under-deliver
            # effective PM token coverage on balanced CI lanes.
            echo "512:512:64:64:64"
            ;;
        "30")
            # 30B models: ~60GB, reduced settings
            echo "1024:512:40:40:48"
            ;;
        "40")
            # 40B models: ~80GB, conservative settings
            echo "1024:512:36:36:32"
            ;;
        "moe")
            # MoE models (Mixtral-8x7B): ~90GB effective
            # Moderate sequence length, conservative batch due to expert memory.
            echo "1024:512:40:40:8"
            ;;
        "70"|"72")
            # 70-72B models: ~140-144GB, ultra-conservative settings
            # Keep headroom for baseline/edited overlap and overhead checks.
            # Settings chosen to avoid double-loading baseline and edited models during overhead checks:
            # - seq_len=128: Minimal KV cache
            # - stride=64: Maintains 50% overlap
            # - windows=8+8: Minimal window count
            # - eval_batch=2: Minimal batch to avoid OOM
            echo "128:64:8:8:2"
            ;;
        *)
            # Unknown - use safe defaults
            echo "1024:512:40:40:32"
            ;;
    esac
}
export -f get_model_invarlock_config

# GPU placement is handled by the dynamic scheduler (required_gpus + reservations).
# There is no fixed GPU→model mapping.
