#!/usr/bin/env bash
# validation_preflight.sh - Model selection, network, preflight, and preset preparation
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Usage: sourced by validation_suite.sh

# ============================================================
# MODEL SELECTION (DEFAULT FULL SUITE)
# ============================================================
# Defaults are ungated/public. run_suite.sh overrides these via its suite definitions.
# Approx VRAM below is weights-only; exact per-task memory is computed from
# `model_profile.json` after download.

# Small models (fit on a single high-memory GPU under typical settings)
# Note: leave a MODEL_N unset to use the default below; set it to an empty string to disable.
if [[ ! ${MODEL_1+x} ]]; then MODEL_1="mistralai/Mistral-7B-v0.1"; fi           # ~14 GB
if [[ ! ${MODEL_2+x} ]]; then MODEL_2="Qwen/Qwen2.5-7B"; fi                     # ~14 GB
if [[ ! ${MODEL_3+x} ]]; then MODEL_3="Qwen/Qwen2.5-14B"; fi                    # ~28 GB

# Medium/MoE models
if [[ ! ${MODEL_4+x} ]]; then MODEL_4="Qwen/Qwen2.5-32B"; fi                    # ~64 GB
if [[ ! ${MODEL_5+x} ]]; then MODEL_5="01-ai/Yi-34B"; fi                        # ~68 GB
if [[ ! ${MODEL_6+x} ]]; then MODEL_6="mistralai/Mixtral-8x7B-v0.1"; fi         # ~90 GB

# Additional permissive-license defaults with tuned edit parameters.
if [[ ! ${MODEL_7+x} ]]; then MODEL_7="Qwen/Qwen3-8B"; fi                       # ~16 GB
if [[ ! ${MODEL_8+x} ]]; then MODEL_8=""; fi                                    # reserved

pack_model_list() {
    local -a models=(
        "${MODEL_1:-}" "${MODEL_2:-}" "${MODEL_3:-}" "${MODEL_4:-}"
        "${MODEL_5:-}" "${MODEL_6:-}" "${MODEL_7:-}" "${MODEL_8:-}"
    )
    local model
    for model in "${models[@]}"; do
        [[ -n "${model}" ]] && printf '%s\n' "${model}"
    done
    return 0
}

pack_model_list_array() {
    PACK_MODEL_LIST=()
    if command -v mapfile >/dev/null 2>&1; then
        mapfile -t PACK_MODEL_LIST < <(pack_model_list)
        return 0
    fi
    while IFS= read -r model; do
        [[ -n "${model}" ]] || continue
        PACK_MODEL_LIST+=("${model}")
    done < <(pack_model_list)
}

pack_model_revisions_path() {
    local path="${PACK_MODEL_REVISIONS_FILE:-${OUTPUT_DIR}/state/model_revisions.json}"
    echo "${path}"
}

pack_load_model_revisions() {
    local path
    path="$(pack_model_revisions_path)"
    if [[ -f "${path}" ]]; then
        PACK_MODEL_REVISIONS_FILE="${path}"
        export PACK_MODEL_REVISIONS_FILE
        local gated_models
        if ! gated_models="$(
            jq -r '.models | to_entries[]? | select((.value.gated==true) or (.value.private==true)) | .key' \
                "${path}" 2>/dev/null
        )"; then
            echo "ERROR: Failed to parse model revisions file: ${path}" >&2
            return 1
        fi
        local gated
        gated="$(printf '%s\n' "${gated_models}" | head -n 1)"
        if [[ -n "${gated}" ]]; then
            echo "ERROR: model_revisions.json includes gated/private models; evidence packs require ungated models." >&2
            return 1
        fi
        return 0
    fi
    return 1
}

pack_model_revision() {
    local model_id="$1"
    local path
    path="$(pack_model_revisions_path)"
    [[ -f "${path}" ]] || return 1
    jq -r --arg model_id "${model_id}" '.models[$model_id].revision // ""' "${path}" 2>/dev/null
}

pack_preflight_models() {
    local output_dir="$1"
    shift
    local -a models=("$@")
    if [[ "${PACK_NET}" != "1" ]]; then
        error_exit "Preflight requires --net 1 (PACK_NET=1)."
    fi
    if [[ ${#models[@]} -eq 0 ]]; then
        error_exit "No models provided for preflight."
    fi

    mkdir -p "${output_dir}/state"
    local out_file="${output_dir}/state/model_revisions.json"
    local repo_root
    repo_root="$(_pack_validation_repo_root)"
    python3 "${repo_root}/scripts/evidence_packs/python/validation_state.py" preflight-models "${out_file}" "${models[@]}" || return 1
    PACK_MODEL_REVISIONS_FILE="${out_file}"
    export PACK_MODEL_REVISIONS_FILE
}

# Edit Configuration
EDIT_TYPE="${EDIT_TYPE:-quant_rtn}"
EDIT_BITS="${EDIT_BITS:-8}"
EDIT_GROUP_SIZE="${EDIT_GROUP_SIZE:-128}"
EDIT_SCOPE="${EDIT_SCOPE:-ffn}"

# Edit Types to test (6 generated families × clean/stress variants)
# Clean specs use tuned edit presets; use "clean" sentinel.
EDIT_TYPES_CLEAN=(
    "quant_rtn:clean:ffn"        # Clean external RTN simulation artifact (calibrated bits/group_size on FFN)
    "fp8_quant:clean:ffn"        # Clean FP8 (calibrated format on FFN)
    "magnitude_prune:clean:ffn"  # Clean pruning (calibrated sparsity on FFN)
    "lowrank_svd:clean:ffn"      # Clean low-rank (calibrated rank on FFN)
    "lora_merge:clean:attn"      # Clean deterministic merged LoRA-style delta
    "fine_tune:clean:ffn"        # Clean deterministic tiny fine-tune-style update
)

EDIT_TYPES_STRESS=(
    "quant_rtn:4:32:all"         # 4-bit group-wise RTN on all
    "fp8_quant:e5m2:all"         # FP8 E5M2 on all (stress)
    "magnitude_prune:0.5:all"    # 50% sparsity on all
    "lowrank_svd:32:all"         # rank-32 SVD on all
    "lora_merge:8:64:all"        # larger deterministic merged LoRA-style delta
    "fine_tune:0.0005:3:all"     # larger deterministic tiny fine-tune-style update
)

# Tuned edit presets (external inputs; required for clean edits)
PACK_TUNED_EDIT_PARAMS_FILE="${PACK_TUNED_EDIT_PARAMS_FILE:-}"
# Optional calibration preset reuse (skip calibration runs, copy presets in)
PACK_CALIBRATION_PRESET_DIR="${PACK_CALIBRATION_PRESET_DIR:-}"
PACK_CALIBRATION_PRESET_FILE="${PACK_CALIBRATION_PRESET_FILE:-}"
# Delete edited/error models after evaluation to keep disk usage bounded.
# Override with PACK_CLEANUP_MODELS=0 to retain model variants for debugging.
PACK_CLEANUP_MODELS="${PACK_CLEANUP_MODELS:-1}"
export PACK_CLEANUP_MODELS

# InvarLock Configuration - BASE DEFAULTS (will be overridden per-model)
# WikiText-2 validation has ~1174 usable samples
# These are conservative defaults that work for largest models (70B+)
# Smaller models will get more generous settings via get_model_invarlock_config()
INVARLOCK_PREVIEW_WINDOWS="${INVARLOCK_PREVIEW_WINDOWS:-32}"
INVARLOCK_FINAL_WINDOWS="${INVARLOCK_FINAL_WINDOWS:-32}"
INVARLOCK_DATASET="${INVARLOCK_DATASET:-wikitext2}"
INVARLOCK_TIER="${INVARLOCK_TIER:-balanced}"
INVARLOCK_SEQ_LEN="${INVARLOCK_SEQ_LEN:-512}"
INVARLOCK_STRIDE="${INVARLOCK_STRIDE:-256}"
INVARLOCK_EVAL_BATCH="${INVARLOCK_EVAL_BATCH:-32}"

# Experiment Configuration
DRIFT_CALIBRATION_RUNS="${DRIFT_CALIBRATION_RUNS:-5}"
CLEAN_EDIT_RUNS="${CLEAN_EDIT_RUNS:-3}"
STRESS_EDIT_RUNS="${STRESS_EDIT_RUNS:-2}"
RUN_ERROR_INJECTION="${RUN_ERROR_INJECTION:-true}"

# Memory planning overheads (GB) for task budgeting.
MODEL_LOAD_OVERHEAD_GB="${MODEL_LOAD_OVERHEAD_GB:-4}"
EDIT_OVERHEAD_GB="${EDIT_OVERHEAD_GB:-8}"
BATCH_EDIT_OVERHEAD_GB="${BATCH_EDIT_OVERHEAD_GB:-8}"
INVARLOCK_OVERHEAD_GB="${INVARLOCK_OVERHEAD_GB:-6}"

# Task timeout (seconds). Set to 0 or empty to disable.
export TASK_TIMEOUT_DEFAULT="${TASK_TIMEOUT_DEFAULT:-21600}"

# Output - supports resume by specifying existing directory.
# When executed, the entrypoint populates a date-stamped default.
OUTPUT_DIR="${OUTPUT_DIR:-}"

# ============================================================
# HUGGINGFACE CACHE LOCATION (CRITICAL ON GPU NODES)
# ============================================================
# HuggingFace defaults to writing caches under ~/.cache (e.g., /root/.cache when
# running as root). On many GPU nodes, / or /root is small, causing silent ENOSPC
# failures during dataset/model downloads while GPUs sit idle.
#
# Default behavior for this suite: co-locate caches under OUTPUT_DIR so they land
# on the same (usually large) filesystem as the run artifacts.
#
# Override by exporting HF_HOME / HF_HUB_CACHE / HF_DATASETS_CACHE before running
# this script.
pack_setup_hf_cache_dirs() {
    if [[ -z "${OUTPUT_DIR:-}" ]]; then
        echo "ERROR: OUTPUT_DIR is not set; use --out or PACK_OUTPUT_DIR." >&2
        return 1
    fi
    export HF_HOME="${HF_HOME:-${OUTPUT_DIR}/.hf}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
    if ! mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"; then
        echo "ERROR: Failed to create HuggingFace cache directories under: ${HF_HOME}" >&2
        return 1
    fi
    return 0
}

pack_preflight_datasets() {
    # Evidence pack runs are often launched with PACK_NET=0 (offline) after a prior net-enabled
    # preflight/population step. If HF caches are per-run (default: OUTPUT_DIR/.hf), a new
    # OUTPUT_DIR can look "cold" and cause calibration to fail later. Preflight once here to
    # fail fast with a clear message and, when PACK_NET=1, warm the cache.
    log_section "PHASE 0: DATASET PREFLIGHT"

    local repo_root
    repo_root="$(_pack_validation_repo_root)"

    if python3 "${repo_root}/scripts/evidence_packs/python/runtime_tools.py" dataset-preflight | tee -a "${LOG_FILE}"; then
        log "Dataset preflight: OK"
        return 0
    fi

    echo "" | tee -a "${LOG_FILE}"
    echo "ERROR: Dataset preflight failed (INVARLOCK_DATASET=${INVARLOCK_DATASET:-wikitext2})." | tee -a "${LOG_FILE}" >&2
    if [[ "${PACK_NET}" != "1" ]]; then
        echo "       Offline mode is enabled (PACK_NET=0)." | tee -a "${LOG_FILE}" >&2
        echo "       Fix options:" | tee -a "${LOG_FILE}" >&2
        echo "         1) Re-run with --net 1 once to populate the dataset cache, then run offline." | tee -a "${LOG_FILE}" >&2
        echo "         2) Use --resume to reuse an existing OUTPUT_DIR (its .hf cache already contains the dataset)." | tee -a "${LOG_FILE}" >&2
        echo "         3) Export HF_HOME/HF_DATASETS_CACHE to a shared cache directory before running." | tee -a "${LOG_FILE}" >&2
    fi
    error_exit "Dataset preflight failed."
}

pack_run_determinism_repeats() {
    local repeats="${PACK_REPEATS:-0}"
    if [[ -z "${OUTPUT_DIR:-}" ]]; then
        echo "ERROR: OUTPUT_DIR is not set; use --out or PACK_OUTPUT_DIR." >&2
        return 1
    fi
    if [[ -z "${repeats}" || "${repeats}" == "0" ]]; then
        return 0
    fi
    if ! [[ "${repeats}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: PACK_REPEATS must be an integer" >&2
        return 1
    fi

    if [[ -z "${PACK_MODEL_LIST[*]:-}" ]]; then
        pack_model_list_array
    fi

    local model_id="${PACK_MODEL_LIST[0]:-}"
    if [[ -z "${model_id}" ]]; then
        echo "ERROR: PACK_REPEATS requested but no models configured." >&2
        return 1
    fi

    local model_name
    model_name=$(sanitize_model_name "${model_id}")
    local model_output_dir="${OUTPUT_DIR}/${model_name}"
    local baseline_path=""
    if [[ -f "${model_output_dir}/.baseline_path" ]]; then
        baseline_path="$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)"
    fi
    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: PACK_REPEATS requires a baseline path for ${model_name}." >&2
        return 1
    fi

    local edit_spec=""
    local repeat_mode="clean"
    if [[ ${#EDIT_TYPES_CLEAN[@]} -gt 0 ]]; then
        edit_spec="${EDIT_TYPES_CLEAN[0]}"
        repeat_mode="clean"
    elif [[ ${#EDIT_TYPES_STRESS[@]} -gt 0 ]]; then
        edit_spec="${EDIT_TYPES_STRESS[0]}"
        repeat_mode="stress"
    else
        echo "ERROR: PACK_REPEATS requested but no edit specs configured." >&2
        return 1
    fi

    local resolved=""
    resolved="$(resolve_edit_params "${model_output_dir}" "${edit_spec}" "${repeat_mode}" 2>/dev/null || echo "")"

    local status=""
    local edit_dir_name=""
    if [[ -n "${resolved}" ]]; then
        status="$(printf '%s' "${resolved}" | jq -r '.status // ""' 2>/dev/null || echo "")"
        edit_dir_name="$(printf '%s' "${resolved}" | jq -r '.edit_dir_name // ""' 2>/dev/null || echo "")"
    fi

    if [[ "${status}" != "selected" || -z "${edit_dir_name}" ]]; then
        echo "ERROR: Determinism repeats requires a selected edit spec (status=${status:-<unset>})." >&2
        return 1
    fi

    local edit_path="${model_output_dir}/models/${edit_dir_name}"
    if [[ ! -d "${edit_path}" ]]; then
        echo "ERROR: Determinism repeats requires an existing edit dir: ${edit_path}" >&2
        return 1
    fi
    local edit_name="${edit_dir_name}"

    local preset_dir="${OUTPUT_DIR}/presets"
    local det_dir="${OUTPUT_DIR}/determinism/${model_name}/${edit_name}"
    mkdir -p "${det_dir}" || return 1

    local -a certs=()
    local run
    for run in $(seq 1 "${repeats}"); do
        run_invarlock_evaluate "${edit_path}" "${baseline_path}" "${det_dir}" "repeat_${run}" "${preset_dir}" "${model_name}" "0" || return 1
        local cert_path="${det_dir}/repeat_${run}/evaluation.report.json"
        if [[ -f "${cert_path}" ]]; then
            certs+=("${cert_path}")
        fi
    done

    mkdir -p "${OUTPUT_DIR}/analysis" || return 1
    local repo_root
    repo_root="$(_pack_validation_repo_root)"
    python3 "${repo_root}/scripts/evidence_packs/python/validation_state.py" \
        determinism-repeats-summary \
        "${OUTPUT_DIR}/analysis/determinism_repeats.json" \
        "${model_id}" \
        "${edit_name}" \
        "${repeats}" \
        "${PACK_DETERMINISM}" \
        "${PACK_SUITE}" \
        "${certs[@]}"
}

# Resume support - skip completed steps if output files exist
RESUME_MODE="${RESUME_MODE:-true}"

# ============ GPU OPTIMIZATION FLAGS ============
# GPU selection is configured at runtime:
# - If `CUDA_VISIBLE_DEVICES` is explicitly set (e.g., by Slurm or the user), it is respected.
# - Otherwise, the harness detects available GPUs and uses all of them.
# The selected pool is exported as `GPU_ID_LIST` (physical GPU indices) for scheduler/workers.

# TF32 / cuDNN benchmark behavior depends on PACK_DETERMINISM:
# - throughput: enable TF32 + benchmark for speed (script-level runs).
# - strict: avoid overriding determinism-friendly flags; rely on InvarLock presets.
if [[ "${PACK_DETERMINISM}" == "strict" ]]; then
    export NVIDIA_TF32_OVERRIDE=0
    export CUDNN_BENCHMARK=0
else
    export NVIDIA_TF32_OVERRIDE=1
    export CUDNN_BENCHMARK=1
fi

# Enable text-level deduplication
export INVARLOCK_DEDUP_TEXTS=1

# Memory optimization for large-model runs
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold:0.9"
unset PYTORCH_ALLOC_CONF 2>/dev/null || true

# Force deterministic workspace config with larger workspace (strict only)
if [[ "${PACK_DETERMINISM}" == "strict" ]]; then
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
else
    unset CUBLAS_WORKSPACE_CONFIG 2>/dev/null || true
fi

# Keep CUDA caching enabled for maximum memory reuse
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

pack_apply_network_mode() {
    local mode="${1:-${PACK_NET}}"
    mode=$(echo "${mode}" | tr '[:upper:]' '[:lower:]')
    case "${mode}" in
        1|true|yes|on)
            PACK_NET=1
            export INVARLOCK_ALLOW_NETWORK=1
            export HF_DATASETS_OFFLINE=0
            export TRANSFORMERS_OFFLINE=0
            export HF_HUB_OFFLINE=0
            export HF_HUB_DISABLE_TELEMETRY=1
            ;;
        *)
            PACK_NET=0
            export INVARLOCK_ALLOW_NETWORK=0
            export HF_DATASETS_OFFLINE=1
            export TRANSFORMERS_OFFLINE=1
            export HF_HUB_OFFLINE=1
            export HF_HUB_DISABLE_TELEMETRY=1
            ;;
    esac
}

pack_apply_network_mode "${PACK_NET}"

pack_configure_hf_access() {
    if [[ "${PACK_NET}" != "1" ]]; then
        return 0
    fi

    export HF_HUB_TIMEOUT="${HF_HUB_TIMEOUT:-60}"
    export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
    export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
    export HF_HUB_MAX_RETRIES="${HF_HUB_MAX_RETRIES:-10}"

    if [[ -n "${HF_ENDPOINT:-}" ]]; then
        return 0
    fi

    local primary="${HF_PRIMARY_ENDPOINT:-https://huggingface.co}"
    local mirror="${HF_MIRROR_ENDPOINT:-https://hf-mirror.com}"
    local test_path="${HF_ENDPOINT_TEST_PATH:-/datasets/cais/mmlu/resolve/main/README.md}"
    local test_timeout="${HF_ENDPOINT_TEST_TIMEOUT:-3}"

    if command -v curl >/dev/null 2>&1; then
        if curl -I --max-time "${test_timeout}" "${primary}${test_path}" >/dev/null 2>&1; then
            export HF_ENDPOINT="${primary}"
        elif curl -I --max-time "${test_timeout}" "${mirror}${test_path}" >/dev/null 2>&1; then
            export HF_ENDPOINT="${mirror}"
        else
            export HF_ENDPOINT="${primary}"
        fi
    else
        export HF_ENDPOINT="${primary}"
    fi
}
# PM acceptance range used during validation
# These bounds help avoid unnecessary gate failures during validation runs
export INVARLOCK_PM_ACCEPTANCE_MIN="${INVARLOCK_PM_ACCEPTANCE_MIN:-0.90}"
export INVARLOCK_PM_ACCEPTANCE_MAX="${INVARLOCK_PM_ACCEPTANCE_MAX:-1.10}"

# Flash attention flag - will be set dynamically based on availability
export FLASH_ATTENTION_AVAILABLE="false"

# FP8 support flag - detected in setup
export FP8_NATIVE_SUPPORT="false"

# Target memory fraction (0.92 = 92% of available) - optimal zone
export CUDA_MEMORY_FRACTION=0.92

if ! declare -F _pack_script_dir >/dev/null 2>&1; then
    _pack_script_dir() {
        cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
    }
fi

pack_source_libs() {
    # ============ LIB MODULES FOR DYNAMIC SCHEDULING ============
    SCRIPT_DIR="$(_pack_script_dir)"
    export SCRIPT_DIR  # Export for subshell workers

    # Determine lib root - support the reorganized repo layout and older
    # packaged layouts where all modules lived directly under lib/.
    if [[ -f "${SCRIPT_DIR}/../tasks/task_serialization.sh" ]]; then
        LIB_DIR="${SCRIPT_DIR}/.."
    elif [[ -f "${SCRIPT_DIR}/task_serialization.sh" ]]; then
        LIB_DIR="${SCRIPT_DIR}"
    elif [[ -d "${SCRIPT_DIR}/lib" && -f "${SCRIPT_DIR}/lib/tasks/task_serialization.sh" ]]; then
        LIB_DIR="${SCRIPT_DIR}/lib"
    elif [[ -d "${SCRIPT_DIR}/lib" && -f "${SCRIPT_DIR}/lib/task_serialization.sh" ]]; then
        LIB_DIR="${SCRIPT_DIR}/lib"
    elif [[ -d "${SCRIPT_DIR}/../lib" && -f "${SCRIPT_DIR}/../lib/tasks/task_serialization.sh" ]]; then
        LIB_DIR="${SCRIPT_DIR}/../lib"
    elif [[ -d "${SCRIPT_DIR}/../lib" && -f "${SCRIPT_DIR}/../lib/task_serialization.sh" ]]; then
        LIB_DIR="${SCRIPT_DIR}/../lib"
    else
        LIB_DIR="${SCRIPT_DIR}"
    fi
    LIB_DIR="$(cd "${LIB_DIR}" 2>/dev/null && pwd || echo "${LIB_DIR}")"
    export LIB_DIR  # Export for subshell workers

    local task_serialization_path="${LIB_DIR}/tasks/task_serialization.sh"
    local queue_manager_path="${LIB_DIR}/queue/queue_manager.sh"
    local scheduler_path="${LIB_DIR}/queue/scheduler.sh"
    local task_functions_path="${LIB_DIR}/tasks/task_functions.sh"
    local gpu_worker_path="${LIB_DIR}/queue/gpu_worker.sh"
    local fault_tolerance_path="${LIB_DIR}/core/fault_tolerance.sh"
    if [[ -f "${LIB_DIR}/task_serialization.sh" ]]; then
        task_serialization_path="${LIB_DIR}/task_serialization.sh"
        queue_manager_path="${LIB_DIR}/queue_manager.sh"
        scheduler_path="${LIB_DIR}/scheduler.sh"
        task_functions_path="${LIB_DIR}/task_functions.sh"
        gpu_worker_path="${LIB_DIR}/gpu_worker.sh"
        fault_tolerance_path="${LIB_DIR}/fault_tolerance.sh"
    fi

    # Source dynamic scheduling modules (required - optimal configuration)
    if [[ -f "${task_serialization_path}" ]]; then
        source "${task_serialization_path}"
        TASK_SERIALIZATION_LOADED=1
        export -n TASK_SERIALIZATION_LOADED 2>/dev/null || true
    else
        echo "ERROR: task_serialization.sh not found (dynamic scheduling is required)" >&2
        return 1
    fi

    if [[ -f "${queue_manager_path}" ]]; then
        source "${queue_manager_path}"
        QUEUE_MANAGER_LOADED=1
        export -n QUEUE_MANAGER_LOADED 2>/dev/null || true
    else
        echo "ERROR: queue_manager.sh not found" >&2
        return 1
    fi

    if [[ -f "${scheduler_path}" ]]; then
        source "${scheduler_path}"
        SCHEDULER_LOADED=1
        export -n SCHEDULER_LOADED 2>/dev/null || true
    else
        echo "ERROR: scheduler.sh not found" >&2
        return 1
    fi

    if [[ -f "${task_functions_path}" ]]; then
        source "${task_functions_path}"
        TASK_FUNCTIONS_LOADED=1
        export -n TASK_FUNCTIONS_LOADED 2>/dev/null || true
    else
        echo "ERROR: task_functions.sh not found" >&2
        return 1
    fi

    if [[ -f "${gpu_worker_path}" ]]; then
        source "${gpu_worker_path}"
        GPU_WORKER_LOADED=1
        export -n GPU_WORKER_LOADED 2>/dev/null || true
    else
        echo "ERROR: gpu_worker.sh not found" >&2
        return 1
    fi

    if [[ -f "${fault_tolerance_path}" ]]; then
        source "${fault_tolerance_path}"
        FAULT_TOLERANCE_LOADED=1
        export -n FAULT_TOLERANCE_LOADED 2>/dev/null || true
    fi

    return 0
}

# Fallback resolver for clean edit specs using tuned presets when task_functions isn't sourced.
if ! declare -F resolve_edit_params >/dev/null 2>&1; then
    :  # xtrace marker for branch coverage (function defs are not traced)
resolve_edit_params() {
    local model_output_dir="$1"
    local edit_spec="$2"
    local version_hint="${3:-}"

    python3 "${_PACK_VALIDATION_PY_DIR}/task_tools.py" resolve-edit-params \
        "${model_output_dir}" "${edit_spec}" "${version_hint}"
}
fi

pack_setup_output_dirs() {
    # ============ SETUP ============
    mkdir -p "${OUTPUT_DIR}"/{logs,models,evals,reports,analysis,reports,presets,workers,state} || return 1
    LOG_FILE="${OUTPUT_DIR}/logs/main.log"

    # Create a lock file for thread-safe logging
    LOG_LOCK="${OUTPUT_DIR}/logs/.log_lock"
    return 0
}

pack_prepare_scenarios_manifest() {
    local repo_root
    repo_root="$(_pack_validation_repo_root)"
    local src="${PACK_SCENARIOS_MANIFEST_FILE:-${repo_root}/scripts/evidence_packs/scenarios.json}"
    if [[ -f "${src}" ]]; then
        mkdir -p "${OUTPUT_DIR}/state"
        local dest="${OUTPUT_DIR}/state/scenarios.json"
        local rendered="${OUTPUT_DIR}/state/.scenarios.json.rendered.$$"
        local suite="${PACK_SUITE:-subset}"
        local scenario_ids_csv="${PACK_SCENARIO_IDS:-}"
        local include_deployable="${PACK_INCLUDE_DEPLOYABLE_EDITS:-0}"
        local deploy_backends_csv="${PACK_DEPLOY_BACKENDS:-}"
        _pack_validation_state render-scenarios \
            --src "${src}" \
            --out "${rendered}" \
            --suite "${suite}" \
            --scenario-ids "${scenario_ids_csv}" \
            --include-deployable "${include_deployable}" \
            --deploy-backends "${deploy_backends_csv}"

        if [[ "${RESUME_FLAG:-false}" == "true" && -f "${dest}" ]]; then
            if ! cmp -s "${dest}" "${rendered}"; then
                rm -f "${rendered}"
                error_exit "Resume run scenario manifest differs from the current contract; start a fresh OUTPUT_DIR instead of --resume."
            fi
        fi

        local non_runnable_deployable
        non_runnable_deployable="$(pack_non_runnable_deployable_ids "${rendered}")"
        if [[ -n "${non_runnable_deployable}" ]]; then
            rm -f "${rendered}"
            error_exit "Deployable scenario(s) are contract placeholders and are not runnable yet: ${non_runnable_deployable}"
        fi

        mv "${rendered}" "${dest}"
    fi
}

pack_resolve_active_scenarios_manifest() {
    if [[ -n "${OUTPUT_DIR:-}" ]]; then
        local state_manifest="${OUTPUT_DIR}/state/scenarios.json"
        if [[ -f "${state_manifest}" ]]; then
            printf '%s\n' "${state_manifest}"
            return 0
        fi
    fi

    local repo_root
    repo_root="$(_pack_validation_repo_root)"
    local src="${PACK_SCENARIOS_MANIFEST_FILE:-${repo_root}/scripts/evidence_packs/scenarios.json}"
    if [[ -f "${src}" ]]; then
        printf '%s\n' "${src}"
        return 0
    fi

    return 1
}

pack_count_edit_scenarios() {
    local scenarios_file=""
    scenarios_file="$(pack_resolve_active_scenarios_manifest 2>/dev/null || true)"

    if [[ -n "${scenarios_file}" && -f "${scenarios_file}" ]]; then
        local source_label="scenarios.json"
        if [[ -n "${OUTPUT_DIR:-}" && "${scenarios_file}" == "${OUTPUT_DIR}/state/scenarios.json" ]]; then
            source_label="state/scenarios.json"
        fi
        local counts=""
        counts="$(_pack_validation_state count-edit-scenarios "${scenarios_file}" --source-label "${source_label}" 2>/dev/null || true)"
        if [[ "${counts}" =~ ^[0-9]+\|[0-9]+\| ]]; then
            printf '%s\n' "${counts}"
            return 0
        fi
    fi

    printf '%s|%s|defaults\n' "${#EDIT_TYPES_CLEAN[@]}" "${#EDIT_TYPES_STRESS[@]}"
}

pack_resolve_tuned_edit_params_file() {
    if [[ -n "${PACK_TUNED_EDIT_PARAMS_FILE:-}" ]]; then
        return 0
    fi

    local repo_root
    repo_root="$(_pack_validation_repo_root)"
    local candidate
    for candidate in \
        "${repo_root}/scripts/evidence_packs/tuned_edit_params.json" \
        "${repo_root}/scripts/evidence_packs/presets/tuned_edit_params.json"
    do
        if [[ -f "${candidate}" ]]; then
            PACK_TUNED_EDIT_PARAMS_FILE="${candidate}"
            export PACK_TUNED_EDIT_PARAMS_FILE
            return 0
        fi
    done
}

pack_prepare_tuned_edit_params() {
    if [[ ${CLEAN_EDIT_RUNS:-0} -le 0 ]]; then
        return 0
    fi

    pack_resolve_tuned_edit_params_file
    if [[ -z "${PACK_TUNED_EDIT_PARAMS_FILE:-}" ]]; then
        error_exit "Missing PACK_TUNED_EDIT_PARAMS_FILE for clean edit presets."
    fi
    if [[ ! -f "${PACK_TUNED_EDIT_PARAMS_FILE}" ]]; then
        error_exit "Tuned edit preset file not found: ${PACK_TUNED_EDIT_PARAMS_FILE}"
    fi

    mkdir -p "${OUTPUT_DIR}/state"
    local dest="${OUTPUT_DIR}/state/tuned_edit_params.json"
    cp "${PACK_TUNED_EDIT_PARAMS_FILE}" "${dest}"
    PACK_TUNED_EDIT_PARAMS_FILE="${dest}"
    export PACK_TUNED_EDIT_PARAMS_FILE
}

pack_validate_tuned_edit_params() {
    if [[ ${CLEAN_EDIT_RUNS:-0} -le 0 ]]; then
        return 0
    fi

    local model_csv
    model_csv=$(printf '%s\n' "${PACK_MODEL_LIST[@]}" | paste -sd "," -)
    local model_names_csv=""
    for model_id in "${PACK_MODEL_LIST[@]}"; do
        local model_name
        model_name=$(sanitize_model_name "${model_id}")
        if [[ -z "${model_names_csv}" ]]; then
            model_names_csv="${model_name}"
        else
            model_names_csv="${model_names_csv},${model_name}"
        fi
    done
    local edit_types_csv
    edit_types_csv=$(printf '%s\n' "${EDIT_TYPES_CLEAN[@]}" | awk -F: '{print $1}' | sort -u | paste -sd "," -)
    local repo_root
    repo_root="$(_pack_validation_repo_root)"
    local canonical_tuned_edit_params_file=""
    if [[ -f "${repo_root}/scripts/evidence_packs/tuned_edit_params.json" ]]; then
        canonical_tuned_edit_params_file="${repo_root}/scripts/evidence_packs/tuned_edit_params.json"
    elif [[ -f "${repo_root}/scripts/evidence_packs/presets/tuned_edit_params.json" ]]; then
        canonical_tuned_edit_params_file="${repo_root}/scripts/evidence_packs/presets/tuned_edit_params.json"
    fi
    local -a validate_args=(
        "${repo_root}/scripts/evidence_packs/python/validation_state.py"
        validate-tuned-edit-params
        --file "${PACK_TUNED_EDIT_PARAMS_FILE}"
        --models "${model_csv}"
        --model-names "${model_names_csv}"
        --edit-types "${edit_types_csv}"
    )
    if [[ -n "${canonical_tuned_edit_params_file}" ]]; then
        validate_args+=(--canonical-file "${canonical_tuned_edit_params_file}")
    fi
    if [[ "${PACK_ALLOW_NONCANONICAL_TUNED_EDIT_PARAMS:-0}" == "1" ]]; then
        validate_args+=(--allow-noncanonical)
    fi
    python3 "${validate_args[@]}" \
        || return 1
}

pack_prepare_calibration_presets() {
    if [[ -z "${PACK_CALIBRATION_PRESET_DIR:-}" && -z "${PACK_CALIBRATION_PRESET_FILE:-}" ]]; then
        return 0
    fi

    if [[ -n "${PACK_CALIBRATION_PRESET_FILE:-}" && ! -f "${PACK_CALIBRATION_PRESET_FILE}" ]]; then
        error_exit "Calibration preset file not found: ${PACK_CALIBRATION_PRESET_FILE}"
    fi

    mkdir -p "${OUTPUT_DIR}/presets"

    for model_id in "${PACK_MODEL_LIST[@]}"; do
        local model_name
        model_name=$(sanitize_model_name "${model_id}")
        if [[ -n "${PACK_CALIBRATION_PRESET_FILE:-}" ]]; then
            local src="${PACK_CALIBRATION_PRESET_FILE}"
            local ext="${src##*.}"
            local dest="${OUTPUT_DIR}/presets/calibrated_preset_${model_name}.${ext}"
            cp "${src}" "${dest}"
        else
            local copied_any="false"
            for ext in yaml yml json; do
                local candidate=""
                for candidate in \
                    "${PACK_CALIBRATION_PRESET_DIR}/calibrated_preset_${model_name}.${ext}" \
                    "${PACK_CALIBRATION_PRESET_DIR}/calibrated_preset_${model_name}"__*.${ext}; do
                    [[ -f "${candidate}" ]] || continue
                    cp "${candidate}" "${OUTPUT_DIR}/presets/$(basename "${candidate}")"
                    copied_any="true"
                done
            done
            if [[ "${copied_any}" != "true" ]]; then
                error_exit "Missing calibration preset for ${model_id} in ${PACK_CALIBRATION_PRESET_DIR:-<unset>}."
            fi
        fi
    done

    PACK_PRESET_READY="true"
    export PACK_PRESET_READY
    DRIFT_CALIBRATION_RUNS=0
    export DRIFT_CALIBRATION_RUNS
}

pack_validate_guard_calibration() {
    local runs="${DRIFT_CALIBRATION_RUNS:-5}"
    if ! [[ "${runs}" =~ ^[0-9]+$ ]]; then
        runs=5
    fi
    if [[ ${runs} -le 0 && -z "${PACK_CALIBRATION_PRESET_DIR:-}" && -z "${PACK_CALIBRATION_PRESET_FILE:-}" ]]; then
        error_exit "Guard calibration disabled (DRIFT_CALIBRATION_RUNS=0) without a calibration preset file/dir."
    fi
}

pack_validate_runtime_provenance() {
    local repo_root
    repo_root="$(_pack_validation_repo_root)"
    python3 "${repo_root}/scripts/evidence_packs/python/runtime_tools.py" \
        remote-setup-smoke \
        --only-runtime-provenance || return 1
}
