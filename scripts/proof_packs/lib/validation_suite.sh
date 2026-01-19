#!/usr/bin/env bash
# validation_suite.sh
# ==========================================================
# InvarLock Proof Pack Validation Suite
# ==========================================================
# Version: proof-packs-v1
# Dependencies: bash 4+, jq, python3, invarlock CLI, nvidia-smi
# Hardware-agnostic: runs on NVIDIA GPUs where models fit VRAM.
# Designed for multi-GPU scheduling with dynamic work-stealing.
#
# EDIT TYPES (4 types × 2 versions = 8 tests per model):
# - Quantization RTN (group-wise): clean tuned preset per model, 4-bit stress
# - FP8 Quantization: clean tuned preset per model, E5M2 stress
# - Magnitude Pruning: clean tuned preset per model, 50% stress
# - Low-Rank SVD: clean tuned preset per model, rank-32 stress
#
# MODEL SUITES:
# - Defined in scripts/proof_packs/suites.sh (ungated-only models).
# - Subset targets single-GPU runs; full targets multi-GPU hardware.
#
# EXECUTION FLOW:
# 1. Optional preflight to pin model revisions
# 2. Launch models across available GPUs
# 3. Each GPU runs: calibration → edits → error injection
# 4. Compile reports → final verdict
# ==========================================================

# Dynamic scheduling is always enabled.
# Static scheduling has been removed.
# Uses a "small_first" priority strategy. Multi-GPU is used only when the
# per-task profile exceeds per-GPU memory; adaptive under-allocation is disabled
# by default to avoid OOM.

# Initialize pids array early (used by cleanup trap when executed)
declare -a pids=()


# Split modules (keeps validation_suite.sh focused on orchestration).
_PACK_VALIDATION_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source model creation helpers, but preserve the caller's SCRIPT_DIR (run_suite.sh uses it).
_pack_prev_script_dir_was_set=0
_pack_prev_script_dir_value=""
if [[ ${SCRIPT_DIR+x} ]]; then
    _pack_prev_script_dir_was_set=1
    _pack_prev_script_dir_value="${SCRIPT_DIR}"
fi
# shellcheck source=model_creation.sh
source "${_PACK_VALIDATION_LIB_DIR}/model_creation.sh"
export MODEL_CREATION_LOADED=1
if [[ ${_pack_prev_script_dir_was_set} -eq 1 ]]; then
    SCRIPT_DIR="${_pack_prev_script_dir_value}"
else
    unset SCRIPT_DIR 2>/dev/null || true
fi
unset _pack_prev_script_dir_was_set _pack_prev_script_dir_value

# shellcheck source=config_generator.sh
source "${_PACK_VALIDATION_LIB_DIR}/config_generator.sh"
# shellcheck source=result_compiler.sh
source "${_PACK_VALIDATION_LIB_DIR}/result_compiler.sh"


# ============ CLEANUP TRAP ============
cleanup() {
    local exit_code=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Script interrupted or finished with exit code: ${exit_code}"

    # Kill any background processes we spawned
    # Check if pids array exists and has elements
    if [[ ${#pids[@]} -gt 0 ]]; then
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "Terminating background process: $pid"
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done
    fi
    # Clean up lock file
    rm -f "${LOG_LOCK:-}" 2>/dev/null || true

    exit ${exit_code}
}

# Trap + strict mode are enabled only when executed as a script (not when sourced).

pack_require_bash4() {
    # Associative arrays require bash 4.0+
    if ! pack_is_bash4; then
        echo "ERROR: This script requires bash 4.0 or later (have ${BASH_VERSION})" >&2
        echo "       Associative arrays are not supported in bash ${BASH_VERSION}" >&2
        return 1
    fi
    return 0
}

# Overrideable hook for tests (simulate bash4 on bash3 without env vars).
pack_is_bash4() {
    [[ "${BASH_VERSINFO[0]}" -ge 4 ]]
}

# ============ VERSION ============
SCRIPT_VERSION="proof-packs-v1"

# ============ PACK CONFIGURATION ============
# Settings tuned for multi-GPU proof packs; defaults stay conservative.

# GPU Configuration (auto-detected at runtime unless explicitly set)
NUM_GPUS="${NUM_GPUS:-}"
export GPU_MEMORY_GB="${GPU_MEMORY_GB:-}"

# Determinism/throughput toggle for this harness (independent of InvarLock CLI presets).
# - throughput (default): enable TF32 + cuDNN benchmark for maximum speed.
# - strict: prefer deterministic-friendly flags and avoid overriding CLI presets.
PACK_DETERMINISM="${PACK_DETERMINISM:-throughput}"
case "${PACK_DETERMINISM}" in
    strict|throughput)
        :
        ;;
    *)
        PACK_DETERMINISM="throughput"
        ;;
esac
export PACK_DETERMINISM

PACK_SUITE="${PACK_SUITE:-subset}"
PACK_NET="${PACK_NET:-0}"
PACK_REPEATS="${PACK_REPEATS:-0}"
PACK_OUTPUT_DIR="${PACK_OUTPUT_DIR:-}"
PACK_MODEL_REVISIONS_FILE="${PACK_MODEL_REVISIONS_FILE:-}"
if [[ -n "${PACK_OUTPUT_DIR}" && -z "${OUTPUT_DIR:-}" ]]; then
    OUTPUT_DIR="${PACK_OUTPUT_DIR}"
fi

# ============================================================
# MODEL SELECTION (DEFAULT FULL SUITE)
# ============================================================
# Defaults are ungated/public. run_suite.sh overrides these via suites.sh.
# Approx VRAM below is weights-only; exact per-task memory is computed from
# `model_profile.json` after download.

# Small models (fit on a single high-memory GPU under typical settings)
# Note: leave a MODEL_N unset to use the default below; set it to an empty string to disable.
if [[ ! ${MODEL_1+x} ]]; then MODEL_1="mistralai/Mistral-7B-v0.1"; fi           # ~14 GB
if [[ ! ${MODEL_2+x} ]]; then MODEL_2="NousResearch/Llama-2-13b-hf"; fi         # ~26 GB
if [[ ! ${MODEL_3+x} ]]; then MODEL_3="Qwen/Qwen2.5-14B"; fi                    # ~28 GB

# Medium/MoE models
if [[ ! ${MODEL_4+x} ]]; then MODEL_4="Qwen/Qwen2.5-32B"; fi                    # ~64 GB
if [[ ! ${MODEL_5+x} ]]; then MODEL_5="01-ai/Yi-34B"; fi                        # ~68 GB
if [[ ! ${MODEL_6+x} ]]; then MODEL_6="mistralai/Mixtral-8x7B-v0.1"; fi         # ~90 GB

# Large models
if [[ ! ${MODEL_7+x} ]]; then MODEL_7="NousResearch/Llama-2-70b-hf"; fi         # ~140 GB
if [[ ! ${MODEL_8+x} ]]; then MODEL_8="Qwen/Qwen1.5-72B"; fi                    # ~144 GB

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
            echo "ERROR: model_revisions.json includes gated/private models; proof packs require ungated models." >&2
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
    repo_root="$(cd "${_PACK_VALIDATION_LIB_DIR}/../../.." && pwd)"
    python3 "${repo_root}/scripts/proof_packs/python/preflight_models.py" "${out_file}" "${models[@]}" || return 1
    PACK_MODEL_REVISIONS_FILE="${out_file}"
    export PACK_MODEL_REVISIONS_FILE
}

# Edit Configuration
EDIT_TYPE="${EDIT_TYPE:-quant_rtn}"
EDIT_BITS="${EDIT_BITS:-8}"
EDIT_GROUP_SIZE="${EDIT_GROUP_SIZE:-128}"
EDIT_SCOPE="${EDIT_SCOPE:-ffn}"

# Edit Types to test (4 types × 2 versions each)
# Clean specs use tuned edit presets; use "clean" sentinel.
EDIT_TYPES_CLEAN=(
    "quant_rtn:clean:ffn"        # Clean RTN (calibrated bits/group_size on FFN)
    "fp8_quant:clean:ffn"        # Clean FP8 (calibrated format on FFN)
    "magnitude_prune:clean:ffn"  # Clean pruning (calibrated sparsity on FFN)
    "lowrank_svd:clean:ffn"      # Clean low-rank (calibrated rank on FFN)
)

EDIT_TYPES_STRESS=(
    "quant_rtn:4:32:all"         # 4-bit group-wise RTN on all
    "fp8_quant:e5m2:all"         # FP8 E5M2 on all (stress)
    "magnitude_prune:0.5:all"    # 50% sparsity on all
    "lowrank_svd:32:all"         # rank-32 SVD on all
)

# Tuned edit presets (external inputs; required for clean edits)
PACK_TUNED_EDIT_PARAMS_FILE="${PACK_TUNED_EDIT_PARAMS_FILE:-}"
# Optional calibration preset reuse (skip calibration runs, copy presets in)
PACK_CALIBRATION_PRESET_DIR="${PACK_CALIBRATION_PRESET_DIR:-}"
PACK_CALIBRATION_PRESET_FILE="${PACK_CALIBRATION_PRESET_FILE:-}"

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
# Override by exporting HF_HOME / HF_HUB_CACHE / HF_DATASETS_CACHE /
# TRANSFORMERS_CACHE before running this script.
pack_setup_hf_cache_dirs() {
    if [[ -z "${OUTPUT_DIR:-}" ]]; then
        echo "ERROR: OUTPUT_DIR is not set; use --out or PACK_OUTPUT_DIR." >&2
        return 1
    fi
    export HF_HOME="${HF_HOME:-${OUTPUT_DIR}/.hf}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
    if ! mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"; then
        echo "ERROR: Failed to create HuggingFace cache directories under: ${HF_HOME}" >&2
        return 1
    fi
    return 0
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
        run_invarlock_certify "${edit_path}" "${baseline_path}" "${det_dir}" "repeat_${run}" "${preset_dir}" "${model_name}" "0" || return 1
        local cert_path="${det_dir}/repeat_${run}/evaluation.cert.json"
        if [[ -f "${cert_path}" ]]; then
            certs+=("${cert_path}")
        fi
    done

    mkdir -p "${OUTPUT_DIR}/analysis" || return 1
    local repo_root
    repo_root="$(cd "${_PACK_VALIDATION_LIB_DIR}/../../.." && pwd)"
    python3 "${repo_root}/scripts/proof_packs/python/determinism_repeats_summary.py" \
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
export INVARLOCK_PM_ACCEPTANCE_MAX="${INVARLOCK_PM_ACCEPTANCE_MAX:-1.20}"

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

    # Determine lib directory - support lib/ and flat layouts.
    if [[ -f "${SCRIPT_DIR}/task_serialization.sh" ]]; then
        LIB_DIR="${SCRIPT_DIR}"
    elif [[ -d "${SCRIPT_DIR}/lib" && -f "${SCRIPT_DIR}/lib/task_serialization.sh" ]]; then
        LIB_DIR="${SCRIPT_DIR}/lib"
    elif [[ -d "${SCRIPT_DIR}/../lib" && -f "${SCRIPT_DIR}/../lib/task_serialization.sh" ]]; then
        LIB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/lib"
    else
        LIB_DIR="${SCRIPT_DIR}"
    fi
    export LIB_DIR  # Export for subshell workers

    # Source dynamic scheduling modules (required - optimal configuration)
    if [[ -f "${LIB_DIR}/task_serialization.sh" ]]; then
        source "${LIB_DIR}/task_serialization.sh"
        export TASK_SERIALIZATION_LOADED=1
    else
        echo "ERROR: lib/task_serialization.sh not found (dynamic scheduling is required)" >&2
        return 1
    fi

    if [[ -f "${LIB_DIR}/queue_manager.sh" ]]; then
        source "${LIB_DIR}/queue_manager.sh"
        export QUEUE_MANAGER_LOADED=1
    else
        echo "ERROR: lib/queue_manager.sh not found" >&2
        return 1
    fi

    if [[ -f "${LIB_DIR}/scheduler.sh" ]]; then
        source "${LIB_DIR}/scheduler.sh"
        export SCHEDULER_LOADED=1
    else
        echo "ERROR: lib/scheduler.sh not found" >&2
        return 1
    fi

    if [[ -f "${LIB_DIR}/task_functions.sh" ]]; then
        source "${LIB_DIR}/task_functions.sh"
        export TASK_FUNCTIONS_LOADED=1
    else
        echo "ERROR: lib/task_functions.sh not found" >&2
        return 1
    fi

    if [[ -f "${LIB_DIR}/gpu_worker.sh" ]]; then
        source "${LIB_DIR}/gpu_worker.sh"
        export GPU_WORKER_LOADED=1
    else
        echo "ERROR: lib/gpu_worker.sh not found" >&2
        return 1
    fi

    if [[ -f "${LIB_DIR}/fault_tolerance.sh" ]]; then
        source "${LIB_DIR}/fault_tolerance.sh"
        export FAULT_TOLERANCE_LOADED=1
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

    python3 - "${model_output_dir}" "${edit_spec}" "${version_hint}" <<-'PY'
	import json
	import os
	import sys
	from pathlib import Path

model_output_dir = Path(sys.argv[1])
edit_spec = sys.argv[2] if len(sys.argv) > 2 else ""
version_hint = sys.argv[3] if len(sys.argv) > 3 else ""

parts = edit_spec.split(":") if edit_spec else []
edit_type = parts[0] if parts else ""
param1 = parts[1] if len(parts) > 1 else ""
param2 = parts[2] if len(parts) > 2 else ""
scope = parts[3] if len(parts) > 3 else ""

if edit_type != "quant_rtn" and not scope:
    scope = param2
    param2 = ""

if edit_type == "quant_rtn" and not scope:
    if param1 and param2:
        scope = param2
        param2 = ""

clean_spec = param1 == "clean"
status = "selected"
reason = ""
edit_dir_name = ""

if clean_spec:
    tuned_path = (os.environ.get("PACK_TUNED_EDIT_PARAMS_FILE") or "").strip()
    model_id_path = model_output_dir / ".model_id"
    model_id = ""
    if model_id_path.exists():
        try:
            model_id = model_id_path.read_text().strip()
        except Exception:
            model_id = ""
    model_key = model_id or model_output_dir.name

    def _load_tuned_entry():
        if not tuned_path:
            return {}, "missing", "missing_tuned_edit_params_file"
        path = Path(tuned_path)
        if not path.exists():
            return {}, "missing", "missing_tuned_edit_params_file"
        try:
            data = json.loads(path.read_text())
        except Exception:
            return {}, "invalid", "invalid_tuned_edit_params_file"
        if not isinstance(data, dict):
            return {}, "invalid", "invalid_tuned_edit_params_file"

        entry_map = {}
        models = data.get("models")
        if isinstance(models, dict):
            entry_map = (
                models.get(model_key)
                or models.get(model_id)
                or models.get(model_output_dir.name)
                or {}
            )
        if not entry_map and isinstance(data.get(edit_type), dict):
            entry_map = data
        defaults = data.get("defaults")
        entry = entry_map.get(edit_type) or (defaults.get(edit_type) if isinstance(defaults, dict) else {}) or {}
        if not isinstance(entry, dict):
            entry = {}
        status = str(entry.get("status") or "missing")
        reason = str(entry.get("reason") or "")
        return entry, status, reason

    entry, status, reason = _load_tuned_entry()
    if status == "selected":
        if edit_type == "quant_rtn":
            param1 = str(entry.get("bits", ""))
            param2 = str(entry.get("group_size", ""))
            scope = str(entry.get("scope") or scope or "")
        elif edit_type == "fp8_quant":
            param1 = str(entry.get("format", ""))
            scope = str(entry.get("scope") or scope or "")
        elif edit_type == "magnitude_prune":
            param1 = str(entry.get("sparsity", ""))
            scope = str(entry.get("scope") or scope or "")
        elif edit_type == "lowrank_svd":
            param1 = str(entry.get("rank", ""))
            scope = str(entry.get("scope") or scope or "")
        edit_dir_name = str(entry.get("edit_dir_name") or "")
else:
    def _is_int(val):
        try:
            int(val)
            return True
        except Exception:
            return False

    def _is_float(val):
        try:
            float(val)
            return True
        except Exception:
            return False

    if edit_type == "quant_rtn":
        if not (_is_int(param1) and _is_int(param2)):
            status = "invalid"
            reason = "invalid_quant_params"
    elif edit_type == "magnitude_prune":
        if not _is_float(param1):
            status = "invalid"
            reason = "invalid_prune_sparsity"
    elif edit_type == "lowrank_svd":
        if not _is_int(param1):
            status = "invalid"
            reason = "invalid_lowrank_rank"
    elif edit_type == "fp8_quant":
        if not param1:
            status = "invalid"
            reason = "invalid_fp_format"

version = version_hint or ("clean" if clean_spec else "")

if status == "selected" and not edit_dir_name:
    if edit_type == "quant_rtn":
        edit_dir_name = f"quant_{param1}bit_{version}" if version else ""
    elif edit_type == "fp8_quant":
        edit_dir_name = f"fp8_{param1}_{version}" if version else ""
    elif edit_type == "magnitude_prune":
        try:
            pct = int(float(param1) * 100)
        except Exception:
            pct = 0
        edit_dir_name = f"prune_{pct}pct_{version}" if version else ""
    elif edit_type == "lowrank_svd":
        edit_dir_name = f"svd_rank{param1}_{version}" if version else ""
    else:
        edit_dir_name = f"{edit_type}_{version}" if version else ""

payload = {
    "status": status,
    "reason": reason,
    "edit_type": edit_type,
    "param1": param1,
    "param2": param2,
    "scope": scope,
    "version": version,
    "edit_dir_name": edit_dir_name,
}
print(json.dumps(payload))
PY
}
fi

pack_setup_output_dirs() {
    # ============ SETUP ============
    mkdir -p "${OUTPUT_DIR}"/{logs,models,evals,certificates,analysis,reports,presets,workers,state} || return 1
    LOG_FILE="${OUTPUT_DIR}/logs/main.log"

    # Create a lock file for thread-safe logging
    LOG_LOCK="${OUTPUT_DIR}/logs/.log_lock"
    return 0
}

pack_prepare_scenarios_manifest() {
    local repo_root
    repo_root="$(cd "${_PACK_VALIDATION_LIB_DIR}/../../.." && pwd)"
    local src="${repo_root}/scripts/proof_packs/scenarios.json"
    if [[ -f "${src}" ]]; then
        mkdir -p "${OUTPUT_DIR}/state"
        cp "${src}" "${OUTPUT_DIR}/state/scenarios.json"
    fi
}

pack_resolve_tuned_edit_params_file() {
    if [[ -n "${PACK_TUNED_EDIT_PARAMS_FILE:-}" ]]; then
        return 0
    fi

    local repo_root
    repo_root="$(cd "${_PACK_VALIDATION_LIB_DIR}/../../.." && pwd)"
    local candidate
    for candidate in \
        "${repo_root}/scripts/proof_packs/tuned_edit_params.json" \
        "${repo_root}/scripts/proof_packs/presets/tuned_edit_params.json"
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
    repo_root="$(cd "${_PACK_VALIDATION_LIB_DIR}/../../.." && pwd)"
    python3 "${repo_root}/scripts/proof_packs/python/validate_tuned_edit_params.py" \
        --file "${PACK_TUNED_EDIT_PARAMS_FILE}" \
        --models "${model_csv}" \
        --model-names "${model_names_csv}" \
        --edit-types "${edit_types_csv}" || return 1
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
        local src=""
        if [[ -n "${PACK_CALIBRATION_PRESET_FILE:-}" ]]; then
            src="${PACK_CALIBRATION_PRESET_FILE}"
        else
            for ext in yaml yml json; do
                local candidate="${PACK_CALIBRATION_PRESET_DIR}/calibrated_preset_${model_name}.${ext}"
                if [[ -f "${candidate}" ]]; then
                    src="${candidate}"
                    break
                fi
            done
        fi
        if [[ -z "${src}" ]]; then
            error_exit "Missing calibration preset for ${model_id} in ${PACK_CALIBRATION_PRESET_DIR:-<unset>}."
        fi
        local ext="${src##*.}"
        local dest="${OUTPUT_DIR}/presets/calibrated_preset_${model_name}.${ext}"
        cp "${src}" "${dest}"
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

    local edits_total=$(( ${#EDIT_TYPES_CLEAN[@]} + ${#EDIT_TYPES_STRESS[@]} ))
    local errors_total=0
    if [[ "${RUN_ERROR_INJECTION}" == "true" ]]; then
        errors_total=9  # nan_injection, inf_injection, shape_mismatch, missing_tensors, extreme_quant, scale_explosion, rank_collapse, norm_collapse, weight_tying_break
    fi

    local baseline_mode="${PACK_BASELINE_STORAGE_MODE:-snapshot_symlink}"
    local baseline_copy=1
    if [[ "${baseline_mode}" == "snapshot_symlink" ]]; then
        baseline_copy=0  # baseline files are symlinks to HF hub cache blobs
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
    [[ -z "${free_gb}" ]] && return 0

    local planned_gb=""
    planned_gb=$(estimate_planned_model_storage_gb 2>/dev/null || echo "")
    [[ -z "${planned_gb}" ]] && return 0

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
    log "       Baseline storage mode: ${PACK_BASELINE_STORAGE_MODE:-snapshot_symlink} (snapshot_symlink avoids a full extra baseline copy)."
    log "       Fix: mount a larger volume and set OUTPUT_DIR, or run the subset suite, or set RUN_ERROR_INJECTION=false."
    log "       Override (not recommended): PACK_SKIP_DISK_PREFLIGHT=1"

    # Resume mode may already have artifacts; allow user to proceed if explicitly resuming.
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
    cat > "${OUTPUT_DIR}/state/disk_pressure.json" << EOF
{
  "detected_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "free_gb": ${free_gb},
  "min_free_gb": ${min_gb},
  "output_dir": "${OUTPUT_DIR}"
}
EOF
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
    repo_root="$(cd "${_PACK_VALIDATION_LIB_DIR}/../../.." && pwd)"
    env_report=$(python3 "${repo_root}/scripts/proof_packs/python/env_report.py")
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
check_dependencies() {
    log_section "PHASE 0: DEPENDENCY CHECK"

    local missing=()

    # Check Python
    command -v python3 >/dev/null 2>&1 || missing+=("python3")

    # Check PyTorch with CUDA
    python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null || missing+=("torch+cuda")

    # Check transformers
    python3 -c "import transformers; print(f'Transformers {transformers.__version__}')" 2>/dev/null || missing+=("transformers")

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
                log "Flash Attention 2: Not found, attempting install..."
                # Use timeout to prevent hanging on slow builds
                if timeout 600 python3 -m pip install flash-attn --no-build-isolation 2>&1 | tee -a "${LOG_FILE}"; then
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
                    log "WARNING: flash-attn install failed (build error), using eager attention"
                    log "         This is OK - script will work without flash attention, just slower."
                fi
            fi
        fi
    fi

    # Check PyYAML
    python3 -c "import yaml" 2>/dev/null || python3 -m pip install pyyaml

    # Check protobuf (required by many HuggingFace models)
    if ! python3 -c "import google.protobuf" 2>/dev/null; then
        log "Installing protobuf..."
        python3 -m pip install protobuf
    fi

    # Check sentencepiece (required by many tokenizers)
    if ! python3 -c "import sentencepiece" 2>/dev/null; then
        log "Installing sentencepiece..."
        python3 -m pip install sentencepiece
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

# ============ MODEL SETUP WITH PROOF PACK OPTIMIZATIONS ============
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

    # Download with proof pack optimizations
    mkdir -p "${model_dir}"

    local success_marker="${model_dir}/.download_success"
    rm -f "${success_marker}"

    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    {
        PACK_MODEL_REVISION="${revision}" CUDA_VISIBLE_DEVICES="${cuda_devices}" python3 << EOF
import json
import os
import sys
from pathlib import Path

model_id = "${model_id}"
output_dir = Path("${model_dir}/baseline")
success_marker = Path("${success_marker}")
flash_available = "${FLASH_ATTENTION_AVAILABLE}" == "true"
revision = os.environ.get("PACK_MODEL_REVISION") or None

# Storage strategy for the baseline directory:
# - snapshot_symlink (default): create baseline/ as symlinks to HF hub cache blobs (saves ~1× weights on disk)
# - snapshot_copy: copy snapshot files into baseline/ (duplicates hub cache)
# - save_pretrained: load with transformers and write a full baseline copy (also duplicates hub cache)
baseline_mode = os.environ.get("PACK_BASELINE_STORAGE_MODE", "snapshot_symlink").strip().lower()

output_dir.mkdir(parents=True, exist_ok=True)

rev_label = f"@{revision}" if revision else ""
print(f"Downloading {model_id}{rev_label} (proof pack optimized)...")
print(f"Baseline storage mode: {baseline_mode}")
if os.environ.get("HF_HUB_CACHE"):
    print(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}")
print(f"Flash Attention 2: {'enabled' if flash_available else 'disabled'}")


def model_supports_flash_attention(model_id: str) -> bool:
    no_fa2_models = [
        "falcon",
        "mpt-",
        "gpt2",
        "bloom",
        "opt-",
        "gpt-j",
        "gpt-neo",
        "codegen",
        "santacoder",
        "stablelm",
    ]
    model_lower = model_id.lower()
    return not any(pattern in model_lower for pattern in no_fa2_models)


def sanitize_generation_config(model_dir: Path) -> None:
    gen_path = model_dir / "generation_config.json"
    if not gen_path.is_file():
        return
    try:
        gen = json.loads(gen_path.read_text())
    except Exception:
        return

    if gen.get("do_sample") is False:
        temp = gen.get("temperature", None)
        if temp not in (None, 1.0):
            print(f"Fixing generation_config.json: clearing temperature={temp} (do_sample=False)")
            gen["temperature"] = None
        top_p = gen.get("top_p", None)
        if top_p not in (None, 1.0):
            print(f"Fixing generation_config.json: clearing top_p={top_p} (do_sample=False)")
            gen["top_p"] = None
        try:
            gen_path.write_text(json.dumps(gen, indent=2) + "\n")
        except Exception:
            pass


def write_model_profile(model_dir: Path, model_id: str) -> None:
    weights_bytes = 0
    for pat in ("*.safetensors", "*.bin"):
        for fp in model_dir.glob(pat):
            try:
                weights_bytes += fp.stat().st_size
            except OSError:
                pass

    cfg_path = model_dir / "config.json"
    config = {}
    if cfg_path.is_file():
        try:
            config = json.loads(cfg_path.read_text())
        except Exception:
            config = {}

    profile = {
        "model_id": model_id,
        "revision": revision,
        "weights_bytes": weights_bytes,
        "weights_gb": round(weights_bytes / (1024**3), 3),
        "hidden_size": config.get("hidden_size"),
        "num_layers": config.get("num_hidden_layers"),
        "num_heads": config.get("num_attention_heads"),
        "num_kv_heads": config.get("num_key_value_heads") or config.get("num_attention_heads"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "dtype_bytes": 2,
    }
    (model_dir / "model_profile.json").write_text(json.dumps(profile, indent=2) + "\n")


def download_snapshot(repo_id: str, model_dir: Path, mode: str) -> None:
    from huggingface_hub import snapshot_download

    local_dir_use_symlinks = mode == "snapshot_symlink"
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=local_dir_use_symlinks,
        cache_dir=os.environ.get("HF_HUB_CACHE"),
        resume_download=True,
        revision=revision,
    )


try:
    if baseline_mode in ("snapshot_symlink", "snapshot_copy"):
        try:
            download_snapshot(model_id, output_dir, baseline_mode)
            sanitize_generation_config(output_dir)
            write_model_profile(output_dir, model_id)
            success_marker.touch()
            print(f"Saved to {output_dir} (snapshot)")
            sys.exit(0)
        except Exception as snap_err:
            print(f"WARNING: snapshot_download failed, falling back to save_pretrained: {snap_err}", file=sys.stderr)
            baseline_mode = "save_pretrained"

    # Fallback: create a full baseline copy via transformers (duplicates HF hub cache).
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mode = os.environ.get("PACK_DETERMINISM", "throughput").strip().lower()
    if mode == "strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cache_dir = os.environ.get("HF_HUB_CACHE")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=cache_dir, revision=revision
    )
    tokenizer.save_pretrained(output_dir)

    use_fa2 = flash_available and model_supports_flash_attention(model_id)
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "cache_dir": cache_dir,
        "revision": revision,
    }
    if use_fa2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print(f"Using Flash Attention 2 for {model_id}")
    else:
        print(f"Using eager attention for {model_id} (FA2 not supported or unavailable)")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except Exception as fa2_err:
        if use_fa2 and "flash" in str(fa2_err).lower():
            print(f"Flash Attention 2 failed, falling back to eager attention: {fa2_err}")
            del model_kwargs["attn_implementation"]
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        else:
            raise

    # Fix invalid generation config before saving (some models have temperature/top_p without do_sample)
    if hasattr(model, "generation_config"):
        gen_config = model.generation_config
        if getattr(gen_config, "do_sample", True) is False:
            if getattr(gen_config, "temperature", 1.0) not in (None, 1.0):
                print(f"Fixing generation_config: clearing temperature={gen_config.temperature} (do_sample=False)")
                gen_config.temperature = None
            if getattr(gen_config, "top_p", 1.0) not in (None, 1.0):
                print(f"Fixing generation_config: clearing top_p={gen_config.top_p} (do_sample=False)")
                gen_config.top_p = None

    model.save_pretrained(output_dir, safe_serialization=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory.empty_cache()

    sanitize_generation_config(output_dir)
    write_model_profile(output_dir, model_id)
    success_marker.touch()
    print(f"Saved to {output_dir} (save_pretrained)")

except Exception as e:
    print(f"ERROR: Model download failed: {e}", file=sys.stderr)
    sys.exit(1)
EOF
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
    local config_file="${model_path}/config.json"
    if [[ ! -f "${config_file}" ]]; then
        echo "7"
        return
    fi

    # Returns model size bucket for batch optimization
    # Also detects MoE architectures (Mixtral) which need special handling
    # Note: config_file is passed as argument to avoid shell injection issues
    local params=$(python3 -c "
import json
import sys
try:
    config_path = sys.argv[1]
    config = json.load(open(config_path))

    # Extract architecture parameters
    h = config.get('hidden_size', 4096)
    l = config.get('num_hidden_layers', 32)
    v = config.get('vocab_size', 32000)
    i = config.get('intermediate_size', h * 4)  # FFN intermediate size

    # Detect MoE architecture (Mixtral style)
    num_experts = config.get('num_local_experts', 1)
    if num_experts == 1:
        num_experts = config.get('num_experts', 1)

    # Better parameter estimation formula:
    # - Embedding: vocab_size * hidden_size
    # - Attention per layer: 4 * hidden_size^2 (Q,K,V,O projections)
    # - FFN per layer: 3 * hidden_size * intermediate_size (SwiGLU/gate has 3 matrices)
    # - LM head: hidden_size * vocab_size
    embedding_params = v * h
    attention_per_layer = 4 * h * h
    ffn_per_layer = 3 * h * i  # gate_proj, up_proj, down_proj
    lm_head = h * v

    base_params = (embedding_params + l * (attention_per_layer + ffn_per_layer) + lm_head) / 1e9

    # For MoE, each expert has its own FFN, but we only activate some at a time
    # Memory scales with total params (all experts loaded), so multiply FFN contribution
    if num_experts > 1:
        moe_ffn = l * ffn_per_layer * num_experts
        base_params = (embedding_params + l * attention_per_layer + moe_ffn + lm_head) / 1e9
        print('moe')
    elif base_params > 55:  # 70B/72B models
        print('70')
    elif base_params > 28:  # 30B-40B models
        print('40')
    elif base_params > 18:  # 20B-30B models (Qwen2.5-32B etc)
        print('30')
    elif base_params > 10:  # 13B-14B models
        print('13')
    else:
        print('7')
except Exception as e:
    # Debug: uncomment to see why detection fails
    # import sys; print(f'estimate_model_params error: {e}', file=sys.stderr)
    print('7')
" "${config_file}" 2>/dev/null)
    echo "${params:-7}"
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
            echo "2048:1024:64:64:96"
            ;;
        "13")
            # 13-14B models: ~26-28GB, moderate settings
            # Note: estimate_model_params() returns "13" for both 13B and 14B
            echo "1536:768:48:48:64"
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
            # Moderate sequence length, smaller batch due to expert memory
            echo "1024:512:40:40:24"
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

# ============ MAIN - DYNAMIC GPU SCHEDULING (v2.0) ============
main_dynamic() {
    local start_time=$(date +%s)
    local gpu_mem="${PACK_GPU_MEM_GB:-${GPU_MEMORY_GB:-}}"
    local gpu_count_label="${NUM_GPUS:-auto}"
    [[ -z "${gpu_mem}" ]] && gpu_mem="auto"
    [[ -z "${gpu_count_label}" ]] && gpu_count_label="auto"

    echo "========================================================================"
    echo "  InvarLock Proof Pack Suite v${SCRIPT_VERSION}"
    echo "  ${gpu_mem}GB x ${gpu_count_label} GPU DYNAMIC SCHEDULING"
    echo "========================================================================"
    echo ""

    check_dependencies
    configure_gpu_pool
    pack_model_list_array

    # Disk pressure preflight (Slurm/Ray-style node health gate).
    # Abort before starting work to avoid half-written artifacts when storage is nearly full.
    local min_free="${MIN_FREE_DISK_GB:-200}"
    if ! [[ "${min_free}" =~ ^[0-9]+$ ]]; then
        min_free=200
    fi
    local free_gb=""
    free_gb=$(get_free_disk_gb "${OUTPUT_DIR}" 2>/dev/null || echo "")
    if [[ -n "${free_gb}" && ${free_gb} -lt ${min_free} ]]; then
        handle_disk_pressure "${free_gb}" "${min_free}"
    fi

    # Disk capacity preflight based on planned model/edit storage.
    # This prevents expensive GPU time from being spent only to later hit ENOSPC.
    disk_preflight

    setup_pack_environment

    log "Output directory: ${OUTPUT_DIR}"
    log "GPU pool: ${NUM_GPUS} GPU(s) [${GPU_ID_LIST}]"
    local model_count
    model_count=$(pack_model_list | wc -l | tr -d ' ')
    log "Models: ${model_count} (PACK_SUITE=${PACK_SUITE})"
    local scenarios_file="${OUTPUT_DIR}/state/scenarios.json"
    local clean_scenarios=0
    local stress_scenarios=0
    local error_scenarios=0
    local edit_scenarios_source="defaults"
    local error_scenarios_source="defaults"

    if command -v jq >/dev/null 2>&1 && [[ -f "${scenarios_file}" ]]; then
        clean_scenarios="$(jq -r '[.scenarios[] | select(.generation.kind=="edit" and .generation.version=="clean")] | length' "${scenarios_file}" 2>/dev/null || echo 0)"
        stress_scenarios="$(jq -r '[.scenarios[] | select(.generation.kind=="edit" and .generation.version=="stress")] | length' "${scenarios_file}" 2>/dev/null || echo 0)"
        error_scenarios="$(jq -r '[.scenarios[] | select(.generation.kind=="error")] | length' "${scenarios_file}" 2>/dev/null || echo 0)"
        edit_scenarios_source="state/scenarios.json"
        error_scenarios_source="state/scenarios.json"
    fi

    if ! [[ "${clean_scenarios}" =~ ^[0-9]+$ ]]; then
        clean_scenarios=0
    fi
    if ! [[ "${stress_scenarios}" =~ ^[0-9]+$ ]]; then
        stress_scenarios=0
    fi
    if ! [[ "${error_scenarios}" =~ ^[0-9]+$ ]]; then
        error_scenarios=0
    fi

    # Match queue_manager fallback behavior when the manifest is missing or incomplete.
    if [[ ${clean_scenarios} -le 0 || ${stress_scenarios} -le 0 ]]; then
        clean_scenarios=4
        stress_scenarios=4
        edit_scenarios_source="defaults"
    fi

    local clean_runs="${CLEAN_EDIT_RUNS:-0}"
    if ! [[ "${clean_runs}" =~ ^-?[0-9]+$ ]]; then
        clean_runs=0
    fi
    if [[ ${clean_runs} -lt 0 ]]; then
        clean_runs=0
    fi

    local stress_runs="${STRESS_EDIT_RUNS:-0}"
    if ! [[ "${stress_runs}" =~ ^-?[0-9]+$ ]]; then
        stress_runs=0
    fi
    if [[ ${stress_runs} -lt 0 ]]; then
        stress_runs=0
    fi

    local edit_scenarios_total=$((clean_scenarios + stress_scenarios))
    local edit_certify_clean=$((clean_scenarios * clean_runs))
    local edit_certify_stress=$((stress_scenarios * stress_runs))
    local edit_certify_total=$((edit_certify_clean + edit_certify_stress))

    log "Edit scenarios: ${clean_scenarios} clean + ${stress_scenarios} stress = ${edit_scenarios_total} per model (${edit_scenarios_source})"
    log "Edit certify runs: clean=${clean_scenarios}×${clean_runs}=${edit_certify_clean}, stress=${stress_scenarios}×${stress_runs}=${edit_certify_stress} (total=${edit_certify_total} per model)"

    if [[ "${RUN_ERROR_INJECTION:-true}" == "true" ]]; then
        if [[ ${error_scenarios} -le 0 ]]; then
            error_scenarios=9
            error_scenarios_source="defaults"
        fi
        log "Error scenarios: ${error_scenarios} (RUN_ERROR_INJECTION=true) (${error_scenarios_source})"
    else
        log "Error scenarios: disabled (RUN_ERROR_INJECTION=false)"
    fi
    log "Tuned edit presets: ${PACK_TUNED_EDIT_PARAMS_FILE:-<unset>}"
    if [[ "${PACK_PRESET_READY:-false}" == "true" ]]; then
        log "Calibration presets: reuse (${OUTPUT_DIR}/presets)"
    else
        log "Calibration presets: ${DRIFT_CALIBRATION_RUNS:-5} run(s)"
    fi
    log "Scheduling: DYNAMIC (work-stealing enabled)"
    log ""

    # Initialize queue
    log_section "PHASE 1: INITIALIZING TASK QUEUE"

    # Check for --resume mode: skip task generation if queue already exists with tasks
    local existing_queue="${OUTPUT_DIR}/queue"
    local skip_task_generation="false"
    local resume_total_tasks=0
    local resume_existing_running=0
    local resume_existing_failed=0

    if [[ "${RESUME_FLAG}" == "true" && -d "${existing_queue}" ]]; then
        # Count existing tasks across all queues
        local existing_pending=$(find "${existing_queue}/pending" -name "*.task" 2>/dev/null | wc -l | tr -d ' ')
        local existing_ready=$(find "${existing_queue}/ready" -name "*.task" 2>/dev/null | wc -l | tr -d ' ')
        local existing_running=$(find "${existing_queue}/running" -name "*.task" 2>/dev/null | wc -l | tr -d ' ')
        local existing_completed=$(find "${existing_queue}/completed" -name "*.task" 2>/dev/null | wc -l | tr -d ' ')
        local existing_failed=$(find "${existing_queue}/failed" -name "*.task" 2>/dev/null | wc -l | tr -d ' ')
        local existing_total=$((existing_pending + existing_ready + existing_running + existing_completed + existing_failed))

        if [[ ${existing_total} -gt 0 ]]; then
            skip_task_generation="true"
            resume_total_tasks="${existing_total}"
            resume_existing_running="${existing_running}"
            resume_existing_failed="${existing_failed}"
            log "RESUME MODE: Found existing queue with ${existing_total} tasks"
            log "  Pending: ${existing_pending}, Ready: ${existing_ready}, Running: ${existing_running}"
            log "  Completed: ${existing_completed}, Failed: ${existing_failed}"
        fi
    fi

    init_queue "${OUTPUT_DIR}"
    # Clear any previous shutdown markers so new workers can start cleanly (important for --resume).
    rm -f "${OUTPUT_DIR}/workers/SHUTDOWN" "${OUTPUT_DIR}/workers"/gpu_*.shutdown 2>/dev/null || true
    # Initialize GPU reservation tracking for multi-GPU tasks before workers start.
    if type init_gpu_reservations &>/dev/null; then
        init_gpu_reservations "${OUTPUT_DIR}"
        log "GPU reservations dir: ${GPU_RESERVATION_DIR:-unset}; GPUs: $(list_run_gpu_ids | tr '\n' ',' | sed 's/,$//')"
    fi
    export QUEUE_DIR GPU_RESERVATION_DIR  # Export for subshell workers

    local total_tasks=0
    if [[ "${skip_task_generation}" == "true" ]]; then
        log "Skipping task generation (--resume mode)"
        # Reclaim any stuck running tasks from a previous run (kills stray procs, releases GPU reservations).
        if [[ ${resume_existing_running} -gt 0 ]]; then
            log "Reclaiming ${resume_existing_running} orphaned running task(s) for resume..."
            local gpu_id
            for gpu_id in $(list_run_gpu_ids); do
                reclaim_orphaned_tasks "${gpu_id}" >> "${LOG_FILE}" 2>&1 || true
            done
        fi

        # Move failed tasks back to pending for retry, clearing retry/backoff state for immediate resume.
        if [[ ${resume_existing_failed} -gt 0 ]]; then
            log "Resetting ${resume_existing_failed} failed task(s) back to pending for resume..."
            local task_file
            for task_file in "${QUEUE_DIR}/failed"/*.task; do
                [[ -f "${task_file}" ]] || continue
                local tmp_file="${task_file}.resume.$$"
                jq '(.params // {}) as $p
                    | .status="pending"
                    | .retries=0
                    | .gpu_id=-1
                    | .assigned_gpus=null
                    | .started_at=null
                    | .completed_at=null
                    | .error_msg=null
                    | .params=($p + {retry_after:null,last_error_type:null})' "${task_file}" > "${tmp_file}" 2>/dev/null \
                    && mv "${tmp_file}" "${task_file}" 2>/dev/null \
                    || { rm -f "${tmp_file}" 2>/dev/null || true; }
                mv "${task_file}" "${QUEUE_DIR}/pending/" 2>/dev/null || true
            done
        fi

        # Re-resolve dependencies after reclaim/reset.
        if type resolve_dependencies &>/dev/null; then
            local moved=$(resolve_dependencies)
            log "Re-resolved dependencies: moved ${moved} tasks to ready queue"
        fi
    else
        # Generate all tasks
        log "Generating tasks for all models..."
        log "Config: CLEAN_EDIT_RUNS=${CLEAN_EDIT_RUNS}, STRESS_EDIT_RUNS=${STRESS_EDIT_RUNS}, RUN_ERROR_INJECTION=${RUN_ERROR_INJECTION}, DRIFT_CALIBRATION_RUNS=${DRIFT_CALIBRATION_RUNS}, PACK_PRESET_READY=${PACK_PRESET_READY:-false}, PACK_USE_BATCH_EDITS=${PACK_USE_BATCH_EDITS:-auto}"
        local model_csv
        model_csv=$(printf '%s\n' "${PACK_MODEL_LIST[@]}" | paste -sd "," -)
        log "Models: ${model_csv:-<none>}"
        generate_all_tasks "${PACK_MODEL_LIST[@]}"
    fi

    if type refresh_task_memory_from_profiles &>/dev/null; then
        refresh_task_memory_from_profiles "${OUTPUT_DIR}"
    fi
    if type export_memory_plan &>/dev/null; then
        export_memory_plan "${OUTPUT_DIR}"
    fi

    # Resolve initial dependencies on fresh runs so workers can start immediately (avoid idle GPUs).
    if [[ "${skip_task_generation}" != "true" ]] && type resolve_dependencies &>/dev/null; then
        local moved_initial=0
        moved_initial=$(resolve_dependencies 2>/dev/null) || moved_initial=0
        log "Resolved initial dependencies: moved ${moved_initial} task(s) to ready queue"
    fi
    if type demote_ready_tasks_for_calibration_only &>/dev/null; then
        demote_ready_tasks_for_calibration_only 2>/dev/null || true
    fi

    total_tasks=$(count_tasks "pending")
    total_tasks=$((total_tasks + $(count_tasks "ready")))
    total_tasks=$((total_tasks + $(count_tasks "completed")))
    log "Total tasks in queue: ${total_tasks} (pending+ready: $(($(count_tasks "pending") + $(count_tasks "ready"))))"

    # Launch worker pool
    log_section "PHASE 2: LAUNCHING GPU WORKERS"
    log "Starting ${NUM_GPUS} GPU workers with dynamic task scheduling..."

    # Initialize log files
    for gpu_id in $(list_run_gpu_ids); do
        touch "${OUTPUT_DIR}/logs/gpu_${gpu_id}.log"
    done

    start_worker() {
        local gpu_id="$1"
        local action="${2:-Starting}"

        # Avoid duplicating a live worker on the same GPU
        local pid_file="${OUTPUT_DIR}/workers/gpu_${gpu_id}.pid"
        if [[ -f "${pid_file}" ]]; then
            local existing_pid
            existing_pid=$(cat "${pid_file}" 2>/dev/null || true)
            if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
                log "  GPU ${gpu_id}: worker already running (PID ${existing_pid}), skipping start"
                return 0
            fi
        fi

        log "  GPU ${gpu_id}: ${action} worker"
        # Run in subshell that sources libraries (bash functions don't inherit to background processes)
        # Note: SCRIPT_DIR, LIB_DIR, QUEUE_DIR, OUTPUT_DIR must all be exported before this point
        (
            # Re-source all necessary modules in the subshell context
            source "${LIB_DIR}/task_serialization.sh"
            source "${LIB_DIR}/queue_manager.sh"
            source "${LIB_DIR}/scheduler.sh"
            source "${LIB_DIR}/task_functions.sh"
            source "${LIB_DIR}/gpu_worker.sh"
            [[ -f "${LIB_DIR}/fault_tolerance.sh" ]] && source "${LIB_DIR}/fault_tolerance.sh"
            gpu_worker "${gpu_id}" "${OUTPUT_DIR}"
        ) >> "${OUTPUT_DIR}/logs/gpu_${gpu_id}.log" 2>&1 &
        pids[${gpu_id}]=$!
        echo "${pids[${gpu_id}]}" > "${OUTPUT_DIR}/workers/gpu_${gpu_id}.pid"
    }

    update_progress() {
        local total="$1"
        local completed="$2"
        local failed="$3"
        local status="$4"

        mkdir -p "${OUTPUT_DIR}/state"
        cat > "${OUTPUT_DIR}/state/progress.json" <<EOF
{
  "total_tasks": ${total},
  "completed_tasks": ${completed},
  "failed_tasks": ${failed},
  "status": "${status}",
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    }

    # Unified monitor loop: progress + dependency resolution + worker health
    log_section "PHASE 3: MONITORING PROGRESS"
    local check_interval=60
    local worker_timeout="${WORKER_TIMEOUT:-2700}"
    local workers_started=0
    while true; do
        if [[ ${workers_started} -eq 0 ]]; then
            for gpu_id in $(list_run_gpu_ids); do
                start_worker "${gpu_id}" "Starting"
            done
            workers_started=1
        fi

        sleep "${check_interval}"

        # Disk pressure check (Slurm/Ray-style node health gate).
        # Abort early to avoid corrupting artifacts when storage is nearly full.
        local min_free="${MIN_FREE_DISK_GB:-200}"
        if ! [[ "${min_free}" =~ ^[0-9]+$ ]]; then
            min_free=200
        fi
        local free_gb=""
        free_gb=$(get_free_disk_gb "${OUTPUT_DIR}" 2>/dev/null || echo "")
        if [[ -n "${free_gb}" && ${free_gb} -lt ${min_free} ]]; then
            handle_disk_pressure "${free_gb}" "${min_free}"
        fi

        # Check if done
        if [[ "${PACK_SUITE_MODE:-full}" == "calibrate-only" ]]; then
            local preset_total=0
            local preset_completed=0
            preset_total=$(find "${QUEUE_DIR}" -type f -name "*_GENERATE_PRESET_*.task" 2>/dev/null | wc -l | tr -d ' ')
            preset_completed=$(find "${QUEUE_DIR}/completed" -type f -name "*_GENERATE_PRESET_*.task" 2>/dev/null | wc -l | tr -d ' ')
            if [[ ${preset_total} -gt 0 && ${preset_completed} -eq ${preset_total} ]]; then
                log "Calibration-only: generated ${preset_completed}/${preset_total} calibrated preset(s); stopping early"
                if type signal_shutdown &>/dev/null; then
                    signal_shutdown "${OUTPUT_DIR}"
                else
                    touch "${OUTPUT_DIR}/workers/SHUTDOWN"
                fi
                local summary_stats=""
                summary_stats="$(get_queue_stats 2>/dev/null || true)"
                if [[ -n "${summary_stats}" ]]; then
                    IFS=':' read -r pending ready running completed failed total <<< "${summary_stats}"
                    update_progress "${total:-0}" "${completed:-0}" "${failed:-0}" "complete"
                fi
                break
            fi
        fi
        if is_queue_empty; then
            if type signal_shutdown &>/dev/null; then
                signal_shutdown "${OUTPUT_DIR}"
            else
                touch "${OUTPUT_DIR}/workers/SHUTDOWN"
            fi
            local summary_stats=""
            summary_stats="$(get_queue_stats 2>/dev/null || true)"
            if [[ -n "${summary_stats}" ]]; then
                IFS=':' read -r pending ready running completed failed total <<< "${summary_stats}"
                update_progress "${total:-0}" "${completed:-0}" "${failed:-0}" "complete"
            fi
            break
        fi

        # Check each worker for liveness and heartbeat
        for gpu_id in $(list_run_gpu_ids); do
            local pid_file="${OUTPUT_DIR}/workers/gpu_${gpu_id}.pid"
            local heartbeat_file="${OUTPUT_DIR}/workers/gpu_${gpu_id}.heartbeat"
            local status_file="${OUTPUT_DIR}/workers/gpu_${gpu_id}.status"

            [[ -f "${pid_file}" ]] || continue
            local pid
            pid=$(cat "${pid_file}" 2>/dev/null || true)
            [[ -z "${pid}" ]] && continue

            if ! kill -0 "${pid}" 2>/dev/null; then
                log "WARNING: Worker GPU ${gpu_id} (PID ${pid}) died"
                wait "${pid}" 2>/dev/null || true
                reclaim_orphaned_tasks "${gpu_id}"
                start_worker "${gpu_id}" "Restarting"
                continue
            fi

            if [[ -f "${heartbeat_file}" ]]; then
                local heartbeat_mtime
                heartbeat_mtime=$(stat -c %Y "${heartbeat_file}" 2>/dev/null || stat -f %m "${heartbeat_file}" 2>/dev/null || echo "")
                if [[ -n "${heartbeat_mtime}" ]]; then
                    local heartbeat_age=$(( $(date +%s) - heartbeat_mtime ))
                    if [[ ${heartbeat_age} -gt ${worker_timeout} ]]; then
                        local status
                        status=$(cat "${status_file}" 2>/dev/null || echo "unknown")
                        log "WARNING: Worker GPU ${gpu_id} stuck (no heartbeat for ${heartbeat_age}s, status: ${status})"
                        kill -9 "${pid}" 2>/dev/null || true
                        wait "${pid}" 2>/dev/null || true
                        reclaim_orphaned_tasks "${gpu_id}"
                        start_worker "${gpu_id}" "Restarting stuck"
                    fi
                fi
            fi
        done

        # Centralized dependency resolution - moved from worker loops to reduce lock contention.
        # Only the monitor (single process) calls this, avoiding 8 workers competing for queue lock.
        local deps_moved=0
        deps_moved=$(resolve_dependencies 2>/dev/null) || deps_moved=0
        if [[ ${deps_moved} -gt 0 ]]; then
            log "Monitor: Promoted ${deps_moved} task(s) from pending to ready queue"
        fi
        local deps_canceled=0
        if type cancel_tasks_with_failed_dependencies &>/dev/null; then
            deps_canceled=$(cancel_tasks_with_failed_dependencies "${CANCEL_BLOCKED_TASKS_GRACE_SECONDS:-90}" 2>/dev/null) || deps_canceled=0
            if [[ ${deps_canceled} -gt 0 ]]; then
                log "Monitor: Marked ${deps_canceled} task(s) failed due to failed dependencies"
            fi
        fi

        # Print progress
        local_stats="$(get_queue_stats 2>/dev/null || true)"
        if [[ -z "${local_stats}" ]]; then
            log "Progress: queue stats unavailable"
            continue
        fi
        IFS=':' read -r pending ready running completed failed total <<< "${local_stats}"
        pending=${pending:-0}
        ready=${ready:-0}
        running=${running:-0}
        completed=${completed:-0}
        failed=${failed:-0}
        total=${total:-0}

        pct=0
        [[ ${total} -gt 0 ]] && pct=$((completed * 100 / total))

        log "Progress: ${completed}/${total} tasks (${pct}%) | Running: ${running} | Ready: ${ready} | Failed: ${failed}"
        update_progress "${total}" "${completed}" "${failed}" "running"

        # Apply work-stealing boost if needed
        apply_work_stealing_boost 2>/dev/null || true
    done

    # Wait for all workers
    log "Waiting for all workers to complete..."
    local failed=0
    for gpu_id in $(list_run_gpu_ids); do
        local pid="${pids[${gpu_id}]:-}"
        if [[ -n "${pid}" ]]; then
            if wait "${pid}"; then
                log "  GPU ${gpu_id}: Worker completed successfully"
            else
                log "  GPU ${gpu_id}: Worker failed"
                failed=$((failed + 1))
            fi
        fi
    done

    # Print final queue stats
    print_queue_stats

    if [[ ${failed} -gt 0 ]]; then
        log "WARNING: ${failed} GPU worker(s) failed"
    fi

    # Check for failed tasks
    local failed_tasks=$(count_tasks "failed")
    if [[ ${failed_tasks} -gt 0 ]]; then
        log "WARNING: ${failed_tasks} task(s) failed"
        log "Failed tasks:"
        for task_file in "${QUEUE_DIR}/failed"/*.task; do
            [[ -f "${task_file}" ]] || continue
            local task_id=$(get_task_id "${task_file}")
            local error=$(get_task_field "${task_file}" "error_msg")
            log "  - ${task_id}: ${error:-unknown error}"
        done
    fi

    if [[ "${PACK_SUITE_MODE:-full}" == "calibrate-only" ]]; then
        log_section "CALIBRATION CHECKPOINT"
        log "Calibration-only run stopped after preset generation."
        log "Presets: ${OUTPUT_DIR}/presets/"
        log "To continue: OUTPUT_DIR=${OUTPUT_DIR} $0 --run-only"
        return 0
    fi

    if [[ -n "${PACK_REPEATS:-}" && "${PACK_REPEATS}" != "0" ]]; then
        log_section "DETERMINISM REPEATS"
        if ! pack_run_determinism_repeats; then
            log "WARNING: Determinism repeats failed; see logs for details."
        fi
    fi

    log_section "PHASE 4: ANALYSIS"
    compile_results
    run_analysis
    generate_verdict

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_section "COMPLETE"
    log "Total time: $((duration / 3600))h $(((duration % 3600) / 60))m $((duration % 60))s"
    log "Tasks completed: $(count_tasks "completed")/${total_tasks}"
    log "Report: ${OUTPUT_DIR}/reports/final_verdict.txt"
    log "Presets: ${OUTPUT_DIR}/presets/"
}

# ============ MAIN ============
# Dynamic scheduling with work-stealing is the only supported mode (v2.1.0)
# Static scheduling was removed as it's less efficient.
main() {
    main_dynamic "$@"
}

pack_run_suite() {
    # Enable strict mode only for actual suite execution (tests may source this file).
    set -uo pipefail
    trap cleanup EXIT INT TERM HUP QUIT

    pack_require_bash4 || return 1

    if [[ -z "${OUTPUT_DIR:-}" ]]; then
        echo "ERROR: OUTPUT_DIR is not set; use run_suite.sh --out or PACK_OUTPUT_DIR." >&2
        return 1
    fi
    # Optionally normalize OUTPUT_DIR to an absolute path (set PACK_OUTPUT_DIR_ABSOLUTE=true).
    if [[ -n "${OUTPUT_DIR}" && "${PACK_OUTPUT_DIR_ABSOLUTE:-false}" == "true" ]]; then
        OUTPUT_DIR="$(cd "$(dirname "${OUTPUT_DIR}")" && pwd)/$(basename "${OUTPUT_DIR}")"
    fi
    PACK_OUTPUT_DIR="${OUTPUT_DIR}"
    export PACK_OUTPUT_DIR

    pack_apply_network_mode "${PACK_NET}"
    pack_source_libs || return 1
    pack_setup_output_dirs || return 1
    pack_prepare_scenarios_manifest || return 1
    pack_setup_hf_cache_dirs || return 1

    pack_model_list_array
    if [[ ${#PACK_MODEL_LIST[@]} -eq 0 ]]; then
        error_exit "No models configured for PACK_SUITE=${PACK_SUITE}."
    fi

    pack_prepare_tuned_edit_params || return 1
    pack_validate_tuned_edit_params || return 1
    pack_prepare_calibration_presets || return 1
    pack_validate_guard_calibration || return 1

    if [[ "${PACK_NET}" == "1" ]]; then
        pack_preflight_models "${OUTPUT_DIR}" "${PACK_MODEL_LIST[@]}" || return 1
    else
        if ! pack_load_model_revisions; then
            error_exit "Offline mode requires model revisions. Run with --net 1 to preflight."
        fi
    fi

    main_dynamic
}
