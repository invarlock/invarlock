#!/usr/bin/env bash
# validation_suite.sh
# ==========================================================
# InvarLock Evidence Pack Validation Suite
# ==========================================================
# Version: evidence-packs-v1
# Dependencies: bash 4+, jq, python3, invarlock CLI, nvidia-smi
# Hardware-agnostic: runs on NVIDIA GPUs where models fit VRAM.
# Designed for multi-GPU scheduling with dynamic work-stealing.
#
# EDIT TYPES (6 generated families, clean and stress scenario variants):
# - Quantization RTN (group-wise): clean tuned preset per model, 4-bit stress
# - FP8 dequantized external-subject simulation: clean tuned preset per model, E5M2 stress
# - Dense magnitude-pruned validation checkpoint: clean tuned preset per model, 50% stress
# - Dense low-rank-SVD approximated validation checkpoint: clean tuned preset per model, rank-32 stress
# - LoRA merged-adapter validation checkpoint: clean tuned preset per model, rank/alpha stress
# - Tiny fine-tune validation checkpoint: clean tuned preset per model, learning-rate/step stress
#
# MODEL SUITES:
# - Defined in scripts/evidence_packs/run_suite.sh (ungated-only models).
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
_PACK_VALIDATION_LIB_ROOT="$(cd "${_PACK_VALIDATION_LIB_DIR}/.." && pwd)"
_PACK_VALIDATION_PY_DIR="${_PACK_VALIDATION_LIB_ROOT}/../python"

_pack_validation_repo_root() {
    if [[ "$(basename "${_PACK_VALIDATION_LIB_DIR}")" == "validation" ]]; then
        cd "${_PACK_VALIDATION_LIB_DIR}/../../../.." && pwd
    else
        cd "${_PACK_VALIDATION_LIB_DIR}/../../.." && pwd
    fi
}

# Source model creation helpers, but preserve the caller's SCRIPT_DIR (run_suite.sh uses it).
_pack_prev_script_dir_was_set=0
_pack_prev_script_dir_value=""
if [[ ${SCRIPT_DIR+x} ]]; then
    _pack_prev_script_dir_was_set=1
    _pack_prev_script_dir_value="${SCRIPT_DIR}"
fi
# shellcheck source=../tasks/model_creation.sh
source "${_PACK_VALIDATION_LIB_ROOT}/tasks/model_creation.sh"
MODEL_CREATION_LOADED=1
export -n MODEL_CREATION_LOADED 2>/dev/null || true
if [[ ${_pack_prev_script_dir_was_set} -eq 1 ]]; then
    SCRIPT_DIR="${_pack_prev_script_dir_value}"
else
    unset SCRIPT_DIR 2>/dev/null || true
fi
unset _pack_prev_script_dir_was_set _pack_prev_script_dir_value

# shellcheck source=../config/config_generator.sh
source "${_PACK_VALIDATION_LIB_ROOT}/config/config_generator.sh"

_pack_result_compiler_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

compile_results() {
    mkdir -p "${OUTPUT_DIR}/analysis" "${OUTPUT_DIR}/reports"
}

run_analysis() {
    # Optional, non-gating analysis artifacts can be written under ${OUTPUT_DIR}/analysis.
    mkdir -p "${OUTPUT_DIR}/analysis"
}

generate_verdict() {
    log_section "FINAL VERDICT"

    local root
    root="$(_pack_result_compiler_root)"

    if [[ -f "${OUTPUT_DIR}/state/scenarios.json" ]]; then
        python3 "${root}/python/verdict_generator.py" \
            --output-dir "${OUTPUT_DIR}" \
            --manifest "${OUTPUT_DIR}/state/scenarios.json"
    else
        python3 "${root}/python/verdict_generator.py" --output-dir "${OUTPUT_DIR}"
    fi
    log "Wrote: ${OUTPUT_DIR}/reports/final_verdict.txt"
    log "Wrote: ${OUTPUT_DIR}/reports/final_verdict.json"
}


# ============ CLEANUP TRAP ============
cleanup() {
    local exit_code=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Script interrupted or finished with exit code: ${exit_code}"

    # Kill any background processes we spawned.
    local had_nounset="0"
    local spawned_pids=()
    case "$-" in
        *u*)
            had_nounset="1"
            set +u
            ;;
    esac
    spawned_pids=("${pids[@]}")
    if [[ ${#spawned_pids[@]} -gt 0 ]]; then
        for pid in "${spawned_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "Terminating background process: $pid"
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done
    fi
    if [[ "${had_nounset}" == "1" ]]; then
        set -u
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

_pack_validation_state() {
    python3 "${_PACK_VALIDATION_PY_DIR}/validation_state.py" "$@"
}

pack_non_runnable_deployable_ids() {
    local scenarios_file="$1"
    _pack_validation_state non-runnable-deployable-ids "${scenarios_file}"
}

pack_read_final_verdict() {
    local verdict_path="$1"
    _pack_validation_state final-verdict "${verdict_path}"
}

# ============ VERSION ============
SCRIPT_VERSION="evidence-packs-v1"

# ============ PACK CONFIGURATION ============
# Settings tuned for multi-GPU evidence packs; defaults stay conservative.

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

# shellcheck source=validation_preflight.sh
if ! declare -F pack_apply_network_mode >/dev/null 2>&1; then
    source "${_PACK_VALIDATION_LIB_DIR}/validation_preflight.sh"
fi
PACK_VALIDATION_PREFLIGHT_LOADED=1
export -n PACK_VALIDATION_PREFLIGHT_LOADED 2>/dev/null || true
# shellcheck source=validation_runtime.sh
if ! declare -F check_dependencies >/dev/null 2>&1; then
    source "${_PACK_VALIDATION_LIB_DIR}/validation_runtime.sh"
fi
PACK_VALIDATION_RUNTIME_LOADED=1
export -n PACK_VALIDATION_RUNTIME_LOADED 2>/dev/null || true
# shellcheck source=validation_dynamic.sh
if ! declare -F main_dynamic >/dev/null 2>&1; then
    source "${_PACK_VALIDATION_LIB_DIR}/validation_dynamic.sh"
fi
PACK_VALIDATION_DYNAMIC_LOADED=1
export -n PACK_VALIDATION_DYNAMIC_LOADED 2>/dev/null || true

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
    pack_preflight_datasets || return 1

    pack_model_list_array
    if [[ ${#PACK_MODEL_LIST[@]} -eq 0 ]]; then
        error_exit "No models configured for PACK_SUITE=${PACK_SUITE}."
    fi

    # Suite modes can change what parts of the task graph are generated.
    # This keeps feedback loops fast when iterating on a specific subsystem.
    case "${PACK_SUITE_MODE:-full}" in
        calibrate-only)
            # Calibration-only runs do not execute clean/stress/error scenarios.
            # Avoid requiring tuned edit presets during this phase.
            CLEAN_EDIT_RUNS=0
            STRESS_EDIT_RUNS=0
            RUN_ERROR_INJECTION="false"
            export CLEAN_EDIT_RUNS STRESS_EDIT_RUNS RUN_ERROR_INJECTION
            ;;
        errors-only)
            # Error-only runs are for rapidly iterating on error injection + guard probes.
            # Keep calibration (or preset reuse) enabled so detectors have a baseline.
            CLEAN_EDIT_RUNS=0
            STRESS_EDIT_RUNS=0
            RUN_ERROR_INJECTION="true"
            export CLEAN_EDIT_RUNS STRESS_EDIT_RUNS RUN_ERROR_INJECTION
            ;;
    esac

    pack_prepare_tuned_edit_params || return 1
    pack_validate_tuned_edit_params || return 1
    pack_prepare_calibration_presets || return 1
    pack_validate_guard_calibration || return 1

    # Net-enabled preflight uses optional python deps (huggingface_hub) and should
    # not run before we validate/install dependencies.
    if [[ "${PACK_NET}" == "1" && "${PACK_DEPENDENCIES_CHECKED:-0}" != "1" ]]; then
        check_dependencies
        PACK_DEPENDENCIES_CHECKED=1
        export PACK_DEPENDENCIES_CHECKED
    fi

    pack_validate_runtime_provenance || return 1

    if [[ "${PACK_NET}" == "1" ]]; then
        pack_preflight_models "${OUTPUT_DIR}" "${PACK_MODEL_LIST[@]}" || return 1
    else
        if ! pack_load_model_revisions; then
            error_exit "Offline mode requires model revisions. Run with --net 1 to preflight."
        fi
    fi

    main_dynamic
}
