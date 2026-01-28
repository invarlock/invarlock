#!/usr/bin/env bash
# task_functions.sh - Atomic task implementations for dynamic scheduling
# Version: proof-packs-v1 (InvarLock Proof Pack Suite)
# Dependencies: jq, python3, invarlock CLI, task_serialization.sh
# Usage: sourced by gpu_worker.sh/validation_suite.sh for per-task execution
#
# Each function executes a single atomic task type with explicit parameters.
# These are extracted from the original monolithic process_model() function
# to enable parallel execution across GPUs.

# Source dependencies
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime.sh
source "${SCRIPT_DIR}/runtime.sh"
[[ -z "${QUEUE_MANAGER_LOADED:-}" ]] && source "${SCRIPT_DIR}/queue_manager.sh" && export QUEUE_MANAGER_LOADED=1

# ============ FALLBACK FUNCTIONS ============
# These provide fallback implementations when main script functions aren't available
# (e.g., when running in subshell workers that only source lib modules)

# Detect model size from model name/path string
# Returns: 7, 13, 30, 40, 70, moe
_get_model_size_from_name() {
    local model_id="$1"
    local model_lower=$(printf '%s' "${model_id}" | tr '[:upper:]' '[:lower:]')

    # Check for MoE architecture first
    if [[ "${model_lower}" =~ mixtral || "${model_lower}" =~ 8x7b || "${model_lower}" =~ moe ]]; then
        echo "moe"
    # Check for 70B/72B models (largest)
    elif [[ "${model_lower}" =~ 70b || "${model_lower}" =~ 72b || "${model_lower}" =~ 65b ]]; then
        echo "70"
    # Check for 40B models
    elif [[ "${model_lower}" =~ 40b || "${model_lower}" =~ 34b ]]; then
        echo "40"
    # Check for 30B models
    elif [[ "${model_lower}" =~ 30b || "${model_lower}" =~ 32b || "${model_lower}" =~ 33b ]]; then
        echo "30"
    # Check for 13B/14B models
    elif [[ "${model_lower}" =~ 13b || "${model_lower}" =~ 14b ]]; then
        echo "13"
    # Default to 7B
    else
        echo "7"
    fi
}

# Get model-aware InvarLock configuration (fallback implementation)
# Returns: seq_len:stride:preview_n:final_n:eval_batch
_get_model_invarlock_config_fallback() {
    local model_size="$1"  # 7, 13, 30, 40, 70, moe

    # Conservative defaults that satisfy CI pairing/coverage floors for proof packs.
    # Use zero overlap (stride == seq_len) and ≥180 windows to avoid E001/E005
    # if workers start without the main suite wrapper.
    case "${model_size}" in
        "7")
            # Shorter sequences are substantially faster on WT-2 (many samples are
            # short, so longer seq_len mostly pads).
            echo "512:512:192:192:96"
            ;;
        "13")
            echo "1536:1536:192:192:64"
            ;;
        "30")
            echo "1024:1024:192:192:48"
            ;;
        "40")
            echo "1024:1024:192:192:32"
            ;;
        "moe")
            echo "1024:1024:192:192:24"
            ;;
        "70"|"72")
            # Minimal sequence length to cap KV cache, but still meet coverage floors.
            echo "128:128:192:192:2"
            ;;
        *)
            # Safe default
            echo "1024:1024:192:192:32"
            ;;
    esac
}

_bootstrap_replicates_floor_for_tier() {
    local tier="${1:-balanced}"
    case "${tier}" in
        conservative)
            echo "1500"
            ;;
        balanced)
            echo "1200"
            ;;
        aggressive)
            echo "800"
            ;;
        *)
            echo "1200"
            ;;
    esac
}

_resolve_bootstrap_replicates() {
    local model_size="$1"
    local tier="${2:-balanced}"

    local bootstrap_replicates=2000
    if _is_large_model "${model_size}"; then
        bootstrap_replicates=1000
    fi
    if [[ -n "${INVARLOCK_BOOTSTRAP_N:-}" ]]; then
        bootstrap_replicates="${INVARLOCK_BOOTSTRAP_N}"
    fi

    local floor=""
    floor="$(_bootstrap_replicates_floor_for_tier "${tier}")"
    if [[ "${bootstrap_replicates}" =~ ^[0-9]+$ && "${floor}" =~ ^[0-9]+$ ]]; then
        if [[ "${bootstrap_replicates}" -lt "${floor}" ]]; then
            bootstrap_replicates="${floor}"
        fi
    fi

    echo "${bootstrap_replicates}"
}

_default_ci_min_windows() {
    local seq_len="${1:-}"

    if [[ -n "${INVARLOCK_CERT_MIN_WINDOWS:-}" ]]; then
        echo "${INVARLOCK_CERT_MIN_WINDOWS}"
        return
    fi

    local default_windows=256
    # The balanced tier enforces a 50k token minimum; short seq_len on short-text
    # datasets (e.g., WikiText-2) can fall below that floor due to padding.
    if [[ "${seq_len}" =~ ^[0-9]+$ && "${seq_len}" -le 256 ]]; then
        default_windows=352
    fi

    echo "${default_windows}"
}

# Wrapper to get model size - tries main script function first, then fallback
_estimate_model_size() {
    local model_path="$1"

    # Try main script's estimate_model_params first
    if type estimate_model_params &>/dev/null; then
        estimate_model_params "${model_path}"
        return
    fi

    # Fallback: detect from model name/path
    _get_model_size_from_name "${model_path}"
}

# Wrapper to get InvarLock config - tries main script function first, then fallback
_get_invarlock_config() {
    local model_size="$1"

    # Try main script's get_model_invarlock_config first
    if type get_model_invarlock_config &>/dev/null; then
        get_model_invarlock_config "${model_size}"
        return
    fi

    # Use fallback
    _get_model_invarlock_config_fallback "${model_size}"
}

_task_create_model_variant() {
    local baseline_path="$1"
    local output_path="$2"
    local edit_type="$3"
    local param1="${4:-}"
    local param2="${5:-}"
    local scope="${6:-}"
    local gpu_id="${7:-0}"

    if type create_model_variant &>/dev/null; then
        create_model_variant "${baseline_path}" "${output_path}" "${edit_type}" "${param1}" "${param2}" "${scope}" "${gpu_id}"
        return $?
    fi

    case "${edit_type}" in
        "quant_rtn")
            if ! type create_edited_model &>/dev/null; then
                echo "ERROR: create_edited_model not available" >&2
                return 1
            fi
            create_edited_model "${baseline_path}" "${output_path}" "quant_rtn" "${param1}" "${param2}" "${scope}" "${gpu_id}"
            ;;
        "fp8_quant")
            if ! type create_fp8_model &>/dev/null; then
                echo "ERROR: create_fp8_model not available" >&2
                return 1
            fi
            create_fp8_model "${baseline_path}" "${output_path}" "${param1}" "${scope}" "${gpu_id}"
            ;;
        "magnitude_prune")
            if ! type create_pruned_model &>/dev/null; then
                echo "ERROR: create_pruned_model not available" >&2
                return 1
            fi
            create_pruned_model "${baseline_path}" "${output_path}" "${param1}" "${scope}" "${gpu_id}"
            ;;
        "lowrank_svd")
            if ! type create_lowrank_model &>/dev/null; then
                echo "ERROR: create_lowrank_model not available" >&2
                return 1
            fi
            create_lowrank_model "${baseline_path}" "${output_path}" "${param1}" "${scope}" "${gpu_id}"
            ;;
        *)
            echo "ERROR: Unknown edit type: ${edit_type}" >&2
            return 1
            ;;
    esac
}

_task_get_model_revision() {
    local model_id="$1"
    if type pack_model_revision &>/dev/null; then
        pack_model_revision "${model_id}"
        return
    fi
    local path="${PACK_MODEL_REVISIONS_FILE:-${OUTPUT_DIR:-}/state/model_revisions.json}"
    [[ -f "${path}" ]] || return 0
    _cmd_python "${SCRIPT_DIR}/../python/get_model_revision.py" "${path}" "${model_id}" 2>/dev/null
}

# Check if model is large (30B+) and needs special handling.
# Changed threshold from 70 to 30 to fix hang on 30-40B models:
# - Skips overhead check (avoids loading model twice, which can exceed 180GB)
_is_large_model() {
    local model_size="$1"
    if [[ "${model_size}" == "moe" ]]; then
        return 0
    fi
    if [[ "${model_size}" =~ ^[0-9]+$ ]]; then
        [[ ${model_size} -ge 30 ]]
        return
    fi
    [[ "${model_size}" =~ 30 || "${model_size}" =~ 32 || "${model_size}" =~ 34 || "${model_size}" =~ 40 || "${model_size}" =~ 70 || "${model_size}" =~ 72 || "${model_size}" =~ 65 || "${model_size}" =~ 80 || "${model_size}" =~ 90 ]]
}

# Resolve the concrete InvarLock adapter name for a model path/ID.
_resolve_invarlock_adapter() {
    local model_id="$1"
    if [[ -z "${model_id}" ]]; then
        return 1
    fi
    _cmd_python "${SCRIPT_DIR}/../python/resolve_invarlock_adapter.py" "${model_id}"
}

_validate_certify_baseline_report() {
    local report_path="$1"
    local expected_adapter="$2"
    local expected_profile="$3"
    local expected_tier="$4"

    if [[ -z "${report_path}" || ! -f "${report_path}" ]]; then
        return 1
    fi

    _cmd_python "${SCRIPT_DIR}/../python/validate_baseline_report.py" \
        "${report_path}" "${expected_adapter}" "${expected_profile}" "${expected_tier}"
}

_ensure_certify_baseline_report() {
    local baseline_root="$1"
    local abs_baseline_path="$2"
    local profile_flag="$3"
    local tier="$4"
    local seq_len="$5"
    local stride="$6"
    local preview_n="$7"
    local final_n="$8"
    local eval_batch="$9"
    local bootstrap_replicates="${10}"
    local model_size="${11}"
    local log_file="${12}"

    mkdir -p "${baseline_root}"
    local abs_baseline_root
    abs_baseline_root="$(cd "${baseline_root}" && pwd)"

    local baseline_report_file="${abs_baseline_root}/baseline_report.json"

    local adapter_name
    adapter_name="$(_resolve_invarlock_adapter "${abs_baseline_path}" 2>/dev/null || true)"
    adapter_name="$(printf '%s' "${adapter_name}" | xargs)"
    if [[ -z "${adapter_name}" ]]; then
        # Fallback for odd environments; must match what invarlock certify will resolve.
        adapter_name="hf_causal"
    fi

    if [[ -f "${baseline_report_file}" ]]; then
        if _validate_certify_baseline_report "${baseline_report_file}" "${adapter_name}" "${profile_flag}" "${tier}" 2>/dev/null; then
            echo "${baseline_report_file}"
            return 0
        fi
        rm -f "${baseline_report_file}"
    fi

    local lock_dir="${abs_baseline_root}/.baseline_lock"
    if mkdir "${lock_dir}" 2>/dev/null; then
        # Re-check after acquiring the lock.
        if [[ -f "${baseline_report_file}" ]]; then
            if _validate_certify_baseline_report "${baseline_report_file}" "${adapter_name}" "${profile_flag}" "${tier}" 2>/dev/null; then
                rmdir "${lock_dir}" 2>/dev/null || true
                echo "${baseline_report_file}"
                return 0
            fi
            rm -f "${baseline_report_file}"
        fi

        echo "  Generating reusable baseline report (adapter=${adapter_name}, tier=${tier})" >> "${log_file}"

        local baseline_config_root="${abs_baseline_root}/config_root"
        mkdir -p "${baseline_config_root}/runtime/profiles"
        cat > "${baseline_config_root}/runtime/profiles/ci.yaml" << YAML
model:
  device_map: "auto"
  dtype: "bfloat16"
  trust_remote_code: true
  low_cpu_mem_usage: true
dataset:
  seq_len: ${seq_len}
  stride: ${stride}
  preview_n: ${preview_n}
  final_n: ${final_n}
eval:
  bootstrap:
    replicates: ${bootstrap_replicates}
    alpha: 0.05
YAML

        local guards_order_csv="${PACK_GUARDS_ORDER:-}"
        local -a raw_guards_order=()
        if [[ -n "${guards_order_csv}" ]]; then
            IFS=',' read -ra raw_guards_order <<< "${guards_order_csv}"
        fi
        local -a guards_order=()
        local g
        for g in "${raw_guards_order[@]}"; do
            g="$(echo "${g}" | xargs)"
            [[ -z "${g}" ]] && continue
            guards_order+=("${g}")
        done
        if [[ ${#guards_order[@]} -eq 0 ]]; then
            guards_order=("invariants" "spectral" "rmt" "variance" "invariants")
        fi
        local guards_order_yaml=""
        for g in "${guards_order[@]}"; do
            guards_order_yaml+=$'    - '"${g}"$'\n'
        done

        local baseline_yaml="${abs_baseline_root}/baseline_noop.yaml"
        cat > "${baseline_yaml}" << YAML
model:
  id: "${abs_baseline_path}"
  adapter: "${adapter_name}"
  device: "auto"
  device_map: "auto"
  dtype: "bfloat16"
  trust_remote_code: true
  low_cpu_mem_usage: true

dataset:
  provider: "${INVARLOCK_DATASET:-wikitext2}"
  split: validation
  seq_len: ${seq_len}
  stride: ${stride}
  preview_n: ${preview_n}
  final_n: ${final_n}
  seed: 42

edit:
  name: "noop"
  plan: {}

guards:
  order:
${guards_order_yaml}

auto:
  enabled: true
  tier: "${tier}"

eval:
  bootstrap:
    replicates: ${bootstrap_replicates}
    alpha: 0.05
  batch_size: ${eval_batch}
YAML

        local baseline_out="${abs_baseline_root}/runs"
        mkdir -p "${baseline_out}"

        local -a extra_env=()
        if _is_large_model "${model_size}"; then
            extra_env+=(INVARLOCK_SKIP_OVERHEAD_CHECK=1)
        fi
        extra_env+=(INVARLOCK_STORE_EVAL_WINDOWS=1)
        extra_env+=("INVARLOCK_CONFIG_ROOT=${baseline_config_root}")

        local exit_code=0
        env "${extra_env[@]}" invarlock run \
            --config "${baseline_yaml}" \
            --profile "${profile_flag}" \
            --tier "${tier}" \
            --out "${baseline_out}" \
            --edit-label "noop" >> "${log_file}" 2>&1 || exit_code=$?

        if [[ ${exit_code} -eq 0 ]]; then
            local report_file
            report_file=$(find "${baseline_out}" -mindepth 2 -maxdepth 2 -name "report.json" -type f 2>/dev/null | sort | tail -1)
            if [[ -n "${report_file}" && -f "${report_file}" ]]; then
                local tmp_report="${baseline_report_file}.tmp"
                cp "${report_file}" "${tmp_report}" 2>/dev/null || true
                if [[ -f "${tmp_report}" ]]; then
                    mv "${tmp_report}" "${baseline_report_file}" 2>/dev/null || true
                fi
            fi
        fi

        rmdir "${lock_dir}" 2>/dev/null || true

        if [[ -f "${baseline_report_file}" ]] && _validate_certify_baseline_report "${baseline_report_file}" "${adapter_name}" "${profile_flag}" "${tier}" 2>/dev/null; then
            echo "${baseline_report_file}"
            return 0
        fi
        rm -f "${baseline_report_file}"
        return 1
    fi

    local wait_interval="${PACK_BASELINE_REPORT_WAIT_INTERVAL_SECS:-2}"
    local wait_secs="${PACK_BASELINE_REPORT_WAIT_SECS:-240}"
    if _is_large_model "${model_size}"; then
        wait_secs="${PACK_BASELINE_REPORT_WAIT_SECS_LARGE:-1800}"
    fi
    if ! [[ "${wait_interval}" =~ ^[0-9]+$ ]] || [[ "${wait_interval}" -lt 1 ]]; then
        wait_interval=2
    fi
    if ! [[ "${wait_secs}" =~ ^[0-9]+$ ]] || [[ "${wait_secs}" -lt 1 ]]; then
        wait_secs=240
    fi
    local wait_iters=$((wait_secs / wait_interval))
    if [[ "${wait_iters}" -lt 1 ]]; then
        wait_iters=1
    fi

    echo "  Waiting for baseline report to be generated by another worker... (timeout=${wait_secs}s)" >> "${log_file}"
    for _ in $(seq 1 "${wait_iters}"); do
        if [[ -f "${baseline_report_file}" ]] && _validate_certify_baseline_report "${baseline_report_file}" "${adapter_name}" "${profile_flag}" "${tier}" 2>/dev/null; then
            echo "${baseline_report_file}"
            return 0
        fi
        _sleep "${wait_interval}"
    done

    return 1
}

# Resolve an edit spec to concrete parameters and directory name.
# Returns JSON with status, edit_dir_name, and resolved params.
resolve_edit_params() {
    local model_output_dir="$1"
    local edit_spec="$2"
    local version_hint="${3:-}"

    _cmd_python "${SCRIPT_DIR}/../python/resolve_edit_params.py" \
        "${model_output_dir}" "${edit_spec}" "${version_hint}"
}

# Resolve task timeout in seconds (empty/0 disables).
_get_task_timeout() {
    local task_type="$1"
    local default_timeout="${TASK_TIMEOUT_DEFAULT:-}"
    local override_var="TASK_TIMEOUT_${task_type}"
    local override="${!override_var:-}"
    local timeout="${override:-${default_timeout}}"

    if [[ -z "${timeout}" || "${timeout}" == "0" || "${timeout}" == "none" ]]; then
        return
    fi
    if [[ "${timeout}" =~ ^[0-9]+$ ]]; then
        echo "${timeout}"
    fi
}

_kill_task_process_group() {
    local pid="$1"
    local pgid=""
    local self_pgid=""

    pgid=$(_cmd_ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ')
    self_pgid=$(_cmd_ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ')

    if [[ -n "${pgid}" && -n "${self_pgid}" && "${pgid}" != "${self_pgid}" ]]; then
        _cmd_kill -TERM -- "-${pgid}" 2>/dev/null || true
        _sleep 5
        _cmd_kill -KILL -- "-${pgid}" 2>/dev/null || true
    else
        _cmd_kill -TERM "${pid}" 2>/dev/null || true
        _sleep 5
        _cmd_kill -KILL "${pid}" 2>/dev/null || true
    fi
}

_write_model_profile() {
    local baseline_dir="$1"
    local model_id="$2"
    local profile_path="${baseline_dir}/model_profile.json"

    [[ -f "${profile_path}" ]] && return 0
    [[ -d "${baseline_dir}" ]] || return 1

    _cmd_python "${SCRIPT_DIR}/../python/write_model_profile.py" \
        "${baseline_dir}" "${model_id}" >/dev/null 2>&1 || true
}

# ============ TASK EXECUTOR ============

# Execute a task based on its type
# Usage: execute_task <task_file> <gpu_id> <output_dir>
execute_task() {
    local task_file="$1"
    local gpu_id="$2"
    local output_dir="$3"

    local task_id=$(get_task_id "${task_file}")
    local task_type=$(get_task_type "${task_file}")
    local model_id=$(get_task_field "${task_file}" "model_id")
    local model_name=$(get_task_field "${task_file}" "model_name")
    local params=$(get_task_params "${task_file}")

    # Get assigned GPUs from task file (multi-GPU support)
    # If assigned_gpus is set (e.g., "2,3,4,5" for 4-GPU tasks), use that
    # Otherwise fall back to the single gpu_id parameter
    local assigned_gpus=$(get_task_assigned_gpus "${task_file}")
    assigned_gpus="${assigned_gpus// /}"
    if [[ -z "${assigned_gpus}" || "${assigned_gpus}" == "null" ]]; then
        assigned_gpus="${gpu_id}"
    fi

    # Create task-specific log
    local task_log="${output_dir}/logs/tasks/${task_id}.log"
    mkdir -p "$(dirname "${task_log}")"

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Starting task: ${task_id}" >> "${task_log}"
    echo "  Type: ${task_type}" >> "${task_log}"
    echo "  Model: ${model_id}" >> "${task_log}"
    echo "  GPU(s): ${assigned_gpus}" >> "${task_log}"
    echo "  Params: ${params}" >> "${task_log}"

    # Set GPU(s) for this task - use assigned_gpus from task file (multi-GPU support)
    # This is set once here and inherited by all task function commands
    # This must be unconditional to ensure multi-GPU tasks get all their GPUs
    export CUDA_VISIBLE_DEVICES="${assigned_gpus}"
    export TASK_ID="${task_id}"
    export TASK_PARAMS="${params}"
    export TASK_TYPE="${task_type}"

    # Set PM acceptance range to avoid gate failures during validation
    # These bounds are calibrated for typical validation runs; adjust if needed
    export INVARLOCK_PM_ACCEPTANCE_MIN="${INVARLOCK_PM_ACCEPTANCE_MIN:-0.90}"
    export INVARLOCK_PM_ACCEPTANCE_MAX="${INVARLOCK_PM_ACCEPTANCE_MAX:-1.20}"

    local task_pid_file=""
    if [[ -n "${QUEUE_DIR:-}" && -d "${QUEUE_DIR}/running" ]]; then
        task_pid_file="${QUEUE_DIR}/running/${task_id}.pid"
    fi

    local exit_code=0
    local task_timeout=""
    task_timeout=$(_get_task_timeout "${task_type}")

    local job_control_enabled=0
    case $- in
        *m*)
            job_control_enabled=1
            ;;
    esac
    if [[ ${job_control_enabled} -eq 0 ]]; then
        set -m
    fi

    (
        local exit_code=0
        case "${task_type}" in
            SETUP_BASELINE)
                task_setup_baseline "${model_id}" "${model_name}" "${gpu_id}" "${output_dir}" "${task_log}" || exit_code=$?
                ;;
            CALIBRATION_RUN)
                local run=$(echo "${params}" | jq -r '.run // 1')
                local seed=$(echo "${params}" | jq -r '.seed // 42')
                task_calibration_run "${model_name}" "${gpu_id}" "${run}" "${seed}" "${output_dir}" "${task_log}" || exit_code=$?
                ;;
            CREATE_EDIT)
                local edit_spec=$(echo "${params}" | jq -r '.edit_spec // ""')
                local version=$(echo "${params}" | jq -r '.version // "clean"')
                task_create_edit "${model_name}" "${gpu_id}" "${edit_spec}" "${version}" "${output_dir}" "${task_log}" || exit_code=$?
                ;;
            CREATE_EDITS_BATCH)
                # v2.1.0: Batch edit creation - loads model once, creates all 8 edits
                local edit_specs=$(echo "${params}" | jq -r '.edit_specs // "[]"')
                task_create_edits_batch "${model_name}" "${gpu_id}" "${edit_specs}" "${output_dir}" "${task_log}" || exit_code=$?
                ;;
            CERTIFY_EDIT)
                local edit_spec=$(echo "${params}" | jq -r '.edit_spec // ""')
                local version=$(echo "${params}" | jq -r '.version // "clean"')
                local run=$(echo "${params}" | jq -r '.run // 1')
                task_certify_edit "${model_name}" "${gpu_id}" "${edit_spec}" "${version}" "${run}" "${output_dir}" "${task_log}" || exit_code=$?
                ;;
            CREATE_ERROR)
                local error_type=$(echo "${params}" | jq -r '.error_type // ""')
                task_create_error "${model_name}" "${gpu_id}" "${error_type}" "${output_dir}" "${task_log}" || exit_code=$?
                ;;
            CERTIFY_ERROR)
                local error_type=$(echo "${params}" | jq -r '.error_type // ""')
                task_certify_error "${model_name}" "${gpu_id}" "${error_type}" "${output_dir}" "${task_log}" || exit_code=$?
                ;;
            GENERATE_PRESET)
                task_generate_preset "${model_name}" "${output_dir}" "${task_log}" || exit_code=$?
                ;;
            *)
                echo "ERROR: Unknown task type: ${task_type}" >> "${task_log}"
                exit_code=1
                ;;
        esac

        exit ${exit_code}
    ) &

    local task_pid=$!
    if [[ ${job_control_enabled} -eq 0 ]]; then
        set +m
    fi
    if [[ -n "${task_pid_file}" ]]; then
        echo "${task_pid}" > "${task_pid_file}"
    fi

    local timeout_marker=""
    local timeout_pid=""
    if [[ -n "${task_timeout}" ]]; then
        timeout_marker="${output_dir}/logs/tasks/${task_id}.timeout"
        rm -f "${timeout_marker}"
        (
            _sleep "${task_timeout}"
            if _cmd_kill -0 "${task_pid}" 2>/dev/null; then
                echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Task ${task_id} exceeded timeout (${task_timeout}s), terminating" >> "${task_log}"
                echo "${task_timeout}" > "${timeout_marker}"
                _kill_task_process_group "${task_pid}"
            fi
        ) &
        timeout_pid=$!
    fi

    exit_code=0
    wait "${task_pid}" || exit_code=$?

    if [[ -n "${timeout_pid}" ]]; then
        _cmd_kill "${timeout_pid}" 2>/dev/null || true
        wait "${timeout_pid}" 2>/dev/null || true
    fi

    if [[ -n "${timeout_marker}" && -f "${timeout_marker}" ]]; then
        exit_code=124
        rm -f "${timeout_marker}"
    fi

    if [[ -n "${task_pid_file}" ]]; then
        rm -f "${task_pid_file}"
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Task ${task_id} finished with exit code: ${exit_code}" >> "${task_log}"

    return ${exit_code}
}

# ============ TASK: SETUP_BASELINE ============

# Download and setup baseline model
# Usage: task_setup_baseline <model_id> <model_name> <gpu_id> <output_dir> <log_file>
task_setup_baseline() {
    local model_id="$1"
    local model_name="$2"
    local gpu_id="$3"
    local output_dir="$4"
    local log_file="$5"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Setting up baseline: ${model_id}" >> "${log_file}"

    # Check if already exists (resume mode)
    if [[ -d "${baseline_dir}" && -f "${baseline_dir}/config.json" ]]; then
        echo "  Baseline already exists, skipping download" >> "${log_file}"
        # Store baseline path for other tasks
        echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
        # Also store original model_id for model size detection in other tasks
        echo "${model_id}" > "${model_output_dir}/.model_id"
        _write_model_profile "${baseline_dir}" "${model_id}"
        if type update_model_task_memory &>/dev/null; then
            update_model_task_memory "${model_name}" "${output_dir}" "${model_id}"
        fi
        return 0
    fi

    mkdir -p "${model_output_dir}"/{models,evals,certificates}

    # Use the main script's setup_model function if available
    if type setup_model &>/dev/null; then
        local baseline_path
        local exit_code=0
        baseline_path=$(setup_model "${model_id}" "${gpu_id}") || exit_code=$?

        if [[ ${exit_code} -eq 0 && -n "${baseline_path}" && -d "${baseline_path}" ]]; then
            echo "  Baseline ready at: ${baseline_path}" >> "${log_file}"
            echo "${baseline_path}" > "${model_output_dir}/.baseline_path"
            # Store original model_id for model size detection
            echo "${model_id}" > "${model_output_dir}/.model_id"
            _write_model_profile "${baseline_path}" "${model_id}"
            if type update_model_task_memory &>/dev/null; then
                update_model_task_memory "${model_name}" "${output_dir}" "${model_id}"
            fi
            return 0
        else
            echo "  ERROR: Failed to setup baseline" >> "${log_file}"
            return 1
        fi
    else
        # Inline implementation
        echo "  Downloading model ${model_id}..." >> "${log_file}"

        local revision=""
        revision=$(_task_get_model_revision "${model_id}" || true)
        if [[ -z "${revision}" ]]; then
            if [[ "${PACK_NET}" == "1" ]]; then
                echo "  ERROR: Missing pinned revision for ${model_id}; run preflight (--net 1)." >> "${log_file}"
            else
                echo "  ERROR: Offline mode requires model revisions. Run with --net 1 to preflight." >> "${log_file}"
            fi
            return 1
        fi

        if [[ "${PACK_NET}" != "1" ]]; then
            echo "  ERROR: Offline mode requested and baseline not cached for ${model_id}." >> "${log_file}"
            return 1
        fi

        # CUDA_VISIBLE_DEVICES is inherited from execute_task() for multi-GPU support
        local exit_code=0
        PACK_MODEL_REVISION="${revision}" \
            _cmd_python "${SCRIPT_DIR}/../python/download_baseline.py" \
                --model-id "${model_id}" \
                --output-dir "${baseline_dir}" >> "${log_file}" 2>&1 || exit_code=$?

        if [[ ${exit_code} -eq 0 && -f "${baseline_dir}/config.json" ]]; then
            echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
            # Store original model_id for model size detection
            echo "${model_id}" > "${model_output_dir}/.model_id"
            _write_model_profile "${baseline_dir}" "${model_id}"
            if type update_model_task_memory &>/dev/null; then
                update_model_task_memory "${model_name}" "${output_dir}" "${model_id}"
            fi
            return 0
        fi
        return 1
    fi
}

# ============ TASK: CALIBRATION_RUN ==========

# Run single InvarLock calibration
# Usage: task_calibration_run <model_name> <gpu_id> <run_num> <seed> <output_dir> <log_file>
task_calibration_run() {
    local model_name="$1"
    local gpu_id="$2"
    local run_num="$3"
    local seed="$4"
    local output_dir="$5"
    local log_file="$6"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)
    local model_id=$(cat "${model_output_dir}/.model_id" 2>/dev/null || true)
    local run_dir="${model_output_dir}/certificates/calibration/run_${run_num}"

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    # Check if already done
    if [[ -f "${run_dir}/baseline_report.json" || -f "${run_dir}/evaluation.cert.json" ]]; then
        echo "  Calibration run ${run_num} already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Running calibration run ${run_num} (seed=${seed})" >> "${log_file}"

    mkdir -p "${run_dir}"

    # Get model-aware config using wrapper functions (try main script, then fallback)
    # First try to get model size from baseline path, then from stored model_id
    local model_size
    model_size=$(_estimate_model_size "${baseline_path}")
    if [[ -z "${model_size}" || "${model_size}" == "7" ]] && [[ -n "${model_id}" ]]; then
        # Fallback: detect from model_id string
        model_size=$(_get_model_size_from_name "${model_id}")
    fi

    # Get configuration for this model size
    local config
    config=$(_get_invarlock_config "${model_size}")

    IFS=':' read -r seq_len stride preview_n final_n eval_batch <<< "${config}"
    local params_json="${TASK_PARAMS:-}"
    local applied_override=0
    if [[ -n "${params_json}" && "${params_json}" != "null" ]]; then
        local override_seq_len override_stride override_batch
        override_seq_len=$(echo "${params_json}" | jq -r '.seq_len // empty' 2>/dev/null)
        override_stride=$(echo "${params_json}" | jq -r '.stride // empty' 2>/dev/null)
        override_batch=$(echo "${params_json}" | jq -r '.batch_size // empty' 2>/dev/null)
        if [[ "${override_seq_len}" =~ ^[0-9]+$ ]]; then
            seq_len="${override_seq_len}"
            applied_override=1
        fi
        if [[ "${override_stride}" =~ ^[0-9]+$ ]]; then
            stride="${override_stride}"
            applied_override=1
        fi
        if [[ "${override_batch}" =~ ^[0-9]+$ ]]; then
            eval_batch="${override_batch}"
            applied_override=1
        fi
        if [[ "${stride}" -gt "${seq_len}" ]]; then
            stride=$((seq_len / 2))
            [[ ${stride} -lt 1 ]] && stride=1
            applied_override=1
        fi
    fi

    # Force non-overlapping windows during calibration to avoid pairing mismatches
    stride="${seq_len}"

    echo "  Model size: ${model_size}, Config: seq=${seq_len}, stride=${stride}, windows=${preview_n}+${final_n}, batch=${eval_batch}" >> "${log_file}"
    if [[ ${applied_override} -eq 1 ]]; then
        echo "  OOM override applied: seq=${seq_len}, stride=${stride}, batch=${eval_batch}" >> "${log_file}"
    fi

    # For large models, use INVARLOCK_SKIP_OVERHEAD_CHECK to avoid loading
    # both baseline and edited models simultaneously (which would exceed 180GB).
    local profile_flag="ci"
    local min_windows
    min_windows="$(_default_ci_min_windows "${seq_len}")"
    if [[ "${profile_flag}" == "ci" && "${min_windows}" =~ ^[0-9]+$ && "${min_windows}" -gt 0 ]]; then
        if [[ "${preview_n}" -lt "${min_windows}" || "${final_n}" -lt "${min_windows}" ]]; then
            preview_n="${min_windows}"
            final_n="${min_windows}"
            applied_override=1
            echo "  CI window override: preview=${preview_n}, final=${final_n}" >> "${log_file}"
        fi
    fi
    local bootstrap_replicates=2000


    if _is_large_model "${model_size}"; then
        bootstrap_replicates=1000
    fi
    if [[ -n "${INVARLOCK_BOOTSTRAP_N:-}" ]]; then
        bootstrap_replicates="${INVARLOCK_BOOTSTRAP_N}"
    fi

    echo "  Calibration: enforcing window_overlap_fraction=0.0" >> "${log_file}"
    local -a extra_env=(INVARLOCK_WINDOW_OVERLAP_FRACTION=0.0 INVARLOCK_SKIP_OVERHEAD_CHECK=1)
    if _is_large_model "${model_size}"; then
        echo "  Large model (${model_size}): SKIP_OVERHEAD_CHECK=1" >> "${log_file}"
    fi

    local config_root_base
    config_root_base="$(cd "${run_dir}" && pwd)"
    local config_root="${config_root_base}/config_root"
    mkdir -p "${config_root}/runtime/profiles"
    cat > "${config_root}/runtime/profiles/ci.yaml" << YAML
model:
  device_map: "auto"
  dtype: "bfloat16"
  trust_remote_code: true
  low_cpu_mem_usage: true
dataset:
  preview_n: ${preview_n}
  final_n: ${final_n}
eval:
  bootstrap:
    replicates: ${bootstrap_replicates}
    alpha: 0.05
YAML

    extra_env+=("INVARLOCK_CONFIG_ROOT=${config_root}")

    # Generate config YAML
    local config_yaml="${run_dir}/calibration_config.yaml"
    local guards_order_csv="${PACK_GUARDS_ORDER:-}"
    local -a raw_guards_order=()
    if [[ -n "${guards_order_csv}" ]]; then
        IFS=',' read -ra raw_guards_order <<< "${guards_order_csv}"
    fi
    local -a guards_order=()
    local g
    for g in "${raw_guards_order[@]}"; do
        g="$(echo "${g}" | xargs)"
        [[ -z "${g}" ]] && continue
        guards_order+=("${g}")
    done
    if [[ ${#guards_order[@]} -eq 0 ]]; then
        guards_order=("invariants" "spectral" "rmt" "variance" "invariants")
    fi
    local guards_order_yaml=""
    for g in "${guards_order[@]}"; do
        guards_order_yaml+=$'    - '"${g}"$'\n'
    done

    cat > "${config_yaml}" << YAML_EOF
model:
  id: "${baseline_path}"
  adapter: "hf_auto"
  device: "auto"
  device_map: "auto"
  dtype: "bfloat16"
  trust_remote_code: true
  low_cpu_mem_usage: true

dataset:
  provider: "${INVARLOCK_DATASET:-wikitext2}"
  preview_n: ${preview_n}
  final_n: ${final_n}
  seq_len: ${seq_len}
  stride: ${stride}
  seed: ${seed}

edit:
  name: "noop"

guards:
  order:
${guards_order_yaml}

eval:
  bootstrap:
    replicates: ${bootstrap_replicates}
    parallel: true
  batch_size: ${eval_batch}
  window_overlap_fraction: 0.0

auto:
  enabled: true
  tier: "${INVARLOCK_TIER:-balanced}"
  probes: 0
YAML_EOF

    local profile_flag="ci"

    # CUDA_VISIBLE_DEVICES is inherited from execute_task() for multi-GPU support
    local exit_code=0
    env "${extra_env[@]}" invarlock run \
        --config "${config_yaml}" \
        --profile "${profile_flag}" \
        --out "${run_dir}" \
        >> "${log_file}" 2>&1 || exit_code=$?

    # Copy report to standard location
    local report_file=$(find "${run_dir}" -name "report*.json" -type f 2>/dev/null | head -1)
    if [[ -n "${report_file}" ]]; then
        cp "${report_file}" "${run_dir}/baseline_report.json" 2>/dev/null || true
        _cmd_python "${SCRIPT_DIR}/../python/certificate_from_report.py" \
            --report "${report_file}" \
            --out "${run_dir}/evaluation.cert.json" >> "${log_file}" 2>&1 || true
    fi

    return ${exit_code}
}

# ============ TASK: GENERATE_PRESET ============

# Generate calibrated preset from calibration runs
# Usage: task_generate_preset <model_name> <output_dir> <log_file>
task_generate_preset() {
    local model_name="$1"
    local output_dir="$2"
    local log_file="$3"

    local model_output_dir="${output_dir}/${model_name}"
    local cal_dir="${model_output_dir}/certificates/calibration"
    local preset_dir="${output_dir}/presets"
    local preset_base="${preset_dir}/calibrated_preset_${model_name}"

    if [[ -f "${preset_base}.yaml" || -f "${preset_base}.json" ]]; then
        echo "  Preset already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Generating calibrated preset" >> "${log_file}"

    mkdir -p "${preset_dir}"

    # Get baseline path and model_id to estimate model size
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)
    local model_id=$(cat "${model_output_dir}/.model_id" 2>/dev/null || true)

    # Get model-aware config for seq_len/stride using wrapper functions
    # (these handle fallback when main script functions aren't available)
    local model_size
    model_size=$(_estimate_model_size "${baseline_path}")
    if [[ -z "${model_size}" || "${model_size}" == "7" ]] && [[ -n "${model_id}" ]]; then
        # Fallback: detect from model_id string
        model_size=$(_get_model_size_from_name "${model_id}")
    fi

    # Get config using wrapper (tries main script, then fallback)
    local config
    config=$(_get_invarlock_config "${model_size}")

    IFS=':' read -r seq_len stride preview_n final_n eval_batch <<< "${config}"

    # Export for use in Python script
    export PRESET_SEQ_LEN="${seq_len}"
    export PRESET_STRIDE="${stride}"
    export PRESET_PREVIEW_N="${preview_n}"
    export PRESET_FINAL_N="${final_n}"

    local exit_code=0
    local proof_packs_dir
    proof_packs_dir="$(cd "${SCRIPT_DIR}/.." && pwd)"
    local generator="${proof_packs_dir}/python/preset_generator.py"
    _cmd_python "${generator}" \
        --cal-dir "${cal_dir}" \
        --preset-file "${preset_base}.yaml" \
        --model-name "${model_name}" \
        --model-path "${baseline_path}" \
        --tier "${INVARLOCK_TIER:-balanced}" \
        --dataset-provider "${INVARLOCK_DATASET:-wikitext2}" \
        --seq-len "${seq_len}" \
        --stride "${stride}" \
        --preview-n "${preview_n}" \
        --final-n "${final_n}" \
        --edit-types "quant_rtn,fp8_quant,magnitude_prune,lowrank_svd" \
        >> "${log_file}" 2>&1 || exit_code=$?

    return ${exit_code}
}

# ============ TASK: CREATE_EDIT ============

# Create edited model
# Usage: task_create_edit <model_name> <gpu_id> <edit_spec> <version> <output_dir> <log_file>
task_create_edit() {
    local model_name="$1"
    local gpu_id="$2"
    local edit_spec="$3"
    local version="$4"
    local output_dir="$5"
    local log_file="$6"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    local resolved
    resolved=$(resolve_edit_params "${model_output_dir}" "${edit_spec}" "${version}")
    local status
    status=$(echo "${resolved}" | jq -r '.status')
    if [[ "${status}" == "skipped" ]]; then
        echo "  Clean edit skipped by tuned preset: ${edit_spec}" >> "${log_file}"
        return 0
    fi
    if [[ "${status}" != "selected" ]]; then
        echo "ERROR: Unable to resolve edit spec (${edit_spec}): ${status}" >> "${log_file}"
        return 1
    fi

    local edit_type param1 param2 scope edit_dir_name
    edit_type=$(echo "${resolved}" | jq -r '.edit_type')
    param1=$(echo "${resolved}" | jq -r '.param1')
    param2=$(echo "${resolved}" | jq -r '.param2')
    scope=$(echo "${resolved}" | jq -r '.scope')
    edit_dir_name=$(echo "${resolved}" | jq -r '.edit_dir_name')
    if [[ -z "${edit_dir_name}" || "${edit_dir_name}" == "null" ]]; then
        echo "ERROR: Empty edit_dir_name for ${edit_spec}" >> "${log_file}"
        return 1
    fi

    local edit_path="${model_output_dir}/models/${edit_dir_name}"

    # Check if already exists
    if [[ -d "${edit_path}" && -f "${edit_path}/config.json" ]]; then
        echo "  Edit ${edit_dir_name} already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Creating edit: ${edit_dir_name}" >> "${log_file}"

    local create_rc=0
    _task_create_model_variant "${baseline_path}" "${edit_path}" "${edit_type}" "${param1}" "${param2}" "${scope}" "${gpu_id}" >> "${log_file}" 2>&1 || create_rc=$?
    if [[ ${create_rc} -ne 0 ]]; then
        return 1
    fi

    # Verify creation
    if [[ -d "${edit_path}" && -f "${edit_path}/config.json" ]]; then
        echo "  Created: ${edit_path}" >> "${log_file}"
        return 0
    else
        echo "  ERROR: Failed to create edit" >> "${log_file}"
        return 1
    fi
}

# ============ TASK: CREATE_EDITS_BATCH ============

# Create all edited models with single model load (Batch optimization)
# This loads the baseline model once and creates all 8 edits, avoiding 8× model load overhead
# Usage: task_create_edits_batch <model_name> <gpu_id> <edit_specs_json> <output_dir> <log_file>
task_create_edits_batch() {
    local model_name="$1"
    local gpu_id="$2"
    local edit_specs_json="$3"
    local output_dir="$4"
    local log_file="$5"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Creating batch edits (8 edits with single model load)" >> "${log_file}"
    echo "  Baseline: ${baseline_path}" >> "${log_file}"

    # Process each edit spec using Python for efficient batch creation
    # CUDA_VISIBLE_DEVICES is inherited from execute_task() for multi-GPU support
    local exit_code=0
    _cmd_python "${SCRIPT_DIR}/../python/create_edits_batch.py" \
        --baseline "${baseline_path}" \
        --model-output-dir "${model_output_dir}" \
        --edit-specs-json "${edit_specs_json}" >> "${log_file}" 2>&1 || exit_code=$?

    if [[ ${exit_code} -eq 0 ]]; then
        echo "  Batch edit creation complete" >> "${log_file}"
    else
        echo "  ERROR: Batch edit creation failed" >> "${log_file}"
    fi

    return ${exit_code}
}

# ============ TASK: CERTIFY_EDIT ============

# Run InvarLock certify on edited model
# Usage: task_certify_edit <model_name> <gpu_id> <edit_spec> <version> <run_num> <output_dir> <log_file>
task_certify_edit() {
    local model_name="$1"
    local gpu_id="$2"
    local edit_spec="$3"
    local version="$4"
    local run_num="$5"
    local output_dir="$6"
    local log_file="$7"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)
    local model_id=$(cat "${model_output_dir}/.model_id" 2>/dev/null || true)
    local preset_dir="${output_dir}/presets"

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    local resolved
    resolved=$(resolve_edit_params "${model_output_dir}" "${edit_spec}" "${version}")
    local status
    status=$(echo "${resolved}" | jq -r '.status')
    if [[ "${status}" == "skipped" ]]; then
        echo "  Clean edit skipped by tuned preset: ${edit_spec}" >> "${log_file}"
        return 0
    fi
    if [[ "${status}" != "selected" ]]; then
        echo "ERROR: Unable to resolve edit spec (${edit_spec}): ${status}" >> "${log_file}"
        return 1
    fi
    local edit_dir_name
    edit_dir_name=$(echo "${resolved}" | jq -r '.edit_dir_name')

    local edit_path="${model_output_dir}/models/${edit_dir_name}"
    local cert_dir="${model_output_dir}/certificates/${edit_dir_name}/run_${run_num}"
    local cert_file="${cert_dir}/evaluation.cert.json"

    if [[ ! -d "${edit_path}" ]]; then
        echo "ERROR: Edit model not found: ${edit_path}" >> "${log_file}"
        return 1
    fi

    local abs_baseline_path
    abs_baseline_path="$(cd "$(dirname "${baseline_path}")" && pwd)/$(basename "${baseline_path}")"
    local abs_edit_path
    abs_edit_path="$(cd "$(dirname "${edit_path}")" && pwd)/$(basename "${edit_path}")"

    if [[ -f "${cert_file}" ]]; then
        echo "  Certification for ${edit_dir_name} run ${run_num} already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Certifying: ${edit_dir_name} run ${run_num}" >> "${log_file}"

    mkdir -p "${cert_dir}"

    local abs_cert_dir
    abs_cert_dir="$(cd "${cert_dir}" && pwd)"
    local abs_log_file
    abs_log_file="$(cd "$(dirname "${log_file}")" && pwd)/$(basename "${log_file}")"
    cert_dir="${abs_cert_dir}"
    cert_file="${cert_dir}/evaluation.cert.json"
    log_file="${abs_log_file}"

    # Get model size for config and profile decision
    local model_size
    model_size=$(_estimate_model_size "${baseline_path}")
    if [[ -z "${model_size}" || "${model_size}" == "7" ]] && [[ -n "${model_id}" ]]; then
        model_size=$(_get_model_size_from_name "${model_id}")
    fi

    # Get model-aware config for window counts (needed for CI override)
    local config seq_len stride preview_n final_n eval_batch
    config=$(_get_invarlock_config "${model_size}")
    IFS=':' read -r seq_len stride preview_n final_n eval_batch <<< "${config}"
    local params_json="${TASK_PARAMS:-}"
    local applied_override=0
    if [[ -n "${params_json}" && "${params_json}" != "null" ]]; then
        local override_seq_len override_stride
        override_seq_len=$(echo "${params_json}" | jq -r '.seq_len // empty' 2>/dev/null)
        override_stride=$(echo "${params_json}" | jq -r '.stride // empty' 2>/dev/null)
        if [[ "${override_seq_len}" =~ ^[0-9]+$ ]]; then
            seq_len="${override_seq_len}"
            applied_override=1
        fi
        if [[ "${override_stride}" =~ ^[0-9]+$ ]]; then
            stride="${override_stride}"
            applied_override=1
        fi
        if [[ "${stride}" -gt "${seq_len}" ]]; then
            stride=$((seq_len / 2))
            [[ ${stride} -lt 1 ]] && stride=1
            applied_override=1
        fi
        if [[ ${applied_override} -eq 1 ]]; then
            echo "  OOM override: seq=${seq_len}, stride=${stride}" >> "${log_file}"
        fi
    fi
    if [[ "${stride}" -ne "${seq_len}" ]]; then
        stride="${seq_len}"
        applied_override=1
        echo "  Pairing override: seq=${seq_len}, stride=${stride}" >> "${log_file}"
    fi
    # For large models, use INVARLOCK_SKIP_OVERHEAD_CHECK to avoid loading
    # both baseline and edited models simultaneously.
    local profile_flag="ci"
    local min_windows
    min_windows="$(_default_ci_min_windows "${seq_len}")"
    if [[ "${profile_flag}" == "ci" && "${min_windows}" =~ ^[0-9]+$ && "${min_windows}" -gt 0 ]]; then
        if [[ "${preview_n}" -lt "${min_windows}" || "${final_n}" -lt "${min_windows}" ]]; then
            preview_n="${min_windows}"
            final_n="${min_windows}"
            applied_override=1
            echo "  CI window override: preview=${preview_n}, final=${final_n}" >> "${log_file}"
        fi
    fi
    local tier="${INVARLOCK_TIER:-balanced}"
    local bootstrap_replicates
    bootstrap_replicates="$(_resolve_bootstrap_replicates "${model_size}" "${tier}")"
    local baseline_report_root="${model_output_dir}/baseline_reports/${profile_flag}_${tier}_seq${seq_len}_pv${preview_n}_fn${final_n}"
    local baseline_report_file=""
    baseline_report_file=$(
        _ensure_certify_baseline_report \
            "${baseline_report_root}" \
            "${abs_baseline_path}" \
            "${profile_flag}" \
            "${tier}" \
            "${seq_len}" \
            "${stride}" \
            "${preview_n}" \
            "${final_n}" \
            "${eval_batch}" \
            "${bootstrap_replicates}" \
            "${model_size}" \
            "${log_file}" \
            || true
    )
    local -a baseline_report_args=()
    if [[ -n "${baseline_report_file}" && -f "${baseline_report_file}" ]]; then
        baseline_report_args=(--baseline-report "${baseline_report_file}")
        echo "  Reusing baseline report: ${baseline_report_file}" >> "${log_file}"
    else
        echo "  WARNING: Baseline report unavailable; will run per-cert baseline evaluation" >> "${log_file}"
    fi

    local -a extra_env=()
    if _is_large_model "${model_size}"; then
        extra_env+=(INVARLOCK_SKIP_OVERHEAD_CHECK=1)
        echo "  Large model (${model_size}): SKIP_OVERHEAD_CHECK=1" >> "${log_file}"
    fi
    extra_env+=(INVARLOCK_STORE_EVAL_WINDOWS=1)

    local config_root_base
    config_root_base="$(cd "${cert_dir}" && pwd)"
    local config_root="${config_root_base}/config_root"
    mkdir -p "${config_root}/runtime/profiles"
    cat > "${config_root}/runtime/profiles/ci.yaml" << YAML
model:
  device_map: "auto"
  dtype: "bfloat16"
  trust_remote_code: true
  low_cpu_mem_usage: true
dataset:
  seq_len: ${seq_len}
  stride: ${stride}
  preview_n: ${preview_n}
  final_n: ${final_n}
eval:
  bootstrap:
    replicates: ${bootstrap_replicates}
    alpha: 0.05
YAML

    extra_env+=("INVARLOCK_CONFIG_ROOT=${config_root}")

    local edit_type=""
    IFS=':' read -r edit_type _ _ _ <<< "${edit_spec}"

    # Find calibrated preset (must have seq_len/stride embedded)
    local preset_file=""
    if [[ -n "${edit_type}" ]]; then
        for ext in yaml json; do
            local f="${preset_dir}/calibrated_preset_${model_name}__${edit_type}.${ext}"
            if [[ -f "${f}" ]]; then
                preset_file="${f}"
                echo "  Using edit-type preset: ${preset_file}" >> "${log_file}"
                break
            fi
        done
    fi
    if [[ -z "${preset_file}" ]]; then
    for ext in yaml json; do
        local f="${preset_dir}/calibrated_preset_${model_name}.${ext}"
        if [[ -f "${f}" ]]; then
            preset_file="${f}"
            break
        fi
    done
    fi

    # If no preset found, we need to create one with model-specific params
    if [[ -z "${preset_file}" || ! -f "${preset_file}" ]]; then
        echo "  WARNING: No preset found for ${model_name}, creating minimal preset" >> "${log_file}"

        # Config already parsed above (seq_len, stride, preview_n, final_n, eval_batch)
        # Create minimal preset with seq_len/stride
        mkdir -p "${preset_dir}"
        preset_file="${preset_dir}/calibrated_preset_${model_name}.yaml"
        cat > "${preset_file}" << PRESET_YAML
dataset:
  provider: wikitext2
  split: validation
  seq_len: ${seq_len}
  stride: ${stride}
  preview_n: ${preview_n}
  final_n: ${final_n}
  seed: 42
PRESET_YAML
        echo "  Created preset: ${preset_file}" >> "${log_file}"
    fi

    # Run certify in isolated working directory to avoid temp file race conditions
    # (invarlock creates .certify_tmp/ in current directory which conflicts in parallel runs)
    local work_dir="${cert_dir}/.workdir"
    mkdir -p "${work_dir}"
    local abs_preset_file
    abs_preset_file="$(cd "$(dirname "${preset_file}")" && pwd)/$(basename "${preset_file}")"
    local edit_label
    edit_label="$(basename "${abs_edit_path}")"
    edit_label="${edit_label%_stress}"
    edit_label="${edit_label%_clean}"
    if [[ -z "${edit_label}" ]]; then
        edit_label="custom"
    fi

    # CUDA_VISIBLE_DEVICES is inherited from execute_task() for multi-GPU support
    local exit_code=0
    (
        cd "${work_dir}" || exit 1
        env "${extra_env[@]}" invarlock certify \
            --source "${abs_baseline_path}" \
            "${baseline_report_args[@]}" \
            --edited "${abs_edit_path}" \
            --edit-label "${edit_label}" \
            --profile "${profile_flag}" \
            --tier "${tier}" \
            --out "${cert_dir}" \
            --cert-out "${cert_dir}" \
            --preset "${abs_preset_file}" >> "${log_file}" 2>&1
    ) || exit_code=$?

    # Find and copy certificate (only the canonical cert)
    if [[ ! -f "${cert_file}" ]]; then
        local found_cert
        found_cert=$(find "${cert_dir}" -name "evaluation.cert.json" -type f 2>/dev/null | head -1)
        if [[ -n "${found_cert}" && -f "${found_cert}" && "${found_cert}" != "${cert_file}" ]]; then
            cp "${found_cert}" "${cert_file}" 2>/dev/null || true
        fi
    fi

    return ${exit_code}
}

# ============ TASK: CREATE_ERROR ============

# Create error-injected model
# Usage: task_create_error <model_name> <gpu_id> <error_type> <output_dir> <log_file>
task_create_error() {
    local model_name="$1"
    local gpu_id="$2"
    local error_type="$3"
    local output_dir="$4"
    local log_file="$5"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)
    local error_path="${model_output_dir}/models/error_${error_type}"

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    if [[ -d "${error_path}" && -f "${error_path}/config.json" ]]; then
        echo "  Error model ${error_type} already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Creating error model: ${error_type}" >> "${log_file}"

    if type create_error_model &>/dev/null; then
        create_error_model "${baseline_path}" "${error_path}" "${error_type}" "${gpu_id}" >> "${log_file}" 2>&1
    else
        echo "ERROR: create_error_model not available" >> "${log_file}"
        return 1
    fi

    if [[ -d "${error_path}" && -f "${error_path}/config.json" ]]; then
        echo "  Created: ${error_path}" >> "${log_file}"
        return 0
    else
        echo "  ERROR: Failed to create error model" >> "${log_file}"
        return 1
    fi
}

# ============ TASK: CERTIFY_ERROR ============

# Certify error-injected model
# Usage: task_certify_error <model_name> <gpu_id> <error_type> <output_dir> <log_file>
task_certify_error() {
    local model_name="$1"
    local gpu_id="$2"
    local error_type="$3"
    local output_dir="$4"
    local log_file="$5"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)
    local model_id=$(cat "${model_output_dir}/.model_id" 2>/dev/null || true)
    local error_path="${model_output_dir}/models/error_${error_type}"
    local cert_dir="${model_output_dir}/certificates/errors/${error_type}"
    local cert_file="${cert_dir}/evaluation.cert.json"
    local preset_dir="${output_dir}/presets"

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    if [[ ! -d "${error_path}" ]]; then
        echo "ERROR: Error model not found: ${error_path}" >> "${log_file}"
        return 1
    fi

    local abs_baseline_path
    abs_baseline_path="$(cd "$(dirname "${baseline_path}")" && pwd)/$(basename "${baseline_path}")"
    local abs_error_path
    abs_error_path="$(cd "$(dirname "${error_path}")" && pwd)/$(basename "${error_path}")"

    if [[ -f "${cert_file}" ]]; then
        echo "  Certification for error ${error_type} already exists, skipping" >> "${log_file}"
        return 0
    fi

    # Repair known config inconsistencies for missing_tensors error models created by older pack versions.
    if [[ "${error_type}" == "missing_tensors" ]]; then
        local repair_script="${SCRIPT_DIR}/../python/repair_missing_tensors_config.py"
        if [[ -f "${repair_script}" && -f "${abs_baseline_path}/config.json" && -f "${abs_error_path}/config.json" ]]; then
            _cmd_python "${repair_script}" "${abs_baseline_path}/config.json" "${abs_error_path}/config.json" >> "${log_file}" 2>&1 || true
        fi
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Certifying error model: ${error_type}" >> "${log_file}"

    mkdir -p "${cert_dir}"

    local abs_cert_dir
    abs_cert_dir="$(cd "${cert_dir}" && pwd)"
    local abs_log_file
    abs_log_file="$(cd "$(dirname "${log_file}")" && pwd)/$(basename "${log_file}")"
    cert_dir="${abs_cert_dir}"
    cert_file="${cert_dir}/evaluation.cert.json"
    log_file="${abs_log_file}"

    # Get model size for config and profile decision
    local model_size
    model_size=$(_estimate_model_size "${baseline_path}")
    if [[ -z "${model_size}" || "${model_size}" == "7" ]] && [[ -n "${model_id}" ]]; then
        model_size=$(_get_model_size_from_name "${model_id}")
    fi

    # Get model-aware config for window counts (needed for CI override)
    local config seq_len stride preview_n final_n eval_batch
    config=$(_get_invarlock_config "${model_size}")
    IFS=':' read -r seq_len stride preview_n final_n eval_batch <<< "${config}"
    local params_json="${TASK_PARAMS:-}"
    local applied_override=0
    if [[ -n "${params_json}" && "${params_json}" != "null" ]]; then
        local override_seq_len override_stride
        override_seq_len=$(echo "${params_json}" | jq -r '.seq_len // empty' 2>/dev/null)
        override_stride=$(echo "${params_json}" | jq -r '.stride // empty' 2>/dev/null)
        if [[ "${override_seq_len}" =~ ^[0-9]+$ ]]; then
            seq_len="${override_seq_len}"
            applied_override=1
        fi
        if [[ "${override_stride}" =~ ^[0-9]+$ ]]; then
            stride="${override_stride}"
            applied_override=1
        fi
        if [[ "${stride}" -gt "${seq_len}" ]]; then
            stride=$((seq_len / 2))
            [[ ${stride} -lt 1 ]] && stride=1
            applied_override=1
        fi
        if [[ ${applied_override} -eq 1 ]]; then
            echo "  OOM override: seq=${seq_len}, stride=${stride}" >> "${log_file}"
        fi
    fi
    if [[ "${stride}" -ne "${seq_len}" ]]; then
        stride="${seq_len}"
        applied_override=1
        echo "  Pairing override: seq=${seq_len}, stride=${stride}" >> "${log_file}"
    fi

    # For large models, use INVARLOCK_SKIP_OVERHEAD_CHECK to avoid loading
    # both baseline and edited models simultaneously (which would exceed 180GB).
    local profile_flag="ci"
    local min_windows
    min_windows="$(_default_ci_min_windows "${seq_len}")"
    if [[ "${profile_flag}" == "ci" && "${min_windows}" =~ ^[0-9]+$ && "${min_windows}" -gt 0 ]]; then
        if [[ "${preview_n}" -lt "${min_windows}" || "${final_n}" -lt "${min_windows}" ]]; then
            preview_n="${min_windows}"
            final_n="${min_windows}"
            applied_override=1
            echo "  CI window override: preview=${preview_n}, final=${final_n}" >> "${log_file}"
        fi
    fi
    local tier="${INVARLOCK_TIER:-balanced}"
    local bootstrap_replicates
    bootstrap_replicates="$(_resolve_bootstrap_replicates "${model_size}" "${tier}")"
    local baseline_report_root="${model_output_dir}/baseline_reports/${profile_flag}_${tier}_seq${seq_len}_pv${preview_n}_fn${final_n}"
    local baseline_report_file=""
    baseline_report_file=$(
        _ensure_certify_baseline_report \
            "${baseline_report_root}" \
            "${abs_baseline_path}" \
            "${profile_flag}" \
            "${tier}" \
            "${seq_len}" \
            "${stride}" \
            "${preview_n}" \
            "${final_n}" \
            "${eval_batch}" \
            "${bootstrap_replicates}" \
            "${model_size}" \
            "${log_file}" \
            || true
    )
    local -a baseline_report_args=()
    if [[ -n "${baseline_report_file}" && -f "${baseline_report_file}" ]]; then
        baseline_report_args=(--baseline-report "${baseline_report_file}")
        echo "  Reusing baseline report: ${baseline_report_file}" >> "${log_file}"
    else
        echo "  WARNING: Baseline report unavailable; will run per-cert baseline evaluation" >> "${log_file}"
    fi

    local -a extra_env=()
    if _is_large_model "${model_size}"; then
        extra_env+=(INVARLOCK_SKIP_OVERHEAD_CHECK=1)
        echo "  Large model (${model_size}): SKIP_OVERHEAD_CHECK=1" >> "${log_file}"
    fi
    extra_env+=(INVARLOCK_STORE_EVAL_WINDOWS=1)

    local config_root_base
    config_root_base="$(cd "${cert_dir}" && pwd)"
    local config_root="${config_root_base}/config_root"
    mkdir -p "${config_root}/runtime/profiles"
    cat > "${config_root}/runtime/profiles/ci.yaml" << YAML
model:
  device_map: "auto"
  dtype: "bfloat16"
  trust_remote_code: true
  low_cpu_mem_usage: true
dataset:
  seq_len: ${seq_len}
  stride: ${stride}
  preview_n: ${preview_n}
  final_n: ${final_n}
eval:
  bootstrap:
    replicates: ${bootstrap_replicates}
    alpha: 0.05
YAML

    extra_env+=("INVARLOCK_CONFIG_ROOT=${config_root}")

    # Find calibrated preset (must have seq_len/stride embedded)
    local preset_file=""
    for ext in yaml json; do
        local f="${preset_dir}/calibrated_preset_${model_name}.${ext}"
        if [[ -f "${f}" ]]; then
            preset_file="${f}"
            break
        fi
    done

    # If no preset found, we need to create one with model-specific params
    if [[ -z "${preset_file}" || ! -f "${preset_file}" ]]; then
        echo "  WARNING: No preset found for ${model_name}, creating minimal preset" >> "${log_file}"

        # Config already parsed above (seq_len, stride, preview_n, final_n, eval_batch)
        # Create minimal preset with seq_len/stride
        mkdir -p "${preset_dir}"
        preset_file="${preset_dir}/calibrated_preset_${model_name}.yaml"
        cat > "${preset_file}" << PRESET_YAML
dataset:
  provider: wikitext2
  split: validation
  seq_len: ${seq_len}
  stride: ${stride}
  preview_n: ${preview_n}
  final_n: ${final_n}
  seed: 42
PRESET_YAML
        echo "  Created preset: ${preset_file}" >> "${log_file}"
    fi

    # Run certify in isolated working directory to avoid temp file race conditions
    # (invarlock creates .certify_tmp/ in current directory which conflicts in parallel runs)
    local work_dir="${cert_dir}/.workdir"
    mkdir -p "${work_dir}"
    local abs_preset_file
    abs_preset_file="$(cd "$(dirname "${preset_file}")" && pwd)/$(basename "${preset_file}")"

    # CUDA_VISIBLE_DEVICES is inherited from execute_task() for multi-GPU support
    local exit_code=0
    (
        cd "${work_dir}" || exit 1
        env "${extra_env[@]}" invarlock certify \
            --source "${abs_baseline_path}" \
            "${baseline_report_args[@]}" \
            --edited "${abs_error_path}" \
            --profile "${profile_flag}" \
            --tier "${tier}" \
            --out "${cert_dir}" \
            --cert-out "${cert_dir}" \
            --preset "${abs_preset_file}" >> "${log_file}" 2>&1
    ) || exit_code=$?

    # Find and copy certificate (only the canonical cert)
    if [[ ! -f "${cert_file}" ]]; then
        local found_cert
        found_cert=$(find "${cert_dir}" -name "evaluation.cert.json" -type f 2>/dev/null | head -1)
        if [[ -n "${found_cert}" && -f "${found_cert}" && "${found_cert}" != "${cert_file}" ]]; then
            cp "${found_cert}" "${cert_file}" 2>/dev/null || true
        fi
    fi

    return ${exit_code}
}
