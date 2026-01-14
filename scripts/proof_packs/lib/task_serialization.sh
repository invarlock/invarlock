#!/usr/bin/env bash
# task_serialization.sh - Task JSON handling for dynamic GPU scheduling
# Version: proof-packs-v1 (InvarLock Proof Pack Suite)
# Dependencies: jq
# Usage: sourced by queue_manager.sh/gpu_worker.sh to read/write queue task files
#
# Provides functions to:
# - Create task records as JSON files
# - Parse task fields from JSON
# - Validate task structure
# - Calculate required GPU count for multi-GPU model distribution

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime.sh
source "${SCRIPT_DIR}/runtime.sh"

# Ensure jq is available (required for JSON handling). Keep this safe-to-source:
# fail only when a jq-dependent function is invoked.
TASK_SERIALIZATION_HAS_JQ="1"
if ! command -v jq &>/dev/null; then
    TASK_SERIALIZATION_HAS_JQ="0"
fi

_task_serialization_require_jq() {
    if [[ "${TASK_SERIALIZATION_HAS_JQ}" != "1" ]]; then
        echo "ERROR: jq is required for task serialization. Install with: sudo apt-get install jq" >&2
        return 1
    fi
}

# ============ TASK FIELD DEFINITIONS ============
# Task record format (stored as JSON):
#   task_id:         Unique identifier (e.g., "model0_SETUP_BASELINE_001_abcd")
#   task_type:       One of: SETUP_BASELINE, EVAL_BASELINE, CALIBRATION_RUN,
#                    CREATE_EDIT, EVAL_EDIT, CERTIFY_EDIT, CREATE_ERROR,
#                    CERTIFY_ERROR, GENERATE_PRESET
#   model_id:        Full HuggingFace model ID (e.g., "mistralai/Mistral-7B-v0.1")
#   model_name:      Sanitized name for paths (e.g., "mistral-7b-v0.1")
#   model_size_gb:   Estimated GPU memory requirement in GB
#   required_gpus:   Number of GPUs needed based on per-device memory planning
#   assigned_gpus:   Comma-separated list of assigned GPU IDs (set at runtime)
#   status:          One of: pending, ready, running, completed, failed
#   gpu_id:          Assigned GPU (-1 if unassigned) - deprecated, use assigned_gpus
#   dependencies:    Array of task_ids that must complete first
#   params:          Task-specific parameters object
#   priority:        Scheduling priority (0-100, higher = more urgent)
#   retries:         Number of retry attempts made
#   max_retries:     Maximum allowed retries
#   created_at:      ISO 8601 timestamp
#   started_at:      ISO 8601 timestamp (null if not started)
#   completed_at:    ISO 8601 timestamp (null if not completed)
#   error_msg:       Error message if failed (null otherwise)

# ============ MULTI-GPU CALCULATION ============

# GPU memory per device (defaults to GPU_MEMORY_GB when available)
GPU_MEMORY_PER_DEVICE="${GPU_MEMORY_PER_DEVICE:-${GPU_MEMORY_GB:-80}}"

# Calculate required GPUs based on model memory size.
# Usage: calculate_required_gpus <model_size_gb>
# Returns: integer GPU count (min 1).
#
# Proof pack optimization: dynamic sizing based on per-GPU memory.
# - Prefer single GPU when model fits within GPU_MEMORY_PER_DEVICE.
# - Scale to multiple GPUs when required memory exceeds per-GPU capacity.
#
calculate_required_gpus() {
    local model_size_gb="$1"
    local per_device="${GPU_MEMORY_PER_DEVICE:-80}"
    local max_gpus="${NUM_GPUS:-8}"

    if ! [[ "${model_size_gb}" =~ ^[0-9]+$ ]]; then
        echo "1"
        return
    fi
    if ! [[ "${per_device}" =~ ^[0-9]+$ ]] || [[ "${per_device}" -le 0 ]]; then
        per_device=80
    fi
    if ! [[ "${max_gpus}" =~ ^[0-9]+$ ]]; then
        max_gpus=8
    fi
    [[ ${max_gpus} -lt 1 ]] && max_gpus=1
    if [[ ${model_size_gb} -le ${per_device} ]]; then
        echo "1"
        return
    fi

    local required=$(( (model_size_gb + per_device - 1) / per_device ))
    if [[ ${required} -gt ${max_gpus} ]]; then
        required=${max_gpus}
    fi
    echo "${required}"
}

# ============ TASK CREATION ============

# Create a new task file in the queue
# Usage: create_task <queue_dir> <task_type> <model_id> <model_name> <model_size_gb> <dependencies> <params_json> [priority]
# Returns: task_id on stdout, creates task file in queue/pending/
create_task() {
    _task_serialization_require_jq || return 1

    local queue_dir="$1"
    local task_type="$2"
    local model_id="$3"
    local model_name="$4"
    local model_size_gb="$5"
    local dependencies="$6"  # Comma-separated or JSON array
    local params_json="$7"
    local priority="${8:-50}"

    if ! [[ "${model_size_gb}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: model_size_gb must be an integer GB value, got: ${model_size_gb}" >&2
        return 1
    fi
    if ! [[ "${priority}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: priority must be an integer, got: ${priority}" >&2
        return 1
    fi

    # Generate unique task_id
    local sequence="${TASK_SEQUENCE:-1}"
    if ! [[ "${sequence}" =~ ^[0-9]+$ ]]; then
        sequence="1"
    fi
    local rand_suffix
    rand_suffix=$(printf '%04x' "$((RANDOM % 65536))")
    local task_id="${model_name}_${task_type}_$(printf '%03d' "${sequence}")_${rand_suffix}"

    # Calculate required GPUs based on model size
    local required_gpus=$(calculate_required_gpus "${model_size_gb}")

    # Convert dependencies to JSON array if comma-separated
    local deps_array
    if [[ "${dependencies}" == "["* ]]; then
        if ! printf '%s' "${dependencies}" | jq -e 'type == "array"' &>/dev/null; then
            echo "ERROR: Invalid dependencies JSON array: ${dependencies}" >&2
            return 1
        fi
        deps_array="${dependencies}"
    elif [[ -z "${dependencies}" || "${dependencies}" == "none" ]]; then
        deps_array="[]"
    else
        # Convert "dep1,dep2,dep3" to ["dep1","dep2","dep3"]
        deps_array="$(printf '%s' "${dependencies}" | tr ',' '\n' | jq -R . | jq -s .)" || {
            echo "ERROR: Failed to parse dependencies: ${dependencies}" >&2
            return 1
        }
    fi

    # Ensure params is valid JSON
    local params
    if [[ -z "${params_json}" || "${params_json}" == "null" ]]; then
        params="{}"
    else
        params="${params_json}"
    fi

    # Validate params is valid JSON object
    if ! printf '%s' "${params}" | jq -e 'type == "object"' &>/dev/null; then
        echo "ERROR: Invalid params JSON (expected object): ${params}" >&2
        return 1
    fi

    local created_at
    created_at=$(_now_iso)

    # Create task JSON with multi-GPU fields
    # Note: started_at, completed_at, error_msg use literal null in jq, not --arg
    local task_json
    task_json="$(jq -n \
        --arg task_id "${task_id}" \
        --arg task_type "${task_type}" \
        --arg model_id "${model_id}" \
        --arg model_name "${model_name}" \
        --argjson model_size_gb "${model_size_gb}" \
        --argjson required_gpus "${required_gpus}" \
        --arg status "pending" \
        --argjson gpu_id "-1" \
        --argjson dependencies "${deps_array}" \
        --argjson params "${params}" \
        --argjson priority "${priority}" \
        --argjson retries "0" \
        --argjson max_retries "3" \
        --arg created_at "${created_at}" \
        '{
            task_id: $task_id,
            task_type: $task_type,
            model_id: $model_id,
            model_name: $model_name,
            model_size_gb: $model_size_gb,
            required_gpus: $required_gpus,
            assigned_gpus: null,
            status: $status,
            gpu_id: $gpu_id,
            dependencies: $dependencies,
            params: $params,
            priority: $priority,
            retries: $retries,
            max_retries: $max_retries,
            created_at: $created_at,
            started_at: null,
            completed_at: null,
            error_msg: null
        }'
    )" || {
        echo "ERROR: Failed to build task JSON for ${task_id}" >&2
        return 1
    }

    # Write to pending queue
    local pending_dir="${queue_dir}/pending"
    if ! mkdir -p "${pending_dir}"; then
        echo "ERROR: Failed to create pending queue dir: ${pending_dir}" >&2
        return 1
    fi

    local task_file="${pending_dir}/${task_id}.task"
    local tmp_file="${task_file}.tmp.${BASHPID:-$$}"
    if ! printf '%s\n' "${task_json}" > "${tmp_file}"; then
        rm -f "${tmp_file}"
        echo "ERROR: Failed to write task file: ${task_file}" >&2
        return 1
    fi
    if ! mv "${tmp_file}" "${task_file}"; then
        rm -f "${tmp_file}"
        echo "ERROR: Failed to finalize task file: ${task_file}" >&2
        return 1
    fi

    echo "${task_id}"
}

# ============ TASK FIELD ACCESS ============

# Get a field from a task file
# Usage: get_task_field <task_file> <field_name>
get_task_field() {
    _task_serialization_require_jq || return 1

    local task_file="$1"
    local field="$2"

    if [[ ! -f "${task_file}" ]]; then
        echo "ERROR: Task file not found: ${task_file}" >&2
        return 1
    fi

    jq -r --arg field "${field}" '.[$field] // empty' "${task_file}"
}

# Get multiple fields as tab-separated values
# Usage: get_task_fields <task_file> <field1> <field2> ...
get_task_fields() {
    _task_serialization_require_jq || return 1

    local task_file="$1"
    shift
    local fields=("$@")

    if [[ ! -f "${task_file}" ]]; then
        echo "ERROR: Task file not found: ${task_file}" >&2
        return 1
    fi

    local field
    local jq_expr=""
    for field in "${fields[@]}"; do
        local esc="${field//\\/\\\\}"
        esc="${esc//\"/\\\"}"
        jq_expr+=".[\"${esc}\"], "
    done
    jq_expr="${jq_expr%, }"

    jq -r "[${jq_expr}] | @tsv" "${task_file}"
}

# Get task id from a task file
# Usage: get_task_id <task_file>
get_task_id() {
    get_task_field "$1" "task_id"
}

# Get task type from a task file
# Usage: get_task_type <task_file>
get_task_type() {
    get_task_field "$1" "task_type"
}

# Get model size from a task file
# Usage: get_task_model_size <task_file>
get_task_model_size() {
    get_task_field "$1" "model_size_gb"
}

# Get required GPUs from a task file
# Usage: get_task_required_gpus <task_file>
get_task_required_gpus() {
    local val
    val=$(get_task_field "$1" "required_gpus")
    # Default to 1 if missing or non-numeric (backward compatibility).
    if ! [[ "${val}" =~ ^[0-9]+$ ]]; then
        val="1"
    fi
    echo "${val}"
}

# Get assigned GPUs from a task file (comma-separated list)
# Usage: get_task_assigned_gpus <task_file>
get_task_assigned_gpus() {
    get_task_field "$1" "assigned_gpus"
}

# Get task dependencies as newline-separated list
# Usage: get_task_dependencies <task_file>
get_task_dependencies() {
    _task_serialization_require_jq || return 1

    local task_file="$1"
    jq -r '.dependencies[]?' "${task_file}" 2>/dev/null
}

# Get task params as JSON object
# Usage: get_task_params <task_file>
get_task_params() {
    _task_serialization_require_jq || return 1

    local task_file="$1"
    jq -r '.params // {}' "${task_file}"
}

# Get a specific param value
# Usage: get_task_param <task_file> <param_name>
get_task_param() {
    _task_serialization_require_jq || return 1

    local task_file="$1"
    local param="$2"
    jq -r --arg param "${param}" '(.params // {})[$param] // empty' "${task_file}"
}

# ============ TASK FIELD UPDATE ============

# Update a field in a task file (atomic operation)
# Usage: update_task_field <task_file> <field_name> <value> [is_json]
update_task_field() {
    _task_serialization_require_jq || return 1

    local task_file="$1"
    local field="$2"
    local value="$3"
    local is_json="${4:-false}"

    if [[ ! -f "${task_file}" ]]; then
        echo "ERROR: Task file not found: ${task_file}" >&2
        return 1
    fi

    local tmp_file="${task_file}.tmp.${BASHPID:-$$}"

    local jq_rc=0
    if [[ "${is_json}" == "true" ]]; then
        jq --arg field "${field}" --argjson val "${value}" '.[$field] = $val' "${task_file}" > "${tmp_file}" || jq_rc=$?
    else
        jq --arg field "${field}" --arg val "${value}" '.[$field] = $val' "${task_file}" > "${tmp_file}" || jq_rc=$?
    fi

    if [[ ${jq_rc} -eq 0 ]]; then
        if mv "${tmp_file}" "${task_file}"; then
            return 0
        fi
    fi
    rm -f "${tmp_file}"
    return 1
}

# Update task status
# Usage: update_task_status <task_file> <new_status>
update_task_status() {
    update_task_field "$1" "status" "$2"
}

# Assign GPU to task
# Usage: assign_task_gpu <task_file> <gpu_id>
assign_task_gpu() {
    update_task_field "$1" "gpu_id" "$2" "true"
}

# Mark task as started
# Usage: mark_task_started <task_file> <gpu_id>
mark_task_started() {
    _task_serialization_require_jq || return 1

    local task_file="$1"
    local gpu_id="$2"
    local now
    now=$(_now_iso)

    local tmp_file="${task_file}.tmp.${BASHPID:-$$}"
    local jq_rc=0
    jq --argjson gpu "${gpu_id}" --arg time "${now}" \
        '.status = "running" | .gpu_id = $gpu | .started_at = $time | .completed_at = null | .error_msg = null' \
        "${task_file}" > "${tmp_file}" || jq_rc=$?

    if [[ ${jq_rc} -eq 0 ]]; then
        if mv "${tmp_file}" "${task_file}"; then
            return 0
        fi
    fi
    rm -f "${tmp_file}"
    return 1
}

# Mark task as started with multiple GPUs
# Usage: mark_task_started_multi <task_file> <gpu_ids_csv>
# Example: mark_task_started_multi task.json "0,1,2,3"
mark_task_started_multi() {
    _task_serialization_require_jq || return 1

    local task_file="$1"
    local gpu_ids="$2"  # Comma-separated GPU IDs
    local now
    now=$(_now_iso)

    # Extract first GPU as primary gpu_id for backward compatibility
    local primary_gpu="${gpu_ids%%,*}"

    local tmp_file="${task_file}.tmp.${BASHPID:-$$}"
    local jq_rc=0
    jq --argjson gpu "${primary_gpu}" --arg gpus "${gpu_ids}" --arg time "${now}" \
        '.status = "running" | .gpu_id = $gpu | .assigned_gpus = $gpus | .started_at = $time | .completed_at = null | .error_msg = null' \
        "${task_file}" > "${tmp_file}" || jq_rc=$?

    if [[ ${jq_rc} -eq 0 ]]; then
        if mv "${tmp_file}" "${task_file}"; then
            return 0
        fi
    fi
    rm -f "${tmp_file}"
    return 1
}

# Mark task as completed
# Usage: mark_task_completed <task_file>
mark_task_completed() {
    _task_serialization_require_jq || return 1

    local task_file="$1"
    local now
    now=$(_now_iso)

    local tmp_file="${task_file}.tmp.${BASHPID:-$$}"
    local jq_rc=0
    jq --arg time "${now}" \
        '.status = "completed" | .completed_at = $time | .error_msg = null' \
        "${task_file}" > "${tmp_file}" || jq_rc=$?

    if [[ ${jq_rc} -eq 0 ]]; then
        if mv "${tmp_file}" "${task_file}"; then
            return 0
        fi
    fi
    rm -f "${tmp_file}"
    return 1
}

# Mark task as failed
# Usage: mark_task_failed <task_file> <error_message>
mark_task_failed() {
    _task_serialization_require_jq || return 1

    local task_file="$1"
    local error_msg="$2"
    local now
    now=$(_now_iso)

    local tmp_file="${task_file}.tmp.${BASHPID:-$$}"
    local jq_rc=0
    jq --arg time "${now}" --arg err "${error_msg}" \
        '.status = "failed" | .completed_at = $time | .error_msg = $err' \
        "${task_file}" > "${tmp_file}" || jq_rc=$?

    if [[ ${jq_rc} -eq 0 ]]; then
        if mv "${tmp_file}" "${task_file}"; then
            return 0
        fi
    fi
    rm -f "${tmp_file}"
    return 1
}

# Increment retry count
# Usage: increment_task_retries <task_file>
increment_task_retries() {
    _task_serialization_require_jq || return 1

    local task_file="$1"

    local tmp_file="${task_file}.tmp.${BASHPID:-$$}"
    local jq_rc=0
    jq '.retries = (((.retries // 0) | tonumber? // 0) + 1)' "${task_file}" > "${tmp_file}" || jq_rc=$?

    if [[ ${jq_rc} -eq 0 ]]; then
        if mv "${tmp_file}" "${task_file}"; then
            return 0
        fi
    fi
    rm -f "${tmp_file}"
    return 1
}

# Update task params (merge with existing)
# Usage: update_task_params <task_file> <params_json>
update_task_params() {
    _task_serialization_require_jq || return 1

    local task_file="$1"
    local new_params="$2"

    local tmp_file="${task_file}.tmp.${BASHPID:-$$}"
    local jq_rc=0
    jq --argjson new "${new_params}" '.params = ((if (.params | type) == "object" then .params else {} end) + $new)' \
        "${task_file}" > "${tmp_file}" || jq_rc=$?

    if [[ ${jq_rc} -eq 0 ]]; then
        if mv "${tmp_file}" "${task_file}"; then
            return 0
        fi
    fi
    rm -f "${tmp_file}"
    return 1
}

# ============ TASK VALIDATION ============

# Validate task file structure
# Usage: validate_task <task_file>
# Returns: 0 if valid, 1 if invalid (with error on stderr)
validate_task() {
    _task_serialization_require_jq || return 1

    local task_file="$1"

    if [[ ! -f "${task_file}" ]]; then
        echo "ERROR: Task file not found: ${task_file}" >&2
        return 1
    fi

    # Check if valid JSON
    if ! jq . "${task_file}" &>/dev/null; then
        echo "ERROR: Invalid JSON in task file: ${task_file}" >&2
        return 1
    fi

    # Check required fields
    local required_fields=("task_id" "task_type" "model_id" "model_name" "status")
    local field
    for field in "${required_fields[@]}"; do
        local value=$(jq -r ".${field} // empty" "${task_file}")
        if [[ -z "${value}" ]]; then
            echo "ERROR: Missing required field '${field}' in: ${task_file}" >&2
            return 1
        fi
    done

    # Validate task_type
    local task_type=$(get_task_type "${task_file}")
    # v2.1.0-b200: Added CREATE_EDITS_BATCH and split eval benchmarks (EVAL_MMLU, etc.)
    local valid_types="SETUP_BASELINE EVAL_BASELINE CALIBRATION_RUN CREATE_EDIT CREATE_EDITS_BATCH EVAL_EDIT EVAL_MMLU EVAL_HELLASWAG EVAL_ARC EVAL_WINOGRANDE CERTIFY_EDIT CREATE_ERROR CERTIFY_ERROR GENERATE_PRESET"
    if [[ ! " ${valid_types} " =~ " ${task_type} " ]]; then
        echo "ERROR: Invalid task_type '${task_type}' in: ${task_file}" >&2
        return 1
    fi

    # Validate status
    local status=$(get_task_field "${task_file}" "status")
    local valid_statuses="pending ready running completed failed"
    if [[ ! " ${valid_statuses} " =~ " ${status} " ]]; then
        echo "ERROR: Invalid status '${status}' in: ${task_file}" >&2
        return 1
    fi

    return 0
}

# ============ TASK SUMMARY ============

# Print task summary (for logging/debugging)
# Usage: print_task_summary <task_file>
print_task_summary() {
    _task_serialization_require_jq || return 1

    local task_file="$1"

    if [[ ! -f "${task_file}" ]]; then
        echo "Task file not found: ${task_file}"
        return 1
    fi

    jq -r '[.task_id, .task_type, .model_name, .status, (.model_size_gb | tostring) + "GB"] | join(" | ")' "${task_file}"
}

# Print all tasks in a directory
# Usage: print_queue_summary <queue_dir> <status>
print_queue_summary() {
    local queue_dir="$1"
    local status="$2"
    local dir="${queue_dir}/${status}"
    local task_file

    if [[ ! -d "${dir}" ]]; then
        echo "No ${status} tasks"
        return
    fi

    echo "=== $(printf '%s' "${status}" | tr '[:lower:]' '[:upper:]') TASKS ==="
    for task_file in "${dir}"/*.task; do
        [[ -f "${task_file}" ]] && print_task_summary "${task_file}"
    done
}

# ============ MODEL SIZE ESTIMATION ============

# Estimate GPU memory requirement for a model (in GB)
# Usage: estimate_model_memory <model_id_or_path> [task_type]
#
# CRITICAL: At task generation time, model_id is a HuggingFace ID like "Qwen/Qwen1.5-72B"
# The model hasn't been downloaded yet, so we CANNOT use estimate_model_params (needs config.json)
# We MUST parse the model name to estimate size.
#
# This function handles both:
# - HuggingFace model IDs (e.g., "Qwen/Qwen1.5-72B", "NousResearch/Llama-2-70b-hf")
# - Local paths (e.g., "/path/to/model") - can use config.json if available
estimate_model_memory() {
    local model_id="$1"
    local task_type="${2:-EVAL_BASELINE}"

    # Determine model size bucket
    # IMPORTANT: Initialize with empty string to avoid "unbound variable" error with set -u
    local size_bucket=""

    # Check if this is a local path with config.json (can use accurate estimation)
    # Local paths: start with / or ./ or contain spaces, OR are directories
    if [[ -d "${model_id}" && -f "${model_id}/config.json" ]]; then
        # Local path with config.json - use accurate estimation if function available
        if type estimate_model_params &>/dev/null; then
            size_bucket=$(estimate_model_params "${model_id}")
        fi
    fi

    # If size_bucket is empty or "7" (default fallback), use name-based estimation
    # This is critical for HuggingFace IDs where config.json doesn't exist yet
    if [[ -z "${size_bucket:-}" || "${size_bucket}" == "7" ]]; then
        local model_lower=$(printf '%s' "${model_id}" | tr '[:upper:]' '[:lower:]')

        # Check for 70B+ models (need ~140-154 GB)
        if [[ "${model_lower}" =~ 70b || "${model_lower}" =~ 72b || "${model_lower}" =~ 65b ]]; then
            size_bucket="70"
        # Check for 40B models (need ~80 GB)
        elif [[ "${model_lower}" =~ 40b ]]; then
            size_bucket="40"
        # Check for 30-34B models (need ~60-68 GB)
        elif [[ "${model_lower}" =~ 30b || "${model_lower}" =~ 32b || "${model_lower}" =~ 34b ]]; then
            size_bucket="30"
        # Check for 13-14B models (need ~26-28 GB)
        elif [[ "${model_lower}" =~ 13b || "${model_lower}" =~ 14b ]]; then
            size_bucket="13"
        # Check for MoE models like Mixtral (need ~90 GB)
        elif [[ "${model_lower}" =~ mixtral || "${model_lower}" =~ 8x7b ]]; then
            size_bucket="moe"
        # Default to 7B models
        else
            size_bucket="7"
        fi
    fi

    # Base memory in GB
    local base_memory
    case "${size_bucket}" in
        "70"|"72") base_memory=140 ;;
        "moe")     base_memory=90 ;;
        "40")      base_memory=80 ;;
        "30")      base_memory=64 ;;
        "13")      base_memory=26 ;;
        *)         base_memory=14 ;;
    esac

    local is_large="false"
    if [[ "${size_bucket}" =~ ^[0-9]+$ && ${size_bucket} -ge 70 ]]; then
        is_large="true"
    fi

    # Multiplier by task type
    local multiplier
    if [[ "${is_large}" == "true" ]]; then
        case "${task_type}" in
            "SETUP_BASELINE")  multiplier="1.0" ;;
            "EVAL_BASELINE")   multiplier="1.1" ;;
            "CALIBRATE_CLEAN") multiplier="1.2" ;;
            "CALIBRATION_RUN") multiplier="1.05" ;;
            "CREATE_EDIT")     multiplier="1.2" ;;
            "EVAL_EDIT")       multiplier="1.1" ;;
            "CERTIFY_EDIT")    multiplier="1.05" ;;
            "CREATE_ERROR")    multiplier="1.15" ;;
            "CERTIFY_ERROR")   multiplier="1.05" ;;
            "GENERATE_PRESET") multiplier="0.1" ;;  # CPU only
            *)                 multiplier="1.1" ;;
        esac
    else
        case "${task_type}" in
            "SETUP_BASELINE")  multiplier="1.0" ;;
            "EVAL_BASELINE")   multiplier="1.2" ;;
            "CALIBRATE_CLEAN") multiplier="1.4" ;;
            "CALIBRATION_RUN") multiplier="1.1" ;;
            "CREATE_EDIT")     multiplier="1.5" ;;
            "EVAL_EDIT")       multiplier="1.2" ;;
            "CERTIFY_EDIT")    multiplier="1.1" ;;
            "CREATE_ERROR")    multiplier="1.3" ;;
            "CERTIFY_ERROR")   multiplier="1.1" ;;
            "GENERATE_PRESET") multiplier="0.1" ;;  # CPU only
            *)                 multiplier="1.2" ;;
        esac
    fi

    # Calculate using awk for floating point (avoids bc dependency)
    local result
    result=$(awk -v base="${base_memory}" -v mult="${multiplier}" 'BEGIN { printf "%.0f", base * mult }')
    # Add safety margin (smaller for large models to avoid false "does not fit" skips)
    local safety_pct=10
    if [[ "${is_large}" == "true" ]]; then
        safety_pct=5
    fi
    result=$((result * (100 + safety_pct) / 100))
    echo "${result}"
}
