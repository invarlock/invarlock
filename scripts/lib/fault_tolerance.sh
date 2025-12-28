#!/usr/bin/env bash
# fault_tolerance.sh - Retry logic and error recovery
# Version: v2.0.1 (InvarLock B200 Validation Suite)
# Dependencies: queue_manager.sh, scheduler.sh, jq
# Usage: sourced optionally by gpu_worker.sh for retry/backoff handling
#
# Provides functions to:
# - Detect different failure types (OOM, transient, permanent)
# - Implement retry with exponential backoff
# - Handle OOM-specific recovery (batch size reduction)
# - Track error statistics

# Source dependencies
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -z "${QUEUE_MANAGER_LOADED:-}" ]] && source "${SCRIPT_DIR}/queue_manager.sh" && export QUEUE_MANAGER_LOADED=1
[[ -z "${SCHEDULER_LOADED:-}" ]] && source "${SCRIPT_DIR}/scheduler.sh" && export SCHEDULER_LOADED=1

export FAULT_TOLERANCE_LOADED=1

# ============ CONFIGURATION ============

# Maximum retry attempts per task
MAX_RETRIES="${MAX_RETRIES:-3}"

# Base backoff time in seconds
RETRY_BACKOFF_BASE="${RETRY_BACKOFF_BASE:-30}"

# Maximum backoff time
RETRY_BACKOFF_MAX="${RETRY_BACKOFF_MAX:-300}"

# ============ ERROR DETECTION ============

# Detect OOM error in log file
# Usage: detect_oom <log_file>
detect_oom() {
    local log_file="$1"

    if [[ ! -f "${log_file}" ]]; then
        return 1
    fi

    if grep -q -E "CUDA out of memory|torch\.cuda\.OutOfMemoryError|RuntimeError: CUDA error: out of memory|OOM|Killed" "${log_file}" 2>/dev/null; then
        return 0  # OOM detected
    fi

    return 1
}

# Detect transient/recoverable errors
# Usage: detect_transient_error <log_file>
detect_transient_error() {
    local log_file="$1"

    if [[ ! -f "${log_file}" ]]; then
        return 1
    fi

    # Network errors, temporary file issues, etc.
    local transient_patterns=(
        "ConnectionError"
        "TimeoutError"
        "Connection reset"
        "Connection refused"
        "Temporary failure"
        "Network is unreachable"
        "Resource temporarily unavailable"
        "Permission denied.*tmp"
        "No space left on device"  # Temporary, might resolve
        "rate limit"
        "503"  # Service unavailable
        "502"  # Bad gateway
    )

    for pattern in "${transient_patterns[@]}"; do
        if grep -q -i "${pattern}" "${log_file}" 2>/dev/null; then
            return 0  # Transient error detected
        fi
    done

    return 1
}

# Detect permanent/fatal errors
# Usage: detect_permanent_error <log_file>
detect_permanent_error() {
    local log_file="$1"

    if [[ ! -f "${log_file}" ]]; then
        return 1
    fi

    local permanent_patterns=(
        "Model.*not found"
        "ValueError: .* is not a valid"
        "FileNotFoundError"
        "ModuleNotFoundError"
        "ImportError"
        "SyntaxError"
        "TypeError"
        "AssertionError"
        "Invalid configuration"
        "Architecture not supported"
        "device-side assert"
        "vectorized_gather_kernel"
        "CUDA error: device-side assert triggered"
    )

    for pattern in "${permanent_patterns[@]}"; do
        if grep -q -E "${pattern}" "${log_file}" 2>/dev/null; then
            return 0  # Permanent error detected
        fi
    done

    return 1
}

# Classify error type
# Usage: classify_error <log_file>
# Returns: "oom", "transient", "permanent", or "unknown"
classify_error() {
    local log_file="$1"

    if detect_oom "${log_file}"; then
        echo "oom"
    elif detect_permanent_error "${log_file}"; then
        echo "permanent"
    elif detect_transient_error "${log_file}"; then
        echo "transient"
    else
        echo "unknown"
    fi
}

# ============ RETRY LOGIC ============

# Calculate backoff time with exponential increase
# Usage: calculate_backoff <retry_count>
calculate_backoff() {
    local retry_count="$1"

    # Exponential backoff: base * 2^retry
    local backoff=$((RETRY_BACKOFF_BASE * (2 ** retry_count)))

    # Add jitter (±20%)
    local jitter=$((backoff * (RANDOM % 40 - 20) / 100))
    backoff=$((backoff + jitter))

    # Cap at maximum
    if [[ ${backoff} -gt ${RETRY_BACKOFF_MAX} ]]; then
        backoff=${RETRY_BACKOFF_MAX}
    fi

    echo "${backoff}"
}

# Decide if task should be retried
# Usage: should_retry_task <task_file> <error_type>
# Returns: 0 if should retry, 1 if not
should_retry_task() {
    local task_file="$1"
    local error_type="$2"

    # Get current retry count
    local retries=$(get_task_field "${task_file}" "retries")
    local max_retries=$(get_task_field "${task_file}" "max_retries")

    [[ -z "${retries}" ]] && retries=0
    [[ -z "${max_retries}" ]] && max_retries=${MAX_RETRIES}

    # Don't retry permanent errors
    if [[ "${error_type}" == "permanent" ]]; then
        echo "Permanent error detected, not retrying" >&2
        return 1
    fi

    # OOM gets fewer retries (likely to fail again)
    if [[ "${error_type}" == "oom" ]]; then
        local oom_max=$((max_retries / 2 + 1))
        if [[ ${retries} -ge ${oom_max} ]]; then
            echo "OOM max retries (${oom_max}) exceeded" >&2
            return 1
        fi
    fi

    # Check general retry limit
    if [[ ${retries} -ge ${max_retries} ]]; then
        echo "Max retries (${max_retries}) exceeded" >&2
        return 1
    fi

    return 0
}

# Maybe retry a failed task (non-blocking version)
# Usage: maybe_retry_task <task_id>
# Note: This schedules a retry by moving the task back to pending queue.
#       The backoff is enforced via a "retry_after" timestamp in the task params.
maybe_retry_task() {
    local task_id="$1"

    # Find task file
    local task_file=$(find_task "${task_id}")

    if [[ -z "${task_file}" || ! -f "${task_file}" ]]; then
        echo "Task ${task_id} not found for retry" >&2
        return 1
    fi

    # Get task log
    local task_log="${QUEUE_DIR}/../logs/tasks/${task_id}.log"

    # Classify error
    local error_type=$(classify_error "${task_log}")

    # Check if should retry
    if ! should_retry_task "${task_file}" "${error_type}"; then
        echo "Task ${task_id} will not be retried (${error_type})"
        return 1
    fi

    # Calculate backoff
    local retries=$(get_task_field "${task_file}" "retries")
    local backoff=$(calculate_backoff "${retries}")

    # Calculate retry timestamp (non-blocking: set a "not before" time)
    local retry_after=$(date -u -d "+${backoff} seconds" +"%Y-%m-%dT%H:%M:%SZ")

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scheduling retry for ${task_id} at ${retry_after} (attempt $((retries + 1)), error: ${error_type})"

    # Update task with retry_after timestamp
    update_task_params "${task_file}" '{"retry_after": "'"${retry_after}"'", "last_error_type": "'"${error_type}"'"}'

    # Retry the task immediately (scheduler enforces retry_after backoff).
    retry_task "${task_id}"

    return $?
}

# Check if task retry delay has elapsed
# Usage: is_retry_ready <task_file>
# Returns: 0 if ready, 1 if still in backoff period
is_retry_ready() {
    local task_file="$1"

    local retry_after=$(get_task_param "${task_file}" "retry_after")

    if [[ -z "${retry_after}" || "${retry_after}" == "null" ]]; then
        return 0  # No retry delay, ready immediately
    fi

    # Compare retry_after with current time
    local retry_epoch
    local now_epoch=$(date +%s)
    retry_epoch=$(date -d "${retry_after}" "+%s" 2>/dev/null || echo "0")

    if [[ ${now_epoch} -ge ${retry_epoch} ]]; then
        return 0  # Past retry time, ready
    fi

    return 1  # Still in backoff
}

# ============ OOM RECOVERY ============

# Handle OOM by reducing memory usage
# Usage: handle_oom_task <task_file> <gpu_id> <log_file>
handle_oom_task() {
    local task_file="$1"
    local gpu_id="$2"
    local log_file="$3"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Handling OOM for task $(get_task_id "${task_file}")"

    # 1. Clear GPU memory
    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    print(f'Cleared GPU {torch.cuda.current_device()} memory')
" 2>/dev/null || true

    # 2. Get current params
    local params=$(get_task_params "${task_file}")
    local task_type=$(get_task_type "${task_file}")

    # 3. Reduce batch size or other memory params
    local current_batch=$(echo "${params}" | jq -r '.batch_size // 32')
    local new_batch=$((current_batch / 2))

    if [[ ${new_batch} -lt 1 ]]; then
        new_batch=1
    fi

    # 4. Reduce sequence length if applicable
    local current_seq=$(echo "${params}" | jq -r '.seq_len // 512')
    local new_seq=$((current_seq / 2))

    if [[ ${new_seq} -lt 128 ]]; then
        new_seq=128
    fi

    # 5. Update params
    local new_params=$(echo "${params}" | jq --argjson bs "${new_batch}" --argjson seq "${new_seq}" \
        '. + {batch_size: $bs, seq_len: $seq, oom_recovery: true}')

    update_task_params "${task_file}" "${new_params}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] OOM recovery: reduced batch_size ${current_batch} → ${new_batch}, seq_len ${current_seq} → ${new_seq}"

    return 0
}

# ============ ERROR STATISTICS ============

# Record error for statistics
# Usage: record_error <task_id> <error_type> <error_msg> <output_dir>
record_error() {
    local task_id="$1"
    local error_type="$2"
    local error_msg="$3"
    local output_dir="$4"

    local error_log="${output_dir}/state/errors.json"

    # Create error entry
    local error_entry=$(jq -n \
        --arg task_id "${task_id}" \
        --arg error_type "${error_type}" \
        --arg error_msg "${error_msg}" \
        --arg timestamp "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
        '{task_id: $task_id, error_type: $error_type, error_msg: $error_msg, timestamp: $timestamp}'
    )

    # Append to log
    if [[ -f "${error_log}" ]]; then
        jq --argjson entry "${error_entry}" '. + [$entry]' "${error_log}" > "${error_log}.tmp"
        mv "${error_log}.tmp" "${error_log}"
    else
        echo "[${error_entry}]" > "${error_log}"
    fi
}

# Get error statistics
# Usage: get_error_stats <output_dir>
get_error_stats() {
    local output_dir="$1"
    local error_log="${output_dir}/state/errors.json"

    if [[ ! -f "${error_log}" ]]; then
        echo '{"total": 0, "by_type": {}}'
        return
    fi

    jq '{
        total: length,
        by_type: (group_by(.error_type) | map({key: .[0].error_type, value: length}) | from_entries),
        recent: (sort_by(.timestamp) | reverse | .[0:5])
    }' "${error_log}"
}

# Print error summary
# Usage: print_error_summary <output_dir>
print_error_summary() {
    local output_dir="$1"

    echo "=== ERROR SUMMARY ==="

    local stats=$(get_error_stats "${output_dir}")
    local total=$(echo "${stats}" | jq -r '.total')

    echo "Total Errors: ${total}"
    echo ""

    echo "By Type:"
    echo "${stats}" | jq -r '.by_type | to_entries[] | "  \(.key): \(.value)"'
    echo ""

    echo "Recent Errors:"
    echo "${stats}" | jq -r '.recent[]? | "  [\(.timestamp)] \(.task_id): \(.error_type)"'
}

# ============ HEALTH CHECK ============

# Check if system is healthy for task execution
# Usage: health_check <gpu_id>
# Returns: 0 if healthy, 1 if not
health_check() {
    local gpu_id="$1"

    # Check GPU is accessible
    if ! nvidia-smi -i "${gpu_id}" &>/dev/null; then
        echo "GPU ${gpu_id} not accessible"
        return 1
    fi

    # Check GPU has enough free memory (at least 1GB)
    local free_mem=$(get_gpu_available_memory "${gpu_id}" 2>/dev/null)
    if [[ -z "${free_mem}" || "${free_mem}" -lt 1 ]]; then
        echo "GPU ${gpu_id} has insufficient memory (${free_mem:-0} GB free)"
        return 1
    fi

    # Check disk space (at least 10GB)
    local free_disk
    free_disk=$(df -BG . 2>/dev/null | awk 'NR==2 {gsub(/G/,""); print $4}')
    if [[ -z "${free_disk}" || ${free_disk} -lt 10 ]]; then
        echo "Low disk space (${free_disk:-unknown} GB free)"
        return 1
    fi

    # Check Python is available
    if ! python3 -c "import torch" &>/dev/null; then
        echo "PyTorch not available"
        return 1
    fi

    return 0
}

# ============ CLEANUP ============

# Clean up failed task artifacts
# Usage: cleanup_failed_task <task_id> <output_dir>
cleanup_failed_task() {
    local task_id="$1"
    local output_dir="$2"

    # Don't delete logs - keep for debugging
    # Only clean up partial/corrupted model artifacts

    local task_file=$(find_task "${task_id}")
    if [[ -z "${task_file}" ]]; then
        return 0
    fi

    local task_type=$(get_task_type "${task_file}")
    local model_name=$(get_task_field "${task_file}" "model_name")
    local params=$(get_task_params "${task_file}")

    case "${task_type}" in
        CREATE_EDIT)
            # Clean up partial edit model
            local edit_spec=$(echo "${params}" | jq -r '.edit_spec // ""')
            local version=$(echo "${params}" | jq -r '.version // "clean"')

            # Find and remove incomplete model directory
            local model_dir="${output_dir}/${model_name}/models"
            for dir in "${model_dir}"/*_${version}; do
                if [[ -d "${dir}" && ! -f "${dir}/config.json" ]]; then
                    echo "Removing incomplete model: ${dir}"
                    rm -rf "${dir}"
                fi
            done
            ;;
        CREATE_ERROR)
            local error_type=$(echo "${params}" | jq -r '.error_type // ""')
            local error_dir="${output_dir}/${model_name}/models/error_${error_type}"

            if [[ -d "${error_dir}" && ! -f "${error_dir}/config.json" ]]; then
                echo "Removing incomplete error model: ${error_dir}"
                rm -rf "${error_dir}"
            fi
            ;;
    esac
}

# Clean up all failed task artifacts
# Usage: cleanup_all_failed <output_dir>
cleanup_all_failed() {
    local output_dir="$1"

    echo "Cleaning up failed task artifacts..."

    for task_file in "${QUEUE_DIR}/failed"/*.task; do
        [[ -f "${task_file}" ]] || continue

        local task_id=$(get_task_id "${task_file}")
        cleanup_failed_task "${task_id}" "${output_dir}"
    done

    echo "Cleanup complete"
}
