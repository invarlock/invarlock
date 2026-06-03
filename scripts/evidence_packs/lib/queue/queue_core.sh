#!/usr/bin/env bash
# queue_core.sh - Queue setup, locking, and summary helpers
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Dependencies: task_serialization.sh, jq
# Usage: sourced by gpu_worker.sh and scheduler to manage task lifecycle
#
# Provides functions to:
# - Initialize and manage queue directories
# - Atomically claim/complete/fail tasks
# - Resolve task dependencies
# - Track queue statistics

# Source task serialization if not already sourced
export QUEUE_CORE_LOADED=1
QUEUE_MANAGER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${QUEUE_MANAGER_SCRIPT_DIR}"
# shellcheck source=../core/runtime.sh
source "${SCRIPT_DIR}/../core/runtime.sh"
if [[ -z "${TASK_SERIALIZATION_LOADED:-}" ]]; then
    source "${SCRIPT_DIR}/../tasks/task_serialization.sh"
    export TASK_SERIALIZATION_LOADED=1
fi
SCRIPT_DIR="${QUEUE_MANAGER_SCRIPT_DIR}"

_pack_queue_pack_root() {
    local script_dir="${SCRIPT_DIR:-${QUEUE_MANAGER_SCRIPT_DIR}}"
    if [[ "$(basename "${script_dir}")" == "queue" ]]; then
        cd "${script_dir}/../.." && pwd
    else
        cd "${script_dir}/.." && pwd
    fi
}

# ============ QUEUE STRUCTURE ============
# ${QUEUE_DIR}/
# ├── pending/          # Tasks waiting for dependencies
# ├── ready/            # Tasks ready to run (dependencies met)
# ├── running/          # Currently executing tasks
# ├── completed/        # Successfully finished tasks
# ├── failed/           # Failed tasks (may retry)
# └── queue.lock        # Global lock for queue operations

# ============ QUEUE INITIALIZATION ============

# Initialize queue directory structure
# Usage: init_queue <output_dir>
init_queue() {
    local output_dir="$1"
    export QUEUE_DIR="${output_dir}/queue"

    mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed}
    mkdir -p "${output_dir}/workers"
    mkdir -p "${output_dir}/logs/tasks"
    mkdir -p "${output_dir}/state"

    # Create lock file
    touch "${QUEUE_DIR}/queue.lock"

    # Initialize state file
    cat > "${output_dir}/state/progress.json" << EOF
{
    "initialized_at": "$(_now_iso)",
    "total_tasks": 0,
    "completed_tasks": 0,
    "failed_tasks": 0,
    "status": "initializing"
}
EOF

    echo "${QUEUE_DIR}"
}

# ============ QUEUE LOCKING ============

# Acquire queue lock (blocking with timeout)
# Usage: acquire_queue_lock [timeout_seconds]
#
# Uses mkdir-based locking which is atomic and avoids file descriptor
# inheritance issues when workers are spawned as subshells.
acquire_queue_lock() {
    local timeout="${1:-30}"
    local lock_file="${QUEUE_DIR}/queue.lock"
    local lock_dir="${lock_file}.d"
    local my_pid="${BASHPID:-$$}"
    local now
    now=$(_now_epoch)
    local deadline=$((now + timeout))

    while true; do
        # Try to create lock directory (atomic operation)
        if mkdir "${lock_dir}" 2>/dev/null; then
            # Successfully acquired lock - record owner
            echo "${my_pid}" > "${lock_dir}/owner" 2>/dev/null || true
            export QUEUE_LOCK_DIR="${lock_dir}"
            return 0
        fi

        # Check if we've exceeded timeout
        now=$(_now_epoch)
        if [[ ${now} -ge ${deadline} ]]; then
            local owner_pid=""
            if [[ -f "${lock_dir}/owner" ]]; then
                owner_pid=$(cat "${lock_dir}/owner" 2>/dev/null || true)
            fi
            echo "WARN: Failed to acquire queue lock after ${timeout}s (owner_pid=${owner_pid:-unknown})" >&2
            return 1
        fi

        # Check for stale lock (owner process no longer exists)
        local owner_pid=""
        if [[ -f "${lock_dir}/owner" ]]; then
            owner_pid=$(cat "${lock_dir}/owner" 2>/dev/null || true)
        fi
        if [[ -n "${owner_pid}" ]]; then
            if ! _pid_is_alive "${owner_pid}"; then
                # Owner process is gone - remove stale lock
                rm -rf "${lock_dir}" 2>/dev/null || true
                continue
            fi
        else
            # Owner file missing/empty: likely a crash between mkdir and writing owner.
            # Treat as stale if it persists beyond a short grace period.
            local no_owner_grace="${QUEUE_LOCK_NOOWNER_STALE_SECONDS:-30}"
            if ! [[ "${no_owner_grace}" =~ ^[0-9]+$ ]]; then
                no_owner_grace=30
            fi
            local lock_mtime=""
            lock_mtime=$(_file_mtime_epoch "${lock_dir}" 2>/dev/null || echo "")
            if [[ -n "${lock_mtime}" ]]; then
                local lock_age=$((now - lock_mtime))
                if [[ ${lock_age} -ge ${no_owner_grace} ]]; then
                    rm -rf "${lock_dir}" 2>/dev/null || true
                    continue
                fi
            fi
        fi

        # Brief sleep before retry (100ms)
        _sleep 0.1
    done
}

# Release queue lock
# Usage: release_queue_lock
release_queue_lock() {
    if [[ -n "${QUEUE_LOCK_DIR:-}" && -d "${QUEUE_LOCK_DIR}" ]]; then
        # Verify we own the lock before releasing
        local my_pid="${BASHPID:-$$}"
        local owner_pid=""
        if [[ -f "${QUEUE_LOCK_DIR}/owner" ]]; then
            owner_pid=$(cat "${QUEUE_LOCK_DIR}/owner" 2>/dev/null || true)
        fi
        if [[ -z "${owner_pid}" || "${owner_pid}" == "${my_pid}" ]]; then
            rm -rf "${QUEUE_LOCK_DIR}" 2>/dev/null || true
        fi
        unset QUEUE_LOCK_DIR
    fi
}

# Execute action with queue lock
# Usage: with_queue_lock <command> [args...]
with_queue_lock() {
    acquire_queue_lock || return 1
    local result=0
    "$@" || result=$?
    release_queue_lock
    return ${result}
}

# ============ TASK OPERATIONS ============

# Add a task to the queue (wrapper around create_task)
# Usage: add_task <task_type> <model_id> <model_name> <model_size_gb> <dependencies> <params_json> [priority]
add_task() {
    local task_type="$1"
    local model_id="$2"
    local model_name="$3"
    local model_size_gb="$4"
    local dependencies="$5"
    local params_json="$6"
    local priority="${7:-50}"

    # Increment task sequence
    export TASK_SEQUENCE=$((${TASK_SEQUENCE:-0} + 1))

    create_task "${QUEUE_DIR}" "${task_type}" "${model_id}" "${model_name}" \
        "${model_size_gb}" "${dependencies}" "${params_json}" "${priority}"
}

# Capture add_task output without command substitution so TASK_SEQUENCE changes
# stay in the current shell and task ids do not silently reuse the same sequence.
# Usage: capture_add_task <output_var> <task_type> <model_id> <model_name> <model_size_gb> <dependencies> <params_json> [priority]
capture_add_task() {
    local output_var="$1"
    shift

    local capture_file
    capture_file="$(mktemp "${TMPDIR:-/tmp}/invarlock-add-task.XXXXXX")" || return 1

    local rc=0
    add_task "$@" > "${capture_file}" || rc=$?

    local task_id=""
    if [[ -f "${capture_file}" ]]; then
        task_id="$(tail -n 1 "${capture_file}" | tr -d '\r\n')"
        rm -f "${capture_file}"
    fi

    if [[ ${rc} -ne 0 ]]; then
        return "${rc}"
    fi
    if [[ -z "${task_id}" ]]; then
        echo "ERROR: add_task did not return a task id" >&2
        return 1
    fi

    printf -v "${output_var}" '%s' "${task_id}"
}

# Get list of task files by status
# Usage: get_tasks_by_status <status>
# Returns: newline-separated list of task file paths
get_tasks_by_status() {
    local status="$1"
    local dir="${QUEUE_DIR}/${status}"

    if [[ -d "${dir}" ]]; then
        find "${dir}" -name "*.task" -type f 2>/dev/null | sort
    fi
}

# Count tasks by status
# Usage: count_tasks <status>
count_tasks() {
    local status="$1"
    local dir="${QUEUE_DIR}/${status}"

    if [[ -d "${dir}" ]]; then
        find "${dir}" -name "*.task" -type f 2>/dev/null | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

# Get queue statistics
# Usage: get_queue_stats
get_queue_stats() {
    local pending=$(count_tasks "pending")
    local ready=$(count_tasks "ready")
    local running=$(count_tasks "running")
    local completed=$(count_tasks "completed")
    local failed=$(count_tasks "failed")
    local total=$((pending + ready + running + completed + failed))

    echo "${pending}:${ready}:${running}:${completed}:${failed}:${total}"
}

# Print queue statistics
# Usage: print_queue_stats
print_queue_stats() {
    IFS=':' read -r pending ready running completed failed total <<< "$(get_queue_stats)"

    echo "=== QUEUE STATUS ==="
    echo "Pending:   ${pending}"
    echo "Ready:     ${ready}"
    echo "Running:   ${running}"
    echo "Completed: ${completed}"
    echo "Failed:    ${failed}"
    echo "Total:     ${total}"
}

# Check if queue is empty (all work done or failed)
# Usage: is_queue_empty
is_queue_empty() {
    IFS=':' read -r pending ready running completed failed total <<< "$(get_queue_stats)"

    [[ $((pending + ready + running)) -eq 0 ]]
}

# Check if all tasks are complete (none pending/ready/running/failed)
# Usage: is_queue_complete
is_queue_complete() {
    IFS=':' read -r pending ready running completed failed total <<< "$(get_queue_stats)"

    [[ $((pending + ready + running)) -eq 0 && ${failed} -eq 0 ]]
}

count_pending_tasks_blocked_by_failed_dependencies() {
    local blocked=0
    local task_file
    local dep

    for task_file in "${QUEUE_DIR}/pending"/*.task; do
        [[ -f "${task_file}" ]] || continue

        local blocked_by_failed_dep="false"
        while IFS= read -r dep; do
            [[ -n "${dep}" ]] || continue
            if [[ -f "${QUEUE_DIR}/failed/${dep}.task" ]]; then
                blocked_by_failed_dep="true"
                break
            fi
        done < <(get_task_dependencies "${task_file}" 2>/dev/null || true)

        if [[ "${blocked_by_failed_dep}" == "true" ]]; then
            blocked=$((blocked + 1))
        fi
    done

    echo "${blocked}"
}

queue_terminal_state() {
    IFS=':' read -r pending ready running completed failed total <<< "$(get_queue_stats)"

    if [[ $((pending + ready + running)) -eq 0 ]]; then
        if [[ ${failed} -eq 0 ]]; then
            echo "completed"
        else
            echo "completed_with_failures"
        fi
        return 0
    fi

    if [[ ${pending} -gt 0 && ${ready} -eq 0 && ${running} -eq 0 && ${failed} -gt 0 ]]; then
        local blocked_pending=0
        blocked_pending="$(count_pending_tasks_blocked_by_failed_dependencies)"
        if [[ "${blocked_pending}" == "${pending}" ]]; then
            echo "blocked_failed_dependencies"
            return 0
        fi
    fi

    return 1
}
