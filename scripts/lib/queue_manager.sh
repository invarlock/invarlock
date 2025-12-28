#!/usr/bin/env bash
# queue_manager.sh - File-based task queue with atomic operations
# Version: v2.0.1 (InvarLock B200 Validation Suite)
# Dependencies: task_serialization.sh, jq
# Usage: sourced by gpu_worker.sh and scheduler to manage task lifecycle
#
# Provides functions to:
# - Initialize and manage queue directories
# - Atomically claim/complete/fail tasks
# - Resolve task dependencies
# - Track queue statistics

# Source task serialization if not already sourced
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${TASK_SERIALIZATION_LOADED:-}" ]]; then
    source "${SCRIPT_DIR}/task_serialization.sh"
    export TASK_SERIALIZATION_LOADED=1
fi

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
    "initialized_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
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
    local deadline=$(($(date +%s) + timeout))

    while true; do
        # Try to create lock directory (atomic operation)
        if mkdir "${lock_dir}" 2>/dev/null; then
            # Successfully acquired lock - record owner
            echo "${my_pid}" > "${lock_dir}/owner" 2>/dev/null || true
            export QUEUE_LOCK_DIR="${lock_dir}"
            return 0
        fi

        # Check if we've exceeded timeout
        local now=$(date +%s)
        if [[ ${now} -ge ${deadline} ]]; then
            echo "ERROR: Failed to acquire queue lock after ${timeout}s" >&2
            return 1
        fi

        # Check for stale lock (owner process no longer exists)
        local owner_pid=""
        if [[ -f "${lock_dir}/owner" ]]; then
            owner_pid=$(cat "${lock_dir}/owner" 2>/dev/null)
        fi
        if [[ -n "${owner_pid}" && ! -d "/proc/${owner_pid}" ]]; then
            # Owner process is gone - remove stale lock
            rm -rf "${lock_dir}" 2>/dev/null || true
            continue
        fi

        # Brief sleep before retry (100ms)
        sleep 0.1
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
            owner_pid=$(cat "${QUEUE_LOCK_DIR}/owner" 2>/dev/null)
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

# ============ TASK STATE TRANSITIONS ============

# Move task from pending to ready (when dependencies are satisfied)
# Usage: mark_task_ready <task_id>
mark_task_ready() {
    local task_id="$1"
    local src="${QUEUE_DIR}/pending/${task_id}.task"
    local dst="${QUEUE_DIR}/ready/${task_id}.task"

    if [[ -f "${src}" ]]; then
        update_task_status "${src}" "ready"
        mv "${src}" "${dst}"
        return 0
    fi
    return 1
}

# Claim a task for execution (atomic move to running)
# Usage: claim_task <task_id> <gpu_id>
# Returns: 0 if successful, 1 if task no longer available
claim_task() {
    local task_id="$1"
    local gpu_id="$2"
    local src="${QUEUE_DIR}/ready/${task_id}.task"
    local dst="${QUEUE_DIR}/running/${task_id}.task"

    # Use queue lock for atomic operation
    acquire_queue_lock 5 || return 1

    if [[ -f "${src}" ]]; then
        mark_task_started "${src}" "${gpu_id}"
        mv "${src}" "${dst}"
        release_queue_lock
        return 0
    fi

    release_queue_lock
    return 1
}

# Complete a task (move from running to completed)
# Usage: complete_task <task_id>
complete_task() {
    local task_id="$1"
    local src="${QUEUE_DIR}/running/${task_id}.task"
    local dst="${QUEUE_DIR}/completed/${task_id}.task"

    if [[ -f "${src}" ]]; then
        mark_task_completed "${src}"
        mv "${src}" "${dst}"
        rm -f "${QUEUE_DIR}/running/${task_id}.pid"

        # Update progress state
        update_progress_state

        return 0
    fi
    return 1
}

# Fail a task (move from running to failed)
# Usage: fail_task <task_id> <error_message>
fail_task() {
    local task_id="$1"
    local error_msg="$2"
    local src="${QUEUE_DIR}/running/${task_id}.task"
    local dst="${QUEUE_DIR}/failed/${task_id}.task"

    if [[ -f "${src}" ]]; then
        mark_task_failed "${src}" "${error_msg}"
        mv "${src}" "${dst}"
        rm -f "${QUEUE_DIR}/running/${task_id}.pid"

        # Update progress state
        update_progress_state

        return 0
    fi
    return 1
}

# Retry a failed task (move from failed to pending)
# Usage: retry_task <task_id>
retry_task() {
    local task_id="$1"
    local src="${QUEUE_DIR}/failed/${task_id}.task"
    local dst="${QUEUE_DIR}/pending/${task_id}.task"

    if [[ -f "${src}" ]]; then
        local retries=$(get_task_field "${src}" "retries")
        local max_retries=$(get_task_field "${src}" "max_retries")

        if [[ ${retries} -lt ${max_retries} ]]; then
            increment_task_retries "${src}"
            update_task_status "${src}" "pending"
            mv "${src}" "${dst}"
            return 0
        else
            echo "Task ${task_id} has exceeded max retries (${max_retries})" >&2
            return 1
        fi
    fi
    return 1
}

# Reclaim orphaned tasks (tasks stuck in running without active worker)
# Usage: reclaim_orphaned_tasks <gpu_id>
reclaim_orphaned_tasks() {
    local gpu_id="$1"
    local count=0
    local running_dir="${QUEUE_DIR}/running"

    for task_file in "${running_dir}"/*.task; do
        [[ -f "${task_file}" ]] || continue

        local task_gpu=$(get_task_field "${task_file}" "gpu_id")
        if [[ "${task_gpu}" == "${gpu_id}" ]]; then
            local task_id=$(get_task_id "${task_file}")
            echo "Reclaiming orphaned task: ${task_id} from GPU ${gpu_id}"

            local assigned_gpus
            assigned_gpus=$(get_task_field "${task_file}" "assigned_gpus")
            [[ -z "${assigned_gpus}" || "${assigned_gpus}" == "null" ]] && assigned_gpus="${gpu_id}"

            local pid_file="${running_dir}/${task_id}.pid"
            local pid_killed="false"

            if [[ -f "${pid_file}" ]]; then
                local pid
                pid=$(cat "${pid_file}" 2>/dev/null || true)
                if [[ -n "${pid}" && "${pid}" =~ ^[0-9]+$ ]]; then
                    if kill -0 "${pid}" 2>/dev/null; then
                        echo "  Killing task process group for ${task_id} (PID ${pid})"
                        kill -TERM "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
                        sleep 2
                        kill -KILL "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
                        pid_killed="true"
                    fi
                fi
                rm -f "${pid_file}"
            fi

            if [[ "${pid_killed}" != "true" && -n "${assigned_gpus}" ]]; then
                local can_kill="true"
                if [[ -n "${GPU_RESERVATION_DIR:-}" ]]; then
                    IFS=',' read -ra gpus <<< "${assigned_gpus}"
                    for gid in "${gpus[@]}"; do
                        local lock_file="${GPU_RESERVATION_DIR}/gpu_${gid}.lock"
                        local owner
                        owner=$(cat "${lock_file}" 2>/dev/null || true)
                        if [[ -z "${owner}" || "${owner}" != "${task_id}" ]]; then
                            can_kill="false"
                            break
                        fi
                    done
                fi

                if [[ "${can_kill}" == "true" ]]; then
                    if type kill_gpu_processes &>/dev/null; then
                        echo "  Killing GPU processes on ${assigned_gpus} for ${task_id}"
                        kill_gpu_processes "${assigned_gpus}"
                    fi
                fi
            fi

            if type release_gpus &>/dev/null; then
                release_gpus "${task_id}"
            fi

            # Move back to pending for re-execution
            update_task_status "${task_file}" "pending"
            assign_task_gpu "${task_file}" "-1"
            update_task_field "${task_file}" "assigned_gpus" "null" "true" 2>/dev/null || true
            mv "${task_file}" "${QUEUE_DIR}/pending/"
            count=$((count + 1))
        fi
    done

    echo "Reclaimed ${count} orphaned tasks from GPU ${gpu_id}"
}

# ============ DEPENDENCY RESOLUTION ============

# Check if all dependencies of a task are completed
# Usage: check_dependencies_met <task_file>
# Returns: 0 if all deps completed, 1 otherwise
check_dependencies_met() {
    local task_file="$1"

    # Use mapfile for safe array population (handles special chars)
    local -a deps=()
    while IFS= read -r dep; do
        [[ -n "${dep}" ]] && deps+=("${dep}")
    done < <(get_task_dependencies "${task_file}")

    if [[ ${#deps[@]} -eq 0 ]]; then
        return 0  # No dependencies
    fi

    for dep_id in "${deps[@]}"; do
        if [[ ! -f "${QUEUE_DIR}/completed/${dep_id}.task" ]]; then
            return 1  # Dependency not completed
        fi
    done

    return 0  # All dependencies completed
}

# Cancel tasks whose dependencies have permanently failed.
# This prevents the queue from stalling forever when an upstream task fails.
# Inspired by Slurm dependency semantics (dependent jobs do not start if parent fails).
#
# Usage: cancel_tasks_with_failed_dependencies [grace_seconds]
# Returns: number of tasks moved pending->failed.
cancel_tasks_with_failed_dependencies() {
    local grace="${1:-${CANCEL_BLOCKED_TASKS_GRACE_SECONDS:-90}}"
    local canceled=0

    if ! [[ "${grace}" =~ ^[0-9]+$ ]]; then
        grace=90
    fi

    acquire_queue_lock 10 || { echo "0"; return 0; }

    for task_file in "${QUEUE_DIR}/pending"/*.task; do
        [[ -f "${task_file}" ]] || continue

        local -a deps=()
        while IFS= read -r dep; do
            [[ -n "${dep}" ]] && deps+=("${dep}")
        done < <(get_task_dependencies "${task_file}")

        [[ ${#deps[@]} -eq 0 ]] && continue

        local -a failed_deps=()
        for dep_id in "${deps[@]}"; do
            local dep_file="${QUEUE_DIR}/failed/${dep_id}.task"
            [[ -f "${dep_file}" ]] || continue

            # Use mtime as the failure timestamp to avoid GNU date parsing assumptions.
            local dep_mtime=""
            dep_mtime=$(stat -c %Y "${dep_file}" 2>/dev/null || stat -f %m "${dep_file}" 2>/dev/null || echo "")
            if [[ -z "${dep_mtime}" ]]; then
                failed_deps+=("${dep_id}")
                continue
            fi
            local dep_age=$(( $(date +%s) - dep_mtime ))
            if [[ ${dep_age} -ge ${grace} ]]; then
                failed_deps+=("${dep_id}")
            fi
        done

        if [[ ${#failed_deps[@]} -gt 0 ]]; then
            local msg
            msg="Dependency failed: $(IFS=','; echo "${failed_deps[*]}")"
            mark_task_failed "${task_file}" "${msg}" 2>/dev/null || true
            mv "${task_file}" "${QUEUE_DIR}/failed/"
            canceled=$((canceled + 1))
        fi
    done

    release_queue_lock

    if [[ ${canceled} -gt 0 ]]; then
        update_progress_state 2>/dev/null || true
    fi

    echo "${canceled}"
}

# Resolve dependencies and move ready tasks from pending to ready
# Usage: resolve_dependencies
# Returns: number of tasks moved to ready
resolve_dependencies() {
    local moved=0

    acquire_queue_lock 10 || return 0

    for task_file in "${QUEUE_DIR}/pending"/*.task; do
        [[ -f "${task_file}" ]] || continue

        if check_dependencies_met "${task_file}"; then
            local task_id=$(get_task_id "${task_file}")
            update_task_status "${task_file}" "ready"
            mv "${task_file}" "${QUEUE_DIR}/ready/"
            moved=$((moved + 1))
        fi
    done

    release_queue_lock
    echo "${moved}"
}

# Update dependents after task completion
# Usage: update_dependents <completed_task_id>
update_dependents() {
    local completed_id="$1"

    # Check all pending tasks for this dependency
    for task_file in "${QUEUE_DIR}/pending"/*.task; do
        [[ -f "${task_file}" ]] || continue

        local deps=$(get_task_dependencies "${task_file}" | tr '\n' ' ')
        if [[ " ${deps} " =~ " ${completed_id} " ]]; then
            if check_dependencies_met "${task_file}"; then
                local task_id=$(get_task_id "${task_file}")
                mark_task_ready "${task_id}"
            fi
        fi
    done
}

# ============ PROGRESS TRACKING ============

# Update progress state file
# Usage: update_progress_state
update_progress_state() {
    local state_file="${QUEUE_DIR}/../state/progress.json"

    IFS=':' read -r pending ready running completed failed total <<< "$(get_queue_stats)"

    local status="running"
    if [[ $((pending + ready + running)) -eq 0 ]]; then
        if [[ ${failed} -eq 0 ]]; then
            status="completed"
        else
            status="completed_with_failures"
        fi
    fi

    local progress_pct=0
    if [[ ${total} -gt 0 ]]; then
        progress_pct=$((completed * 100 / total))
    fi

    cat > "${state_file}" << EOF
{
    "updated_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "total_tasks": ${total},
    "pending_tasks": ${pending},
    "ready_tasks": ${ready},
    "running_tasks": ${running},
    "completed_tasks": ${completed},
    "failed_tasks": ${failed},
    "progress_pct": ${progress_pct},
    "status": "${status}"
}
EOF
}

# ============ TASK SEARCH ============

# Find task by ID across all queues
# Usage: find_task <task_id>
# Returns: full path to task file, or empty if not found
find_task() {
    local task_id="$1"

    for status in pending ready running completed failed; do
        local path="${QUEUE_DIR}/${status}/${task_id}.task"
        if [[ -f "${path}" ]]; then
            echo "${path}"
            return 0
        fi
    done

    return 1
}

# Find tasks by model name
# Usage: find_tasks_by_model <model_name> [status]
find_tasks_by_model() {
    local model_name="$1"
    local status="${2:-}"

    local search_dirs=()
    if [[ -n "${status}" ]]; then
        search_dirs=("${QUEUE_DIR}/${status}")
    else
        search_dirs=("${QUEUE_DIR}/pending" "${QUEUE_DIR}/ready" "${QUEUE_DIR}/running" "${QUEUE_DIR}/completed" "${QUEUE_DIR}/failed")
    fi

    for dir in "${search_dirs[@]}"; do
        for task_file in "${dir}"/*.task; do
            [[ -f "${task_file}" ]] || continue
            local task_model=$(get_task_field "${task_file}" "model_name")
            if [[ "${task_model}" == "${model_name}" ]]; then
                echo "${task_file}"
            fi
        done
    done
}

# Refresh task memory estimates for any models with existing profiles.
# Usage: refresh_task_memory_from_profiles <output_dir>
refresh_task_memory_from_profiles() {
    local output_dir="$1"

    for model_dir in "${output_dir}"/*; do
        [[ -d "${model_dir}" ]] || continue

        local model_name
        model_name=$(basename "${model_dir}")
        local model_id=""
        if [[ -f "${model_dir}/.model_id" ]]; then
            model_id=$(cat "${model_dir}/.model_id" 2>/dev/null || true)
        fi

        update_model_task_memory "${model_name}" "${output_dir}" "${model_id}"
    done
}

export_memory_plan() {
    local output_dir="$1"
    local plan_dir="${output_dir}/analysis"
    local plan_file="${plan_dir}/memory_plan.csv"
    local tmp_file="${plan_file}.tmp.$$"

    mkdir -p "${plan_dir}"

    local csv_escape
    csv_escape() {
        local val="$1"
        val="${val//\"/\"\"}"
        printf "\"%s\"" "${val}"
    }

    echo "task_id,status,task_type,model_name,model_id,model_size_gb,required_gpus,adaptive_gpus,assigned_gpus,priority" > "${tmp_file}"

    local status
    for status in pending ready running completed failed; do
        [[ -d "${QUEUE_DIR}/${status}" ]] || continue
        for task_file in "${QUEUE_DIR}/${status}"/*.task; do
            [[ -f "${task_file}" ]] || continue

            local task_id
            local task_type
            local model_name
            local model_id
            local model_size_gb
            local required_gpus
            local adaptive_gpus
            local assigned_gpus
            local priority

            task_id=$(get_task_id "${task_file}")
            task_type=$(get_task_type "${task_file}")
            model_name=$(get_task_field "${task_file}" "model_name")
            model_id=$(get_task_field "${task_file}" "model_id")
            model_size_gb=$(get_task_field "${task_file}" "model_size_gb")
            required_gpus=$(get_task_field "${task_file}" "required_gpus")
            adaptive_gpus=$(get_task_field "${task_file}" "adaptive_gpus")
            assigned_gpus=$(get_task_field "${task_file}" "assigned_gpus")
            priority=$(get_task_field "${task_file}" "priority")

            echo "$(csv_escape "${task_id}"),$(csv_escape "${status}"),$(csv_escape "${task_type}"),$(csv_escape "${model_name}"),$(csv_escape "${model_id}"),$(csv_escape "${model_size_gb}"),$(csv_escape "${required_gpus}"),$(csv_escape "${adaptive_gpus}"),$(csv_escape "${assigned_gpus}"),$(csv_escape "${priority}")" >> "${tmp_file}"
        done
    done

    mv "${tmp_file}" "${plan_file}"
}

# Find tasks by type
# Usage: find_tasks_by_type <task_type> [status]
find_tasks_by_type() {
    local task_type="$1"
    local status="${2:-}"

    local search_dirs=()
    if [[ -n "${status}" ]]; then
        search_dirs=("${QUEUE_DIR}/${status}")
    else
        search_dirs=("${QUEUE_DIR}/pending" "${QUEUE_DIR}/ready" "${QUEUE_DIR}/running" "${QUEUE_DIR}/completed" "${QUEUE_DIR}/failed")
    fi

    for dir in "${search_dirs[@]}"; do
        for task_file in "${dir}"/*.task; do
            [[ -f "${task_file}" ]] || continue
            local type=$(get_task_type "${task_file}")
            if [[ "${type}" == "${task_type}" ]]; then
                echo "${task_file}"
            fi
        done
    done
}

# ============ TASK GRAPH GENERATION ============

# Generate all tasks for a model
# Usage: generate_model_tasks <model_idx> <model_id> <model_name>
generate_model_tasks() {
    local model_idx="$1"
    local model_id="$2"
    local model_name="$3"

    # Calculate model size for memory estimation
    local base_size=$(estimate_model_memory "${model_id}" "EVAL_BASELINE")

    # Decide whether to use batch edit creation or per-edit tasks.
    # Deep-copying a 70B+ model for batch edits can exceed per-GPU memory,
    # so we disable CREATE_EDITS_BATCH for very large models and fall back
    # to legacy per-edit CREATE_EDIT tasks in that case.
    local use_batch="true"
    local model_lower
    model_lower=$(echo "${model_id}" | tr '[:upper:]' '[:lower:]')
    if [[ "${model_lower}" =~ 70b || "${model_lower}" =~ 72b || "${model_lower}" =~ 65b || "${model_lower}" =~ mixtral || "${model_lower}" =~ 8x7b || "${model_lower}" =~ moe ]]; then
        use_batch="false"
    elif [[ -n "${base_size}" ]]; then
        # For very large models, EVAL_BASELINE estimates can still be high.
        # Treat anything >=170GB as "large" and avoid batch edits.
        if [[ "${base_size}" -ge 170 ]]; then
            use_batch="false"
        fi
    fi

    # Track task IDs for dependencies
    local task_ids=()

    # 1. SETUP_BASELINE (no dependencies)
    local setup_id=$(add_task "SETUP_BASELINE" "${model_id}" "${model_name}" \
        "$(estimate_model_memory "${model_id}" "SETUP_BASELINE")" \
        "none" '{"model_idx": '"${model_idx}"'}' 90)
    task_ids+=("${setup_id}")
    echo "Created: ${setup_id}"

    # 2. EVAL_BASELINE (depends on setup)
    local eval_base_id=$(add_task "EVAL_BASELINE" "${model_id}" "${model_name}" \
        "$(estimate_model_memory "${model_id}" "EVAL_BASELINE")" \
        "${setup_id}" '{}' 80)
    task_ids+=("${eval_base_id}")
    echo "Created: ${eval_base_id}"

    # 3. CALIBRATION_RUN × N (depend on setup)
    local cal_ids=()
    local calibration_runs="${DRIFT_CALIBRATION_RUNS:-5}"
    for run in $(seq 1 "${calibration_runs}"); do
        local cal_id=$(add_task "CALIBRATION_RUN" "${model_id}" "${model_name}" \
            "$(estimate_model_memory "${model_id}" "CALIBRATION_RUN")" \
            "${setup_id}" '{"run": '"${run}"', "seed": '"$((41 + run))"'}' 85)
        cal_ids+=("${cal_id}")
        task_ids+=("${cal_id}")
        echo "Created: ${cal_id}"
    done

    # 4. GENERATE_PRESET (depends on all calibration runs)
    local cal_deps=$(IFS=','; echo "${cal_ids[*]}")
    local preset_id=$(add_task "GENERATE_PRESET" "${model_id}" "${model_name}" \
        5 "${cal_deps}" '{}' 75)
    task_ids+=("${preset_id}")
    echo "Created: ${preset_id}"

    # 5/6. Edit creation + eval/certify
    # Clean edits (3 certify runs each)
    local clean_edits=("quant_rtn:8:128:ffn" "fp4_quant:e2m1:ffn" "magnitude_prune:0.1:ffn" "lowrank_svd:256:ffn")
    # Stress edits (2 certify runs each)
    local stress_edits=("quant_rtn:4:32:all" "fp4_quant:aggressive:all" "magnitude_prune:0.5:all" "lowrank_svd:32:all")

    if [[ "${use_batch}" == "true" ]]; then
        # CREATE_EDITS_BATCH - Create all 8 edits with a single model load (Batch optimization)
        # This replaces 8 separate CREATE_EDIT tasks, reducing model load overhead from 8× to 1×.
        local all_edit_specs='[
            {"spec": "quant_rtn:8:128:ffn", "version": "clean"},
            {"spec": "fp4_quant:e2m1:ffn", "version": "clean"},
            {"spec": "magnitude_prune:0.1:ffn", "version": "clean"},
            {"spec": "lowrank_svd:256:ffn", "version": "clean"},
            {"spec": "quant_rtn:4:32:all", "version": "stress"},
            {"spec": "fp4_quant:aggressive:all", "version": "stress"},
            {"spec": "magnitude_prune:0.5:all", "version": "stress"},
            {"spec": "lowrank_svd:32:all", "version": "stress"}
        ]'
        local batch_edit_id=$(add_task "CREATE_EDITS_BATCH" "${model_id}" "${model_name}" \
            "$(estimate_model_memory "${model_id}" "CREATE_EDIT")" \
            "${setup_id}" '{"edit_specs": '"${all_edit_specs}"'}' 70)
        task_ids+=("${batch_edit_id}")
        echo "Created: ${batch_edit_id} (batch creates 8 edits with single model load)"

        # Eval and Certify tasks for each edit (depend on batch creation)
        for edit_spec in "${clean_edits[@]}"; do
            generate_eval_certify_tasks "${model_id}" "${model_name}" "${batch_edit_id}" "${preset_id}" \
                "${edit_spec}" "clean" 3
        done
        for edit_spec in "${stress_edits[@]}"; do
            generate_eval_certify_tasks "${model_id}" "${model_name}" "${batch_edit_id}" "${preset_id}" \
                "${edit_spec}" "stress" 2
        done
    else
        # For very large models (70B+), avoid batch edit creation to prevent OOM from deep copies.
        # Fall back to legacy per-edit CREATE_EDIT + EVAL_EDIT + CERTIFY_EDIT tasks.
        echo "Model ${model_name}: estimated ${base_size}GB for eval - disabling CREATE_EDITS_BATCH, using per-edit tasks"

        for edit_spec in "${clean_edits[@]}"; do
            generate_edit_tasks "${model_id}" "${model_name}" "${setup_id}" "${preset_id}" \
                "${edit_spec}" "clean" 3
        done
        for edit_spec in "${stress_edits[@]}"; do
            generate_edit_tasks "${model_id}" "${model_name}" "${setup_id}" "${preset_id}" \
                "${edit_spec}" "stress" 2
        done
    fi

    # 7. Error injection tests (5 types)
    local error_types=("nan_injection" "inf_injection" "extreme_quant" "scale_explosion" "zero_layer")
    for error_type in "${error_types[@]}"; do
        # CREATE_ERROR
        local error_create_id=$(add_task "CREATE_ERROR" "${model_id}" "${model_name}" \
            "$(estimate_model_memory "${model_id}" "CREATE_ERROR")" \
            "${setup_id}" '{"error_type": "'"${error_type}"'"}' 60)
        echo "Created: ${error_create_id}"

        # CERTIFY_ERROR
        local error_cert_id=$(add_task "CERTIFY_ERROR" "${model_id}" "${model_name}" \
            "$(estimate_model_memory "${model_id}" "CERTIFY_ERROR")" \
            "${error_create_id},${preset_id}" '{"error_type": "'"${error_type}"'"}' 55)
        echo "Created: ${error_cert_id}"
    done

    echo "Generated tasks for model ${model_name}"
}

# Generate eval and certify tasks for an edit (Split Eval optimization)
# Usage: generate_eval_certify_tasks <model_id> <model_name> <batch_edit_id> <preset_id> <edit_spec> <version> <cert_runs>
# Note: Edit creation is handled by CREATE_EDITS_BATCH; this function creates eval+certify tasks
generate_eval_certify_tasks() {
    local model_id="$1"
    local model_name="$2"
    local batch_edit_id="$3"
    local preset_id="$4"
    local edit_spec="$5"
    local version="$6"
    local cert_runs="$7"

    # Split Eval: 4 parallel benchmark tasks instead of 1 monolithic EVAL_EDIT
    # This enables better parallelization across GPUs (4× parallelism opportunity)
    local benchmarks=("mmlu" "hellaswag" "arc" "winogrande")
    local eval_ids=()

    for benchmark in "${benchmarks[@]}"; do
        local task_type="EVAL_${benchmark^^}"  # uppercase: EVAL_MMLU, EVAL_HELLASWAG, etc.
        local eval_id=$(add_task "${task_type}" "${model_id}" "${model_name}" \
            "$(estimate_model_memory "${model_id}" "EVAL_EDIT")" \
            "${batch_edit_id}" '{"edit_spec": "'"${edit_spec}"'", "benchmark": "'"${benchmark}"'"}' 65)
        eval_ids+=("${eval_id}")
        echo "Created: ${eval_id} (${benchmark} eval)"
    done

    # CERTIFY_EDIT depends on batch edit creation + preset (not on evals - they run in parallel)
    # Certification uses pre-computed reference values, not eval results
    for run in $(seq 1 "${cert_runs}"); do
        local cert_id=$(add_task "CERTIFY_EDIT" "${model_id}" "${model_name}" \
            "$(estimate_model_memory "${model_id}" "CERTIFY_EDIT")" \
            "${batch_edit_id},${preset_id}" '{"edit_spec": "'"${edit_spec}"'", "version": "'"${version}"'", "run": '"${run}"'}' 65)
        echo "Created: ${cert_id}"
    done
}

# Legacy: Generate tasks for an edit type (deprecated - use generate_eval_certify_tasks)
# Kept for backwards compatibility with external scripts
# Usage: generate_edit_tasks <model_id> <model_name> <setup_id> <preset_id> <edit_spec> <version> <cert_runs>
generate_edit_tasks() {
    local model_id="$1"
    local model_name="$2"
    local setup_id="$3"
    local preset_id="$4"
    local edit_spec="$5"
    local version="$6"
    local cert_runs="$7"

    echo "WARNING: generate_edit_tasks is deprecated; use CREATE_EDITS_BATCH + generate_eval_certify_tasks" >&2

    # CREATE_EDIT (legacy single edit)
    local edit_id=$(add_task "CREATE_EDIT" "${model_id}" "${model_name}" \
        "$(estimate_model_memory "${model_id}" "CREATE_EDIT")" \
        "${setup_id}" '{"edit_spec": "'"${edit_spec}"'", "version": "'"${version}"'"}' 70)
    echo "Created: ${edit_id}"

    # EVAL_EDIT (legacy monolithic eval)
    local eval_id=$(add_task "EVAL_EDIT" "${model_id}" "${model_name}" \
        "$(estimate_model_memory "${model_id}" "EVAL_EDIT")" \
        "${edit_id}" '{"edit_spec": "'"${edit_spec}"'"}' 65)
    echo "Created: ${eval_id}"

    # CERTIFY_EDIT × cert_runs
    for run in $(seq 1 "${cert_runs}"); do
        local cert_id=$(add_task "CERTIFY_EDIT" "${model_id}" "${model_name}" \
            "$(estimate_model_memory "${model_id}" "CERTIFY_EDIT")" \
            "${edit_id},${preset_id}" '{"edit_spec": "'"${edit_spec}"'", "version": "'"${version}"'", "run": '"${run}"'}' 65)
        echo "Created: ${cert_id}"
    done
}

# Generate all tasks for all models
# Usage: generate_all_tasks <models_array>
# Note: models_array should be passed as quoted array string
generate_all_tasks() {
    local model_1="$1"
    local model_2="$2"
    local model_3="$3"
    local model_4="$4"
    local model_5="$5"
    local model_6="$6"
    local model_7="$7"
    local model_8="$8"

    local models=("${model_1}" "${model_2}" "${model_3}" "${model_4}" \
                  "${model_5}" "${model_6}" "${model_7}" "${model_8}")

    export TASK_SEQUENCE=0

    for idx in "${!models[@]}"; do
        local model_id="${models[$idx]}"
        if [[ -n "${model_id}" ]]; then
            # Use full model id (including org) for filesystem-safe name to avoid collisions
            # Example: meta-llama/Llama-2-7b-hf -> meta-llama__llama-2-7b-hf
            local model_name
            model_name=$(echo "${model_id}" \
                | tr '[:upper:]' '[:lower:]' \
                | sed 's#/#__#g' \
                | tr ' ' '_' \
                | tr -cd '[:alnum:]_-')
            echo ""
            echo "=== Generating tasks for model $((idx + 1)): ${model_name} ==="
            generate_model_tasks "$((idx + 1))" "${model_id}" "${model_name}"
        fi
    done

    # Initial dependency resolution
    echo ""
    echo "=== Resolving initial dependencies ==="
    local moved=$(resolve_dependencies)
    echo "Moved ${moved} tasks to ready queue"

    # Update state
    update_progress_state

    echo ""
    print_queue_stats
}

# Update task memory estimates for a model based on its on-disk profile.
# Usage: update_model_task_memory <model_name> <output_dir> [model_id]
update_model_task_memory() {
    local model_name="$1"
    local output_dir="$2"
    local model_id="${3:-}"
    local profile_path=""
    local baseline_path_file="${output_dir}/${model_name}/.baseline_path"
    local baseline_path=""

    if [[ -f "${baseline_path_file}" ]]; then
        baseline_path=$(cat "${baseline_path_file}" 2>/dev/null || true)
        if [[ -n "${baseline_path}" ]]; then
            profile_path="${baseline_path}/model_profile.json"
        fi
    fi

    if [[ -z "${profile_path}" || ! -f "${profile_path}" ]]; then
        profile_path="${output_dir}/${model_name}/models/baseline/model_profile.json"
    fi

    if [[ -n "${model_id}" && ( -z "${profile_path}" || ! -f "${profile_path}" ) ]]; then
        local model_basename
        model_basename=$(basename "${model_id}" | tr '[:upper:]' '[:lower:]' | tr '/' '_')
        local model_sanitized
        model_sanitized=$(echo "${model_id}" \
            | tr '[:upper:]' '[:lower:]' \
            | sed 's#/#__#g' \
            | tr ' ' '_' \
            | tr -cd '[:alnum:]_-')
        local candidate
        for candidate in \
            "${output_dir}/models/${model_sanitized}/baseline/model_profile.json" \
            "${output_dir}/models/${model_basename}/baseline/model_profile.json"; do
            if [[ -f "${candidate}" ]]; then
                profile_path="${candidate}"
                break
            fi
        done
    fi

    [[ -f "${profile_path}" ]] || return 0

    local queue_dirs=("${QUEUE_DIR}/pending" "${QUEUE_DIR}/ready")
    for dir in "${queue_dirs[@]}"; do
        for task_file in "${dir}"/*.task; do
            [[ -f "${task_file}" ]] || continue
            local task_model
            task_model=$(get_task_field "${task_file}" "model_name")
            [[ "${task_model}" == "${model_name}" ]] || continue

            local task_type
            task_type=$(get_task_type "${task_file}")

            local result
            result=$(TASK_TYPE="${task_type}" MODEL_ID="${model_id}" PROFILE_PATH="${profile_path}" \
                EVAL_BATCH_SIZE_SMALL="${EVAL_BATCH_SIZE_SMALL:-auto:16}" \
                EVAL_BATCH_SIZE_MEDIUM="${EVAL_BATCH_SIZE_MEDIUM:-auto:8}" \
                EVAL_BATCH_SIZE_LARGE="${EVAL_BATCH_SIZE_LARGE:-auto:4}" \
                EVAL_BATCH_SIZE_MOE="${EVAL_BATCH_SIZE_MOE:-auto:6}" \
                EVAL_CONTEXT_LEN="${EVAL_CONTEXT_LEN:-2048}" \
                MODEL_LOAD_OVERHEAD_GB="${MODEL_LOAD_OVERHEAD_GB:-4}" \
                EDIT_OVERHEAD_GB="${EDIT_OVERHEAD_GB:-8}" \
                BATCH_EDIT_OVERHEAD_GB="${BATCH_EDIT_OVERHEAD_GB:-8}" \
                EVAL_OVERHEAD_GB="${EVAL_OVERHEAD_GB:-6}" \
                INVARLOCK_OVERHEAD_GB="${INVARLOCK_OVERHEAD_GB:-6}" \
                GPU_MEMORY_PER_DEVICE="${GPU_MEMORY_PER_DEVICE:-${GPU_MEMORY_GB:-180}}" \
                NUM_GPUS="${NUM_GPUS:-8}" \
                python3 - <<'PY'
import json
import math
import os
import re
from pathlib import Path

profile_path = Path(os.environ["PROFILE_PATH"])
task_type = os.environ.get("TASK_TYPE", "")
model_id = (os.environ.get("MODEL_ID") or "").lower()

try:
    profile = json.loads(profile_path.read_text())
except Exception:
    raise SystemExit(0)

if not model_id:
    model_id = str(profile.get("model_id", "")).lower()

weights_gb = profile.get("weights_gb") or 0.0
if not weights_gb:
    weights_bytes = profile.get("weights_bytes") or 0
    weights_gb = weights_bytes / (1024 ** 3) if weights_bytes else 0.0

hidden_size = profile.get("hidden_size")
num_layers = profile.get("num_layers")
num_heads = profile.get("num_heads")
num_kv_heads = profile.get("num_kv_heads") or num_heads
max_pos = profile.get("max_position_embeddings")
dtype_bytes = profile.get("dtype_bytes") or 2

def parse_batch(val, default):
    if not val:
        return default
    val = str(val).strip()
    if val.startswith("auto:"):
        try:
            return int(val.split(":", 1)[1])
        except Exception:
            return default
    try:
        return int(val)
    except Exception:
        return default

def size_category():
    if any(t in model_id for t in ("mixtral", "moe", "8x7b")):
        return "moe"
    if weights_gb >= 120:
        return "70"
    if weights_gb >= 80:
        return "40"
    if weights_gb >= 60:
        return "30"
    if weights_gb >= 24:
        return "13"
    return "7"

category = size_category()

invarlock_cfg = {
    "7":  (2048, 96),
    "13": (1536, 64),
    "30": (1024, 48),
    "40": (1024, 32),
    "moe": (1024, 24),
    "70": (128, 2),
}

seq_len_invarlock, batch_invarlock = invarlock_cfg.get(category, (1024, 32))

eval_batch = {
    "moe": parse_batch(os.environ.get("EVAL_BATCH_SIZE_MOE"), 6),
    "70": parse_batch(os.environ.get("EVAL_BATCH_SIZE_LARGE"), 4),
    "40": parse_batch(os.environ.get("EVAL_BATCH_SIZE_MEDIUM"), 8),
    "30": parse_batch(os.environ.get("EVAL_BATCH_SIZE_MEDIUM"), 8),
    "13": parse_batch(os.environ.get("EVAL_BATCH_SIZE_SMALL"), 16),
    "7":  parse_batch(os.environ.get("EVAL_BATCH_SIZE_SMALL"), 16),
}.get(category, parse_batch(os.environ.get("EVAL_BATCH_SIZE_SMALL"), 16))

eval_context = int(os.environ.get("EVAL_CONTEXT_LEN", "2048"))
seq_len_eval = eval_context
if isinstance(max_pos, int) and max_pos > 0:
    seq_len_eval = min(eval_context, max_pos)

def kv_cache_gb(batch, seq_len):
    if not all(isinstance(x, int) and x > 0 for x in (hidden_size, num_layers, num_heads)):
        return 0.0
    kv_heads = num_kv_heads if isinstance(num_kv_heads, int) and num_kv_heads > 0 else num_heads
    head_dim = hidden_size // num_heads if num_heads else 0
    if head_dim == 0:
        return 0.0
    elems = 2 * num_layers * batch * seq_len * kv_heads * head_dim
    return elems * dtype_bytes / (1024 ** 3)

load_overhead = float(os.environ.get("MODEL_LOAD_OVERHEAD_GB", "4"))
edit_overhead = float(os.environ.get("EDIT_OVERHEAD_GB", "8"))
batch_overhead = float(os.environ.get("BATCH_EDIT_OVERHEAD_GB", "8"))
eval_overhead = float(os.environ.get("EVAL_OVERHEAD_GB", "6"))
inv_overhead = float(os.environ.get("INVARLOCK_OVERHEAD_GB", "6"))

if task_type == "GENERATE_PRESET":
    required = 5.0
elif task_type == "SETUP_BASELINE":
    required = weights_gb + load_overhead
elif task_type == "CREATE_EDITS_BATCH":
    required = (weights_gb * 2.0) + batch_overhead
elif task_type in ("CREATE_EDIT", "CREATE_ERROR"):
    required = weights_gb + edit_overhead
elif task_type in ("EVAL_BASELINE", "EVAL_EDIT", "EVAL_MMLU", "EVAL_HELLASWAG", "EVAL_ARC", "EVAL_WINOGRANDE"):
    required = weights_gb + kv_cache_gb(eval_batch, seq_len_eval) + eval_overhead
elif task_type in ("CALIBRATION_RUN", "CERTIFY_EDIT", "CERTIFY_ERROR"):
    required = weights_gb + kv_cache_gb(batch_invarlock, seq_len_invarlock) + inv_overhead
else:
    required = weights_gb + eval_overhead

per_device = int(os.environ.get("GPU_MEMORY_PER_DEVICE", "180"))
max_gpus = int(os.environ.get("NUM_GPUS", "8"))
required_mem = int(math.ceil(required))
required_gpus = max(1, int(math.ceil(required_mem / per_device)))
if max_gpus > 0:
    required_gpus = min(required_gpus, max_gpus)

print(f"{required_mem} {required_gpus}")
PY
            )

            local required_mem=""
            local required_gpus=""
            read -r required_mem required_gpus <<< "${result}"

            if [[ -n "${required_mem}" && "${required_mem}" =~ ^[0-9]+$ ]]; then
                update_task_field "${task_file}" "model_size_gb" "${required_mem}" "true" 2>/dev/null || true
            fi
            if [[ -n "${required_gpus}" && "${required_gpus}" =~ ^[0-9]+$ ]]; then
                update_task_field "${task_file}" "required_gpus" "${required_gpus}" "true" 2>/dev/null || true
            fi
        done
    done

    if type export_memory_plan &>/dev/null; then
        export_memory_plan "${output_dir}"
    fi
}
