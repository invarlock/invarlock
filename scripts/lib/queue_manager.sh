#!/usr/bin/env bash
# queue_manager.sh - File-based task queue with atomic operations
# Version: v2.0.1 (InvarLock B200 Validation Suite)
# Dependencies: task_serialization.sh, flock
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
acquire_queue_lock() {
    local timeout="${1:-30}"
    local lock_file="${QUEUE_DIR}/queue.lock"

    local fd
    exec {fd}>"${lock_file}"

    if flock -w "${timeout}" "${fd}"; then
        # Store fd in global var for later release
        export QUEUE_LOCK_FD="${fd}"
        return 0
    else
        exec {fd}>&-
        echo "ERROR: Failed to acquire queue lock after ${timeout}s" >&2
        return 1
    fi
}

# Release queue lock
# Usage: release_queue_lock
release_queue_lock() {
    if [[ -n "${QUEUE_LOCK_FD:-}" ]]; then
        exec {QUEUE_LOCK_FD}>&-
        unset QUEUE_LOCK_FD
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

    for task_file in "${QUEUE_DIR}/running"/*.task; do
        [[ -f "${task_file}" ]] || continue

        local task_gpu=$(get_task_field "${task_file}" "gpu_id")
        if [[ "${task_gpu}" == "${gpu_id}" ]]; then
            local task_id=$(get_task_id "${task_file}")
            echo "Reclaiming orphaned task: ${task_id} from GPU ${gpu_id}"

            # Move back to pending for re-execution
            update_task_status "${task_file}" "pending"
            assign_task_gpu "${task_file}" "-1"
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

    # 3. CALIBRATION_RUN × 5 (depend on setup)
    local cal_ids=()
    for run in $(seq 1 5); do
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

    # 5. Clean edits (4 types)
    local clean_edits=("quant_rtn:8:128:ffn" "fp4_quant:e2m1:ffn" "magnitude_prune:0.1:ffn" "lowrank_svd:256:ffn")
    for edit_spec in "${clean_edits[@]}"; do
        generate_edit_tasks "${model_id}" "${model_name}" "${setup_id}" "${preset_id}" \
            "${edit_spec}" "clean" 3
    done

    # 6. Stress edits (4 types)
    local stress_edits=("quant_rtn:4:32:all" "fp4_quant:aggressive:all" "magnitude_prune:0.5:all" "lowrank_svd:32:all")
    for edit_spec in "${stress_edits[@]}"; do
        generate_edit_tasks "${model_id}" "${model_name}" "${setup_id}" "${preset_id}" \
            "${edit_spec}" "stress" 2
    done

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

# Generate tasks for an edit type
# Usage: generate_edit_tasks <model_id> <model_name> <setup_id> <preset_id> <edit_spec> <version> <cert_runs>
generate_edit_tasks() {
    local model_id="$1"
    local model_name="$2"
    local setup_id="$3"
    local preset_id="$4"
    local edit_spec="$5"
    local version="$6"
    local cert_runs="$7"

    # CREATE_EDIT
    local edit_id=$(add_task "CREATE_EDIT" "${model_id}" "${model_name}" \
        "$(estimate_model_memory "${model_id}" "CREATE_EDIT")" \
        "${setup_id}" '{"edit_spec": "'"${edit_spec}"'", "version": "'"${version}"'"}' 70)
    echo "Created: ${edit_id}"

    # EVAL_EDIT
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
            # basename already strips directory, so tr '/' is unnecessary
            # Use tr to make names filesystem-safe (lowercase, replace special chars)
            local model_name=$(basename "${model_id}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr -cd '[:alnum:]_-')
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
