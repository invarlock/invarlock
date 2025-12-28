#!/usr/bin/env bash
# gpu_worker.sh - GPU worker loop for dynamic task execution
# Version: v2.2.0 (InvarLock B200 Validation Suite)
# Dependencies: scheduler.sh, task_functions.sh, queue_manager.sh
# Usage: spawned by b200_validation_suite.sh for each GPU worker
#
# Each worker runs on a dedicated GPU and continuously:
# 1. Checks available GPU memory
# 2. Finds a suitable task from the ready queue (with multi-GPU awareness)
# 3. Pre-checks OOM risk before execution
# 4. Executes the task (may use multiple GPUs for large models)
# 5. Reports completion/failure, releases GPU reservations, purges memory
# 6. Repeats until queue is empty

# Source dependencies
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -z "${SCHEDULER_LOADED:-}" ]] && source "${SCRIPT_DIR}/scheduler.sh" && export SCHEDULER_LOADED=1
[[ -z "${TASK_FUNCTIONS_LOADED:-}" ]] && source "${SCRIPT_DIR}/task_functions.sh" && export TASK_FUNCTIONS_LOADED=1
if [[ -z "${MODEL_CREATION_LOADED:-}" && -f "${SCRIPT_DIR}/model_creation.sh" ]]; then
    source "${SCRIPT_DIR}/model_creation.sh"
    export MODEL_CREATION_LOADED=1
fi
[[ -z "${TASK_SERIALIZATION_LOADED:-}" ]] && source "${SCRIPT_DIR}/task_serialization.sh" && export TASK_SERIALIZATION_LOADED=1
# Fault tolerance is optional - use subshell to handle source failure
if [[ -z "${FAULT_TOLERANCE_LOADED:-}" ]]; then
    source "${SCRIPT_DIR}/fault_tolerance.sh" 2>/dev/null || true
fi

# ============ WORKER CONFIGURATION ============

# Heartbeat interval in seconds
WORKER_HEARTBEAT_INTERVAL="${WORKER_HEARTBEAT_INTERVAL:-30}"

# Idle sleep time when no tasks available
WORKER_IDLE_SLEEP="${WORKER_IDLE_SLEEP:-5}"

# Maximum consecutive failures before worker shutdown
WORKER_MAX_FAILURES="${WORKER_MAX_FAILURES:-10}"

# ============ WORKER STATE MANAGEMENT ============

# Initialize worker
# Usage: init_worker <gpu_id> <output_dir>
init_worker() {
    local gpu_id="$1"
    local output_dir="$2"

    local workers_dir="${output_dir}/workers"
    mkdir -p "${workers_dir}"

    # Write PID file
    # IMPORTANT: Use BASHPID not $$ because $$ doesn't change in subshells
    # BASHPID gives the actual PID of the current bash process
    local worker_pid="${BASHPID:-$$}"
    echo "${worker_pid}" > "${workers_dir}/gpu_${gpu_id}.pid"

    # Initialize status
    echo "starting" > "${workers_dir}/gpu_${gpu_id}.status"

    # Initialize heartbeat
    touch "${workers_dir}/gpu_${gpu_id}.heartbeat"

    # Write worker info
    cat > "${workers_dir}/gpu_${gpu_id}.info" << EOF
{
    "gpu_id": ${gpu_id},
    "pid": ${worker_pid},
    "started_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "status": "running"
}
EOF

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Worker initialized (PID ${worker_pid})"
}

# Update worker status
# Usage: update_worker_status <gpu_id> <output_dir> <status> [task_id]
update_worker_status() {
    local gpu_id="$1"
    local output_dir="$2"
    local status="$3"
    local task_id="${4:-}"

    local status_file="${output_dir}/workers/gpu_${gpu_id}.status"

    if [[ -n "${task_id}" ]]; then
        echo "${status}:${task_id}" > "${status_file}"
    else
        echo "${status}" > "${status_file}"
    fi
}

# Update heartbeat
# Usage: update_heartbeat <gpu_id> <output_dir>
update_heartbeat() {
    local gpu_id="$1"
    local output_dir="$2"

    touch "${output_dir}/workers/gpu_${gpu_id}.heartbeat"
}

# Start background heartbeat process
# Usage: start_heartbeat_thread <gpu_id> <output_dir> <interval>
# Returns: PID of heartbeat process (store in _HEARTBEAT_PID)
start_heartbeat_thread() {
    local gpu_id="$1"
    local output_dir="$2"
    local interval="${3:-30}"

    (
        while true; do
            touch "${output_dir}/workers/gpu_${gpu_id}.heartbeat" 2>/dev/null || break
            sleep "${interval}"
        done
    ) &
    echo $!
}

# Stop background heartbeat process
# Usage: stop_heartbeat_thread <pid>
stop_heartbeat_thread() {
    local hb_pid="$1"

    if [[ -n "${hb_pid}" ]] && kill -0 "${hb_pid}" 2>/dev/null; then
        kill "${hb_pid}" 2>/dev/null || true
        wait "${hb_pid}" 2>/dev/null || true
    fi
}

# Check if shutdown requested
# Usage: should_shutdown <gpu_id> <output_dir>
should_shutdown() {
    local gpu_id="$1"
    local output_dir="$2"

    # Check for global shutdown signal
    [[ -f "${output_dir}/workers/SHUTDOWN" ]] && return 0

    # Check for worker-specific shutdown
    [[ -f "${output_dir}/workers/gpu_${gpu_id}.shutdown" ]] && return 0

    return 1
}

# Signal all workers to shutdown
# Usage: signal_shutdown <output_dir>
signal_shutdown() {
    local output_dir="$1"

    touch "${output_dir}/workers/SHUTDOWN"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Shutdown signal sent to all workers"
}

# ============ MAIN WORKER LOOP ============

# Run GPU worker loop
# Usage: gpu_worker <gpu_id> <output_dir>
gpu_worker() {
    local gpu_id="$1"
    local output_dir="$2"

    # Initialize
    init_worker "${gpu_id}" "${output_dir}"
    # Ensure GPU reservation tracking directory is initialized for this run.
    if type init_gpu_reservations &>/dev/null; then
        init_gpu_reservations "${output_dir}"
    fi

    local consecutive_failures=0
    local tasks_completed=0
    local last_heartbeat=$(date +%s)

    # Set GPU for this worker
    export CUDA_VISIBLE_DEVICES="${gpu_id}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Worker started"
    update_worker_status "${gpu_id}" "${output_dir}" "idle"

    while true; do
        # Check for shutdown signal
        if should_shutdown "${gpu_id}" "${output_dir}"; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Shutdown signal received"
            break
        fi

        # Heartbeat update
        local now=$(date +%s)
        if [[ $((now - last_heartbeat)) -ge ${WORKER_HEARTBEAT_INTERVAL} ]]; then
            update_heartbeat "${gpu_id}" "${output_dir}"
            last_heartbeat=${now}
        fi

        # NOTE: resolve_dependencies() removed from worker loop to reduce queue lock contention.
        # Dependency resolution is now centralized in the main script's monitor loop.
        # Workers only claim tasks from the ready queue; the monitor promotes pending->ready.

        # Get available GPU memory
        local available_mem=$(get_gpu_available_memory "${gpu_id}")

        if [[ -z "${available_mem}" || "${available_mem}" -le 0 ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Cannot query GPU memory, retrying..."
            sleep "${WORKER_IDLE_SLEEP}"
            continue
        fi

        # Find and claim a task
        update_worker_status "${gpu_id}" "${output_dir}" "searching"

        local task_file=$(find_and_claim_task "${available_mem}" "${gpu_id}")

        if [[ -z "${task_file}" || ! -f "${task_file}" ]]; then
            # No suitable task found

            # Check if queue is completely empty
            if is_queue_empty; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Queue empty, worker shutting down"
                break
            fi

            # Check if just pending tasks (waiting for dependencies)
            local pending=$(count_tasks "pending")
            local ready=$(count_tasks "ready")
            local running=$(count_tasks "running")

            if [[ ${ready} -eq 0 && ${pending} -gt 0 ]]; then
                # All ready tasks taken, wait for deps to resolve
                update_worker_status "${gpu_id}" "${output_dir}" "waiting_deps"
            else
                update_worker_status "${gpu_id}" "${output_dir}" "idle"
            fi

            sleep "${WORKER_IDLE_SLEEP}"
            continue
        fi

        # Got a task - execute it
        local task_id=$(get_task_id "${task_file}")
        local task_type=$(get_task_type "${task_file}")
        local model_name=$(get_task_field "${task_file}" "model_name")
        local assigned_gpus=$(get_task_assigned_gpus "${task_file}")
        [[ -z "${assigned_gpus}" || "${assigned_gpus}" == "null" ]] && assigned_gpus="${gpu_id}"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Starting task ${task_id} (${task_type}) on GPUs: ${assigned_gpus}"
        update_worker_status "${gpu_id}" "${output_dir}" "running" "${task_id}"

        # OOM Pre-check: Verify we have enough memory before running
        if type check_oom_safe &>/dev/null; then
            if ! check_oom_safe "${task_file}" "${assigned_gpus}"; then
                local risk_level="unknown"
                if type get_oom_risk_level &>/dev/null; then
                    risk_level=$(get_oom_risk_level "${task_file}" "${assigned_gpus}")
                fi
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: OOM RISK (${risk_level}) for ${task_id}, attempting memory cleanup first..."

                # Try to purge GPU memory before running
                if type purge_multi_gpu_memory &>/dev/null; then
                    purge_multi_gpu_memory "${assigned_gpus}"
                    sleep 2  # Wait for memory to be freed
                fi

                # Re-check after cleanup
                if ! check_oom_safe "${task_file}" "${assigned_gpus}"; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: WARNING - Memory still tight after cleanup, proceeding with caution"
                fi
            fi
        fi

        # Start background heartbeat thread during task execution
        # This ensures heartbeat stays fresh even during long model loads (Yi-34B: 20+ min)
        local heartbeat_pid
        heartbeat_pid=$(start_heartbeat_thread "${gpu_id}" "${output_dir}" "${WORKER_HEARTBEAT_INTERVAL}")

        # Execute task with CUDA_VISIBLE_DEVICES set to assigned GPUs
        local exit_code=0
        CUDA_VISIBLE_DEVICES="${assigned_gpus}" execute_task "${task_file}" "${gpu_id}" "${output_dir}" || exit_code=$?

        # Stop heartbeat thread now that task is complete
        stop_heartbeat_thread "${heartbeat_pid}"

        # Handle result
        if [[ ${exit_code} -eq 0 ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Task ${task_id} completed successfully"

            complete_task "${task_id}"

            # Release GPU reservations for multi-GPU tasks
            if type release_task_gpus &>/dev/null; then
                release_task_gpus "${task_id}"
            fi

            # Purge GPU memory immediately after task completion
            if type purge_multi_gpu_memory &>/dev/null; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Purging GPU memory after task completion"
                purge_multi_gpu_memory "${assigned_gpus}"
            fi

            # Dependency promotion is centralized in the main script's monitor loop
            # (resolve_dependencies) to avoid lock contention across workers.

            consecutive_failures=0
            tasks_completed=$((tasks_completed + 1))
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Task ${task_id} FAILED (exit code: ${exit_code})"

            # Extract error from task log
            local task_log="${output_dir}/logs/tasks/${task_id}.log"
            local error_msg="Exit code ${exit_code}"

            if [[ ${exit_code} -eq 124 ]]; then
                error_msg="Timeout: task exceeded limit"
            elif [[ -f "${task_log}" ]]; then
                # Check for OOM
                if grep -q "CUDA out of memory" "${task_log}" 2>/dev/null; then
                    error_msg="OOM: CUDA out of memory"
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: OOM detected - attempting aggressive memory cleanup"

                    # Aggressive memory cleanup after OOM
                    if type purge_multi_gpu_memory &>/dev/null; then
                        purge_multi_gpu_memory "${assigned_gpus}"
                    fi

                    # Try OOM recovery if available
                    if type handle_oom_task &>/dev/null; then
                        handle_oom_task "${task_file}" "${gpu_id}" "${task_log}"
                    fi
                fi
            fi

            # If CUDA context is poisoned by device-side assert, fail the task and exit
            if [[ -f "${task_log}" ]] && grep -qE "device-side assert|vectorized_gather_kernel" "${task_log}" 2>/dev/null; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: CUDA context may be poisoned, exiting for restart"
                fail_task "${task_id}" "CUDA device-side assert - context poisoned"
                if type release_task_gpus &>/dev/null; then
                    release_task_gpus "${task_id}"
                fi
                exit 1
            fi

            fail_task "${task_id}" "${error_msg}"

            # Release GPU reservations for multi-GPU tasks
            if type release_task_gpus &>/dev/null; then
                release_task_gpus "${task_id}"
            fi

            # Purge GPU memory after failure (especially important after OOM)
            if type purge_multi_gpu_memory &>/dev/null; then
                purge_multi_gpu_memory "${assigned_gpus}"
            fi

            # Check if we should retry
            if type maybe_retry_task &>/dev/null; then
                maybe_retry_task "${task_id}"
            fi

            consecutive_failures=$((consecutive_failures + 1))

            # Check for too many consecutive failures
            if [[ ${consecutive_failures} -ge ${WORKER_MAX_FAILURES} ]]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Too many consecutive failures (${consecutive_failures}), shutting down"
                break
            fi
        fi

        update_worker_status "${gpu_id}" "${output_dir}" "idle"

        # Brief pause between tasks to allow memory cleanup
        sleep 1
    done

    # Cleanup
    update_worker_status "${gpu_id}" "${output_dir}" "stopped"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Worker stopped (completed ${tasks_completed} tasks)"

    # Update worker info - read existing started_at if available
    local original_started_at
    if [[ -f "${output_dir}/workers/gpu_${gpu_id}.info" ]]; then
        original_started_at=$(jq -r '.started_at // empty' "${output_dir}/workers/gpu_${gpu_id}.info" 2>/dev/null)
    fi
    [[ -z "${original_started_at}" ]] && original_started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

    # Use BASHPID for accurate PID in subshells (same as init_worker)
    local final_pid="${BASHPID:-$$}"
    cat > "${output_dir}/workers/gpu_${gpu_id}.info" << EOF
{
    "gpu_id": ${gpu_id},
    "pid": ${final_pid},
    "started_at": "${original_started_at}",
    "stopped_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "status": "stopped",
    "tasks_completed": ${tasks_completed},
    "consecutive_failures": ${consecutive_failures}
}
EOF
}

# ============ WORKER POOL MANAGEMENT ============

# Launch all GPU workers
# Usage: launch_worker_pool <output_dir> <num_gpus>
# Returns: array of worker PIDs
launch_worker_pool() {
    local output_dir="$1"
    local num_gpus="${2:-8}"

    declare -a worker_pids=()

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching ${num_gpus} GPU workers..."

    local -a gpu_ids=()
    if [[ -n "${GPU_ID_LIST:-}" ]]; then
        IFS=',' read -ra gpu_ids <<< "${GPU_ID_LIST}"
        gpu_ids=("${gpu_ids[@]:0:${num_gpus}}")
    else
        for gpu_id in $(seq 0 $((num_gpus - 1))); do
            gpu_ids+=("${gpu_id}")
        done
    fi

    for gpu_id in "${gpu_ids[@]}"; do
        gpu_worker "${gpu_id}" "${output_dir}" &
        local pid=$!
        worker_pids+=("${pid}")

        # Write PID to file
        echo "${pid}" > "${output_dir}/workers/gpu_${gpu_id}.pid"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched worker for GPU ${gpu_id} (PID ${pid})"
    done

    # Return PIDs
    echo "${worker_pids[*]}"
}

# Wait for all workers to complete
# Usage: wait_for_workers <worker_pids...>
wait_for_workers() {
    local pids=("$@")
    local failed=0

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for ${#pids[@]} workers to complete..."

    for i in "${!pids[@]}"; do
        local pid="${pids[$i]}"

        if wait "${pid}"; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Worker ${i} (PID ${pid}) completed successfully"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Worker ${i} (PID ${pid}) FAILED"
            failed=$((failed + 1))
        fi
    done

    if [[ ${failed} -gt 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: ${failed} worker(s) failed"
        return 1
    fi

    return 0
}

# ============ WORKER MONITORING ============

# Monitor workers and restart dead ones
# Usage: monitor_workers <output_dir> <num_gpus>
monitor_workers() {
    local output_dir="$1"
    local num_gpus="$2"
    local check_interval="${3:-30}"
    # Default timeout increased from 5 min to 45 min to accommodate large model loads
    # (Yi-34B takes 20+ min to load 15 checkpoint shards)
    local worker_timeout="${4:-2700}"  # 45 minutes

    while true; do
        # Check if all work is done
        if is_queue_empty && is_queue_complete; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] All tasks complete, stopping monitor"
            signal_shutdown "${output_dir}"
            break
        fi

        # Check each worker
        local -a gpu_ids=()
        if [[ -n "${GPU_ID_LIST:-}" ]]; then
            IFS=',' read -ra gpu_ids <<< "${GPU_ID_LIST}"
            gpu_ids=("${gpu_ids[@]:0:${num_gpus}}")
        else
            for gpu_id in $(seq 0 $((num_gpus - 1))); do
                gpu_ids+=("${gpu_id}")
            done
        fi

        for gpu_id in "${gpu_ids[@]}"; do
            local pid_file="${output_dir}/workers/gpu_${gpu_id}.pid"
            local heartbeat_file="${output_dir}/workers/gpu_${gpu_id}.heartbeat"
            local status_file="${output_dir}/workers/gpu_${gpu_id}.status"

            if [[ ! -f "${pid_file}" ]]; then
                continue
            fi

            local pid=$(cat "${pid_file}")

            # Check if process is alive
            if ! kill -0 "${pid}" 2>/dev/null; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Worker GPU ${gpu_id} (PID ${pid}) died"

                # Reclaim orphaned tasks
                reclaim_orphaned_tasks "${gpu_id}"

                # Restart worker
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting worker for GPU ${gpu_id}"
                gpu_worker "${gpu_id}" "${output_dir}" &
                local new_pid=$!
                echo "${new_pid}" > "${pid_file}"

                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarted GPU ${gpu_id} worker (new PID ${new_pid})"
                continue
            fi

            # Check for stale heartbeat (stuck worker)
            if [[ -f "${heartbeat_file}" ]]; then
                local heartbeat_age
                heartbeat_age=$(( $(date +%s) - $(stat -c %Y "${heartbeat_file}") ))

                if [[ ${heartbeat_age} -gt ${worker_timeout} ]]; then
                    local status=$(cat "${status_file}" 2>/dev/null || echo "unknown")

                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Worker GPU ${gpu_id} stuck (no heartbeat for ${heartbeat_age}s, status: ${status})"

                    # Kill stuck worker
                    kill -9 "${pid}" 2>/dev/null || true

                    # Reclaim tasks
                    reclaim_orphaned_tasks "${gpu_id}"

                    # Restart
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting stuck worker for GPU ${gpu_id}"
                    gpu_worker "${gpu_id}" "${output_dir}" &
                    local new_pid=$!
                    echo "${new_pid}" > "${pid_file}"
                fi
            fi
        done

        sleep "${check_interval}"
    done
}

# Get worker summary
# Usage: get_worker_summary <output_dir> <num_gpus>
get_worker_summary() {
    local output_dir="$1"
    local num_gpus="$2"

    echo "=== WORKER STATUS ==="

    local -a gpu_ids=()
    if [[ -n "${GPU_ID_LIST:-}" ]]; then
        IFS=',' read -ra gpu_ids <<< "${GPU_ID_LIST}"
        gpu_ids=("${gpu_ids[@]:0:${num_gpus}}")
    else
        for gpu_id in $(seq 0 $((num_gpus - 1))); do
            gpu_ids+=("${gpu_id}")
        done
    fi

    for gpu_id in "${gpu_ids[@]}"; do
        local status_file="${output_dir}/workers/gpu_${gpu_id}.status"
        local pid_file="${output_dir}/workers/gpu_${gpu_id}.pid"

        local status="unknown"
        local pid="N/A"
        local alive="no"

        [[ -f "${status_file}" ]] && status=$(cat "${status_file}")
        [[ -f "${pid_file}" ]] && pid=$(cat "${pid_file}")

        if [[ "${pid}" != "N/A" ]] && kill -0 "${pid}" 2>/dev/null; then
            alive="yes"
        fi

        echo "  GPU ${gpu_id}: ${status} (PID ${pid}, alive: ${alive})"
    done
}
