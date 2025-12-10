#!/usr/bin/env bash
# gpu_worker.sh - GPU worker loop for dynamic task execution
# Version: v2.0.1 (InvarLock B200 Validation Suite)
# Dependencies: scheduler.sh, task_functions.sh, queue_manager.sh
# Usage: spawned by invarlock_definitive_validation_b200.sh for each GPU worker
#
# Each worker runs on a dedicated GPU and continuously:
# 1. Checks available GPU memory
# 2. Finds a suitable task from the ready queue
# 3. Executes the task
# 4. Reports completion/failure
# 5. Repeats until queue is empty

# Source dependencies
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -z "${SCHEDULER_LOADED:-}" ]] && source "${SCRIPT_DIR}/scheduler.sh" && export SCHEDULER_LOADED=1
[[ -z "${TASK_FUNCTIONS_LOADED:-}" ]] && source "${SCRIPT_DIR}/task_functions.sh" && export TASK_FUNCTIONS_LOADED=1
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
    echo "$$" > "${workers_dir}/gpu_${gpu_id}.pid"
    
    # Initialize status
    echo "starting" > "${workers_dir}/gpu_${gpu_id}.status"
    
    # Initialize heartbeat
    touch "${workers_dir}/gpu_${gpu_id}.heartbeat"
    
    # Write worker info
    cat > "${workers_dir}/gpu_${gpu_id}.info" << EOF
{
    "gpu_id": ${gpu_id},
    "pid": $$,
    "started_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "status": "running"
}
EOF
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Worker initialized (PID $$)"
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
        
        # Resolve any newly ready dependencies
        resolve_dependencies >/dev/null 2>&1
        
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
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Starting task ${task_id} (${task_type})"
        update_worker_status "${gpu_id}" "${output_dir}" "running" "${task_id}"
        
        # Execute task
        local exit_code=0
        execute_task "${task_file}" "${gpu_id}" "${output_dir}" || exit_code=$?
        
        # Handle result
        if [[ ${exit_code} -eq 0 ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Task ${task_id} completed successfully"
            
            complete_task "${task_id}"
            update_dependents "${task_id}"
            
            consecutive_failures=0
            tasks_completed=$((tasks_completed + 1))
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Task ${task_id} FAILED (exit code: ${exit_code})"
            
            # Extract error from task log
            local task_log="${output_dir}/logs/tasks/${task_id}.log"
            local error_msg="Exit code ${exit_code}"
            
            if [[ -f "${task_log}" ]]; then
                # Check for OOM
                if grep -q "CUDA out of memory" "${task_log}" 2>/dev/null; then
                    error_msg="OOM: CUDA out of memory"
                    
                    # Try OOM recovery if available
                    if type handle_oom_task &>/dev/null; then
                        handle_oom_task "${task_file}" "${gpu_id}" "${task_log}"
                    fi
                fi
            fi
            
            fail_task "${task_id}" "${error_msg}"
            
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
    
    cat > "${output_dir}/workers/gpu_${gpu_id}.info" << EOF
{
    "gpu_id": ${gpu_id},
    "pid": $$,
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
    
    for gpu_id in $(seq 0 $((num_gpus - 1))); do
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
    local worker_timeout="${4:-300}"  # 5 minutes
    
    while true; do
        # Check if all work is done
        if is_queue_empty && is_queue_complete; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] All tasks complete, stopping monitor"
            signal_shutdown "${output_dir}"
            break
        fi
        
        # Check each worker
        for gpu_id in $(seq 0 $((num_gpus - 1))); do
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
    
    for gpu_id in $(seq 0 $((num_gpus - 1))); do
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
