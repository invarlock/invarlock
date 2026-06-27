#!/usr/bin/env bash
# validation_dynamic.sh - Dynamic GPU scheduling orchestration
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Usage: sourced by validation_suite.sh

# ============ MAIN - DYNAMIC GPU SCHEDULING (v2.0) ============
main_dynamic() {
    local start_time=$(date +%s)
    local suite_status=0
    local gpu_mem="${PACK_GPU_MEM_GB:-${GPU_MEMORY_GB:-}}"
    local gpu_count_label="${NUM_GPUS:-auto}"
    [[ -z "${gpu_mem}" ]] && gpu_mem="auto"
    [[ -z "${gpu_count_label}" ]] && gpu_count_label="auto"

    echo "========================================================================"
    echo "  InvarLock Evidence Pack Suite v${SCRIPT_VERSION}"
    echo "  ${gpu_mem}GB x ${gpu_count_label} GPU DYNAMIC SCHEDULING"
    echo "========================================================================"
    echo ""

    if [[ "${PACK_DEPENDENCIES_CHECKED:-0}" != "1" ]]; then
        check_dependencies
        PACK_DEPENDENCIES_CHECKED=1
        export PACK_DEPENDENCIES_CHECKED
    fi
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
    local clean_scenarios=0
    local stress_scenarios=0
    local error_scenarios=0
    local edit_scenarios_source="defaults"
    local error_scenarios_source="defaults"

    local scenarios_file="${OUTPUT_DIR}/state/scenarios.json"
    local edit_counts=""
    edit_counts="$(pack_count_edit_scenarios)"
    IFS='|' read -r clean_scenarios stress_scenarios edit_scenarios_source <<< "${edit_counts}"

    if ! [[ "${clean_scenarios}" =~ ^[0-9]+$ ]]; then
        clean_scenarios=0
    fi
    if ! [[ "${stress_scenarios}" =~ ^[0-9]+$ ]]; then
        stress_scenarios=0
    fi
    if [[ -f "${scenarios_file}" ]]; then
        error_scenarios="$(_pack_validation_state count-generation-kind "${scenarios_file}" --kind error 2>/dev/null || echo 0)"
        error_scenarios_source="state/scenarios.json"
    fi
    if ! [[ "${error_scenarios}" =~ ^[0-9]+$ ]]; then
        error_scenarios=0
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
    local edit_evaluate_clean=$((clean_scenarios * clean_runs))
    local edit_evaluate_stress=$((stress_scenarios * stress_runs))
    local edit_evaluate_total=$((edit_evaluate_clean + edit_evaluate_stress))

    log "Edit scenarios: ${clean_scenarios} clean + ${stress_scenarios} stress = ${edit_scenarios_total} per model (${edit_scenarios_source})"
    log "Edit evaluate runs: clean=${clean_scenarios}×${clean_runs}=${edit_evaluate_clean}, stress=${stress_scenarios}×${stress_runs}=${edit_evaluate_stress} (total=${edit_evaluate_total} per model)"

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

        # Move failed tasks back to pending only when explicitly requested.
        # Failed tasks often mean persistent OOM/config/dependency problems; silently
        # retrying them during a long resumed evidence run can waste GPU time.
        if [[ ${resume_existing_failed} -gt 0 ]]; then
            if [[ "${PACK_RETRY_FAILED_ON_RESUME:-0}" != "1" ]]; then
                error_exit "Resume found ${resume_existing_failed} failed task(s). Inspect or fix the failures, then set PACK_RETRY_FAILED_ON_RESUME=1 to retry them explicitly."
            fi
            log "Resetting ${resume_existing_failed} failed task(s) back to pending for explicit resume retry..."
            local task_file
            for task_file in "${QUEUE_DIR}/failed"/*.task; do
                [[ -f "${task_file}" ]] || continue
                _pack_validation_state reset-task-for-resume "${task_file}" 2>/dev/null || true
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
            set +e
            set +u
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Worker bootstrap starting"
            source_worker_lib() {
                local label="$1"
                local path="$2"
                local required_function="${3:-}"
                local source_rc=0
                source "${path}"
                source_rc=$?
                if [[ -n "${required_function}" ]] && ! declare -F "${required_function}" >/dev/null 2>&1; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: source missing ${label} function=${required_function} rc=${source_rc}"
                fi
                if [[ ${source_rc} -ne 0 ]]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: source status ${label} rc=${source_rc} after symbol validation"
                fi
            }
            # Re-source all necessary modules in the subshell context
            if [[ -f "${LIB_DIR}/tasks/task_serialization.sh" ]]; then
                source_worker_lib "task_serialization" "${LIB_DIR}/tasks/task_serialization.sh" "get_task_id"
                source_worker_lib "queue_manager" "${LIB_DIR}/queue/queue_manager.sh" "count_tasks"
                source_worker_lib "scheduler" "${LIB_DIR}/queue/scheduler.sh" "find_and_claim_task"
                source_worker_lib "task_functions" "${LIB_DIR}/tasks/task_functions.sh" "execute_task"
                source_worker_lib "gpu_worker" "${LIB_DIR}/queue/gpu_worker.sh" "gpu_worker"
                [[ -f "${LIB_DIR}/core/fault_tolerance.sh" ]] && source "${LIB_DIR}/core/fault_tolerance.sh"
            else
                source_worker_lib "task_serialization" "${LIB_DIR}/task_serialization.sh" "get_task_id"
                source_worker_lib "queue_manager" "${LIB_DIR}/queue_manager.sh" "count_tasks"
                source_worker_lib "scheduler" "${LIB_DIR}/scheduler.sh" "find_and_claim_task"
                source_worker_lib "task_functions" "${LIB_DIR}/task_functions.sh" "execute_task"
                source_worker_lib "gpu_worker" "${LIB_DIR}/gpu_worker.sh" "gpu_worker"
                [[ -f "${LIB_DIR}/fault_tolerance.sh" ]] && source "${LIB_DIR}/fault_tolerance.sh"
            fi
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Worker libraries ready"
            gpu_worker "${gpu_id}" "${OUTPUT_DIR}"
            worker_rc=$?
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Worker bootstrap exiting rc=${worker_rc}"
            exit "${worker_rc}"
        ) >> "${OUTPUT_DIR}/logs/gpu_${gpu_id}.log" 2>&1 &
        pids[${gpu_id}]=$!
        echo "${pids[${gpu_id}]}" > "${OUTPUT_DIR}/workers/gpu_${gpu_id}.pid"
    }

    update_progress() {
        local total="$1"
        local completed="$2"
        local failed="$3"
        local status="$4"
        local detail="${5:-}"

        mkdir -p "${OUTPUT_DIR}/state"
        cat > "${OUTPUT_DIR}/state/progress.json" <<EOF
{
  "total_tasks": ${total},
  "completed_tasks": ${completed},
  "failed_tasks": ${failed},
  "status": "${status}",
  "detail": "${detail}",
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    }

    # Unified monitor loop: progress + dependency resolution + worker health
    log_section "PHASE 3: MONITORING PROGRESS"
    local check_interval=60
    local worker_timeout="${WORKER_TIMEOUT:-2700}"
    local workers_started=0
    local terminal_queue_state=""
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

        local terminal_state=""
        terminal_state="$(queue_terminal_state 2>/dev/null || true)"
        if [[ "${terminal_state}" == "blocked_failed_dependencies" ]]; then
            log "Queue reached resumable terminal state: blocked_failed_dependencies"
            if type signal_shutdown &>/dev/null; then
                signal_shutdown "${OUTPUT_DIR}"
            else
                touch "${OUTPUT_DIR}/workers/SHUTDOWN"
            fi
            update_progress "${total}" "${completed}" "${failed}" "${terminal_state}" "all pending tasks are blocked on failed dependencies"
            terminal_queue_state="${terminal_state}"
            suite_status=1
            break
        fi

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
        suite_status=1
    fi

    # Check for failed tasks
    local failed_tasks=$(count_tasks "failed")
    if [[ ${failed_tasks} -gt 0 ]]; then
        log "WARNING: ${failed_tasks} task(s) failed"
        suite_status=1
        log "Failed tasks:"
        for task_file in "${QUEUE_DIR}/failed"/*.task; do
            [[ -f "${task_file}" ]] || continue
            local task_id=$(get_task_id "${task_file}")
            local error=$(get_task_field "${task_file}" "error_msg")
            log "  - ${task_id}: ${error:-unknown error}"
        done
    fi

    if [[ -n "${terminal_queue_state}" ]]; then
        log_section "BLOCKED"
        log "Queue entered resumable terminal state: ${terminal_queue_state}"
        log "Resume after fixing failed work with: OUTPUT_DIR=${OUTPUT_DIR} $0 --resume"
        return "${suite_status}"
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
    if ! python3 "${_PACK_VALIDATION_PY_DIR}/validation_state.py" \
        evaluation-optimization-summary \
        "${OUTPUT_DIR}" \
        --out "${OUTPUT_DIR}/results/analysis/evaluation_optimization_summary.json" >> "${LOG_FILE}" 2>&1; then
        log "WARNING: Failed to write evaluation optimization summary."
    fi
    local verdict_file="${OUTPUT_DIR}/reports/final_verdict.json"
    local verdict_status
    verdict_status="$(pack_read_final_verdict "${verdict_file}")"
    if [[ "${verdict_status}" == "FAIL" ]]; then
        log "ERROR: Final verdict is ${verdict_status}; suite marked failed."
        suite_status=1
    elif [[ "${verdict_status}" != "PASS" ]]; then
        log "WARNING: Final verdict status is ${verdict_status}; unable to enforce PASS gate."
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_section "COMPLETE"
    log "Total time: $((duration / 3600))h $(((duration % 3600) / 60))m $((duration % 60))s"
    log "Tasks completed: $(count_tasks "completed")/${total_tasks}"
    log "Report: ${OUTPUT_DIR}/reports/final_verdict.txt"
    log "Presets: ${OUTPUT_DIR}/presets/"
    return "${suite_status}"
}
