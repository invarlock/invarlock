#!/usr/bin/env bash
# scheduler_selection.sh - Task priority, selection, work stealing, and metrics
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Usage: sourced by scheduler.sh

SCHEDULER_MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scheduler_core.sh
[[ -z "${SCHEDULER_CORE_LOADED:-}" ]] && source "${SCHEDULER_MODULE_DIR}/scheduler_core.sh"
# shellcheck source=scheduler_gpu_runtime.sh
[[ -z "${SCHEDULER_GPU_RUNTIME_LOADED:-}" ]] && source "${SCHEDULER_MODULE_DIR}/scheduler_gpu_runtime.sh" && export SCHEDULER_GPU_RUNTIME_LOADED=1
# shellcheck source=scheduler_reservations.sh
[[ -z "${SCHEDULER_RESERVATIONS_LOADED:-}" ]] && source "${SCHEDULER_MODULE_DIR}/scheduler_reservations.sh" && export SCHEDULER_RESERVATIONS_LOADED=1

# ============ PRIORITY CALCULATION ============

# Scheduling strategy: "small_first"

# Calculate dynamic priority for a task
# Usage: calculate_task_priority <task_file>
# Returns: priority score (0-100, higher = more urgent)
#
# Strategy: "small_first"
#   - Lower-memory tasks get higher priority to maximize early parallelism
#   - Multi-GPU scaling is based on per-task memory; adaptive under-allocation is
#     intentionally disabled by default to avoid OOM.
calculate_task_priority() {
    local task_file="$1"
    local blocked_count_override="${2:-}"
    local task_id_override="${3:-}"

    # Base priority from task file
    local base_priority=$(get_task_field "${task_file}" "priority")
    [[ "${base_priority}" =~ ^-?[0-9]+$ ]] || base_priority=50

    # Get model size for boosting
    local model_size=$(get_task_field "${task_file}" "model_size_gb")
    [[ "${model_size}" =~ ^[0-9]+$ ]] || model_size=14

    local boost=0

    # small_first strategy: small models get higher priority
    # Small models can run in parallel on all 8 GPUs
    if [[ ${model_size} -lt 30 ]]; then
        boost=$((boost + 30))  # 7B-14B: highest priority
    elif [[ ${model_size} -lt 70 ]]; then
        boost=$((boost + 20))  # 30B-40B: medium priority
    elif [[ ${model_size} -lt 100 ]]; then
        boost=$((boost + 10))  # MoE: lower priority
    fi
    # 70B+ gets no boost - runs after smaller tasks

    # Boost critical task types (SETUP must run first)
    local task_type=$(get_task_type "${task_file}")
    if [[ "${task_type}" == "SETUP_BASELINE" ]]; then
        boost=$((boost + 50))  # Always run setup first
    elif [[ "${task_type}" == "CALIBRATION_RUN" ]]; then
        boost=$((boost + 20))  # Needed before evaluate
    fi

    # Boost tasks that unblock many others.
    # NOTE: The raw blocked_count computation can be expensive if done per-candidate
    # (it scans all pending tasks). Callers should pass a precomputed value.
    local task_id="${task_id_override}"
    if [[ -z "${task_id}" ]]; then
        task_id=$(get_task_id "${task_file}")
    fi
    local blocked_count="${blocked_count_override}"
    if [[ -z "${blocked_count}" ]]; then
        blocked_count=$(count_blocked_by_task "${task_id}")
    fi
    if ! [[ "${blocked_count}" =~ ^[0-9]+$ ]]; then
        blocked_count=0
    fi
    local blocked_boost=$((blocked_count * 2))
    [[ ${blocked_boost} -gt 40 ]] && blocked_boost=40  # Cap only the unblock boost
    boost=$((boost + blocked_boost))

    # Age boost (prevent starvation)
    local created_at=$(get_task_field "${task_file}" "created_at")
    if [[ -n "${created_at}" ]]; then
        local created_epoch
        created_epoch=$(_iso_to_epoch "${created_at}")
        local now
        now=$(_now_epoch)
        local age_min=$(( (now - created_epoch) / 60 ))
        local age_boost=$((age_min / 5))
        [[ ${age_boost} -gt 10 ]] && age_boost=10
        boost=$((boost + age_boost))
    fi

    # Fairness penalty for models with many running tasks
    local model_name=$(get_task_field "${task_file}" "model_name")
    local running_for_model=$(count_running_for_model "${model_name}")
    local fairness_penalty=$((running_for_model * 3))
    [[ ${fairness_penalty} -gt 15 ]] && fairness_penalty=15

    # Final priority
    local final_priority=$((base_priority + boost - fairness_penalty))

    # Clamp to 0-100
    [[ ${final_priority} -lt 0 ]] && final_priority=0
    [[ ${final_priority} -gt 100 ]] && final_priority=100

    echo "${final_priority}"
}

# Count pending tasks blocked by completion of a given task
# Usage: count_blocked_by_task <task_id>
count_blocked_by_task() {
    local blocking_id="$1"
    local count=0
    local task_file

    for task_file in "${QUEUE_DIR}/pending"/*.task; do
        [[ -f "${task_file}" ]] || continue

        local deps=$(get_task_dependencies "${task_file}" | tr '\n' ' ')
        if [[ " ${deps} " =~ " ${blocking_id} " ]]; then
            count=$((count + 1))
        fi
    done

    echo "${count}"
}

# Count running tasks for a specific model
# Usage: count_running_for_model <model_name>
count_running_for_model() {
    local model_name="$1"
    local count=0
    local task_file

    for task_file in "${QUEUE_DIR}/running"/*.task; do
        [[ -f "${task_file}" ]] || continue

        local task_model=$(get_task_field "${task_file}" "model_name")
        if [[ "${task_model}" == "${model_name}" ]]; then
            count=$((count + 1))
        fi
    done

    echo "${count}"
}

# ============ TASK SELECTION ============

# Find the best task that fits in available memory with multi-GPU awareness
# Usage: find_best_task <available_memory_gb> <gpu_id>
# Returns: task_id of best task, or empty if none suitable
find_best_task() {
    local available_mem="$1"
    local gpu_id="$2"
    local gid
    local task_file
    [[ "${available_mem}" =~ ^[0-9]+$ ]] || available_mem=0

    # Clean up stale reservations first
    cleanup_stale_reservations

    # Check if this GPU is already reserved by another task
    if ! is_gpu_available "${gpu_id}"; then
        # This GPU is reserved for a multi-GPU task
        # Check if we're part of that task's GPU set
        local reservation_owner=""
        if [[ -n "${GPU_RESERVATION_DIR}" ]]; then
            local lock_file="${GPU_RESERVATION_DIR}/gpu_${gpu_id}.lock"
            [[ -f "${lock_file}" ]] && reservation_owner=$(cat "${lock_file}" 2>/dev/null || true)
        fi

        # If GPU is reserved by another task, skip task selection
        if [[ -n "${reservation_owner}" ]]; then
            echo ""
            return 1
        fi
    fi

    if ! is_gpu_usable "${gpu_id}"; then
        echo ""
        return 1
    fi

    # Count available GPUs for adaptive allocation decisions
    local total_available=0
    for gid in $(list_gpu_ids); do
        is_gpu_usable "${gid}" && total_available=$((total_available + 1))
    done

    # Adaptive safety margin based on available memory
    # For high-memory GPUs:
    #   - High memory (>=160GB): 2% margin (allows 157GB for 154GB 70B models)
    #   - Medium memory (>=80GB): 5% margin
    #   - Low memory (<80GB): 10% margin for safety
    local effective_mem
    if [[ ${available_mem} -ge 160 ]]; then
        # Nearly full GPU - minimal margin for large models
        effective_mem=$((available_mem * 98 / 100))
    elif [[ ${available_mem} -ge 80 ]]; then
        # Medium availability - moderate margin
        effective_mem=$((available_mem * 95 / 100))
    else
        # Low availability - conservative margin
        effective_mem=$((available_mem * 90 / 100))
    fi
    local mem_tolerance="${SCHEDULER_MEM_TOLERANCE_GB:-8}"
    if ! [[ "${mem_tolerance}" =~ ^[0-9]+$ ]]; then
        mem_tolerance=0
    fi

    local best_task=""
    local best_priority=-1
    local best_required_gpus=1
    local best_actual_gpus=1  # May differ from required if using adaptive allocation

    # Precompute "blocked by this task" counts once per scan (bash 3.2 compatible).
    # This avoids O(ready * pending) behavior from calling count_blocked_by_task
    # for every candidate task (which would scan all pending tasks each time).
    local -a blocked_ids=()
    local -a blocked_counts=()
    local pending_files=("${QUEUE_DIR}/pending"/*.task)
    if [[ -f "${pending_files[0]}" ]]; then
        while IFS=$'\t' read -r dep_id dep_count; do
            [[ -n "${dep_id}" && "${dep_count}" =~ ^[0-9]+$ ]] || continue
            blocked_ids+=("${dep_id}")
            blocked_counts+=("${dep_count}")
        done < <(
            jq -r -s '
                reduce .[] as $t ({}; reduce ($t.dependencies[]?) as $d (. ; .[$d] = (.[$d] // 0) + 1))
                | to_entries[]
                | "\(.key)\t\(.value)"
            ' "${pending_files[@]}" 2>/dev/null || true
        )
    fi

    # Scan ready queue for suitable tasks
    for task_file in "${QUEUE_DIR}/ready"/*.task; do
        [[ -f "${task_file}" ]] || continue

        # Check retry backoff delay (if is_retry_ready is available)
        if type is_retry_ready &>/dev/null; then
            if ! is_retry_ready "${task_file}"; then
                continue  # Task still in backoff period
            fi
        fi

        # Get task info
        local required_mem=$(get_task_field "${task_file}" "model_size_gb")
        [[ "${required_mem}" =~ ^[0-9]+$ ]] || required_mem=20

        # Get required GPUs (new field, default to calculated value)
        local required_gpus=$(get_task_field "${task_file}" "required_gpus")
        if [[ -z "${required_gpus}" || "${required_gpus}" == "null" ]]; then
            required_gpus=$(get_required_gpus "${required_mem}")
        fi
        [[ "${required_gpus}" =~ ^[0-9]+$ ]] || required_gpus=$(get_required_gpus "${required_mem}")
        [[ ${required_gpus} -lt 1 ]] && required_gpus=1

        # Get minimum viable GPUs for this task
        local min_gpus=$(get_minimum_gpus "${required_mem}")

        # Determine actual GPUs to use (may be less than required for adaptive allocation)
        local actual_gpus=${required_gpus}

        # For single-GPU tasks, check memory fit
        if [[ ${required_gpus} -eq 1 ]]; then
            local max_allowed=${effective_mem}
            if [[ ${required_mem} -ge 140 ]]; then
                max_allowed=$((effective_mem + mem_tolerance))
            fi
            if [[ ${required_mem} -gt ${max_allowed} ]]; then
                continue  # Task doesn't fit
            fi
        else
            # For multi-GPU tasks, check if enough GPUs are available
            local per_gpu_required=0
            if [[ "${required_mem}" =~ ^[0-9]+$ && ${required_gpus} -gt 0 ]]; then
                per_gpu_required=$(( (required_mem + required_gpus - 1) / required_gpus ))
            fi
            local available_gpu_list=$(get_available_gpus "${required_gpus}" "false" "${gpu_id}" "${per_gpu_required}")
            if [[ -z "${available_gpu_list}" ]]; then
                # Not enough GPUs for optimal allocation
                # Try adaptive allocation if no other work is available
                if should_use_adaptive_gpus "${total_available}" "${required_gpus}" "${min_gpus}"; then
                    # Use available GPUs (at least min_gpus) with reduced parallelism
                    actual_gpus=${total_available}
                    [[ ${actual_gpus} -gt ${required_gpus} ]] && actual_gpus=${required_gpus}
                    [[ ${actual_gpus} -lt ${min_gpus} ]] && continue
                    if [[ ${actual_gpus} -gt 0 ]]; then
                        per_gpu_required=$(( (required_mem + actual_gpus - 1) / actual_gpus ))
                    fi
                    available_gpu_list=$(get_available_gpus "${actual_gpus}" "false" "${gpu_id}" "${per_gpu_required}")
                    [[ -z "${available_gpu_list}" ]] && continue
                else
                    continue  # Not enough GPUs available and shouldn't adapt
                fi
            fi
        fi

        # Calculate priority
        local task_id
        task_id=$(get_task_id "${task_file}")
        local blocked_count=0
        local idx
        for idx in "${!blocked_ids[@]}"; do
            if [[ "${blocked_ids[$idx]}" == "${task_id}" ]]; then
                blocked_count="${blocked_counts[$idx]}"
                break
            fi
        done
        local priority
        priority=$(calculate_task_priority "${task_file}" "${blocked_count}" "${task_id}")

        # Slight penalty for running with fewer GPUs than optimal (OOM risk)
        if [[ ${actual_gpus} -lt ${required_gpus} ]]; then
            priority=$((priority - 5))  # Small penalty for suboptimal allocation
        fi

        if [[ ${priority} -gt ${best_priority} ]]; then
            best_priority=${priority}
            best_task="${task_id}"
            best_required_gpus=${required_gpus}
            best_actual_gpus=${actual_gpus}
        fi
    done

    echo "${best_task}"
}

# Find and claim a task atomically with multi-GPU support and adaptive allocation
# Usage: find_and_claim_task <available_memory_gb> <gpu_id>
# Returns: path to claimed task file, or empty if none
#
# IMPORTANT: ALL tasks (single and multi-GPU) reserve their GPUs to prevent
# concurrent task conflicts where multi-GPU tasks might try to use a GPU
# that already has a single-GPU task running on it.
#
# LOCK OPTIMIZATION (v2.2.1): This function uses optimistic concurrency to
# reduce lock contention. The task selection (find_best_task) is done WITHOUT
# holding the scheduler lock. Only the actual claim + reserve operation holds
# the lock, then immediately releases it. This prevents the nested lock pattern
# where scheduler lock was held while waiting for queue lock in claim_task().
find_and_claim_task() {
    local available_mem="$1"
    local gpu_id="$2"
    local gid

    local result=1
    local claimed_file=""

    # PHASE 1: Task selection WITHOUT scheduler lock (optimistic)
    # This allows multiple workers to scan in parallel
    local task_id
    task_id=$(find_best_task "${available_mem}" "${gpu_id}")

    if [[ -z "${task_id}" ]]; then
        return 1  # No suitable task found
    fi

    # Pre-compute GPU allocation info before acquiring lock
    local task_file="${QUEUE_DIR}/ready/${task_id}.task"
    if [[ ! -f "${task_file}" ]]; then
        return 1  # Task was already claimed
    fi

    local required_mem=$(get_task_field "${task_file}" "model_size_gb")
    [[ "${required_mem}" =~ ^[0-9]+$ ]] || required_mem=20

    local required_gpus=$(get_task_field "${task_file}" "required_gpus")
    if [[ -z "${required_gpus}" || "${required_gpus}" == "null" ]]; then
        required_gpus=$(get_required_gpus "${required_mem}")
    fi
    [[ "${required_gpus}" =~ ^[0-9]+$ ]] || required_gpus=$(get_required_gpus "${required_mem}")
    [[ ${required_gpus} -lt 1 ]] && required_gpus=1

    local min_gpus
    min_gpus=$(get_minimum_gpus "${required_mem}")

    # Pre-compute GPU allocation outside the scheduler lock.
    # This avoids slow nvidia-smi calls while holding the lock, which can stall all workers.
    local total_available
    total_available=$(count_available_gpus 2>/dev/null || echo "0")
    if ! [[ "${total_available}" =~ ^[0-9]+$ ]]; then
        total_available=0
    fi

    local gpu_list=""
    local actual_gpus=${required_gpus}
    if [[ ${required_gpus} -gt 1 ]]; then
        local per_gpu_required=0
        if [[ "${required_mem}" =~ ^[0-9]+$ && ${required_gpus} -gt 0 ]]; then
            per_gpu_required=$(( (required_mem + required_gpus - 1) / required_gpus ))
        fi

        gpu_list=$(get_available_gpus "${required_gpus}" "false" "${gpu_id}" "${per_gpu_required}")
        if [[ -z "${gpu_list}" ]]; then
            # Try adaptive allocation if not enough optimal GPUs.
            if should_use_adaptive_gpus "${total_available}" "${required_gpus}" "${min_gpus}"; then
                actual_gpus=${total_available}
                [[ ${actual_gpus} -gt ${required_gpus} ]] && actual_gpus=${required_gpus}
                [[ ${actual_gpus} -lt ${min_gpus} ]] && return 1
                if [[ ${actual_gpus} -gt 0 ]]; then
                    per_gpu_required=$(( (required_mem + actual_gpus - 1) / actual_gpus ))
                fi
                gpu_list=$(get_available_gpus "${actual_gpus}" "false" "${gpu_id}" "${per_gpu_required}")
                [[ -z "${gpu_list}" ]] && return 1
                echo "[ADAPTIVE] Running ${task_id} with ${actual_gpus} GPUs instead of optimal ${required_gpus}" >&2
            else
                return 1
            fi
        fi
    else
        # Single-GPU task - use the worker's assigned GPU.
        gpu_list="${gpu_id}"
        actual_gpus=1
    fi

    [[ -z "${gpu_list}" ]] && return 1

    # PHASE 2: Short-lived lock for claim + reserve
    # Only hold lock during the actual atomic operations
    local lock_timeout="${SCHEDULER_LOCK_TIMEOUT:-10}"
    if ! [[ "${lock_timeout}" =~ ^[0-9]+$ ]]; then
        lock_timeout=10
    fi
    acquire_scheduler_lock "${lock_timeout}" || return 1

    # Revalidate - task may have been claimed while we prepared
    task_file="${QUEUE_DIR}/ready/${task_id}.task"
    if [[ ! -f "${task_file}" ]]; then
        release_scheduler_lock
        return 1  # Task was claimed by another worker
    fi

    # Reserve GPUs BEFORE releasing scheduler lock
    if ! reserve_gpus "${task_id}" "${gpu_list}"; then
        if [[ "${SCHEDULER_DEBUG:-false}" == "true" ]]; then
            echo "[scheduler] reserve_gpus failed task=${task_id} gpus=${gpu_list}" >&2
        fi
        release_scheduler_lock
        return 1  # Failed to reserve GPUs (GPU already in use)
    fi

    # Release scheduler lock BEFORE calling claim_task
    # This breaks the nested lock pattern (scheduler lock → queue lock)
    release_scheduler_lock

    # PHASE 3: Claim task (uses queue lock internally, but we no longer hold scheduler lock)
    if claim_task "${task_id}" "${gpu_id}"; then
        # Update task with assigned GPUs
        local running_file="${QUEUE_DIR}/running/${task_id}.task"
        if [[ -f "${running_file}" ]]; then
            update_task_field "${running_file}" "assigned_gpus" "${gpu_list}"
            update_task_field "${running_file}" "required_gpus" "${required_gpus}" "true"
            # Track if we used adaptive allocation
            if [[ ${actual_gpus} -lt ${required_gpus} ]]; then
                update_task_field "${running_file}" "adaptive_gpus" "${actual_gpus}" "true"
            fi
        fi
        claimed_file="${running_file}"
        result=0
    else
        # Failed to claim - release reserved GPUs
        release_gpus "${task_id}"
    fi

    if [[ ${result} -eq 0 ]]; then
        echo "${claimed_file}"
    fi
    return ${result}
}

# Release task GPUs when task completes or fails
# Usage: release_task_gpus <task_id>
release_task_gpus() {
    local task_id="$1"
    release_gpus "${task_id}"
}

# ============ WORK STEALING ============

# Boost priority of tasks for models that are falling behind
# Usage: apply_work_stealing_boost
apply_work_stealing_boost() {
    # Calculate completion rate per model
    local -a models=()
    local -a model_completion=()
    local -a model_total=()
    local status
    local task_file
    local model
    local dir

    for status in completed pending ready running; do
        for task_file in "${QUEUE_DIR}/${status}"/*.task; do
            [[ -f "${task_file}" ]] || continue

            local model=$(get_task_field "${task_file}" "model_name")
            # Skip tasks with empty model names (malformed)
            [[ -z "${model}" ]] && continue
            local idx=-1
            local i
            for i in "${!models[@]}"; do
                if [[ "${models[$i]}" == "${model}" ]]; then
                    idx=$i
                    break
                fi
            done
            if [[ ${idx} -lt 0 ]]; then
                models+=("${model}")
                model_total+=(0)
                model_completion+=(0)
                idx=$((${#models[@]} - 1))
            fi
            model_total[$idx]=$((${model_total[$idx]} + 1))

            if [[ "${status}" == "completed" ]]; then
                model_completion[$idx]=$((${model_completion[$idx]} + 1))
            fi
        done
    done

    # Find average completion rate
    local total_models=0
    local total_rate=0
    local idx
    for idx in "${!models[@]}"; do
        local completed=${model_completion[$idx]}
        local total=${model_total[$idx]}
        if [[ ${total} -gt 0 ]]; then
            local rate=$((completed * 100 / total))
            total_rate=$((total_rate + rate))
            total_models=$((total_models + 1))
        fi
    done

    if [[ ${total_models} -eq 0 ]]; then
        return
    fi

    local avg_rate=$((total_rate / total_models))

    # Identify lagging models without holding the queue lock.
    # Track indices (bash 3.2 compatible; avoids associative arrays).
    local -a lagging_indices=()
    for idx in "${!models[@]}"; do
        model="${models[$idx]}"
        # Skip empty model names
        [[ -z "${model}" ]] && continue
        local completed=${model_completion[$idx]}
        local total=${model_total[$idx]}
        if [[ ${total} -gt 0 ]]; then
            local rate=$((completed * 100 / total))
            if [[ ${rate} -lt $((avg_rate - 10)) ]]; then
                lagging_indices+=("${idx}")
            fi
        fi
    done

    [[ ${#lagging_indices[@]} -eq 0 ]] && return 0

    local max_ready_updates="${WORK_STEAL_MAX_READY_UPDATES:-50}"
    if ! [[ "${max_ready_updates}" =~ ^[0-9]+$ ]]; then
        max_ready_updates=50
    fi

    # Boost pending tasks WITHOUT holding the queue lock.
    # The monitor is the only writer that moves pending->ready, so this is safe and
    # avoids blocking workers trying to claim ready tasks.
    for idx in "${lagging_indices[@]}"; do
        model="${models[$idx]}"
        local completed=${model_completion[$idx]}
        local total=${model_total[$idx]}
        local rate=0
        [[ ${total} -gt 0 ]] && rate=$((completed * 100 / total))

        echo "Boosting priority for lagging model: ${model} (${rate}% vs ${avg_rate}% avg)"

        local pending_dir="${QUEUE_DIR}/pending"
        [[ -d "${pending_dir}" ]] || continue
        for task_file in "${pending_dir}"/*.task; do
            [[ -f "${task_file}" ]] || continue

            local task_model
            task_model=$(get_task_field "${task_file}" "model_name")
            [[ "${task_model}" == "${model}" ]] || continue

            # Don't boost very large tasks; they can monopolize GPUs.
            local model_size
            model_size=$(get_task_field "${task_file}" "model_size_gb")
            if [[ -n "${model_size}" && "${model_size}" -ge 120 ]]; then
                continue
            fi

            local current_priority
            current_priority=$(get_task_field "${task_file}" "priority")
            [[ -z "${current_priority}" || "${current_priority}" == "null" ]] && current_priority=50
            if [[ ${current_priority} -ge 95 ]]; then
                continue
            fi

            local boosted=$((current_priority + 15))
            [[ ${boosted} -gt 100 ]] && boosted=100
            update_task_field "${task_file}" "priority" "${boosted}" "true"
        done
    done

    # Boost ready tasks WITH the queue lock to avoid racing claim_task() moves.
    # Limit the number of updates per cycle to keep lock hold time bounded.
    if ! acquire_queue_lock 1; then
        return 0
    fi

    local updated_ready=0
    for idx in "${lagging_indices[@]}"; do
        model="${models[$idx]}"
        [[ ${updated_ready} -ge ${max_ready_updates} ]] && break

        local ready_dir="${QUEUE_DIR}/ready"
        [[ -d "${ready_dir}" ]] || continue
        for task_file in "${ready_dir}"/*.task; do
            [[ -f "${task_file}" ]] || continue

            local task_model
            task_model=$(get_task_field "${task_file}" "model_name")
            [[ "${task_model}" == "${model}" ]] || continue

            local model_size
            model_size=$(get_task_field "${task_file}" "model_size_gb")
            if [[ -n "${model_size}" && "${model_size}" -ge 120 ]]; then
                continue
            fi

            local current_priority
            current_priority=$(get_task_field "${task_file}" "priority")
            [[ -z "${current_priority}" || "${current_priority}" == "null" ]] && current_priority=50
            if [[ ${current_priority} -ge 95 ]]; then
                continue
            fi

            local boosted=$((current_priority + 15))
            [[ ${boosted} -gt 100 ]] && boosted=100
            update_task_field "${task_file}" "priority" "${boosted}" "true"
            updated_ready=$((updated_ready + 1))
            [[ ${updated_ready} -ge ${max_ready_updates} ]] && break
        done
    done

    release_queue_lock
}

# ============ SCHEDULING METRICS ============

# Get scheduling stats for monitoring
# Usage: get_scheduling_stats
get_scheduling_stats() {
    local ready_count=$(count_tasks "ready")
    local running_count=$(count_tasks "running")
    local task_file

    # Calculate average wait time in ready queue
    local total_wait=0
    local wait_count=0
    local now
    now=$(_now_epoch)

    for task_file in "${QUEUE_DIR}/ready"/*.task; do
        [[ -f "${task_file}" ]] || continue

        local created_at=$(get_task_field "${task_file}" "created_at")
        if [[ -n "${created_at}" ]]; then
            local created_epoch
            created_epoch=$(_iso_to_epoch "${created_at}")
            [[ -z "${created_epoch}" || "${created_epoch}" -le 0 ]] && created_epoch="${now}"
            local wait=$((now - created_epoch))
            total_wait=$((total_wait + wait))
            wait_count=$((wait_count + 1))
        fi
    done

    local avg_wait=0
    [[ ${wait_count} -gt 0 ]] && avg_wait=$((total_wait / wait_count))

    # Get memory stats per GPU
    local gpu_mem_stats=""
    local gpu_id
    for gpu_id in $(list_gpu_ids); do
        local free=$(get_gpu_available_memory "${gpu_id}")
        local total=$(get_gpu_total_memory "${gpu_id}")
        local used=$((total - free))
        gpu_mem_stats="${gpu_mem_stats}GPU${gpu_id}:${used}/${total}GB "
    done

    echo "Ready: ${ready_count}, Running: ${running_count}, AvgWait: ${avg_wait}s | ${gpu_mem_stats}"
}

# Print detailed scheduling report
# Usage: print_scheduling_report
print_scheduling_report() {
    echo "=== SCHEDULING REPORT ==="
    echo ""
    local task_file

    # Queue stats
    print_queue_stats
    echo ""

    # GPU memory
    echo "=== GPU MEMORY ==="
    local gpu_id
    for gpu_id in $(list_gpu_ids); do
        local free=$(get_gpu_available_memory "${gpu_id}")
        local total=$(get_gpu_total_memory "${gpu_id}")
        local util=$(get_gpu_utilization "${gpu_id}")
        echo "GPU ${gpu_id}: ${free}/${total} GB free, ${util}% utilization"
    done
    echo ""

    # Running tasks
    echo "=== RUNNING TASKS ==="
    for task_file in "${QUEUE_DIR}/running"/*.task; do
        [[ -f "${task_file}" ]] || continue
        local task_id=$(get_task_id "${task_file}")
        local model=$(get_task_field "${task_file}" "model_name")
        local type=$(get_task_type "${task_file}")
        local gpu
        gpu="$(get_task_field "${task_file}" "assigned_gpus")"
        local size=$(get_task_field "${task_file}" "model_size_gb")
        echo "  ${task_id}: ${model}/${type} on GPU ${gpu:-unassigned} (${size}GB)"
    done
    echo ""

    # Top 5 ready tasks by priority
    echo "=== TOP READY TASKS ==="
    local count=0
    # Use safe file iteration with find instead of ls
    # Note: We limit to 5 tasks in the loop body, not with head (for portability)
    while IFS= read -r -d '' task_file; do
        [[ -f "${task_file}" ]] || continue
        local task_id=$(get_task_id "${task_file}")
        local model=$(get_task_field "${task_file}" "model_name")
        local type=$(get_task_type "${task_file}")
        local priority=$(calculate_task_priority "${task_file}")
        local size=$(get_task_field "${task_file}" "model_size_gb")
        echo "  ${task_id}: ${model}/${type} (pri=${priority}, ${size}GB)"
        count=$((count + 1))
        [[ ${count} -ge 5 ]] && break
    done < <(find "${QUEUE_DIR}/ready" -name "*.task" -type f -print0 2>/dev/null) || true
}
