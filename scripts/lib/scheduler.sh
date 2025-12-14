#!/usr/bin/env bash
# scheduler.sh - Memory-aware task scheduling and priority management
# Version: v2.0.1 (InvarLock B200 Validation Suite)
# Dependencies: queue_manager.sh, nvidia-smi
# Usage: sourced by gpu_worker.sh to select tasks per GPU memory headroom
#
# Provides functions to:
# - Calculate task priorities dynamically
# - Find tasks that fit in available GPU memory
# - Implement work-stealing priority boosting

# Source dependencies
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -z "${QUEUE_MANAGER_LOADED:-}" ]] && source "${SCRIPT_DIR}/queue_manager.sh" && export QUEUE_MANAGER_LOADED=1

# ============ GPU MEMORY MANAGEMENT ============

# Get available GPU memory in GB
# Usage: get_gpu_available_memory <gpu_id>
get_gpu_available_memory() {
    local gpu_id="$1"

    # Query nvidia-smi for free memory in MiB
    local free_mib
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | head -1)

    if [[ -z "${free_mib}" ]]; then
        echo "0"
        return 1
    fi

    # Convert MiB to GB (1024 MiB = 1 GiB ≈ 1.07 GB)
    local free_gb=$((free_mib / 1024))
    echo "${free_gb}"
}

# Get total GPU memory in GB
# Usage: get_gpu_total_memory <gpu_id>
get_gpu_total_memory() {
    local gpu_id="$1"

    local total_mib
    total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | head -1)

    if [[ -z "${total_mib}" ]]; then
        echo "180"  # Default to B200 size
        return 1
    fi

    local total_gb=$((total_mib / 1024))
    echo "${total_gb}"
}

# Get GPU utilization percentage
# Usage: get_gpu_utilization <gpu_id>
get_gpu_utilization() {
    local gpu_id="$1"

    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | head -1 || echo "0"
}

# Check if GPU is idle (no processes)
# Usage: is_gpu_idle <gpu_id>
is_gpu_idle() {
    local gpu_id="$1"

    local processes
    processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu_id}" 2>/dev/null | wc -l | tr -d ' ')

    [[ "${processes}" -eq 0 ]]
}

# ============ PRIORITY CALCULATION ============

# Calculate dynamic priority for a task
# Usage: calculate_task_priority <task_file>
# Returns: priority score (0-100, higher = more urgent)
calculate_task_priority() {
    local task_file="$1"

    # Base priority from task file
    local base_priority=$(get_task_field "${task_file}" "priority")
    [[ -z "${base_priority}" ]] && base_priority=50

    # Get model size for boosting
    local model_size=$(get_task_field "${task_file}" "model_size_gb")
    [[ -z "${model_size}" ]] && model_size=14

    local boost=0

    # Boost 1: Large models get priority (they're the bottleneck)
    # 70B+ models: +20 priority
    # 40B+ models: +10 priority
    # 30B+ models: +5 priority
    if [[ ${model_size} -ge 120 ]]; then
        boost=$((boost + 20))
    elif [[ ${model_size} -ge 70 ]]; then
        boost=$((boost + 10))
    elif [[ ${model_size} -ge 50 ]]; then
        boost=$((boost + 5))
    fi

    # Boost 2: Tasks blocking many others get priority
    local task_id=$(get_task_id "${task_file}")
    local blocked_count=$(count_blocked_by_task "${task_id}")
    boost=$((boost + (blocked_count * 2)))
    [[ ${boost} -gt 40 ]] && boost=40  # Cap at +40

    # Boost 3: Older tasks get slight priority (prevent starvation)
    local created_at=$(get_task_field "${task_file}" "created_at")
    if [[ -n "${created_at}" ]]; then
        # Convert to epoch and calculate age in minutes
        local created_epoch
        created_epoch=$(date -d "${created_at}" "+%s" 2>/dev/null || echo "0")
        local now=$(date "+%s")
        local age_min=$(( (now - created_epoch) / 60 ))
        # +1 priority per 5 minutes waiting
        local age_boost=$((age_min / 5))
        [[ ${age_boost} -gt 10 ]] && age_boost=10  # Cap at +10
        boost=$((boost + age_boost))
    fi

    # Reduce priority for models with many running tasks (fairness)
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

# Find the best task that fits in available memory
# Usage: find_best_task <available_memory_gb> <gpu_id>
# Returns: task_id of best task, or empty if none suitable
find_best_task() {
    local available_mem="$1"
    local gpu_id="$2"

    # Adaptive safety margin based on available memory
    # For B200 (180GB GPUs):
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

    local best_task=""
    local best_priority=-1

    # Scan ready queue for suitable tasks
    for task_file in "${QUEUE_DIR}/ready"/*.task; do
        [[ -f "${task_file}" ]] || continue

        # Check retry backoff delay (if is_retry_ready is available)
        if type is_retry_ready &>/dev/null; then
            if ! is_retry_ready "${task_file}"; then
                continue  # Task still in backoff period
            fi
        fi

        # Check memory requirement
        local required_mem=$(get_task_field "${task_file}" "model_size_gb")
        [[ -z "${required_mem}" ]] && required_mem=20

        if [[ ${required_mem} -gt ${effective_mem} ]]; then
            continue  # Task doesn't fit
        fi

        # Calculate priority
        local priority=$(calculate_task_priority "${task_file}")

        if [[ ${priority} -gt ${best_priority} ]]; then
            best_priority=${priority}
            best_task=$(get_task_id "${task_file}")
        fi
    done

    echo "${best_task}"
}

# Find and claim a task atomically
# Usage: find_and_claim_task <available_memory_gb> <gpu_id>
# Returns: path to claimed task file, or empty if none
find_and_claim_task() {
    local available_mem="$1"
    local gpu_id="$2"

    # Find best task
    local task_id=$(find_best_task "${available_mem}" "${gpu_id}")

    if [[ -z "${task_id}" ]]; then
        return 1  # No suitable task
    fi

    # Try to claim it (atomic)
    if claim_task "${task_id}" "${gpu_id}"; then
        echo "${QUEUE_DIR}/running/${task_id}.task"
        return 0
    fi

    # Someone else claimed it, try again
    return 1
}

# ============ WORK STEALING ============

# Boost priority of tasks for models that are falling behind
# Usage: apply_work_stealing_boost
apply_work_stealing_boost() {
    # Calculate completion rate per model
    declare -A model_completion
    declare -A model_total

    for status in completed pending ready running; do
        for task_file in "${QUEUE_DIR}/${status}"/*.task; do
            [[ -f "${task_file}" ]] || continue

            local model=$(get_task_field "${task_file}" "model_name")
            # Skip tasks with empty model names (malformed)
            [[ -z "${model}" ]] && continue
            model_total[${model}]=$((${model_total[${model}]:-0} + 1))

            if [[ "${status}" == "completed" ]]; then
                model_completion[${model}]=$((${model_completion[${model}]:-0} + 1))
            fi
        done
    done

    # Find average completion rate
    local total_models=0
    local total_rate=0
    for model in "${!model_total[@]}"; do
        local completed=${model_completion[${model}]:-0}
        local total=${model_total[${model}]}
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

    # Boost models falling behind average
    for model in "${!model_total[@]}"; do
        # Skip empty model names
        [[ -z "${model}" ]] && continue
        local completed=${model_completion[${model}]:-0}
        local total=${model_total[${model}]}
        if [[ ${total} -gt 0 ]]; then
            local rate=$((completed * 100 / total))

            if [[ ${rate} -lt $((avg_rate - 10)) ]]; then
                # Model is falling behind, boost its pending/ready tasks
                echo "Boosting priority for lagging model: ${model} (${rate}% vs ${avg_rate}% avg)"

                # Iterate directories separately to handle missing directories gracefully
                for dir in "${QUEUE_DIR}/pending" "${QUEUE_DIR}/ready"; do
                    [[ -d "${dir}" ]] || continue
                    for task_file in "${dir}"/*.task; do
                        [[ -f "${task_file}" ]] || continue

                    local task_model=$(get_task_field "${task_file}" "model_name")
                    if [[ "${task_model}" == "${model}" ]]; then
                        local current_priority=$(get_task_field "${task_file}" "priority")
                        local boosted=$((current_priority + 15))
                        [[ ${boosted} -gt 100 ]] && boosted=100
                        update_task_field "${task_file}" "priority" "${boosted}" "true"
                    fi
                done
            done
        fi
    fi
done
}

# ============ SCHEDULING METRICS ============

# Get scheduling stats for monitoring
# Usage: get_scheduling_stats
get_scheduling_stats() {
    local ready_count=$(count_tasks "ready")
    local running_count=$(count_tasks "running")

    # Calculate average wait time in ready queue
    local total_wait=0
    local wait_count=0
    local now=$(date "+%s")

    for task_file in "${QUEUE_DIR}/ready"/*.task; do
        [[ -f "${task_file}" ]] || continue

        local created_at=$(get_task_field "${task_file}" "created_at")
        if [[ -n "${created_at}" ]]; then
            local created_epoch
            created_epoch=$(date -d "${created_at}" "+%s" 2>/dev/null || echo "${now}")
            local wait=$((now - created_epoch))
            total_wait=$((total_wait + wait))
            wait_count=$((wait_count + 1))
        fi
    done

    local avg_wait=0
    [[ ${wait_count} -gt 0 ]] && avg_wait=$((total_wait / wait_count))

    # Get memory stats per GPU
    local gpu_mem_stats=""
    for gpu_id in $(seq 0 $((${NUM_GPUS:-8} - 1))); do
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

    # Queue stats
    print_queue_stats
    echo ""

    # GPU memory
    echo "=== GPU MEMORY ==="
    for gpu_id in $(seq 0 $((${NUM_GPUS:-8} - 1))); do
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
        local gpu=$(get_task_field "${task_file}" "gpu_id")
        local size=$(get_task_field "${task_file}" "model_size_gb")
        echo "  ${task_id}: ${model}/${type} on GPU ${gpu} (${size}GB)"
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
    done < <(find "${QUEUE_DIR}/ready" -name "*.task" -type f -print0 2>/dev/null)
}
