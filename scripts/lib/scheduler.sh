#!/usr/bin/env bash
# scheduler.sh - Memory-aware task scheduling and priority management
# Version: v2.2.0 (InvarLock B200 Validation Suite)
# Dependencies: queue_manager.sh, task_serialization.sh, nvidia-smi
# Usage: sourced by gpu_worker.sh to select tasks per GPU memory headroom
#
# Provides functions to:
# - Calculate task priorities dynamically
# - Find tasks that fit in available GPU memory
# - Implement work-stealing priority boosting
# - Multi-GPU model distribution (4 GPUs for 70B+, 2 for medium/MoE, 1 for small)
# - GPU reservation protection to prevent double-booking large model GPUs
# - OOM protection with pre-allocation memory checks
# - Non-sequential GPU allocation (any available GPUs, not just 0,1,2,3)
# - Adaptive GPU allocation when optimal GPU count unavailable

# Source dependencies
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -z "${QUEUE_MANAGER_LOADED:-}" ]] && source "${SCRIPT_DIR}/queue_manager.sh" && export QUEUE_MANAGER_LOADED=1
[[ -z "${TASK_SERIALIZATION_LOADED:-}" ]] && source "${SCRIPT_DIR}/task_serialization.sh" && export TASK_SERIALIZATION_LOADED=1

# ============ GPU POOL MANAGEMENT ============
# Track which GPUs are reserved for multi-GPU tasks

# Directory for GPU reservation files
# Preserve any exported value (e.g., set by the main script/worker init).
GPU_RESERVATION_DIR="${GPU_RESERVATION_DIR:-}"
GPU_MIN_FREE_GB="${GPU_MIN_FREE_GB:-10}"
GPU_REQUIRE_IDLE="${GPU_REQUIRE_IDLE:-true}"

# Initialize GPU reservation tracking
# Usage: init_gpu_reservations <output_dir>
init_gpu_reservations() {
    local output_dir="$1"
    GPU_RESERVATION_DIR="${output_dir}/workers/gpu_reservations"
    mkdir -p "${GPU_RESERVATION_DIR}"
    export GPU_RESERVATION_DIR
}

# ============ SCHEDULER LOCKING ============

# Get scheduler lock file path (prefer queue dir, fallback to GPU reservation dir)
# Usage: scheduler_lock_file
scheduler_lock_file() {
    if [[ -n "${QUEUE_DIR:-}" ]]; then
        echo "${QUEUE_DIR}/scheduler.lock"
        return 0
    fi
    if [[ -n "${GPU_RESERVATION_DIR:-}" ]]; then
        echo "${GPU_RESERVATION_DIR}/scheduler.lock"
        return 0
    fi
    echo ""
}

# Acquire scheduler lock (serialize task selection/reservation)
# Usage: acquire_scheduler_lock [timeout_seconds]
acquire_scheduler_lock() {
    local timeout="${1:-10}"
    local lock_file
    lock_file="$(scheduler_lock_file)"
    [[ -z "${lock_file}" ]] && return 0

    touch "${lock_file}" 2>/dev/null || true
    local fd
    exec {fd}>"${lock_file}"

    if flock -w "${timeout}" "${fd}"; then
        export SCHEDULER_LOCK_FD="${fd}"
        return 0
    fi

    exec {fd}>&-
    echo "ERROR: Failed to acquire scheduler lock after ${timeout}s" >&2
    return 1
}

# Release scheduler lock
# Usage: release_scheduler_lock
release_scheduler_lock() {
    if [[ -n "${SCHEDULER_LOCK_FD:-}" ]]; then
        exec {SCHEDULER_LOCK_FD}>&-
        unset SCHEDULER_LOCK_FD
    fi
}

# Get the number of GPUs required for a model size
# Usage: get_required_gpus <model_size_gb>
# Returns: 1, 2, or 4
get_required_gpus() {
    local model_size_gb="$1"

    # Large models (70B+): ~140GB+ → need 4 GPUs for tensor parallelism
    # Note: Even on B200 180GB, having headroom for edits/evals benefits from multi-GPU
    if [[ ${model_size_gb} -ge 120 ]]; then
        echo "4"
    # Medium models (30B-40B) and MoE (~90GB): benefit from 2 GPUs
    elif [[ ${model_size_gb} -ge 60 ]]; then
        echo "2"
    # Small models (7B-14B): single GPU is sufficient
    else
        echo "1"
    fi
}

# Get minimum viable GPUs for a model (can run with reduced parallelism)
# Usage: get_minimum_gpus <model_size_gb>
# Returns: minimum GPUs (1 for most, 2 for 70B+ on B200 180GB)
get_minimum_gpus() {
    local model_size_gb="$1"

    # 70B+ models need at least 2 GPUs (can't fit on single B200)
    if [[ ${model_size_gb} -ge 120 ]]; then
        echo "2"
    # All other models can run on 1 GPU if needed
    else
        echo "1"
    fi
}

# Check if adaptive GPU allocation should be used
# Returns 0 if we should try fewer GPUs, 1 otherwise
should_use_adaptive_gpus() {
    local available_gpu_count="$1"
    local required_gpus="$2"
    local min_gpus="$3"

    # If we have exactly what we need, no adaptation needed
    [[ ${available_gpu_count} -ge ${required_gpus} ]] && return 1

    # If we have at least minimum, and no single-GPU tasks are waiting, adapt
    if [[ ${available_gpu_count} -ge ${min_gpus} ]]; then
        # Check if there are any single-GPU tasks waiting
        local single_gpu_tasks=0
        for task_file in "${QUEUE_DIR}/ready"/*.task; do
            [[ -f "${task_file}" ]] || continue
            local task_req=$(get_task_field "${task_file}" "required_gpus")
            [[ -z "${task_req}" || "${task_req}" == "null" ]] && task_req=1
            if [[ ${task_req} -eq 1 ]]; then
                single_gpu_tasks=$((single_gpu_tasks + 1))
            fi
        done

        # Adapt if no single-GPU tasks are waiting (GPUs would be idle otherwise)
        [[ ${single_gpu_tasks} -eq 0 ]] && return 0
    fi

    return 1
}

# Get required GPUs from model size category string
# Usage: get_required_gpus_from_category <model_size_category>
# model_size_category: 7, 13, 30, 40, 70, moe
get_required_gpus_from_category() {
    local category="$1"

    case "${category}" in
        "70"|"72")
            echo "4"
            ;;
        "40"|"moe"|"30")
            echo "2"
            ;;
        *)
            echo "1"
            ;;
    esac
}

# Reserve GPUs for a task
# Usage: reserve_gpus <task_id> <gpu_list>
# gpu_list: comma-separated GPU IDs (e.g., "0,1,2,3")
# Returns: 0 on success, 1 on failure
reserve_gpus() {
    local task_id="$1"
    local gpu_list="$2"

    [[ -z "${GPU_RESERVATION_DIR}" ]] && return 1

    # Check if any GPU is already reserved
    IFS=',' read -ra gpus <<< "${gpu_list}"
    for gpu_id in "${gpus[@]}"; do
        local lock_file="${GPU_RESERVATION_DIR}/gpu_${gpu_id}.lock"
        if [[ -f "${lock_file}" ]]; then
            local existing_task=$(cat "${lock_file}" 2>/dev/null)
            # Check if the reservation is stale (task no longer running)
            if [[ -n "${existing_task}" ]]; then
                local task_file="${QUEUE_DIR}/running/${existing_task}.task"
                if [[ -f "${task_file}" ]]; then
                    # GPU is legitimately reserved
                    return 1
                fi
                # Stale reservation - clean it up
                rm -f "${lock_file}"
            fi
        fi
    done

    # Create reservations atomically
    for gpu_id in "${gpus[@]}"; do
        local lock_file="${GPU_RESERVATION_DIR}/gpu_${gpu_id}.lock"
        echo "${task_id}" > "${lock_file}"
    done

    # Record which GPUs this task is using
    echo "${gpu_list}" > "${GPU_RESERVATION_DIR}/task_${task_id}.gpus"

    return 0
}

# Release GPUs for a task
# Usage: release_gpus <task_id>
release_gpus() {
    local task_id="$1"

    [[ -z "${GPU_RESERVATION_DIR}" ]] && return 0

    local gpu_file="${GPU_RESERVATION_DIR}/task_${task_id}.gpus"
    if [[ -f "${gpu_file}" ]]; then
        local gpu_list=$(cat "${gpu_file}")
        IFS=',' read -ra gpus <<< "${gpu_list}"
        for gpu_id in "${gpus[@]}"; do
            local lock_file="${GPU_RESERVATION_DIR}/gpu_${gpu_id}.lock"
            # Only remove if we own the lock
            local owner=$(cat "${lock_file}" 2>/dev/null)
            if [[ "${owner}" == "${task_id}" ]]; then
                rm -f "${lock_file}"
            fi
        done
        rm -f "${gpu_file}"
    fi
}

# Check if a GPU is available (not reserved)
# Usage: is_gpu_available <gpu_id>
is_gpu_available() {
    local gpu_id="$1"

    [[ -z "${GPU_RESERVATION_DIR}" ]] && return 0

    local lock_file="${GPU_RESERVATION_DIR}/gpu_${gpu_id}.lock"
    if [[ -f "${lock_file}" ]]; then
        local task_id=$(cat "${lock_file}" 2>/dev/null)
        if [[ -n "${task_id}" ]]; then
            # Verify the task is still running
            local task_file="${QUEUE_DIR}/running/${task_id}.task"
            if [[ -f "${task_file}" ]]; then
                return 1  # GPU is reserved
            fi
            # Stale reservation - clean up
            rm -f "${lock_file}"
        fi
    fi
    return 0
}

# Check if a GPU is available and usable (reservation + idle/free memory).
# Usage: is_gpu_usable <gpu_id>
is_gpu_usable() {
    local gpu_id="$1"

    if ! is_gpu_available "${gpu_id}"; then
        return 1
    fi

    local free_mem
    free_mem=$(get_gpu_available_memory "${gpu_id}")
    if [[ -z "${free_mem}" || "${free_mem}" -lt ${GPU_MIN_FREE_GB} ]]; then
        return 1
    fi

    if [[ "${GPU_REQUIRE_IDLE}" == "true" ]]; then
        is_gpu_idle "${gpu_id}" || return 1
    fi

    return 0
}

# Get list of available GPUs (non-sequential - any available GPUs)
# Usage: get_available_gpus <num_gpus> [prefer_spread] [must_include]
# Returns: comma-separated list of available GPU IDs, or empty if not enough available
# Note: Does NOT require sequential GPUs - can return "0,3,5,7" instead of "0,1,2,3"
# If prefer_spread=true, tries to spread across GPU pairs for better memory bandwidth
# If must_include is set, the returned list will include that GPU or return empty.
get_available_gpus() {
    local num_needed="$1"
    local prefer_spread="${2:-false}"
    local must_include="${3:-}"
    local total_gpus="${NUM_GPUS:-8}"

    local available=()
    for gpu_id in $(seq 0 $((total_gpus - 1))); do
        if is_gpu_usable "${gpu_id}"; then
            available+=("${gpu_id}")
        fi
    done

    if [[ -n "${must_include}" ]]; then
        local found="false"
        for gpu_id in "${available[@]}"; do
            if [[ "${gpu_id}" == "${must_include}" ]]; then
                found="true"
                break
            fi
        done
        if [[ "${found}" != "true" ]]; then
            echo ""
            return 1
        fi
    fi

    if [[ ${#available[@]} -lt ${num_needed} ]]; then
        echo ""
        return 1
    fi

    # Select GPUs - prefer spread if requested (better for large models)
    local selected=()
    if [[ -n "${must_include}" ]]; then
        selected+=("${must_include}")
        for gpu_id in "${available[@]}"; do
            [[ "${gpu_id}" == "${must_include}" ]] && continue
            selected+=("${gpu_id}")
            [[ ${#selected[@]} -ge ${num_needed} ]] && break
        done
    elif [[ "${prefer_spread}" == "true" && ${num_needed} -ge 2 ]]; then
        # Try to pick GPUs that are spread out (e.g., 0,2,4,6 instead of 0,1,2,3)
        # This can improve memory bandwidth on some systems
        local step=$(( ${#available[@]} / num_needed ))
        [[ ${step} -lt 1 ]] && step=1
        local idx=0
        for i in $(seq 0 $((num_needed - 1))); do
            selected+=("${available[$idx]}")
            idx=$((idx + step))
            [[ ${idx} -ge ${#available[@]} ]] && idx=$((${#available[@]} - 1))
        done
    else
        # Just take first N available (any non-sequential GPUs that are free)
        selected=("${available[@]:0:${num_needed}}")
    fi

    if [[ ${#selected[@]} -lt ${num_needed} ]]; then
        echo ""
        return 1
    fi

    local IFS=','
    echo "${selected[*]}"
}

# Count currently available GPUs
# Usage: count_available_gpus
count_available_gpus() {
    local total_gpus="${NUM_GPUS:-8}"
    local count=0

    for gpu_id in $(seq 0 $((total_gpus - 1))); do
        is_gpu_usable "${gpu_id}" && count=$((count + 1))
    done

    echo "${count}"
}

# Get GPUs assigned to a task
# Usage: get_task_gpus <task_id>
get_task_gpus() {
    local task_id="$1"

    [[ -z "${GPU_RESERVATION_DIR}" ]] && echo "" && return 1

    local gpu_file="${GPU_RESERVATION_DIR}/task_${task_id}.gpus"
    if [[ -f "${gpu_file}" ]]; then
        cat "${gpu_file}"
    else
        echo ""
    fi
}

# Clean up stale GPU reservations (call periodically)
# Usage: cleanup_stale_reservations
cleanup_stale_reservations() {
    [[ -z "${GPU_RESERVATION_DIR}" ]] && return 0

    for lock_file in "${GPU_RESERVATION_DIR}"/gpu_*.lock; do
        [[ -f "${lock_file}" ]] || continue

        local task_id=$(cat "${lock_file}" 2>/dev/null)
        if [[ -n "${task_id}" ]]; then
            local task_file="${QUEUE_DIR}/running/${task_id}.task"
            if [[ ! -f "${task_file}" ]]; then
                # Task is no longer running - clean up
                rm -f "${lock_file}"
                rm -f "${GPU_RESERVATION_DIR}/task_${task_id}.gpus"
            fi
        fi
    done
}

# ============ OOM PROTECTION ============

# Pre-check if a task will fit in available GPU memory with safety margin
# Usage: check_oom_safe <task_file> <gpu_ids_csv>
# Returns: 0 if safe, 1 if OOM risk detected
check_oom_safe() {
    local task_file="$1"
    local gpu_ids="$2"

    if [[ ! -f "${task_file}" ]]; then
        return 1
    fi

    local model_size=$(get_task_field "${task_file}" "model_size_gb")

    [[ -z "${model_size}" ]] && model_size=20

    # Use the actually assigned GPU list, not the nominal required_gpus field.
    local assigned_gpu_count=1
    IFS=',' read -ra _assigned_gpus <<< "${gpu_ids}"
    [[ ${#_assigned_gpus[@]} -gt 0 ]] && assigned_gpu_count=${#_assigned_gpus[@]}

    # Calculate memory per GPU (distributed across GPUs)
    local mem_per_gpu=$((model_size / assigned_gpu_count))

    # model_size_gb already includes task multipliers + safety margin.
    # Apply only a small extra headroom for fragmentation.
    local required_mem=$(awk -v base="${mem_per_gpu}" 'BEGIN { printf "%.0f", base * 1.05 }')

    # Check each assigned GPU has enough memory
    IFS=',' read -ra gpus <<< "${gpu_ids}"
    for gpu_id in "${gpus[@]}"; do
        local available=$(get_gpu_available_memory "${gpu_id}")
        if [[ ${available} -lt ${required_mem} ]]; then
            echo "[OOM_CHECK] GPU ${gpu_id} has ${available}GB free but task needs ~${required_mem}GB. RISK DETECTED." >&2
            return 1
        fi
    done

    return 0
}

# Get OOM risk level for a task
# Usage: get_oom_risk_level <task_file> <available_gpus_csv>
# Returns: "low", "medium", "high", or "critical"
get_oom_risk_level() {
    local task_file="$1"
    local gpu_ids="$2"

    local model_size=$(get_task_field "${task_file}" "model_size_gb")

    [[ -z "${model_size}" ]] && model_size=20

    # Use actual assigned GPU count (may differ from required_gpus in adaptive mode).
    local assigned_gpu_count=1
    IFS=',' read -ra _assigned_gpus <<< "${gpu_ids}"
    [[ ${#_assigned_gpus[@]} -gt 0 ]] && assigned_gpu_count=${#_assigned_gpus[@]}

    # Calculate memory per GPU
    local mem_per_gpu=$((model_size / assigned_gpu_count))

    # Get minimum available memory across assigned GPUs
    local min_available=999999
    IFS=',' read -ra gpus <<< "${gpu_ids}"
    for gpu_id in "${gpus[@]}"; do
        local available=$(get_gpu_available_memory "${gpu_id}")
        [[ ${available} -lt ${min_available} ]] && min_available=${available}
    done

    # Calculate headroom percentage
    if [[ ${min_available} -le 0 ]]; then
        echo "critical"
        return
    fi
    local headroom=$(( (min_available - mem_per_gpu) * 100 / min_available ))

    if [[ ${headroom} -lt 5 ]]; then
        echo "critical"  # Less than 5% headroom - very likely OOM
    elif [[ ${headroom} -lt 15 ]]; then
        echo "high"      # 5-15% headroom - significant risk
    elif [[ ${headroom} -lt 30 ]]; then
        echo "medium"    # 15-30% headroom - some risk for memory-intensive ops
    else
        echo "low"       # >30% headroom - should be safe
    fi
}

# Clear GPU memory by running torch.cuda.empty_cache() equivalent
# Usage: purge_gpu_memory <gpu_id>
# Note: This runs a Python snippet to force CUDA memory cleanup
purge_gpu_memory() {
    local gpu_id="$1"

    # Run Python to clear CUDA cache
    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
" 2>/dev/null || true
}

# Purge memory on multiple GPUs
# Usage: purge_multi_gpu_memory <gpu_ids_csv>
purge_multi_gpu_memory() {
    local gpu_ids="$1"

    IFS=',' read -ra gpus <<< "${gpu_ids}"
    for gpu_id in "${gpus[@]}"; do
        purge_gpu_memory "${gpu_id}"
    done
}

# Kill all compute processes on the provided GPUs.
# Usage: kill_gpu_processes <gpu_ids_csv>
kill_gpu_processes() {
    local gpu_ids="$1"
    [[ -z "${gpu_ids}" ]] && return 0

    IFS=',' read -ra gpus <<< "${gpu_ids}"
    for gpu_id in "${gpus[@]}"; do
        local pids
        pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu_id}" 2>/dev/null | tr -d ' ' | awk '/^[0-9]+$/')
        [[ -z "${pids}" ]] && continue

        for pid in ${pids}; do
            kill -TERM "${pid}" 2>/dev/null || true
        done

        sleep 1

        for pid in ${pids}; do
            kill -KILL "${pid}" 2>/dev/null || true
        done
    done
}

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

# Scheduling strategy: "small_first"

# Calculate dynamic priority for a task
# Usage: calculate_task_priority <task_file>
# Returns: priority score (0-100, higher = more urgent)
#
# Strategy: "small_first" + adaptive GPU allocation
#   - Small models (7B-14B) get highest priority
#   - Maximizes parallelism when all 8 GPUs can run small tasks
#   - Large models (70B+) run later with adaptive GPU count
calculate_task_priority() {
    local task_file="$1"

    # Base priority from task file
    local base_priority=$(get_task_field "${task_file}" "priority")
    [[ -z "${base_priority}" ]] && base_priority=50

    # Get model size for boosting
    local model_size=$(get_task_field "${task_file}" "model_size_gb")
    [[ -z "${model_size}" ]] && model_size=14

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
        boost=$((boost + 20))  # Needed before certify
    fi

    # Boost tasks that unblock many others
    local task_id=$(get_task_id "${task_file}")
    local blocked_count=$(count_blocked_by_task "${task_id}")
    boost=$((boost + (blocked_count * 2)))
    [[ ${boost} -gt 40 ]] && boost=40  # Cap at +40

    # Age boost (prevent starvation)
    local created_at=$(get_task_field "${task_file}" "created_at")
    if [[ -n "${created_at}" ]]; then
        local created_epoch
        created_epoch=$(date -d "${created_at}" "+%s" 2>/dev/null || echo "0")
        local now=$(date "+%s")
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

# Find the best task that fits in available memory with multi-GPU awareness
# Usage: find_best_task <available_memory_gb> <gpu_id>
# Returns: task_id of best task, or empty if none suitable
find_best_task() {
    local available_mem="$1"
    local gpu_id="$2"

    # Clean up stale reservations first
    cleanup_stale_reservations

    # Check if this GPU is already reserved by another task
    if ! is_gpu_available "${gpu_id}"; then
        # This GPU is reserved for a multi-GPU task
        # Check if we're part of that task's GPU set
        local reservation_owner=""
        if [[ -n "${GPU_RESERVATION_DIR}" ]]; then
            local lock_file="${GPU_RESERVATION_DIR}/gpu_${gpu_id}.lock"
            [[ -f "${lock_file}" ]] && reservation_owner=$(cat "${lock_file}" 2>/dev/null)
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
    for gid in $(seq 0 $((${NUM_GPUS:-8} - 1))); do
        is_gpu_usable "${gid}" && total_available=$((total_available + 1))
    done

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
    local best_required_gpus=1
    local best_actual_gpus=1  # May differ from required if using adaptive allocation

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
        [[ -z "${required_mem}" ]] && required_mem=20

        # Get required GPUs (new field, default to calculated value)
        local required_gpus=$(get_task_field "${task_file}" "required_gpus")
        if [[ -z "${required_gpus}" || "${required_gpus}" == "null" ]]; then
            required_gpus=$(get_required_gpus "${required_mem}")
        fi

        # Get minimum viable GPUs for this task
        local min_gpus=$(get_minimum_gpus "${required_mem}")

        # Determine actual GPUs to use (may be less than required for adaptive allocation)
        local actual_gpus=${required_gpus}

        # For single-GPU tasks, check memory fit
        if [[ ${required_gpus} -eq 1 ]]; then
            if [[ ${required_mem} -gt ${effective_mem} ]]; then
                continue  # Task doesn't fit
            fi
        else
            # For multi-GPU tasks, check if enough GPUs are available
            local available_gpu_list=$(get_available_gpus "${required_gpus}" "false" "${gpu_id}")
            if [[ -z "${available_gpu_list}" ]]; then
                # Not enough GPUs for optimal allocation
                # Try adaptive allocation if no other work is available
                if should_use_adaptive_gpus "${total_available}" "${required_gpus}" "${min_gpus}"; then
                    # Use available GPUs (at least min_gpus) with reduced parallelism
                    actual_gpus=${total_available}
                    [[ ${actual_gpus} -gt ${required_gpus} ]] && actual_gpus=${required_gpus}
                    [[ ${actual_gpus} -lt ${min_gpus} ]] && continue
                    available_gpu_list=$(get_available_gpus "${actual_gpus}" "false" "${gpu_id}")
                    [[ -z "${available_gpu_list}" ]] && continue
                else
                    continue  # Not enough GPUs available and shouldn't adapt
                fi
            fi
        fi

        # Calculate priority
        local priority=$(calculate_task_priority "${task_file}")

        # Slight penalty for running with fewer GPUs than optimal (OOM risk)
        if [[ ${actual_gpus} -lt ${required_gpus} ]]; then
            priority=$((priority - 5))  # Small penalty for suboptimal allocation
        fi

        if [[ ${priority} -gt ${best_priority} ]]; then
            best_priority=${priority}
            best_task=$(get_task_id "${task_file}")
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
find_and_claim_task() {
    local available_mem="$1"
    local gpu_id="$2"

    local result=1
    local claimed_file=""

    acquire_scheduler_lock 10 || return 1

    while true; do
        # Find best task
        local task_id
        task_id=$(find_best_task "${available_mem}" "${gpu_id}")

        if [[ -z "${task_id}" ]]; then
            break  # No suitable task
        fi

        # Get the task file to check GPU requirements
        local task_file="${QUEUE_DIR}/ready/${task_id}.task"
        if [[ ! -f "${task_file}" ]]; then
            break  # Task was claimed by someone else
        fi

        # Get required GPUs
        local required_mem=$(get_task_field "${task_file}" "model_size_gb")
        [[ -z "${required_mem}" ]] && required_mem=20

        local required_gpus=$(get_task_field "${task_file}" "required_gpus")
        if [[ -z "${required_gpus}" || "${required_gpus}" == "null" ]]; then
            required_gpus=$(get_required_gpus "${required_mem}")
        fi

        # Get minimum viable GPUs for adaptive allocation
        local min_gpus
        min_gpus=$(get_minimum_gpus "${required_mem}")

        # Count available GPUs
        local total_available=0
        for gid in $(seq 0 $((${NUM_GPUS:-8} - 1))); do
            is_gpu_usable "${gid}" && total_available=$((total_available + 1))
        done

        # Determine GPU list and reserve GPUs
        # ALL tasks reserve their GPUs to prevent conflicts
        local gpu_list=""
        local actual_gpus=${required_gpus}

        if [[ ${required_gpus} -gt 1 ]]; then
            gpu_list=$(get_available_gpus "${required_gpus}" "false" "${gpu_id}")

            if [[ -z "${gpu_list}" ]]; then
                # Try adaptive allocation if not enough optimal GPUs
                if should_use_adaptive_gpus "${total_available}" "${required_gpus}" "${min_gpus}"; then
                    actual_gpus=${total_available}
                    [[ ${actual_gpus} -gt ${required_gpus} ]] && actual_gpus=${required_gpus}
                    [[ ${actual_gpus} -lt ${min_gpus} ]] && break
                    gpu_list=$(get_available_gpus "${actual_gpus}" "false" "${gpu_id}")
                    [[ -z "${gpu_list}" ]] && break

                    echo "[ADAPTIVE] Running ${task_id} with ${actual_gpus} GPUs instead of optimal ${required_gpus}" >&2
                else
                    break  # Not enough GPUs available
                fi
            fi
        else
            # Single-GPU task - use the worker's assigned GPU
            gpu_list="${gpu_id}"
            actual_gpus=1
        fi

        # Reserve GPUs for ALL tasks (both single and multi-GPU)
        # This prevents multi-GPU tasks from trying to use a GPU
        # that already has a single-GPU task running on it
        if ! reserve_gpus "${task_id}" "${gpu_list}"; then
            break  # Failed to reserve GPUs (GPU already in use)
        fi

        # Try to claim the task (atomic)
        # IMPORTANT: claim_task expects a single GPU/worker id (integer), not a CSV list.
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
            break
        fi

        # Failed to claim - release reserved GPUs
        release_gpus "${task_id}"
        break
    done

    release_scheduler_lock

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
                        local model_size=$(get_task_field "${task_file}" "model_size_gb")
                        if [[ -n "${model_size}" && "${model_size}" -ge 120 ]]; then
                            continue
                        fi
                        local current_priority=$(get_task_field "${task_file}" "priority")
                        if [[ ${current_priority} -ge 95 ]]; then
                            continue
                        fi
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
