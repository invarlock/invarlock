#!/usr/bin/env bash
# scheduler_core.sh - Scheduler setup, GPU cache, and lock helpers
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Dependencies: queue_manager.sh, task_serialization.sh, nvidia-smi
# Usage: sourced by gpu_worker.sh to select tasks per GPU memory headroom
#
# Provides functions to:
# - Calculate task priorities dynamically
# - Find tasks that fit in available GPU memory
# - Implement work-stealing priority boosting
# - Multi-GPU task distribution based on per-GPU memory (profile-driven)
# - GPU reservation protection to prevent double-booking large model GPUs
# - OOM protection with pre-allocation memory checks
# - Non-sequential GPU allocation (any available GPUs, not just 0,1,2,3)
# - Adaptive under-allocation logic (disabled by default via get_minimum_gpus)

# Source dependencies
export SCHEDULER_CORE_LOADED=1
SCHEDULER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SCHEDULER_SCRIPT_DIR}"
# shellcheck source=../core/runtime.sh
source "${SCHEDULER_SCRIPT_DIR}/../core/runtime.sh"
[[ -z "${QUEUE_MANAGER_LOADED:-}" ]] && source "${SCHEDULER_SCRIPT_DIR}/queue_manager.sh" && export QUEUE_MANAGER_LOADED=1
[[ -z "${TASK_SERIALIZATION_LOADED:-}" ]] && source "${SCHEDULER_SCRIPT_DIR}/../tasks/task_serialization.sh" && export TASK_SERIALIZATION_LOADED=1

# ============ GPU POOL MANAGEMENT ============
# Track which GPUs are reserved for multi-GPU tasks

# Directory for GPU reservation files
# Preserve any exported value (e.g., set by the main script/worker init).
GPU_RESERVATION_DIR="${GPU_RESERVATION_DIR:-}"
GPU_MIN_FREE_GB="${GPU_MIN_FREE_GB:-10}"
GPU_REQUIRE_IDLE="${GPU_REQUIRE_IDLE:-true}"

# ============ GPU STATE CACHE ============
# Cache nvidia-smi results to reduce latency under scheduler lock
# TTL in seconds (default 5s - balance freshness vs performance)
GPU_CACHE_TTL="${GPU_CACHE_TTL:-5}"

# Reservation TTL in seconds (default 60s - how long a ready-queue reservation is valid)
# This handles the case where a worker dies after reserving but before claiming.
GPU_RESERVATION_TTL="${GPU_RESERVATION_TTL:-60}"
# Per-task reservation lock timeout (seconds) to serialize reservations per task.
GPU_RESERVATION_LOCK_TIMEOUT="${GPU_RESERVATION_LOCK_TIMEOUT:-5}"

# ============ GPU ID LIST HELPERS ============
# GPU_ID_LIST is the comma-separated set of *physical* GPU indices to use for this run.
# It is set/exported by the main harness. If unset, fall back to 0..NUM_GPUS-1.
list_gpu_ids() {
    if [[ -n "${GPU_ID_LIST:-}" ]]; then
        echo "${GPU_ID_LIST}" | tr -d ' ' | tr ',' '\n' | sed '/^$/d'
    else
        local total_gpus="${NUM_GPUS:-8}"
        if ! [[ "${total_gpus}" =~ ^[0-9]+$ ]]; then
            total_gpus=8
        fi
        if [[ ${total_gpus} -lt 1 ]]; then
            total_gpus=1
        fi
        seq 0 $((total_gpus - 1))
    fi
}

# Get cache file path for GPU state
# Usage: _gpu_cache_file <gpu_id>
_gpu_cache_file() {
    local gpu_id="$1"
    if [[ -n "${GPU_RESERVATION_DIR:-}" ]]; then
        echo "${GPU_RESERVATION_DIR}/.gpu_cache_${gpu_id}"
    else
        echo ""
    fi
}

# Read cached GPU state if valid (within TTL)
# Usage: _read_gpu_cache <gpu_id> <field>
# Returns: cached value or empty if cache miss/stale
_read_gpu_cache() {
    local gpu_id="$1"
    local field="$2"
    local cache_file
    cache_file=$(_gpu_cache_file "${gpu_id}")
    [[ -z "${cache_file}" || ! -f "${cache_file}" ]] && return 1

    local cache_time
    cache_time=$(_file_mtime_epoch "${cache_file}" 2>/dev/null)
    [[ -z "${cache_time}" ]] && return 1

    local ttl="${GPU_CACHE_TTL}"
    if ! [[ "${ttl}" =~ ^[0-9]+$ ]]; then
        ttl=5
    fi

    local now
    now=$(_now_epoch)
    local age=$((now - cache_time))
    if [[ ${age} -gt ${ttl} ]]; then
        return 1  # Cache expired
    fi

    # Read field from cache (format: field=value per line).
    # Under `set -euo pipefail`, `grep` returns 1 on no match which would
    # otherwise abort the caller; treat missing field as a cache miss.
    grep "^${field}=" "${cache_file}" 2>/dev/null | cut -d'=' -f2 || true
}

# Write GPU state to cache
# Usage: _write_gpu_cache <gpu_id> <free_mem> <is_idle>
_write_gpu_cache() {
    local gpu_id="$1"
    local free_mem="$2"
    local is_idle="$3"
    local cache_file
    cache_file=$(_gpu_cache_file "${gpu_id}")
    [[ -z "${cache_file}" ]] && return 0

    local tmp="${cache_file}.tmp.${BASHPID:-$$}"
    cat > "${tmp}" 2>/dev/null << EOF
free_mem=${free_mem}
is_idle=${is_idle}
EOF
    mv -f "${tmp}" "${cache_file}" 2>/dev/null || true
    rm -f "${tmp}" 2>/dev/null || true
}

# Refresh GPU cache for a single GPU (call nvidia-smi once for both values)
# Usage: _refresh_gpu_cache <gpu_id>
_refresh_gpu_cache() {
    local gpu_id="$1"

    # Query nvidia-smi once for free memory
    local free_mib
    free_mib=$(_cmd_nvidia_smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | head -1 || true)
    local free_gb=0
    if [[ "${free_mib}" =~ ^[0-9]+$ ]]; then
        free_gb=$((free_mib / 1024))
    fi

    # Query nvidia-smi once for running processes
    # Note: When no processes are running, nvidia-smi returns empty output.
    # Count only actual PID lines (numbers) to avoid empty line issues.
    local raw_output
    raw_output=$(_cmd_nvidia_smi --query-compute-apps=pid --format=csv,noheader -i "${gpu_id}" 2>/dev/null || true)
    local processes=0
    if [[ -n "${raw_output}" ]]; then
        processes=$(echo "${raw_output}" | grep -cE '^[0-9]+' 2>/dev/null || echo "0")
    fi
    local is_idle="false"
    [[ "${processes}" -eq 0 ]] && is_idle="true"

    # Write to cache
    _write_gpu_cache "${gpu_id}" "${free_gb}" "${is_idle}"
}

# Refresh GPU cache for all GPUs (batch nvidia-smi call for efficiency)
# Usage: refresh_all_gpu_cache
refresh_all_gpu_cache() {
    local gpu_id
    for gpu_id in $(list_gpu_ids); do
        _refresh_gpu_cache "${gpu_id}" 2>/dev/null || true
    done
}

# Initialize GPU reservation tracking
# Usage: init_gpu_reservations <output_dir>
init_gpu_reservations() {
    local output_dir="$1"
    GPU_RESERVATION_DIR="${output_dir}/workers/gpu_reservations"
    mkdir -p "${GPU_RESERVATION_DIR}"
    export GPU_RESERVATION_DIR

    # Pre-populate GPU cache
    refresh_all_gpu_cache 2>/dev/null || true
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
#
# IMPORTANT: Uses a mkdir-based lock which is atomic on POSIX filesystems and
# avoids file descriptor inheritance issues when workers are spawned as subshells.
#
# For subshell workers (spawned with ( ... ) &), FD-based flock approaches can
# be problematic because:
# 1. $$ doesn't change in subshells, only BASHPID does
# 2. File descriptors can be inherited/shared across subshells
# 3. shared FDs can cause contention and surprises
acquire_scheduler_lock() {
    local timeout="${1:-10}"
    local lock_file
    lock_file="$(scheduler_lock_file)"
    [[ -z "${lock_file}" ]] && return 0

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
            export SCHEDULER_LOCK_DIR="${lock_dir}"
            return 0
        fi

        # Check if we've exceeded timeout
        now=$(_now_epoch)
        if [[ ${now} -ge ${deadline} ]]; then
            local owner_pid=""
            if [[ -f "${lock_dir}/owner" ]]; then
                owner_pid=$(cat "${lock_dir}/owner" 2>/dev/null || true)
            fi
            echo "WARN: Failed to acquire scheduler lock after ${timeout}s (owner_pid=${owner_pid:-unknown})" >&2
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
            local no_owner_grace="${SCHEDULER_LOCK_NOOWNER_STALE_SECONDS:-30}"
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

# Release scheduler lock
# Usage: release_scheduler_lock
release_scheduler_lock() {
    if [[ -n "${SCHEDULER_LOCK_DIR:-}" && -d "${SCHEDULER_LOCK_DIR}" ]]; then
        # Verify we own the lock before releasing
        local my_pid="${BASHPID:-$$}"
        local owner_pid=""
        if [[ -f "${SCHEDULER_LOCK_DIR}/owner" ]]; then
            owner_pid=$(cat "${SCHEDULER_LOCK_DIR}/owner" 2>/dev/null || true)
        fi
        if [[ -z "${owner_pid}" || "${owner_pid}" == "${my_pid}" ]]; then
            rm -rf "${SCHEDULER_LOCK_DIR}" 2>/dev/null || true
        fi
        unset SCHEDULER_LOCK_DIR
    fi
}

# Get the number of GPUs required for a model size
# Usage: get_required_gpus <model_size_gb>
# Returns: integer GPU count (>=1)
#
# Evidence pack optimization: uses calculate_required_gpus from task_serialization.sh
# which accounts for per-device memory (GPU_MEMORY_GB/GPU_MEMORY_PER_DEVICE).
get_required_gpus() {
    local model_size_gb="$1"

    # Delegate to task_serialization.sh which has the per-device memory logic
    calculate_required_gpus "${model_size_gb}"
}

# Get minimum viable GPUs for a model.
# Usage: get_minimum_gpus <model_size_gb>
# Returns: minimum GPUs (defaults to required_gpus to disable under-allocation)
get_minimum_gpus() {
    local model_size_gb="$1"

    calculate_required_gpus "${model_size_gb}"
}

# Check if adaptive GPU under-allocation should be used.
# Returns 0 if we should try fewer GPUs, 1 otherwise.
#
# NOTE: With get_minimum_gpus() == get_required_gpus(), adaptive allocation is
# effectively disabled by default (prevents OOM from under-reserving).
should_use_adaptive_gpus() {
    local available_gpu_count="$1"
    local required_gpus="$2"
    local min_gpus="$3"
    local task_file

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
            [[ "${task_req}" =~ ^[0-9]+$ ]] || task_req=1
            if [[ ${task_req} -eq 1 ]]; then
                single_gpu_tasks=$((single_gpu_tasks + 1))
            fi
        done

        # Adapt if no single-GPU tasks are waiting (GPUs would be idle otherwise)
        [[ ${single_gpu_tasks} -eq 0 ]] && return 0
    fi

    return 1
}

# Task-level reservation lock (serialize reservations per task)
# Usage: _acquire_task_reservation_lock <task_id> [timeout_seconds]
SCRIPT_DIR="${SCHEDULER_SCRIPT_DIR}"
