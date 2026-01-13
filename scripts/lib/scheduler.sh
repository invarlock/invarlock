#!/usr/bin/env bash
# scheduler.sh - Memory-aware task scheduling and priority management
# Version: v2.2.1 (InvarLock B200 Validation Suite)
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime.sh
source "${SCRIPT_DIR}/runtime.sh"
[[ -z "${QUEUE_MANAGER_LOADED:-}" ]] && source "${SCRIPT_DIR}/queue_manager.sh" && export QUEUE_MANAGER_LOADED=1
[[ -z "${TASK_SERIALIZATION_LOADED:-}" ]] && source "${SCRIPT_DIR}/task_serialization.sh" && export TASK_SERIALIZATION_LOADED=1

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
# B200 OPTIMIZATION (v2.2.1): Uses calculate_required_gpus from task_serialization.sh
# which accounts for per-device memory (defaults to 180GB on B200).
get_required_gpus() {
    local model_size_gb="$1"

    # Delegate to task_serialization.sh which has the B200-optimized logic
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

# Legacy GPU sizing by category (kept for backwards compatibility; unused by the
# current profile-based planner).
# Usage: get_required_gpus_from_category <model_size_category>
get_required_gpus_from_category() {
    local category="$1"

    case "${category}" in
        "70"|"72")
            # Conservative: reserve a single GPU to match single-GPU loading behavior.
            echo "1"
            ;;
        "moe")
            # Mixtral-8x7B needs ~90GB - fits on single B200 (180GB)
            echo "1"
            ;;
        "40"|"30")
            # 30-40B models need ~60-80GB - fit on single B200
            echo "1"
            ;;
        *)
            # 7B-14B models - single GPU
            echo "1"
            ;;
    esac
}

# Task-level reservation lock (serialize reservations per task)
# Usage: _acquire_task_reservation_lock <task_id> [timeout_seconds]
_acquire_task_reservation_lock() {
    local task_id="$1"
    local timeout="${2:-${GPU_RESERVATION_LOCK_TIMEOUT}}"
    if ! [[ "${timeout}" =~ ^[0-9]+$ ]]; then
        timeout=5
    fi
    local lock_dir="${GPU_RESERVATION_DIR}/task_${task_id}.lock.d"
    local my_pid="${BASHPID:-$$}"
    local now
    now=$(_now_epoch)
    local deadline=$((now + timeout))

    while true; do
        if mkdir "${lock_dir}" 2>/dev/null; then
            echo "${my_pid}" > "${lock_dir}/owner" 2>/dev/null || true
            return 0
        fi

        now=$(_now_epoch)
        if [[ ${now} -ge ${deadline} ]]; then
            return 1
        fi

        local owner_pid=""
        if [[ -f "${lock_dir}/owner" ]]; then
            owner_pid=$(cat "${lock_dir}/owner" 2>/dev/null || true)
        fi
        if [[ -n "${owner_pid}" ]]; then
            if ! _pid_is_alive "${owner_pid}"; then
                rm -rf "${lock_dir}" 2>/dev/null || true
                continue
            fi
        else
            local no_owner_grace="${GPU_RESERVATION_LOCK_NOOWNER_STALE_SECONDS:-30}"
            if ! [[ "${no_owner_grace}" =~ ^[0-9]+$ ]]; then
                no_owner_grace=30
            fi
            local now_epoch
            now_epoch=$(_now_epoch)
            local lock_mtime=""
            lock_mtime=$(_file_mtime_epoch "${lock_dir}" 2>/dev/null || echo "")
            if [[ -n "${lock_mtime}" ]]; then
                local lock_age=$((now_epoch - lock_mtime))
                if [[ ${lock_age} -ge ${no_owner_grace} ]]; then
                    rm -rf "${lock_dir}" 2>/dev/null || true
                    continue
                fi
            fi
        fi

        _sleep 0.05
    done
}

# Usage: _release_task_reservation_lock <task_id>
_release_task_reservation_lock() {
    local task_id="$1"
    local lock_dir="${GPU_RESERVATION_DIR}/task_${task_id}.lock.d"
    local my_pid="${BASHPID:-$$}"

    if [[ -d "${lock_dir}" ]]; then
        local owner_pid=""
        if [[ -f "${lock_dir}/owner" ]]; then
            owner_pid=$(cat "${lock_dir}/owner" 2>/dev/null || true)
        fi
        if [[ -z "${owner_pid}" || "${owner_pid}" == "${my_pid}" ]]; then
            rm -rf "${lock_dir}" 2>/dev/null || true
        fi
    fi
}

# Cleanup all reservation files for a task id.
# Usage: _cleanup_task_reservation <task_id>
_cleanup_task_reservation() {
    local task_id="$1"
    local lock_file

    [[ -z "${GPU_RESERVATION_DIR}" ]] && return 0

    for lock_file in "${GPU_RESERVATION_DIR}"/gpu_*.lock; do
        [[ -f "${lock_file}" ]] || continue
        local owner=$(cat "${lock_file}" 2>/dev/null | head -1 || true)
        if [[ "${owner}" == "${task_id}" ]]; then
            rm -f "${lock_file}"
        fi
    done
    rm -f "${GPU_RESERVATION_DIR}/task_${task_id}.gpus" "${GPU_RESERVATION_DIR}/task_${task_id}.meta"
}

# Reserve GPUs for a task
# Usage: reserve_gpus <task_id> <gpu_list>
# gpu_list: comma-separated GPU IDs (e.g., "0,1,2,3")
# Returns: 0 on success, 1 on failure
#
# RACE CONDITION PROTECTION:
# 1. Check if task is already reserved elsewhere (prevent double-reservation)
# 2. Check if any requested GPU is already reserved by a valid task
# 3. Store owner PID and timestamp for TTL-based cleanup
# 4. Use per-task lock file for atomic reservation
#
# NOTE: Reservations are considered valid if the task exists in ready OR running queue
# AND the reservation is within TTL (or running). This prevents stale reservations from
# dead workers blocking GPUs indefinitely.
reserve_gpus() {
    local task_id="$1"
    local gpu_list="$2"
    gpu_list="${gpu_list// /}"
    local my_pid="${BASHPID:-$$}"
    local gpu_id
    local -a gpus=()

    [[ -z "${GPU_RESERVATION_DIR}" ]] && return 1

    if ! _acquire_task_reservation_lock "${task_id}"; then
        return 1
    fi

    # STEP 1: Check if this task is already reserved elsewhere
    # This prevents two workers from reserving the same task on different GPUs
    if _is_reservation_valid "${task_id}"; then
        _release_task_reservation_lock "${task_id}"
        return 1
    fi
    _cleanup_task_reservation "${task_id}"

    # STEP 2: Check if any requested GPU is already reserved by a valid task
    IFS=',' read -ra gpus <<< "${gpu_list}"
    if [[ ${#gpus[@]} -eq 0 ]]; then
        _release_task_reservation_lock "${task_id}"
        return 1
    fi
    for gpu_id in "${gpus[@]}"; do
        local lock_file="${GPU_RESERVATION_DIR}/gpu_${gpu_id}.lock"
        if [[ -f "${lock_file}" ]]; then
            local existing_task=$(cat "${lock_file}" 2>/dev/null | head -1 || true)
            if [[ -n "${existing_task}" && "${existing_task}" != "${task_id}" ]]; then
                # Check if the existing reservation is valid
                if _is_reservation_valid "${existing_task}"; then
                    # GPU is reserved by a valid task
                    _release_task_reservation_lock "${task_id}"
                    return 1
                fi
                # Stale reservation - clean it up
                _cleanup_task_reservation "${existing_task}"
            fi
        fi
    done

    # STEP 3: Create reservations with metadata
    local now
    now=$(_now_epoch)

    # Write metadata file first (atomically via temp file)
    local meta_file="${GPU_RESERVATION_DIR}/task_${task_id}.meta"
    local meta_tmp="${meta_file}.${my_pid}.tmp"
    cat > "${meta_tmp}" << EOF
timestamp=${now}
owner_pid=${my_pid}
gpu_list=${gpu_list}
EOF
    mv -f "${meta_tmp}" "${meta_file}" 2>/dev/null || {
        rm -f "${meta_tmp}"
        _release_task_reservation_lock "${task_id}"
        return 1
    }

    # Write GPU lock files
    for gpu_id in "${gpus[@]}"; do
        local lock_file="${GPU_RESERVATION_DIR}/gpu_${gpu_id}.lock"
        local lock_tmp="${lock_file}.${my_pid}.tmp"
        printf '%s\n' "${task_id}" > "${lock_tmp}" 2>/dev/null \
            && mv -f "${lock_tmp}" "${lock_file}" 2>/dev/null \
            || {
                rm -f "${lock_tmp}" 2>/dev/null || true
                _cleanup_task_reservation "${task_id}"
                _release_task_reservation_lock "${task_id}"
                return 1
            }
        rm -f "${lock_tmp}" 2>/dev/null || true
    done

    # Write GPU list file
    local gpus_file="${GPU_RESERVATION_DIR}/task_${task_id}.gpus"
    local gpus_tmp="${gpus_file}.${my_pid}.tmp"
    printf '%s\n' "${gpu_list}" > "${gpus_tmp}" 2>/dev/null \
        && mv -f "${gpus_tmp}" "${gpus_file}" 2>/dev/null \
        || {
            rm -f "${gpus_tmp}" 2>/dev/null || true
            _cleanup_task_reservation "${task_id}"
            _release_task_reservation_lock "${task_id}"
            return 1
        }
    rm -f "${gpus_tmp}" 2>/dev/null || true

    _release_task_reservation_lock "${task_id}"
    return 0
}

# Helper: Check if a reservation is valid
# Usage: _is_reservation_valid <task_id>
# Returns: 0 if valid, 1 if stale
_is_reservation_valid() {
    local task_id="$1"

    # Check if task is in running queue - always valid
    local running_file="${QUEUE_DIR}/running/${task_id}.task"
    if [[ -f "${running_file}" ]]; then
        return 0
    fi

    # Check if task is in ready queue
    local ready_file="${QUEUE_DIR}/ready/${task_id}.task"
    if [[ ! -f "${ready_file}" ]]; then
        return 1  # Task not in ready or running - stale
    fi

    # Task is in ready queue - check metadata for TTL
    local meta_file="${GPU_RESERVATION_DIR}/task_${task_id}.meta"
    local res_time=""
    local res_pid=""
    if [[ -f "${meta_file}" ]]; then
        res_time=$(grep "^timestamp=" "${meta_file}" 2>/dev/null | cut -d'=' -f2 || true)
        res_pid=$(grep "^owner_pid=" "${meta_file}" 2>/dev/null | cut -d'=' -f2 || true)
    else
        # Fallback to file mtime if metadata is missing
        local gpus_file="${GPU_RESERVATION_DIR}/task_${task_id}.gpus"
        if [[ -f "${gpus_file}" ]]; then
            res_time=$(_file_mtime_epoch "${gpus_file}" 2>/dev/null)
        fi
    fi

    [[ -z "${res_time}" ]] && return 1

    local now
    now=$(_now_epoch)
    local ttl="${GPU_RESERVATION_TTL}"
    if ! [[ "${ttl}" =~ ^[0-9]+$ ]]; then
        ttl=60
    fi
    local age=$((now - ${res_time:-0}))
    if [[ ${age} -ge ${ttl} ]]; then
        return 1
    fi

    if [[ -n "${res_pid}" ]]; then
        _pid_is_alive "${res_pid}" || return 1
    fi

    return 0
}

# Release GPUs for a task
# Usage: release_gpus <task_id>
#
# NOTE: This function only releases GPU locks owned by the specified task.
# It also cleans up the metadata file created during reservation.
release_gpus() {
    local task_id="$1"

    [[ -z "${GPU_RESERVATION_DIR}" ]] && return 0

    _cleanup_task_reservation "${task_id}"
}

# Check if a GPU is available (not reserved)
# Usage: is_gpu_available <gpu_id>
#
# NOTE: Uses _is_reservation_valid which checks:
# 1. Task is in ready OR running queue
# 2. For ready-queue tasks: reservation is within TTL and owner is alive
is_gpu_available() {
    local gpu_id="$1"

    [[ -z "${GPU_RESERVATION_DIR}" ]] && return 0

    local lock_file="${GPU_RESERVATION_DIR}/gpu_${gpu_id}.lock"
    if [[ -f "${lock_file}" ]]; then
        local task_id=$(cat "${lock_file}" 2>/dev/null | head -1 || true)
        if [[ -n "${task_id}" ]]; then
            # Check if reservation is still valid (uses TTL + owner PID for ready-queue tasks)
            if _is_reservation_valid "${task_id}"; then
                return 1  # GPU is reserved by a valid task
            fi
            # Stale reservation - clean up all related files
            _cleanup_task_reservation "${task_id}"
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

    local min_free="${GPU_MIN_FREE_GB}"
    if ! [[ "${min_free}" =~ ^[0-9]+$ ]]; then
        min_free=10
    fi

    # Relax gates for single-GPU single-model runs
    if [[ "${NUM_GPUS:-}" == "1" ]] || [[ "${GPU_ID_LIST:-}" =~ ^0$ ]]; then
        min_free=0
        GPU_REQUIRE_IDLE="false"
    fi

    local free_mem
    free_mem=$(get_gpu_available_memory "${gpu_id}")
    if ! [[ "${free_mem}" =~ ^[0-9]+$ ]] || [[ "${free_mem}" -lt ${min_free} ]]; then
        return 1
    fi

    if [[ "${GPU_REQUIRE_IDLE}" == "true" ]]; then
        is_gpu_idle "${gpu_id}" || return 1
    fi

    return 0
}

# Get list of available GPUs (non-sequential - any available GPUs)
# Usage: get_available_gpus <num_gpus> [prefer_spread] [must_include] [min_free_gb]
# Returns: comma-separated list of available GPU IDs, or empty if not enough available
# Note: Does NOT require sequential GPUs - can return "0,3,5,7" instead of "0,1,2,3"
# If prefer_spread=true, tries to spread across GPU pairs for better memory bandwidth
# If must_include is set, the returned list will include that GPU or return empty.
get_available_gpus() {
    local num_needed="$1"
    [[ "${num_needed}" =~ ^[0-9]+$ ]] || num_needed=1
    [[ ${num_needed} -lt 1 ]] && num_needed=1
    local prefer_spread="${2:-false}"
    local must_include="${3:-}"
    local min_free_gb="${4:-}"
    local gpu_id

    local -a available=()
    for gpu_id in $(list_gpu_ids); do
        if is_gpu_usable "${gpu_id}"; then
            if [[ -n "${min_free_gb}" && "${min_free_gb}" =~ ^[0-9]+$ ]]; then
                local free_mem
                free_mem=$(get_gpu_available_memory "${gpu_id}")
                [[ -n "${free_mem}" && "${free_mem}" -lt ${min_free_gb} ]] && continue
            fi
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
    local -a selected=()
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
    local count=0
    local gpu_id

    for gpu_id in $(list_gpu_ids); do
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
        cat "${gpu_file}" 2>/dev/null || true
    else
        echo ""
    fi
}

# Clean up stale GPU reservations (call periodically)
# Usage: cleanup_stale_reservations
#
# Uses _is_reservation_valid which checks:
# 1. Task is in ready OR running queue
# 2. For ready-queue tasks: reservation is within TTL and owner PID is alive
cleanup_stale_reservations() {
    [[ -z "${GPU_RESERVATION_DIR}" ]] && return 0
    local lock_file

    for lock_file in "${GPU_RESERVATION_DIR}"/gpu_*.lock; do
        [[ -f "${lock_file}" ]] || continue

        local task_id=$(cat "${lock_file}" 2>/dev/null | head -1 || true)
        if [[ -n "${task_id}" ]]; then
            # Check if reservation is still valid (uses TTL + owner PID for ready-queue tasks)
            if _is_reservation_valid "${task_id}"; then
                continue  # Valid reservation
            fi
            # Stale reservation - clean up all related files
            _cleanup_task_reservation "${task_id}"
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
    gpu_ids="${gpu_ids// /}"
    local gpu_id
    local -a gpus=()

    if [[ ! -f "${task_file}" ]]; then
        return 1
    fi

    local model_size=$(get_task_field "${task_file}" "model_size_gb")

    [[ -z "${model_size}" ]] && model_size=20

    local gpu_count=1
    IFS=',' read -ra gpus <<< "${gpu_ids}"
    gpu_count=${#gpus[@]}
    [[ ${gpu_count} -lt 1 ]] && gpu_count=1

    # For multi-GPU tasks, assume sharded load across assigned GPUs.
    local mem_per_gpu=$(( (model_size + gpu_count - 1) / gpu_count ))

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
    gpu_ids="${gpu_ids// /}"
    local gpu_id
    local -a gpus=()

    local model_size=$(get_task_field "${task_file}" "model_size_gb")

    [[ -z "${model_size}" ]] && model_size=20

    local gpu_count=1
    IFS=',' read -ra gpus <<< "${gpu_ids}"
    gpu_count=${#gpus[@]}
    [[ ${gpu_count} -lt 1 ]] && gpu_count=1

    # For multi-GPU tasks, assume sharded load across assigned GPUs.
    local mem_per_gpu=$(( (model_size + gpu_count - 1) / gpu_count ))

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
    CUDA_VISIBLE_DEVICES="${gpu_id}" _cmd_python -c "
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
    gpu_ids="${gpu_ids// /}"
    local gpu_id
    local -a gpus=()

    IFS=',' read -ra gpus <<< "${gpu_ids}"
    for gpu_id in "${gpus[@]}"; do
        purge_gpu_memory "${gpu_id}"
    done
}

# Kill all compute processes on the provided GPUs.
# Usage: kill_gpu_processes <gpu_ids_csv>
kill_gpu_processes() {
    local gpu_ids="$1"
    gpu_ids="${gpu_ids// /}"
    [[ -z "${gpu_ids}" ]] && return 0
    local gpu_id
    local -a gpus=()

    IFS=',' read -ra gpus <<< "${gpu_ids}"
    for gpu_id in "${gpus[@]}"; do
        local pids
        pids=$(_cmd_nvidia_smi --query-compute-apps=pid --format=csv,noheader -i "${gpu_id}" 2>/dev/null | tr -d ' ' | awk '/^[0-9]+$/' || true)
        [[ -z "${pids}" ]] && continue

        for pid in ${pids}; do
            _cmd_kill -TERM "${pid}" 2>/dev/null || true
        done

        _sleep 1

        for pid in ${pids}; do
            _cmd_kill -KILL "${pid}" 2>/dev/null || true
        done
    done
}

# ============ GPU MEMORY MANAGEMENT ============

# Get available GPU memory in GB (with caching to reduce nvidia-smi calls)
# Usage: get_gpu_available_memory <gpu_id>
get_gpu_available_memory() {
    local gpu_id="$1"

    # Try to read from cache first
    local cached_val
    cached_val=$(_read_gpu_cache "${gpu_id}" "free_mem")
    if [[ -n "${cached_val}" && "${cached_val}" =~ ^[0-9]+$ ]]; then
        echo "${cached_val}"
        return 0
    fi

    local nsmi_timeout="${GPU_NSMI_TIMEOUT:-2}"
    [[ -z "${nsmi_timeout}" || ! "${nsmi_timeout}" =~ ^[0-9]+$ ]] && nsmi_timeout=2

    # Cache miss - query nvidia-smi for free memory in MiB, but fall back quickly
    local free_mib
    free_mib=$(timeout "${nsmi_timeout}" _cmd_nvidia_smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | head -1 || true)

    if ! [[ "${free_mib}" =~ ^[0-9]+$ ]]; then
        echo "180"
        return 0
    fi

    # Convert MiB to GB (1024 MiB = 1 GiB ≈ 1.07 GB)
    local free_gb=$((free_mib / 1024))

    # Update cache (also refresh idle status since we're querying)
    # Count only actual PID lines (numbers) to avoid empty line issues
    local raw_output
    raw_output=$(timeout "${nsmi_timeout}" _cmd_nvidia_smi --query-compute-apps=pid --format=csv,noheader -i "${gpu_id}" 2>/dev/null || true)
    local processes=0
    if [[ -n "${raw_output}" ]]; then
        processes=$(echo "${raw_output}" | grep -cE '^[0-9]+' 2>/dev/null || echo "0")
    fi
    local is_idle="false"
    [[ "${processes}" -eq 0 ]] && is_idle="true"
    _write_gpu_cache "${gpu_id}" "${free_gb}" "${is_idle}"

    echo "${free_gb}"
}

# Get total GPU memory in GB
# Usage: get_gpu_total_memory <gpu_id>
get_gpu_total_memory() {
    local gpu_id="$1"

    local total_mib
    total_mib=$(_cmd_nvidia_smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | head -1 || true)

    if ! [[ "${total_mib}" =~ ^[0-9]+$ ]]; then
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

    _cmd_nvidia_smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | head -1 || echo "0"
}

# Check if GPU is idle (no processes) - with caching to reduce nvidia-smi calls
# Usage: is_gpu_idle <gpu_id>
is_gpu_idle() {
    local gpu_id="$1"

    # Try to read from cache first
    local cached_idle
    cached_idle=$(_read_gpu_cache "${gpu_id}" "is_idle")
    if [[ -n "${cached_idle}" ]]; then
        [[ "${cached_idle}" == "true" ]]
        return $?
    fi

    # Cache miss - query nvidia-smi
    # Note: When no processes are running, nvidia-smi returns empty output.
    # We use grep -c to count actual PID lines (numbers only) to avoid
    # counting empty lines or error messages.
    local raw_output
    raw_output=$(_cmd_nvidia_smi --query-compute-apps=pid --format=csv,noheader -i "${gpu_id}" 2>/dev/null || true)

    local processes=0
    if [[ -n "${raw_output}" ]]; then
        # Count only lines that contain actual PIDs (numeric values)
        processes=$(echo "${raw_output}" | grep -cE '^[0-9]+' 2>/dev/null || echo "0")
    fi

    local is_idle="false"
    [[ "${processes}" -eq 0 ]] && is_idle="true"

    # Update cache (also refresh free memory since we're querying)
    local free_mib
    free_mib=$(_cmd_nvidia_smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | head -1 || true)
    local free_gb=0
    if [[ "${free_mib}" =~ ^[0-9]+$ ]]; then
        free_gb=$((free_mib / 1024))
    fi
    _write_gpu_cache "${gpu_id}" "${free_gb}" "${is_idle}"

    [[ "${is_idle}" == "true" ]]
}

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
        boost=$((boost + 20))  # Needed before certify
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
    done < <(find "${QUEUE_DIR}/ready" -name "*.task" -type f -print0 2>/dev/null) || true
}
