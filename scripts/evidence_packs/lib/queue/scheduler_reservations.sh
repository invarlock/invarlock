#!/usr/bin/env bash
# scheduler_reservations.sh - GPU reservation and availability helpers
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Usage: sourced by scheduler.sh

SCHEDULER_MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scheduler_core.sh
[[ -z "${SCHEDULER_CORE_LOADED:-}" ]] && source "${SCHEDULER_MODULE_DIR}/scheduler_core.sh"
# shellcheck source=scheduler_gpu_runtime.sh
[[ -z "${SCHEDULER_GPU_RUNTIME_LOADED:-}" ]] && source "${SCHEDULER_MODULE_DIR}/scheduler_gpu_runtime.sh" && export SCHEDULER_GPU_RUNTIME_LOADED=1

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
    free_mem=$(get_gpu_available_memory "${gpu_id}" 2>/dev/null || echo "0")
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
                free_mem=$(get_gpu_available_memory "${gpu_id}" 2>/dev/null || echo "0")
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
