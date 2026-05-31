#!/usr/bin/env bash
# queue_lifecycle.sh - Task state transitions and orphan reclamation
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Usage: sourced by queue_manager.sh

QUEUE_MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=queue_core.sh
[[ -z "${QUEUE_CORE_LOADED:-}" ]] && source "${QUEUE_MODULE_DIR}/queue_core.sh"

# ============ TASK STATE TRANSITIONS ============

# Move task from pending to ready (when dependencies are satisfied)
# Usage: mark_task_ready <task_id>
mark_task_ready() {
    local task_id="$1"
    local src="${QUEUE_DIR}/pending/${task_id}.task"
    local dst="${QUEUE_DIR}/ready/${task_id}.task"

    if [[ -f "${src}" ]]; then
        update_task_status "${src}" "ready" \
            && mv "${src}" "${dst}" 2>/dev/null \
            || return 1
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
    local lock_timeout="${QUEUE_CLAIM_LOCK_TIMEOUT:-5}"
    if ! [[ "${lock_timeout}" =~ ^[0-9]+$ ]]; then
        lock_timeout=5
    fi
    acquire_queue_lock "${lock_timeout}" || return 1

    if [[ -f "${src}" ]]; then
        mark_task_started "${src}" "${gpu_id}" \
            && mv "${src}" "${dst}" 2>/dev/null \
            || { release_queue_lock; return 1; }
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

    acquire_queue_lock 10 || return 1

    if [[ ! -f "${src}" ]]; then
        release_queue_lock
        return 1
    fi

    mark_task_completed "${src}" \
        && mv "${src}" "${dst}" 2>/dev/null \
        || { release_queue_lock; return 1; }
    rm -f "${QUEUE_DIR}/running/${task_id}.pid"
    release_queue_lock

    # Update progress state outside the queue lock (derived from filesystem state).
    update_progress_state 2>/dev/null || true
    return 0
}

# Fail a task (move from running to failed)
# Usage: fail_task <task_id> <error_message>
fail_task() {
    local task_id="$1"
    local error_msg="$2"
    local src="${QUEUE_DIR}/running/${task_id}.task"
    local dst="${QUEUE_DIR}/failed/${task_id}.task"

    acquire_queue_lock 10 || return 1

    if [[ ! -f "${src}" ]]; then
        release_queue_lock
        return 1
    fi

    mark_task_failed "${src}" "${error_msg}" \
        && mv "${src}" "${dst}" 2>/dev/null \
        || { release_queue_lock; return 1; }
    rm -f "${QUEUE_DIR}/running/${task_id}.pid"
    release_queue_lock

    update_progress_state 2>/dev/null || true
    return 0
}

# Retry a failed task (move from failed to pending)
# Usage: retry_task <task_id>
retry_task() {
    local task_id="$1"
    local src="${QUEUE_DIR}/failed/${task_id}.task"

    if [[ -f "${src}" ]]; then
        acquire_queue_lock 10 || return 1
        # Re-check under lock
        if [[ ! -f "${src}" ]]; then
            release_queue_lock
            return 1
        fi

        local retries
        retries=$(get_task_field "${src}" "retries")
        local max_retries
        max_retries=$(get_task_field "${src}" "max_retries")

        if ! [[ "${retries}" =~ ^[0-9]+$ ]]; then
            retries=0
        fi
        if ! [[ "${max_retries}" =~ ^[0-9]+$ ]]; then
            max_retries=3
        fi

        if [[ ${retries} -lt ${max_retries} ]]; then
            # On retry, clear assignment fields so the scheduler can pick fresh GPUs.
            local target_status="pending"
            local target_dir="${QUEUE_DIR}/pending"
            if check_dependencies_met "${src}"; then
                target_status="ready"
                target_dir="${QUEUE_DIR}/ready"
            fi

            _runtime_python queue_state.py retry-task \
                --task-file "${src}" \
                --status "${target_status}" \
                || { release_queue_lock; return 1; }

            mv "${src}" "${target_dir}/" 2>/dev/null || { release_queue_lock; return 1; }
            release_queue_lock
            return 0
        fi

        release_queue_lock
        echo "Task ${task_id} has exceeded max retries (${max_retries})" >&2
        return 1
    fi
    return 1
}

# Reclaim orphaned tasks (tasks stuck in running without active worker)
# Usage: reclaim_orphaned_tasks <gpu_id>
reclaim_orphaned_tasks() {
    local gpu_id="$1"
    local count=0
    local running_dir="${QUEUE_DIR}/running"
    local task_file
    local -a task_ids=()
    local -a task_assigned=()
    local -a task_pids=()

    # Collect candidates under the queue lock, but do not kill processes while holding it.
    acquire_queue_lock 10 || { echo "Reclaimed 0 orphaned tasks from GPU ${gpu_id}"; return 0; }
    for task_file in "${running_dir}"/*.task; do
        [[ -f "${task_file}" ]] || continue

        local task_id
        task_id=$(get_task_id "${task_file}")

        local pid_file="${running_dir}/${task_id}.pid"
        local pid=""
        if [[ -f "${pid_file}" ]]; then
            pid=$(cat "${pid_file}" 2>/dev/null || true)
        fi

        local assigned_gpus
        assigned_gpus=$(get_task_assigned_gpus "${task_file}")
        assigned_gpus="${assigned_gpus// /}"

        local should_reclaim="false"
        if [[ -n "${assigned_gpus}" && "${assigned_gpus}" != "null" && "${assigned_gpus%%,*}" == "${gpu_id}" ]]; then
            should_reclaim="true"
        elif [[ -n "${pid}" && ( -z "${assigned_gpus}" || "${assigned_gpus}" == "null" ) ]]; then
            # Older or partially-updated running tasks may still have a live PID file
            # before assigned_gpus is recorded. Reclaim those inconsistent entries too.
            should_reclaim="true"
        fi

        if [[ "${should_reclaim}" == "true" ]]; then
            task_ids+=("${task_id}")
            task_assigned+=("${assigned_gpus}")
            task_pids+=("${pid}")
        fi
    done
    release_queue_lock

    local idx
    for idx in "${!task_ids[@]}"; do
        local task_id="${task_ids[$idx]}"
        local assigned_gpus="${task_assigned[$idx]}"
        local pid="${task_pids[$idx]}"
        local pid_killed="false"

        echo "Reclaiming orphaned task: ${task_id} from GPU ${gpu_id}"

        if [[ -n "${pid}" && "${pid}" =~ ^[0-9]+$ ]]; then
            if _cmd_kill -0 "${pid}" 2>/dev/null; then
                echo "  Killing task process group for ${task_id} (PID ${pid})"
                _cmd_kill -TERM "-${pid}" 2>/dev/null || _cmd_kill -TERM "${pid}" 2>/dev/null || true
                _sleep 2
                _cmd_kill -KILL "-${pid}" 2>/dev/null || _cmd_kill -KILL "${pid}" 2>/dev/null || true
                pid_killed="true"
            fi
        fi

        if [[ "${pid_killed}" != "true" && -n "${assigned_gpus}" ]]; then
            local can_kill="true"
            local gid
            local -a gpus=()
            if [[ -n "${GPU_RESERVATION_DIR:-}" ]]; then
                assigned_gpus="${assigned_gpus// /}"
                IFS=',' read -ra gpus <<< "${assigned_gpus}"
                for gid in "${gpus[@]}"; do
                    local lock_file="${GPU_RESERVATION_DIR}/gpu_${gid}.lock"
                    local owner
                    owner=$(cat "${lock_file}" 2>/dev/null || true)
                    if [[ -z "${owner}" || "${owner}" != "${task_id}" ]]; then
                        can_kill="false"
                        break
                    fi
                done
            fi

            if [[ "${can_kill}" == "true" ]]; then
                if type kill_gpu_processes &>/dev/null; then
                    echo "  Killing GPU processes on ${assigned_gpus} for ${task_id}"
                    kill_gpu_processes "${assigned_gpus}"
                fi
            fi
        fi

        if type release_gpus &>/dev/null; then
            release_gpus "${task_id}"
        fi
    done

    # Move tasks back to pending under the queue lock.
    acquire_queue_lock 10 || { echo "Reclaimed ${count} orphaned tasks from GPU ${gpu_id}"; return 0; }
    for idx in "${!task_ids[@]}"; do
        local task_id="${task_ids[$idx]}"
        local task_path="${running_dir}/${task_id}.task"
        [[ -f "${task_path}" ]] || continue

        update_task_status "${task_path}" "pending" || continue
        update_task_field "${task_path}" "assigned_gpus" "null" "true" 2>/dev/null || true
        rm -f "${running_dir}/${task_id}.pid" 2>/dev/null || true
        mv "${task_path}" "${QUEUE_DIR}/pending/" 2>/dev/null || continue
        count=$((count + 1))
    done
    release_queue_lock

    echo "Reclaimed ${count} orphaned tasks from GPU ${gpu_id}"
}
