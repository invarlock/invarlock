#!/usr/bin/env bash
# queue_dependencies.sh - Dependency resolution and dependent promotion
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Usage: sourced by queue_manager.sh

QUEUE_MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=queue_core.sh
[[ -z "${QUEUE_CORE_LOADED:-}" ]] && source "${QUEUE_MODULE_DIR}/queue_core.sh"
# shellcheck source=queue_lifecycle.sh
[[ -z "${QUEUE_LIFECYCLE_LOADED:-}" ]] && source "${QUEUE_MODULE_DIR}/queue_lifecycle.sh" && export QUEUE_LIFECYCLE_LOADED=1

# ============ DEPENDENCY RESOLUTION ============

# Check if all dependencies of a task are completed
# Usage: check_dependencies_met <task_file>
# Returns: 0 if all deps completed, 1 otherwise
check_dependencies_met() {
    local task_file="$1"
    local dep_id

    [[ -f "${task_file}" ]] || return 1

    local deps_output=""
    deps_output="$(get_task_dependencies "${task_file}")" || return 1

    local dep=""
    local -a deps=()
    while IFS= read -r dep; do
        [[ -n "${dep}" ]] && deps+=("${dep}")
    done < <(printf '%s\n' "${deps_output}")

    if [[ ${#deps[@]} -eq 0 ]]; then
        return 0  # No dependencies
    fi

    for dep_id in "${deps[@]}"; do
        if [[ ! -f "${QUEUE_DIR}/completed/${dep_id}.task" ]]; then
            return 1  # Dependency not completed
        fi
    done

    return 0  # All dependencies completed
}

# Cancel tasks whose dependencies have permanently failed.
# This prevents the queue from stalling forever when an upstream task fails.
# Inspired by Slurm dependency semantics (dependent jobs do not start if parent fails).
#
# Usage: cancel_tasks_with_failed_dependencies [grace_seconds]
# Returns: number of tasks moved pending->failed.
cancel_tasks_with_failed_dependencies() {
    local grace="${1:-${CANCEL_BLOCKED_TASKS_GRACE_SECONDS:-90}}"
    local canceled=0

    if ! [[ "${grace}" =~ ^[0-9]+$ ]]; then
        grace=90
    fi

    local -a candidate_task_ids=()
    local -a candidate_failed_deps_csv=()
    local task_file
    local dep_id
    local dep

    local now=""
    now=$(_now_epoch)

    # Scan WITHOUT holding the queue lock to avoid starving workers trying to claim tasks.
    for task_file in "${QUEUE_DIR}/pending"/*.task; do
        [[ -f "${task_file}" ]] || continue

        local -a deps=()
        while IFS= read -r dep; do
            [[ -n "${dep}" ]] && deps+=("${dep}")
        done < <(get_task_dependencies "${task_file}")

        [[ ${#deps[@]} -eq 0 ]] && continue

        local -a failed_deps=()
        for dep_id in "${deps[@]}"; do
            local dep_file="${QUEUE_DIR}/failed/${dep_id}.task"
            [[ -f "${dep_file}" ]] || continue

            # Use mtime as the failure timestamp to avoid GNU date parsing assumptions.
            local dep_mtime=""
            dep_mtime=$(_file_mtime_epoch "${dep_file}" 2>/dev/null || echo "")
            if [[ -z "${dep_mtime}" ]]; then
                failed_deps+=("${dep_id}")
                continue
            fi
            local dep_age=$(( now - dep_mtime ))
            if [[ ${dep_age} -ge ${grace} ]]; then
                failed_deps+=("${dep_id}")
            fi
        done

        if [[ ${#failed_deps[@]} -gt 0 ]]; then
            local task_id
            task_id=$(get_task_id "${task_file}")
            candidate_task_ids+=("${task_id}")
            candidate_failed_deps_csv+=("$(IFS=','; echo "${failed_deps[*]}")")
        fi
    done

    [[ ${#candidate_task_ids[@]} -eq 0 ]] && { echo "0"; return 0; }

    # Apply cancellations under the queue lock, re-checking that deps are still failed.
    acquire_queue_lock 10 || { echo "0"; return 0; }

    local idx
    local now_apply
    now_apply=$(_now_epoch)
    for idx in "${!candidate_task_ids[@]}"; do
        local task_id="${candidate_task_ids[$idx]}"
        local task_file="${QUEUE_DIR}/pending/${task_id}.task"
        [[ -f "${task_file}" ]] || continue

        local deps_csv="${candidate_failed_deps_csv[$idx]}"
        local -a still_failed=()
        local -a deps=()

        IFS=',' read -ra deps <<< "${deps_csv}"
        for dep_id in "${deps[@]}"; do
            [[ -n "${dep_id}" ]] || continue
            local dep_file="${QUEUE_DIR}/failed/${dep_id}.task"
            [[ -f "${dep_file}" ]] || continue

            local dep_mtime=""
            dep_mtime=$(_file_mtime_epoch "${dep_file}" 2>/dev/null || echo "")
            if [[ -z "${dep_mtime}" ]]; then
                still_failed+=("${dep_id}")
                continue
            fi
            local dep_age=$(( now_apply - dep_mtime ))
            if [[ ${dep_age} -ge ${grace} ]]; then
                still_failed+=("${dep_id}")
            fi
        done

        if [[ ${#still_failed[@]} -gt 0 ]]; then
            local msg
            msg="Dependency failed: $(IFS=','; echo "${still_failed[*]}")"
            mark_task_failed "${task_file}" "${msg}" 2>/dev/null || true
            mv "${task_file}" "${QUEUE_DIR}/failed/" 2>/dev/null || continue
            canceled=$((canceled + 1))
        fi
    done

    release_queue_lock

    if [[ ${canceled} -gt 0 ]]; then
        update_progress_state 2>/dev/null || true
    fi

    echo "${canceled}"
}

# Resolve dependencies and move ready tasks from pending to ready
# Usage: resolve_dependencies
# Returns: number of tasks moved to ready
resolve_dependencies() {
    local moved=0

    local -a candidate_task_ids=()
    local task_file
    local task_id
    local suite_mode="${PACK_SUITE_MODE:-full}"

    # Scan WITHOUT holding the queue lock to avoid starving workers trying to claim tasks.
    for task_file in "${QUEUE_DIR}/pending"/*.task; do
        [[ -f "${task_file}" ]] || continue

        if check_dependencies_met "${task_file}"; then
            if [[ "${suite_mode}" == "calibrate-only" ]]; then
                local task_type=""
                task_type="$(get_task_type "${task_file}" 2>/dev/null || true)"
                case "${task_type}" in
                    SETUP_BASELINE|CALIBRATION_RUN|GENERATE_PRESET)
                        :
                        ;;
                    *)
                        continue
                        ;;
                esac
            fi
            task_id="$(get_task_id "${task_file}")" || continue
            candidate_task_ids+=("${task_id}")
        fi
    done

    [[ ${#candidate_task_ids[@]} -eq 0 ]] && { echo "0"; return 0; }

    acquire_queue_lock 10 || return 0

    for task_id in "${candidate_task_ids[@]}"; do
        local task_file="${QUEUE_DIR}/pending/${task_id}.task"
        [[ -f "${task_file}" ]] || continue
        if check_dependencies_met "${task_file}"; then
            if [[ "${suite_mode}" == "calibrate-only" ]]; then
                local task_type=""
                task_type="$(get_task_type "${task_file}" 2>/dev/null || true)"
                case "${task_type}" in
                    SETUP_BASELINE|CALIBRATION_RUN|GENERATE_PRESET)
                        :
                        ;;
                    *)
                        continue
                        ;;
                esac
            fi
            update_task_status "${task_file}" "ready" \
                && mv "${task_file}" "${QUEUE_DIR}/ready/" 2>/dev/null \
                && moved=$((moved + 1)) \
                || true
        fi
    done

    release_queue_lock
    echo "${moved}"
}

# Demote any "ready" tasks that are disallowed under a calibration-only run.
# This is a safety net for resumed queues where non-calibration tasks might
# already be in the ready queue when switching into calibrate-only mode.
demote_ready_tasks_for_calibration_only() {
    local suite_mode="${PACK_SUITE_MODE:-full}"
    [[ "${suite_mode}" == "calibrate-only" ]] || return 0

    acquire_queue_lock 10 || return 0

    local task_file
    for task_file in "${QUEUE_DIR}/ready"/*.task; do
        [[ -f "${task_file}" ]] || continue
        local task_type=""
        task_type="$(get_task_type "${task_file}" 2>/dev/null || true)"
        case "${task_type}" in
            SETUP_BASELINE|CALIBRATION_RUN|GENERATE_PRESET)
                :
                ;;
            *)
                update_task_status "${task_file}" "pending" 2>/dev/null || true
                mv "${task_file}" "${QUEUE_DIR}/pending/" 2>/dev/null || true
                ;;
        esac
    done

    release_queue_lock
    return 0
}

# Update dependents after task completion
# Usage: update_dependents <completed_task_id>
update_dependents() {
    local completed_id="$1"
    local task_file

    # Check all pending tasks for this dependency
    for task_file in "${QUEUE_DIR}/pending"/*.task; do
        [[ -f "${task_file}" ]] || continue

        local deps=""
        deps="$(get_task_dependencies "${task_file}" 2>/dev/null | tr '\n' ' ' || true)"
        if [[ " ${deps} " =~ " ${completed_id} " ]]; then
            if check_dependencies_met "${task_file}"; then
                local task_id=""
                task_id="$(get_task_id "${task_file}")" || continue
                mark_task_ready "${task_id}" 2>/dev/null || true
            fi
        fi
    done
}
