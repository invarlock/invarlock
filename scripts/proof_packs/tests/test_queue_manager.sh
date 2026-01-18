#!/usr/bin/env bash

test_acquire_queue_lock_recovers_stale_owner_pid() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    # Force stale-owner recovery regardless of host (/proc vs ps).
    _pid_is_alive() { return 1; }

    local lock_dir="${QUEUE_DIR}/queue.lock.d"
    mkdir -p "${lock_dir}"
    echo "99999" > "${lock_dir}/owner"

    acquire_queue_lock 5
    assert_dir_exists "${QUEUE_LOCK_DIR}" "lock acquired"
    release_queue_lock
}

test_claim_complete_fail_and_retry_transitions() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local t1
    t1="$(add_task "SETUP_BASELINE" "org/model" "model" "14" "none" '{}' "50")"

    # pending -> ready
    assert_eq "1" "$(resolve_dependencies)" "no-deps task promoted"
    assert_file_exists "${QUEUE_DIR}/ready/${t1}.task" "task in ready"

    # ready -> running
    run claim_task "${t1}" "2"
    assert_rc "0" "${RUN_RC}" "claim_task ok"
    assert_file_exists "${QUEUE_DIR}/running/${t1}.task" "task in running"
    assert_eq "running" "$(jq -r '.status' "${QUEUE_DIR}/running/${t1}.task")" "status running"
    assert_eq "2" "$(jq -r '.gpu_id' "${QUEUE_DIR}/running/${t1}.task")" "gpu_id set"

    # running -> completed
    run complete_task "${t1}"
    assert_rc "0" "${RUN_RC}" "complete_task ok"
    assert_file_exists "${QUEUE_DIR}/completed/${t1}.task" "task completed"
    assert_eq "completed" "$(jq -r '.status' "${QUEUE_DIR}/completed/${t1}.task")" "status completed"

    # running -> failed
    local t2
    t2="$(add_task "SETUP_BASELINE" "org/model" "model2" "14" "none" '{}' "50")"
    resolve_dependencies >/dev/null
    claim_task "${t2}" "0" >/dev/null
    run fail_task "${t2}" "boom"
    assert_rc "0" "${RUN_RC}" "fail_task ok"
    assert_file_exists "${QUEUE_DIR}/failed/${t2}.task" "task failed"
    assert_eq "failed" "$(jq -r '.status' "${QUEUE_DIR}/failed/${t2}.task")" "status failed"
    assert_eq "boom" "$(jq -r '.error_msg' "${QUEUE_DIR}/failed/${t2}.task")" "error msg set"

    # failed -> pending (retry) when deps not met
    run retry_task "${t2}"
    assert_rc "0" "${RUN_RC}" "retry_task ok"
    assert_file_exists "${QUEUE_DIR}/ready/${t2}.task" "no-deps retry goes ready"
    assert_eq "1" "$(jq -r '.retries' "${QUEUE_DIR}/ready/${t2}.task")" "retries incremented"
    assert_eq "-1" "$(jq -r '.gpu_id' "${QUEUE_DIR}/ready/${t2}.task")" "gpu reset"
    assert_eq "null" "$(jq -r '.assigned_gpus' "${QUEUE_DIR}/ready/${t2}.task")" "assigned_gpus reset"
}

test_resolve_dependencies_filters_non_calibration_tasks_in_calibration_only_mode() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    export PACK_SUITE_MODE="calibrate-only"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local setup_id
    setup_id="$(add_task "SETUP_BASELINE" "org/model" "model" "14" "none" '{}' "50")"
    local eval_id
    eval_id="$(add_task "EVAL_BASELINE" "org/model" "model" "14" "${setup_id}" '{}' "50")"

    assert_eq "1" "$(resolve_dependencies)" "setup task promoted"
    assert_file_exists "${QUEUE_DIR}/ready/${setup_id}.task" "setup ready"
    assert_file_exists "${QUEUE_DIR}/pending/${eval_id}.task" "eval remains pending until deps met"

    claim_task "${setup_id}" "0" >/dev/null
    complete_task "${setup_id}" >/dev/null

    assert_eq "0" "$(resolve_dependencies)" "eval not promoted under calibration-only"
    assert_file_exists "${QUEUE_DIR}/pending/${eval_id}.task" "eval still pending"
    [[ ! -f "${QUEUE_DIR}/ready/${eval_id}.task" ]] || t_fail "eval should not be ready under calibration-only"
}

test_demote_ready_tasks_for_calibration_only_moves_disallowed_ready_to_pending() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    export PACK_SUITE_MODE="calibrate-only"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local allowed_id
    allowed_id="$(add_task "SETUP_BASELINE" "org/model" "model" "14" "none" '{}' "50")"
    local disallowed_id
    disallowed_id="$(add_task "EVAL_BASELINE" "org/model" "model" "14" "none" '{}' "50")"

    update_task_status "${QUEUE_DIR}/pending/${allowed_id}.task" "ready"
    mv "${QUEUE_DIR}/pending/${allowed_id}.task" "${QUEUE_DIR}/ready/${allowed_id}.task"

    update_task_status "${QUEUE_DIR}/pending/${disallowed_id}.task" "ready"
    mv "${QUEUE_DIR}/pending/${disallowed_id}.task" "${QUEUE_DIR}/ready/${disallowed_id}.task"

    demote_ready_tasks_for_calibration_only

    assert_file_exists "${QUEUE_DIR}/ready/${allowed_id}.task" "allowed task stays ready"
    assert_file_exists "${QUEUE_DIR}/pending/${disallowed_id}.task" "disallowed task demoted"
    assert_eq "pending" "$(jq -r '.status' "${QUEUE_DIR}/pending/${disallowed_id}.task")" "status updated to pending"
}

test_claim_task_returns_nonzero_when_mark_task_started_fails() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local t1
    t1="$(add_task "SETUP_BASELINE" "org/model" "model" "14" "none" '{}' "50")"
    resolve_dependencies >/dev/null

    mark_task_started() { return 1; }

    run claim_task "${t1}" "0"
    assert_rc "1" "${RUN_RC}" "claim_task fails when mark_task_started fails"
    assert_file_exists "${QUEUE_DIR}/ready/${t1}.task" "task remains ready"
    [[ ! -f "${QUEUE_DIR}/running/${t1}.task" ]] || t_fail "expected task not moved to running on start failure"
}

test_complete_task_returns_nonzero_when_mark_task_completed_fails() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local t1
    t1="$(add_task "SETUP_BASELINE" "org/model" "model" "14" "none" '{}' "50")"
    resolve_dependencies >/dev/null
    claim_task "${t1}" "0" >/dev/null

    mark_task_completed() { return 1; }

    run complete_task "${t1}"
    assert_rc "1" "${RUN_RC}" "complete_task fails when mark_task_completed fails"
    assert_file_exists "${QUEUE_DIR}/running/${t1}.task" "task remains running"
    [[ ! -f "${QUEUE_DIR}/completed/${t1}.task" ]] || t_fail "expected task not moved to completed on completion failure"
}

test_fail_task_returns_nonzero_when_mark_task_failed_fails() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local t1
    t1="$(add_task "SETUP_BASELINE" "org/model" "model" "14" "none" '{}' "50")"
    resolve_dependencies >/dev/null
    claim_task "${t1}" "0" >/dev/null

    mark_task_failed() { return 1; }

    run fail_task "${t1}" "boom"
    assert_rc "1" "${RUN_RC}" "fail_task fails when mark_task_failed fails"
    assert_file_exists "${QUEUE_DIR}/running/${t1}.task" "task remains running"
    [[ ! -f "${QUEUE_DIR}/failed/${t1}.task" ]] || t_fail "expected task not moved to failed on failure update error"
}

test_retry_task_respects_max_retries() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local task_id="tmax"
    jq -n '{task_id:"tmax", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"failed", retries:3, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:"x", gpu_id:0, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/failed/${task_id}.task"

    run retry_task "${task_id}"
    assert_rc "1" "${RUN_RC}" "exceeded retries fails"
    assert_match 'exceeded max retries' "${RUN_ERR}" "message"
}

test_retry_task_sanitizes_missing_retry_fields() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local task_id="tmissing"
    jq -n '{task_id:"tmissing", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"failed", retries:null, max_retries:null, created_at:"x", started_at:null, completed_at:null, error_msg:"x", gpu_id:0, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/failed/${task_id}.task"

    run retry_task "${task_id}"
    assert_rc "0" "${RUN_RC}" "retry_task succeeds with null retry fields"
    assert_file_exists "${QUEUE_DIR}/ready/${task_id}.task" "task moved to ready"
    assert_eq "1" "$(jq -r '.retries' "${QUEUE_DIR}/ready/${task_id}.task")" "retries incremented from default"
}

test_cancel_tasks_with_failed_dependencies_moves_pending_to_failed() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    # Create failed dep task.
    jq -n '{task_id:"dep", task_type:"SETUP_BASELINE", model_id:"m", model_name:"d", status:"failed", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:"x", gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/failed/dep.task"

    # Create pending task that depends on failed dep.
    jq -n '{task_id:"child", task_type:"EVAL_BASELINE", model_id:"m", model_name:"c", status:"pending", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:["dep"], params:{}, priority:50}' \
        > "${QUEUE_DIR}/pending/child.task"

    local canceled
    canceled="$(cancel_tasks_with_failed_dependencies 0)"
    assert_eq "1" "${canceled}" "child canceled"
    assert_file_exists "${QUEUE_DIR}/failed/child.task" "moved to failed"
    assert_eq "failed" "$(jq -r '.status' "${QUEUE_DIR}/failed/child.task")" "status failed"
    assert_match 'Dependency failed' "$(jq -r '.error_msg' "${QUEUE_DIR}/failed/child.task")" "error msg"
}

test_queue_lock_timeout_and_no_owner_stale_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    _sleep() { :; }
    _now_epoch() { echo "100"; }

    # Timeout branch with owner_pid read.
    _pid_is_alive() { return 0; }
    mkdir -p "${QUEUE_DIR}/queue.lock.d"
    echo "4242" > "${QUEUE_DIR}/queue.lock.d/owner"
    local rc=0
    acquire_queue_lock 0 || rc=$?
    assert_rc "1" "${rc}" "acquire_queue_lock times out"

    # Ownerless lock dir: invalid no-owner grace coerces to 30 and stale lock removed.
    rm -rf "${QUEUE_DIR}/queue.lock.d"
    mkdir -p "${QUEUE_DIR}/queue.lock.d"
    QUEUE_LOCK_NOOWNER_STALE_SECONDS="nope"
    _file_mtime_epoch() { echo "0"; }
    acquire_queue_lock 1
    release_queue_lock
}

test_queue_task_listing_and_count_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    jq -n '{task_id:"a", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"ready", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/ready/a.task"

    local tasks
    tasks="$(get_tasks_by_status "ready")"
    assert_match 'ready/a\.task' "${tasks}" "get_tasks_by_status lists tasks"

    assert_eq "0" "$(count_tasks "no_such_status")" "missing status dir counts as 0"
}

test_mark_task_ready_and_claim_lock_timeout_validation() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    jq -n '{task_id:"t1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"pending", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/pending/t1.task"

    mark_task_ready "t1"
    assert_file_exists "${QUEUE_DIR}/ready/t1.task" "mark_task_ready moves file"

    QUEUE_CLAIM_LOCK_TIMEOUT="bad"
    claim_task "t1" "2"
    assert_file_exists "${QUEUE_DIR}/running/t1.task" "claim_task moves to running"
}

test_complete_fail_and_retry_missing_file_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local rc=0
    rc=0; complete_task "missing" || rc=$?; assert_rc "1" "${rc}" "complete_task fails when missing"
    rc=0; fail_task "missing" "boom" || rc=$?; assert_rc "1" "${rc}" "fail_task fails when missing"

    local task_id="race"
    jq -n '{task_id:"race", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"failed", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:"x", gpu_id:0, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/failed/${task_id}.task"

    acquire_queue_lock() { rm -f "${QUEUE_DIR}/failed/${task_id}.task"; return 0; }
    release_queue_lock() { return 0; }

    rc=0
    retry_task "${task_id}" || rc=$?
    assert_rc "1" "${rc}" "retry_task returns 1 when file disappears under lock"
}

test_reclaim_orphaned_tasks_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    _sleep() { :; }

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    _cmd_kill() {
        if [[ "${1:-}" == "-0" ]]; then
            return 0
        fi
        return 0
    }

    local reservation_dir="${TEST_TMPDIR}/reservations"
    mkdir -p "${reservation_dir}"
    export GPU_RESERVATION_DIR="${reservation_dir}"

    local called_kill_gpu="0"
    kill_gpu_processes() { called_kill_gpu="1"; }
    local released_ids=""
    release_gpus() { released_ids+="${1},"; }

    # Task A: has pid, kill path taken.
    jq -n '{task_id:"a", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"running", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:0, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/running/a.task"
    echo "123" > "${QUEUE_DIR}/running/a.pid"

    # Task B: owner mismatch => can_kill false (reservation check).
    jq -n '{task_id:"b", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"running", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:0, assigned_gpus:"0,1", dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/running/b.task"
    echo "" > "${reservation_dir}/gpu_0.lock"

    # Task C: owners match => kill_gpu_processes + release_gpus.
    jq -n '{task_id:"c", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"running", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:0, assigned_gpus:"0,1", dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/running/c.task"
    echo "c" > "${reservation_dir}/gpu_0.lock"
    echo "c" > "${reservation_dir}/gpu_1.lock"

    reclaim_orphaned_tasks 0 >/dev/null

    assert_file_exists "${QUEUE_DIR}/pending/a.task" "task moved back to pending"
    assert_file_exists "${QUEUE_DIR}/pending/b.task" "task moved back to pending"
    assert_file_exists "${QUEUE_DIR}/pending/c.task" "task moved back to pending"
    assert_eq "1" "${called_kill_gpu}" "kill_gpu_processes invoked"
    assert_match 'a,' "${released_ids}" "release_gpus called for task a"
    assert_match 'b,' "${released_ids}" "release_gpus called for task b"
    assert_match 'c,' "${released_ids}" "release_gpus called for task c"
}

test_check_dependencies_met_and_update_dependents_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    jq -n '{task_id:"child", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"pending", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:["dep"], params:{}, priority:50}' \
        > "${QUEUE_DIR}/pending/child.task"

    local rc=0
    check_dependencies_met "${QUEUE_DIR}/pending/child.task" || rc=$?
    assert_rc "1" "${rc}" "dependency not completed"

    jq -n '{task_id:"dep", task_type:"SETUP_BASELINE", model_id:"m", model_name:"d", status:"completed", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/completed/dep.task"

    update_dependents "dep"
    assert_file_exists "${QUEUE_DIR}/ready/child.task" "update_dependents promotes to ready"
}

test_cancel_tasks_with_failed_dependencies_invalid_grace_and_mtime_missing_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }
    _now_epoch() { echo "100"; }
    _file_mtime_epoch() { echo ""; }

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    jq -n '{task_id:"dep", task_type:"SETUP_BASELINE", model_id:"m", model_name:"d", status:"failed", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:"x", gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/failed/dep.task"

    jq -n '{task_id:"child", task_type:"EVAL_BASELINE", model_id:"m", model_name:"c", status:"pending", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:["dep"], params:{}, priority:50}' \
        > "${QUEUE_DIR}/pending/child.task"

    assert_eq "1" "$(cancel_tasks_with_failed_dependencies "bad")" "cancels with invalid grace coerced"
    assert_file_exists "${QUEUE_DIR}/failed/child.task" "moved to failed"
}

test_update_progress_state_status_and_percent_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    jq -n '{task_id:"t", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"completed", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:"x", error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/completed/t.task"

    update_progress_state
    assert_match '\"status\": \"completed\"' "$(cat "${out_dir}/state/progress.json")" "completed status"

    rm -f "${QUEUE_DIR}/completed/t.task"
    jq -n '{task_id:"t", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"failed", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:"x", error_msg:"x", gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/failed/t.task"
    update_progress_state
    assert_match '\"status\": \"completed_with_failures\"' "$(cat "${out_dir}/state/progress.json")" "completed_with_failures status"
}

test_find_and_refresh_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    jq -n '{task_id:"m1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"m1", status:"pending", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/pending/m1.task"

    assert_match 'pending/m1\.task' "$(find_tasks_by_model "m1" "pending")" "find_tasks_by_model with status"
    assert_match 'pending/m1\.task' "$(find_tasks_by_model "m1")" "find_tasks_by_model all statuses"
    assert_match 'pending/m1\.task' "$(find_tasks_by_type "SETUP_BASELINE" "pending")" "find_tasks_by_type with status"
    assert_match 'pending/m1\.task' "$(find_tasks_by_type "SETUP_BASELINE")" "find_tasks_by_type all statuses"

    local calls=""
    update_model_task_memory() { calls+="$1|$3;"; }
    mkdir -p "${out_dir}/m1"
    echo "org/model" > "${out_dir}/m1/.model_id"
    refresh_task_memory_from_profiles "${out_dir}"
    assert_match 'm1\\|org/model' "${calls}" "refresh passes through model_id"
}

test_generate_model_tasks_use_batch_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local seq=0
    add_task() { seq=$((seq + 1)); echo "t${seq}"; }
    generate_eval_certify_tasks() { :; }
    generate_edit_tasks() { :; }

    estimate_model_memory() {
        local model_id="$1"
        local task_type="${2:-}"
        case "${model_id}:${task_type}" in
            *":EVAL_BASELINE") echo "${EST_BASELINE:-}" ;;
            *) echo "14" ;;
        esac
    }

    EST_BASELINE=""
    generate_model_tasks "1" "org/model-medium" "model-medium" >/dev/null

    EST_BASELINE="160"
    generate_model_tasks "1" "org/model-small" "model-small" >/dev/null

    EST_BASELINE="175"
    generate_model_tasks "1" "org/model-large-by-size" "model-large" >/dev/null

    EST_BASELINE="175"
    generate_model_tasks "1" "org/model-70B" "model-70b" >/dev/null
}

test_generate_model_tasks_sanitizes_invalid_calibration_runs() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local calls="${TEST_TMPDIR}/calls"
    : > "${calls}"
    add_task() {
        local task_type="$1"
        printf '%s\n' "${task_type}" >> "${calls}"
        local count
        count=$(wc -l < "${calls}" | tr -d ' ')
        echo "t${count}"
    }
    generate_eval_certify_tasks() { :; }
    generate_edit_tasks() { :; }
    estimate_model_memory() { echo "14"; }

    DRIFT_CALIBRATION_RUNS="nope"
    generate_model_tasks "1" "org/model" "model" >/dev/null
    local cal_count
    cal_count="$(awk '/^CALIBRATION_RUN$/ {c++} END {print c+0}' "${calls}")"
    assert_eq "5" "${cal_count}" "invalid calibration runs default to 5"
}

test_generate_model_tasks_disables_batch_for_large_memory_and_uses_manifest_fallbacks() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local calls="${TEST_TMPDIR}/calls"
    : > "${calls}"

    add_task() {
        local task_type="$1"
        local deps="$5"
        local params_json="$6"
        printf '%s|%s|%s\n' "${task_type}" "${deps}" "${params_json}" >> "${calls}"
        local count
        count=$(wc -l < "${calls}" | tr -d ' ')
        echo "t${count}"
    }

    # Simulate a large model memory estimate (>=170GB) without tripping the name heuristics.
    estimate_model_memory() { echo "200"; }

    # Force manifest fallbacks by making `command -v jq` report jq missing.
    command() {
        if [[ "${1:-}" == "-v" && "${2:-}" == "jq" ]]; then
            return 1
        fi
        builtin command "$@"
    }

    CLEAN_EDIT_RUNS=1
    STRESS_EDIT_RUNS=1
    DRIFT_CALIBRATION_RUNS=0
    PACK_PRESET_READY=1
    RUN_ERROR_INJECTION=true
    export CLEAN_EDIT_RUNS STRESS_EDIT_RUNS DRIFT_CALIBRATION_RUNS PACK_PRESET_READY RUN_ERROR_INJECTION

    generate_model_tasks "1" "org/Thing-40B" "model" >/dev/null

    local all_calls
    all_calls="$(cat "${calls}")"
    [[ "${all_calls}" != *"CREATE_EDITS_BATCH"* ]] || t_fail "expected large memory to disable batch edits"
    assert_match "CREATE_EDIT\\|" "${all_calls}" "per-edit create tasks emitted"

    # preset_ready=1 should be normalized to true so certify tasks are emitted even with DRIFT_CALIBRATION_RUNS=0.
    assert_match "CERTIFY_EDIT\\|" "${all_calls}" "certify tasks emitted when preset_ready=1"

    # scenarios.json + jq fallback defaults should be used.
    assert_match "quant_rtn:clean:ffn" "${all_calls}" "fallback clean edit spec used"
    assert_match "weight_tying_break" "${all_calls}" "fallback error type used"
}

test_generate_edit_tasks_sanitizes_cert_runs() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local calls="${TEST_TMPDIR}/calls"
    : > "${calls}"

    add_task() {
        local task_type="$1"
        printf '%s\n' "${task_type}" >> "${calls}"
        local count
        count=$(wc -l < "${calls}" | tr -d ' ')
        echo "t${count}"
    }
    estimate_model_memory() { echo "14"; }

    local cert_count
    cert_count="$(awk '/^CERTIFY_EDIT$/ {c++} END {print c+0}' "${calls}")"
    assert_eq "0" "${cert_count}" "precondition: no certify tasks yet"

    : > "${calls}"
    generate_certify_tasks "m" "n" "edit" "preset" "spec" "clean" "bad" >/dev/null
    cert_count="$(awk '/^CERTIFY_EDIT$/ {c++} END {print c+0}' "${calls}")"
    assert_eq "1" "${cert_count}" "invalid cert_runs defaults to 1"

    : > "${calls}"
    generate_certify_tasks "m" "n" "edit" "preset" "spec" "clean" "-2" >/dev/null
    cert_count="$(awk '/^CERTIFY_EDIT$/ {c++} END {print c+0}' "${calls}")"
    assert_eq "0" "${cert_count}" "negative cert_runs clamps to 0"

    : > "${calls}"
    generate_edit_tasks "m" "n" "setup" "preset" "spec" "clean" "bad" >/dev/null
    cert_count="$(awk '/^CERTIFY_EDIT$/ {c++} END {print c+0}' "${calls}")"
    local create_count
    create_count="$(awk '/^CREATE_EDIT$/ {c++} END {print c+0}' "${calls}")"
    assert_eq "1" "${create_count}" "create_edit task still created"
    assert_eq "1" "${cert_count}" "invalid cert_runs defaults to 1"

    : > "${calls}"
    generate_edit_tasks "m" "n" "setup" "preset" "spec" "clean" "-1" >/dev/null
    cert_count="$(awk '/^CERTIFY_EDIT$/ {c++} END {print c+0}' "${calls}")"
    create_count="$(awk '/^CREATE_EDIT$/ {c++} END {print c+0}' "${calls}")"
    assert_eq "1" "${create_count}" "create_edit task created with negative cert_runs"
    assert_eq "0" "${cert_count}" "negative cert_runs clamps to 0"
}

test_generate_all_tasks_and_update_model_task_memory_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local calls=""
    generate_model_tasks() { calls+="$2;"; }
    resolve_dependencies() { echo "0"; }
    update_progress_state() { :; }
    print_queue_stats() { :; }

    generate_all_tasks "" "org/model" "" "" "" "" "" ""
    assert_match 'org/model' "${calls}" "generate_all_tasks invokes model generator for non-empty ids"

    # update_model_task_memory profile selection + numeric result updates.
    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    _cmd_python() { echo "123 2"; }

    jq -n '{task_id:"t", task_type:"SETUP_BASELINE", model_id:"m", model_name:"m1", status:"pending", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/pending/t.task"

    local model_name="m1"
    mkdir -p "${out_dir}/${model_name}"
    local baseline_path="${TEST_TMPDIR}/baseline"
    mkdir -p "${baseline_path}"
    echo "${baseline_path}" > "${out_dir}/${model_name}/.baseline_path"
    echo '{}' > "${baseline_path}/model_profile.json"

    update_model_task_memory "${model_name}" "${out_dir}" ""
    assert_eq "123" "$(jq -r '.model_size_gb' "${QUEUE_DIR}/pending/t.task")" "model_size_gb updated"
    assert_eq "2" "$(jq -r '.required_gpus' "${QUEUE_DIR}/pending/t.task")" "required_gpus updated"
    assert_file_exists "${out_dir}/analysis/memory_plan.csv" "export_memory_plan runs"

    # Fallback: no baseline path file, candidate search using model_id.
    rm -f "${out_dir}/${model_name}/.baseline_path"
    local sanitized="org__model"
    mkdir -p "${out_dir}/models/${sanitized}/baseline"
    echo '{}' > "${out_dir}/models/${sanitized}/baseline/model_profile.json"
    update_model_task_memory "${model_name}" "${out_dir}" "org/model"
}

test_with_queue_lock_returns_nonzero_when_lock_acquire_fails() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    acquire_queue_lock() { return 1; }
    should_not_run() { t_fail "with_queue_lock should not execute action when lock fails"; }

    run with_queue_lock should_not_run
    assert_rc "1" "${RUN_RC}" "with_queue_lock returns error when lock unavailable"
}

test_with_queue_lock_runs_action_and_propagates_status() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    failing_action() { return 42; }
    run with_queue_lock failing_action
    assert_rc "42" "${RUN_RC}" "with_queue_lock returns action rc"
}

test_acquire_queue_lock_sleeps_when_lock_held_by_live_owner() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local lock_dir="${QUEUE_DIR}/queue.lock.d"
    mkdir -p "${lock_dir}"
    echo "123" > "${lock_dir}/owner"

    local now_state="${TEST_TMPDIR}/now_epoch.calls"
    : >"${now_state}"
    _now_epoch() {
        # NOTE: acquire_queue_lock captures _now_epoch via command substitution, which runs in a subshell.
        # Persist call counts via a file so each invocation can advance deterministically.
        local n=0
        n="$(cat "${now_state}" 2>/dev/null || echo "0")"
        n=$((n + 1))
        printf '%s' "${n}" >"${now_state}"
        case "${n}" in
            1|2) echo "0" ;;  # start + first loop iteration
            *) echo "1" ;;     # hit deadline on second loop iteration
        esac
    }
    _pid_is_alive() { return 0; }

    local slept=0
    _sleep() { slept=$((slept + 1)); }

    run acquire_queue_lock 1
    assert_rc "1" "${RUN_RC}" "times out while lock held"
    assert_eq "1" "${slept}" "sleeps before retry"
}

test_print_queue_stats_and_is_queue_complete_cover_success_and_failure() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local task_id="t1"
    jq -n '{task_id:"t1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"failed", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:"x", gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/failed/${task_id}.task"

    local pending_id="t2"
    jq -n '{task_id:"t2", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"pending", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/pending/${pending_id}.task"

    assert_match 'QUEUE STATUS' "$(print_queue_stats)" "queue status header"
    ! is_queue_empty
    ! is_queue_complete

    rm -f "${QUEUE_DIR}/pending/${pending_id}.task"
    is_queue_empty

    rm -f "${QUEUE_DIR}/failed/${task_id}.task"
    is_queue_complete
}

test_mark_task_ready_and_claim_task_return_nonzero_when_source_missing() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    run mark_task_ready "nope"
    assert_rc "1" "${RUN_RC}" "mark_task_ready returns 1 when source missing"

    run claim_task "nope" "0"
    assert_rc "1" "${RUN_RC}" "claim_task returns 1 when source missing"
}

test_mark_task_ready_returns_nonzero_when_update_task_status_fails() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    jq -n '{task_id:"t1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"pending", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/pending/t1.task"

    update_task_status() { return 1; }

    run mark_task_ready "t1"
    assert_rc "1" "${RUN_RC}" "mark_task_ready returns error when update_task_status fails"
    assert_file_exists "${QUEUE_DIR}/pending/t1.task" "task remains pending"
    [[ ! -f "${QUEUE_DIR}/ready/t1.task" ]] || t_fail "expected task not moved to ready on status update failure"
}

test_check_dependencies_met_returns_nonzero_when_task_json_is_invalid() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    echo '{not-json' > "${QUEUE_DIR}/pending/bad.task"
    run check_dependencies_met "${QUEUE_DIR}/pending/bad.task"
    assert_rc "1" "${RUN_RC}" "invalid task json treated as unmet dependencies"
}

test_check_dependencies_met_returns_nonzero_when_task_file_missing() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    run check_dependencies_met "${QUEUE_DIR}/pending/missing.task"
    assert_rc "1" "${RUN_RC}" "missing task file treated as unmet dependencies"
}

test_generate_certify_tasks_and_generate_edit_tasks_create_expected_tasks() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }
    estimate_model_memory() { echo "14"; }

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local out
    out="$(generate_certify_tasks "org/model" "m" "edit1" "preset1" "quant_rtn:8:128:ffn" "clean" "1")"
    assert_match 'Created: ' "${out}" "creates certify tasks"

    local pending_count
    pending_count="$(ls "${QUEUE_DIR}/pending"/*.task 2>/dev/null | wc -l | tr -d ' ')"
    assert_eq "1" "${pending_count}" "creates 1 certify task"

    local task_file task_type version
    for task_file in "${QUEUE_DIR}/pending"/*.task; do
        task_type="$(jq -r '.task_type' "${task_file}")"
        assert_eq "CERTIFY_EDIT" "${task_type}" "certify task type"
        version="$(jq -r '.params.version // ""' "${task_file}")"
        assert_eq "clean" "${version}" "certify task carries version hint"
    done

    run generate_edit_tasks "org/model" "m" "setup1" "preset1" "quant_rtn:8:128:ffn" "clean" "1"
    assert_match 'deprecated' "${RUN_ERR}" "generate_edit_tasks warns"
}

test_task_ops_short_circuit_when_lock_acquire_fails() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    acquire_queue_lock() { return 1; }

    run claim_task "t1" "0"
    assert_rc "1" "${RUN_RC}" "claim_task returns non-zero when lock unavailable"

    run complete_task "t1"
    assert_rc "1" "${RUN_RC}" "complete_task returns non-zero when lock unavailable"

    run fail_task "t1" "boom"
    assert_rc "1" "${RUN_RC}" "fail_task returns non-zero when lock unavailable"
}

test_retry_task_short_circuits_when_lock_acquire_fails() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    jq -n '{task_id:"t1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"failed", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:"x", gpu_id:0, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/failed/t1.task"

    acquire_queue_lock() { return 1; }
    run retry_task "t1"
    assert_rc "1" "${RUN_RC}" "retry_task returns non-zero when lock unavailable"
}

test_retry_task_returns_nonzero_when_task_missing() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    run retry_task "no_such_task"
    assert_rc "1" "${RUN_RC}" "retry_task returns non-zero when task file is missing"
}

test_retry_task_atomic_update_failure_triggers_error_block() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    jq -n '{task_id:"t1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"failed", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:"x", gpu_id:0, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/failed/t1.task"

    local real_jq
    real_jq="$(command -v jq)"
    jq() {
        if [[ "${1:-}" == "--arg" && "${2:-}" == "status" ]]; then
            return 1
        fi
        "${real_jq}" "$@"
    }

    run retry_task "t1"
    assert_rc "1" "${RUN_RC}" "jq failure triggers error path"
}

test_retry_task_move_failure_returns_error() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    jq -n '{task_id:"t1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"failed", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:"x", gpu_id:0, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/failed/t1.task"

    mv() {
        if [[ "${2:-}" == "${QUEUE_DIR}/ready/" || "${2:-}" == "${QUEUE_DIR}/pending/" ]]; then
            return 1
        fi
        command mv "$@"
    }

    run retry_task "t1"
    assert_rc "1" "${RUN_RC}" "move failure triggers error return"
    assert_file_exists "${QUEUE_DIR}/failed/t1.task" "task remains in failed when move fails"
}

test_update_progress_state_returns_nonzero_when_atomic_move_fails() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    mv() { return 1; }

    run update_progress_state
    assert_rc "1" "${RUN_RC}" "mv failure returns non-zero"
}

test_queue_manager_find_task_returns_path() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    jq -n '{task_id:"t1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"ready", dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/ready/t1.task"

    assert_eq "${QUEUE_DIR}/ready/t1.task" "$(find_task "t1")" "find_task returns path"

    run find_task "missing"
    assert_rc "1" "${RUN_RC}" "find_task returns non-zero when missing"
    assert_eq "" "${RUN_OUT}" "find_task returns empty output when missing"
}

test_queue_manager_resolve_dependencies_skips_disallowed_tasks() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    export PACK_SUITE_MODE="calibrate-only"
    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    jq -n '{task_id:"t1", task_type:"EVAL_BASELINE", model_id:"m", model_name:"n", status:"pending", dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/pending/t1.task"

    check_dependencies_met() { return 0; }

    assert_eq "0" "$(resolve_dependencies)" "disallowed task skipped"
    assert_file_exists "${QUEUE_DIR}/pending/t1.task" "task stays pending"
}


test_queue_manager_resolve_dependencies_skips_on_second_pass_after_type_change() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    export PACK_SUITE_MODE="calibrate-only"
    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    jq -n '{task_id:"t1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"pending", dependencies:[], params:{}}' \
        > "${QUEUE_DIR}/pending/t1.task"

    check_dependencies_met() { return 0; }
    local type_calls_file="${TEST_TMPDIR}/type_calls"
    echo "0" > "${type_calls_file}"
    get_task_type() {
        local count
        count="$(cat "${type_calls_file}")"
        count=$((count + 1))
        echo "${count}" > "${type_calls_file}"
        if [[ ${count} -eq 1 ]]; then
            echo "SETUP_BASELINE"
        else
            echo "EVAL_BASELINE"
        fi
    }

    assert_eq "0" "$(resolve_dependencies)" "second-pass disallowed task skipped"
    assert_file_exists "${QUEUE_DIR}/pending/t1.task" "task remains pending after second-pass skip"
}

test_generate_model_tasks_branch_coverage() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local calls="${TEST_TMPDIR}/calls"
    : > "${calls}"
    add_task() {
        local task_type="$1"
        printf '%s\n' "${task_type}" >> "${calls}"
        local count
        count=$(wc -l < "${calls}" | tr -d ' ')
        echo "t${count}"
    }
    estimate_model_memory() { echo "14"; }
    generate_eval_certify_tasks() { :; }
    generate_edit_tasks() { :; }

    PACK_USE_BATCH_EDITS="true"
    CLEAN_EDIT_RUNS="1"
    STRESS_EDIT_RUNS="1"
    DRIFT_CALIBRATION_RUNS=1
    RUN_ERROR_INJECTION="true"
    generate_model_tasks "1" "org/model" "model" >/dev/null

    CLEAN_EDIT_RUNS="-1"
    STRESS_EDIT_RUNS="-1"
    generate_model_tasks "2" "org/model" "model" >/dev/null

    PACK_USE_BATCH_EDITS="false"
    CLEAN_EDIT_RUNS=2
    STRESS_EDIT_RUNS=2
    DRIFT_CALIBRATION_RUNS=1
    RUN_ERROR_INJECTION="true"
    generate_model_tasks "3" "org/model" "model" >/dev/null

    DRIFT_CALIBRATION_RUNS=0
    PACK_PRESET_READY="true"
    RUN_ERROR_INJECTION="true"
    generate_model_tasks "4" "org/model" "model" >/dev/null

    PACK_PRESET_READY="false"
    RUN_ERROR_INJECTION="false"
    generate_model_tasks "5" "org/model" "model" >/dev/null
}


test_generate_model_tasks_additional_batch_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/queue_manager.sh"

    local calls="${TEST_TMPDIR}/calls_extra"
    : > "${calls}"
    add_task() {
        local task_type="$1"
        printf '%s\n' "${task_type}" >> "${calls}"
        local count
        count=$(wc -l < "${calls}" | tr -d ' ')
        echo "t${count}"
    }
    estimate_model_memory() { echo "14"; }
    generate_eval_certify_tasks() { :; }
    generate_edit_tasks() { :; }

    PACK_USE_BATCH_EDITS="true"
    CLEAN_EDIT_RUNS=1
    STRESS_EDIT_RUNS=1
    DRIFT_CALIBRATION_RUNS=1
    RUN_ERROR_INJECTION="true"
    generate_model_tasks "1" "org/model" "model" >/dev/null
    assert_match "CALIBRATION_RUN" "$(cat "${calls}")" "calibration task created"
    local error_count
    error_count="$(awk '/^CERTIFY_ERROR$/ {c++} END {print c+0}' "${calls}")"
    assert_eq "5" "${error_count}" "certify error tasks created"

    : > "${calls}"
    CLEAN_EDIT_RUNS=""
    STRESS_EDIT_RUNS=1
    DRIFT_CALIBRATION_RUNS=1
    RUN_ERROR_INJECTION="false"
    generate_model_tasks "2" "org/model" "model" >/dev/null

    : > "${calls}"
    CLEAN_EDIT_RUNS="-1"
    STRESS_EDIT_RUNS=1
    DRIFT_CALIBRATION_RUNS=1
    generate_model_tasks "3" "org/model" "model" >/dev/null

    : > "${calls}"
    CLEAN_EDIT_RUNS=1
    STRESS_EDIT_RUNS=""
    DRIFT_CALIBRATION_RUNS=1
    generate_model_tasks "4" "org/model" "model" >/dev/null

    : > "${calls}"
    CLEAN_EDIT_RUNS=1
    STRESS_EDIT_RUNS="-1"
    DRIFT_CALIBRATION_RUNS=1
    generate_model_tasks "5" "org/model" "model" >/dev/null

    : > "${calls}"
    CLEAN_EDIT_RUNS=1
    STRESS_EDIT_RUNS=1
    DRIFT_CALIBRATION_RUNS=0
    RUN_ERROR_INJECTION="true"
    generate_model_tasks "6" "org/model" "model" >/dev/null
}
