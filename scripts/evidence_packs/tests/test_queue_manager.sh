#!/usr/bin/env bash

test_acquire_queue_lock_recovers_stale_owner_pid() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local t1
    t1="$(add_task "SETUP_BASELINE" "org/model" "model" "14" "none" '{}' "50")"

    # pending -> ready
    assert_eq "1" "$(resolve_dependencies)" "no-deps task moved to ready"
    assert_file_exists "${QUEUE_DIR}/ready/${t1}.task" "task in ready"

    # ready -> running
    run claim_task "${t1}" "2"
    assert_rc "0" "${RUN_RC}" "claim_task ok"
    assert_file_exists "${QUEUE_DIR}/running/${t1}.task" "task in running"
    assert_eq "running" "$(jq -r '.status' "${QUEUE_DIR}/running/${t1}.task")" "status running"
    assert_eq "2" "$(jq -r '.assigned_gpus' "${QUEUE_DIR}/running/${t1}.task")" "assigned_gpus set"

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
    assert_eq "null" "$(jq -r '.assigned_gpus' "${QUEUE_DIR}/ready/${t2}.task")" "assigned_gpus reset"
}

test_resolve_dependencies_filters_non_calibration_tasks_in_calibration_only_mode() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    export PACK_SUITE_MODE="calibrate-only"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local setup_id
    setup_id="$(add_task "SETUP_BASELINE" "org/model" "model" "14" "none" '{}' "50")"
    local eval_id
    eval_id="$(add_task "EVAL_BASELINE" "org/model" "model" "14" "${setup_id}" '{}' "50")"

    assert_eq "1" "$(resolve_dependencies)" "setup task moved to ready"
    assert_file_exists "${QUEUE_DIR}/ready/${setup_id}.task" "setup ready"
    assert_file_exists "${QUEUE_DIR}/pending/${eval_id}.task" "eval remains pending until deps met"

    claim_task "${setup_id}" "0" >/dev/null
    complete_task "${setup_id}" >/dev/null

    assert_eq "0" "$(resolve_dependencies)" "eval not moved to ready under calibration-only"
    assert_file_exists "${QUEUE_DIR}/pending/${eval_id}.task" "eval still pending"
    [[ ! -f "${QUEUE_DIR}/ready/${eval_id}.task" ]] || t_fail "eval should not be ready under calibration-only"
}

test_demote_ready_tasks_for_calibration_only_moves_disallowed_ready_to_pending() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local task_id="tmax"
    write_queue_task failed "${task_id}" SETUP_BASELINE n '{"retries":3, "error_msg":"x"}'

    run retry_task "${task_id}"
    assert_rc "1" "${RUN_RC}" "exceeded retries fails"
    assert_match 'exceeded max retries' "${RUN_ERR}" "message"
}

test_retry_task_sanitizes_missing_retry_fields() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local task_id="tmissing"
    write_queue_task failed "${task_id}" SETUP_BASELINE n '{"retries":null, "max_retries":null, "error_msg":"x"}'

    run retry_task "${task_id}"
    assert_rc "0" "${RUN_RC}" "retry_task succeeds with null retry fields"
    assert_file_exists "${QUEUE_DIR}/ready/${task_id}.task" "task moved to ready"
    assert_eq "1" "$(jq -r '.retries' "${QUEUE_DIR}/ready/${task_id}.task")" "retries incremented from default"
}

test_cancel_tasks_with_failed_dependencies_moves_pending_to_failed() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    # Create failed dep task.
    write_queue_task failed dep SETUP_BASELINE d '{"error_msg":"x"}'

    # Create pending task that depends on failed dep.
    write_queue_task pending child EVAL_BASELINE c '{"dependencies":["dep"]}'

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    write_queue_task ready a

    local tasks
    tasks="$(get_tasks_by_status "ready")"
    assert_match 'ready/a\.task' "${tasks}" "get_tasks_by_status lists tasks"

    assert_eq "0" "$(count_tasks "no_such_status")" "missing status dir counts as 0"
}

test_mark_task_ready_and_claim_lock_timeout_validation() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    write_queue_task pending t1

    mark_task_ready "t1"
    assert_file_exists "${QUEUE_DIR}/ready/t1.task" "mark_task_ready moves file"

    QUEUE_CLAIM_LOCK_TIMEOUT="bad"
    claim_task "t1" "2"
    assert_file_exists "${QUEUE_DIR}/running/t1.task" "claim_task moves to running"
}

test_complete_fail_and_retry_missing_file_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local rc=0
    rc=0; complete_task "missing" || rc=$?; assert_rc "1" "${rc}" "complete_task fails when missing"
    rc=0; fail_task "missing" "boom" || rc=$?; assert_rc "1" "${rc}" "fail_task fails when missing"

    local task_id="race"
    write_queue_task failed "${task_id}" SETUP_BASELINE n '{"error_msg":"x"}'

    acquire_queue_lock() { rm -f "${QUEUE_DIR}/failed/${task_id}.task"; return 0; }
    release_queue_lock() { return 0; }

    rc=0
    retry_task "${task_id}" || rc=$?
    assert_rc "1" "${rc}" "retry_task returns 1 when file disappears under lock"
}

test_reclaim_orphaned_tasks_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

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
    write_queue_task running a
    echo "123" > "${QUEUE_DIR}/running/a.pid"

    # Task B: owner mismatch => can_kill false (reservation check).
    write_queue_task running b SETUP_BASELINE n '{"assigned_gpus":"0,1"}'
    echo "" > "${reservation_dir}/gpu_0.lock"

    # Task C: owners match => kill_gpu_processes + release_gpus.
    write_queue_task running c SETUP_BASELINE n '{"assigned_gpus":"0,1"}'
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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    write_queue_task pending child SETUP_BASELINE n '{"dependencies":["dep"]}'

    local rc=0
    check_dependencies_met "${QUEUE_DIR}/pending/child.task" || rc=$?
    assert_rc "1" "${rc}" "dependency not completed"

    write_queue_task completed dep SETUP_BASELINE d

    update_dependents "dep"
    assert_file_exists "${QUEUE_DIR}/ready/child.task" "update_dependents promotes to ready"
}

test_cancel_tasks_with_failed_dependencies_invalid_grace_and_mtime_missing_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }
    _now_epoch() { echo "100"; }
    _file_mtime_epoch() { echo ""; }

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    write_queue_task failed dep SETUP_BASELINE d '{"error_msg":"x"}'

    write_queue_task pending child EVAL_BASELINE c '{"dependencies":["dep"]}'

    assert_eq "1" "$(cancel_tasks_with_failed_dependencies "bad")" "cancels with invalid grace coerced"
    assert_file_exists "${QUEUE_DIR}/failed/child.task" "moved to failed"
}

test_update_progress_state_status_and_percent_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    write_queue_task completed t SETUP_BASELINE n '{"completed_at":"x"}'

    update_progress_state
    assert_match '\"status\": \"completed\"' "$(cat "${out_dir}/state/progress.json")" "completed status"

    rm -f "${QUEUE_DIR}/completed/t.task"
    write_queue_task failed t SETUP_BASELINE n '{"completed_at":"x", "error_msg":"x"}'
    update_progress_state
    assert_match '\"status\": \"completed_with_failures\"' "$(cat "${out_dir}/state/progress.json")" "completed_with_failures status"

    write_queue_task pending child EVAL_BASELINE n '{"dependencies":["t"]}'
    update_progress_state
    assert_match '\"status\": \"blocked_failed_dependencies\"' "$(cat "${out_dir}/state/progress.json")" "blocked dependency state"
}

test_find_and_refresh_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    write_queue_task pending m1 SETUP_BASELINE m1

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local seq=0
    add_task() { seq=$((seq + 1)); echo "t${seq}"; }
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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

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

    DRIFT_CALIBRATION_RUNS="nope"
    generate_model_tasks "1" "org/model" "model" >/dev/null
    local cal_count
    cal_count="$(awk '/^CALIBRATION_RUN$/ {c++} END {print c+0}' "${calls}")"
    assert_eq "5" "${cal_count}" "invalid calibration runs default to 5"
}

test_capture_add_task_advances_sequence_without_subshell_loss() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    export TASK_SEQUENCE=0

    local t1=""
    local t2=""
    capture_add_task t1 "SETUP_BASELINE" "org/model" "model" "14" "none" '{}' "50"
    capture_add_task t2 "CALIBRATION_RUN" "org/model" "model" "14" "${t1}" '{"run":1}' "50"

    assert_match '_001_' "${t1}" "first captured task uses sequence 001"
    assert_match '_002_' "${t2}" "second captured task increments sequence"
    assert_eq "2" "${TASK_SEQUENCE}" "task sequence preserved in parent shell"
}

test_capture_add_task_error_paths() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local captured=""
    mktemp() { return 1; }
    local rc=0
    set +e
    capture_add_task captured "SETUP_BASELINE" "org/model" "model" "14" "none" '{}' "50"
    rc=$?
    set -e
    assert_rc "1" "${rc}" "capture_add_task fails when temp file creation fails"
    unset -f mktemp

    add_task() { return 7; }
    run capture_add_task captured "SETUP_BASELINE" "org/model" "model" "14" "none" '{}' "50"
    assert_rc "7" "${RUN_RC}" "capture_add_task propagates add_task failures"

    add_task() { return 0; }
    run capture_add_task captured "SETUP_BASELINE" "org/model" "model" "14" "none" '{}' "50"
    assert_rc "1" "${RUN_RC}" "capture_add_task rejects empty add_task output"
}

test_generate_model_tasks_disables_batch_for_large_memory_and_uses_manifest_fallbacks() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

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

    # preset_ready=1 should be normalized to true so evaluate tasks are emitted even with DRIFT_CALIBRATION_RUNS=0.
    assert_match "evaluate_EDIT\\|" "${all_calls}" "evaluate tasks emitted when preset_ready=1"

    # scenarios.json + jq fallback defaults should be used.
    assert_match "quant_rtn:clean:ffn" "${all_calls}" "fallback clean edit spec used"
    assert_match "lora_merge:clean:attn" "${all_calls}" "fallback LoRA clean edit spec used"
    assert_match "fine_tune:clean:ffn" "${all_calls}" "fallback fine-tune clean edit spec used"
    assert_match "lora_merge:8:64:all" "${all_calls}" "fallback LoRA stress edit spec used"
    assert_match "fine_tune:0.0005:3:all" "${all_calls}" "fallback fine-tune stress edit spec used"
    assert_match "weight_tying_break" "${all_calls}" "fallback error type used"
}


test_generate_model_tasks_nonbatch_edit_dependencies_match_create_specs() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    estimate_model_memory() { echo "14"; }
    DRIFT_CALIBRATION_RUNS=1
    CLEAN_EDIT_RUNS=1
    STRESS_EDIT_RUNS=1
    PACK_USE_BATCH_EDITS="false"
    RUN_ERROR_INJECTION="false"
    export DRIFT_CALIBRATION_RUNS CLEAN_EDIT_RUNS STRESS_EDIT_RUNS PACK_USE_BATCH_EDITS RUN_ERROR_INJECTION

    mkdir -p "${out_dir}/state"
    cat > "${out_dir}/state/scenarios.json" <<'EOF'
{
  "scenarios": [
    {"id": "quant_clean", "generation": {"kind": "edit", "edit_spec": "quant_rtn:clean:ffn", "version": "clean"}},
    {"id": "fp8_clean", "generation": {"kind": "edit", "edit_spec": "fp8_quant:clean:ffn", "version": "clean"}},
    {"id": "prune_clean", "generation": {"kind": "edit", "edit_spec": "magnitude_prune:clean:ffn", "version": "clean"}},
    {"id": "svd_clean", "generation": {"kind": "edit", "edit_spec": "lowrank_svd:clean:ffn", "version": "clean"}},
    {"id": "lora_clean", "generation": {"kind": "edit", "edit_spec": "lora_merge:clean:attn", "version": "clean"}},
    {"id": "fine_clean", "generation": {"kind": "edit", "edit_spec": "fine_tune:clean:ffn", "version": "clean"}},
    {"id": "prune_stress", "generation": {"kind": "edit", "edit_spec": "magnitude_prune:0.5:all", "version": "stress"}},
    {"id": "svd_stress", "generation": {"kind": "edit", "edit_spec": "lowrank_svd:32:all", "version": "stress"}},
    {"id": "lora_stress", "generation": {"kind": "edit", "edit_spec": "lora_merge:8:64:all", "version": "stress"}},
    {"id": "fine_stress", "generation": {"kind": "edit", "edit_spec": "fine_tune:0.0005:3:all", "version": "stress"}}
  ]
}
EOF

    generate_model_tasks "1" "org/model" "model" >/dev/null

    local create_count
    create_count="$(find "${QUEUE_DIR}" -type f -name '*_CREATE_EDIT_*.task' | wc -l | tr -d ' ')"
    assert_eq "10" "${create_count}" "all requested non-batch create tasks emitted"

    local duplicate_ids
    duplicate_ids="$(
        find "${QUEUE_DIR}" -type f -name '*_CREATE_EDIT_*.task' -exec jq -r '.task_id' {} \; \
            | sort | uniq -d
    )"
    assert_eq "" "${duplicate_ids}" "create task ids remain unique"

    while IFS= read -r eval_task; do
        [[ -n "${eval_task}" ]] || continue
        local edit_spec
        local version
        local create_dep
        edit_spec="$(jq -r '.params.edit_spec' "${eval_task}")"
        version="$(jq -r '.params.version' "${eval_task}")"
        create_dep="$(jq -r '.dependencies[0]' "${eval_task}")"
        [[ -n "${create_dep}" && "${create_dep}" != "null" ]] || t_fail "missing create dependency for ${eval_task}"
        local create_task="${QUEUE_DIR}/pending/${create_dep}.task"
        assert_file_exists "${create_task}" "create dependency task exists for ${edit_spec}"
        assert_eq "${edit_spec}" "$(jq -r '.params.edit_spec' "${create_task}")" "edit dependency matches edit spec"
        assert_eq "${version}" "$(jq -r '.params.version' "${create_task}")" "edit dependency matches version"
    done < <(find "${QUEUE_DIR}" -type f -name '*_evaluate_EDIT_*.task' | sort)
}

test_generate_model_tasks_adds_eager_baseline_report_dependency() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    estimate_model_memory() { echo "14"; }
    DRIFT_CALIBRATION_RUNS=0
    CLEAN_EDIT_RUNS=1
    STRESS_EDIT_RUNS=0
    PACK_PRESET_READY=1
    PACK_USE_BATCH_EDITS="false"
    RUN_ERROR_INJECTION="false"
    export DRIFT_CALIBRATION_RUNS CLEAN_EDIT_RUNS STRESS_EDIT_RUNS PACK_PRESET_READY PACK_USE_BATCH_EDITS RUN_ERROR_INJECTION

    generate_model_tasks "1" "org/model" "model" >/dev/null

    local baseline_task
    baseline_task="$(find "${QUEUE_DIR}" -type f -name '*_SETUP_EVALUATE_BASELINE_REPORT_*.task' | sort | head -1)"
    assert_file_exists "${baseline_task}" "eager baseline report task created"

    local baseline_id
    baseline_id="$(jq -r '.task_id' "${baseline_task}")"
    local setup_dep
    setup_dep="$(jq -r '.dependencies[0]' "${baseline_task}")"
    [[ -n "${setup_dep}" && "${setup_dep}" != "null" ]] || t_fail "baseline report task missing setup dependency"

    while IFS= read -r eval_task; do
        [[ -n "${eval_task}" ]] || continue
        local has_dep
        has_dep="$(jq -r --arg id "${baseline_id}" '(.dependencies // []) | index($id) != null' "${eval_task}")"
        assert_eq "true" "${has_dep}" "evaluate task waits for eager baseline report"
    done < <(find "${QUEUE_DIR}" -type f -name '*_evaluate_EDIT_*.task' | sort)
}

test_generate_model_tasks_can_disable_eager_baseline_report_dependency() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    estimate_model_memory() { echo "14"; }
    DRIFT_CALIBRATION_RUNS=0
    CLEAN_EDIT_RUNS=1
    STRESS_EDIT_RUNS=0
    PACK_PRESET_READY=1
    PACK_USE_BATCH_EDITS="false"
    PACK_EAGER_BASELINE_REPORT=0
    RUN_ERROR_INJECTION="false"
    export DRIFT_CALIBRATION_RUNS CLEAN_EDIT_RUNS STRESS_EDIT_RUNS PACK_PRESET_READY PACK_USE_BATCH_EDITS PACK_EAGER_BASELINE_REPORT RUN_ERROR_INJECTION

    generate_model_tasks "1" "org/model" "model" >/dev/null

    local baseline_count
    baseline_count="$(find "${QUEUE_DIR}" -type f -name '*_SETUP_EVALUATE_BASELINE_REPORT_*.task' | wc -l | tr -d ' ')"
    assert_eq "0" "${baseline_count}" "eager baseline report task disabled"
}

test_generate_evaluate_tasks_sanitizes_cert_runs() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

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
    cert_count="$(awk '/^evaluate_EDIT$/ {c++} END {print c+0}' "${calls}")"
    assert_eq "0" "${cert_count}" "precondition: no evaluate tasks yet"

    : > "${calls}"
    generate_evaluate_tasks "m" "n" "edit" "preset" "spec" "clean" "bad" >/dev/null
    cert_count="$(awk '/^evaluate_EDIT$/ {c++} END {print c+0}' "${calls}")"
    assert_eq "1" "${cert_count}" "invalid cert_runs defaults to 1"

    : > "${calls}"
    generate_evaluate_tasks "m" "n" "edit" "preset" "spec" "clean" "-2" >/dev/null
    cert_count="$(awk '/^evaluate_EDIT$/ {c++} END {print c+0}' "${calls}")"
    assert_eq "0" "${cert_count}" "negative cert_runs clamps to 0"

}

test_generate_all_tasks_and_update_model_task_memory_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

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

    write_queue_task pending t SETUP_BASELINE m1

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

test_update_model_task_memory_preserves_existing_reservation_floor() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    _cmd_python() { echo "96 1"; }

    write_queue_task pending t CALIBRATION_RUN mixtral '{"model_id":"mistralai/Mixtral-8x7B-v0.1", "model_size_gb":480, "required_gpus":4, "params":{"run":1}, "priority":85}'

    mkdir -p "${out_dir}/mixtral"
    local baseline_path="${TEST_TMPDIR}/baseline"
    mkdir -p "${baseline_path}"
    echo "${baseline_path}" > "${out_dir}/mixtral/.baseline_path"
    echo '{}' > "${baseline_path}/model_profile.json"

    update_model_task_memory "mixtral" "${out_dir}" "mistralai/Mixtral-8x7B-v0.1"
    assert_eq "480" "$(jq -r '.model_size_gb' "${QUEUE_DIR}/pending/t.task")" "profile refinement keeps stricter memory floor"
    assert_eq "4" "$(jq -r '.required_gpus' "${QUEUE_DIR}/pending/t.task")" "profile refinement keeps stricter GPU floor"
}

test_update_model_task_memory_allows_single_gpu_downsize_after_refinement() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    _runtime_python() {
        assert_eq "queue_state.py" "${1:-}" "memory refinement uses runtime helper module resolution"
        assert_eq "estimate-task-memory" "${2:-}" "memory refinement invokes estimate-task-memory"
        echo "44 1"
    }

    write_queue_task ready t CALIBRATION_RUN olmo '{"model_id":"allenai/OLMo-2-1124-7B", "model_size_gb":82, "required_gpus":1, "params":{"run":1}, "priority":85}'

    mkdir -p "${out_dir}/olmo"
    local baseline_path="${TEST_TMPDIR}/baseline"
    mkdir -p "${baseline_path}"
    echo "${baseline_path}" > "${out_dir}/olmo/.baseline_path"
    echo '{}' > "${baseline_path}/model_profile.json"

    update_model_task_memory "olmo" "${out_dir}" "allenai/OLMo-2-1124-7B"
    assert_eq "44" "$(jq -r '.model_size_gb' "${QUEUE_DIR}/ready/t.task")" "single-GPU tasks can downsize after refined estimate"
    assert_eq "1" "$(jq -r '.required_gpus' "${QUEUE_DIR}/ready/t.task")" "single-GPU requirement unchanged"
}

test_estimate_task_memory_reserves_full_host_for_moe_execution() {
    mock_reset

    local profile="${TEST_TMPDIR}/profile.json"
    jq -n '{model_id:"mistralai/Mixtral-8x7B-v0.1", weights_gb:90, hidden_size:4096, num_layers:32, num_heads:32, num_kv_heads:8, dtype_bytes:2}' > "${profile}"

    local result
    result="$(TASK_TYPE=CALIBRATION_RUN MODEL_ID="mistralai/Mixtral-8x7B-v0.1" PROFILE_PATH="${profile}" GPU_MEMORY_PER_DEVICE=140 NUM_GPUS=4 python3 "${TEST_ROOT}/scripts/evidence_packs/python/queue_state.py" estimate-task-memory)"
    assert_eq "421 4" "${result}" "MoE calibration reserves the full 4-GPU host"

    result="$(TASK_TYPE=SETUP_BASELINE MODEL_ID="mistralai/Mixtral-8x7B-v0.1" PROFILE_PATH="${profile}" GPU_MEMORY_PER_DEVICE=140 NUM_GPUS=4 python3 "${TEST_ROOT}/scripts/evidence_packs/python/queue_state.py" estimate-task-memory)"
    assert_eq "94 1" "${result}" "MoE baseline setup stays single-GPU sized"
}

test_estimate_task_memory_uses_runtime_sized_7b_windows() {
    mock_reset

    local profile="${TEST_TMPDIR}/profile.json"
    jq -n '{model_id:"allenai/OLMo-2-1124-7B", weights_gb:14, hidden_size:4096, num_layers:32, num_heads:32, num_kv_heads:32, dtype_bytes:2}' > "${profile}"

    local result
    result="$(TASK_TYPE=CALIBRATION_RUN MODEL_ID="allenai/OLMo-2-1124-7B" PROFILE_PATH="${profile}" GPU_MEMORY_PER_DEVICE=80 NUM_GPUS=1 python3 "${TEST_ROOT}/scripts/evidence_packs/python/queue_state.py" estimate-task-memory)"
    assert_eq "44 1" "${result}" "7B calibration memory estimate stays aligned with runtime-sized windows on 80 GB GPUs"
}

test_with_queue_lock_returns_nonzero_when_lock_acquire_fails() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    failing_action() { return 42; }
    run with_queue_lock failing_action
    assert_rc "42" "${RUN_RC}" "with_queue_lock returns action rc"
}

test_acquire_queue_lock_sleeps_when_lock_held_by_live_owner() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local task_id="t1"
    write_queue_task failed "${task_id}" SETUP_BASELINE n '{"error_msg":"x"}'

    local pending_id="t2"
    write_queue_task pending "${pending_id}"

    assert_match 'QUEUE STATUS' "$(print_queue_stats)" "queue status header"
    ! is_queue_empty
    ! is_queue_complete

    rm -f "${QUEUE_DIR}/pending/${pending_id}.task"
    is_queue_empty

    rm -f "${QUEUE_DIR}/failed/${task_id}.task"
    is_queue_complete
}

test_queue_terminal_state_reports_blocked_failed_dependencies() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    write_queue_task failed dep SETUP_BASELINE n '{"completed_at":"x", "error_msg":"x"}'
    write_queue_task pending child EVAL_BASELINE n '{"dependencies":["dep"]}'

    assert_eq "1" "$(count_pending_tasks_blocked_by_failed_dependencies)" "blocked pending count"
    assert_eq "blocked_failed_dependencies" "$(queue_terminal_state)" "queue reports blocked terminal state"
}

test_mark_task_ready_and_claim_task_return_nonzero_when_source_missing() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    write_queue_task pending t1

    update_task_status() { return 1; }

    run mark_task_ready "t1"
    assert_rc "1" "${RUN_RC}" "mark_task_ready returns error when update_task_status fails"
    assert_file_exists "${QUEUE_DIR}/pending/t1.task" "task remains pending"
    [[ ! -f "${QUEUE_DIR}/ready/t1.task" ]] || t_fail "expected task not moved to ready on status update failure"
}

test_check_dependencies_met_returns_nonzero_when_task_json_is_invalid() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    echo '{not-json' > "${QUEUE_DIR}/pending/bad.task"
    run check_dependencies_met "${QUEUE_DIR}/pending/bad.task"
    assert_rc "1" "${RUN_RC}" "invalid task json treated as unmet dependencies"
}

test_check_dependencies_met_returns_nonzero_when_task_file_missing() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    run check_dependencies_met "${QUEUE_DIR}/pending/missing.task"
    assert_rc "1" "${RUN_RC}" "missing task file treated as unmet dependencies"
}

test_generate_evaluate_tasks_create_expected_tasks() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }
    estimate_model_memory() { echo "14"; }

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local out
    out="$(generate_evaluate_tasks "org/model" "m" "edit1" "preset1" "quant_rtn:8:128:ffn" "clean" "1")"
    assert_match 'Created: ' "${out}" "creates evaluate tasks"

    local pending_count
    pending_count="$(ls "${QUEUE_DIR}/pending"/*.task 2>/dev/null | wc -l | tr -d ' ')"
    assert_eq "1" "${pending_count}" "creates 1 evaluate task"

    local task_file task_type version
    for task_file in "${QUEUE_DIR}/pending"/*.task; do
        task_type="$(jq -r '.task_type' "${task_file}")"
        assert_eq "evaluate_EDIT" "${task_type}" "evaluate task type"
        version="$(jq -r '.params.version // ""' "${task_file}")"
        assert_eq "clean" "${version}" "evaluate task carries version hint"
    done

}

test_task_ops_short_circuit_when_lock_acquire_fails() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    write_queue_task failed t1 SETUP_BASELINE n '{"error_msg":"x"}'

    acquire_queue_lock() { return 1; }
    run retry_task "t1"
    assert_rc "1" "${RUN_RC}" "retry_task returns non-zero when lock unavailable"
}

test_retry_task_returns_nonzero_when_task_missing() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    run retry_task "no_such_task"
    assert_rc "1" "${RUN_RC}" "retry_task returns non-zero when task file is missing"
}

test_retry_task_atomic_update_failure_triggers_error_block() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    write_queue_task failed t1 SETUP_BASELINE n '{"error_msg":"x"}'

    _runtime_python() {
        if [[ "${1:-}" == "queue_state.py" && "${2:-}" == "retry-task" ]]; then
            return 1
        fi
        command "${TEST_REAL_PYTHON3}" "${TEST_ROOT}/scripts/evidence_packs/python/${1}" "${@:2}"
    }

    run retry_task "t1"
    assert_rc "1" "${RUN_RC}" "queue-state update failure triggers error path"
}

test_retry_task_move_failure_returns_error() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    write_queue_task failed t1 SETUP_BASELINE n '{"error_msg":"x"}'

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    mv() { return 1; }

    run update_progress_state
    assert_rc "1" "${RUN_RC}" "mv failure returns non-zero"
}

test_queue_manager_find_task_returns_path() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    write_queue_task ready t1

    assert_eq "${QUEUE_DIR}/ready/t1.task" "$(find_task "t1")" "find_task returns path"

    run find_task "missing"
    assert_rc "1" "${RUN_RC}" "find_task returns non-zero when missing"
    assert_eq "" "${RUN_OUT}" "find_task returns empty output when missing"
}

test_queue_manager_resolve_dependencies_skips_disallowed_tasks() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    export PACK_SUITE_MODE="calibrate-only"
    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    write_queue_task pending t1 EVAL_BASELINE

    check_dependencies_met() { return 0; }

    assert_eq "0" "$(resolve_dependencies)" "disallowed task skipped"
    assert_file_exists "${QUEUE_DIR}/pending/t1.task" "task stays pending"
}


test_queue_manager_resolve_dependencies_skips_on_second_pass_after_type_change() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    export PACK_SUITE_MODE="calibrate-only"
    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    write_queue_task pending t1

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

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
    generate_eval_evaluate_tasks() { :; }
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

test_generate_model_tasks_defaults_error_types_when_manifest_missing_errors() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local calls="${TEST_TMPDIR}/calls_default_errors"
    : > "${calls}"
    add_task() {
        local task_type="$1"
        local params="${6:-}"
        printf '%s\t%s\n' "${task_type}" "${params}" >> "${calls}"
        local count
        count=$(wc -l < "${calls}" | tr -d ' ')
        echo "t${count}"
    }
    estimate_model_memory() { echo "14"; }
    generate_eval_evaluate_tasks() { :; }
    jq() { :; }

    # Point SCRIPT_DIR at a temporary pack root that has a manifest without any
    # error-injection scenarios so generate_model_tasks falls back to defaults.
    local pack_root="${TEST_TMPDIR}/pack_no_errors"
    mkdir -p "${pack_root}/lib"
    SCRIPT_DIR="${pack_root}/lib"

    cat > "${pack_root}/scenarios.json" <<'EOF'
{
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {
      "id": "quant_4bit_clean",
      "generation": {"kind": "edit", "edit_spec": "quant_rtn:clean:ffn", "version": "clean"}
    },
    {
      "id": "quant_4bit_stress",
      "generation": {"kind": "edit", "edit_spec": "quant_rtn:4:32:all", "version": "stress"}
    }
  ]
}
EOF

    PACK_USE_BATCH_EDITS="true"
    CLEAN_EDIT_RUNS="1"
    STRESS_EDIT_RUNS="1"
    DRIFT_CALIBRATION_RUNS=1
    RUN_ERROR_INJECTION="true"
    generate_model_tasks "1" "org/model" "model" >/dev/null

    assert_match "CREATE_ERROR" "$(cat "${calls}")" "default error tasks created"
    assert_match "\"error_type\"[[:space:]]*:[[:space:]]*\"nan_injection\"" "$(cat "${calls}")" "includes nan injection"
    assert_match "\"error_type\"[[:space:]]*:[[:space:]]*\"inf_injection\"" "$(cat "${calls}")" "includes inf injection"

    local created
    created="$(grep -c '^CREATE_ERROR' "${calls}" || true)"
    assert_eq "9" "${created}" "fallback creates 9 default error injections"
}

test_generate_model_tasks_state_manifest_without_errors_does_not_fallback_to_defaults() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local calls="${TEST_TMPDIR}/calls_state_manifest_no_errors"
    : > "${calls}"
    add_task() {
        local task_type="$1"
        local params="${6:-}"
        printf '%s\t%s\n' "${task_type}" "${params}" >> "${calls}"
        local count
        count=$(wc -l < "${calls}" | tr -d ' ')
        echo "t${count}"
    }
    estimate_model_memory() { echo "14"; }
    generate_eval_evaluate_tasks() { :; }

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null
    mkdir -p "${out_dir}/state"
    cat > "${out_dir}/state/scenarios.json" <<'EOF'
{
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {
      "id": "quant_4bit_clean",
      "generation": {"kind": "edit", "edit_spec": "quant_rtn:clean:ffn", "version": "clean"}
    },
    {
      "id": "quant_4bit_stress",
      "generation": {"kind": "edit", "edit_spec": "quant_rtn:4:32:all", "version": "stress"}
    }
  ]
}
EOF

    PACK_USE_BATCH_EDITS="true"
    CLEAN_EDIT_RUNS="1"
    STRESS_EDIT_RUNS="1"
    DRIFT_CALIBRATION_RUNS=1
    RUN_ERROR_INJECTION="true"
    generate_model_tasks "1" "org/model" "model" >/dev/null

    local created
    created="$(grep -c '^CREATE_ERROR' "${calls}" || true)"
    assert_eq "0" "${created}" "state manifest without errors remains authoritative"
}

test_generate_model_tasks_propagates_error_env_from_manifest() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local calls="${TEST_TMPDIR}/calls_error_env"
    : > "${calls}"
    add_task() {
        local task_type="$1"
        local params="${6:-}"
        printf '%s\t%s\n' "${task_type}" "${params}" >> "${calls}"
        local count
        count=$(wc -l < "${calls}" | tr -d ' ')
        echo "t${count}"
    }
    estimate_model_memory() { echo "14"; }
    generate_eval_evaluate_tasks() { :; }
    local pack_root="${TEST_TMPDIR}/pack_error_env"
    mkdir -p "${pack_root}/lib"
    SCRIPT_DIR="${pack_root}/lib"

    cat > "${pack_root}/scenarios.json" <<'EOF'
{
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {
      "id": "quant_4bit_clean",
      "generation": {"kind": "edit", "edit_spec": "quant_rtn:clean:ffn", "version": "clean"}
    },
    {
      "id": "quant_4bit_stress",
      "generation": {"kind": "edit", "edit_spec": "quant_rtn:3:32:all", "version": "stress"}
    },
    {
      "id": "rmt_norm_noise",
      "generation": {
        "kind": "error",
        "error_type": "rmt_norm_noise",
        "env": {
          "INVARLOCK_RMT_PROBE_MODE": "anisotropy",
          "INVARLOCK_RMT_ANISO_BLEND": "0.75"
        }
      }
    }
  ]
}
EOF

    PACK_USE_BATCH_EDITS="true"
    CLEAN_EDIT_RUNS="1"
    STRESS_EDIT_RUNS="1"
    DRIFT_CALIBRATION_RUNS=1
    RUN_ERROR_INJECTION="true"
    generate_model_tasks "1" "org/model" "model" >/dev/null

    local create_params
    create_params="$(awk -F '\t' '$1=="CREATE_ERROR"{print $2; exit}' "${calls}")"
    assert_match '"error_type":"rmt_norm_noise"' "${create_params}" "error_type retained"
    assert_eq "anisotropy" "$(printf '%s' "${create_params}" | jq -r '.error_env.INVARLOCK_RMT_PROBE_MODE')" "env mode propagated"
    assert_eq "0.75" "$(printf '%s' "${create_params}" | jq -r '.error_env.INVARLOCK_RMT_ANISO_BLEND')" "env blend propagated"
}

test_generate_model_tasks_applies_model_specific_error_env_overrides() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local calls="${TEST_TMPDIR}/calls_error_env_override"
    : > "${calls}"
    add_task() {
        local task_type="$1"
        local params="${6:-}"
        printf '%s\t%s\n' "${task_type}" "${params}" >> "${calls}"
        local count
        count=$(wc -l < "${calls}" | tr -d ' ')
        echo "t${count}"
    }
    estimate_model_memory() { echo "14"; }
    generate_eval_evaluate_tasks() { :; }
    local pack_root="${TEST_TMPDIR}/pack_error_env_override"
    mkdir -p "${pack_root}/lib"
    SCRIPT_DIR="${pack_root}/lib"

    cat > "${pack_root}/scenarios.json" <<'EOF'
{
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {
      "id": "quant_4bit_clean",
      "generation": {"kind": "edit", "edit_spec": "quant_rtn:clean:ffn", "version": "clean"}
    },
    {
      "id": "quant_4bit_stress",
      "generation": {"kind": "edit", "edit_spec": "quant_rtn:3:32:all", "version": "stress"}
    },
    {
      "id": "rmt_norm_noise",
      "generation": {
        "kind": "error",
        "error_type": "rmt_norm_noise",
        "env": {
          "INVARLOCK_RMT_PROBE_MODE": "anisotropy",
          "INVARLOCK_RMT_ANISO_BLEND": "0.75",
          "INVARLOCK_RMT_ANISO_MAX_PARAMS": "10"
        },
        "env_by_model": {
          "org/model": {
            "INVARLOCK_RMT_ANISO_BLEND": "0.60",
            "INVARLOCK_RMT_ANISO_MAX_PARAMS": "4"
          }
        }
      }
    }
  ]
}
EOF

    PACK_USE_BATCH_EDITS="true"
    CLEAN_EDIT_RUNS="1"
    STRESS_EDIT_RUNS="1"
    DRIFT_CALIBRATION_RUNS=1
    RUN_ERROR_INJECTION="true"
    generate_model_tasks "1" "org/model" "model" >/dev/null

    local create_params
    create_params="$(awk -F '\t' '$1=="CREATE_ERROR"{print $2; exit}' "${calls}")"
    assert_eq "anisotropy" "$(printf '%s' "${create_params}" | jq -r '.error_env.INVARLOCK_RMT_PROBE_MODE')" "base env key preserved"
    assert_eq "0.60" "$(printf '%s' "${create_params}" | jq -r '.error_env.INVARLOCK_RMT_ANISO_BLEND')" "model-specific blend override applied"
    assert_eq "4" "$(printf '%s' "${create_params}" | jq -r '.error_env.INVARLOCK_RMT_ANISO_MAX_PARAMS')" "model-specific max params override applied"
}


test_generate_model_tasks_additional_batch_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

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
    generate_eval_evaluate_tasks() { :; }
    PACK_USE_BATCH_EDITS="true"
    CLEAN_EDIT_RUNS=1
    STRESS_EDIT_RUNS=1
    DRIFT_CALIBRATION_RUNS=1
    RUN_ERROR_INJECTION="true"
    generate_model_tasks "1" "org/model" "model" >/dev/null
    assert_match "CALIBRATION_RUN" "$(cat "${calls}")" "calibration task created"
    local error_count
    error_count="$(awk '/^evaluate_ERROR$/ {c++} END {print c+0}' "${calls}")"
    local expected_error_count
    expected_error_count="$(
        jq '.scenarios | map(select(.generation.kind=="error")) | length' \
            "${TEST_ROOT}/scripts/evidence_packs/scenarios.json"
    )"
    assert_eq "${expected_error_count}" "${error_count}" "evaluate error tasks created"

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

test_generate_model_tasks_honors_one_sided_state_manifest_without_edit_fallback() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local calls="${TEST_TMPDIR}/calls_one_sided_manifest"
    : > "${calls}"
    add_task() {
        local task_type="$1"
        local params="${6:-}"
        printf '%s\t%s\n' "${task_type}" "${params}" >> "${calls}"
        local count
        count=$(wc -l < "${calls}" | tr -d ' ')
        echo "t${count}"
    }
    estimate_model_memory() { echo "14"; }
    generate_eval_evaluate_tasks() { :; }

    local run_root="${TEST_TMPDIR}/run_one_clean"
    export QUEUE_DIR="${run_root}/queue"
    mkdir -p "${QUEUE_DIR}" "${run_root}/state"
    cat > "${run_root}/state/scenarios.json" <<'EOF'
{
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {
      "id": "svd_rank32_clean",
      "generation": {"kind": "edit", "edit_spec": "lowrank_svd:clean:ffn", "version": "clean"}
    }
  ]
}
EOF

    PACK_USE_BATCH_EDITS="false"
    PACK_PRESET_READY="true"
    CLEAN_EDIT_RUNS="1"
    STRESS_EDIT_RUNS="1"
    DRIFT_CALIBRATION_RUNS="0"
    RUN_ERROR_INJECTION="false"

    generate_model_tasks "1" "org/model" "model" >/dev/null

    local create_count
    create_count="$(awk -F '\t' '$1=="CREATE_EDIT"{c++} END {print c+0}' "${calls}")"
    assert_eq "1" "${create_count}" "one clean scenario creates one edit task"
    assert_match "lowrank_svd:clean:ffn" "$(cat "${calls}")" "selected clean edit spec used"

    local all_calls
    all_calls="$(cat "${calls}")"
    if [[ "${all_calls}" == *"quant_rtn:clean:ffn"* || "${all_calls}" == *"fp8_quant:clean:ffn"* || "${all_calls}" == *"magnitude_prune:clean:ffn"* || "${all_calls}" == *'"version": "stress"'* || "${all_calls}" == *'"version":"stress"'* ]]; then
        t_fail "one-sided manifest should not fall back to default clean/stress edit sets"
    fi
}

test_generate_model_tasks_prefers_run_state_manifest_and_skips_blank_error_entries() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local calls="${TEST_TMPDIR}/calls_state_manifest"
    : > "${calls}"
    add_task() {
        local task_type="$1"
        local params="${6:-}"
        printf '%s\t%s\n' "${task_type}" "${params}" >> "${calls}"
        local count
        count=$(wc -l < "${calls}" | tr -d ' ')
        echo "t${count}"
    }
    estimate_model_memory() { echo "14"; }
    generate_eval_evaluate_tasks() { :; }

    local run_root="${TEST_TMPDIR}/run_root"
    export QUEUE_DIR="${run_root}/queue"
    mkdir -p "${QUEUE_DIR}" "${run_root}/state"
    cat > "${run_root}/state/scenarios.json" <<'EOF'
{
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {
      "id": "error_blank",
      "generation": {"kind": "error", "error_type": "", "env": {"INVARLOCK_SKIP": "1"}}
    },
    {
      "id": "error_real",
      "generation": {
        "kind": "error",
        "error_type": "rmt_norm_noise",
        "env": {"INVARLOCK_RMT_PROBE_MODE": "anisotropy"}
      }
    }
  ]
}
EOF

    PACK_PRESET_READY="true"
    RUN_ERROR_INJECTION="true"
    CLEAN_EDIT_RUNS="bad"
    STRESS_EDIT_RUNS="also-bad"

    generate_model_tasks "1" "org/model" "model" >/dev/null

    local create_count
    create_count="$(awk -F '\t' '$1=="CREATE_ERROR"{c++} END {print c+0}' "${calls}")"
    assert_eq "1" "${create_count}" "blank manifest error entries are skipped"

    local create_params
    create_params="$(awk -F '\t' '$1=="CREATE_ERROR"{print $2; exit}' "${calls}")"
    assert_eq "rmt_norm_noise" "$(printf '%s' "${create_params}" | jq -r '.error_type')" "state manifest error type used"
    assert_eq "anisotropy" "$(printf '%s' "${create_params}" | jq -r '.error_env.INVARLOCK_RMT_PROBE_MODE')" "state manifest env propagated"
}

test_generate_model_tasks_falls_back_to_plain_error_json_when_jq_binary_missing() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local calls="${TEST_TMPDIR}/calls_no_jq"
    : > "${calls}"
    add_task() {
        local task_type="$1"
        local params="${6:-}"
        printf '%s\t%s\n' "${task_type}" "${params}" >> "${calls}"
        local count
        count=$(wc -l < "${calls}" | tr -d ' ')
        echo "t${count}"
    }
    estimate_model_memory() { echo "14"; }
    generate_eval_evaluate_tasks() { :; }
    type() { return 1; }

    PACK_PRESET_READY="true"
    RUN_ERROR_INJECTION="true"
    CLEAN_EDIT_RUNS="0"
    STRESS_EDIT_RUNS="0"

    generate_model_tasks "1" "org/model" "model" >/dev/null

    local create_params
    create_params="$(awk -F '\t' '$1=="CREATE_ERROR"{print $2; exit}' "${calls}")"
    assert_match '"error_type": "nan_injection"' "${create_params}" "fallback params use stringified JSON"
    if [[ "${create_params}" == *'"error_env"'* ]]; then
        t_fail "fallback params should omit error_env when jq is unavailable"
    fi

    unset -f type
}

test_queue_manager_terminal_state_covers_completed_variants() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    get_queue_stats() { echo "0:0:0:2:0:2"; }
    assert_eq "completed" "$(queue_terminal_state)" "no failures yields completed terminal state"

    get_queue_stats() { echo "0:0:0:1:1:2"; }
    assert_eq "completed_with_failures" "$(queue_terminal_state)" "failures yield completed_with_failures terminal state"

    get_queue_stats() { echo "0:1:1:0:0:2"; }
    run queue_terminal_state
    assert_rc "1" "${RUN_RC}" "active queue is not terminal"
}

test_queue_manager_direct_module_source_guards() {
    mock_reset

    (
        # shellcheck source=../queue_dependencies.sh
        source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_dependencies.sh"
        declare -F check_dependencies_met >/dev/null
        declare -F mark_task_ready >/dev/null
    )

    (
        # shellcheck source=../queue_generation.sh
        source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_generation.sh"
        declare -F generate_all_tasks >/dev/null
        declare -F check_dependencies_met >/dev/null
    )
}

test_queue_lock_stale_owner_reads_owner_and_removes_lock() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    local lock_dir="${QUEUE_DIR}/queue.lock.d"
    mkdir -p "${lock_dir}"
    printf '%s\n' "999999" > "${lock_dir}/owner"

    _pid_is_alive() { return 1; }
    _now_epoch() { echo "100"; }

    acquire_queue_lock 1
    assert_dir_exists "${QUEUE_LOCK_DIR}" "lock reacquired after stale owner cleanup"
    release_queue_lock
}

test_queue_lifecycle_move_failures_release_locks() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    write_queue_task ready ready_mv
    write_queue_task running complete_mv
    write_queue_task running fail_mv

    mark_task_started() { return 0; }
    mark_task_completed() { return 0; }
    mark_task_failed() { return 0; }
    update_progress_state() { :; }
    mv() { return 1; }

    run claim_task "ready_mv" "0"
    assert_rc "1" "${RUN_RC}" "claim_task returns non-zero when ready->running move fails"
    [[ -z "${QUEUE_LOCK_DIR:-}" ]] || t_fail "claim_task should release lock after move failure"

    run complete_task "complete_mv"
    assert_rc "1" "${RUN_RC}" "complete_task returns non-zero when running->completed move fails"
    [[ -z "${QUEUE_LOCK_DIR:-}" ]] || t_fail "complete_task should release lock after move failure"

    run fail_task "fail_mv" "boom"
    assert_rc "1" "${RUN_RC}" "fail_task returns non-zero when running->failed move fails"
    [[ -z "${QUEUE_LOCK_DIR:-}" ]] || t_fail "fail_task should release lock after move failure"
}

test_queue_dependency_cancellation_rechecks_age_and_updates_progress() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    _now_epoch() { echo "100"; }
    _file_mtime_epoch() { echo "0"; }
    local progress_updates=0
    update_progress_state() { progress_updates=$((progress_updates + 1)); }

    write_queue_task failed dep SETUP_BASELINE d '{"error_msg":"x"}'
    write_queue_task pending child EVAL_BASELINE c '{"dependencies":["dep"]}'

    run cancel_tasks_with_failed_dependencies 10
    assert_rc "0" "${RUN_RC}" "aged failed dependency cancellation succeeds"
    assert_eq "1" "${RUN_OUT}" "aged failed dependency cancels child"
    assert_file_exists "${QUEUE_DIR}/failed/child.task" "child moved to failed"
    assert_eq "1" "${progress_updates}" "progress updated after cancellation"
}

test_queue_dependency_ready_transition_and_demote_branches_under_calibrate_only() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    export PACK_SUITE_MODE="calibrate-only"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    write_queue_task pending allowed CALIBRATION_RUN
    write_queue_task pending blocked EVAL_BASELINE

    assert_eq "1" "$(resolve_dependencies)" "only calibration task moved to ready"
    assert_file_exists "${QUEUE_DIR}/ready/allowed.task" "allowed task moved to ready"
    assert_file_exists "${QUEUE_DIR}/pending/blocked.task" "blocked task stays pending"

    mv "${QUEUE_DIR}/pending/blocked.task" "${QUEUE_DIR}/ready/blocked.task"
    update_task_status "${QUEUE_DIR}/ready/blocked.task" "ready"
    demote_ready_tasks_for_calibration_only
    assert_file_exists "${QUEUE_DIR}/pending/blocked.task" "disallowed ready task demoted"
}

test_queue_generation_error_paths_and_skip_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    _runtime_python() { return 1; }
    run update_progress_state
    assert_rc "1" "${RUN_RC}" "progress update returns non-zero when queue_state helper fails"

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
    resolve_dependencies() { echo "0"; }
    update_progress_state() { :; }
    print_queue_stats() { :; }

    DRIFT_CALIBRATION_RUNS=0
    PACK_PRESET_READY=false
    CLEAN_EDIT_RUNS=1
    STRESS_EDIT_RUNS=0
    RUN_ERROR_INJECTION=false
    export DRIFT_CALIBRATION_RUNS PACK_PRESET_READY CLEAN_EDIT_RUNS STRESS_EDIT_RUNS RUN_ERROR_INJECTION
    assert_match 'Skipping edit creation \(no calibrated preset available\)' "$(generate_model_tasks 1 "org/model" "model")" "no-preset branch reached"

    : > "${calls}"
    PACK_PRESET_READY=true
    CLEAN_EDIT_RUNS=-3
    STRESS_EDIT_RUNS=-2
    assert_match 'Skipping edit creation \(CLEAN_EDIT_RUNS=0, STRESS_EDIT_RUNS=0\)' "$(generate_model_tasks 1 "org/model" "model")" "negative run counts clamp to zero"
}

test_generate_evaluate_and_all_tasks_branch_representatives() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local calls="${TEST_TMPDIR}/calls"
    : > "${calls}"
    add_task() {
        local task_type="$1"
        local deps="$5"
        printf '%s|%s\n' "${task_type}" "${deps}" >> "${calls}"
        local count
        count=$(wc -l < "${calls}" | tr -d ' ')
        echo "t${count}"
    }
    estimate_model_memory() { echo "14"; }

    generate_evaluate_tasks "org/model" "model" "edit1" "preset1" "spec" "clean" "bad" >/dev/null
    assert_match 'evaluate_EDIT\|edit1,preset1' "$(cat "${calls}")" "preset dependency included for invalid cert_runs default"

    : > "${calls}"
    generate_evaluate_tasks "org/model" "model" "edit1" "" "spec" "clean" "-1" >/dev/null
    assert_eq "" "$(cat "${calls}")" "negative cert_runs creates no evaluate tasks"

    local generated=""
    generate_model_tasks() { generated+="$1:$2:$3;"; }
    resolve_dependencies() { echo "0"; }
    update_progress_state() { :; }
    print_queue_stats() { :; }

    generate_all_tasks "" "Org/Model Name" >/dev/null
    assert_match '2:Org/Model Name:org__model_name' "${generated}" "non-empty model id normalized and generated"
}

test_queue_memory_plan_profile_fallbacks_and_model_id_candidates() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local out_dir="${TEST_TMPDIR}/out"
    init_queue "${out_dir}" >/dev/null

    _runtime_python() { echo "33 1"; }

    write_queue_task pending direct SETUP_BASELINE model '{"model_size_gb":10, "required_gpus":1}'
    mkdir -p "${out_dir}/model/models/baseline"
    echo '{}' > "${out_dir}/model/models/baseline/model_profile.json"
    update_model_task_memory "model" "${out_dir}" ""
    assert_eq "33" "$(jq -r '.model_size_gb' "${QUEUE_DIR}/pending/direct.task")" "model-local profile fallback used"

    write_queue_task ready candidate SETUP_BASELINE candidate '{"model_id":"Org/Other Model", "model_size_gb":10, "required_gpus":1}'
    mkdir -p "${out_dir}/models/org__other_model/baseline"
    echo '{}' > "${out_dir}/models/org__other_model/baseline/model_profile.json"
    update_model_task_memory "candidate" "${out_dir}" "Org/Other Model"
    assert_eq "33" "$(jq -r '.model_size_gb' "${QUEUE_DIR}/ready/candidate.task")" "model_id sanitized profile candidate used"

    mkdir -p "${out_dir}/profiled"
    echo "Org/Profiled" > "${out_dir}/profiled/.model_id"
    refresh_task_memory_from_profiles "${out_dir}"
}

test_queue_manager_remaining_core_dependency_and_lifecycle_branches() {
    (
        mock_reset
        # shellcheck source=../queue_manager.sh
        source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

        local pack_root="${TEST_TMPDIR}/pack_root"
        mkdir -p "${pack_root}/tools"
        SCRIPT_DIR="${pack_root}/tools"
        assert_eq "${pack_root}" "$(_pack_queue_pack_root)" "non-queue script dir resolves pack root"

        local captured=""
        local rc=0
        mktemp() { return 1; }
        set +e
        capture_add_task captured "SETUP_BASELINE" "org/model" "model" "14" "none" '{}' "50"
        rc=$?
        set -e
        assert_rc "1" "${rc}" "capture_add_task returns 1 when temp creation fails"
        unset -f mktemp

        add_task() { return 6; }
        run capture_add_task captured "SETUP_BASELINE" "org/model" "model" "14" "none" '{}' "50"
        assert_rc "6" "${RUN_RC}" "capture_add_task propagates add_task rc"

        add_task() { return 0; }
        run capture_add_task captured "SETUP_BASELINE" "org/model" "model" "14" "none" '{}' "50"
        assert_rc "1" "${RUN_RC}" "capture_add_task rejects empty add_task output"
    )

    (
        mock_reset
        # shellcheck source=../queue_manager.sh
        source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

        local out_dir="${TEST_TMPDIR}/dep_out"
        init_queue "${out_dir}" >/dev/null

        run check_dependencies_met "${QUEUE_DIR}/pending/missing.task"
        assert_rc "1" "${RUN_RC}" "missing dependency file returns unmet"

        write_queue_task pending baddeps SETUP_BASELINE n '{"dependencies":["dep"]}'
        get_task_dependencies() { return 1; }
        run check_dependencies_met "${QUEUE_DIR}/pending/baddeps.task"
        assert_rc "1" "${RUN_RC}" "dependency parser failure returns unmet"
    )

    (
        mock_reset
        # shellcheck source=../queue_manager.sh
        source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

        local out_dir="${TEST_TMPDIR}/cancel_out"
        init_queue "${out_dir}" >/dev/null

        _now_epoch() { echo "100"; }
        _file_mtime_epoch() { echo ""; }

        write_queue_task failed dep SETUP_BASELINE d
        write_queue_task pending child EVAL_BASELINE c '{"dependencies":["dep"]}'

        assert_eq "1" "$(cancel_tasks_with_failed_dependencies "not-a-number")" "invalid grace still cancels failed dependency"
        assert_file_exists "${QUEUE_DIR}/failed/child.task" "blocked child moved to failed"
        assert_match 'Dependency failed: dep' "$(jq -r '.error_msg' "${QUEUE_DIR}/failed/child.task")" "failed dependency named in error"
    )

    (
        mock_reset
        # shellcheck source=../queue_manager.sh
        source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

        local out_dir="${TEST_TMPDIR}/dep_promote_out"
        init_queue "${out_dir}" >/dev/null

        write_queue_task completed dep SETUP_BASELINE d
        write_queue_task pending child EVAL_BASELINE c '{"dependencies":["dep"]}'

        update_dependents "dep"
        assert_file_exists "${QUEUE_DIR}/ready/child.task" "dependent moved to ready when completed dependency is present"
    )

    (
        mock_reset
        # shellcheck source=../queue_manager.sh
        source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

        local out_dir="${TEST_TMPDIR}/lifecycle_out"
        init_queue "${out_dir}" >/dev/null

        run complete_task "missing"
        assert_rc "1" "${RUN_RC}" "complete_task releases and fails for missing running task"

        run fail_task "missing" "boom"
        assert_rc "1" "${RUN_RC}" "fail_task releases and fails for missing running task"

        write_queue_task failed race
        acquire_queue_lock() { rm -f "${QUEUE_DIR}/failed/race.task"; return 0; }
        release_queue_lock() { return 0; }
        run retry_task "race"
        assert_rc "1" "${RUN_RC}" "retry_task fails if failed file disappears after lock"
    )
}

test_generate_model_tasks_remaining_batch_and_nonbatch_paths() {
    (
        mock_reset
        # shellcheck source=../queue_manager.sh
        source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

        local calls="${TEST_TMPDIR}/batch_calls"
        : > "${calls}"
        add_task() {
            local task_type="$1"
            local deps="$5"
            local params="${6:-}"
            printf '%s|%s|%s\n' "${task_type}" "${deps}" "${params}" >> "${calls}"
            local count
            count=$(wc -l < "${calls}" | tr -d ' ')
            echo "b${count}"
        }
        estimate_model_memory() { echo "14"; }

        local pack_root="${TEST_TMPDIR}/pack_batch"
        mkdir -p "${pack_root}/lib"
        SCRIPT_DIR="${pack_root}/lib"
        cat > "${pack_root}/scenarios.json" <<'EOF'
{
  "scenarios": [
    {"id": "clean", "generation": {"kind": "edit", "edit_spec": "clean_spec", "version": "clean"}},
    {"id": "stress", "generation": {"kind": "edit", "edit_spec": "stress_spec", "version": "stress"}}
  ]
}
EOF

        PACK_USE_BATCH_EDITS="true"
        PACK_CLEANUP_MODELS="1"
        DRIFT_CALIBRATION_RUNS="1"
        CLEAN_EDIT_RUNS="1"
        STRESS_EDIT_RUNS="1"
        RUN_ERROR_INJECTION="false"

        generate_model_tasks "1" "org/model" "model" >/dev/null

        local all_calls
        all_calls="$(cat "${calls}")"
        assert_match 'CREATE_EDITS_BATCH\|' "${all_calls}" "batch edit task created"
        assert_match 'evaluate_EDIT\|b5,b3,b4\|' "${all_calls}" "clean batch evaluate waits for edit batch preset and baseline report"
        assert_match 'evaluate_EDIT\|b5,b3,b4\|' "${all_calls}" "stress batch evaluate waits for edit batch preset and baseline report"
        assert_eq "2" "$(awk -F '|' '$1=="CLEANUP_EDIT"{c++} END {print c+0}' "${calls}")" "batch cleanup tasks created for both edit versions"
    )

    (
        mock_reset
        # shellcheck source=../queue_manager.sh
        source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

        local calls="${TEST_TMPDIR}/nonbatch_calls"
        : > "${calls}"
        add_task() {
            local task_type="$1"
            local deps="$5"
            local params="${6:-}"
            printf '%s|%s|%s\n' "${task_type}" "${deps}" "${params}" >> "${calls}"
            local count
            count=$(wc -l < "${calls}" | tr -d ' ')
            echo "n${count}"
        }
        estimate_model_memory() { echo "14"; }

        local pack_root="${TEST_TMPDIR}/pack_nonbatch"
        mkdir -p "${pack_root}/lib"
        SCRIPT_DIR="${pack_root}/lib"
        cat > "${pack_root}/scenarios.json" <<'EOF'
{
  "scenarios": [
    {"id": "clean", "generation": {"kind": "edit", "edit_spec": "clean_spec", "version": "clean"}},
    {"id": "stress", "generation": {"kind": "edit", "edit_spec": "stress_spec", "version": "stress"}}
  ]
}
EOF

        PACK_USE_BATCH_EDITS="false"
        PACK_CLEANUP_MODELS="1"
        DRIFT_CALIBRATION_RUNS="1"
        CLEAN_EDIT_RUNS="1"
        STRESS_EDIT_RUNS="1"
        RUN_ERROR_INJECTION="false"

        generate_model_tasks "1" "org/model" "model" >/dev/null

        local all_calls
        all_calls="$(cat "${calls}")"
        assert_eq "2" "$(awk -F '|' '$1=="CREATE_EDIT"{c++} END {print c+0}' "${calls}")" "nonbatch creates one edit per manifest version"
        assert_match 'evaluate_EDIT\|n5,n3,n4\|' "${all_calls}" "clean nonbatch evaluate waits for clean edit preset and baseline"
        assert_match 'evaluate_EDIT\|n8,n3,n4\|' "${all_calls}" "stress nonbatch evaluate waits for stress edit preset and baseline"
        assert_eq "2" "$(awk -F '|' '$1=="CLEANUP_EDIT"{c++} END {print c+0}' "${calls}")" "nonbatch cleanup tasks created for both edit versions"
    )

    (
        mock_reset
        # shellcheck source=../queue_manager.sh
        source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

        local calls="${TEST_TMPDIR}/large_memory_calls"
        : > "${calls}"
        add_task() {
            local task_type="$1"
            printf '%s\n' "${task_type}" >> "${calls}"
            local count
            count=$(wc -l < "${calls}" | tr -d ' ')
            echo "m${count}"
        }
        estimate_model_memory() {
            if [[ "${2:-}" == "evaluate_EDIT" ]]; then
                echo "175"
            else
                echo "14"
            fi
        }

        PACK_CLEANUP_MODELS="0"
        DRIFT_CALIBRATION_RUNS="0"
        PACK_PRESET_READY="false"
        CLEAN_EDIT_RUNS="0"
        STRESS_EDIT_RUNS="0"
        RUN_ERROR_INJECTION="false"
        unset PACK_USE_BATCH_EDITS

        assert_match 'Skipping edit creation \(CLEAN_EDIT_RUNS=0, STRESS_EDIT_RUNS=0\)' \
            "$(generate_model_tasks "1" "org/model-medium" "model-medium")" \
            "large memory branch still produces a valid no-edit graph"
        assert_match '^SETUP_BASELINE$' "$(cat "${calls}")" "setup task still created in large memory case"
    )
}

test_generate_model_tasks_remaining_manifest_fallback_paths() {
    (
        mock_reset
        # shellcheck source=../queue_manager.sh
        source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

        local out_dir="${TEST_TMPDIR}/state_out"
        init_queue "${out_dir}" >/dev/null
        cat > "${out_dir}/state/scenarios.json" <<'EOF'
{
  "scenarios": [
    {"id": "clean", "generation": {"kind": "edit", "edit_spec": "state_clean_spec", "version": "clean"}},
    {"id": "blank_error", "generation": {"kind": "error", "error_type": "", "env": {"SHOULD_SKIP": "1"}}},
    {"id": "real_error", "generation": {"kind": "error", "error_type": "state_error", "env": {"MODE": "state"}}}
  ]
}
EOF

        local calls="${TEST_TMPDIR}/state_calls"
        : > "${calls}"
        add_task() {
            local task_type="$1"
            local params="${6:-}"
            printf '%s|%s\n' "${task_type}" "${params}" >> "${calls}"
            local count
            count=$(wc -l < "${calls}" | tr -d ' ')
            echo "s${count}"
        }
        estimate_model_memory() { echo "14"; }

        PACK_USE_BATCH_EDITS="true"
        PACK_PRESET_READY="true"
        DRIFT_CALIBRATION_RUNS="0"
        CLEAN_EDIT_RUNS="1"
        STRESS_EDIT_RUNS="0"
        RUN_ERROR_INJECTION="true"

        generate_model_tasks "1" "org/model" "model" >/dev/null

        local all_calls
        all_calls="$(cat "${calls}")"
        assert_match 'state_clean_spec' "${all_calls}" "state manifest edit spec used"
        assert_eq "1" "$(awk -F '|' '$1=="CREATE_ERROR"{c++} END {print c+0}' "${calls}")" "blank state-manifest error skipped"
        assert_match '"error_type":"state_error"' "${all_calls}" "state manifest error emitted"
    )

    (
        mock_reset
        # shellcheck source=../queue_manager.sh
        source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

        local calls="${TEST_TMPDIR}/fallback_calls"
        : > "${calls}"
        add_task() {
            local task_type="$1"
            local params="${6:-}"
            printf '%s|%s\n' "${task_type}" "${params}" >> "${calls}"
            local count
            count=$(wc -l < "${calls}" | tr -d ' ')
            echo "f${count}"
        }
        estimate_model_memory() { echo "14"; }

        local pack_root="${TEST_TMPDIR}/pack_without_manifest"
        mkdir -p "${pack_root}/lib"
        SCRIPT_DIR="${pack_root}/lib"
        command() {
            if [[ "${1:-}" == "-v" && "${2:-}" == "jq" ]]; then
                return 1
            fi
            builtin command "$@"
        }
        type() {
            if [[ "${1:-}" == "-P" && "${2:-}" == "jq" ]]; then
                return 1
            fi
            builtin type "$@"
        }

        PACK_USE_BATCH_EDITS="true"
        PACK_PRESET_READY="true"
        DRIFT_CALIBRATION_RUNS="0"
        CLEAN_EDIT_RUNS="1"
        STRESS_EDIT_RUNS="1"
        RUN_ERROR_INJECTION="true"

        generate_model_tasks "1" "org/model" "model" >/dev/null

        local all_calls
        all_calls="$(cat "${calls}")"
        assert_match 'quant_rtn:clean:ffn' "${all_calls}" "fallback clean edit set used when manifest cannot load"
        assert_match 'fine_tune:0.0005:3:all' "${all_calls}" "fallback stress edit set used when manifest cannot load"
        grep -Fxq 'CREATE_ERROR|{"error_type": "nan_injection"}' "${calls}" \
            || t_fail "plain error JSON used when jq binary is unavailable actual='${all_calls}'"
    )
}

test_generate_evaluate_and_all_tasks_remaining_owner_branches() {
    mock_reset
    # shellcheck source=../queue_manager.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/queue/queue_manager.sh"

    local calls="${TEST_TMPDIR}/evaluate_calls"
    : > "${calls}"
    add_task() {
        local task_type="$1"
        local deps="$5"
        printf '%s|%s\n' "${task_type}" "${deps}" >> "${calls}"
        local count
        count=$(wc -l < "${calls}" | tr -d ' ')
        echo "e${count}"
    }
    estimate_model_memory() { echo "14"; }

    generate_evaluate_tasks "org/model" "model" "edit1" "preset1" "spec" "clean" "bad" >/dev/null
    assert_match 'evaluate_EDIT\|edit1,preset1' "$(cat "${calls}")" "invalid cert run count defaults to one evaluate task with preset dependency"

    : > "${calls}"
    generate_evaluate_tasks "org/model" "model" "edit1" "" "spec" "clean" "1" >/dev/null
    assert_match 'evaluate_EDIT\|edit1$' "$(cat "${calls}")" "empty preset leaves evaluate dependency on edit only"

    : > "${calls}"
    generate_evaluate_tasks "org/model" "model" "edit1" "preset1" "spec" "clean" "-3" >/dev/null
    assert_eq "" "$(cat "${calls}")" "negative cert run count emits no evaluate tasks"

    local generated=""
    generate_model_tasks() { generated+="$1:$2:$3;"; }
    resolve_dependencies() { echo "0"; }
    update_progress_state() { :; }
    print_queue_stats() { :; }

    generate_all_tasks "" "Org/Model Name" >/dev/null
    assert_match '2:Org/Model Name:org__model_name' "${generated}" "generate_all_tasks normalizes non-empty model ids"
}
