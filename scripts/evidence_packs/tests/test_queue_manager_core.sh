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
