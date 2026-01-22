#!/usr/bin/env bash

test_classify_error_precedence() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    local log="${TEST_TMPDIR}/task.log"

    printf "CUDA out of memory\nNo space left on device\n" > "${log}"
    assert_eq "oom" "$(classify_error "${log}")" "oom wins"

    printf "No space left on device\n" > "${log}"
    assert_eq "permanent" "$(classify_error "${log}")" "permanent"

    printf "ConnectionError\n" > "${log}"
    assert_eq "transient" "$(classify_error "${log}")" "transient"

    printf "something else\n" > "${log}"
    assert_eq "unknown" "$(classify_error "${log}")" "unknown"
}

test_calculate_backoff_uses_jitter_hook_and_caps() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    RETRY_BACKOFF_BASE=10
    RETRY_BACKOFF_MAX=100

    _rand_jitter_ms() { echo "0"; }
    assert_eq "10" "$(calculate_backoff 0)" "no jitter base"
    assert_eq "40" "$(calculate_backoff 2)" "exponential"

    # Max positive jitter (+20%) still caps.
    _rand_jitter_ms() { echo "$1"; }
    assert_eq "100" "$(calculate_backoff 4)" "caps at max"
}

test_calculate_backoff_sanitizes_non_numeric_retry_count_and_jitter() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    RETRY_BACKOFF_BASE=10
    RETRY_BACKOFF_MAX=100

    _rand_jitter_ms() { echo "bogus"; }
    assert_eq "10" "$(calculate_backoff "bogus")" "non-numeric retry_count + jitter sanitize to base"
}

test_calculate_backoff_defaults_invalid_config() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    RETRY_BACKOFF_BASE="bad"
    RETRY_BACKOFF_MAX="bad"
    _rand_jitter_ms() { echo "0"; }

    assert_eq "30" "$(calculate_backoff 0)" "invalid base/max fallback to defaults"
}

test_should_retry_task_branches() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    local task="${TEST_TMPDIR}/t.task"
    jq -n '{task_id:"t", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"failed", retries:2, max_retries:4, created_at:"x", started_at:null, completed_at:null, error_msg:"x", gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${task}"

    run should_retry_task "${task}" "permanent"
    assert_rc "1" "${RUN_RC}" "permanent not retried"

    run should_retry_task "${task}" "transient"
    assert_rc "0" "${RUN_RC}" "transient retried"

    # OOM max is (max_retries/2 + 1) => 3 when max_retries=4.
    jq '.retries = 3' "${task}" > "${task}.tmp" && mv "${task}.tmp" "${task}"
    run should_retry_task "${task}" "oom"
    assert_rc "1" "${RUN_RC}" "oom limited retries"

    jq '.retries = 4' "${task}" > "${task}.tmp" && mv "${task}.tmp" "${task}"
    run should_retry_task "${task}" "unknown"
    assert_rc "1" "${RUN_RC}" "max retries enforced"
}

test_should_retry_task_sanitizes_missing_retry_fields() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    MAX_RETRIES=3

    local task="${TEST_TMPDIR}/t.task"
    jq -n '{task_id:"t", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"failed", created_at:"x", started_at:null, completed_at:null, error_msg:"x", gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${task}"

    run should_retry_task "${task}" "transient"
    assert_rc "0" "${RUN_RC}" "missing retries/max_retries defaults to retryable"
}

test_should_retry_task_sanitizes_invalid_max_retries_env() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    MAX_RETRIES="nope"

    local task="${TEST_TMPDIR}/t.task"
    jq -n '{task_id:"t", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"failed", retries:2, created_at:"x", started_at:null, completed_at:null, error_msg:"x", gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${task}"

    run should_retry_task "${task}" "transient"
    assert_rc "0" "${RUN_RC}" "invalid MAX_RETRIES falls back to default"
    assert_eq "0" "$(echo "${RUN_OUT}${RUN_ERR}" | grep -c 'integer expression expected' || true)" "no integer expression errors"
}

test_maybe_retry_task_sets_retry_after_and_moves_task() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    _now_iso_plus_seconds() { echo "2025-01-01T00:00:10Z"; }
    _rand_jitter_ms() { echo "0"; }
    RETRY_BACKOFF_BASE=10

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/logs/tasks"
    export QUEUE_DIR="${out}/queue"
    mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed}

    local task_id="t1"
    jq -n '{task_id:"t1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"failed", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:"x", gpu_id:0, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/failed/${task_id}.task"
    printf "ConnectionError\n" > "${out}/logs/tasks/${task_id}.log"

    run maybe_retry_task "${task_id}"
    assert_rc "0" "${RUN_RC}" "retry scheduled"

    assert_file_exists "${QUEUE_DIR}/ready/${task_id}.task" "moved out of failed"
    assert_eq "2025-01-01T00:00:10Z" "$(jq -r '.params.retry_after' "${QUEUE_DIR}/ready/${task_id}.task")" "retry_after set"
    assert_eq "transient" "$(jq -r '.params.last_error_type' "${QUEUE_DIR}/ready/${task_id}.task")" "last_error_type set"
    assert_eq "1" "$(jq -r '.retries' "${QUEUE_DIR}/ready/${task_id}.task")" "retries incremented"
}

test_maybe_retry_task_sanitizes_non_numeric_retries_and_returns_error_when_update_task_params_fails() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    _now_iso_plus_seconds() { echo "2025-01-01T00:00:10Z"; }
    _rand_jitter_ms() { echo "0"; }
    RETRY_BACKOFF_BASE=10

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/logs/tasks"
    export QUEUE_DIR="${out}/queue"
    mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed}

    local task_id="t1"
    jq -n '{task_id:"t1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"failed", retries:null, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:"x", gpu_id:0, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/failed/${task_id}.task"
    printf "ConnectionError\n" > "${out}/logs/tasks/${task_id}.log"

    update_task_params() { return 1; }

    run maybe_retry_task "${task_id}"
    assert_rc "1" "${RUN_RC}" "update_task_params failure returns non-zero"
    assert_file_exists "${QUEUE_DIR}/failed/${task_id}.task" "task remains failed when params update fails"
    [[ ! -f "${QUEUE_DIR}/ready/${task_id}.task" ]] || t_fail "expected task not moved to ready when params update fails"
}

test_is_retry_ready_gates_on_retry_after() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    _now_epoch() { echo "0"; }
    local task="${TEST_TMPDIR}/t.task"
    jq -n '{task_id:"t", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"ready", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:[], params:{retry_after:"2025-01-01T00:00:10Z"}, priority:50}' \
        > "${task}"

    ! is_retry_ready "${task}"

    _now_epoch() { echo "$(_iso_to_epoch "2025-01-01T00:00:10Z")"; }
    is_retry_ready "${task}"
}

test_detect_helpers_return_false_on_missing_log_file() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    ! detect_oom "${TEST_TMPDIR}/nope.log"
    ! detect_transient_error "${TEST_TMPDIR}/nope.log"
    ! detect_permanent_error "${TEST_TMPDIR}/nope.log"
}

test_maybe_retry_task_missing_task_and_non_retryable_paths() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/logs/tasks"
    export QUEUE_DIR="${out}/queue"
    mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed}

    # Task not found for retry.
    if maybe_retry_task "missing" >/dev/null; then
        t_fail "expected maybe_retry_task to fail when task cannot be found"
    fi

    # Not retryable path (permanent error).
    local task_id="t1"
    jq -n '{task_id:"t1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"failed", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:"x", gpu_id:0, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${QUEUE_DIR}/failed/${task_id}.task"
    printf "No space left on device\n" > "${out}/logs/tasks/${task_id}.log"
    if maybe_retry_task "${task_id}" >/dev/null; then
        t_fail "expected maybe_retry_task to refuse permanent errors"
    fi
}

test_is_retry_ready_defaults_ready_when_retry_after_missing() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    local task="${TEST_TMPDIR}/t.task"
    jq -n '{task_id:"t", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"ready", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${task}"
    is_retry_ready "${task}"
}

test_handle_oom_task_clamps_batch_and_seq_len_minimums() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    fixture_write "python3.stub" ""
    fixture_write "python3.rc" "0"

    local task="${TEST_TMPDIR}/t.task"
    jq -n '{task_id:"t", task_type:"EVAL_BASELINE", model_id:"m", model_name:"n", status:"running", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:0, assigned_gpus:null, dependencies:[], params:{batch_size:1, seq_len:100}, priority:50}' \
        > "${task}"

    handle_oom_task "${task}" 0 "${TEST_TMPDIR}/log.txt"

    assert_eq "1" "$(jq -r '.params.batch_size' "${task}")" "batch_size clamped to 1"
    assert_eq "128" "$(jq -r '.params.seq_len' "${task}")" "seq_len clamped to 128"
    assert_eq "true" "$(jq -r '.params.oom_recovery' "${task}")" "oom_recovery flag set"
}

test_handle_oom_task_sanitizes_non_numeric_batch_and_seq() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    fixture_write "python3.stub" ""
    fixture_write "python3.rc" "0"

    local task="${TEST_TMPDIR}/t.task"
    jq -n '{task_id:"t", task_type:"EVAL_BASELINE", model_id:"m", model_name:"n", status:"running", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:0, assigned_gpus:null, dependencies:[], params:{batch_size:"nope", seq_len:"bad"}, priority:50}' \
        > "${task}"

    handle_oom_task "${task}" 0 "${TEST_TMPDIR}/log.txt"

    assert_eq "16" "$(jq -r '.params.batch_size' "${task}")" "non-numeric batch_size falls back to 32 then halves"
    assert_eq "256" "$(jq -r '.params.seq_len' "${task}")" "non-numeric seq_len falls back to 512 then halves"
    assert_eq "true" "$(jq -r '.params.oom_recovery' "${task}")" "oom_recovery flag set"
}

test_record_error_and_get_error_stats_cover_create_and_append_paths() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/state"

    # Missing log file returns defaults.
    assert_match '"total": 0' "$(get_error_stats "${out}")" "missing error log defaults"

    # Create new log, then append.
    record_error "t1" "oom" "x" "${out}"
    record_error "t2" "transient" "y" "${out}"
    assert_eq "2" "$(jq 'length' "${out}/state/errors.json")" "two error entries recorded"
}

test_record_error_returns_nonzero_and_preserves_existing_invalid_log() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/state"
    printf "not json\n" > "${out}/state/errors.json"

    run record_error "t1" "oom" "x" "${out}"
    assert_rc "1" "${RUN_RC}" "invalid existing log returns non-zero"
    assert_eq "not json" "$(cat "${out}/state/errors.json" | tr -d '\n')" "existing invalid log preserved"
}

test_record_error_returns_nonzero_when_errors_log_move_fails() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/state"

    mv() {
        local dst="${2:-}"
        if [[ "${1:-}" == "-f" ]]; then
            dst="${3:-}"
        fi
        if [[ "${dst}" == "${out}/state/errors.json" ]]; then
            return 1
        fi
        command mv "$@"
    }

    run record_error "t1" "oom" "x" "${out}"
    assert_rc "1" "${RUN_RC}" "mv failure returns non-zero"
    [[ ! -f "${out}/state/errors.json" ]] || t_fail "expected errors log not created when move fails"
}

test_health_check_error_branches_for_gpu_mem_disk_and_python() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    # nvidia-smi failure.
    _cmd_nvidia_smi() { return 1; }
    if health_check 0 >/dev/null; then
        t_fail "expected health_check to fail when nvidia-smi fails"
    fi

    # low/empty memory.
    _cmd_nvidia_smi() { return 0; }
    get_gpu_available_memory() { echo ""; }
    if health_check 0 >/dev/null; then
        t_fail "expected health_check to fail with insufficient memory"
    fi

    # low disk space.
    get_gpu_available_memory() { echo "10"; }
    mock_df_set_output $'Filesystem  1G-blocks  Used Available Use% Mounted on\n/dev/mock      1000    10        5   1% /\n'
    if health_check 0 >/dev/null; then
        t_fail "expected health_check to fail with low disk"
    fi

    # python missing torch.
    mock_df_set_output $'Filesystem  1G-blocks  Used Available Use% Mounted on\n/dev/mock      1000    10       990   1% /\n'
    _cmd_python() { return 1; }
    if health_check 0 >/dev/null; then
        t_fail "expected health_check to fail when torch import fails"
    fi
}

test_health_check_handles_non_numeric_disk_output() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    _cmd_nvidia_smi() { return 0; }
    get_gpu_available_memory() { echo "10"; }
    mock_df_set_output $'Filesystem  1G-blocks  Used Available Use% Mounted on\n/dev/mock      1000    10      foo   1% /\n'
    _cmd_python() { return 0; }

    run health_check 0
    assert_rc "1" "${RUN_RC}" "non-numeric disk output fails cleanly"
    assert_match 'Low disk space' "${RUN_OUT}${RUN_ERR}" "message emitted"
    assert_eq "0" "$(echo "${RUN_OUT}${RUN_ERR}" | grep -c 'integer expression expected' || true)" "avoid integer expression expected errors"
}

test_cleanup_failed_task_branches_cover_not_found_and_incomplete_artifacts() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    local out="${TEST_TMPDIR}/out"
    export QUEUE_DIR="${out}/queue"
    mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed}

    cleanup_failed_task "missing" "${out}"

    local model_name="m"
    local model_dir="${out}/${model_name}/models"
    mkdir -p "${model_dir}/keep_clean" "${model_dir}/drop_clean"
    echo "{}" > "${model_dir}/keep_clean/config.json"

    jq -n '{task_id:"t1", task_type:"CREATE_EDIT", model_id:"m", model_name:"m", status:"failed", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:"x", gpu_id:0, assigned_gpus:null, dependencies:[], params:{edit_spec:"quant_rtn:4:32:attn", version:"clean"}, priority:50}' \
        > "${QUEUE_DIR}/failed/t1.task"
    cleanup_failed_task "t1" "${out}"
    [[ ! -d "${model_dir}/drop_clean" ]] || t_fail "expected incomplete edit dir removed"

    local error_dir="${model_dir}/error_cuda_assert"
    mkdir -p "${error_dir}"
    jq -n '{task_id:"t2", task_type:"CREATE_ERROR", model_id:"m", model_name:"m", status:"failed", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:"x", gpu_id:0, assigned_gpus:null, dependencies:[], params:{error_type:"cuda_assert"}, priority:50}' \
        > "${QUEUE_DIR}/failed/t2.task"
    cleanup_failed_task "t2" "${out}"
    [[ ! -d "${error_dir}" ]] || t_fail "expected incomplete error dir removed"
}

test_get_error_stats_print_error_summary_and_cleanup_all_failed_cover_success_paths() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/state"

    record_error "t1" "oom" "x" "${out}"
    record_error "t2" "transient" "y" "${out}"

    local stats
    stats="$(get_error_stats "${out}")"
    assert_match '"total": 2' "${stats}" "stats include totals"

    assert_match 'Total Errors: 2' "$(print_error_summary "${out}")" "print_error_summary prints totals"

    export QUEUE_DIR="${out}/queue"
    mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed}
    jq -n '{task_id:"t1", task_type:"CREATE_EDIT", model_id:"m", model_name:"n", status:"failed", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:"x", gpu_id:-1, assigned_gpus:null, dependencies:[], params:{edit_spec:"quant_rtn:8:128:ffn", version:"clean"}, priority:50}' \
        > "${QUEUE_DIR}/failed/t1.task"

    local called=""
    cleanup_failed_task() { called+="${1},"; }
    run cleanup_all_failed "${out}"
    assert_rc "0" "${RUN_RC}" "cleanup_all_failed returns 0"
    assert_match 'Cleanup complete' "${RUN_OUT}" "cleanup_all_failed summary"
    assert_match 't1,' "${called}" "cleanup_failed_task invoked for failed task"
}

test_health_check_returns_zero_when_all_checks_pass() {
    mock_reset
    # shellcheck source=../fault_tolerance.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/fault_tolerance.sh"

    _cmd_nvidia_smi() { return 0; }
    get_gpu_available_memory() { echo "100"; }
    _cmd_df() { printf '%s\n' "Filesystem  1G-blocks  Used Available Use% Mounted on" "/dev/mock      1000    10       990   1% /"; }
    _cmd_python() { return 0; }

    health_check 0 >/dev/null
}
