#!/usr/bin/env bash

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
