#!/usr/bin/env bash

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
