#!/usr/bin/env bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/validation_suite_test_helpers.sh"

test_pack_validation_main_dynamic_exits_with_resumable_blocked_state() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    check_dependencies() { :; }
    configure_gpu_pool() { NUM_GPUS=1; GPU_ID_LIST="0"; export NUM_GPUS GPU_ID_LIST; }
    disk_preflight() { :; }
    setup_pack_environment() { :; }
    handle_disk_pressure() { return 0; }

    RESUME_FLAG="false"

    init_queue() {
        QUEUE_DIR="${OUTPUT_DIR}/queue"
        GPU_RESERVATION_DIR="${OUTPUT_DIR}/gpu_reservations"
        mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed} "${OUTPUT_DIR}/workers" "${OUTPUT_DIR}/logs" "${GPU_RESERVATION_DIR}"
        export QUEUE_DIR GPU_RESERVATION_DIR
    }
    generate_all_tasks() { :; }
    resolve_dependencies() { echo 0; }
    cancel_tasks_with_failed_dependencies() { echo 0; }
    get_queue_stats() { echo "1:0:0:0:1:2"; }
    queue_terminal_state() { echo "blocked_failed_dependencies"; }
    apply_work_stealing_boost() { :; }
    count_tasks() {
        if [[ "${1:-}" == "failed" ]]; then
            echo 1
        else
            echo 0
        fi
    }
    print_queue_stats() { :; }
    get_task_id() { echo "dep"; }
    get_task_field() { echo "dependency failed"; }
    compile_results() { echo "compile" >> "${TEST_TMPDIR}/analysis.calls"; }
    run_analysis() { echo "analysis" >> "${TEST_TMPDIR}/analysis.calls"; }
    generate_verdict() { echo "verdict" >> "${TEST_TMPDIR}/analysis.calls"; }

    list_run_gpu_ids() { printf '0\n'; }
    is_queue_empty() { return 1; }
    sleep() { :; }
    get_free_disk_gb() { echo "999"; }
    signal_shutdown() { echo "shutdown:$1" >> "${TEST_TMPDIR}/shutdown.calls"; }

    local stub_lib="${TEST_TMPDIR}/stub_lib"
    mkdir -p "${stub_lib}"
    for f in task_serialization.sh queue_manager.sh scheduler.sh task_functions.sh fault_tolerance.sh; do
        printf '%s\n' "#!/usr/bin/env bash" > "${stub_lib}/${f}"
    done
    cat > "${stub_lib}/gpu_worker.sh" <<'EOF'
#!/usr/bin/env bash
gpu_worker() { return 0; }
EOF
    LIB_DIR="${stub_lib}"
    export LIB_DIR

    init_queue
    printf '{"status":"failed"}\n' > "${QUEUE_DIR}/failed/dep.task"

    run main_dynamic
    assert_rc "1" "${RUN_RC}" "blocked queue exits nonzero"
    assert_file_exists "${TEST_TMPDIR}/shutdown.calls" "blocked queue signals shutdown"
    assert_match '"status": "blocked_failed_dependencies"' "$(cat "${OUTPUT_DIR}/state/progress.json")" "progress records blocked state"
    assert_match '"detail": "all pending tasks are blocked on failed dependencies"' "$(cat "${OUTPUT_DIR}/state/progress.json")" "progress records blocked detail"
    [[ ! -f "${TEST_TMPDIR}/analysis.calls" ]] || t_fail "analysis should not run after blocked terminal state"
}

test_pack_validation_main_dynamic_fresh_task_generation_and_touch_shutdown_branch() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    check_dependencies() { :; }
    configure_gpu_pool() { NUM_GPUS=1; GPU_ID_LIST="0"; export NUM_GPUS GPU_ID_LIST; }
    disk_preflight() { :; }
    setup_pack_environment() { :; }
    handle_disk_pressure() { return 0; }

    RESUME_FLAG="false"

    init_queue() {
        QUEUE_DIR="${OUTPUT_DIR}/queue"
        mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed} "${OUTPUT_DIR}/workers"
        export QUEUE_DIR
    }
    generate_all_tasks() { echo "generated" >> "${TEST_TMPDIR}/tasks.calls"; }
    resolve_dependencies() { echo 0; }
    count_tasks() { echo 0; }
    print_queue_stats() { :; }
    compile_results() { :; }
    run_analysis() { :; }
    generate_verdict() { :; }

    list_run_gpu_ids() { printf '0\n'; }
    is_queue_empty() { return 0; }

    local stub_lib="${TEST_TMPDIR}/stub_lib"
    mkdir -p "${stub_lib}"
    for f in task_serialization.sh queue_manager.sh scheduler.sh task_functions.sh fault_tolerance.sh; do
        printf '%s\n' "#!/usr/bin/env bash" > "${stub_lib}/${f}"
    done
    cat > "${stub_lib}/gpu_worker.sh" <<'EOF'
#!/usr/bin/env bash
gpu_worker() { return 0; }
EOF
    LIB_DIR="${stub_lib}"
    export LIB_DIR

    MIN_FREE_DISK_GB="bogus"
    get_free_disk_gb() { echo "999"; }

    main_dynamic
    assert_file_exists "${TEST_TMPDIR}/tasks.calls" "fresh run generates tasks"
    assert_file_exists "${OUTPUT_DIR}/workers/SHUTDOWN" "touch shutdown when signal_shutdown missing"
}

test_pack_validation_main_dynamic_scenario_summary_branch_coverage() {
    mock_reset

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/jq" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "bogus"
EOF
    chmod +x "${bin_dir}/jq"
    export PATH="${bin_dir}:$PATH"
    hash -r 2>/dev/null || true

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    check_dependencies() { :; }
    configure_gpu_pool() { NUM_GPUS=1; GPU_ID_LIST="0"; export NUM_GPUS GPU_ID_LIST; }
    disk_preflight() { :; }
    setup_pack_environment() { :; }
    handle_disk_pressure() { return 0; }

    init_queue() {
        QUEUE_DIR="${OUTPUT_DIR}/queue"
        mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed} "${OUTPUT_DIR}/workers" "${OUTPUT_DIR}/logs"
        export QUEUE_DIR
    }
    generate_all_tasks() { :; }
    resolve_dependencies() { echo 0; }
    count_tasks() { echo 0; }
    print_queue_stats() { :; }
    compile_results() { :; }
    run_analysis() { :; }
    generate_verdict() { :; }
    list_run_gpu_ids() { printf ''; }
    is_queue_empty() { return 0; }
    get_free_disk_gb() { echo "999"; }

    local stub_lib="${TEST_TMPDIR}/stub_lib"
    mkdir -p "${stub_lib}"
    for f in task_serialization.sh queue_manager.sh scheduler.sh task_functions.sh fault_tolerance.sh; do
        printf '%s\n' "#!/usr/bin/env bash" > "${stub_lib}/${f}"
    done
    cat > "${stub_lib}/gpu_worker.sh" <<'EOF'
#!/usr/bin/env bash
gpu_worker() { return 0; }
EOF
    LIB_DIR="${stub_lib}"
    export LIB_DIR

    local scenarios_json='{"schema":"evidence_pack_scenarios_v1","schema_version":1,"scenarios":[]}'

    # Non-numeric runs + error injection enabled
    OUTPUT_DIR="${TEST_TMPDIR}/out_scenario_1"
    pack_setup_output_dirs
    mkdir -p "${OUTPUT_DIR}/state"
    printf '%s\n' "${scenarios_json}" > "${OUTPUT_DIR}/state/scenarios.json"
    CLEAN_EDIT_RUNS="bogus"
    STRESS_EDIT_RUNS="bogus"
    RUN_ERROR_INJECTION="true"
    RESUME_FLAG="false"
    main_dynamic

    # Negative runs + error injection disabled (covers else branch)
    OUTPUT_DIR="${TEST_TMPDIR}/out_scenario_2"
    pack_setup_output_dirs
    mkdir -p "${OUTPUT_DIR}/state"
    printf '%s\n' "${scenarios_json}" > "${OUTPUT_DIR}/state/scenarios.json"
    CLEAN_EDIT_RUNS="-1"
    STRESS_EDIT_RUNS="-1"
    RUN_ERROR_INJECTION="false"
    RESUME_FLAG="false"
    main_dynamic
}

test_pack_validation_main_dynamic_calibrate_only_stops_after_presets_even_with_pending_tasks() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    export OUTPUT_DIR
    export PACK_SUITE_MODE="calibrate-only"

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    check_dependencies() { :; }
    configure_gpu_pool() { NUM_GPUS=1; GPU_ID_LIST="0"; export NUM_GPUS GPU_ID_LIST; }
    disk_preflight() { :; }
    setup_pack_environment() { :; }
    handle_disk_pressure() { return 0; }

    RESUME_FLAG="true"

    init_queue() {
        QUEUE_DIR="${OUTPUT_DIR}/queue"
        mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed} "${OUTPUT_DIR}/workers"
        export QUEUE_DIR
    }
    resolve_dependencies() { echo 0; }
    reclaim_orphaned_tasks() { :; }
    print_queue_stats() { :; }
    count_tasks() { echo 0; }

    # Ensure queue is NOT empty (pending contains non-calibration task), but
    # calibration-only early exit triggers because all preset tasks are completed.
    init_queue
    printf '{"status":"pending"}\n' > "${QUEUE_DIR}/pending/model_EVAL_BASELINE_001_dead.task"
    printf '{"status":"completed","task_type":"GENERATE_PRESET"}\n' > "${QUEUE_DIR}/completed/model_GENERATE_PRESET_001_beef.task"

    list_run_gpu_ids() { printf '0\n'; }

    # Stub worker scripts so start_worker doesn't run real gpu_worker.
    local stub_lib="${TEST_TMPDIR}/stub_lib"
    mkdir -p "${stub_lib}"
    for f in task_serialization.sh queue_manager.sh scheduler.sh task_functions.sh fault_tolerance.sh; do
        printf '%s\n' "#!/usr/bin/env bash" > "${stub_lib}/${f}"
    done
    cat > "${stub_lib}/gpu_worker.sh" <<'EOF'
#!/usr/bin/env bash
gpu_worker() { return 0; }
EOF
    LIB_DIR="${stub_lib}"
    export LIB_DIR

    compile_results() { echo "compile" >> "${TEST_TMPDIR}/analysis.calls"; }
    run_analysis() { echo "analysis" >> "${TEST_TMPDIR}/analysis.calls"; }
    generate_verdict() { echo "verdict" >> "${TEST_TMPDIR}/analysis.calls"; }

    signal_shutdown() { echo "shutdown:$1" >> "${TEST_TMPDIR}/shutdown.calls"; }
    get_free_disk_gb() { echo "999"; }

    main_dynamic

    assert_file_exists "${TEST_TMPDIR}/shutdown.calls" "calibration-only run signals shutdown early"
    [[ ! -f "${TEST_TMPDIR}/analysis.calls" ]] || t_fail "analysis should not run for calibration-only mode"
}

test_pack_validation_main_wrapper_parses_progress_and_reports_failed_tasks_offline() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    log() { :; }
    log_section() { :; }

    check_dependencies() { :; }
    configure_gpu_pool() { NUM_GPUS=1; GPU_ID_LIST="0"; export NUM_GPUS GPU_ID_LIST; }
    disk_preflight() { :; }
    setup_pack_environment() { :; }

    RESUME_FLAG="false"
    init_queue() {
        QUEUE_DIR="${OUTPUT_DIR}/queue"
        GPU_RESERVATION_DIR="${OUTPUT_DIR}/gpu_reservations"
        mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed} "${OUTPUT_DIR}/workers" "${GPU_RESERVATION_DIR}"
        export QUEUE_DIR GPU_RESERVATION_DIR
    }
    init_gpu_reservations() { GPU_RESERVATION_DIR="${OUTPUT_DIR}/gpu_reservations"; export GPU_RESERVATION_DIR; }
    generate_all_tasks() { :; }

    resolve_dependencies() { echo 0; }
    get_queue_stats() { echo "1:2:3:4:5:6"; }
    apply_work_stealing_boost() { echo "boost" >> "${TEST_TMPDIR}/boost.calls"; }

    mkdir -p "${OUTPUT_DIR}/queue/failed"
    : > "${OUTPUT_DIR}/queue/failed/t2.task"
    get_task_id() { echo "${1}" >> "${TEST_TMPDIR}/task_id.calls"; echo "t2"; }
    get_task_field() { echo "${1}:${2}" >> "${TEST_TMPDIR}/task_field.calls"; echo "boom"; }

    count_tasks() { [[ "${1:-}" == "failed" ]] && echo 1 || echo 0; }
    print_queue_stats() { :; }
    list_run_gpu_ids() { printf '0\n'; }

    local empty_checks=0
    is_queue_empty() {
        empty_checks=$((empty_checks + 1))
        [[ ${empty_checks} -ge 2 ]]
    }
    get_free_disk_gb() { echo "999"; }

    python3() {
        echo "python3 $*" >> "${TEST_TMPDIR}/python3.calls"
        return 0
    }

    local stub_lib="${TEST_TMPDIR}/stub_lib"
    mkdir -p "${stub_lib}"
    for f in task_serialization.sh queue_manager.sh scheduler.sh task_functions.sh fault_tolerance.sh; do
        printf '%s\n' "#!/usr/bin/env bash" > "${stub_lib}/${f}"
    done
    cat > "${stub_lib}/gpu_worker.sh" <<'EOF'
#!/usr/bin/env bash
gpu_worker() { return 0; }
EOF
    LIB_DIR="${stub_lib}"
    export LIB_DIR

    kill() { return 0; }

    run main
    assert_rc "1" "${RUN_RC}" "main fails closed when failed tasks are present"
    assert_file_exists "${TEST_TMPDIR}/boost.calls" "progress path applies work-stealing boost"
    assert_file_exists "${TEST_TMPDIR}/task_id.calls" "failed task reporting reads task ids"
    assert_file_exists "${TEST_TMPDIR}/python3.calls" "analysis steps invoke python3"
}

test_pack_validation_setup_output_dirs_returns_nonzero_when_output_dir_is_file() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    OUTPUT_DIR="${TEST_TMPDIR}/out_file"
    echo "not a dir" > "${OUTPUT_DIR}"

    run pack_setup_output_dirs
    assert_rc "1" "${RUN_RC}" "mkdir failure propagates as non-zero"
}


test_pack_validation_pack_output_dir_defaults_to_pack_output_dir() {
    mock_reset

    unset OUTPUT_DIR
    PACK_OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    assert_eq "${PACK_OUTPUT_DIR}" "${OUTPUT_DIR}" "PACK_OUTPUT_DIR seeds OUTPUT_DIR"
    unset PACK_OUTPUT_DIR OUTPUT_DIR
}


test_pack_validation_pack_model_list_and_revisions_branches() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    mkdir -p "${OUTPUT_DIR}/state"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    MODEL_1="org/model1"
    MODEL_2=""
    enable -n mapfile 2>/dev/null || true
    pack_model_list_array
    assert_eq "org/model1" "${PACK_MODEL_LIST[0]}" "fallback populates model list"
    enable mapfile 2>/dev/null || true

    mapfile() {
        local flag="$1"
        local target="$2"
        local -a values=()
        while IFS= read -r line; do
            values+=("${line}")
        done
        if [[ "${flag}" == "-t" ]]; then
            eval "${target}=()"
            local value
            for value in "${values[@]}"; do
                eval "${target}+=(\"${value}\")"
            done
        fi
    }
    PACK_MODEL_LIST=()
    pack_model_list_array
    assert_eq "org/model1" "${PACK_MODEL_LIST[0]}" "mapfile populates model list"
    unset -f mapfile

    rm -f "${OUTPUT_DIR}/state/model_revisions.json"
    run pack_load_model_revisions
    assert_rc "1" "${RUN_RC}" "missing revisions file returns non-zero"

    run pack_model_revision "org/model1"
    assert_rc "1" "${RUN_RC}" "missing revisions file returns non-zero"

    echo '{"models":{"org/model1":{"revision":"abc"}}}' > "${OUTPUT_DIR}/state/model_revisions.json"
    run pack_load_model_revisions
    assert_rc "0" "${RUN_RC}" "load revisions succeeds"
    assert_eq "${OUTPUT_DIR}/state/model_revisions.json" "${PACK_MODEL_REVISIONS_FILE}" "revisions file set"

    run pack_model_revision "org/model1"
    assert_rc "0" "${RUN_RC}" "revision lookup succeeds"
    assert_eq "abc" "${RUN_OUT}" "revision returned"

    echo '{not_json' > "${OUTPUT_DIR}/state/model_revisions.json"
    run pack_load_model_revisions
    assert_rc "1" "${RUN_RC}" "invalid revisions file fails"
    assert_match "Failed to parse model revisions file" "${RUN_ERR}" "parse failure reported"

    echo '{"models":{"org/model1":{"gated":true}}}' > "${OUTPUT_DIR}/state/model_revisions.json"
    run pack_load_model_revisions
    assert_rc "1" "${RUN_RC}" "gated model revisions fail"
}

test_pack_validation_fallback_resolve_edit_params_executes_python() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local model_output_dir="${TEST_TMPDIR}/model_out"
    mkdir -p "${model_output_dir}"

    run resolve_edit_params "${model_output_dir}" "quant_rtn:4:32:ffn" "stress"
    assert_rc "0" "${RUN_RC}" "resolve_edit_params succeeds"
    assert_match "\"status\": \"selected\"" "${RUN_OUT}" "resolver returns selected status"
}


test_pack_validation_preflight_models_error_branches() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    mock_python3_stub_enable
    fixture_write "python3.rc" "0"

    local error_log="${TEST_TMPDIR}/error.msg"
    : > "${error_log}"
    error_exit() { echo "$1" >> "${error_log}"; return 0; }

    PACK_NET="0"
    run pack_preflight_models "${OUTPUT_DIR}" "org/model"
    assert_match "Preflight requires" "$(cat "${error_log}")" "preflight requires net"

    PACK_NET="1"
    set +u
    run pack_preflight_models "${OUTPUT_DIR}"
    set -u
    assert_match "No models provided" "$(cat "${error_log}")" "preflight requires models"

    fixture_write "python3.rc" "2"
    run pack_preflight_models "${OUTPUT_DIR}" "org/model"
    assert_rc "1" "${RUN_RC}" "python failure returns non-zero"
}


test_pack_validation_setup_hf_cache_dirs_requires_output_dir() {
    mock_reset

    OUTPUT_DIR=""
    PACK_OUTPUT_DIR=""
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    run pack_setup_hf_cache_dirs
    assert_rc "1" "${RUN_RC}" "missing OUTPUT_DIR fails"
}


test_pack_validation_estimate_planned_model_storage_mapfile() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    EDIT_TYPES_CLEAN=("quant_rtn:clean:ffn")
    EDIT_TYPES_STRESS=()
    RUN_ERROR_INJECTION="false"

    pack_model_list() { printf '%s\n' "org/model"; }
    estimate_model_weights_gb() { echo "10"; }

    mapfile() {
        local flag="$1"
        local target="$2"
        local -a values=()
        while IFS= read -r line; do
            values+=("${line}")
        done
        if [[ "${flag}" == "-t" ]]; then
            eval "${target}=()"
            local value
            for value in "${values[@]}"; do
                eval "${target}+=(\"${value}\")"
            done
        fi
    }

    local total
    total="$(estimate_planned_model_storage_gb)"
    unset -f mapfile
    assert_eq "20" "${total}" "planned storage sums weights and edits"
}


test_pack_validation_estimate_planned_storage_honors_one_sided_state_manifest() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    mkdir -p "${OUTPUT_DIR}/state"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    cat > "${OUTPUT_DIR}/state/scenarios.json" <<'EOF'
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

    RUN_ERROR_INJECTION="false"
    PACK_CLEANUP_MODELS="0"

    pack_model_list() { printf '%s\n' "org/model"; }
    estimate_model_weights_gb() { echo "10"; }

    local total
    total="$(estimate_planned_model_storage_gb)"
    assert_eq "20" "${total}" "one clean scenario counts as one edit copy without fallback expansion"
}


test_pack_validation_setup_model_revision_branches() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    LOG_FILE="${TEST_TMPDIR}/log.txt"
    : > "${LOG_FILE}"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    error_exit() { echo "$1" > "${TEST_TMPDIR}/error.msg"; return 1; }

    pack_model_revision() { echo ""; }
    PACK_NET="1"
    run setup_model "org/model" "0"
    assert_rc "1" "${RUN_RC}" "missing revision fails with net"

    PACK_NET="0"
    run setup_model "org/model" "0"
    assert_rc "1" "${RUN_RC}" "missing revision fails offline"

    pack_model_revision() { echo "rev1"; }
    PACK_NET="0"
    run setup_model "org/model" "0"
    assert_rc "1" "${RUN_RC}" "offline missing cache fails"
}


test_pack_validation_generate_invarlock_config_guard_order() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local cfg="${TEST_TMPDIR}/cfg.yaml"
    PACK_GUARDS_ORDER="variance,invariants"
    generate_invarlock_config "model" "${cfg}" "edit"
    assert_match "variance" "$(cat "${cfg}")" "guard order uses csv"

    PACK_GUARDS_ORDER=" , "
    generate_invarlock_config "model" "${cfg}" "edit"
    assert_match "invariants" "$(cat "${cfg}")" "guard order defaults when empty"
}


test_pack_validation_run_determinism_repeats_branch_coverage() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    OUTPUT_DIR=""
    PACK_REPEATS="1"
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "requires OUTPUT_DIR"

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    pack_setup_output_dirs

    PACK_REPEATS="0"
    run pack_run_determinism_repeats
    assert_rc "0" "${RUN_RC}" "zero repeats returns success"

    PACK_REPEATS="bad"
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "invalid repeats fails"

    PACK_REPEATS="1"
    PACK_MODEL_LIST=()
    pack_model_list() { :; }
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "missing models fails"

    pack_model_list() { printf '%s\n' "org/model"; }
    PACK_MODEL_LIST=()
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "missing baseline path fails"

    local model_id="org/model"
    local model_name
    model_name="$(sanitize_model_name "${model_id}")"
    local model_output_dir="${OUTPUT_DIR}/${model_name}"
    mkdir -p "${model_output_dir}"
    local baseline_dir="${TEST_TMPDIR}/baseline"
    mkdir -p "${baseline_dir}"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"

    PACK_MODEL_LIST=("${model_id}")
    PACK_REPEATS="1"

    declare -a EDIT_TYPES_CLEAN=()
    declare -a EDIT_TYPES_STRESS=()
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "missing edit specs fails"

    declare -a EDIT_TYPES_CLEAN=()
    EDIT_TYPES_STRESS=("quant_rtn:4:32:ffn")
    resolve_edit_params() { return 1; }
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "resolve_edit_params failure returns non-zero"

    resolve_edit_params() {
        jq -n '{status:"skipped", edit_dir_name:""}'
    }
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "non-selected edit spec fails"

    resolve_edit_params() {
        jq -n '{status:"selected", edit_dir_name:"missing_edit_dir"}'
    }
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "missing edit dir fails"

    mkdir -p "${model_output_dir}/models/existing_edit"
    resolve_edit_params() {
        jq -n '{status:"selected", edit_dir_name:"existing_edit"}'
    }
    run_invarlock_evaluate() { return 1; }
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "evaluate failure returns non-zero"

    resolve_edit_params() {
        jq -n '{status:"selected", edit_dir_name:"existing_edit"}'
    }
    run_invarlock_evaluate() { return 0; }
    mkdir() {
        for arg in "$@"; do
            if [[ "${arg}" == *"/determinism/"* ]]; then
                return 1
            fi
        done
        command mkdir "$@"
    }
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "determinism mkdir failure returns non-zero"
    unset -f mkdir

    resolve_edit_params() {
        jq -n '{status:"selected", edit_dir_name:"existing_edit"}'
    }
    run_invarlock_evaluate() { return 0; }
    mkdir() {
        for arg in "$@"; do
            if [[ "${arg}" == *"/analysis" ]]; then
                return 1
            fi
        done
        command mkdir "$@"
    }
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "analysis mkdir failure returns non-zero"
    unset -f mkdir
}
