#!/usr/bin/env bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/validation_suite_test_helpers.sh"

test_pack_validation_source_libs_prefers_lib_subdir() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local root="${TEST_TMPDIR}/pkg"
    mkdir -p "${root}/lib"
    for f in task_serialization.sh queue_manager.sh scheduler.sh task_functions.sh gpu_worker.sh; do
        printf '%s\n' "#!/usr/bin/env bash" > "${root}/lib/${f}"
    done

    _pack_script_dir() { echo "${root}"; }
    pack_source_libs
    assert_eq "${root}/lib" "${LIB_DIR}" "lib subdir selected"
}


test_pack_validation_source_libs_uses_parent_lib_dir() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local root="${TEST_TMPDIR}/pkg"
    mkdir -p "${root}/lib" "${root}/child"
    for f in task_serialization.sh queue_manager.sh scheduler.sh task_functions.sh gpu_worker.sh; do
        printf '%s\n' "#!/usr/bin/env bash" > "${root}/lib/${f}"
    done

    _pack_script_dir() { echo "${root}/child"; }
    pack_source_libs
    assert_eq "${root}/lib" "${LIB_DIR}" "parent lib dir selected"
}


test_pack_validation_main_dynamic_demote_ready_tasks_for_calibration_only() {
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
    generate_all_tasks() { :; }
    resolve_dependencies() { echo 0; }
    demote_ready_tasks_for_calibration_only() { echo "demote" > "${TEST_TMPDIR}/demote.calls"; }
    count_tasks() { echo 0; }
    print_queue_stats() { :; }
    compile_results() { :; }
    run_analysis() { :; }
    generate_verdict() { :; }
    list_run_gpu_ids() { printf '0\n'; }

    local empty_checks=0
    is_queue_empty() {
        empty_checks=$((empty_checks + 1))
        [[ ${empty_checks} -ge 1 ]]
    }
    get_free_disk_gb() { echo "999"; }

    PACK_PRESET_READY="true"
    log() { echo "$*" >> "${TEST_TMPDIR}/log.msg"; }
    log_section() { :; }

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

    main_dynamic
    assert_file_exists "${TEST_TMPDIR}/demote.calls" "demote_ready_tasks_for_calibration_only invoked"
    assert_match "Calibration presets: reuse" "$(cat "${TEST_TMPDIR}/log.msg")" "preset reuse logged"
}


test_pack_validation_main_dynamic_calibrate_only_without_signal_shutdown() {
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

    init_queue
    printf '{"status":"pending"}\n' > "${QUEUE_DIR}/pending/model_EVAL_BASELINE_001_dead.task"
    printf '{"status":"completed","task_type":"GENERATE_PRESET"}\n' > "${QUEUE_DIR}/completed/model_GENERATE_PRESET_001_beef.task"

    list_run_gpu_ids() { printf '0\n'; }

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

    compile_results() { :; }
    run_analysis() { :; }
    generate_verdict() { :; }
    get_free_disk_gb() { echo "999"; }
    get_queue_stats() { echo "0:0:0:1:0:1"; }

    main_dynamic
    assert_file_exists "${OUTPUT_DIR}/workers/SHUTDOWN" "touch shutdown when signal_shutdown missing"
    assert_file_exists "${OUTPUT_DIR}/state/progress.json" "progress.json written when summary stats are present"
    assert_match "\"status\": \"complete\"" "$(cat "${OUTPUT_DIR}/state/progress.json")" "complete progress state recorded"
}


test_pack_validation_main_dynamic_warns_on_determinism_repeats_failure() {
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
    generate_all_tasks() { :; }
    resolve_dependencies() { echo 0; }
    count_tasks() { echo 0; }
    print_queue_stats() { :; }
    compile_results() { :; }
    run_analysis() { :; }
    generate_verdict() { :; }
    list_run_gpu_ids() { printf '0\n'; }

    local empty_checks=0
    is_queue_empty() {
        empty_checks=$((empty_checks + 1))
        [[ ${empty_checks} -ge 1 ]]
    }
    get_free_disk_gb() { echo "999"; }

    PACK_REPEATS="1"
    pack_run_determinism_repeats() { echo "repeats" > "${TEST_TMPDIR}/repeats.calls"; return 1; }
    log() { echo "$*" >> "${TEST_TMPDIR}/log.msg"; }
    log_section() { :; }

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

    main_dynamic
    assert_file_exists "${TEST_TMPDIR}/repeats.calls" "determinism repeats invoked"
}


test_pack_validation_pack_run_suite_branches() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    cleanup() { return 0; }
    pack_apply_network_mode() { :; }
    pack_preflight_datasets() { :; }
    check_dependencies() { :; }
    pack_prepare_tuned_edit_params() { :; }
    pack_validate_tuned_edit_params() { :; }
    pack_prepare_calibration_presets() { :; }
    pack_validate_guard_calibration() { :; }
    pack_validate_runtime_provenance() { :; }

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    PACK_NET="0"

    pack_require_bash4() { return 1; }
    run pack_run_suite
    assert_rc "1" "${RUN_RC}" "bash4 requirement enforced"
    trap - EXIT INT TERM HUP QUIT

    pack_require_bash4() { return 0; }
    OUTPUT_DIR=""
    run pack_run_suite
    assert_rc "1" "${RUN_RC}" "missing output dir fails"
    trap - EXIT INT TERM HUP QUIT

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    pack_source_libs() { return 1; }
    run pack_run_suite
    assert_rc "1" "${RUN_RC}" "pack_source_libs failure returns non-zero"
    trap - EXIT INT TERM HUP QUIT

    pack_source_libs() { return 0; }
    pack_setup_output_dirs() { return 1; }
    run pack_run_suite
    assert_rc "1" "${RUN_RC}" "pack_setup_output_dirs failure returns non-zero"
    trap - EXIT INT TERM HUP QUIT

    pack_setup_output_dirs() { return 0; }
    pack_prepare_scenarios_manifest() { return 1; }
    pack_setup_hf_cache_dirs() { return 0; }
    run pack_run_suite
    assert_rc "1" "${RUN_RC}" "pack_prepare_scenarios_manifest failure returns non-zero"
    trap - EXIT INT TERM HUP QUIT

    pack_prepare_scenarios_manifest() { return 0; }
    pack_setup_hf_cache_dirs() { return 1; }
    run pack_run_suite
    assert_rc "1" "${RUN_RC}" "pack_setup_hf_cache_dirs failure returns non-zero"
    trap - EXIT INT TERM HUP QUIT

    pack_setup_hf_cache_dirs() { return 0; }
    pack_model_list_array() { PACK_MODEL_LIST=("model"); }
    pack_load_model_revisions() { return 0; }
    main_dynamic() { :; }
    PACK_OUTPUT_DIR_ABSOLUTE="true"
    local original_dir
    original_dir="$(pwd)"
    cd "${TEST_TMPDIR}"
    OUTPUT_DIR="rel_out"
    PACK_NET="0"
    run pack_run_suite
    cd "${original_dir}"
    assert_rc "0" "${RUN_RC}" "absolute output dir path succeeds"
    assert_match '^/' "${OUTPUT_DIR}" "output dir normalized to absolute"
    trap - EXIT INT TERM HUP QUIT
    PACK_OUTPUT_DIR_ABSOLUTE="false"

    OUTPUT_DIR="${TEST_TMPDIR}/out_fail_prep_tuned"
    pack_prepare_tuned_edit_params() { return 1; }
    run pack_run_suite
    assert_rc "1" "${RUN_RC}" "pack_prepare_tuned_edit_params failure returns non-zero"
    trap - EXIT INT TERM HUP QUIT

    OUTPUT_DIR="${TEST_TMPDIR}/out_fail_validate_tuned"
    pack_prepare_tuned_edit_params() { return 0; }
    pack_validate_tuned_edit_params() { return 1; }
    run pack_run_suite
    assert_rc "1" "${RUN_RC}" "pack_validate_tuned_edit_params failure returns non-zero"
    trap - EXIT INT TERM HUP QUIT

    OUTPUT_DIR="${TEST_TMPDIR}/out_fail_prepare_calibration"
    pack_validate_tuned_edit_params() { return 0; }
    pack_prepare_calibration_presets() { return 1; }
    run pack_run_suite
    assert_rc "1" "${RUN_RC}" "pack_prepare_calibration_presets failure returns non-zero"
    trap - EXIT INT TERM HUP QUIT

    OUTPUT_DIR="${TEST_TMPDIR}/out_fail_validate_calibration"
    pack_prepare_calibration_presets() { return 0; }
    pack_validate_guard_calibration() { return 1; }
    run pack_run_suite
    assert_rc "1" "${RUN_RC}" "pack_validate_guard_calibration failure returns non-zero"
    trap - EXIT INT TERM HUP QUIT
    pack_validate_guard_calibration() { return 0; }

    OUTPUT_DIR="${TEST_TMPDIR}/out_fail_runtime_provenance"
    pack_validate_runtime_provenance() { return 1; }
    run pack_run_suite
    assert_rc "1" "${RUN_RC}" "pack_validate_runtime_provenance failure returns non-zero"
    trap - EXIT INT TERM HUP QUIT
    pack_validate_runtime_provenance() { return 0; }

    pack_model_list_array() { PACK_MODEL_LIST=(); }
    local error_log="${TEST_TMPDIR}/error.calls"
    : > "${error_log}"
    error_exit() { echo "$1" >> "${error_log}"; return 0; }
    OUTPUT_DIR="${TEST_TMPDIR}/out2"
    run pack_run_suite
    assert_rc "0" "${RUN_RC}" "missing model list triggers error_exit"
    assert_match "No models configured" "$(cat "${error_log}")" "error_exit called for empty models"
    trap - EXIT INT TERM HUP QUIT

    pack_model_list_array() { PACK_MODEL_LIST=("model"); }
    pack_preflight_models() { echo "preflight" > "${TEST_TMPDIR}/preflight.calls"; }
    OUTPUT_DIR="${TEST_TMPDIR}/out3"
    PACK_NET="1"
    run pack_run_suite
    assert_rc "0" "${RUN_RC}" "preflight path succeeds"
    assert_file_exists "${TEST_TMPDIR}/preflight.calls" "preflight invoked"
    trap - EXIT INT TERM HUP QUIT

    pack_preflight_models() { return 1; }
    OUTPUT_DIR="${TEST_TMPDIR}/out3_fail_preflight"
    PACK_NET="1"
    run pack_run_suite
    assert_rc "1" "${RUN_RC}" "preflight failure returns non-zero"
    trap - EXIT INT TERM HUP QUIT

    PACK_NET="0"
    pack_load_model_revisions() { return 1; }
    local offline_log="${TEST_TMPDIR}/offline.calls"
    : > "${offline_log}"
    error_exit() { echo "$1" >> "${offline_log}"; return 0; }
    OUTPUT_DIR="${TEST_TMPDIR}/out4"
    run pack_run_suite
    assert_rc "0" "${RUN_RC}" "offline revisions failure triggers error_exit"
    assert_match "Offline mode requires" "$(cat "${offline_log}")" "offline error recorded"
    trap - EXIT INT TERM HUP QUIT
}

test_pack_validation_pack_run_suite_calibrate_only_skips_tuned_edit_params_validation() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    cleanup() { return 0; }
    pack_require_bash4() { return 0; }
    pack_apply_network_mode() { :; }
    pack_source_libs() { return 0; }
    pack_prepare_scenarios_manifest() { return 0; }
    pack_setup_hf_cache_dirs() { return 0; }
    pack_preflight_datasets() { :; }
    pack_validate_runtime_provenance() { :; }

    # Ensure the model list would fail tuned preset validation if it ran.
    pack_model_list_array() { PACK_MODEL_LIST=("Qwen/Qwen2.5-14B"); }

    PACK_SUITE_MODE="calibrate-only"
    PACK_SUITE="full"
    PACK_NET="0"
    OUTPUT_DIR="${TEST_TMPDIR}/out_calib_only"
    export PACK_SUITE_MODE PACK_SUITE PACK_NET OUTPUT_DIR

    pack_load_model_revisions() { return 0; }
    main_dynamic() { :; }

    run pack_run_suite
    assert_rc "0" "${RUN_RC}" "calibrate-only skips tuned edit preset validation"
    trap - EXIT INT TERM HUP QUIT
}

test_pack_validation_pack_run_suite_errors_only_skips_tuned_edit_params_validation() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    cleanup() { return 0; }
    pack_require_bash4() { return 0; }
    pack_apply_network_mode() { :; }
    pack_source_libs() { return 0; }
    pack_prepare_scenarios_manifest() { return 0; }
    pack_setup_hf_cache_dirs() { return 0; }
    pack_preflight_datasets() { :; }
    pack_validate_runtime_provenance() { :; }

    # Ensure the model list would fail tuned preset validation if it ran.
    pack_model_list_array() { PACK_MODEL_LIST=("Qwen/Qwen2.5-14B"); }

    PACK_SUITE_MODE="errors-only"
    PACK_SUITE="full"
    PACK_NET="0"
    OUTPUT_DIR="${TEST_TMPDIR}/out_errors_only"
    export PACK_SUITE_MODE PACK_SUITE PACK_NET OUTPUT_DIR

    pack_load_model_revisions() { return 0; }
    main_dynamic() {
        printf '%s\n' "${CLEAN_EDIT_RUNS:-}|${STRESS_EDIT_RUNS:-}|${RUN_ERROR_INJECTION:-}" > "${TEST_TMPDIR}/errors_only.env"
    }

    run pack_run_suite
    assert_rc "0" "${RUN_RC}" "errors-only skips tuned edit preset validation"
    assert_file_exists "${TEST_TMPDIR}/errors_only.env" "main_dynamic invoked"
    assert_eq "0|0|true" "$(cat "${TEST_TMPDIR}/errors_only.env")" "errors-only disables edits but keeps error injection enabled"
    trap - EXIT INT TERM HUP QUIT
}
