#!/usr/bin/env bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/validation_suite_test_helpers.sh"

test_pack_validation_main_dynamic_marks_suite_failed_when_final_verdict_fails() {
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
        mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed} "${OUTPUT_DIR}/workers" "${OUTPUT_DIR}/reports"
        export QUEUE_DIR
    }
    generate_all_tasks() { :; }
    resolve_dependencies() { echo 0; }
    count_tasks() { echo 0; }
    print_queue_stats() { :; }
    compile_results() { :; }
    run_analysis() { :; }
    generate_verdict() {
        mkdir -p "${OUTPUT_DIR}/reports"
        printf '%s\n' '{"verdict":"FAIL"}' > "${OUTPUT_DIR}/reports/final_verdict.json"
        printf '%s\n' 'FAIL' > "${OUTPUT_DIR}/reports/final_verdict.txt"
    }
    list_run_gpu_ids() { printf '0\n'; }
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
    python3() {
        if [[ "${1:-}" == */validation_state.py && "${2:-}" == "evaluation-optimization-summary" ]]; then
            return 1
        fi
        command "${TEST_REAL_PYTHON3}" "$@"
    }

    run main_dynamic
    unset -f python3
    assert_rc "1" "${RUN_RC}" "failed final verdict makes suite fail"
    assert_match "Final verdict is FAIL" "${RUN_OUT}${RUN_ERR}" "failure reason is logged"
    assert_match "Failed to write evaluation optimization summary" "${RUN_OUT}${RUN_ERR}" "optimization summary failure is logged as a warning"
}

test_pack_validation_pack_run_suite_returns_nonzero_when_dataset_preflight_fails() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    cleanup() { return 0; }
    pack_apply_network_mode() { :; }
    pack_source_libs() { return 0; }
    pack_setup_output_dirs() { return 0; }
    pack_prepare_scenarios_manifest() { return 0; }
    pack_setup_hf_cache_dirs() { return 0; }
    pack_preflight_datasets() { return 1; }

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    PACK_NET="0"
    pack_require_bash4() { return 0; }

    run pack_run_suite
    assert_rc "1" "${RUN_RC}" "dataset preflight failure propagates from pack_run_suite"
    trap - EXIT INT TERM HUP QUIT
}

test_pack_prepare_scenarios_manifest_filters_source_without_jq() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    _pack_validation_has_jq() { return 1; }

    local manifest="${TEST_TMPDIR}/scenarios_copy.json"
    cat > "${manifest}" <<'EOF'
{
  "_meta": {},
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [{"id": "a", "generation": {"kind": "edit", "edit_spec": "x", "version": "clean"}}]
}
EOF

    PACK_SCENARIOS_MANIFEST_FILE="${manifest}"
    PACK_SCENARIO_IDS=""

    pack_prepare_scenarios_manifest
    assert_eq "subset" "$(jq -r '._meta.applied_suite' "${OUTPUT_DIR}/state/scenarios.json")" "python renderer records suite metadata"
    assert_eq "a" "$(jq -r '.scenarios[0].id' "${OUTPUT_DIR}/state/scenarios.json")" "scenario content preserved under cp fallback"
}

test_pack_validation_estimate_planned_storage_covers_error_fallback_and_batch_cleanup() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    HF_HUB_CACHE=""
    MODEL_1="mistralai/Mistral-7B-v0.1"
    MODEL_2=""
    MODEL_3=""
    MODEL_4=""
    MODEL_5=""
    MODEL_6=""
    MODEL_7=""
    MODEL_8=""
    RUN_ERROR_INJECTION="true"
    PACK_CLEANUP_MODELS="1"
    PACK_USE_BATCH_EDITS="true"
    CLEAN_EDIT_RUNS="oops"
    STRESS_EDIT_RUNS="bad"
    jq() { echo "not-a-number"; }

    local total
    total="$(estimate_planned_model_storage_gb)"
    assert_match '^[0-9]+$' "${total}" "storage estimate falls back when error count is invalid"

    CLEAN_EDIT_RUNS="1"
    STRESS_EDIT_RUNS="2"
    PACK_USE_BATCH_EDITS="true"
    total="$(estimate_planned_model_storage_gb)"
    assert_match '^[0-9]+$' "${total}" "batch cleanup estimate supports truthy pack_use_batch_edits"
}

test_pack_validation_setup_model_early_returns_for_local_or_cached_paths_and_errors_on_failed_download() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs
    PACK_NET=1
    pack_model_revision() { echo "rev"; }

    local local_model="${TEST_TMPDIR}/local_model"
    mkdir -p "${local_model}"
    assert_eq "${local_model}" "$(setup_model "${local_model}" 0)" "local path returns unchanged"

    local model_id="Test/Model"
    local model_name
    model_name="$(sanitize_model_name "${model_id}")"
    mkdir -p "${OUTPUT_DIR}/models/${model_name}/baseline"
    assert_eq "${OUTPUT_DIR}/models/${model_name}/baseline" "$(setup_model "${model_id}" 0)" "cached sanitized baseline preferred"

    rm -rf "${OUTPUT_DIR}/models/${model_name}"
    mkdir -p "${OUTPUT_DIR}/models/model/baseline"
    assert_eq "${OUTPUT_DIR}/models/model/baseline" "$(setup_model "${model_id}" 0)" "cached basename baseline honored"

    rm -rf "${OUTPUT_DIR}/models"
    mkdir -p "${OUTPUT_DIR}/models"

    mock_python3_stub_enable
    local rc=0
    local out
    set +e
    out="$(setup_model "Remote/NoCache" 0)"
    rc=$?
    set -e
    assert_ne "0" "${rc}" "stubbed download without marker fails"
    assert_eq "" "${out}" "failed download returns empty baseline path"
}

test_pack_validation_setup_model_cleans_incomplete_baseline_dir_on_download_failure() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs
    PACK_NET=1
    pack_model_revision() { echo "rev"; }

    local model_id="Remote/Incomplete"
    local model_name
    model_name="$(sanitize_model_name "${model_id}")"
    local baseline_path="${OUTPUT_DIR}/models/${model_name}/baseline"

	local bin_dir="${TEST_TMPDIR}/bin"
	mkdir -p "${bin_dir}"
	cat > "${bin_dir}/python3" <<'EOF'
#!/usr/bin/env bash
	set -euo pipefail

count_file="${TEST_TMPDIR:-}/python3.count"
count=0
if [[ -f "${count_file}" ]]; then
    count="$(cat "${count_file}" 2>/dev/null || echo "0")"
fi
	count=$((count + 1))
	printf '%s\n' "${count}" > "${count_file}"

	output_dir=""
	while [[ $# -gt 0 ]]; do
	    case "$1" in
	        --output-dir)
	            output_dir="${2:-}"
	            shift 2
	            ;;
	        *)
	            shift
	            ;;
	    esac
	done
	[[ -n "${output_dir}" ]] && mkdir -p "${output_dir}"

	# Simulate a download failure by not creating the success marker.
	exit 1
EOF
    chmod +x "${bin_dir}/python3"
    export PATH="${bin_dir}:$PATH"
    hash -r 2>/dev/null || true

    local rc=0
    local out
    set +e
    out="$(setup_model "${model_id}" 0)"
    rc=$?
    set -e
    assert_ne "0" "${rc}" "download failure returns non-zero"
    assert_eq "" "${out}" "download failure returns empty baseline path"
    if [[ -d "${baseline_path}" ]]; then
        t_fail "baseline dir should be removed after download failure baseline_path='${baseline_path}'"
    fi
    assert_eq "1" "$(cat "${TEST_TMPDIR}/python3.count")" "python3 invoked for first attempt"

    set +e
    out="$(setup_model "${model_id}" 0)"
    rc=$?
    set -e
    assert_ne "0" "${rc}" "second attempt should not treat incomplete baseline as cached success"
    assert_eq "" "${out}" "second attempt still returns empty baseline path"
    assert_eq "2" "$(cat "${TEST_TMPDIR}/python3.count")" "python3 invoked again (no stale cached baseline dir)"
}

test_pack_validation_setup_model_succeeds_when_python_stub_creates_success_marker() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs
    PACK_NET=1
    pack_model_revision() { echo "rev"; }

	local bin_dir="${TEST_TMPDIR}/bin"
	mkdir -p "${bin_dir}"
	cat > "${bin_dir}/python3" <<'EOF'
#!/usr/bin/env bash
	set -euo pipefail

	output_dir=""
	marker=""
	while [[ $# -gt 0 ]]; do
	    case "$1" in
	        --output-dir)
	            output_dir="${2:-}"
	            shift 2
	            ;;
	        --success-marker)
	            marker="${2:-}"
	            shift 2
	            ;;
	        *)
	            shift
	            ;;
	    esac
	done

	[[ -n "${output_dir}" ]] && mkdir -p "${output_dir}"
	[[ -n "${marker}" ]] && : > "${marker}"
	exit 0
EOF
    chmod +x "${bin_dir}/python3"
    export PATH="${bin_dir}:$PATH"
    hash -r 2>/dev/null || true

    assert_eq "${bin_dir}/python3" "$(command -v python3)" "python3 resolves to the test stub"

    local rc=0
    local out
    set +e
    out="$(setup_model "Remote/WithMarker" 0)"
    rc=$?
    set -e

    assert_rc "0" "${rc}" "setup_model returns success when marker is present"
    assert_match "/baseline$" "${out}" "returns baseline path on success"
    assert_dir_exists "${out}" "baseline directory created"
}

test_pack_validation_estimate_model_params_defaults_to_7_without_config() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local model_dir="${TEST_TMPDIR}/model"
    mkdir -p "${model_dir}"
    assert_eq "7" "$(estimate_model_params "${model_dir}")" "missing config defaults to 7B"
}

test_pack_validation_estimate_model_params_classifies_when_config_is_present() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local model_dir="${TEST_TMPDIR}/model"
    mkdir -p "${model_dir}"
    cat > "${model_dir}/config.json" <<'EOF'
{"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32000}
EOF

    assert_eq "7" "$(estimate_model_params "${model_dir}")" "config-based estimation classifies small bucket"
}

test_pack_validation_get_model_invarlock_config_covers_all_case_arms() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    assert_eq "512:512:64:64:96" "$(get_model_invarlock_config 7)" "7B config"
    assert_eq "512:512:64:64:64" "$(get_model_invarlock_config 13)" "13B config"
    assert_eq "1024:512:40:40:48" "$(get_model_invarlock_config 30)" "30B config"
    assert_eq "1024:512:36:36:32" "$(get_model_invarlock_config 40)" "40B config"
    assert_eq "1024:512:40:40:8" "$(get_model_invarlock_config moe)" "moe config"
    assert_eq "128:64:8:8:2" "$(get_model_invarlock_config 70)" "70B config"
    assert_eq "1024:512:40:40:32" "$(get_model_invarlock_config unknown)" "default config"
}

test_pack_validation_create_edited_model_quant_rtn_and_unknown_edit_type_branches() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    mock_python3_stub_enable

    create_edited_model "${TEST_TMPDIR}/baseline" "${TEST_TMPDIR}/edited" "quant_rtn" "8" "128" "ffn" "0"

    error_exit() { exit 4; }
    local rc=0
    ( create_edited_model "${TEST_TMPDIR}/baseline" "${TEST_TMPDIR}/edited" "nope" "8" "128" "ffn" "0" ) || rc=$?
    assert_eq "4" "${rc}" "unknown edit type aborts via error_exit"
}

test_pack_validation_generate_invarlock_config_attn_and_strict_accelerator_flags() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    FLASH_ATTENTION_AVAILABLE="true"
    PACK_DETERMINISM="strict"
    local cfg="${TEST_TMPDIR}/cfg.yaml"
    generate_invarlock_config "model" "${cfg}" "edit"
    assert_file_exists "${cfg}" "config generated"
    assert_match "flash_attention_2" "$(cat "${cfg}")" "attn implementation emitted"

    FLASH_ATTENTION_AVAILABLE="false"
    PACK_DETERMINISM="throughput"
    generate_invarlock_config "model" "${cfg}" "edit"
    assert_match "flash_attention_2 not available" "$(cat "${cfg}")" "comment emitted when FA2 unavailable"
}

test_pack_validation_run_single_calibration_large_model_and_report_copy_branch() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    mock_python3_stub_enable
    fixture_write "python3.create_report" ""
    estimate_model_params() { echo "${MODEL_SIZE_RETURN}"; }

    local run_dir="${TEST_TMPDIR}/cal/run1"
    local log_file="${TEST_TMPDIR}/cal/run.log"
    MODEL_SIZE_RETURN="70"
    run_single_calibration "model" "${run_dir}" "42" "1" "1" "1" "${log_file}" "0"
    assert_file_exists "${run_dir}/baseline_report.json" "report copied when present"

    MODEL_SIZE_RETURN="7"
    run_single_calibration "model" "${TEST_TMPDIR}/cal/run2" "42" "1" "1" "1" "${log_file}" "0"
}

test_pack_validation_run_invarlock_calibration_failure_paths_and_labels() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    mock_python3_stub_enable

    estimate_model_params() { echo "${MODEL_SIZE_RETURN}"; }

    MODEL_SIZE_RETURN="moe"
    run_single_calibration() { return 0; }
    run_invarlock_calibration "model" "mixtral" "${TEST_TMPDIR}/cal/moe" "1" "${TEST_TMPDIR}/presets" "0"

    MODEL_SIZE_RETURN="7"
    run_single_calibration() { return 1; }
    local rc=0
    ( run_invarlock_calibration "model" "small" "${TEST_TMPDIR}/cal/fail" "2" "${TEST_TMPDIR}/presets" "0" ) || rc=$?
    assert_ne "0" "${rc}" "all calibration runs failing returns non-zero"
}

test_pack_validation_run_invarlock_evaluate_preset_optional_and_cert_copy_paths() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    mock_python3_stub_enable

    local preset_dir="${TEST_TMPDIR}/presets"
    mkdir -p "${preset_dir}"
    : > "${preset_dir}/calibrated_preset_model.yaml"

    estimate_model_params() { echo "${MODEL_SIZE_RETURN}"; }

    fixture_write "invarlock.create_cert" ""
    MODEL_SIZE_RETURN="70"
    run_invarlock_evaluate "subject" "baseline" "${TEST_TMPDIR}/certs" "run_ok" "${preset_dir}" "model" "0"

    # alt-cert path when canonical cert missing
    rm -f "${TEST_TMPDIR}/fixtures/invarlock.create_cert"
    local cert_dir="${TEST_TMPDIR}/certs/run_alt/cert/nested"
    mkdir -p "${cert_dir}"
    printf '{"ok":true}\n' > "${cert_dir}/evaluation.report.json"
    MODEL_SIZE_RETURN="7"
    run_invarlock_evaluate "subject" "baseline" "${TEST_TMPDIR}/certs" "run_alt" "${preset_dir}" "model" "0"
}

test_pack_validation_main_dynamic_resume_and_monitoring_branches_offline() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    # Stub early heavyweight phases.
    check_dependencies() { :; }
    configure_gpu_pool() { NUM_GPUS=2; GPU_ID_LIST="0,1"; export NUM_GPUS GPU_ID_LIST; }
    disk_preflight() { :; }
    setup_pack_environment() { :; }
    handle_disk_pressure() { echo "disk_pressure:$1:$2" >> "${TEST_TMPDIR}/disk_pressure.calls"; return 0; }

    # Existing queue with tasks for --resume branch coverage.
    RESUME_FLAG="true"
    PACK_RETRY_FAILED_ON_RESUME="1"
    mkdir -p "${OUTPUT_DIR}/queue"/{pending,ready,running,completed,failed}
    printf '{"id":"t1","status":"running"}\n' > "${OUTPUT_DIR}/queue/running/t1.task"
    printf '{"id":"t2","status":"failed"}\n' > "${OUTPUT_DIR}/queue/failed/t2.task"

    init_queue() {
        QUEUE_DIR="${OUTPUT_DIR}/queue"
        GPU_RESERVATION_DIR="${OUTPUT_DIR}/gpu_reservations"
        mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed} "${OUTPUT_DIR}/workers" "${GPU_RESERVATION_DIR}"
        export QUEUE_DIR GPU_RESERVATION_DIR
    }
    init_gpu_reservations() { GPU_RESERVATION_DIR="${OUTPUT_DIR}/gpu_reservations"; export GPU_RESERVATION_DIR; }
    refresh_task_memory_from_profiles() { echo "refresh:$1" >> "${TEST_TMPDIR}/mem.calls"; }
    export_memory_plan() { echo "plan:$1" >> "${TEST_TMPDIR}/mem.calls"; }
    resolve_dependencies() { echo 1; }
    cancel_tasks_with_failed_dependencies() { echo 2; }
    get_queue_stats() { echo ""; }
    apply_work_stealing_boost() { :; }
    reclaim_orphaned_tasks() { echo "reclaim:$1" >> "${TEST_TMPDIR}/reclaim.calls"; }
    count_tasks() {
        if [[ "${1:-}" == "failed" ]]; then
            echo 1
        else
            echo 0
        fi
    }
    print_queue_stats() { :; }
    compile_results() { :; }
    run_analysis() { :; }
    generate_verdict() { :; }

    list_run_gpu_ids() { printf '0\n1\n'; }

    local empty_checks=0
    is_queue_empty() {
        empty_checks=$((empty_checks + 1))
        [[ ${empty_checks} -ge 2 ]]
    }

    # Stub worker scripts so start_worker doesn't run real gpu_worker.
    local stub_lib="${TEST_TMPDIR}/stub_lib"
    mkdir -p "${stub_lib}"
    for f in task_serialization.sh queue_manager.sh scheduler.sh task_functions.sh fault_tolerance.sh; do
        printf '%s\n' "#!/usr/bin/env bash" > "${stub_lib}/${f}"
    done
    cat > "${stub_lib}/gpu_worker.sh" <<'EOF'
#!/usr/bin/env bash
gpu_worker() {
    local gpu_id="$1"
    local output_dir="$2"
    mkdir -p "${output_dir}/workers"
    echo "searching" > "${output_dir}/workers/gpu_${gpu_id}.status"
    : > "${output_dir}/workers/gpu_${gpu_id}.heartbeat"
    if [[ "${gpu_id}" == "1" ]]; then
        return 1
    fi
    return 0
}
EOF

    LIB_DIR="${stub_lib}"
    export LIB_DIR

    WORKER_TIMEOUT=1
    MIN_FREE_DISK_GB="bogus"
    get_free_disk_gb() { echo "1"; }

    # Ensure heartbeat appears stale for GPU 0 so "stuck" branch triggers.
    fixture_append "stat/mtime" "$(printf '%s %s\n' "${OUTPUT_DIR}/workers/gpu_0.heartbeat" "1699990000")"

    kill() {
        local sig="${1:-}"
        local pid="${2:-}"
        if [[ "${sig}" == "-0" ]]; then
            local pid0
            pid0="$(cat "${OUTPUT_DIR}/workers/gpu_0.pid" 2>/dev/null || echo "")"
            [[ -n "${pid0}" && "${pid}" == "${pid0}" ]] && return 0
            return 1
        fi
        return 0
    }

    signal_shutdown() { echo "shutdown:$1" >> "${TEST_TMPDIR}/shutdown.calls"; }

    run main_dynamic
    assert_rc "1" "${RUN_RC}" "main_dynamic fails closed when worker/task failures occur"
    assert_file_exists "${TEST_TMPDIR}/shutdown.calls" "signal_shutdown called on empty queue"
}

test_pack_validation_main_dynamic_skips_live_worker_and_restarts_stale_heartbeat() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out_live_worker"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    check_dependencies() { :; }
    configure_gpu_pool() { NUM_GPUS=1; GPU_ID_LIST="0"; export NUM_GPUS GPU_ID_LIST; }
    pack_model_list_array() { PACK_MODEL_LIST=("org/model"); }
    pack_model_list() { printf '%s\n' "org/model"; }
    disk_preflight() { :; }
    setup_pack_environment() { :; }
    handle_disk_pressure() { return 0; }
    get_free_disk_gb() { echo "999"; }
    pack_count_edit_scenarios() { printf '%s\n' "0|0|fixture"; }
    _pack_validation_state() {
        if [[ "${1:-}" == "count-generation-kind" ]]; then
            printf '%s\n' "0"
            return 0
        fi
        return 0
    }

    RESUME_FLAG="false"
    WORKER_TIMEOUT="1"
    MIN_FREE_DISK_GB="200"

    init_queue() {
        QUEUE_DIR="${OUTPUT_DIR}/queue"
        mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed} "${OUTPUT_DIR}/workers" "${OUTPUT_DIR}/logs"
        printf '%s\n' "4242" > "${OUTPUT_DIR}/workers/gpu_0.pid"
        : > "${OUTPUT_DIR}/workers/gpu_0.heartbeat"
        printf '%s\n' "searching" > "${OUTPUT_DIR}/workers/gpu_0.status"
        export QUEUE_DIR
    }
    generate_all_tasks() { :; }
    refresh_task_memory_from_profiles() { :; }
    export_memory_plan() { :; }
    resolve_dependencies() { echo 0; }
    reclaim_orphaned_tasks() { echo "reclaim:$1" >> "${TEST_TMPDIR}/reclaim.calls"; }
    cancel_tasks_with_failed_dependencies() { echo 0; }
    queue_terminal_state() { echo ""; }
    apply_work_stealing_boost() { echo "boost" >> "${TEST_TMPDIR}/boost.calls"; }
    count_tasks() { echo 0; }
    print_queue_stats() { :; }
    get_queue_stats() { echo "0:0:1:0:0:1"; }
    compile_results() { :; }
    run_analysis() { :; }
    generate_verdict() {
        mkdir -p "${OUTPUT_DIR}/reports"
        printf '%s\n' '{"verdict":"PASS"}' > "${OUTPUT_DIR}/reports/final_verdict.json"
        printf '%s\n' 'PASS' > "${OUTPUT_DIR}/reports/final_verdict.txt"
    }
    pack_read_final_verdict() { echo "PASS"; }
    signal_shutdown() { echo "shutdown:$1" >> "${TEST_TMPDIR}/shutdown.calls"; }

    local empty_checks=0
    is_queue_empty() {
        empty_checks=$((empty_checks + 1))
        [[ ${empty_checks} -ge 2 ]]
    }
    sleep() { :; }
    date() {
        case "${1:-}" in
            +%s) echo "2000" ;;
            +%Y-%m-%dT%H:%M:%SZ) echo "2025-01-01T00:00:00Z" ;;
            *) echo "2025-01-01 00:00:00" ;;
        esac
    }
    stat() {
        if [[ "${1:-}" == "-c" || "${1:-}" == "-f" ]]; then
            echo "1"
            return 0
        fi
        command stat "$@"
    }
    kill() {
        printf '%s\n' "$*" >> "${TEST_TMPDIR}/kill.calls"
        if [[ "${1:-}" == "-0" && "${2:-}" == "4242" ]]; then
            return 0
        fi
        return 0
    }
    python3() { return 0; }

    main_dynamic

    assert_match "worker already running" "$(cat "${LOG_FILE}")" "live worker pid is not duplicated"
    assert_match "stuck \\(no heartbeat" "$(cat "${LOG_FILE}")" "stale heartbeat is detected"
    assert_file_exists "${TEST_TMPDIR}/shutdown.calls" "empty queue signals shutdown"
    assert_file_exists "${TEST_TMPDIR}/boost.calls" "monitor loop reached progress path"
}

test_pack_validation_main_dynamic_starts_worker_with_nested_lib_and_logs_source_status() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out_nested_worker"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    check_dependencies() { :; }
    configure_gpu_pool() { NUM_GPUS=1; GPU_ID_LIST="0"; export NUM_GPUS GPU_ID_LIST; }
    pack_model_list_array() { PACK_MODEL_LIST=("org/model"); }
    pack_model_list() { printf '%s\n' "org/model"; }
    disk_preflight() { :; }
    setup_pack_environment() { :; }
    handle_disk_pressure() { return 0; }
    get_free_disk_gb() { echo "999"; }
    pack_count_edit_scenarios() { printf '%s\n' "0|0|fixture"; }

    RESUME_FLAG="false"

    init_queue() {
        QUEUE_DIR="${OUTPUT_DIR}/queue"
        mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed} "${OUTPUT_DIR}/workers" "${OUTPUT_DIR}/logs"
        export QUEUE_DIR
    }
    generate_all_tasks() { :; }
    refresh_task_memory_from_profiles() { :; }
    export_memory_plan() { :; }
    resolve_dependencies() { echo 0; }
    count_tasks() { echo 0; }
    print_queue_stats() { :; }
    get_queue_stats() { echo "0:0:0:0:0:0"; }
    is_queue_empty() { return 0; }
    sleep() { :; }
    compile_results() { :; }
    run_analysis() { :; }
    generate_verdict() {
        mkdir -p "${OUTPUT_DIR}/reports"
        printf '%s\n' '{"verdict":"PASS"}' > "${OUTPUT_DIR}/reports/final_verdict.json"
        printf '%s\n' 'PASS' > "${OUTPUT_DIR}/reports/final_verdict.txt"
    }
    pack_read_final_verdict() { echo "PASS"; }
    python3() { return 0; }

    local stub_lib="${TEST_TMPDIR}/nested_lib"
    mkdir -p "${stub_lib}/tasks" "${stub_lib}/queue" "${stub_lib}/core"
    cat > "${stub_lib}/tasks/task_serialization.sh" <<'EOF'
#!/usr/bin/env bash
get_task_id() { echo "task"; }
EOF
    cat > "${stub_lib}/queue/queue_manager.sh" <<'EOF'
#!/usr/bin/env bash
count_tasks() { echo 0; }
EOF
    cat > "${stub_lib}/queue/scheduler.sh" <<'EOF'
#!/usr/bin/env bash
return 4
EOF
    cat > "${stub_lib}/tasks/task_functions.sh" <<'EOF'
#!/usr/bin/env bash
execute_task() { return 0; }
EOF
    cat > "${stub_lib}/queue/gpu_worker.sh" <<'EOF'
#!/usr/bin/env bash
gpu_worker() {
    local gpu_id="$1"
    local output_dir="$2"
    echo "started" > "${output_dir}/workers/gpu_${gpu_id}.nested_started"
    return 0
}
EOF
    printf '%s\n' "#!/usr/bin/env bash" > "${stub_lib}/core/fault_tolerance.sh"
    LIB_DIR="${stub_lib}"
    export LIB_DIR

    main_dynamic

    assert_file_exists "${OUTPUT_DIR}/workers/gpu_0.nested_started" "nested-layout worker started"
    assert_match "source status scheduler rc=4" "$(cat "${OUTPUT_DIR}/logs/gpu_0.log")" "nonzero source status is logged"
}

test_pack_validation_main_dynamic_resume_requires_explicit_failed_retry() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    check_dependencies() { :; }
    configure_gpu_pool() { NUM_GPUS=1; GPU_ID_LIST="0"; export NUM_GPUS GPU_ID_LIST; }
    disk_preflight() { :; }
    setup_pack_environment() { :; }

    RESUME_FLAG="true"
    mkdir -p "${OUTPUT_DIR}/queue"/{pending,ready,running,completed,failed}
    printf '{"id":"t2","status":"failed"}\n' > "${OUTPUT_DIR}/queue/failed/t2.task"

    init_queue() {
        QUEUE_DIR="${OUTPUT_DIR}/queue"
        GPU_RESERVATION_DIR="${OUTPUT_DIR}/gpu_reservations"
        mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed} "${GPU_RESERVATION_DIR}"
        export QUEUE_DIR GPU_RESERVATION_DIR
    }

    error_exit() {
        printf '%s\n' "$*" > "${TEST_TMPDIR}/error_exit.txt"
        exit 98
    }

    local rc=0
    ( main_dynamic ) || rc=$?

    assert_eq "98" "${rc}" "resume aborts when failed tasks exist without explicit retry"
    assert_match "PACK_RETRY_FAILED_ON_RESUME=1" "$(cat "${TEST_TMPDIR}/error_exit.txt")" "retry opt-in guidance is emitted"
}
