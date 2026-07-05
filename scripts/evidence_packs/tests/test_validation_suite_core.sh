#!/usr/bin/env bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/validation_suite_test_helpers.sh"

test_pack_validation_cleanup_kills_spawned_pids_and_exits_with_previous_rc() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local rc=0
    (
        set +e
        pids=(111 222)
        LOG_LOCK="${TEST_TMPDIR}/log.lock"

        kill() {
            local sig="${1:-}"
            local pid="${2:-}"
            if [[ "${sig}" == "-0" ]]; then
                [[ "${pid}" == "111" ]]
                return $?
            fi
            return 0
        }

        false
        cleanup
    ) || rc=$?

    assert_rc "1" "${rc}" "cleanup exits with previous rc"
}

test_pack_validation_determinism_strict_sets_compile_off() {
    mock_reset

    PACK_DETERMINISM="strict"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    assert_eq "strict" "${PACK_DETERMINISM}" "strict preserved"
    assert_eq "0" "${NVIDIA_TF32_OVERRIDE}" "strict disables TF32"
    assert_eq "0" "${CUDNN_BENCHMARK}" "strict disables cuDNN benchmark"
    assert_eq ":4096:8" "${CUBLAS_WORKSPACE_CONFIG-}" "strict forces cublas workspace"
}

test_pack_validation_determinism_invalid_defaults_to_throughput() {
    mock_reset

    PACK_DETERMINISM="not-a-preset"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    assert_eq "throughput" "${PACK_DETERMINISM}" "invalid preset coerces to throughput"
    assert_eq "1" "${NVIDIA_TF32_OVERRIDE}" "throughput enables TF32"
    assert_eq "1" "${CUDNN_BENCHMARK}" "throughput enables cuDNN benchmark"
    assert_eq "" "${CUBLAS_WORKSPACE_CONFIG-}" "throughput unsets cublas workspace"
}

test_pack_validation_bash4_guard_reports_error_on_bash3() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    # Simulate bash3 (non-bash4) regardless of the host bash version.
    pack_is_bash4() { return 1; }
    local rc=0
    if pack_require_bash4; then
        rc=0
    else
        rc=$?
    fi
    assert_ne "0" "${rc}" "bash4 guard should fail under bash 3"
}

test_pack_validation_bash4_guard_succeeds_when_bash4_is_reported() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    pack_is_bash4() { return 0; }
    pack_require_bash4
}

test_pack_validation_default_edit_type_sets_include_generated_lora_and_fine_tune() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local clean_joined stress_joined
    clean_joined="$(printf '%s\n' "${EDIT_TYPES_CLEAN[@]}")"
    stress_joined="$(printf '%s\n' "${EDIT_TYPES_STRESS[@]}")"

    assert_match "lora_merge:clean:attn" "${clean_joined}" "clean defaults include generated lora"
    assert_match "fine_tune:clean:ffn" "${clean_joined}" "clean defaults include generated fine-tune"
    assert_match "lora_merge:8:64:all" "${stress_joined}" "stress defaults include generated lora"
    assert_match "fine_tune:0.0005:3:all" "${stress_joined}" "stress defaults include generated fine-tune"
}

test_pack_validation_source_preserves_callers_script_dir() {
    mock_reset

    SCRIPT_DIR="${TEST_TMPDIR}/caller-script-dir"
    export SCRIPT_DIR

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    assert_eq "${TEST_TMPDIR}/caller-script-dir" "${SCRIPT_DIR}" "validation suite sourcing preserves caller SCRIPT_DIR"
    unset SCRIPT_DIR
}

test_pack_validation_pack_is_bash4_default_impl_executes() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    pack_is_bash4 || true
}

test_pack_validation_worker_sources_dependencies_with_inherited_loaded_flags() {
    mock_reset

    local rc=0
    (
        set -euo pipefail
        export SCHEDULER_LOADED=1
        export TASK_FUNCTIONS_LOADED=1
        export MODEL_CREATION_LOADED=1
        export TASK_SERIALIZATION_LOADED=1
        export FAULT_TOLERANCE_LOADED=1
        export QUEUE_CORE_LOADED=1
        export QUEUE_LIFECYCLE_LOADED=1
        export QUEUE_DEPENDENCIES_LOADED=1
        export QUEUE_MEMORY_PLAN_LOADED=1
        export QUEUE_GENERATION_LOADED=1
        export SCHEDULER_CORE_LOADED=1
        export SCHEDULER_GPU_RUNTIME_LOADED=1
        export SCHEDULER_RESERVATIONS_LOADED=1
        export SCHEDULER_SELECTION_LOADED=1
        resolve_edit_params() { :; }

        source ./scripts/evidence_packs/lib/queue/gpu_worker.sh

        declare -F find_and_claim_task >/dev/null
        declare -F execute_task >/dev/null
        declare -F create_model_variant >/dev/null
        declare -F get_task_field >/dev/null
        declare -F _get_task_timeout >/dev/null
        declare -F should_retry_task >/dev/null
        ! env | grep -q '^SCHEDULER_LOADED='
        ! env | grep -q '^TASK_FUNCTIONS_LOADED='
    ) || rc=$?

    assert_rc "0" "${rc}" "worker must source dependencies even when child shell inherits stale loaded flags"
}

test_pack_validation_setup_hf_cache_dirs_errors_when_home_is_file() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local hf_home="${TEST_TMPDIR}/hf_file"
    : > "${hf_home}"
    export HF_HOME="${hf_home}"

    local rc=0
    if pack_setup_hf_cache_dirs; then
        rc=0
    else
        rc=$?
    fi
    assert_ne "0" "${rc}" "expected mkdir failure when HF_HOME is a file"
}

test_pack_validation_setup_hf_cache_dirs_creates_directories_and_returns_zero() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    export HF_HOME="${TEST_TMPDIR}/hf"
    unset HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE

    pack_setup_hf_cache_dirs
    assert_dir_exists "${HF_HOME}" "HF_HOME created"
    assert_dir_exists "${HF_HOME}/hub" "HF_HUB_CACHE created"
    if [[ -n "${TRANSFORMERS_CACHE:-}" ]]; then
        t_fail "TRANSFORMERS_CACHE should remain unset (Transformers v5 uses HF_HOME)"
    fi
    if [[ -d "${HF_HOME}/transformers" ]]; then
        t_fail "TRANSFORMERS_CACHE directory should not be created under HF_HOME"
    fi
}

test_pack_validation_run_determinism_repeats_writes_summary() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    PACK_SUITE="subset"
    PACK_DETERMINISM="strict"
    PACK_REPEATS="2"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    pack_setup_output_dirs

    local model_id="org/model"
    local model_name
    model_name="$(sanitize_model_name "${model_id}")"
    mkdir -p "${OUTPUT_DIR}/${model_name}"
    mkdir -p "${TEST_TMPDIR}/baseline"
    echo "${TEST_TMPDIR}/baseline" > "${OUTPUT_DIR}/${model_name}/.baseline_path"

    PACK_MODEL_LIST=("${model_id}")

    resolve_edit_params() {
        jq -n '{status:"selected", edit_dir_name:"edit_for_repeats"}'
    }
    mkdir -p "${OUTPUT_DIR}/${model_name}/models/edit_for_repeats"

    run_invarlock_evaluate() {
        local output_dir="$3"
        local run_name="$4"
        local run_dir="${output_dir}/${run_name}"
        mkdir -p "${run_dir}"
        local count_file="${TEST_TMPDIR}/repeat.count"
        local count=0
        if [[ -f "${count_file}" ]]; then
            count="$(cat "${count_file}")"
        fi
        count=$((count + 1))
        echo "${count}" > "${count_file}"
        cat > "${run_dir}/evaluation.report.json" << EOF
{"verdict": {"primary_metric_ratio": ${count}.01}}
EOF
    }

    pack_run_determinism_repeats

    local path="${OUTPUT_DIR}/analysis/determinism_repeats.json"
    assert_file_exists "${path}" "determinism repeats file written"
    assert_match "\"completed\": 2" "$(cat "${path}")" "repeat count recorded"
}

test_pack_validation_generate_verdict_writes_reports() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    PACK_GPU_NAME="Mock GPU"
    PACK_GPU_MEM_GB="80"
    PACK_GPU_COUNT="2"
    PACK_SUITE="subset"
    PACK_NET="1"
    PACK_DETERMINISM="throughput"
    PACK_REPEATS="0"

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    pack_setup_output_dirs
    mkdir -p "${OUTPUT_DIR}/analysis"
    cat > "${OUTPUT_DIR}/analysis/correlation_analysis.json" <<'EOF'
{
  "summary": {
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1_score": 1.0,
    "error_detection_rate": 1.0,
    "confidence_score": 95,
    "confidence_level": "HIGH",
    "triage_counts": {"PASS": 1, "REVIEW": 0, "FAIL": 0},
    "degraded_edits": 0,
    "degraded_runs": [],
    "total_tests": 1,
    "models_tested": 1
  },
  "models": {}
}
EOF

    generate_verdict

    assert_file_exists "${OUTPUT_DIR}/reports/final_verdict.txt" "final verdict text written"
    assert_file_exists "${OUTPUT_DIR}/reports/final_verdict.json" "final verdict json written"
    assert_match "VERDICT" "$(cat "${OUTPUT_DIR}/reports/final_verdict.txt")" "verdict content emitted"
}

test_pack_validation_generate_verdict_passes_state_manifest_when_present() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    mkdir -p "${OUTPUT_DIR}/state" "${OUTPUT_DIR}/reports"
    printf '{"scenarios":[]}\n' > "${OUTPUT_DIR}/state/scenarios.json"

    python3() {
        printf '%s\n' "$*" > "${TEST_TMPDIR}/verdict.args"
        mkdir -p "${OUTPUT_DIR}/reports"
        printf 'PASS\n' > "${OUTPUT_DIR}/reports/final_verdict.txt"
        printf '{"verdict":"PASS"}\n' > "${OUTPUT_DIR}/reports/final_verdict.json"
    }

    generate_verdict

    assert_match "--manifest ${OUTPUT_DIR}/state/scenarios.json" "$(cat "${TEST_TMPDIR}/verdict.args")" "state manifest passed to verdict generator"
    assert_file_exists "${OUTPUT_DIR}/reports/final_verdict.json" "manifest verdict json written"
}

test_pack_validation_pack_run_suite_runs_dependency_check_before_preflight_when_net_enabled() {
    mock_reset

    local calls_file="${TEST_TMPDIR}/calls.txt"

    (
        OUTPUT_DIR="${TEST_TMPDIR}/out"
        PACK_NET="1"
        PACK_SUITE="subset"
        source ./scripts/evidence_packs/lib/validation/validation_suite.sh

        # Stub out heavy setup (we only care about call ordering inside pack_run_suite).
        pack_apply_network_mode() { :; }
        pack_source_libs() { :; }
        pack_setup_output_dirs() { :; }
        pack_prepare_scenarios_manifest() { :; }
        pack_setup_hf_cache_dirs() { :; }
        pack_preflight_datasets() { :; }
        pack_model_list_array() { PACK_MODEL_LIST=("mistralai/Mistral-7B-v0.1"); }
        pack_prepare_tuned_edit_params() { :; }
        pack_validate_tuned_edit_params() { :; }
        pack_prepare_calibration_presets() { :; }
        pack_validate_guard_calibration() { :; }
        pack_validate_runtime_provenance() { calls="${calls}runtime,"; }

        calls=""
        check_dependencies() { calls="${calls}check,"; }
        pack_preflight_models() { calls="${calls}preflight,"; }
        main_dynamic() { calls="${calls}main,"; }
        pack_require_bash4() { return 0; }

        pack_run_suite
        printf '%s' "${calls}" > "${calls_file}"
    )

    assert_eq "check,runtime,preflight,main," "$(cat "${calls_file}")" "runtime provenance check precedes net preflight"
}

test_pack_validation_source_libs_nested_layout_succeeds_and_exports_loaded_flags() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    pack_source_libs

    assert_match "/scripts/evidence_packs/lib$" "${LIB_DIR}" "LIB_DIR points at scripts/evidence_packs/lib"
    assert_eq "1" "${TASK_SERIALIZATION_LOADED:-}" "task_serialization loaded"
    assert_eq "1" "${QUEUE_MANAGER_LOADED:-}" "queue_manager loaded"
    assert_eq "1" "${SCHEDULER_LOADED:-}" "scheduler loaded"
    assert_eq "1" "${TASK_FUNCTIONS_LOADED:-}" "task_functions loaded"
    assert_eq "1" "${GPU_WORKER_LOADED:-}" "gpu_worker loaded"
    assert_eq "1" "${FAULT_TOLERANCE_LOADED:-}" "fault_tolerance loaded"
}

test_pack_validation_source_libs_falls_back_to_lib_dir_when_missing() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local sandbox
    sandbox="$(_make_validation_suite_sandbox)"

    rm -rf "${sandbox}/lib"

    local rc=0
    (
        _pack_script_dir() { echo "${sandbox}"; }
        pack_source_libs
    ) || rc=$?
    assert_ne "0" "${rc}" "expected pack_source_libs failure when scripts/evidence_packs/lib is missing"
}

test_pack_validation_source_libs_packaged_v2_layout_succeeds() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local sandbox
    sandbox="$(_make_validation_suite_sandbox)"

    (
        _pack_script_dir() { echo "${sandbox}"; }
        pack_source_libs
        assert_eq "${sandbox}/lib" "${LIB_DIR}" "packaged v2 layout loads lib dir"
        assert_eq "1" "${QUEUE_MANAGER_LOADED:-}" "queue_manager loaded from packaged v2 lib dir"
    )
}

test_pack_validation_source_libs_packaged_v2_layout_errors_when_queue_manager_missing() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local sandbox
    sandbox="$(_make_validation_suite_sandbox)"

    rm -f "${sandbox}/lib/queue/queue_manager.sh"

    local rc=0
    (
        _pack_script_dir() { echo "${sandbox}"; }
        pack_source_libs
    ) || rc=$?
    assert_ne "0" "${rc}" "expected failure when queue_manager is missing in packaged v2 lib dir"
}

test_pack_validation_source_libs_legacy_flat_script_dir_layout_succeeds() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local root="${TEST_TMPDIR}/legacy_flat"
    mkdir -p "${root}"
    local file
    for file in task_serialization.sh queue_manager.sh scheduler.sh task_functions.sh gpu_worker.sh; do
        printf '%s\n' "#!/usr/bin/env bash" > "${root}/${file}"
    done

    _pack_script_dir() { echo "${root}"; }
    pack_source_libs
    assert_eq "${root}" "${LIB_DIR}" "legacy direct lib layout selected"
}

test_pack_validation_source_libs_parent_nested_lib_layout_succeeds() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local root="${TEST_TMPDIR}/parent_nested"
    mkdir -p "${root}/child" "${root}/lib/tasks" "${root}/lib/queue" "${root}/lib/core"
    printf '%s\n' "#!/usr/bin/env bash" > "${root}/lib/tasks/task_serialization.sh"
    printf '%s\n' "#!/usr/bin/env bash" > "${root}/lib/tasks/task_functions.sh"
    printf '%s\n' "#!/usr/bin/env bash" > "${root}/lib/queue/queue_manager.sh"
    printf '%s\n' "#!/usr/bin/env bash" > "${root}/lib/queue/scheduler.sh"
    printf '%s\n' "#!/usr/bin/env bash" > "${root}/lib/queue/gpu_worker.sh"
    printf '%s\n' "#!/usr/bin/env bash" > "${root}/lib/core/fault_tolerance.sh"

    _pack_script_dir() { echo "${root}/child"; }
    pack_source_libs
    assert_eq "${root}/lib" "${LIB_DIR}" "parent nested lib layout selected"
}
