#!/usr/bin/env bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/validation_suite_test_helpers.sh"

test_pack_validate_guard_calibration_errors_when_disabled_without_preset() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    DRIFT_CALIBRATION_RUNS="0"
    unset PACK_CALIBRATION_PRESET_DIR PACK_CALIBRATION_PRESET_FILE

    local rc=0
    ( pack_validate_guard_calibration ) || rc=$?
    assert_ne "0" "${rc}" "DRIFT_CALIBRATION_RUNS=0 without preset triggers error_exit"
    assert_match "Guard calibration disabled" "$(cat "${OUTPUT_DIR}/logs/main.log")" "error logged"
}

test_pack_validation_estimate_planned_model_storage_falls_back_when_mapfile_disabled() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    EDIT_TYPES_CLEAN=("quant_rtn:clean:ffn")
    EDIT_TYPES_STRESS=()
    RUN_ERROR_INJECTION="false"

    pack_model_list() { printf '%s\n' "org/model"; }
    estimate_model_weights_gb() { echo "10"; }

    local had_mapfile="0"
    if command -v mapfile >/dev/null 2>&1; then
        had_mapfile="1"
        enable -n mapfile 2>/dev/null || true
    fi

    local total
    total="$(estimate_planned_model_storage_gb)"
    if [[ "${had_mapfile}" == "1" ]]; then
        enable mapfile 2>/dev/null || true
    fi
    assert_eq "20" "${total}" "planned storage sums weights and edits without mapfile"
}

test_pack_prepare_scenarios_manifest_writes_state_manifest() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    pack_prepare_scenarios_manifest

    assert_file_exists "${OUTPUT_DIR}/state/scenarios.json" "scenarios manifest written into run state"
    assert_eq "evidence_pack_scenarios_v1" "$(jq -r '.schema' "${OUTPUT_DIR}/state/scenarios.json")" "schema set"
    assert_eq "1" "$(jq -r '.schema_version' "${OUTPUT_DIR}/state/scenarios.json")" "schema version set"
    assert_eq "subset" "$(jq -r '._meta.applied_suite' "${OUTPUT_DIR}/state/scenarios.json")" "suite recorded"
    local count
    count="$(jq '.scenarios | length' "${OUTPUT_DIR}/state/scenarios.json")"
    assert_ne "0" "${count}" "scenarios list is non-empty"
    local deployable_count
    deployable_count="$(jq '[.scenarios[] | select(.artifact_class=="deployable_optimized_subject")] | length' "${OUTPUT_DIR}/state/scenarios.json")"
    assert_eq "0" "${deployable_count}" "deployable scenarios are excluded by default"
}

test_pack_prepare_scenarios_manifest_filters_by_suite_tags() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local PACK_SUITE="showcase"

    local manifest="${TEST_TMPDIR}/scenarios.json"
    cat > "${manifest}" <<'EOF'
{
  "_meta": {},
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {"id": "a", "category": "clean", "strictness": "must_pass", "generation": {"kind": "edit", "edit_spec": "x", "version": "clean"}, "suites": ["subset"]},
    {"id": "b", "category": "clean", "strictness": "must_pass", "generation": {"kind": "edit", "edit_spec": "y", "version": "clean"}, "suites": ["showcase"]},
    {"id": "c", "category": "clean", "strictness": "must_pass", "generation": {"kind": "edit", "edit_spec": "z", "version": "clean"}}
  ]
}
EOF
    local PACK_SCENARIOS_MANIFEST_FILE="${manifest}"

    pack_prepare_scenarios_manifest

    local ids
    ids="$(jq -r '.scenarios[].id' "${OUTPUT_DIR}/state/scenarios.json" | sort | paste -sd ',' -)"
    assert_eq "b,c" "${ids}" "filters by suite, but keeps untagged scenarios"
}

test_pack_prepare_scenarios_manifest_filters_by_scenario_ids() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local PACK_SUITE="showcase"

    local manifest="${TEST_TMPDIR}/scenarios.json"
    cat > "${manifest}" <<'EOF'
{
  "_meta": {},
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {"id": "a", "category": "clean", "strictness": "must_pass", "generation": {"kind": "edit", "edit_spec": "x", "version": "clean"}, "suites": ["subset"]},
    {"id": "b", "category": "clean", "strictness": "must_pass", "generation": {"kind": "edit", "edit_spec": "y", "version": "clean"}, "suites": ["showcase"]},
    {"id": "c", "category": "clean", "strictness": "must_pass", "generation": {"kind": "edit", "edit_spec": "z", "version": "clean"}}
  ]
}
EOF
    local PACK_SCENARIOS_MANIFEST_FILE="${manifest}"
    local PACK_SCENARIO_IDS="b"

    pack_prepare_scenarios_manifest

    local ids
    ids="$(jq -r '.scenarios[].id' "${OUTPUT_DIR}/state/scenarios.json" | sort | paste -sd ',' -)"
    assert_eq "b" "${ids}" "filters by scenario id after suite filtering"
}

test_pack_prepare_scenarios_manifest_filters_deployable_backends() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local manifest="${TEST_TMPDIR}/scenarios.json"
    cat > "${manifest}" <<'EOF'
{
  "_meta": {},
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {"id": "quant_4bit_clean", "category": "clean", "artifact_class": "validation_subject_checkpoint", "strictness": "must_pass", "generation": {"kind": "edit", "edit_spec": "quant_rtn:clean:ffn", "version": "clean"}, "suites": ["subset"]},
    {"id": "deploy_gptq", "category": "deployable_clean", "artifact_class": "deployable_optimized_subject", "strictness": "must_pass", "generation": {"kind": "deployable_edit", "backend": "gptq", "edit_spec": "gptq_int4:clean:ffn", "version": "clean"}, "suites": ["deployable"]},
    {"id": "deploy_bnb", "category": "deployable_clean", "artifact_class": "deployable_optimized_subject", "strictness": "must_pass", "generation": {"kind": "deployable_edit", "backend": "bitsandbytes", "edit_spec": "bnb_8bit:clean:ffn", "version": "clean"}, "suites": ["deployable"]}
  ]
}
EOF
    local PACK_SCENARIOS_MANIFEST_FILE="${manifest}"
    local PACK_INCLUDE_DEPLOYABLE_EDITS="1"
    local PACK_DEPLOY_BACKENDS="gptq"

    pack_prepare_scenarios_manifest

    local ids
    ids="$(jq -r '.scenarios[].id' "${OUTPUT_DIR}/state/scenarios.json" | sort | paste -sd ',' -)"
    assert_eq "deploy_gptq,quant_4bit_clean" "${ids}" "deployable scenarios honor backend filter"
}

test_pack_prepare_scenarios_manifest_rejects_non_runnable_deployable() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    local manifest="${TEST_TMPDIR}/scenarios-non-runnable.json"
    cat > "${manifest}" <<'EOF'
{
  "_meta": {},
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {"id": "deploy_gptq", "category": "deployable_clean", "artifact_class": "deployable_optimized_subject", "strictness": "must_pass", "runnable": false, "generation": {"kind": "deployable_edit", "backend": "gptq", "edit_spec": "gptq_int4:clean:ffn", "version": "clean"}, "suites": ["deployable"]}
  ]
}
EOF
    local PACK_SCENARIOS_MANIFEST_FILE="${manifest}"
    local PACK_INCLUDE_DEPLOYABLE_EDITS="1"
    local PACK_DEPLOY_BACKENDS="gptq"

    local rc=0
    ( pack_prepare_scenarios_manifest ) || rc=$?
    assert_ne "0" "${rc}" "non-runnable deployable scenario fails closed"
    assert_match "contract placeholders and are not runnable yet: deploy_gptq" "$(cat "${OUTPUT_DIR}/logs/main.log")" "non-runnable reason is logged"
}

test_pack_prepare_scenarios_manifest_rejects_non_runnable_deployable_without_jq() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    _pack_validation_has_jq() { return 1; }
    pack_setup_output_dirs

    local manifest="${TEST_TMPDIR}/scenarios-non-runnable-no-jq.json"
    cat > "${manifest}" <<'EOF'
{
  "_meta": {},
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {"id": "deploy_gptq", "category": "deployable_clean", "artifact_class": "deployable_optimized_subject", "strictness": "must_pass", "runnable": false, "generation": {"kind": "deployable_edit", "backend": "gptq", "edit_spec": "gptq_int4:clean:ffn", "version": "clean"}, "suites": ["deployable"]}
  ]
}
EOF
    local PACK_SCENARIOS_MANIFEST_FILE="${manifest}"
    local PACK_INCLUDE_DEPLOYABLE_EDITS="1"

    local rc=0
    ( pack_prepare_scenarios_manifest ) || rc=$?
    assert_ne "0" "${rc}" "non-runnable deployable scenario fails closed without jq"
    assert_match "contract placeholders and are not runnable yet: deploy_gptq" "$(cat "${OUTPUT_DIR}/logs/main.log")" "non-runnable reason is logged without jq"
}

test_pack_prepare_scenarios_manifest_resume_errors_on_contract_drift() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    cat > "${OUTPUT_DIR}/state/scenarios.json" <<'EOF'
{
  "_meta": {"applied_suite": "subset"},
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {"id": "a", "category": "clean", "strictness": "must_pass", "generation": {"kind": "edit", "edit_spec": "x", "version": "clean"}}
  ]
}
EOF

    local manifest="${TEST_TMPDIR}/resume-scenarios.json"
    cat > "${manifest}" <<'EOF'
{
  "_meta": {},
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {"id": "a", "category": "clean", "strictness": "must_pass", "generation": {"kind": "edit", "edit_spec": "x", "version": "clean"}},
    {"id": "b", "category": "error_injection", "strictness": "informational", "generation": {"kind": "error", "error_type": "b"}, "requirements": {"primary_guard_required": true}}
  ]
}
EOF
    local PACK_SCENARIOS_MANIFEST_FILE="${manifest}"
    local RESUME_FLAG="true"

    local rc=0
    ( pack_prepare_scenarios_manifest ) || rc=$?
    assert_ne "0" "${rc}" "resume fails closed when scenario contract drift is detected"
    assert_match "Resume run scenario manifest differs from the current contract" "$(cat "${OUTPUT_DIR}/logs/main.log")" "contract drift is logged"
}

test_pack_validation_resolve_active_scenarios_manifest_prefers_state_then_repo_source() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    mkdir -p "${OUTPUT_DIR}/state"
    local state_manifest="${OUTPUT_DIR}/state/scenarios.json"
    printf '%s\n' '{"scenarios":[]}' > "${state_manifest}"
    assert_eq "${state_manifest}" "$(pack_resolve_active_scenarios_manifest)" "state scenarios manifest takes precedence"

    rm -f "${state_manifest}"
    local repo_manifest="${TEST_TMPDIR}/repo-scenarios.json"
    printf '%s\n' '{"scenarios":[]}' > "${repo_manifest}"
    PACK_SCENARIOS_MANIFEST_FILE="${repo_manifest}"
    export PACK_SCENARIOS_MANIFEST_FILE
    assert_eq "${repo_manifest}" "$(pack_resolve_active_scenarios_manifest)" "repo manifest fallback is used when state manifest is absent"

    rm -f "${repo_manifest}"
    run pack_resolve_active_scenarios_manifest
    assert_rc "1" "${RUN_RC}" "manifest resolution fails when neither state nor repo manifest exists"
}

test_pack_validation_estimate_planned_storage_sanitizes_invalid_edit_counts() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    HF_HUB_CACHE=""
    PACK_BASELINE_STORAGE_MODE="snapshot_symlink"
    RUN_ERROR_INJECTION="false"
    pack_model_list() { printf '%s\n' "org/model"; }
    estimate_model_weights_gb() { echo "10"; }
    pack_count_edit_scenarios() { echo "bad|oops|fixture"; }

    local total
    total="$(estimate_planned_model_storage_gb)"
    assert_eq "10" "${total}" "invalid edit counts are sanitized to zero when estimating storage"
}

test_pack_validation_check_dependencies_reports_missing_pinned_requirements() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    PACK_NET="1"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    log_section() { :; }
    log() { printf '%s\n' "$*" >> "${TEST_TMPDIR}/dep.log"; }
    error_exit() { exit 23; }

    run pack_install_pinned_requirement "missing"
    assert_rc "1" "${RUN_RC}" "missing pinned requirement file fails directly"
    assert_match "pinned requirement file missing" "$(cat "${TEST_TMPDIR}/dep.log")" "missing requirement file is reported"

    timeout() { shift; "$@"; }
    command() {
        if [[ "${1:-}" == "-v" && "${2:-}" == "invarlock" ]]; then
            return 0
        fi
        builtin command "$@"
    }
    pack_evidence_pack_requirement_path() {
        if [[ "${1:-}" == "flash-attn" ]]; then
            printf '%s\n' "${TEST_TMPDIR}/missing-flash-attn.txt"
            return 0
        fi
        printf '%s\n' "${TEST_TMPDIR}/requirements/%s.txt" "${1}"
    }
    python3() {
        if [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "--version" ]]; then
            return 0
        fi
        if [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "install" ]]; then
            return 0
        fi
        if [[ "${1:-}" == "-c" ]]; then
            local code="${2:-}"
            case "${code}" in
                *"import torch; assert torch.cuda.is_available"*) return 0 ;;
                *"import transformers"*) return 0 ;;
                *"import invarlock"*) return 0 ;;
                *"import huggingface_hub"*) return 0 ;;
                *"import accelerate"*) return 0 ;;
                *"import yaml"*) return 0 ;;
                *"import google.protobuf"*) return 0 ;;
                *"import sentencepiece"*) return 0 ;;
                *"import flash_attn; print('Flash Attention OK')"*) return 1 ;;
                *"import sysconfig; exit(0 if sysconfig.get_config_var('INCLUDEPY')"*) return 0 ;;
                *"print(sysconfig.get_config_var('INCLUDEPY'))"*) echo "${TEST_TMPDIR}/include"; return 0 ;;
                *"import flash_attn"*) return 1 ;;
            esac
        fi
        return 0
    }

    mkdir -p "${TEST_TMPDIR}/include"
    : > "${TEST_TMPDIR}/include/Python.h"
    check_dependencies
    assert_eq "false" "${FLASH_ATTENTION_AVAILABLE}" "flash-attn remains disabled when pinned requirement file is missing"
    assert_match "pinned flash-attn requirement file missing" "$(cat "${TEST_TMPDIR}/dep.log")" "missing flash-attn requirement is logged"
}

test_pack_validation_main_dynamic_blocked_state_touches_shutdown_without_signal_handler() {
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
    compile_results() { :; }
    run_analysis() { :; }
    generate_verdict() { :; }
    list_run_gpu_ids() { printf '0\n'; }
    is_queue_empty() { return 1; }
    sleep() { :; }
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

    init_queue
    printf '{"status":"failed"}\n' > "${QUEUE_DIR}/failed/dep.task"

    run main_dynamic
    assert_rc "1" "${RUN_RC}" "blocked queue exits nonzero without signal_shutdown"
    assert_file_exists "${OUTPUT_DIR}/workers/SHUTDOWN" "blocked queue touches shutdown marker when signal_shutdown is unavailable"
}

test_pack_validation_main_dynamic_sanitizes_invalid_edit_scenario_counts() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out_invalid_counts"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs
    printf '%s\n' '{"scenarios":[]}' > "${OUTPUT_DIR}/state/scenarios.json"

    check_dependencies() { :; }
    configure_gpu_pool() { NUM_GPUS=1; GPU_ID_LIST="0"; export NUM_GPUS GPU_ID_LIST; }
    disk_preflight() { :; }
    setup_pack_environment() { :; }
    handle_disk_pressure() { return 0; }
    pack_count_edit_scenarios() { echo "bogus|oops|fixture"; }
    _pack_validation_state() {
        if [[ "${1:-}" == "count-generation-kind" ]]; then
            echo "not-a-number"
            return 0
        fi
        return 0
    }

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
    list_run_gpu_ids() { printf '0\n'; }
    is_queue_empty() { return 0; }
    get_free_disk_gb() { echo "999"; }
    RESUME_FLAG="false"
    CLEAN_EDIT_RUNS="1"
    STRESS_EDIT_RUNS="1"
    RUN_ERROR_INJECTION="false"

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
    assert_match "Edit scenarios: 0 clean \\+ 0 stress = 0 per model \\(fixture\\)" "$(cat "${OUTPUT_DIR}/logs/main.log")" "invalid edit scenario counts are sanitized to zero"
}
