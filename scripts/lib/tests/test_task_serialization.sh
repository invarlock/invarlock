#!/usr/bin/env bash

test_calculate_required_gpus_basic() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    GPU_MEMORY_PER_DEVICE=180 NUM_GPUS=8
    assert_eq "1" "$(calculate_required_gpus "abc")" "invalid model size defaults to 1"
    assert_eq "1" "$(calculate_required_gpus "180")" "fits on one GPU"
    assert_eq "2" "$(calculate_required_gpus "181")" "rounds up to 2 GPUs"

    GPU_MEMORY_PER_DEVICE=100 NUM_GPUS=3
    assert_eq "3" "$(calculate_required_gpus "250")" "caps at NUM_GPUS"
}

test_task_serialization_requires_jq_when_missing() {
    mock_reset

    (
        dirname() {
            local path="${1:-}"
            [[ -n "${path}" ]] || return 1
            while [[ "${path}" == */ ]]; do
                path="${path%/}"
            done
            if [[ "${path}" != */* ]]; then
                echo "."
            else
                echo "${path%/*}"
            fi
        }
        export PATH="${TEST_ROOT}/scripts/lib/tests/mocks/bin"
        # shellcheck source=../task_serialization.sh
        source "${TEST_ROOT}/scripts/lib/task_serialization.sh"
        assert_eq "0" "${TASK_SERIALIZATION_HAS_JQ}" "detects jq missing"

        local rc=0
        create_task "${TEST_TMPDIR}/queue" "SETUP_BASELINE" "org/model" "model" "14" "[]" "{}" "50" || rc=$?
        assert_rc "1" "${rc}" "jq-dependent call fails cleanly"
    )
}

test_task_serialization_require_jq_short_circuit_sites() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    _task_serialization_require_jq() { return 1; }

    run get_task_field "${TEST_TMPDIR}/nope.task" "task_id"
    assert_rc "1" "${RUN_RC}" "get_task_field returns non-zero when jq missing"

    run get_task_fields "${TEST_TMPDIR}/nope.task" "task_id" "status"
    assert_rc "1" "${RUN_RC}" "get_task_fields returns non-zero when jq missing"

    run get_task_dependencies "${TEST_TMPDIR}/nope.task"
    assert_rc "1" "${RUN_RC}" "get_task_dependencies returns non-zero when jq missing"

    run get_task_params "${TEST_TMPDIR}/nope.task"
    assert_rc "1" "${RUN_RC}" "get_task_params returns non-zero when jq missing"

    run get_task_param "${TEST_TMPDIR}/nope.task" "seq_len"
    assert_rc "1" "${RUN_RC}" "get_task_param returns non-zero when jq missing"

    run update_task_field "${TEST_TMPDIR}/nope.task" "status" "pending"
    assert_rc "1" "${RUN_RC}" "update_task_field returns non-zero when jq missing"

    run mark_task_started "${TEST_TMPDIR}/nope.task" "0"
    assert_rc "1" "${RUN_RC}" "mark_task_started returns non-zero when jq missing"

    run mark_task_started_multi "${TEST_TMPDIR}/nope.task" "0,1"
    assert_rc "1" "${RUN_RC}" "mark_task_started_multi returns non-zero when jq missing"

    run mark_task_completed "${TEST_TMPDIR}/nope.task"
    assert_rc "1" "${RUN_RC}" "mark_task_completed returns non-zero when jq missing"

    run mark_task_failed "${TEST_TMPDIR}/nope.task" "boom"
    assert_rc "1" "${RUN_RC}" "mark_task_failed returns non-zero when jq missing"

    run increment_task_retries "${TEST_TMPDIR}/nope.task"
    assert_rc "1" "${RUN_RC}" "increment_task_retries returns non-zero when jq missing"

    run update_task_params "${TEST_TMPDIR}/nope.task" '{}'
    assert_rc "1" "${RUN_RC}" "update_task_params returns non-zero when jq missing"

    run validate_task "${TEST_TMPDIR}/nope.task"
    assert_rc "1" "${RUN_RC}" "validate_task returns non-zero when jq missing"

    run print_task_summary "${TEST_TMPDIR}/nope.task"
    assert_rc "1" "${RUN_RC}" "print_task_summary returns non-zero when jq missing"
}

test_calculate_required_gpus_sanitizes_per_device_and_num_gpus() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    GPU_MEMORY_PER_DEVICE="bad" NUM_GPUS=8
    assert_eq "2" "$(calculate_required_gpus "181")" "invalid per-device falls back to default"

    GPU_MEMORY_PER_DEVICE=0 NUM_GPUS=8
    assert_eq "2" "$(calculate_required_gpus "181")" "zero per-device falls back to default"

    GPU_MEMORY_PER_DEVICE=100 NUM_GPUS="nope"
    assert_eq "3" "$(calculate_required_gpus "250")" "invalid NUM_GPUS falls back to default"

    GPU_MEMORY_PER_DEVICE=100 NUM_GPUS=2
    assert_eq "2" "$(calculate_required_gpus "250")" "caps required GPUs at NUM_GPUS"

    GPU_MEMORY_PER_DEVICE=100 NUM_GPUS=0
    assert_eq "1" "$(calculate_required_gpus "250")" "always returns at least one GPU"
}

test_create_task_success_and_invalid_params() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }

    local queue_dir="${TEST_TMPDIR}/queue"
    mkdir -p "${queue_dir}"

    TASK_SEQUENCE=1
    local task_id
    task_id="$(create_task "${queue_dir}" "SETUP_BASELINE" "org/model" "model" "14" "dep1,dep2" '{"k":"v"}' "77")"
    assert_match '^model_SETUP_BASELINE_001_[0-9a-f]{4}$' "${task_id}" "task_id format"

    local task_file="${queue_dir}/pending/${task_id}.task"
    assert_file_exists "${task_file}" "task file created"
    assert_eq "SETUP_BASELINE" "$(jq -r '.task_type' "${task_file}")" "task_type stored"
    assert_eq "2025-01-01T00:00:00Z" "$(jq -r '.created_at' "${task_file}")" "created_at uses _now_iso"
    assert_eq "77" "$(jq -r '.priority' "${task_file}")" "priority stored"
    assert_eq "2" "$(jq -r '.dependencies | length' "${task_file}")" "dependencies parsed"
    assert_eq "v" "$(jq -r '.params.k' "${task_file}")" "params stored"

    # Invalid params JSON fails
    run create_task "${queue_dir}" "SETUP_BASELINE" "org/model" "model" "14" "[]" '{not-json' "50"
    assert_rc "1" "${RUN_RC}" "invalid params JSON should fail"
    assert_match 'Invalid params JSON' "${RUN_ERR}" "error message"
}

test_create_task_rejects_invalid_model_size_gb() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    local queue_dir="${TEST_TMPDIR}/queue"
    mkdir -p "${queue_dir}"

    run create_task "${queue_dir}" "SETUP_BASELINE" "org/model" "model" "nope" "[]" '{}' "50"
    assert_rc "1" "${RUN_RC}" "invalid model_size_gb should fail"
    assert_match 'model_size_gb' "${RUN_ERR}" "error message"
}

test_create_task_rejects_invalid_priority() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    local queue_dir="${TEST_TMPDIR}/queue"
    mkdir -p "${queue_dir}"

    run create_task "${queue_dir}" "SETUP_BASELINE" "org/model" "model" "14" "[]" '{}' "high"
    assert_rc "1" "${RUN_RC}" "invalid priority should fail"
    assert_match 'priority' "${RUN_ERR}" "error message"
}

test_create_task_sanitizes_non_numeric_task_sequence() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }

    local queue_dir="${TEST_TMPDIR}/queue"
    mkdir -p "${queue_dir}"

    TASK_SEQUENCE="nope"
    local task_id
    task_id="$(create_task "${queue_dir}" "SETUP_BASELINE" "org/model" "model" "14" "[]" '{}' "50")"
    assert_match '^model_SETUP_BASELINE_001_[0-9a-f]{4}$' "${task_id}" "sequence defaults to 001 when invalid"
}

test_create_task_rejects_invalid_dependencies_json_array() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    local queue_dir="${TEST_TMPDIR}/queue"
    mkdir -p "${queue_dir}"

    run create_task "${queue_dir}" "SETUP_BASELINE" "org/model" "model" "14" '[bad' '{}' "50"
    assert_rc "1" "${RUN_RC}" "invalid dependencies JSON should fail"
    assert_match 'dependencies' "${RUN_ERR}" "error message"
}

test_create_task_rejects_non_object_params_json() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    local queue_dir="${TEST_TMPDIR}/queue"
    mkdir -p "${queue_dir}"

    run create_task "${queue_dir}" "SETUP_BASELINE" "org/model" "model" "14" "[]" '[]' "50"
    assert_rc "1" "${RUN_RC}" "non-object params JSON should fail"
    assert_match 'expected object' "${RUN_ERR}" "error message"
}

test_create_task_returns_error_when_dependency_parse_fails() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    jq() {
        if [[ "${1:-}" == "-s" ]]; then
            return 2
        fi
        command jq "$@"
    }

    local queue_dir="${TEST_TMPDIR}/queue"
    mkdir -p "${queue_dir}"

    run create_task "${queue_dir}" "SETUP_BASELINE" "org/model" "model" "14" "dep1,dep2" '{}' "50"
    assert_rc "1" "${RUN_RC}" "dependency parse failure returns non-zero"
    assert_match 'Failed to parse dependencies' "${RUN_ERR}" "error message"
    unset -f jq
}

test_create_task_returns_error_when_task_json_build_fails() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    jq() {
        if [[ "${1:-}" == "-n" ]]; then
            return 3
        fi
        command jq "$@"
    }

    local queue_dir="${TEST_TMPDIR}/queue"
    mkdir -p "${queue_dir}"

    run create_task "${queue_dir}" "SETUP_BASELINE" "org/model" "model" "14" "[]" '{}' "50"
    assert_rc "1" "${RUN_RC}" "jq build failure returns non-zero"
    assert_match 'Failed to build task JSON' "${RUN_ERR}" "error message"
    unset -f jq
}

test_create_task_returns_error_when_pending_dir_mkdir_fails() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    local queue_dir="${TEST_TMPDIR}/queue"
    echo "not a dir" > "${queue_dir}"

    run create_task "${queue_dir}" "SETUP_BASELINE" "org/model" "model" "14" "[]" '{}' "50"
    assert_rc "1" "${RUN_RC}" "mkdir failure returns non-zero"
    assert_match 'Failed to create pending queue dir' "${RUN_ERR}" "error message"
}

test_create_task_returns_error_when_task_file_write_fails() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    local queue_dir="${TEST_TMPDIR}/queue"
    mkdir -p "${queue_dir}/pending"
    chmod 500 "${queue_dir}/pending"

    run create_task "${queue_dir}" "SETUP_BASELINE" "org/model" "model" "14" "[]" '{}' "50"
    assert_rc "1" "${RUN_RC}" "write failure returns non-zero"
    assert_match 'Failed to write task file' "${RUN_ERR}" "error message"

    chmod 700 "${queue_dir}/pending"
}

test_create_task_returns_error_when_task_file_finalize_fails() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    mv() { return 4; }

    local queue_dir="${TEST_TMPDIR}/queue"
    mkdir -p "${queue_dir}"

    run create_task "${queue_dir}" "SETUP_BASELINE" "org/model" "model" "14" "[]" '{}' "50"
    assert_rc "1" "${RUN_RC}" "mv failure returns non-zero"
    assert_match 'Failed to finalize task file' "${RUN_ERR}" "error message"
    assert_eq "" "$(ls -A "${queue_dir}/pending" 2>/dev/null || true)" "tmp file cleaned up"

    unset -f mv
}

test_create_task_dependency_and_params_default_branches() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }

    local queue_dir="${TEST_TMPDIR}/queue"
    mkdir -p "${queue_dir}"

    local task_id task_file

    task_id="$(create_task "${queue_dir}" "SETUP_BASELINE" "org/model" "model" "14" "[]" "" )"
    task_file="${queue_dir}/pending/${task_id}.task"
    assert_file_exists "${task_file}" "task created with deps array + empty params"
    assert_eq "0" "$(jq -r '.dependencies | length' "${task_file}")" "deps array preserved"
    assert_eq "{}" "$(jq -c '.params' "${task_file}")" "empty params default to object"

    task_id="$(create_task "${queue_dir}" "SETUP_BASELINE" "org/model" "model" "14" "none" "null" )"
    task_file="${queue_dir}/pending/${task_id}.task"
    assert_eq "0" "$(jq -r '.dependencies | length' "${task_file}")" "none deps default to empty array"
    assert_eq "{}" "$(jq -c '.params' "${task_file}")" "null params default to object"
}

test_task_status_transitions() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }

    local queue_dir="${TEST_TMPDIR}/queue"
    mkdir -p "${queue_dir}/pending"

    local task_id="t1"
    local task_file="${queue_dir}/pending/${task_id}.task"
    jq -n --arg task_id "${task_id}" --arg task_type "SETUP_BASELINE" --arg model_id "m" --arg model_name "n" --arg status "pending" \
        '{task_id:$task_id, task_type:$task_type, model_id:$model_id, model_name:$model_name, status:$status, retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${task_file}"

    mark_task_started "${task_file}" "3"
    assert_eq "running" "$(jq -r '.status' "${task_file}")" "status running"
    assert_eq "3" "$(jq -r '.gpu_id' "${task_file}")" "gpu_id set"
    assert_eq "2025-01-01T00:00:00Z" "$(jq -r '.started_at' "${task_file}")" "started_at set"

    mark_task_completed "${task_file}"
    assert_eq "completed" "$(jq -r '.status' "${task_file}")" "status completed"
    assert_eq "2025-01-01T00:00:00Z" "$(jq -r '.completed_at' "${task_file}")" "completed_at set"

    mark_task_failed "${task_file}" "boom"
    assert_eq "failed" "$(jq -r '.status' "${task_file}")" "status failed"
    assert_eq "boom" "$(jq -r '.error_msg' "${task_file}")" "error_msg set"
}

test_task_serialization_field_access_and_update_error_paths() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    local missing="${TEST_TMPDIR}/missing.task"
    local rc=0

    rc=0
    get_task_field "${missing}" "task_id" || rc=$?
    assert_rc "1" "${rc}" "get_task_field fails on missing file"

    rc=0
    get_task_fields "${missing}" "task_id" "status" || rc=$?
    assert_rc "1" "${rc}" "get_task_fields fails on missing file"

    rc=0
    update_task_field "${missing}" "status" "pending" || rc=$?
    assert_rc "1" "${rc}" "update_task_field fails on missing file"

    local task_file="${TEST_TMPDIR}/task.json"
    jq -n '{task_id:"x", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"pending", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${task_file}"

    update_task_field "${task_file}" "gpu_id" "7" "true"
    assert_eq "7" "$(jq -r '.gpu_id' "${task_file}")" "json update writes numeric value"

    rc=0
    update_task_field "${task_file}" "gpu_id" "nope" "true" || rc=$?
    assert_rc "1" "${rc}" "invalid json value fails"
    assert_eq "7" "$(jq -r '.gpu_id' "${task_file}")" "invalid update does not clobber file"
}

test_validate_task_required_fields_and_enums() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    local task_file="${TEST_TMPDIR}/task.json"

    # Missing file fails
    run validate_task "${task_file}"
    assert_rc "1" "${RUN_RC}" "missing file invalid"

    # Invalid JSON fails
    echo "{bad" > "${task_file}"
    run validate_task "${task_file}"
    assert_rc "1" "${RUN_RC}" "invalid JSON invalid"

    # Missing required field fails
    jq -n '{task_id:"x", task_type:"SETUP_BASELINE", model_id:"m", status:"pending"}' > "${task_file}"
    run validate_task "${task_file}"
    assert_rc "1" "${RUN_RC}" "missing model_name invalid"

    # Invalid task_type fails
    jq -n '{task_id:"x", task_type:"NOPE", model_id:"m", model_name:"n", status:"pending"}' > "${task_file}"
    run validate_task "${task_file}"
    assert_rc "1" "${RUN_RC}" "invalid task_type invalid"

    # Invalid status fails
    jq -n '{task_id:"x", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"nope"}' > "${task_file}"
    run validate_task "${task_file}"
    assert_rc "1" "${RUN_RC}" "invalid status invalid"

    # Valid succeeds
    jq -n '{task_id:"x", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"pending"}' > "${task_file}"
    run validate_task "${task_file}"
    assert_rc "0" "${RUN_RC}" "valid task"
}

test_get_task_fields_and_simple_accessors_return_expected_values() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    local task="${TEST_TMPDIR}/t.task"
    jq -n '{task_id:"t1", task_type:"EVAL_BASELINE", model_id:"m", model_name:"n", status:"ready", model_size_gb:14, required_gpus:2, dependencies:[], params:{}, priority:50}' \
        > "${task}"

    assert_eq $'t1\tEVAL_BASELINE\tn\t2' "$(get_task_fields "${task}" task_id task_type model_name required_gpus)" "tsv field selection"
    assert_eq "14" "$(get_task_model_size "${task}")" "model_size_gb accessor"
    assert_eq "2" "$(get_task_required_gpus "${task}")" "required_gpus accessor"

    # Missing required_gpus defaults to 1.
    jq 'del(.required_gpus)' "${task}" > "${task}.tmp" && mv "${task}.tmp" "${task}"
    assert_eq "1" "$(get_task_required_gpus "${task}")" "required_gpus default"

    # Null required_gpus defaults to 1.
    jq '.required_gpus = null' "${task}" > "${task}.tmp" && mv "${task}.tmp" "${task}"
    assert_eq "1" "$(get_task_required_gpus "${task}")" "required_gpus null defaults"
}

test_print_task_summary_and_queue_summary_print_expected_headers() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    local queue_dir="${TEST_TMPDIR}/queue"
    mkdir -p "${queue_dir}/ready"

    local task="${queue_dir}/ready/t1.task"
    jq -n '{task_id:"t1", task_type:"EVAL_BASELINE", model_id:"m", model_name:"n", status:"ready", model_size_gb:14, required_gpus:1, dependencies:[], params:{}, priority:50}' \
        > "${task}"

    assert_match 't1 \\| EVAL_BASELINE \\| n \\| ready \\| 14GB' "$(print_task_summary "${task}")" "task summary formatting"
    assert_match '=== READY TASKS ===' "$(print_queue_summary "${queue_dir}" "ready")" "queue summary header"
}

test_task_serialization_multi_gpu_and_jq_failure_branches() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    _now_iso() { echo "2025-01-01T00:00:00Z"; }

    local task_file="${TEST_TMPDIR}/task.json"
    jq -n '{task_id:"x", task_type:"SETUP_BASELINE", model_id:"m", model_name:"n", status:"pending", retries:0, max_retries:3, created_at:"x", started_at:null, completed_at:null, error_msg:null, gpu_id:-1, assigned_gpus:null, dependencies:[], params:{}, priority:50}' \
        > "${task_file}"

    mark_task_started_multi "${task_file}" "0,1"
    assert_eq "running" "$(jq -r '.status' "${task_file}")" "started_multi sets running"
    assert_eq "0" "$(jq -r '.gpu_id' "${task_file}")" "primary gpu_id is first id"
    assert_eq "0,1" "$(jq -r '.assigned_gpus' "${task_file}")" "assigned_gpus recorded"

    increment_task_retries "${task_file}"
    assert_eq "1" "$(jq -r '.retries' "${task_file}")" "retries incremented"

    update_task_params "${task_file}" '{"k":1}'
    assert_eq "1" "$(jq -r '.params.k' "${task_file}")" "params merged"

    local bad_file="${TEST_TMPDIR}/bad.json"
    echo "{bad" > "${bad_file}"

    local rc=0
    rc=0; mark_task_started "${bad_file}" "0" || rc=$?; assert_ne "0" "${rc}" "mark_task_started fails on invalid json"
    rc=0; mark_task_started_multi "${bad_file}" "0,1" || rc=$?; assert_ne "0" "${rc}" "mark_task_started_multi fails on invalid json"
    rc=0; mark_task_completed "${bad_file}" || rc=$?; assert_ne "0" "${rc}" "mark_task_completed fails on invalid json"
    rc=0; mark_task_failed "${bad_file}" "boom" || rc=$?; assert_ne "0" "${rc}" "mark_task_failed fails on invalid json"
    rc=0; increment_task_retries "${bad_file}" || rc=$?; assert_ne "0" "${rc}" "increment_task_retries fails on invalid json"
    rc=0; update_task_params "${bad_file}" '{"x":1}' || rc=$?; assert_ne "0" "${rc}" "update_task_params fails on invalid json"
}

test_get_task_param_supports_non_identifier_keys() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    local task_file="${TEST_TMPDIR}/task.json"
    jq -n '{params:{"retry-after":123}}' > "${task_file}"
    assert_eq "123" "$(get_task_param "${task_file}" "retry-after")" "dash keys are supported"
}

test_task_serialization_task_summary_branches() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    local missing="${TEST_TMPDIR}/missing.task"
    local rc=0
    rc=0; print_task_summary "${missing}" || rc=$?; assert_rc "1" "${rc}" "print_task_summary fails on missing file"

    print_queue_summary "${TEST_TMPDIR}/queue" "pending"
}

test_estimate_model_memory_name_buckets() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    local v
    v="$(estimate_model_memory "Qwen/Qwen1.5-72B" "EVAL_BASELINE")"
    assert_match '^[0-9]+$' "${v}" "returns integer"
    [[ "${v}" -ge 100 ]] || t_fail "expected large model memory >= 100, got ${v}"

    v="$(estimate_model_memory "mistralai/Mixtral-8x7B-v0.1" "EVAL_BASELINE")"
    assert_match '^[0-9]+$' "${v}" "returns integer"
    [[ "${v}" -ge 80 ]] || t_fail "expected MoE model memory >= 80, got ${v}"
}

test_estimate_model_memory_name_bucket_arms_and_local_path_override() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    local model_dir="${TEST_TMPDIR}/model"
    mkdir -p "${model_dir}"
    echo '{}' > "${model_dir}/config.json"
    estimate_model_params() { echo "40"; }
    assert_match '^[0-9]+$' "$(estimate_model_memory "${model_dir}" "SETUP_BASELINE")" "local config.json uses estimate_model_params"

    assert_match '^[0-9]+$' "$(estimate_model_memory "org/Thing-40B" "EVAL_BASELINE")" "40B bucket"
    assert_match '^[0-9]+$' "$(estimate_model_memory "org/Thing-32B" "EVAL_BASELINE")" "30B bucket"
    assert_match '^[0-9]+$' "$(estimate_model_memory "org/Thing-13B" "EVAL_BASELINE")" "13B bucket"
    assert_match '^[0-9]+$' "$(estimate_model_memory "org/Thing" "EVAL_BASELINE")" "default 7B bucket"
}

test_estimate_model_memory_multiplier_case_arms_large_and_small() {
    mock_reset
    # shellcheck source=../task_serialization.sh
    source "${TEST_ROOT}/scripts/lib/task_serialization.sh"

    local large="Qwen/Qwen1.5-72B"
    assert_match '^[0-9]+$' "$(estimate_model_memory "${large}" "SETUP_BASELINE")" "large SETUP_BASELINE"
    assert_match '^[0-9]+$' "$(estimate_model_memory "${large}" "CALIBRATION_RUN")" "large CALIBRATION_RUN"
    assert_match '^[0-9]+$' "$(estimate_model_memory "${large}" "CREATE_EDIT")" "large CREATE_EDIT"
    assert_match '^[0-9]+$' "$(estimate_model_memory "${large}" "EVAL_EDIT")" "large EVAL_EDIT"
    assert_match '^[0-9]+$' "$(estimate_model_memory "${large}" "CERTIFY_EDIT")" "large CERTIFY_EDIT"
    assert_match '^[0-9]+$' "$(estimate_model_memory "${large}" "CREATE_ERROR")" "large CREATE_ERROR"
    assert_match '^[0-9]+$' "$(estimate_model_memory "${large}" "CERTIFY_ERROR")" "large CERTIFY_ERROR"
    assert_match '^[0-9]+$' "$(estimate_model_memory "${large}" "GENERATE_PRESET")" "large GENERATE_PRESET"
    assert_match '^[0-9]+$' "$(estimate_model_memory "${large}" "UNKNOWN")" "large default"

    local small="org/Thing-13B"
    assert_match '^[0-9]+$' "$(estimate_model_memory "${small}" "SETUP_BASELINE")" "small SETUP_BASELINE"
    assert_match '^[0-9]+$' "$(estimate_model_memory "${small}" "CALIBRATION_RUN")" "small CALIBRATION_RUN"
    assert_match '^[0-9]+$' "$(estimate_model_memory "${small}" "CREATE_EDIT")" "small CREATE_EDIT"
    assert_match '^[0-9]+$' "$(estimate_model_memory "${small}" "EVAL_EDIT")" "small EVAL_EDIT"
    assert_match '^[0-9]+$' "$(estimate_model_memory "${small}" "CERTIFY_EDIT")" "small CERTIFY_EDIT"
    assert_match '^[0-9]+$' "$(estimate_model_memory "${small}" "CREATE_ERROR")" "small CREATE_ERROR"
    assert_match '^[0-9]+$' "$(estimate_model_memory "${small}" "CERTIFY_ERROR")" "small CERTIFY_ERROR"
    assert_match '^[0-9]+$' "$(estimate_model_memory "${small}" "GENERATE_PRESET")" "small GENERATE_PRESET"
    assert_match '^[0-9]+$' "$(estimate_model_memory "${small}" "UNKNOWN")" "small default"
}
