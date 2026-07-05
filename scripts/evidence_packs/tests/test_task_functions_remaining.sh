#!/usr/bin/env bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/task_functions_test_helpers.sh"

test_task_evaluate_failure_branches_for_schedule_and_assurance_via_run() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"
    stub_resolve_edit_params

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local edit_dir="${model_output_dir}/models/quant_4bit_clean"
    local error_dir="${model_output_dir}/models/error_cuda_assert"
    local log_file="${TEST_TMPDIR}/evaluate_failures.log"
    mkdir -p "${baseline_dir}" "${error_dir}" "${out}/presets" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "{}" > "${error_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    write_minimal_validation_edit_artifact "${edit_dir}" "quant_rtn"
    printf 'dataset:\n  seq_len: 128\n' > "${out}/presets/calibrated_preset_${model_name}.yaml"
    : > "${log_file}"

    _estimate_model_size() { echo "7"; }
    _get_invarlock_config() { echo "128:128:1:1:1"; }
    _apply_effective_ci_schedule() { echo ""; }
    local baseline_report="${TEST_TMPDIR}/baseline_report.json"
    write_minimal_evaluate_baseline_report "${baseline_report}" 128 128 1 1
    _ensure_evaluate_baseline_report() { echo "${baseline_report}"; }
    _cmd_python() {
        if [[ "${1:-}" == *"validate_artifact.py" ]]; then
            return 0
        fi
        if [[ "${1:-}" == *"task_tools.py" && "${2:-}" == "baseline-report-schedule" ]]; then
            echo "bad-schedule"
            return 0
        fi
        return 0
    }

    run task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean 1 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "evaluate_edit rejects invalid reusable baseline report schedule"

    run task_evaluate_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "evaluate_error rejects invalid reusable baseline report schedule"

    _ensure_evaluate_baseline_report() { echo ""; }
    _stage_preset_for_eval() {
        printf '%s\n' "${out}/presets/calibrated_preset_${model_name}.yaml"
    }
    _normalize_staged_preset_for_eval() { return 0; }
    _cmd_python() {
        if [[ "${1:-}" == *"validate_artifact.py" ]]; then
            return 0
        fi
        return 0
    }
    PACK_EVALUATE_ASSURANCE="bad"
    run task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean 2 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "evaluate_edit rejects invalid assurance before invoking invarlock"

    run task_evaluate_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "evaluate_error rejects invalid assurance before invoking invarlock"
    unset PACK_EVALUATE_ASSURANCE
}

test_task_common_direct_branch_arms_are_asserted_via_run() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    run _get_model_size_from_name "Mixtral-8x7B"
    assert_rc "0" "${RUN_RC}" "moe model size branch succeeds"
    assert_eq "moe" "${RUN_OUT}" "moe branch emits model size"
    run _get_model_size_from_name "Qwen-72B"
    assert_eq "70" "${RUN_OUT}" "70B model size branch emits model size"
    run _get_model_size_from_name "Yi-34B"
    assert_eq "40" "${RUN_OUT}" "40B model size branch emits model size"
    run _get_model_size_from_name "Qwen-32B"
    assert_eq "30" "${RUN_OUT}" "30B model size branch emits model size"
    run _get_model_size_from_name "Qwen-14B"
    assert_eq "13" "${RUN_OUT}" "13B model size branch emits model size"
    run _get_model_size_from_name "small"
    assert_eq "7" "${RUN_OUT}" "default model size branch emits 7B"

    local size expected
    for size in 7 13 30 40 moe 70 unknown; do
        case "${size}" in
            7) expected="512:512:192:192:96" ;;
            13) expected="512:512:192:192:64" ;;
            30) expected="1024:1024:192:192:48" ;;
            40) expected="1024:1024:192:192:32" ;;
            moe) expected="1024:1024:192:192:8" ;;
            70) expected="128:128:192:192:2" ;;
            *) expected="1024:1024:192:192:32" ;;
        esac
        run _get_model_invarlock_config_fallback "${size}"
        assert_eq "${expected}" "${RUN_OUT}" "fallback config branch for ${size}"
    done

    run _bootstrap_replicates_floor_for_tier conservative
    assert_eq "1500" "${RUN_OUT}" "conservative bootstrap floor"
    run _bootstrap_replicates_floor_for_tier balanced
    assert_eq "1200" "${RUN_OUT}" "balanced bootstrap floor"
    run _bootstrap_replicates_floor_for_tier aggressive
    assert_eq "800" "${RUN_OUT}" "aggressive bootstrap floor"
    run _bootstrap_replicates_floor_for_tier custom
    assert_eq "1200" "${RUN_OUT}" "unknown bootstrap floor defaults to balanced"

    unset INVARLOCK_BOOTSTRAP_N
    run _resolve_bootstrap_replicates 14 aggressive
    assert_eq "1000" "${RUN_OUT}" "large models keep large bootstrap count above aggressive floor"
    export INVARLOCK_BOOTSTRAP_N="abc"
    run _resolve_bootstrap_replicates 7 balanced
    assert_eq "abc" "${RUN_OUT}" "non-numeric bootstrap override is passed through"
    unset INVARLOCK_BOOTSTRAP_N

    run _plan_effective_ci_schedule "${TEST_TMPDIR}/missing" 7 balanced wikitext2 validation 42
    assert_rc "0" "${RUN_RC}" "non-target effective-ci plan is a no-op"
    assert_eq "" "${RUN_OUT}" "non-target effective-ci plan emits nothing"
    local model_ref="${TEST_TMPDIR}/model_ref"
    mkdir -p "${model_ref}"
    export INVARLOCK_CERT_MIN_WINDOWS="99"
    run _plan_effective_ci_schedule "${model_ref}" 13 balanced wikitext2 validation 42
    assert_rc "0" "${RUN_RC}" "manual window override skips effective-ci planner"
    assert_match '"reason":[[:space:]]*"manual_window_override"' "${RUN_OUT}" "manual override skip reason is emitted"
    unset INVARLOCK_CERT_MIN_WINDOWS
    run _plan_effective_ci_schedule "${TEST_TMPDIR}/missing_ref" 13 balanced wikitext2 validation 42
    assert_rc "0" "${RUN_RC}" "missing model ref skips effective-ci planner"
    assert_match '"reason":[[:space:]]*"missing_model_ref"' "${RUN_OUT}" "missing model ref skip reason is emitted"

    unset -f pack_model_revision || true
    unset PACK_MODEL_REVISIONS_FILE OUTPUT_DIR
    run _task_get_model_revision "org/model"
    assert_rc "0" "${RUN_RC}" "missing revision file is a soft no-op"
    assert_eq "" "${RUN_OUT}" "missing revision file emits no revision"

    export PACK_BASELINE_REPORT_WAIT_SECS="bad"
    run _baseline_report_wait_secs 7 1 1
    assert_eq "240" "${RUN_OUT}" "invalid wait budget falls back to default"
    unset PACK_BASELINE_REPORT_WAIT_SECS
    export PACK_BASELINE_REPORT_WAIT_HEAVY_WINDOW_TOTAL_MIN="bad"
    run _baseline_report_wait_secs 7 400 400
    assert_eq "1800" "${RUN_OUT}" "invalid heavy-window floor falls back and still detects heavy run"
    unset PACK_BASELINE_REPORT_WAIT_HEAVY_WINDOW_TOTAL_MIN
    run _baseline_report_wait_secs 7 x y
    assert_eq "240" "${RUN_OUT}" "non-numeric window counts keep default wait"

    unset PACK_DEFER_REPORT_RENDERING PACK_DEFER_OPTIONAL_REPORT_RENDERING
    run _pack_defer_report_rendering_enabled
    assert_rc "1" "${RUN_RC}" "defer report rendering defaults off"
    PACK_DEFER_OPTIONAL_REPORT_RENDERING="on" run _pack_defer_report_rendering_enabled
    assert_rc "0" "${RUN_RC}" "legacy defer report rendering flag enables deferral"

    local source_file="${TEST_TMPDIR}/runtime_input.json"
    local cert_dir="${TEST_TMPDIR}/cert"
    local log_file="${TEST_TMPDIR}/common_direct.log"
    printf '{}\n' > "${source_file}"
    : > "${log_file}"
    run _stage_runtime_input_for_eval "${source_file}" "${cert_dir}" "${log_file}" ""
    assert_rc "0" "${RUN_RC}" "runtime input staging succeeds without a label"
    assert_file_exists "${cert_dir}/runtime_inputs/runtime_input.json" "runtime input was staged"

    local staged_preset="${TEST_TMPDIR}/staged.yaml"
    local baseline_report="${TEST_TMPDIR}/baseline_report.json"
    local normalize_capture_file="${TEST_TMPDIR}/normalize.args"
    printf 'dataset:\n  seq_len: 64\n' > "${staged_preset}"
    write_minimal_evaluate_baseline_report "${baseline_report}" 256 256 7 8
    _cmd_python() {
        printf '%s\n' "$*" > "${normalize_capture_file}"
        return 0
    }
    unset PYTHON_BIN
    run _normalize_staged_preset_for_eval "${staged_preset}" 128 128 1 1 1 "${log_file}" "${baseline_report}"
    assert_rc "0" "${RUN_RC}" "normalize helper succeeds with a reusable baseline report"
    assert_match "--baseline-report ${baseline_report}" "$(cat "${normalize_capture_file}")" "normalize helper forwards baseline report"
    assert_match "--skip-overhead-check" "$(cat "${normalize_capture_file}")" "normalize helper forwards skip-overhead flag"
    assert_match "from baseline report" "$(cat "${log_file}")" "normalize helper logs baseline-report schedule source"
    [[ "${PYTHON_BIN+x}" != "x" ]] || t_fail "implicit PYTHON_BIN should be unset after normalize success"

    export PYTHON_BIN="/tmp/explicit-python"
    _cmd_python() {
        if [[ "$*" == *"baseline-report-schedule"* ]]; then
            echo "64:64:2:3"
        fi
        return 0
    }
    run _baseline_report_schedule_for_eval "${baseline_report}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "baseline report schedule succeeds with explicit PYTHON_BIN"
    assert_eq "64:64:2:3" "${RUN_OUT}" "baseline report schedule output is returned"
    assert_eq "/tmp/explicit-python" "${PYTHON_BIN}" "explicit PYTHON_BIN restored after schedule success"
    unset PYTHON_BIN

    local profile_dir="${TEST_TMPDIR}/profile_model"
    local profile_calls="${TEST_TMPDIR}/profile.calls"
    mkdir -p "${profile_dir}"
    _cmd_python() {
        printf '%s\n' "$*" > "${profile_calls}"
        return 0
    }
    run _write_model_profile "${profile_dir}" "org/model"
    assert_rc "0" "${RUN_RC}" "model profile writer ignores python helper failures and successes"
    assert_match "write-model-profile ${profile_dir} org/model" "$(cat "${profile_calls}")" "model profile writer invokes task_tools"
}

test_task_baseline_setup_branches_are_asserted_via_run() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_id="org/model-14B"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/setup_baseline.log"
    mkdir -p "${baseline_dir}" "$(dirname "${log_file}")"
    printf '{}\n' > "${baseline_dir}/config.json"
    : > "${log_file}"

    local memory_update="${TEST_TMPDIR}/memory.update"
    update_model_task_memory() {
        printf '%s:%s:%s\n' "$1" "$2" "$3" > "${memory_update}"
    }
    run task_setup_baseline "${model_id}" "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "cached baseline setup succeeds"
    assert_eq "${baseline_dir}" "$(cat "${model_output_dir}/.baseline_path")" "cached baseline path recorded"
    assert_eq "${model_id}" "$(cat "${model_output_dir}/.model_id")" "cached baseline model id recorded"
    assert_match "${model_name}:${out}:${model_id}" "$(cat "${memory_update}")" "cached baseline updates memory plan"

    rm -rf "${model_output_dir}"
    local setup_path="${TEST_TMPDIR}/setup_model_success"
    setup_model() {
        mkdir -p "${setup_path}"
        printf '{}\n' > "${setup_path}/config.json"
        printf '%s\n' "${setup_path}"
    }
    run task_setup_baseline "${model_id}" "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "setup_model success path succeeds"
    assert_eq "${setup_path}" "$(cat "${model_output_dir}/.baseline_path")" "setup_model baseline path recorded"

    rm -rf "${model_output_dir}"
    setup_model() {
        printf '%s\n' "${TEST_TMPDIR}/missing_setup_model"
        return 0
    }
    run task_setup_baseline "${model_id}" "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "setup_model output must point at a directory"
    unset -f setup_model

    rm -rf "${model_output_dir}"
    _task_get_model_revision() { return 0; }
    PACK_NET="0" run task_setup_baseline "${model_id}" "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "offline setup fails when revision is missing"
    assert_match "Offline mode requires model revisions" "$(cat "${log_file}")" "offline missing revision is logged"

    PACK_NET="1" run task_setup_baseline "${model_id}" "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "online setup fails when revision is missing"
    assert_match "Missing pinned revision" "$(cat "${log_file}")" "online missing revision is logged"

    _task_get_model_revision() { echo "rev123"; }
    PACK_NET="0" run task_setup_baseline "${model_id}" "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "offline setup refuses uncached baseline even with a revision"
    assert_match "Offline mode requested and baseline not cached" "$(cat "${log_file}")" "offline uncached baseline is logged"

    _cmd_python() {
        mkdir -p "${baseline_dir}"
        printf '{}\n' > "${baseline_dir}/config.json"
        return 0
    }
    PACK_NET="1" run task_setup_baseline "${model_id}" "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "inline download succeeds when config is written"
    assert_eq "${baseline_dir}" "$(cat "${model_output_dir}/.baseline_path")" "inline baseline path recorded"

    rm -rf "${model_output_dir}"
    _cmd_python() { return 9; }
    PACK_NET="1" run task_setup_baseline "${model_id}" "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "inline download fails when python helper fails"
}

test_task_baseline_report_calibration_and_preset_remaining_branches_via_run() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/baseline_lifecycle.log"
    mkdir -p "${baseline_dir}" "${model_output_dir}/reports/calibration" "$(dirname "${log_file}")"
    printf '{}\n' > "${baseline_dir}/config.json"
    printf '%s\n' "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    printf '%s\n' "Qwen/Qwen2.5-14B" > "${model_output_dir}/.model_id"
    : > "${log_file}"

    _estimate_model_size() { echo ""; }
    _get_model_size_from_name() { echo "14"; }
    _get_invarlock_config() { echo "256:512:1:1:2"; }
    _default_ci_min_windows() { echo "3"; }
    _plan_effective_ci_schedule() { echo ""; }
    _apply_effective_ci_schedule() { return 0; }
    _resolve_bootstrap_replicates() { echo "33"; }
    _ensure_evaluate_baseline_report() {
        printf '%s\n' "$5:$6:$7:$8:$9:${10}:${11}" > "${TEST_TMPDIR}/ensure.args"
        local report="${TEST_TMPDIR}/prepared_baseline.json"
        printf '{}\n' > "${report}"
        printf '%s\n' "${report}"
    }
    run task_setup_evaluate_baseline_report "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "setup evaluate baseline report succeeds without a selected schedule"
    assert_eq "256:256:3:3:2:33:14" "$(cat "${TEST_TMPDIR}/ensure.args")" "setup report applies stride and CI window overrides"
    assert_match "Pairing override: seq=256, stride=256" "$(cat "${log_file}")" "setup report logs pairing override"
    assert_match "CI window override: preview=3, final=3" "$(cat "${log_file}")" "setup report logs CI window override"

    local run_dir="${model_output_dir}/reports/calibration/run_1"
    mkdir -p "${run_dir}"
    printf '{}\n' > "${run_dir}/baseline_report.json"
    run task_calibration_run "${model_name}" 0 1 42 "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "existing calibration report skips run"

    rm -rf "${run_dir}"
    TASK_PARAMS='{"seq_len":2,"stride":99,"batch_size":5}'
    _estimate_model_size() { echo "7"; }
    _get_model_size_from_name() { echo "7"; }
    _get_invarlock_config() { echo "128:128:1:1:1"; }
    _default_ci_min_windows() { echo "0"; }
    _plan_effective_ci_schedule() { echo ""; }
    _apply_effective_ci_schedule() { return 0; }
    _pack_run_from_config() { return 0; }
    _cmd_python() { return 0; }
    run task_calibration_run "${model_name}" 0 1 42 "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "calibration run succeeds with OOM overrides and no report copy"
    assert_match "OOM override applied: seq=2, stride=2, batch=5" "$(cat "${log_file}")" "calibration logs override after stride clamp"
    unset TASK_PARAMS

    mkdir -p "${out}/presets"
    printf 'preset: true\n' > "${out}/presets/calibrated_preset_${model_name}.yaml"
    run task_generate_preset "${model_name}" "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "existing preset skips generation"

    rm -f "${out}/presets/calibrated_preset_${model_name}.yaml"
    _estimate_model_size() { echo ""; }
    _get_model_size_from_name() { echo "14"; }
    _get_invarlock_config() { echo "512:512:4:5:6"; }
    _plan_effective_ci_schedule() { echo ""; }
    _apply_effective_ci_schedule() { return 0; }
    local preset_args="${TEST_TMPDIR}/preset.args"
    _cmd_python() {
        printf '%s\n' "$*" > "${preset_args}"
        return 0
    }
    run task_generate_preset "${model_name}" "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "preset generation succeeds without an effective-ci selected schedule"
    assert_match "--seq-len 512" "$(cat "${preset_args}")" "preset generation forwards fallback seq_len"
    assert_match "--preview-n 4" "$(cat "${preset_args}")" "preset generation forwards fallback preview count"
}

test_execute_task_dispatch_and_timeout_branches_are_asserted_via_run() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local calls="${TEST_TMPDIR}/dispatch.calls"
    mkdir -p "${out}"
    unset QUEUE_DIR

    task_setup_baseline() { printf 'setup:%s:%s:%s:%s\n' "$1" "$2" "$3" "$4" >> "${calls}"; }
    task_calibration_run() { printf 'calibration:%s:%s:%s:%s:%s\n' "$1" "$2" "$3" "$4" "$5" >> "${calls}"; }
    task_setup_evaluate_baseline_report() { printf 'setup_report:%s:%s:%s\n' "$1" "$2" "$3" >> "${calls}"; }
    task_create_edit() { printf 'create_edit:%s:%s:%s:%s:%s\n' "$1" "$2" "$3" "$4" "$5" >> "${calls}"; }
    task_create_edits_batch() { printf 'batch:%s:%s:%s:%s\n' "$1" "$2" "$3" "$4" >> "${calls}"; }
    task_evaluate_edit() { printf 'eval_edit:%s:%s:%s:%s:%s:%s\n' "$1" "$2" "$3" "$4" "$5" "$6" >> "${calls}"; }
    task_cleanup_edit() { printf 'cleanup_edit:%s:%s:%s:%s\n' "$1" "$2" "$3" "$4" >> "${calls}"; }
    task_create_error() { printf 'create_error:%s:%s:%s:%s:%s\n' "$1" "$2" "$3" "$4" "$5" >> "${calls}"; }
    task_evaluate_error() { printf 'eval_error:%s:%s:%s:%s\n' "$1" "$2" "$3" "$4" >> "${calls}"; }
    task_cleanup_error() { printf 'cleanup_error:%s:%s:%s\n' "$1" "$2" "$3" >> "${calls}"; }
    task_generate_preset() { printf 'preset:%s:%s\n' "$1" "$2" >> "${calls}"; }

    make_dispatch_task() {
        local task_id="$1"
        local task_type="$2"
        local params_json="$3"
        local assigned_json="${4:-null}"
        jq -n \
            --arg id "${task_id}" \
            --arg type "${task_type}" \
            --argjson params "${params_json}" \
            --argjson assigned "${assigned_json}" \
            '{task_id:$id, task_type:$type, model_id:"model/id", model_name:"model_name", assigned_gpus:$assigned, params:$params}' \
            > "${TEST_TMPDIR}/${task_id}.task"
    }

    local task_types=(SETUP_BASELINE CALIBRATION_RUN SETUP_EVALUATE_BASELINE_REPORT CREATE_EDIT CREATE_EDITS_BATCH evaluate_EDIT CLEANUP_EDIT CREATE_ERROR evaluate_ERROR CLEANUP_ERROR GENERATE_PRESET)
    local task_type
    for task_type in "${task_types[@]}"; do
        make_dispatch_task "dispatch_${task_type}" "${task_type}" '{"edit_spec":"quant_rtn:4:32:ffn","version":"v1","run":7,"seed":99,"edit_specs":["a"],"error_type":"cuda_assert","error_env":{"INVARLOCK_X":"1"}}' '"1, 2"'
        run execute_task "${TEST_TMPDIR}/dispatch_${task_type}.task" 0 "${out}"
        assert_rc "0" "${RUN_RC}" "execute_task dispatches ${task_type}"
    done
    assert_match "calibration:model_name:0:7:99" "$(cat "${calls}")" "calibration params are parsed"
    assert_match "create_edit:model_name:0:quant_rtn:4:32:ffn:v1" "$(cat "${calls}")" "edit params are parsed"
    assert_match 'create_error:model_name:0:cuda_assert:\{"INVARLOCK_X":"1"\}' "$(cat "${calls}")" "error env params are parsed compactly"
    assert_eq "1,2" "${CUDA_VISIBLE_DEVICES}" "assigned GPUs are stripped of spaces"

    make_dispatch_task "fallback_gpu" "SETUP_BASELINE" '{}' 'null'
    run execute_task "${TEST_TMPDIR}/fallback_gpu.task" 3 "${out}"
    assert_rc "0" "${RUN_RC}" "execute_task falls back to gpu argument when assigned_gpus is null"
    assert_eq "3" "${CUDA_VISIBLE_DEVICES}" "gpu fallback is exported"

    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}/running"
    TASK_TIMEOUT_SETUP_BASELINE="1"
    _sleep() { :; }
    _cmd_kill() { return 0; }
    _kill_task_process_group() {
        printf 'killed:%s\n' "$1" > "${TEST_TMPDIR}/timeout.kill"
    }
    task_setup_baseline() {
        _sleep 0
    }
    make_dispatch_task "timeout_task" "SETUP_BASELINE" '{}' 'null'
    run execute_task "${TEST_TMPDIR}/timeout_task.task" 0 "${out}"
    assert_rc "124" "${RUN_RC}" "execute_task timeout marker forces 124"
    assert_file_exists "${TEST_TMPDIR}/timeout.kill" "timeout path invokes process-group kill"
    [[ ! -f "${QUEUE_DIR}/running/timeout_task.pid" ]] || t_fail "pid file should be removed after timeout"
    unset TASK_TIMEOUT_SETUP_BASELINE QUEUE_DIR
}

test_edit_and_error_lifecycle_remaining_branches_are_asserted_via_run() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local models_root="${model_output_dir}/models"
    local baseline_dir="${models_root}/baseline"
    local log_file="${TEST_TMPDIR}/lifecycle_remaining.log"
    mkdir -p "${baseline_dir}" "$(dirname "${log_file}")"
    printf '{}\n' > "${baseline_dir}/config.json"
    printf '%s\n' "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    : > "${log_file}"

    resolve_edit_params() {
        jq -n '{status:"skipped", edit_dir_name:"ignored"}'
    }
    PACK_CLEANUP_MODELS="0" run task_cleanup_edit "${model_name}" "quant_rtn:4:32:ffn" clean "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "edit cleanup disabled branch succeeds"
    PACK_CLEANUP_MODELS="0" run task_cleanup_error "${model_name}" "cuda_assert" "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "error cleanup disabled branch succeeds"

    PACK_CLEANUP_MODELS="1" run task_cleanup_edit "${model_name}" "quant_rtn:4:32:ffn" clean "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "edit cleanup skipped resolver branch succeeds"

    resolve_edit_params() {
        jq -n '{status:"selected", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:"quant_4bit_clean"}'
    }
    _edit_artifact_complete() {
        [[ -f "$1/.complete" ]]
    }
    _task_create_model_variant() {
        mkdir -p "$2"
        touch "$2/.complete"
    }
    run task_create_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "create_edit success branch verifies completed artifact"
    assert_match "Created: ${model_output_dir}/models/quant_4bit_clean" "$(cat "${log_file}")" "create_edit success is logged"

    rm -rf "${model_output_dir}/models/error_cuda_assert"
    create_error_model() {
        mkdir -p "$2"
        printf '{}\n' > "$2/config.json"
        printf '{}\n' > "$2/error_metadata.json"
    }
    run task_create_error "${model_name}" 0 cuda_assert '{"INVARLOCK_NEW":"value","bad-key":"ignored","INVARLOCK_NUM":3}' "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "create_error success branch accepts sanitized injector env"
    assert_dir_exists "${model_output_dir}/models/error_cuda_assert" "create_error wrote model directory"
    [[ "${INVARLOCK_NEW+x}" != "x" ]] || t_fail "new injector env should be unset after create_error"

    rm -rf "${model_output_dir}/models/error_cuda_assert"
    create_error_model() {
        mkdir -p "$2"
        printf '{}\n' > "$2/config.json"
    }
    run task_create_error "${model_name}" 0 cuda_assert null "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "create_error fails when injector output is incomplete"
    assert_match "Failed to create error model" "$(cat "${log_file}")" "create_error incomplete output is logged"

    mkdir -p "${models_root}/error_cuda_assert"
    run task_cleanup_error "${model_name}" "cuda_assert" "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "cleanup_error removes existing error model"
    [[ ! -e "${models_root}/error_cuda_assert" ]] || t_fail "cleanup_error should remove existing error model"
}
