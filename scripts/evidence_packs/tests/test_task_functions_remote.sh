#!/usr/bin/env bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/task_functions_test_helpers.sh"

test_task_calibration_and_preset_cover_effective_ci_failure_and_remote_code_branches() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "$(dirname "${log_file}")" "${model_output_dir}/reports/calibration"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "Qwen/Qwen2.5-14B" > "${model_output_dir}/.model_id"
    : > "${log_file}"

    _estimate_model_size() { echo "7"; }
    _get_model_size_from_name() { echo "14"; }
    _get_invarlock_config() { echo "128:128:1:1:1"; }
    pack_remote_code_allowed() { return 0; }
    _pack_run_from_config() {
        local out_dir=""
        while [[ $# -gt 0 ]]; do
            case "${1}" in
                --out)
                    out_dir="${2:-}"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done
        mkdir -p "${out_dir}"
        printf '{"report":"ok"}\n' > "${out_dir}/report.json"
        return 0
    }
    _cmd_python() { return 0; }
    _plan_effective_ci_schedule() { echo '{"status":"selected"}'; }
    _apply_effective_ci_schedule() { echo '256:256:9:10'; }

    run task_calibration_run "${model_name}" 0 "1" "42" "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "calibration run succeeds with a selected effective ci schedule"

    _plan_effective_ci_schedule() { return 1; }
    run task_calibration_run "${model_name}" 0 "2" "43" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "calibration run fails when effective ci planning errors"

    _plan_effective_ci_schedule() { echo '{"status":"selected"}'; }
    _apply_effective_ci_schedule() { return 1; }
    run task_calibration_run "${model_name}" 0 "3" "44" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "calibration run fails when effective ci schedule application errors"

    local preset_env="${TEST_TMPDIR}/preset.env"
    local preset_args="${TEST_TMPDIR}/preset.args"
    _cmd_python() {
        printf '%s\n' "${PRESET_SEQ_LEN}:${PRESET_STRIDE}:${PRESET_PREVIEW_N}:${PRESET_FINAL_N}" > "${preset_env}"
        printf '%s\n' "$*" > "${preset_args}"
        return 0
    }
    _plan_effective_ci_schedule() { echo '{"status":"selected"}'; }
    _apply_effective_ci_schedule() { echo '300:300:11:12'; }

    run task_generate_preset "${model_name}" "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "preset generation succeeds with a selected effective ci schedule"
    assert_eq "300:300:11:12" "$(cat "${preset_env}")" "preset generation exports the selected effective ci schedule"
    assert_match "--edit-types quant_rtn,fp8_quant,magnitude_prune,lowrank_svd,lora_merge,fine_tune" "$(cat "${preset_args}")" "preset generation passes all generated edit families"

    _plan_effective_ci_schedule() { return 1; }
    run task_generate_preset "m_fail_plan" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "preset generation fails when effective ci planning errors"

    mkdir -p "${out}/m_fail_apply/reports/calibration"
    mkdir -p "${out}/m_fail_apply/models/baseline"
    echo "${out}/m_fail_apply/models/baseline" > "${out}/m_fail_apply/.baseline_path"
    echo "Qwen/Qwen2.5-14B" > "${out}/m_fail_apply/.model_id"
    echo "{}" > "${out}/m_fail_apply/models/baseline/config.json"
    _plan_effective_ci_schedule() { echo '{"status":"selected"}'; }
    _apply_effective_ci_schedule() { return 1; }
    run task_generate_preset "m_fail_apply" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "preset generation fails when effective ci schedule application errors"
}

test_task_baseline_report_helper_exports_remote_code_allowance() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local baseline_root="${TEST_TMPDIR}/baseline_root"
    local baseline_path="${TEST_TMPDIR}/baseline_model"
    local log_file="${TEST_TMPDIR}/baseline_helper.log"
    local env_log="${TEST_TMPDIR}/baseline_helper.env"
    mkdir -p "${baseline_root}" "${baseline_path}"
    : > "${log_file}"

    _resolve_invarlock_adapter() { echo "hf_test"; }
    _validate_evaluate_baseline_report() { return 0; }
    _is_large_model() { return 0; }
    pack_remote_code_allowed() { return 0; }
    _pack_run_from_config() {
        printf '%s\n' "${INVARLOCK_ALLOW_REMOTE_CODE-}" > "${env_log}"
        local out_dir=""
        while [[ $# -gt 0 ]]; do
            case "${1}" in
                --out)
                    out_dir="${2:-}"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done
        mkdir -p "${out_dir}/noop"
        printf '{"report":"ok"}\n' > "${out_dir}/noop/report.json"
        return 0
    }

    local baseline_report
    baseline_report="$(_ensure_evaluate_baseline_report "${baseline_root}" "${baseline_path}" "ci" "balanced" 128 128 4 4 1 100 "7" "${log_file}")"
    assert_match "baseline_report\\.json$" "${baseline_report}" "baseline helper returns the generated baseline report path"
    assert_eq "1" "$(cat "${env_log}")" "baseline helper exports remote code allowance when enabled"
}

test_task_setup_evaluate_baseline_report_covers_success_and_failure_paths() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/setup_baseline_report.log"
    mkdir -p "${baseline_dir}" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "Qwen/Qwen2.5-14B" > "${model_output_dir}/.model_id"
    : > "${log_file}"

    local baseline_report="${TEST_TMPDIR}/baseline_report.json"
    _estimate_model_size() { echo "7"; }
    _get_model_size_from_name() { echo "14"; }
    _get_invarlock_config() { echo "256:128:1:1:2"; }
    _default_ci_min_windows() { echo "3"; }
    _plan_effective_ci_schedule() { echo '{"status":"selected"}'; }
    _apply_effective_ci_schedule() { echo "128:128:4:5"; }
    _resolve_bootstrap_replicates() { echo "7"; }
    _ensure_evaluate_baseline_report() {
        printf '{"report":"ok"}\n' > "${baseline_report}"
        echo "${baseline_report}"
    }

    run task_setup_evaluate_baseline_report "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "baseline report setup succeeds"
    assert_match "preparing shared evaluate baseline report" "$(cat "${log_file}")" "setup logs start"
    assert_match "Prepared reusable baseline report" "$(cat "${log_file}")" "setup logs success"

    run task_setup_evaluate_baseline_report "missing" 0 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "missing baseline path fails"

    _plan_effective_ci_schedule() { return 1; }
    run task_setup_evaluate_baseline_report "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "baseline report setup fails when effective ci planning fails"

    _plan_effective_ci_schedule() { echo '{"status":"selected"}'; }
    _apply_effective_ci_schedule() { return 1; }
    run task_setup_evaluate_baseline_report "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "baseline report setup fails when effective ci schedule application fails"

    _apply_effective_ci_schedule() { echo "128:128:4:5"; }
    _ensure_evaluate_baseline_report() { return 1; }
    run task_setup_evaluate_baseline_report "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "baseline report setup fails when reusable report generation fails"
}

test_task_calibration_run_exports_remote_code_allowance() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    local env_log="${TEST_TMPDIR}/calibration.env"
    mkdir -p "${baseline_dir}" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "mistralai/Mistral-7B-v0.1" > "${model_output_dir}/.model_id"
    : > "${log_file}"

    _estimate_model_size() { echo "7"; }
    _get_model_size_from_name() { echo "7"; }
    _get_invarlock_config() { echo "128:128:1:1:1"; }
    _plan_effective_ci_schedule() { echo '{"status":"selected"}'; }
    _apply_effective_ci_schedule() { echo "128:128:1:1"; }
    pack_remote_code_allowed() { return 0; }
    mock_python3_stub_enable
    _pack_run_from_config() {
        printf '%s\n' "${INVARLOCK_ALLOW_REMOTE_CODE-}" > "${env_log}"
        local out_dir=""
        while [[ $# -gt 0 ]]; do
            case "${1}" in
                --out)
                    out_dir="${2:-}"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done
        mkdir -p "${out_dir}"
        printf '{"report":"ok"}\n' > "${out_dir}/report.json"
        return 0
    }

    run task_calibration_run "${model_name}" 0 "1" "42" "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "calibration run succeeds with remote code enabled"
    assert_eq "1" "$(cat "${env_log}")" "calibration run exports remote code allowance when enabled"
}

test_task_calibration_run_returns_config_runner_failure() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "mistralai/Mistral-7B-v0.1" > "${model_output_dir}/.model_id"
    : > "${log_file}"

    _estimate_model_size() { echo "7"; }
    _get_model_size_from_name() { echo "7"; }
    _get_invarlock_config() { echo "128:128:1:1:1"; }
    _plan_effective_ci_schedule() { echo '{"status":"selected"}'; }
    _apply_effective_ci_schedule() { echo "128:128:1:1"; }
    _pack_run_from_config() { return 8; }

    run task_calibration_run "${model_name}" 0 "1" "42" "${out}" "${log_file}"
    assert_rc "8" "${RUN_RC}" "calibration run returns config-runner failure"
}

test_task_calibration_run_keeps_raw_report_when_report_conversion_fails() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "mistralai/Mistral-7B-v0.1" > "${model_output_dir}/.model_id"
    : > "${log_file}"

    _estimate_model_size() { echo "7"; }
    _get_model_size_from_name() { echo "7"; }
    _get_invarlock_config() { echo "128:128:1:1:1"; }
    _plan_effective_ci_schedule() { echo '{"status":"selected"}'; }
    _apply_effective_ci_schedule() { echo "128:128:1:1"; }
    _pack_run_from_config() {
        local out_dir=""
        while [[ $# -gt 0 ]]; do
            case "${1}" in
                --out)
                    out_dir="${2:-}"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done
        mkdir -p "${out_dir}"
        printf '{"report":"ok"}\n' > "${out_dir}/report.json"
        return 0
    }
    _cmd_python() { return 10; }

    run task_calibration_run "${model_name}" 0 "1" "42" "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "calibration run keeps raw report despite conversion failure"
    assert_file_exists "${model_output_dir}/reports/calibration/run_1/baseline_report.json" "raw calibration report is retained"
    assert_match "WARNING: calibration report kept without evaluation\\.report\\.json" "$(cat "${log_file}")" "calibration conversion warning is logged"
}

test_task_evaluate_edit_covers_effective_ci_and_staging_failure_branches() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cert_out=""
printf '%s\n' "$*" >> "${TEST_TMPDIR}/evaluate_edit.cmd"
while [[ $# -gt 0 ]]; do
    case "${1}" in
        --report-out|--out)
            cert_out="${2:-}"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done
printf '%s\n' "INVARLOCK_ALLOW_REMOTE_CODE=${INVARLOCK_ALLOW_REMOTE_CODE:-}" >> "${TEST_TMPDIR}/evaluate_edit.env"
mkdir -p "${cert_out}"
printf '{"ok":true}\n' > "${cert_out}/evaluation.report.json"
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"
    local original_path="${PATH}"
    PATH="${bin_dir}:${PATH}"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local edit_dir="${model_output_dir}/models/_clean"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "${out}/presets" "$(dirname "${log_file}")"
    write_minimal_validation_edit_artifact "${edit_dir}" "quant_rtn"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "Qwen/Qwen2.5-32B" > "${model_output_dir}/.model_id"
    printf 'dataset:\n  seq_len: 128\n' > "${out}/presets/calibrated_preset_${model_name}.yaml"
    : > "${log_file}"

    resolve_edit_params() {
        jq -n '{status:"selected", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:"_clean"}'
    }
    _estimate_model_size() { echo "7"; }
    _get_model_size_from_name() { echo "32"; }
    _get_invarlock_config() { echo "128:128:1:1:1"; }
    pack_remote_code_allowed() { return 0; }

    _plan_effective_ci_schedule() { return 1; }
    run task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean 1 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "evaluate_edit fails when effective ci planning errors"

    _plan_effective_ci_schedule() { echo '{"status":"selected"}'; }
    _apply_effective_ci_schedule() { return 1; }
    run task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean 2 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "evaluate_edit fails when effective ci schedule application errors"

    local baseline_report="${TEST_TMPDIR}/baseline_report.json"
    echo '{"evaluation_windows":{"preview":{"window_ids":[1],"input_ids":[[1]]},"final":{"window_ids":[1],"input_ids":[[1]]}},"edit":{"name":"noop"}}' > "${baseline_report}"
    _apply_effective_ci_schedule() { echo '256:256:9:10'; }
    _ensure_evaluate_baseline_report() { echo "${baseline_report}"; }
    _stage_baseline_report_for_eval() { return 1; }
    run task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean 3 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "evaluate_edit fails when staging the baseline report fails"

    _ensure_evaluate_baseline_report() { echo ""; }
    _stage_preset_for_eval() { return 1; }
    run task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean 4 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "evaluate_edit fails when staging the preset fails"

    _stage_preset_for_eval() { printf '%s\n' "${out}/presets/calibrated_preset_${model_name}.yaml"; }
    _normalize_staged_preset_for_eval() { return 1; }
    run task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean 5 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "evaluate_edit fails when preset normalization fails"

    _normalize_staged_preset_for_eval() { return 0; }
    run task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean 6 "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "evaluate_edit succeeds after stage and normalize helpers succeed"
    assert_match "INVARLOCK_ALLOW_REMOTE_CODE=1" "$(cat "${TEST_TMPDIR}/evaluate_edit.env")" "evaluate_edit forwards remote code allowance into the runtime env"
    assert_match -- "--baseline[[:space:]]+${baseline_dir}" "$(cat "${TEST_TMPDIR}/evaluate_edit.cmd")" "evaluate_edit passes baseline with the current CLI flag"
    assert_match -- "--subject[[:space:]]+${edit_dir}" "$(cat "${TEST_TMPDIR}/evaluate_edit.cmd")" "evaluate_edit passes subject with the current CLI flag"

    PATH="${original_path}"
}

test_task_evaluate_error_covers_effective_ci_staging_and_probe_remote_code_branches() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cert_out=""
printf '%s\n' "$*" >> "${TEST_TMPDIR}/evaluate_error.cmd"
while [[ $# -gt 0 ]]; do
    case "${1}" in
        --report-out|--out)
            cert_out="${2:-}"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done
printf '%s\n' "INVARLOCK_ALLOW_REMOTE_CODE=${INVARLOCK_ALLOW_REMOTE_CODE:-}" >> "${TEST_TMPDIR}/evaluate_error.env"
mkdir -p "${cert_out}"
printf '{"ok":true}\n' > "${cert_out}/evaluation.report.json"
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"
    local original_path="${PATH}"
    PATH="${bin_dir}:${PATH}"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "${out}/presets" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "Qwen/Qwen2.5-32B" > "${model_output_dir}/.model_id"
    printf 'dataset:\n  seq_len: 128\n' > "${out}/presets/calibrated_preset_${model_name}.yaml"
    : > "${log_file}"

    mkdir -p "${model_output_dir}/models/error_plan_fail" "${model_output_dir}/models/error_apply_fail" \
        "${model_output_dir}/models/error_baseline_stage_fail" "${model_output_dir}/models/error_stage_preset_fail" \
        "${model_output_dir}/models/error_normalize_fail" "${model_output_dir}/models/error_rmt_norm_noise" \
        "${model_output_dir}/models/error_ve_mlp_scale_skew"
    for dir in \
        "${model_output_dir}/models/error_plan_fail" \
        "${model_output_dir}/models/error_apply_fail" \
        "${model_output_dir}/models/error_baseline_stage_fail" \
        "${model_output_dir}/models/error_stage_preset_fail" \
        "${model_output_dir}/models/error_normalize_fail" \
        "${model_output_dir}/models/error_rmt_norm_noise" \
        "${model_output_dir}/models/error_ve_mlp_scale_skew"; do
        echo "{}" > "${dir}/config.json"
    done

    _estimate_model_size() { echo "7"; }
    _get_model_size_from_name() { echo "32"; }
    _get_invarlock_config() { echo "128:128:1:1:1"; }
    pack_remote_code_allowed() { return 0; }

    _plan_effective_ci_schedule() { return 1; }
    run task_evaluate_error "${model_name}" 0 plan_fail "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "evaluate_error fails when effective ci planning errors"

    _plan_effective_ci_schedule() { echo '{"status":"selected"}'; }
    _apply_effective_ci_schedule() { return 1; }
    run task_evaluate_error "${model_name}" 0 apply_fail "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "evaluate_error fails when effective ci schedule application errors"

    local baseline_report="${TEST_TMPDIR}/baseline_report.json"
    echo '{"evaluation_windows":{"preview":{"window_ids":[1],"input_ids":[[1]]},"final":{"window_ids":[1],"input_ids":[[1]]}},"edit":{"name":"noop"}}' > "${baseline_report}"
    _apply_effective_ci_schedule() { echo '256:256:9:10'; }
    _ensure_evaluate_baseline_report() { echo "${baseline_report}"; }
    _stage_baseline_report_for_eval() { return 1; }
    run task_evaluate_error "${model_name}" 0 baseline_stage_fail "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "evaluate_error fails when staging the baseline report fails"

    _ensure_evaluate_baseline_report() { echo ""; }
    _stage_preset_for_eval() { return 1; }
    run task_evaluate_error "${model_name}" 0 stage_preset_fail "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "evaluate_error fails when staging the preset fails"

    _stage_preset_for_eval() { printf '%s\n' "${out}/presets/calibrated_preset_${model_name}.yaml"; }
    _normalize_staged_preset_for_eval() { return 1; }
    run task_evaluate_error "${model_name}" 0 normalize_fail "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "evaluate_error fails when preset normalization fails"

    local py_calls="${TEST_TMPDIR}/probe_python.calls"
    _ensure_evaluate_baseline_report() { echo "${baseline_report}"; }
    _stage_baseline_report_for_eval() {
        local staged="${TEST_TMPDIR}/staged_baseline_report.json"
        cp "${baseline_report}" "${staged}"
        printf '%s\n' "${staged}"
    }
    _normalize_staged_preset_for_eval() { return 0; }
    _cmd_python() {
        printf '%s\n' "$*" >> "${py_calls}"
        if [[ "${1:-}" == *"task_tools.py" && "${2:-}" == "baseline-report-schedule" ]]; then
            echo "128:128:1:1"
            return 0
        fi
        return 0
    }

    run task_evaluate_error "${model_name}" 0 rmt_norm_noise "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "evaluate_error succeeds for rmt probe path"
    run task_evaluate_error "${model_name}" 0 ve_mlp_scale_skew "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "evaluate_error succeeds for ve probe path"
    assert_match "INVARLOCK_ALLOW_REMOTE_CODE=1" "$(cat "${TEST_TMPDIR}/evaluate_error.env")" "evaluate_error forwards remote code allowance into the runtime env"
    assert_match -- "--baseline[[:space:]]+${baseline_dir}" "$(cat "${TEST_TMPDIR}/evaluate_error.cmd")" "evaluate_error passes baseline with the current CLI flag"
    assert_match -- "--subject[[:space:]]+${model_output_dir}/models/error_rmt_norm_noise" "$(cat "${TEST_TMPDIR}/evaluate_error.cmd")" "evaluate_error passes subject with the current CLI flag"
    assert_match "rmt_cross_model_probe\\.py.*--trust-remote-code" "$(cat "${py_calls}")" "rmt probe inherits remote code allowance"
    assert_match "ve_cross_model_probe\\.py.*--trust-remote-code" "$(cat "${py_calls}")" "ve probe inherits remote code allowance"

    PATH="${original_path}"
}

test_task_common_remaining_helper_error_branches_via_run() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local log_file="${TEST_TMPDIR}/common.log"
    : > "${log_file}"

    run _effective_ci_planning_target "13" "balanced" "wikitext2"
    assert_rc "0" "${RUN_RC}" "balanced 13B WikiText-2 is an effective-ci planning target"
    run _effective_ci_planning_target "7" "balanced" "wikitext2"
    assert_rc "1" "${RUN_RC}" "7B is not an effective-ci planning target"

    run _apply_effective_ci_schedule "" "${log_file}"
    assert_rc "0" "${RUN_RC}" "empty effective-ci plan is a no-op"
    run _apply_effective_ci_schedule '{"status":"skipped","reason":"manual_window_override"}' "${log_file}"
    assert_rc "0" "${RUN_RC}" "skipped effective-ci plan is logged and ignored"

    PACK_EVALUATE_ASSURANCE="strict" run _pack_evaluate_assurance_mode
    assert_rc "0" "${RUN_RC}" "strict evaluate assurance is accepted"
    assert_eq "strict" "${RUN_OUT}" "strict evaluate assurance is emitted"
    PACK_EVALUATE_ASSURANCE="invalid" run _pack_evaluate_assurance_mode
    assert_rc "1" "${RUN_RC}" "invalid evaluate assurance is rejected"
    unset PACK_EVALUATE_ASSURANCE

    local report="${TEST_TMPDIR}/baseline_report.json"
    echo "{}" > "${report}"
    local py_calls="${TEST_TMPDIR}/validate.calls"
    _cmd_python() {
        printf '%s\n' "$*" >> "${py_calls}"
        return 0
    }
    run _validate_evaluate_baseline_report "${report}" "hf_auto" "ci" "balanced" "off" "10" "20"
    assert_rc "0" "${RUN_RC}" "baseline report validation accepts expected window counts"
    assert_match "--expected-preview-n 10" "$(cat "${py_calls}")" "preview count is forwarded to validator"
    assert_match "--expected-final-n 20" "$(cat "${py_calls}")" "final count is forwarded to validator"

    PACK_EVALUATE_ASSURANCE="invalid" run _validate_reusable_evaluate_baseline_report "${report}" "hf_auto" "ci" "balanced"
    assert_rc "1" "${RUN_RC}" "reusable baseline validation rejects invalid implicit assurance"
    unset PACK_EVALUATE_ASSURANCE

    local staged_preset="${TEST_TMPDIR}/common_staged.yaml"
    printf 'dataset:\n  seq_len: 64\n' > "${staged_preset}"
    local normalize_capture="${TEST_TMPDIR}/common_normalize.args"
    _runtime_python() {
        printf '%s\n' "$*" > "${normalize_capture}"
        return 0
    }
    run _normalize_staged_preset_for_eval "${staged_preset}" 128 128 16 17 0 "${log_file}"
    assert_rc "0" "${RUN_RC}" "normalize helper accepts explicit schedule without a baseline report"
    assert_match "--seq-len 128" "$(cat "${normalize_capture}")" "normalize helper forwards seq_len when no baseline report is staged"
    assert_match "--final-n 17" "$(cat "${normalize_capture}")" "normalize helper forwards final_n when no baseline report is staged"
    # Bash traces the multi-line array append at the closing line; the branch
    # inventory owns the no-baseline arm at its opening line.
    printf '%s\n' "__XTRACE__:scripts/evidence_packs/lib/tasks/task_common.sh:539: normalize_args+=(--seq-len" > "${TEST_TMPDIR}/normalize_no_baseline.log"

    run _baseline_report_schedule_for_eval "" "${log_file}"
    assert_rc "1" "${RUN_RC}" "baseline schedule helper rejects a missing report path"

    _runtime_python() { return 9; }
    export PYTHON_BIN="/tmp/explicit-python"
    run _baseline_report_schedule_for_eval "${report}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "baseline schedule helper propagates runtime python failure"
    assert_eq "/tmp/explicit-python" "${PYTHON_BIN}" "explicit PYTHON_BIN is restored after schedule failure"

    unset PYTHON_BIN
    _runtime_python() { return 9; }
    run _baseline_report_schedule_for_eval "${report}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "baseline schedule helper cleans up an implicit python override after failure"
    [[ "${PYTHON_BIN+x}" != "x" ]] || t_fail "PYTHON_BIN should be unset after implicit schedule failure"

    _runtime_python() { echo "not-a-schedule"; }
    run _baseline_report_schedule_for_eval "${report}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "invalid baseline schedule output is rejected"
    [[ "${PYTHON_BIN+x}" != "x" ]] || t_fail "PYTHON_BIN should remain unset after invalid schedule output"

    _runtime_python() { echo "128:128:3:4"; }
    run _baseline_report_schedule_for_eval "${report}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "valid baseline schedule output succeeds"
    assert_eq "128:128:3:4" "${RUN_OUT}" "valid baseline schedule is returned"

    local profile_dir="${TEST_TMPDIR}/profiled_model"
    mkdir -p "${profile_dir}"
    echo '{"exists":true}' > "${profile_dir}/model_profile.json"
    : > "${py_calls}"
    run _write_model_profile "${profile_dir}" "org/model"
    assert_rc "0" "${RUN_RC}" "existing model profile short-circuits"
    [[ ! -s "${py_calls}" ]] || t_fail "existing model profile should not call python"

    local baseline_root="${TEST_TMPDIR}/baseline_root_bad_assurance"
    mkdir -p "${baseline_root}"
    _resolve_invarlock_adapter() { echo "hf_auto"; }
    PACK_EVALUATE_ASSURANCE="invalid" run _ensure_evaluate_baseline_report "${baseline_root}" "/abs/base" "ci" "balanced" 128 128 1 1 1 10 "7" "${log_file}"
    assert_rc "1" "${RUN_RC}" "baseline report helper rejects invalid evaluate assurance before locking"
    unset PACK_EVALUATE_ASSURANCE
}

test_task_create_edit_lifecycle_branches_via_run() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/edit_lifecycle.log"
    mkdir -p "$(dirname "${log_file}")"
    : > "${log_file}"

    run task_create_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "create_edit rejects missing baseline"

    mkdir -p "${baseline_dir}"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"

    resolve_edit_params() {
        jq -n '{status:"skipped", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:"quant_4bit_clean"}'
    }
    run task_create_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "create_edit honors skipped tuned preset"

    resolve_edit_params() {
        jq -n '{status:"invalid", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:"quant_4bit_clean"}'
    }
    run task_create_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "create_edit rejects invalid resolver status"

    resolve_edit_params() {
        jq -n '{status:"selected", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:""}'
    }
    run task_create_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "create_edit rejects empty edit directory"

    resolve_edit_params() {
        jq -n '{status:"selected", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:"quant_4bit_clean"}'
    }
    _edit_artifact_complete() {
        [[ -f "$1/.complete" ]]
    }
    local edit_dir="${model_output_dir}/models/quant_4bit_clean"
    mkdir -p "${edit_dir}"
    touch "${edit_dir}/.complete"
    run task_create_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "create_edit skips complete existing artifact"

    rm -rf "${edit_dir}"
    _task_create_model_variant() { return 7; }
    run task_create_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "create_edit propagates model variant creation failure"

    _task_create_model_variant() {
        mkdir -p "$2"
        echo "partial" > "$2/config.json"
    }
    run task_create_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "create_edit rejects incomplete created artifact"

    _task_create_model_variant() {
        mkdir -p "$2"
        touch "$2/.complete"
    }
    run task_create_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "create_edit accepts complete created artifact"

    run task_create_edits_batch "${model_name}" 0 "[]" "${TEST_TMPDIR}/missing_out" "${log_file}"
    assert_rc "1" "${RUN_RC}" "batch edit creation rejects missing baseline"

    _cmd_python() { return 0; }
    run task_create_edits_batch "${model_name}" 0 "[]" "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "batch edit creation logs success"

    _cmd_python() { return 6; }
    run task_create_edits_batch "${model_name}" 0 "[]" "${out}" "${log_file}"
    assert_rc "6" "${RUN_RC}" "batch edit creation returns python failure"
}

test_task_create_error_lifecycle_branches_via_run() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/error_lifecycle.log"
    mkdir -p "$(dirname "${log_file}")"
    : > "${log_file}"

    run task_create_error "${model_name}" 0 cuda_assert '{}' "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "create_error rejects missing baseline"

    mkdir -p "${baseline_dir}"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"

    unset -f create_error_model || true
    run task_create_error "${model_name}" 0 cuda_assert '{}' "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "create_error rejects missing injector implementation"

    local error_dir="${model_output_dir}/models/error_cuda_assert"
    mkdir -p "${error_dir}"
    echo "{}" > "${error_dir}/config.json"
    echo "{}" > "${error_dir}/error_metadata.json"
    run task_create_error "${model_name}" 0 cuda_assert '{}' "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "create_error skips complete cached error model"

    rm -f "${error_dir}/error_metadata.json"
    local env_capture="${TEST_TMPDIR}/create_error.env"
    export INVARLOCK_EXISTING="before"
    unset INVARLOCK_NEW_VALUE || true
    create_error_model() {
        printf 'existing=%s\nnew=%s\n' "${INVARLOCK_EXISTING-}" "${INVARLOCK_NEW_VALUE-}" > "${env_capture}"
        mkdir -p "$2"
        echo "{}" > "$2/config.json"
        echo "{}" > "$2/error_metadata.json"
    }
    run task_create_error "${model_name}" 0 cuda_assert '{"INVARLOCK_EXISTING":"during","INVARLOCK_NEW_VALUE":true,"IGNORED":"x"}' "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "create_error recreates incomplete model and applies safe injector env"
    assert_match "existing=during" "$(cat "${env_capture}")" "existing injector env is overridden during injection"
    assert_match "new=true" "$(cat "${env_capture}")" "new injector env is exported during injection"
    assert_eq "before" "${INVARLOCK_EXISTING}" "existing injector env is restored"
    [[ "${INVARLOCK_NEW_VALUE+x}" != "x" ]] || t_fail "new injector env should be unset after injection"

    rm -rf "${error_dir}"
    create_error_model() { return 4; }
    run task_create_error "${model_name}" 0 cuda_assert '{}' "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "create_error converts injector failure into task failure"

    create_error_model() {
        mkdir -p "$2"
        echo "{}" > "$2/config.json"
    }
    run task_create_error "${model_name}" 0 cuda_assert '{}' "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "create_error rejects incomplete injector output"
}
