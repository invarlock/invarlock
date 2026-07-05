#!/usr/bin/env bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/task_functions_test_helpers.sh"

test_task_evaluate_edit_reuses_baseline_report_applies_ci_override_and_falls_back_label() {
    mock_reset
    push_active_python_bin
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    fixture_write "invarlock.create_cert" ""
    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
fixtures="${TEST_TMPDIR}/fixtures"
mkdir -p "${fixtures}"
echo "PYTHONPATH=${PYTHONPATH:-}" >> "${fixtures}/invarlock.calls"
echo "INVARLOCK_CONFIG_ROOT=${INVARLOCK_CONFIG_ROOT:-}" >> "${fixtures}/invarlock.calls"
echo "INVARLOCK_STORE_EVAL_WINDOWS=${INVARLOCK_STORE_EVAL_WINDOWS:-}" >> "${fixtures}/invarlock.calls"
echo "HF_HOME=${HF_HOME:-}" >> "${fixtures}/invarlock.calls"
echo "invarlock $*" >> "${fixtures}/invarlock.calls"
target=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --report-out|--out)
            shift
            target="${1:-}"
            ;;
    esac
    shift || true
done
if [[ -n "${target}" && -f "${fixtures}/invarlock.create_cert" ]]; then
    mkdir -p "${target}"
    printf '{"ok":true}\n' > "${target}/evaluation.report.json"
fi
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
    mkdir -p "${baseline_dir}" "$(dirname "${log_file}")" "${model_output_dir}/models"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "org/model" > "${model_output_dir}/.model_id"
    : > "${log_file}"
    export HF_HOME="${TEST_TMPDIR}/hf-home"
    mkdir -p "${HF_HOME}"

    # Force CI window override by returning tiny preview/final windows.
    _estimate_model_size() { echo "30"; }
    _get_invarlock_config() { echo "128:128:1:1:1"; }
    export INVARLOCK_CERT_MIN_WINDOWS="400"

    local baseline_report="${TEST_TMPDIR}/baseline_report.json"
    write_minimal_evaluate_baseline_report "${baseline_report}" 128 128 218 218
    _ensure_evaluate_baseline_report() { echo "${baseline_report}"; }

    resolve_edit_params() {
        jq -n '{status:"selected", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:"_clean"}'
    }
    local edit_dir="${model_output_dir}/models/_clean"
    write_minimal_validation_edit_artifact "${edit_dir}" "quant_rtn"

    mkdir -p "${out}/presets"
    echo "{}" > "${out}/presets/calibrated_preset_${model_name}.yaml"

    task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean 1 "${out}" "${log_file}"

    assert_match "CI window override" "$(cat "${log_file}")" "CI window override applied"
    assert_match "Reusing baseline report" "$(cat "${log_file}")" "baseline report reused"
    assert_match "Staged baseline report for evaluate runtime" "$(cat "${log_file}")" "baseline report staged into cert dir"
    assert_match "Staged preset for evaluate runtime" "$(cat "${log_file}")" "preset staged into cert dir"
    assert_match "Normalized staged preset dataset for evaluate runtime from baseline report" "$(cat "${log_file}")" "preset dataset normalized from reused baseline report for evaluate edit"
    assert_file_exists "${model_output_dir}/reports/_clean/run_1/runtime_inputs/baseline_report.json" "staged baseline report exists for evaluate edit"
    assert_file_exists "${model_output_dir}/reports/_clean/run_1/runtime_inputs/calibrated_preset_${model_name}.yaml" "staged preset exists for evaluate edit"
    local staged_preset_contents
    staged_preset_contents="$(cat "${model_output_dir}/reports/_clean/run_1/runtime_inputs/calibrated_preset_${model_name}.yaml")"
    assert_match "seq_len: 128" "${staged_preset_contents}" "staged preset seq_len normalized for evaluate edit"
    assert_match "stride: 128" "${staged_preset_contents}" "staged preset stride normalized for evaluate edit"
    assert_match "preview_n: 218" "${staged_preset_contents}" "staged preset preview_n normalized for evaluate edit"
    assert_match "final_n: 218" "${staged_preset_contents}" "staged preset final_n normalized for evaluate edit"
    assert_match "Using effective baseline report schedule" "$(cat "${log_file}")" "evaluate edit uses effective baseline schedule"
    assert_match "preview_n: 218" "$(cat "${model_output_dir}/reports/_clean/run_1/config_root/runtime/profiles/ci.yaml")" "profile preview_n matches effective baseline report"
    assert_match "final_n: 218" "$(cat "${model_output_dir}/reports/_clean/run_1/config_root/runtime/profiles/ci.yaml")" "profile final_n matches effective baseline report"

    local calls
    calls="$(cat "${TEST_TMPDIR}/fixtures/invarlock.calls")"
    assert_match "PYTHONPATH=${PACK_REPO_PYTHONPATH}" "${calls}" "absolute repo PYTHONPATH forwarded to evaluate"
    assert_match "INVARLOCK_CONFIG_ROOT=${model_output_dir}/reports/_clean/run_1/config_root" "${calls}" "evaluate forwards config root"
    assert_match "INVARLOCK_STORE_EVAL_WINDOWS=1" "${calls}" "evaluate enables stored windows"
    assert_match "HF_HOME=${HF_HOME}" "${calls}" "evaluate preserves inherited HF cache root"
    assert_match "--baseline-report" "${calls}" "baseline report forwarded to invarlock evaluate"
    assert_match "--assurance off" "${calls}" "evaluate forwards default assurance mode"
    assert_match "--edit-label custom" "${calls}" "empty edit label falls back to custom"
    assert_match "/reports/_clean/run_1/runtime_inputs/baseline_report\\.json" "${calls}" "staged baseline report path forwarded to evaluate"
    assert_match "/reports/_clean/run_1/runtime_inputs/calibrated_preset_${model_name}\\.yaml" "${calls}" "staged preset path forwarded to evaluate"
    assert_match "skip_overhead_check: true" "$(cat "${model_output_dir}/reports/_clean/run_1/config_root/runtime/profiles/ci.yaml")" "large-model evaluate profile carries skip_overhead setting"

    PATH="${original_path}"
    pop_active_python_bin
}

test_normalize_staged_preset_for_eval_handles_sparse_yaml_and_json_inputs() {
    mock_reset
    push_active_python_bin
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local log_file="${TEST_TMPDIR}/normalize.log"
    : > "${log_file}"

    local sparse_yaml="${TEST_TMPDIR}/sparse.yaml"
    cat > "${sparse_yaml}" <<'YAML'
guards:
  spectral:
    max_caps: 15
YAML

    _normalize_staged_preset_for_eval "${sparse_yaml}" 128 128 192 192 1 "${log_file}"
    local sparse_yaml_contents
    sparse_yaml_contents="$(cat "${sparse_yaml}")"
    assert_match "seq_len: 128" "${sparse_yaml_contents}" "yaml preset gets seq_len when dataset is absent"
    assert_match "stride: 128" "${sparse_yaml_contents}" "yaml preset gets stride when dataset is absent"
    assert_match "preview_n: 192" "${sparse_yaml_contents}" "yaml preset gets preview_n when dataset is absent"
    assert_match "final_n: 192" "${sparse_yaml_contents}" "yaml preset gets final_n when dataset is absent"
    assert_match "skip_overhead_check: true" "${sparse_yaml_contents}" "yaml preset gets skip_overhead policy"
    assert_match "guards:" "${sparse_yaml_contents}" "yaml preset keeps existing sections"

    local json_preset="${TEST_TMPDIR}/preset.json"
    echo '{}' > "${json_preset}"
    _normalize_staged_preset_for_eval "${json_preset}" 256 256 200 220 0 "${log_file}"
    local json_contents
    json_contents="$(cat "${json_preset}")"
    assert_match "seq_len: 256" "${json_contents}" "json preset gets seq_len"
    assert_match "stride: 256" "${json_contents}" "json preset gets stride"
    assert_match "preview_n: 200" "${json_contents}" "json preset gets preview_n"
    assert_match "final_n: 220" "${json_contents}" "json preset gets final_n"

    local report_backed_preset="${TEST_TMPDIR}/report-backed.yaml"
    echo '{}' > "${report_backed_preset}"
    local report_backed_baseline="${TEST_TMPDIR}/report-backed-baseline.json"
    write_minimal_evaluate_baseline_report "${report_backed_baseline}" 512 512 218 218
    _normalize_staged_preset_for_eval "${report_backed_preset}" 512 512 400 400 0 "${log_file}" "${report_backed_baseline}"
    local report_backed_contents
    report_backed_contents="$(cat "${report_backed_preset}")"
    assert_match "seq_len: 512" "${report_backed_contents}" "baseline-report backed preset keeps seq_len"
    assert_match "stride: 512" "${report_backed_contents}" "baseline-report backed preset keeps stride"
    assert_match "preview_n: 218" "${report_backed_contents}" "baseline-report backed preset uses actual preview_n"
    assert_match "final_n: 218" "${report_backed_contents}" "baseline-report backed preset uses actual final_n"
    assert_match "from baseline report" "$(cat "${log_file}")" "normalization logs baseline-report source"
    pop_active_python_bin
}

test_task_evaluate_error_reuses_baseline_report_for_nonstructural_errors_and_applies_ci_override() {
    mock_reset
    push_active_python_bin
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    fixture_write "invarlock.create_cert" ""

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local error_dir="${model_output_dir}/models/error_norm_collapse"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "${error_dir}" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "{}" > "${error_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "org/model" > "${model_output_dir}/.model_id"
    : > "${log_file}"

    _get_invarlock_config() { echo "128:128:1:1:1"; }
    export INVARLOCK_CERT_MIN_WINDOWS="400"

    local baseline_report="${TEST_TMPDIR}/baseline_report.json"
    write_minimal_evaluate_baseline_report "${baseline_report}" 128 128 218 218
    _ensure_evaluate_baseline_report() { echo "${baseline_report}"; }

    mkdir -p "${out}/presets"
    echo "{}" > "${out}/presets/calibrated_preset_${model_name}.yaml"

    export PACK_DEFER_REPORT_RENDERING="1"
    task_evaluate_error "${model_name}" 0 norm_collapse "${out}" "${log_file}"
    unset PACK_DEFER_REPORT_RENDERING

    assert_match "CI window override" "$(cat "${log_file}")" "CI window override applied"
    assert_match "Reusing baseline report" "$(cat "${log_file}")" "baseline report reused"
    assert_match "Staged baseline report for evaluate runtime" "$(cat "${log_file}")" "baseline report staged into error cert dir"
    assert_match "Staged preset for evaluate runtime" "$(cat "${log_file}")" "preset staged into error cert dir"
    assert_match "Normalized staged preset dataset for evaluate runtime from baseline report" "$(cat "${log_file}")" "preset dataset normalized from reused baseline report for evaluate error"
    assert_file_exists "${model_output_dir}/reports/errors/norm_collapse/evaluation.report.json" "error cert written"
    assert_file_exists "${model_output_dir}/reports/errors/norm_collapse/runtime_inputs/baseline_report.json" "staged baseline report exists for evaluate error"
    assert_file_exists "${model_output_dir}/reports/errors/norm_collapse/runtime_inputs/calibrated_preset_${model_name}.yaml" "staged preset exists for evaluate error"
    local staged_error_preset_contents
    staged_error_preset_contents="$(cat "${model_output_dir}/reports/errors/norm_collapse/runtime_inputs/calibrated_preset_${model_name}.yaml")"
    assert_match "seq_len: 128" "${staged_error_preset_contents}" "staged preset seq_len normalized for evaluate error"
    assert_match "stride: 128" "${staged_error_preset_contents}" "staged preset stride normalized for evaluate error"
    assert_match "preview_n: 218" "${staged_error_preset_contents}" "staged preset preview_n normalized for evaluate error"
    assert_match "final_n: 218" "${staged_error_preset_contents}" "staged preset final_n normalized for evaluate error"
    assert_match "Using effective baseline report schedule" "$(cat "${log_file}")" "evaluate error uses effective baseline schedule"
    assert_match "preview_n: 218" "$(cat "${model_output_dir}/reports/errors/norm_collapse/config_root/runtime/profiles/ci.yaml")" "error profile preview_n matches effective baseline report"
    assert_match "final_n: 218" "$(cat "${model_output_dir}/reports/errors/norm_collapse/config_root/runtime/profiles/ci.yaml")" "error profile final_n matches effective baseline report"
    local calls
    calls="$(cat "${TEST_TMPDIR}/fixtures/invarlock.calls")"
    assert_match "--assurance off" "${calls}" "error evaluate forwards default assurance mode"
    pop_active_python_bin
}

test_task_evaluate_error_skips_baseline_report_reuse_for_structural_errors() {
    mock_reset
    push_active_python_bin
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    fixture_write "invarlock.create_cert" ""

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local error_dir="${model_output_dir}/models/error_nan_injection"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "${error_dir}" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "{}" > "${error_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "org/model" > "${model_output_dir}/.model_id"
    : > "${log_file}"

    _get_invarlock_config() { echo "128:128:1:1:1"; }
    export INVARLOCK_CERT_MIN_WINDOWS="192"

    local baseline_report="${TEST_TMPDIR}/baseline_report.json"
    echo '{"evaluation_windows":{"preview":{"window_ids":[1],"input_ids":[[1]]},"final":{"window_ids":[1],"input_ids":[[1]]}},"edit":{"name":"noop"}}' > "${baseline_report}"
    local ensure_calls="${TEST_TMPDIR}/ensure.calls"
    _ensure_evaluate_baseline_report() {
        echo called >> "${ensure_calls}"
        echo "${baseline_report}"
    }

    mkdir -p "${out}/presets"
    echo "{}" > "${out}/presets/calibrated_preset_${model_name}.yaml"

    export PACK_DEFER_REPORT_RENDERING="1"
    task_evaluate_error "${model_name}" 0 nan_injection "${out}" "${log_file}"
    unset PACK_DEFER_REPORT_RENDERING

    local log_text
    log_text="$(cat "${log_file}")"
    assert_match "Baseline report reuse disabled for structural error: nan_injection" "${log_text}" "structural error path disables reused baseline report"
    if [[ -f "${ensure_calls}" ]]; then
        t_fail "expected structural error path to skip baseline report lookup"
    fi
    if [[ "${log_text}" =~ Reusing\ baseline\ report ]]; then
        t_fail "expected structural error path to omit baseline report reuse"
    fi
    assert_file_exists "${model_output_dir}/reports/errors/nan_injection/evaluation.report.json" "error cert written"
    if [[ -e "${model_output_dir}/reports/errors/nan_injection/runtime_inputs/baseline_report.json" ]]; then
        t_fail "expected no staged baseline report for structural error"
    fi
    assert_file_exists "${model_output_dir}/reports/errors/nan_injection/runtime_inputs/calibrated_preset_${model_name}.yaml" "staged preset exists for structural error"
    local calls
    calls="$(cat "${TEST_TMPDIR}/fixtures/invarlock.calls")"
    assert_match "--defer-report-rendering" "${calls}" "deferred optional report rendering flag forwarded for error evaluation"
    assert_match "--assurance off" "${calls}" "structural error evaluate forwards default assurance mode"
    if [[ "${calls}" =~ --baseline-report ]]; then
        t_fail "expected structural error evaluate call to omit --baseline-report"
    fi
    pop_active_python_bin
}

test_task_evaluate_error_emits_structural_failure_report_when_structural_eval_cannot_write_one() {
    mock_reset
    push_active_python_bin
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    fixture_write "invarlock.rc" "1"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local error_dir="${model_output_dir}/models/error_inf_injection"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "${error_dir}" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "{}" > "${error_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "org/model" > "${model_output_dir}/.model_id"
    : > "${log_file}"

    _get_invarlock_config() { echo "128:128:1:1:1"; }
    export INVARLOCK_CERT_MIN_WINDOWS="192"

    mkdir -p "${out}/presets"
    echo "{}" > "${out}/presets/calibrated_preset_${model_name}.yaml"

    local cert_dir="${model_output_dir}/reports/errors/inf_injection"
    local source_dir="${cert_dir}/source/000000"
    local edited_dir="${cert_dir}/edited/000000"
    mkdir -p "${source_dir}" "${edited_dir}"
    cat > "${source_dir}/report.json" <<'EOF'
{
  "run_id": "source-run",
  "meta": {
    "model_id": "org/model",
    "adapter": "hf_causal",
    "seed": 7,
    "device": "cpu"
  },
  "data": {
    "dataset": "dummy",
    "split": "validation",
    "seq_len": 8,
    "stride": 8,
    "preview_n": 2,
    "final_n": 2
  },
  "edit": {
    "name": "noop",
    "plan_digest": "x",
    "deltas": {
      "params_changed": 0,
      "layers_modified": 0
    }
  },
  "guards": [],
  "metrics": {
    "primary_metric": {
      "kind": "ppl_causal",
      "unit": "ppl",
      "direction": "lower",
      "aggregation_scope": "token",
      "paired": true,
      "gating_basis": "upper",
      "supports_bootstrap": true,
      "preview": 9.429,
      "final": 8.893,
      "drift_band": {
        "min": 0.8878,
        "max": 1.0859
      }
    }
  },
  "evaluation_windows": {
    "final": {
      "window_ids": [1, 2],
      "logloss": [2.30, 2.31],
      "token_counts": [100, 100]
    }
  },
  "artifacts": {
    "events_path": "",
    "logs_path": "",
    "checkpoint_path": null
  },
  "flags": {
    "guard_recovered": false,
    "rollback_reason": null
  }
}
EOF
    cat > "${source_dir}/runtime.manifest.json" <<'EOF'
{
  "manifest_version": 1,
  "generated_at_utc": "2026-04-19T08:00:00+00:00",
  "verifier_contract_version": "runtime-manifest-v1",
  "report": {
    "path": "/tmp/source.report.json",
    "filename": "report.json",
    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "config": {
    "path": null,
    "sha256": null,
    "source": "missing"
  },
  "execution_mode": "container",
  "runtime": {
    "image_ref": "invarlock-runtime:cuda-local@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "image_digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "container_execution": true,
    "allow_network": true,
    "allow_remote_code": true,
    "allow_third_party_plugins": false
  }
}
EOF
    echo '{"run_id":"edited-run"}' > "${edited_dir}/report.json"
    : > "${edited_dir}/events.jsonl"

    task_evaluate_error "${model_name}" 0 inf_injection "${out}" "${log_file}"

    local cert_path="${model_output_dir}/reports/errors/inf_injection/evaluation.report.json"
    assert_file_exists "${cert_path}" "structural failure report written for structural error"
    assert_file_exists "${model_output_dir}/reports/errors/inf_injection/runtime.manifest.json" "runtime manifest written for structural failure report"
    assert_match "synthesized structural-failure evaluation\\.report\\.json for structural error: inf_injection" "$(cat "${log_file}")" "structural failure log emitted"
    assert_match "evidence-pack-structural-failure-report-v1" "$(cat "${cert_path}")" "structural failure format marker present"
    assert_match '"primary_metric_acceptable":[[:space:]]*false' "$(cat "${cert_path}")" "structural failure report marks primary metric unacceptable"
    assert_match "evidence_pack_structural_failure" "$(cat "${model_output_dir}/reports/errors/inf_injection/runtime.manifest.json")" "runtime manifest records structural failure context"
    pop_active_python_bin
}

test_task_timeout_and_profile_helpers() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    TASK_TIMEOUT_DEFAULT=""
    assert_eq "" "$(_get_task_timeout "X")" "empty timeout returns blank"
    TASK_TIMEOUT_DEFAULT="12"
    assert_eq "12" "$(_get_task_timeout "X")" "numeric timeout returned"

    local kills=""
    _cmd_ps() {
        if [[ "$*" == *"-p 111"* ]]; then
            echo "200"
        else
            echo "100"
        fi
    }
    _cmd_kill() { kills+="$*;"; return 0; }
    _sleep() { :; }
    _kill_task_process_group 111
    _cmd_ps() { echo "100"; }
    _kill_task_process_group 111
    assert_match "-TERM" "${kills}" "kill invoked"

    local rc=0
    _write_model_profile "${TEST_TMPDIR}/missing" "model" || rc=$?
    assert_rc "1" "${rc}" "missing baseline dir returns non-zero"

    PACK_DEFER_REPORT_RENDERING="yes"
    _pack_defer_report_rendering_enabled || t_fail "truthy defer rendering flag should enable optional report deferral"
    unset PACK_DEFER_REPORT_RENDERING
}

test_execute_task_dispatches_all_task_types() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}"
    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}/running"

    TASK_TIMEOUT_DEFAULT=""
    set +m

    task_setup_baseline() { :; }
    task_calibration_run() { :; }
    task_create_edit() { :; }
    task_create_edits_batch() { :; }
    task_evaluate_edit() { :; }
    task_cleanup_edit() { :; }
    task_create_error() { :; }
    task_evaluate_error() { :; }
    task_cleanup_error() { :; }
    task_generate_preset() { :; }
    task_setup_evaluate_baseline_report() { :; }

    make_task() {
        local task_id="$1"
        local task_type="$2"
        local params_json="${3:-}"
        if [[ -z "${params_json}" ]]; then
            params_json="{}"
        fi
        jq -n --arg id "${task_id}" --arg type "${task_type}" --argjson params "${params_json}" \
            '{task_id:$id, task_type:$type, model_id:"m", model_name:"model", status:"pending", assigned_gpus:null, params:$params}' \
            > "${TEST_TMPDIR}/${task_id}.task"
    }

    local types=(SETUP_BASELINE CALIBRATION_RUN SETUP_EVALUATE_BASELINE_REPORT CREATE_EDIT CREATE_EDITS_BATCH evaluate_EDIT CLEANUP_EDIT CREATE_ERROR evaluate_ERROR CLEANUP_ERROR GENERATE_PRESET)
    local type
    for type in "${types[@]}"; do
        make_task "task_${type}" "${type}" '{}'
        execute_task "${TEST_TMPDIR}/task_${type}.task" 0 "${out}"
    done

    make_task "task_unknown" "UNKNOWN" '{}'
    run execute_task "${TEST_TMPDIR}/task_unknown.task" 0 "${out}"
    assert_rc "1" "${RUN_RC}" "unknown task returns non-zero"

    [[ ! -f "${QUEUE_DIR}/running/task_SETUP_BASELINE.pid" ]] || t_fail "expected pid file removed"
}

test_execute_task_handles_job_control_enabled() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}"
    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}/running"

    task_setup_baseline() { :; }

    jq -n '{task_id:"t1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"model", status:"pending", assigned_gpus:null, params:{}}' \
        > "${TEST_TMPDIR}/t1.task"

    set -m
    run execute_task "${TEST_TMPDIR}/t1.task" 0 "${out}"
    assert_rc "0" "${RUN_RC}" "execute_task succeeds with job control enabled"
    set +m
}

test_execute_task_timeout_triggers_marker() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}"
    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}/running"

    TASK_TIMEOUT_DEFAULT="1"
    _sleep() { :; }
    _cmd_kill() { return 0; }
    _kill_task_process_group() { :; }
    task_setup_baseline() { :; }

    jq -n '{task_id:"t1", task_type:"SETUP_BASELINE", model_id:"m", model_name:"model", status:"pending", assigned_gpus:null, params:{}}' \
        > "${TEST_TMPDIR}/t1.task"

    run execute_task "${TEST_TMPDIR}/t1.task" 0 "${out}"
    assert_rc "124" "${RUN_RC}" "timeout returns 124"
}

test_task_setup_baseline_revision_errors() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "$(dirname "${log_file}")"
    : > "${log_file}"

    unset -f setup_model
    _task_get_model_revision() { echo ""; }

    PACK_NET=1
    run task_setup_baseline "org/model" "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "missing revision errors in net mode"

    PACK_NET=0
    run task_setup_baseline "org/model" "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "offline mode without revision errors"

    _task_get_model_revision() { echo "rev"; }
    PACK_NET=0
    run task_setup_baseline "org/model" "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "offline without cache errors"
}

test_task_create_edit_handles_skip_and_invalid() {
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
    : > "${log_file}"

    resolve_edit_params() {
        jq -n '{status:"skipped", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:"quant_4bit_clean"}'
    }
    task_create_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean "${out}" "${log_file}"

    resolve_edit_params() {
        jq -n '{status:"selected", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:""}'
    }
    run task_create_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "empty edit_dir_name errors"

    resolve_edit_params() {
        jq -n '{status:"selected", edit_type:"mystery", param1:"4", param2:"32", scope:"ffn", edit_dir_name:"mystery_clean"}'
    }
    run task_create_edit "${model_name}" 0 "mystery:4:32:ffn" clean "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "unknown edit type errors"
}

test_task_evaluate_edit_skip_and_invalid() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "$(dirname "${log_file}")" "${model_output_dir}/models"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    : > "${log_file}"

    resolve_edit_params() {
        jq -n '{status:"skipped", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:"quant_4bit_clean"}'
    }
    task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean 1 "${out}" "${log_file}"

    resolve_edit_params() {
        jq -n '{status:"invalid", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:"quant_4bit_clean"}'
    }
    run task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean 1 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "invalid resolution errors"

    resolve_edit_params() {
        jq -n '{status:"selected", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:"quant_4bit_clean"}'
    }
    mkdir -p "${model_output_dir}/models/quant_4bit_clean"
    _cmd_python() { return 1; }
    run task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean 1 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "edit metadata validation failure errors"
    unset -f _cmd_python
    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"
}

test_resolve_edit_params_uses_tuned_presets() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    mkdir -p "${model_output_dir}"
    echo "org/model" > "${model_output_dir}/.model_id"

    local tuned_file="${TEST_TMPDIR}/tuned_edit_params.json"
    cat > "${tuned_file}" <<'JSON'
{
  "models": {
    "org/model": {
      "quant_rtn": {
        "status": "selected",
        "bits": 8,
        "group_size": 128,
        "scope": "ffn",
        "edit_dir_name": "quant_8bit_clean"
      }
    }
  },
  "defaults": {
    "fp8_quant": {
      "status": "selected",
      "format": "e4m3fn",
      "scope": "ffn",
      "edit_dir_name": "fp8_e4m3fn_clean"
    },
    "lora_merge": {
      "status": "selected",
      "rank": 4,
      "alpha": 8,
      "scope": "attn",
      "edit_dir_name": "lora_rank4_clean"
    },
    "fine_tune": {
      "status": "selected",
      "learning_rate": 0.0001,
      "steps": 1,
      "scope": "ffn",
      "edit_dir_name": "fine_tune_step1_clean"
    }
  }
}
JSON

    PACK_TUNED_EDIT_PARAMS_FILE="${tuned_file}"
    export PACK_TUNED_EDIT_PARAMS_FILE
    local resolved
    resolved=$(resolve_edit_params "${model_output_dir}" "quant_rtn:clean:ffn" "clean")
    assert_eq "selected" "$(echo "${resolved}" | jq -r '.status')" "quant_rtn resolved"
    assert_eq "8" "$(echo "${resolved}" | jq -r '.param1')" "quant_rtn bits"
    assert_eq "128" "$(echo "${resolved}" | jq -r '.param2')" "quant_rtn group_size"

    resolved=$(resolve_edit_params "${model_output_dir}" "fp8_quant:clean:ffn" "clean")
    assert_eq "selected" "$(echo "${resolved}" | jq -r '.status')" "fp8_quant resolved"
    assert_eq "e4m3fn" "$(echo "${resolved}" | jq -r '.param1')" "fp8_quant format"

    resolved=$(resolve_edit_params "${model_output_dir}" "lora_merge:clean:attn" "clean")
    assert_eq "selected" "$(echo "${resolved}" | jq -r '.status')" "lora_merge resolved"
    assert_eq "4" "$(echo "${resolved}" | jq -r '.param1')" "lora_merge rank"
    assert_eq "8" "$(echo "${resolved}" | jq -r '.param2')" "lora_merge alpha"
    assert_eq "lora_rank4_clean" "$(echo "${resolved}" | jq -r '.edit_dir_name')" "lora_merge edit dir"

    resolved=$(resolve_edit_params "${model_output_dir}" "fine_tune:clean:ffn" "clean")
    assert_eq "selected" "$(echo "${resolved}" | jq -r '.status')" "fine_tune resolved"
    assert_eq "0.0001" "$(echo "${resolved}" | jq -r '.param1')" "fine_tune learning rate"
    assert_eq "1" "$(echo "${resolved}" | jq -r '.param2')" "fine_tune steps"
    assert_eq "fine_tune_step1_clean" "$(echo "${resolved}" | jq -r '.edit_dir_name')" "fine_tune edit dir"

    PACK_TUNED_EDIT_PARAMS_FILE="${TEST_TMPDIR}/missing.json"
    export PACK_TUNED_EDIT_PARAMS_FILE
    resolved=$(resolve_edit_params "${model_output_dir}" "lowrank_svd:clean:ffn" "clean")
    assert_eq "missing" "$(echo "${resolved}" | jq -r '.status')" "missing tuned params file returns missing"
}


test_task_calibration_run_guard_order_branches() {
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
    echo "org/model" > "${model_output_dir}/.model_id"
    : > "${log_file}"

    _estimate_model_size() { echo "7"; }
    _get_model_size_from_name() { echo "7"; }
    _get_invarlock_config() { echo "512:256:1:1:4"; }

    fixture_write "python3.create_report" ""
    mock_python3_stub_enable
    cat > "${TEST_TMPDIR}/fixtures/python3.capture_env_keys" <<'EOF'
PYTHONPATH
INVARLOCK_CONFIG_ROOT
TMPDIR
HF_HOME
EOF
    export TMPDIR="${TEST_TMPDIR}/tmpdir"
    export HF_HOME="${TEST_TMPDIR}/hf-home"
    mkdir -p "${TMPDIR}" "${HF_HOME}"

    PACK_GUARDS_ORDER="variance"
    task_calibration_run "${model_name}" 0 "1" "42" "${out}" "${log_file}"
    assert_match "variance" "$(cat "${model_output_dir}/reports/calibration/run_1/calibration_config.yaml")" "explicit guard order used"
    assert_match "skip_overhead_check: true" "$(cat "${model_output_dir}/reports/calibration/run_1/calibration_config.yaml")" "calibration config carries skip_overhead policy"
    assert_match "skip_overhead_check: true" "$(cat "${model_output_dir}/reports/calibration/run_1/config_root/runtime/profiles/ci.yaml")" "calibration profile carries skip_overhead policy"
    local env_capture
    env_capture="$(cat "${TEST_TMPDIR}/fixtures/python3.env")"
    assert_match "PYTHONPATH=${PACK_REPO_PYTHONPATH}" "${env_capture}" "calibration injects repo pythonpath"
    assert_match "INVARLOCK_CONFIG_ROOT=${model_output_dir}/reports/calibration/run_1/config_root" "${env_capture}" "calibration injects config root"
    assert_match "TMPDIR=${TMPDIR}" "${env_capture}" "calibration preserves tmpdir override"
    assert_match "HF_HOME=${HF_HOME}" "${env_capture}" "calibration preserves HF cache root"

    PACK_GUARDS_ORDER=" , "
    task_calibration_run "${model_name}" 0 "2" "43" "${out}" "${log_file}"
    assert_match "spectral" "$(cat "${model_output_dir}/reports/calibration/run_2/calibration_config.yaml")" "default guard order used"
}

test_task_helper_effective_ci_and_runtime_stage_error_branches() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local plan_json
    plan_json="$(_plan_effective_ci_schedule "${TEST_TMPDIR}/missing-model" "13" "balanced" "wikitext2" "validation" "42")"
    assert_eq "missing_model_ref" "$(printf '%s' "${plan_json}" | jq -r '.reason')" "effective ci planner skips when model ref is missing"

    local log_file="${TEST_TMPDIR}/helper.log"
    : > "${log_file}"

    run _stage_runtime_input_for_eval "${TEST_TMPDIR}/missing" "${TEST_TMPDIR}/cert" "${log_file}" "preset"
    assert_rc "1" "${RUN_RC}" "staging helper fails when source file is missing"

    local source_file="${TEST_TMPDIR}/preset.yaml"
    printf 'dataset:\n  seq_len: 64\n' > "${source_file}"

    mkdir() { return 1; }
    run _stage_runtime_input_for_eval "${source_file}" "${TEST_TMPDIR}/cert_mkdir_fail" "${log_file}" "preset"
    assert_rc "1" "${RUN_RC}" "staging helper fails when runtime_inputs directory cannot be created"
    unset -f mkdir

    cp() { return 1; }
    run _stage_runtime_input_for_eval "${source_file}" "${TEST_TMPDIR}/cert_cp_fail" "${log_file}" "preset"
    assert_rc "1" "${RUN_RC}" "staging helper fails when the staged file cannot be copied"
    unset -f cp

    run _normalize_staged_preset_for_eval "${TEST_TMPDIR}/missing.yaml" 128 128 16 16 0 "${log_file}"
    assert_rc "1" "${RUN_RC}" "normalize helper fails when the staged preset is missing"

    local staged_preset="${TEST_TMPDIR}/staged.yaml"
    printf 'dataset:\n  seq_len: 64\n' > "${staged_preset}"
    export PYTHON_BIN="$(command -v python || command -v python3)"
    local explicit_python="${PYTHON_BIN}"
    _runtime_python() { return 1; }
    run _normalize_staged_preset_for_eval "${staged_preset}" 128 128 16 16 0 "${log_file}"
    assert_rc "1" "${RUN_RC}" "normalize helper propagates runtime python failures with explicit PYTHON_BIN"
    assert_eq "${explicit_python}" "${PYTHON_BIN}" "explicit PYTHON_BIN is restored after normalize failure"

    unset PYTHON_BIN
    run _normalize_staged_preset_for_eval "${staged_preset}" 128 128 16 16 0 "${log_file}"
    assert_rc "1" "${RUN_RC}" "normalize helper propagates runtime python failures without PYTHON_BIN"
    [[ "${PYTHON_BIN+x}" != "x" ]] || t_fail "PYTHON_BIN should be unset after normalize failure without an explicit override"
}
