#!/usr/bin/env bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/task_functions_test_helpers.sh"

test_task_cleanup_edit_and_error_cover_guard_paths() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local models_root="${model_output_dir}/models"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${models_root}/baseline" "$(dirname "${log_file}")"
    : > "${log_file}"

    PACK_CLEANUP_MODELS="0"
    task_cleanup_edit "${model_name}" "quant_rtn:4:32:all" "clean" "${out}" "${log_file}"
    task_cleanup_error "${model_name}" "cuda_assert" "${out}" "${log_file}"

    PACK_CLEANUP_MODELS="1"
    resolve_edit_params() { jq -n '{status:"skipped", edit_dir_name:"ignored"}'; }
    task_cleanup_edit "${model_name}" "quant_rtn:4:32:all" "clean" "${out}" "${log_file}"

    resolve_edit_params() { jq -n '{status:"invalid", edit_dir_name:"ignored"}'; }
    run task_cleanup_edit "${model_name}" "quant_rtn:4:32:all" "clean" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "invalid edit cleanup resolution fails"

    resolve_edit_params() { jq -n '{status:"selected", edit_dir_name:""}'; }
    run task_cleanup_edit "${model_name}" "quant_rtn:4:32:all" "clean" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "empty edit dir name fails"

    rm -rf "${models_root}"
    resolve_edit_params() { jq -n '{status:"selected", edit_dir_name:"quant_4bit_clean"}'; }
    run task_cleanup_edit "${model_name}" "quant_rtn:4:32:all" "clean" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "missing models root fails cleanup"
    mkdir -p "${models_root}/baseline"

    resolve_edit_params() { jq -n '{status:"selected", edit_dir_name:"baseline"}'; }
    run task_cleanup_edit "${model_name}" "quant_rtn:4:32:all" "clean" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "baseline path cleanup is rejected"

    resolve_edit_params() { jq -n '{status:"selected", edit_dir_name:"missing_parent/edit"}'; }
    run task_cleanup_edit "${model_name}" "quant_rtn:4:32:all" "clean" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "missing edit parent is rejected"

    resolve_edit_params() { jq -n '{status:"selected", edit_dir_name:"../../escape"}'; }
    run task_cleanup_edit "${model_name}" "quant_rtn:4:32:all" "clean" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "paths outside models root are rejected"

    resolve_edit_params() { jq -n '{status:"selected", edit_dir_name:"quant_4bit_clean"}'; }
    run task_cleanup_edit "${model_name}" "quant_rtn:4:32:all" "clean" "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "missing edit paths are treated as already cleaned"

    mkdir -p "${models_root}/quant_4bit_clean"
    run task_cleanup_edit "${model_name}" "quant_rtn:4:32:all" "clean" "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "edit cleanup removes existing model directory"
    [[ ! -e "${models_root}/quant_4bit_clean" ]] || t_fail "edit directory removed on cleanup success path='${models_root}/quant_4bit_clean'"

    mkdir -p "${models_root}/quant_4bit_clean"
    rm() {
        if [[ "${1:-}" == "-rf" ]]; then
            return 1
        fi
        command rm "$@"
    }
    run task_cleanup_edit "${model_name}" "quant_rtn:4:32:all" "clean" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "edit cleanup failure propagates"
    unset -f rm

    run task_cleanup_error "${model_name}" "" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "cleanup_error requires an error type"

    rm -rf "${models_root}"
    run task_cleanup_error "${model_name}" "cuda_assert" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "missing error models root fails cleanup"
    mkdir -p "${models_root}/baseline"

    run task_cleanup_error "${model_name}" "missing_parent/error" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "cleanup_error rejects missing parent paths"

    run task_cleanup_error "${model_name}" "../../escape" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "cleanup_error rejects paths outside models root"

    run task_cleanup_error "${model_name}" "cuda_assert" "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "missing error paths are treated as already cleaned"

    mkdir -p "${models_root}/error_cuda_assert"
    run task_cleanup_error "${model_name}" "cuda_assert" "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "error cleanup removes existing model directory"
    [[ ! -e "${models_root}/error_cuda_assert" ]] || t_fail "error directory removed on cleanup success path='${models_root}/error_cuda_assert'"

    mkdir -p "${models_root}/error_cuda_assert"
    rm() {
        if [[ "${1:-}" == "-rf" ]]; then
            return 1
        fi
        command rm "$@"
    }
    run task_cleanup_error "${model_name}" "cuda_assert" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "error cleanup failure propagates"
    unset -f rm
}

test_task_evaluate_error_probe_warning_branches() {
    mock_reset
    push_active_python_bin
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

    _estimate_model_size() { echo "7"; }
    _ensure_evaluate_baseline_report() { echo "${TEST_TMPDIR}/baseline_report.json"; }
    write_minimal_evaluate_baseline_report "${TEST_TMPDIR}/baseline_report.json"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cmd="${1:-}"
shift || true
if [[ "${cmd}" != "evaluate" ]]; then
  exit 0
fi
cert_out=""
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
mkdir -p "${cert_out}"
echo '{}' > "${cert_out}/evaluation.report.json"
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"
    PATH="${bin_dir}:${PATH}"
    export PATH

    _cmd_python() {
        case "${1:-}" in
            *rmt_cross_model_probe.py|*ve_cross_model_probe.py)
                return 9
                ;;
            *)
                command "${PYTHON_BIN}" "$@"
                ;;
        esac
    }

    local rmt_dir="${model_output_dir}/models/error_rmt_norm_noise_case"
    mkdir -p "${rmt_dir}"
    echo "{}" > "${rmt_dir}/config.json"
    run task_evaluate_error "${model_name}" 0 "rmt_norm_noise_case" "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "probe failures do not fail evaluate_error"

    rm -f "${TEST_TMPDIR}/baseline_report.json"
    local rmt_skip_dir="${model_output_dir}/models/error_rmt_norm_noise_skip"
    mkdir -p "${rmt_skip_dir}"
    echo "{}" > "${rmt_skip_dir}/config.json"
    run task_evaluate_error "${model_name}" 0 "rmt_norm_noise_skip" "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "missing probe prerequisites do not fail evaluate_error"

    local ve_skip_dir="${model_output_dir}/models/error_ve_mlp_scale_skew_skip"
    mkdir -p "${ve_skip_dir}"
    echo "{}" > "${ve_skip_dir}/config.json"
    run task_evaluate_error "${model_name}" 0 "ve_mlp_scale_skew_skip" "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "missing ve probe prerequisites do not fail evaluate_error"

    write_minimal_evaluate_baseline_report "${TEST_TMPDIR}/baseline_report.json"
    local ve_dir="${model_output_dir}/models/error_ve_mlp_scale_skew_case"
    mkdir -p "${ve_dir}"
    echo "{}" > "${ve_dir}/config.json"
    run task_evaluate_error "${model_name}" 0 "ve_mlp_scale_skew_case" "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "ve probe failures do not fail evaluate_error"

    local log_text
    log_text="$(cat "${log_file}")"
    assert_match "RMT cross-model probe failed" "${log_text}" "rmt probe failure warning logged"
    assert_match "Skipping RMT cross-model probe" "${log_text}" "rmt probe skip warning logged"
    assert_match "VE cross-model probe failed" "${log_text}" "ve probe failure warning logged"
    assert_match "Skipping VE cross-model probe" "${log_text}" "ve probe skip warning logged"
    pop_active_python_bin
}

test_task_helpers_cover_fallback_branches() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    assert_eq "moe" "$(_get_model_size_from_name "Mixtral-8x7B")" "moe detection"
    assert_eq "13" "$(_get_model_size_from_name "model-13b")" "13B detection"
    assert_eq "70" "$(_get_model_size_from_name "Qwen1.5-72B")" "70B detection"
    assert_eq "40" "$(_get_model_size_from_name "01-ai/Yi-34B")" "40B detection"
    assert_eq "30" "$(_get_model_size_from_name "Qwen2.5-32B")" "30B detection"

    assert_eq "512:512:192:192:64" "$(_get_model_invarlock_config_fallback "13")" "13B config"
    assert_eq "1024:1024:192:192:48" "$(_get_model_invarlock_config_fallback "30")" "30B config"
    assert_eq "1024:1024:192:192:32" "$(_get_model_invarlock_config_fallback "40")" "40B config"
    assert_eq "1024:1024:192:192:8" "$(_get_model_invarlock_config_fallback "moe")" "moe config"
    assert_eq "128:128:192:192:2" "$(_get_model_invarlock_config_fallback "70")" "70B config"
    assert_eq "1024:1024:192:192:32" "$(_get_model_invarlock_config_fallback "unknown")" "fallback config"

    estimate_model_params() { echo "42"; }
    assert_eq "42" "$(_estimate_model_size "model")" "uses estimate_model_params"
    unset -f estimate_model_params
    assert_eq "7" "$(_estimate_model_size "model")" "fallback size"

    get_model_invarlock_config() { echo "custom"; }
    assert_eq "custom" "$(_get_invarlock_config "7")" "uses get_model_invarlock_config"
    unset -f get_model_invarlock_config

    pack_model_revision() { echo "rev"; }
    assert_eq "rev" "$(_task_get_model_revision "org/model")" "uses pack_model_revision"
    unset -f pack_model_revision

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    mkdir -p "${OUTPUT_DIR}/state"
    echo '{"models":{"org/model":{"revision":"abc"}}}' > "${OUTPUT_DIR}/state/model_revisions.json"
    unset PACK_MODEL_REVISIONS_FILE
    assert_eq "abc" "$(_task_get_model_revision "org/model")" "fallback uses model_revisions"

    local resolved
    resolved="$(resolve_edit_params "${TEST_TMPDIR}" "quant_rtn:4:32:ffn" "clean")"
    assert_match "quant_4bit_clean" "${resolved}" "resolve_edit_params builds edit_dir_name"

    _is_large_model "moe" || t_fail "expected moe to be large"
    _is_large_model "model-30b" || t_fail "expected 30b string to be large"
}

test_task_helpers_cover_bootstrap_replicates_floor_logic() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    assert_eq "1500" "$(_bootstrap_replicates_floor_for_tier conservative)" "conservative floor"
    assert_eq "1200" "$(_bootstrap_replicates_floor_for_tier balanced)" "balanced floor"
    assert_eq "800" "$(_bootstrap_replicates_floor_for_tier aggressive)" "aggressive floor"
    assert_eq "1200" "$(_bootstrap_replicates_floor_for_tier unknown)" "default floor"

    export INVARLOCK_BOOTSTRAP_N="500"
    assert_eq "1500" "$(_resolve_bootstrap_replicates "7" conservative)" "floor clamps low bootstrap replicates"
    unset INVARLOCK_BOOTSTRAP_N
}

test_task_baseline_report_helpers_wait_sanitizes_interval_and_large_timeout() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local baseline_root="${TEST_TMPDIR}/baseline_root_sanitize_wait"
    mkdir -p "${baseline_root}"
    local baseline_report="${baseline_root}/baseline_report.json"
    rm -f "${baseline_report}"
    mkdir -p "${baseline_root}/.baseline_lock"

    _resolve_invarlock_adapter() { echo "hf_test"; }
    _validate_evaluate_baseline_report() { return 0; }

    export PACK_BASELINE_REPORT_WAIT_INTERVAL_SECS="0"
    export PACK_BASELINE_REPORT_WAIT_SECS_LARGE="nope"

    _sleep() {
        echo "{}" > "${baseline_report}"
        return 0
    }

    local log_file="${TEST_TMPDIR}/baseline_wait_sanitize.log"
    : > "${log_file}"
    local waited
    waited="$(_ensure_evaluate_baseline_report "${baseline_root}" "/abs/base" "ci" "balanced" 128 128 1 1 1 10 "moe" "${log_file}")"
    assert_eq "${baseline_report}" "${waited}" "wait loop returns once report appears"

    unset PACK_BASELINE_REPORT_WAIT_INTERVAL_SECS PACK_BASELINE_REPORT_WAIT_SECS_LARGE
}

test_task_baseline_report_helpers_wait_iters_floor_to_one() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local baseline_root="${TEST_TMPDIR}/baseline_root_wait_iters"
    mkdir -p "${baseline_root}"
    local baseline_report="${baseline_root}/baseline_report.json"
    rm -f "${baseline_report}"

    local lock_dir="${baseline_root}/.baseline_lock"

    _resolve_invarlock_adapter() { echo "hf_test"; }
    _validate_evaluate_baseline_report() { return 0; }

    export PACK_BASELINE_REPORT_WAIT_INTERVAL_SECS="10"
    export PACK_BASELINE_REPORT_WAIT_SECS="1"

    mkdir() {
        if [[ "${1:-}" == "${lock_dir}" ]]; then
            echo "{}" > "${baseline_report}"
            return 1
        fi
        command mkdir "$@"
    }

    local log_file="${TEST_TMPDIR}/baseline_wait_iters.log"
    : > "${log_file}"
    local waited
    waited="$(_ensure_evaluate_baseline_report "${baseline_root}" "/abs/base" "ci" "balanced" 128 128 1 1 1 10 "7" "${log_file}")"
    assert_eq "${baseline_report}" "${waited}" "wait_iters clamped to 1 still returns report"

    unset -f mkdir
    unset PACK_BASELINE_REPORT_WAIT_INTERVAL_SECS PACK_BASELINE_REPORT_WAIT_SECS
}

test_task_evaluate_error_repairs_missing_tensors_config_when_available() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    fixture_write "invarlock.create_cert" ""

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local error_dir="${model_output_dir}/models/error_missing_tensors"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "${error_dir}" "$(dirname "${log_file}")"
    echo '{"num_hidden_layers": 32}' > "${baseline_dir}/config.json"
    echo '{"num_hidden_layers": 16}' > "${error_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    : > "${log_file}"

    task_evaluate_error "${model_name}" 0 missing_tensors "${out}" "${log_file}"
}

test_task_evaluate_error_emits_rmt_cross_model_probe_for_rmt_norm_noise() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    fixture_write "invarlock.create_cert" ""

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local error_dir="${model_output_dir}/models/error_rmt_norm_noise"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "${error_dir}" "$(dirname "${log_file}")" "${out}/presets"
    echo "{}" > "${baseline_dir}/config.json"
    echo "{}" > "${error_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    : > "${log_file}"

    cat > "${out}/presets/calibrated_preset_${model_name}.yaml" <<'YAML'
dataset:
  provider: wikitext2
  split: validation
YAML

    local probe_baseline_report_file="${TEST_TMPDIR}/baseline_report.json"
    echo '{"evaluation_windows":{"preview":{"input_ids":[[1,2,3]],"attention_masks":[[1,1,1]]},"final":{"input_ids":[[4,5,6]],"attention_masks":[[1,1,1]]}}}' > "${probe_baseline_report_file}"
    _ensure_evaluate_baseline_report() {
        echo "${probe_baseline_report_file}"
    }

    local py_calls="${TEST_TMPDIR}/python.calls"
    _cmd_python() {
        echo "$*" >> "${py_calls}"
        if [[ "${1:-}" == *"task_tools.py" && "${2:-}" == "baseline-report-schedule" ]]; then
            echo "128:128:1:1"
            return 0
        fi
        if [[ "${1:-}" == *"rmt_cross_model_probe.py" ]]; then
            local out_path=""
            local prev=""
            for arg in "$@"; do
                if [[ "${prev}" == "--out" ]]; then
                    out_path="${arg}"
                    break
                fi
                prev="${arg}"
            done
            if [[ -n "${out_path}" ]]; then
                mkdir -p "$(dirname "${out_path}")"
                echo '{"probe":"rmt_cross_model_v1","stable":false}' > "${out_path}"
            fi
        fi
        return 0
    }

    task_evaluate_error "${model_name}" 0 rmt_norm_noise "${out}" "${log_file}"

    local probe_path="${model_output_dir}/reports/errors/rmt_norm_noise/rmt_probe.json"
    assert_file_exists "${probe_path}" "rmt probe artifact emitted"
    assert_match "rmt_cross_model_probe\\.py" "$(cat "${py_calls}")" "rmt probe script invoked"
}

test_task_create_model_variant_dispatch_and_fallback_errors() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    create_model_variant() { echo "main:$*"; return 0; }
    run _task_create_model_variant "/b" "/o" "quant_rtn" "8" "128" "ffn" "0"
    assert_rc "0" "${RUN_RC}" "dispatches to create_model_variant when available"
    assert_match "^main:" "${RUN_OUT}" "create_model_variant called"

    unset -f create_model_variant
    unset -f create_edited_model create_fp8_model create_pruned_model create_lowrank_model create_lora_merged_model create_fine_tuned_model || true

    run _task_create_model_variant "/b" "/o" "quant_rtn" "8" "128" "ffn" "0"
    assert_rc "1" "${RUN_RC}" "quant_rtn without create_edited_model returns non-zero"

    run _task_create_model_variant "/b" "/o" "fp8_quant" "e4m3fn" "" "ffn" "0"
    assert_rc "1" "${RUN_RC}" "fp8_quant without create_fp8_model returns non-zero"

    run _task_create_model_variant "/b" "/o" "magnitude_prune" "0.1" "" "ffn" "0"
    assert_rc "1" "${RUN_RC}" "magnitude_prune without create_pruned_model returns non-zero"

    run _task_create_model_variant "/b" "/o" "lowrank_svd" "8" "" "ffn" "0"
    assert_rc "1" "${RUN_RC}" "lowrank_svd without create_lowrank_model returns non-zero"

    run _task_create_model_variant "/b" "/o" "lora_merge" "4" "8" "attn" "0"
    assert_rc "1" "${RUN_RC}" "lora_merge without create_lora_merged_model returns non-zero"

    run _task_create_model_variant "/b" "/o" "fine_tune" "0.0001" "1" "ffn" "0"
    assert_rc "1" "${RUN_RC}" "fine_tune without create_fine_tuned_model returns non-zero"

    run _task_create_model_variant "/b" "/o" "nope" "" "" "" "0"
    assert_rc "1" "${RUN_RC}" "unknown edit type returns non-zero"
}

test_task_create_model_variant_fallback_success_paths() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    unset -f create_model_variant || true

    create_edited_model() { echo "edited:$*"; return 0; }
    create_fp8_model() { echo "fp8:$*"; return 0; }
    create_pruned_model() { echo "pruned:$*"; return 0; }
    create_lowrank_model() { echo "lowrank:$*"; return 0; }
    create_lora_merged_model() { echo "lora:$*"; return 0; }
    create_fine_tuned_model() { echo "fine:$*"; return 0; }

    run _task_create_model_variant "/b" "/o" "quant_rtn" "8" "128" "ffn" "0"
    assert_rc "0" "${RUN_RC}" "quant_rtn fallback succeeds when helper exists"
    assert_match "^edited:" "${RUN_OUT}" "quant_rtn uses create_edited_model"

    run _task_create_model_variant "/b" "/o" "fp8_quant" "e4m3fn" "" "ffn" "0"
    assert_rc "0" "${RUN_RC}" "fp8_quant fallback succeeds when helper exists"
    assert_match "^fp8:" "${RUN_OUT}" "fp8_quant uses create_fp8_model"

    run _task_create_model_variant "/b" "/o" "magnitude_prune" "0.1" "" "ffn" "0"
    assert_rc "0" "${RUN_RC}" "magnitude_prune fallback succeeds when helper exists"
    assert_match "^pruned:" "${RUN_OUT}" "magnitude_prune uses create_pruned_model"

    run _task_create_model_variant "/b" "/o" "lowrank_svd" "8" "" "ffn" "0"
    assert_rc "0" "${RUN_RC}" "lowrank_svd fallback succeeds when helper exists"
    assert_match "^lowrank:" "${RUN_OUT}" "lowrank_svd uses create_lowrank_model"

    run _task_create_model_variant "/b" "/o" "lora_merge" "4" "8" "attn" "0"
    assert_rc "0" "${RUN_RC}" "lora_merge fallback succeeds when helper exists"
    assert_match "^lora:" "${RUN_OUT}" "lora_merge uses create_lora_merged_model"

    run _task_create_model_variant "/b" "/o" "fine_tune" "0.0001" "1" "ffn" "0"
    assert_rc "0" "${RUN_RC}" "fine_tune fallback succeeds when helper exists"
    assert_match "^fine:" "${RUN_OUT}" "fine_tune uses create_fine_tuned_model"
}

test_task_edit_artifact_probe_helpers_cover_present_and_missing_cases() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local edit_dir="${TEST_TMPDIR}/edit_probe"
    mkdir -p "${edit_dir}"

    run _edit_artifact_has_weights "${edit_dir}"
    assert_rc "1" "${RUN_RC}" "missing weight files are rejected"

    touch "${edit_dir}/foo.safetensors"
    run _edit_artifact_has_weights "${edit_dir}"
    assert_rc "0" "${RUN_RC}" "safetensors shard satisfies weight probe"

    run _edit_artifact_has_tokenizer "${edit_dir}"
    assert_rc "1" "${RUN_RC}" "missing tokenizer files are rejected"

    touch "${edit_dir}/tokenizer.json"
    run _edit_artifact_has_tokenizer "${edit_dir}"
    assert_rc "0" "${RUN_RC}" "tokenizer json satisfies tokenizer probe"
}

test_task_baseline_report_helpers_cover_reuse_lock_race_and_wait_paths() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    run _resolve_invarlock_adapter ""
    assert_ne "0" "${RUN_RC}" "empty adapter input returns non-zero"

    run _validate_evaluate_baseline_report "" "hf" "ci" "balanced"
    assert_ne "0" "${RUN_RC}" "missing baseline report returns non-zero"

    local baseline_root="${TEST_TMPDIR}/baseline_root"
    mkdir -p "${baseline_root}"
    local baseline_report="${baseline_root}/baseline_report.json"

    _resolve_invarlock_adapter() { echo "hf_test"; }
    _validate_evaluate_baseline_report() { return 0; }

    echo "{}" > "${baseline_report}"
    local reuse
    reuse="$(_ensure_evaluate_baseline_report "${baseline_root}" "/abs/base" "ci" "balanced" 128 128 1 1 1 10 "7" "${TEST_TMPDIR}/log.txt")"
    assert_eq "${baseline_report}" "${reuse}" "reuse returns existing baseline report"

    rm -f "${baseline_report}"
    local lock_dir="${baseline_root}/.baseline_lock"
    mkdir() {
        if [[ "${1:-}" == "${lock_dir}" ]]; then
            command mkdir "$@"
            echo "{}" > "${baseline_report}"
            return 0
        fi
        command mkdir "$@"
    }
    local raced
    raced="$(_ensure_evaluate_baseline_report "${baseline_root}" "/abs/base" "ci" "balanced" 128 128 1 1 1 10 "7" "${TEST_TMPDIR}/log.txt")"
    assert_eq "${baseline_report}" "${raced}" "lock re-check returns when report appears"
    unset -f mkdir

    # Wait path: lock already held by another worker.
    rm -f "${baseline_report}"
    mkdir -p "${lock_dir}"
    _sleep() {
        echo "{}" > "${baseline_report}"
        return 0
    }
    local waited_file="${TEST_TMPDIR}/baseline_waited.out"
    local waited_rc=0
    if _ensure_evaluate_baseline_report "${baseline_root}" "/abs/base" "ci" "balanced" 128 128 1 1 1 10 "7" "${TEST_TMPDIR}/log.txt" > "${waited_file}"; then
        waited_rc=0
    else
        waited_rc=$?
    fi
    local waited
    waited="$(cat "${waited_file}")"
    assert_rc "0" "${waited_rc}" "wait loop exits successfully once report appears"
    assert_eq "${baseline_report}" "${waited}" "wait loop returns when report exists"
}

test_task_baseline_report_helpers_execute_python_wrappers() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local calls="${TEST_TMPDIR}/python.calls"
    _cmd_python() {
        echo "python $*" >> "${calls}"
        if [[ $# -eq 3 && "${2:-}" == "resolve-adapter" ]]; then
            echo "hf_auto"
        fi
        return 0
    }

    run _resolve_invarlock_adapter "org/model"
    assert_rc "0" "${RUN_RC}" "adapter resolver runs python wrapper"
    assert_eq "hf_auto" "${RUN_OUT}" "adapter output forwarded"

    local report="${TEST_TMPDIR}/baseline_report.json"
    echo "{}" > "${report}"

    run _validate_evaluate_baseline_report "${report}" "hf_auto" "ci" "balanced"
    assert_rc "0" "${RUN_RC}" "baseline report validation runs python wrapper"
    assert_file_exists "${calls}" "python stub invoked"

    : > "${calls}"
    run _validate_reusable_evaluate_baseline_report "${report}" "hf_auto" "ci" "balanced"
    assert_rc "0" "${RUN_RC}" "reusable baseline report validation runs python wrapper"
    local reusable_call
    reusable_call="$(cat "${calls}")"
    assert_match "validate-baseline-report ${report} hf_auto ci balanced off" "${reusable_call}" "reusable baseline reports validate as cache inputs"
    [[ "${reusable_call}" != *"--expected-preview-n"* ]] || t_fail "reusable baseline validation should not enforce requested preview count"
    [[ "${reusable_call}" != *"--expected-final-n"* ]] || t_fail "reusable baseline validation should not enforce requested final count"

    : > "${calls}"
    PACK_EVALUATE_ASSURANCE="strict" run _validate_reusable_evaluate_baseline_report "${report}" "hf_auto" "ci" "balanced"
    assert_rc "0" "${RUN_RC}" "reusable baseline report validation honors strict evaluate assurance"
    assert_match "validate-baseline-report ${report} hf_auto ci balanced strict" "$(cat "${calls}")" "strict reusable baseline reports validate as strict cache inputs"
}

test_task_baseline_report_helpers_cover_generate_baseline_report_path() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local baseline_root="${TEST_TMPDIR}/baseline_root_generate"
    mkdir -p "${baseline_root}"
    local baseline_report="${baseline_root}/baseline_report.json"
    rm -f "${baseline_report}"

    local log_file="${TEST_TMPDIR}/baseline_generate.log"
    : > "${log_file}"

    _resolve_invarlock_adapter() { echo "hf_test"; }
    _validate_evaluate_baseline_report() { return 0; }

    export PACK_GUARDS_ORDER="invariants, spectral , rmt"
    export HF_HOME="${TEST_TMPDIR}/hf-home"
    mkdir -p "${HF_HOME}"
    fixture_write "python3.create_report_nested" ""
    cat > "${TEST_TMPDIR}/fixtures/python3.capture_env_keys" <<'EOF'
PYTHONPATH
INVARLOCK_CONFIG_ROOT
INVARLOCK_STORE_EVAL_WINDOWS
INVARLOCK_SKIP_OVERHEAD_CHECK
HF_HOME
EOF

    local generated
    generated="$(_ensure_evaluate_baseline_report "${baseline_root}" "/abs/base" "ci" "balanced" 128 128 1 1 1 10 "14" "${log_file}")"

    assert_eq "${baseline_report}" "${generated}" "baseline report path returned"
    assert_file_exists "${baseline_report}" "baseline report generated"
    assert_match '"seed": 42' "$(cat "${baseline_report}")" "baseline report seed is stamped for evaluate reuse"
    assert_match "Generating reusable baseline report" "$(cat "${log_file}")" "generation logged"
    local env_capture
    env_capture="$(cat "${TEST_TMPDIR}/fixtures/python3.env")"
    assert_match "PYTHONPATH=${PACK_REPO_PYTHONPATH}" "${env_capture}" "baseline run forwards repo pythonpath"
    assert_match "INVARLOCK_CONFIG_ROOT=${baseline_root}/config_root" "${env_capture}" "baseline run forwards config root"
    assert_match "INVARLOCK_STORE_EVAL_WINDOWS=1" "${env_capture}" "baseline run stores eval windows"
    assert_match "INVARLOCK_SKIP_OVERHEAD_CHECK=1" "${env_capture}" "large-model baseline run skips overhead check"
    assert_match "HF_HOME=${HF_HOME}" "${env_capture}" "baseline run preserves inherited HF cache root"
    local profile_contents
    profile_contents="$(cat "${baseline_root}/config_root/runtime/profiles/ci.yaml")"
    assert_match "skip_overhead_check: true" "${profile_contents}" "large-model baseline report profile carries skip_overhead policy"
    local baseline_yaml_contents
    baseline_yaml_contents="$(cat "${baseline_root}/baseline_noop.yaml")"
    assert_match "assurance:" "${baseline_yaml_contents}" "baseline report config carries assurance section"
    assert_match "mode: \"off\"" "${baseline_yaml_contents}" "baseline report config defaults evaluate assurance to off"
}

test_task_baseline_report_helpers_warn_when_seed_stamp_fails() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local baseline_root="${TEST_TMPDIR}/baseline_root_stamp_warning"
    mkdir -p "${baseline_root}"
    local baseline_report="${baseline_root}/baseline_report.json"
    local log_file="${TEST_TMPDIR}/baseline_stamp_warning.log"
    : > "${log_file}"

    _resolve_invarlock_adapter() { echo "hf_test"; }
    _validate_evaluate_baseline_report() { return 0; }
    _cmd_python() {
        local script="$1"
        shift || true
        if [[ "${script}" == *"task_tools.py" && "${1:-}" == "stamp-baseline-report-seed" ]]; then
            return 9
        fi
        return 0
    }
    _pack_run_from_config() {
        local out_dir=""
        while [[ $# -gt 0 ]]; do
            if [[ "${1}" == "--out" ]]; then
                out_dir="${2:-}"
                break
            fi
            shift
        done
        mkdir -p "${out_dir}/run_1"
        printf '{"ok":true}\n' > "${out_dir}/run_1/report.json"
    }

    local generated
    generated="$(_ensure_evaluate_baseline_report "${baseline_root}" "/abs/base" "ci" "balanced" 128 128 1 1 1 10 "7" "${log_file}")"

    assert_eq "${baseline_report}" "${generated}" "baseline report path returned even when seed stamp warns"
    assert_file_exists "${baseline_report}" "baseline report is still staged after warning"
    assert_match "WARNING: Failed to stamp baseline report seed" "$(cat "${log_file}")" "seed stamp warning is logged"
}

test_task_baseline_report_helpers_return_runner_failure() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local baseline_root="${TEST_TMPDIR}/baseline_root_runner_failure"
    mkdir -p "${baseline_root}"
    local log_file="${TEST_TMPDIR}/baseline_runner_failure.log"
    : > "${log_file}"

    _resolve_invarlock_adapter() { echo "hf_test"; }
    _validate_evaluate_baseline_report() { return 1; }
    _pack_run_from_config() { return 9; }

    local rc=0
    ( _ensure_evaluate_baseline_report "${baseline_root}" "/abs/base" "ci" "balanced" 128 128 1 1 1 10 "7" "${log_file}" ) || rc=$?
    assert_rc "1" "${rc}" "baseline report helper returns failure when config runner fails"
}

test_task_baseline_report_helpers_remove_invalid_baseline_report_and_timeout_wait() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local baseline_root="${TEST_TMPDIR}/baseline_root_timeout"
    mkdir -p "${baseline_root}"
    local baseline_report="${baseline_root}/baseline_report.json"
    echo "{}" > "${baseline_report}"

    _resolve_invarlock_adapter() { echo "hf_test"; }
    _validate_evaluate_baseline_report() { return 1; }

    mkdir -p "${baseline_root}/.baseline_lock"
    _sleep() { return 0; }

    local log_file="${TEST_TMPDIR}/baseline_timeout.log"
    : > "${log_file}"

    local rc=0
    ( _ensure_evaluate_baseline_report "${baseline_root}" "/abs/base" "ci" "balanced" 128 128 1 1 1 10 "7" "${log_file}" ) || rc=$?
    assert_ne "0" "${rc}" "timeout wait returns non-zero"
    [[ ! -f "${baseline_report}" ]] || t_fail "invalid baseline report should be removed"
}

test_task_baseline_report_helpers_remove_invalid_baseline_report_after_lock_acquired() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    local baseline_root="${TEST_TMPDIR}/baseline_root_lock_rm"
    mkdir -p "${baseline_root}"
    local baseline_report="${baseline_root}/baseline_report.json"
    rm -f "${baseline_report}"

    _resolve_invarlock_adapter() { echo "hf_test"; }
    _validate_evaluate_baseline_report() { return 1; }

    local lock_dir="${baseline_root}/.baseline_lock"
    mkdir() {
        if [[ "${1:-}" == "${lock_dir}" ]]; then
            command mkdir "$@"
            echo "{}" > "${baseline_report}"
            return 0
        fi
        command mkdir "$@"
    }

    local log_file="${TEST_TMPDIR}/baseline_lock_rm.log"
    : > "${log_file}"

    local rc=0
    ( _ensure_evaluate_baseline_report "${baseline_root}" "/abs/base" "ci" "balanced" 128 128 1 1 1 10 "7" "${log_file}" ) || rc=$?
    assert_ne "0" "${rc}" "invalid baseline report triggers error path"
    unset -f mkdir
}
