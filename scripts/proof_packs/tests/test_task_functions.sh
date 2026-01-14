#!/usr/bin/env bash

stub_resolve_edit_params() {
    resolve_edit_params() {
        local model_output_dir="$1"
        local edit_spec="$2"
        local version="${3:-}"

        local edit_type param1 param2 scope
        IFS=':' read -r edit_type param1 param2 scope <<< "${edit_spec}"
        if [[ -z "${scope}" && "${edit_type}" != "quant_rtn" ]]; then
            scope="${param2}"
            param2=""
        fi
        if [[ "${edit_type}" == "quant_rtn" && -z "${scope}" ]]; then
            scope="${param2}"
            param2=""
        fi

        local status="selected"
        local edit_dir_name=""
        case "${edit_type}" in
            quant_rtn)
                edit_dir_name="quant_${param1}bit_${version}"
                ;;
            fp8_quant)
                edit_dir_name="fp8_${param1}_${version}"
                ;;
            fp4_quant)
                edit_dir_name="fp4_${param1}_${version}"
                ;;
            magnitude_prune)
                local pct
                pct=$(echo "${param1}" | awk '{printf "%.0f", $1 * 100}')
                edit_dir_name="prune_${pct}pct_${version}"
                ;;
            lowrank_svd)
                edit_dir_name="svd_rank${param1}_${version}"
                ;;
            *)
                status="invalid"
                ;;
        esac

        jq -n \
            --arg status "${status}" \
            --arg edit_type "${edit_type}" \
            --arg param1 "${param1}" \
            --arg param2 "${param2}" \
            --arg scope "${scope}" \
            --arg version "${version}" \
            --arg edit_dir_name "${edit_dir_name}" \
            '{status:$status, edit_type:$edit_type, param1:$param1, param2:$param2, scope:$scope, version:$version, edit_dir_name:$edit_dir_name}'
    }
}

test_model_size_and_eval_batch_selection() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    fixture_write "python3.stub" ""

    fixture_write "python3.rc" "0"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "$(dirname "${log_file}")"
    : > "${log_file}"

    # Resume branch (baseline exists) + update_model_task_memory hook.
    mkdir -p "${baseline_dir}"
    echo "{}" > "${baseline_dir}/config.json"
    update_model_task_memory() { echo "mem $*" >> "${TEST_TMPDIR}/mem.calls"; }
    task_setup_baseline "org/model" "${model_name}" 0 "${out}" "${log_file}"

    # setup_model branch success.
    rm -rf "${baseline_dir}"
    mkdir -p "${TEST_TMPDIR}/baseline_ready"
    echo "{}" > "${TEST_TMPDIR}/baseline_ready/config.json"
    setup_model() { echo "${TEST_TMPDIR}/baseline_ready"; }
    task_setup_baseline "org/model" "${model_name}" 0 "${out}" "${log_file}"

    # setup_model branch failure.
    setup_model() { return 1; }
    if task_setup_baseline "org/model" "${model_name}" 0 "${out}" "${log_file}"; then
        t_fail "expected setup_model failure to propagate"
    fi

    # Inline python branch (setup_model absent) success.
    unset -f setup_model
    PACK_NET=1
    _task_get_model_revision() { echo "rev"; }
    _cmd_python() {
        mkdir -p "${baseline_dir}"
        echo "{}" > "${baseline_dir}/config.json"
        return 0
    }
    task_setup_baseline "org/model" "${model_name}" 0 "${out}" "${log_file}"

    # Inline python branch failure (python non-zero or config missing) returns 1.
    rm -rf "${baseline_dir}"
    _cmd_python() { return 1; }
    if task_setup_baseline "org/model" "${model_name}" 0 "${out}" "${log_file}"; then
        t_fail "expected inline python failure to return non-zero"
    fi
}

test_task_eval_baseline_branches_for_missing_paths_overrides_and_results_move() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    fixture_write "python3.stub" ""
    fixture_write "python3.rc" "0"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${model_output_dir}/evals" "$(dirname "${log_file}")"
    : > "${log_file}"

    # Missing baseline path errors.
    if task_eval_baseline "${model_name}" 0 "${out}" "${log_file}"; then
        t_fail "expected baseline missing error"
    fi

    # Baseline present + existing results skip.
    mkdir -p "${model_output_dir}/models/baseline"
    echo "${model_output_dir}/models/baseline" > "${model_output_dir}/.baseline_path"
    echo "meta-llama/Llama-2-40b-hf" > "${model_output_dir}/.model_id"
    echo "{}" > "${model_output_dir}/evals/baseline_results.json"
    task_eval_baseline "${model_name}" 0 "${out}" "${log_file}"

    rm -f "${model_output_dir}/evals/baseline_results.json"

    # Fallback to model_id + OOM batch override + results found.
    _estimate_model_size() { echo "7"; }
    export TASK_ID="eval1"
    export TASK_PARAMS='{"batch_size":"auto:2"}'
    local tmp_eval_dir="${model_output_dir}/evals/.tmp/${TASK_ID}"
    mkdir -p "${tmp_eval_dir}"
    echo "{}" > "${tmp_eval_dir}/results.json"
    task_eval_baseline "${model_name}" 0 "${out}" "${log_file}"
    assert_file_exists "${model_output_dir}/evals/baseline_results.json" "results moved"

    # Error branch when no results exist.
    rm -f "${model_output_dir}/evals/baseline_results.json"
    rm -rf "${tmp_eval_dir}"
    mkdir -p "${tmp_eval_dir}"
    run task_eval_baseline "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "missing results returns non-zero"
}

test_task_eval_baseline_returns_nonzero_when_results_move_fails() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    fixture_write "python3.stub" ""
    fixture_write "python3.rc" "0"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${model_output_dir}/evals" "$(dirname "${log_file}")" "${model_output_dir}/models/baseline"
    : > "${log_file}"

    echo "${model_output_dir}/models/baseline" > "${model_output_dir}/.baseline_path"
    echo "meta-llama/Llama-2-40b-hf" > "${model_output_dir}/.model_id"

    export TASK_ID="eval_mv_fail"
    local tmp_eval_dir="${model_output_dir}/evals/.tmp/${TASK_ID}"
    mkdir -p "${tmp_eval_dir}"
    echo "{}" > "${tmp_eval_dir}/results.json"

    mv() {
        if [[ "${2:-}" == "${model_output_dir}/evals/baseline_results.json" ]]; then
            return 1
        fi
        command mv "$@"
    }

    run task_eval_baseline "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "move failure triggers non-zero"
    unset -f mv
}

test_task_calibration_run_and_generate_preset_cover_overrides_large_model_and_report_branches() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    fixture_write "python3.stub" ""
    fixture_write "python3.rc" "0"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "meta-llama/Llama-2-40b-hf" > "${model_output_dir}/.model_id"
    : > "${log_file}"

    # Baseline missing error.
    if task_calibration_run "${model_name}" 0 1 42 "${TEST_TMPDIR}/nope" "${log_file}"; then
        t_fail "expected calibration to fail with missing baseline"
    fi

    # Already done skip.
    local run_dir="${model_output_dir}/certificates/calibration/run_1"
    mkdir -p "${run_dir}"
    echo "{}" > "${run_dir}/baseline_report.json"
    task_calibration_run "${model_name}" 0 1 42 "${out}" "${log_file}"
    rm -rf "${run_dir}"

    # Override parsing + stride clamp + large model bootstrap/env + report handling.
    export INVARLOCK_BOOTSTRAP_N="1234"
    export TASK_ID="cal1"
    export TASK_PARAMS='{"seq_len":100,"stride":200,"batch_size":16}'
    _estimate_model_size() { echo "7"; }

    mkdir -p "${run_dir}"
    echo "{}" > "${run_dir}/report.json"
    task_calibration_run "${model_name}" 0 1 42 "${out}" "${log_file}"

    # Preset skip and fallback to model_id when model_size is 7.
    local preset_dir="${out}/presets"
    mkdir -p "${preset_dir}"
    local preset_file="${preset_dir}/calibrated_preset_${model_name}.yaml"
    echo "{}" > "${preset_file}"
    task_generate_preset "${model_name}" "${out}" "${log_file}"

    rm -f "${preset_file}"
    task_generate_preset "${model_name}" "${out}" "${log_file}"
}

test_task_create_edit_and_batch_edits_cover_success_failure_and_missing_function_branches() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"
    stub_resolve_edit_params

    fixture_write "python3.stub" ""
    fixture_write "python3.rc" "0"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    : > "${log_file}"

    # Missing baseline path errors.
    if task_create_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean "${TEST_TMPDIR}/nope" "${log_file}"; then
        t_fail "expected create_edit to fail without baseline"
    fi

    # Create function stubs that materialize config.json for verification.
    create_edited_model() { mkdir -p "$2"; echo "{}" > "$2/config.json"; }
    create_fp8_model() { mkdir -p "$2"; echo "{}" > "$2/config.json"; }
    create_pruned_model() { mkdir -p "$2"; echo "{}" > "$2/config.json"; }
    create_lowrank_model() { mkdir -p "$2"; echo "{}" > "$2/config.json"; }

    task_create_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean "${out}" "${log_file}"
    task_create_edit "${model_name}" 0 "fp8_quant:e4m3fn:ffn" clean "${out}" "${log_file}"
    task_create_edit "${model_name}" 0 "magnitude_prune:0.1:ffn" clean "${out}" "${log_file}"
    task_create_edit "${model_name}" 0 "lowrank_svd:8:attn" clean "${out}" "${log_file}"

    # Existing edit skips.
    task_create_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean "${out}" "${log_file}"

    # Missing create_* function branches.
    rm -rf "${model_output_dir}/models/fp8_e4m3fn_clean"
    unset -f create_fp8_model
    if task_create_edit "${model_name}" 0 "fp8_quant:e4m3fn:ffn" clean "${out}" "${log_file}"; then
        t_fail "expected missing create_fp8_model to fail"
    fi

    # Verify-failure branch when creation does not produce config.json.
    rm -rf "${model_output_dir}/models/fp8_e4m3fn_clean"
    create_fp8_model() { mkdir -p "$2"; }
    if task_create_edit "${model_name}" 0 "fp8_quant:e4m3fn:ffn" clean "${out}" "${log_file}"; then
        t_fail "expected create_edit verification failure"
    fi

    # Unknown edit type.
    if task_create_edit "${model_name}" 0 "unknown:1:2" clean "${out}" "${log_file}"; then
        t_fail "expected unknown edit type to fail"
    fi

    # Missing create_edited_model / create_pruned_model / create_lowrank_model branches.
    rm -rf "${model_output_dir}/models/quant_4bit_clean"
    unset -f create_edited_model
    if task_create_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean "${out}" "${log_file}"; then
        t_fail "expected missing create_edited_model to fail"
    fi
    rm -rf "${model_output_dir}/models/prune_10pct_clean"
    unset -f create_pruned_model
    if task_create_edit "${model_name}" 0 "magnitude_prune:0.1:ffn" clean "${out}" "${log_file}"; then
        t_fail "expected missing create_pruned_model to fail"
    fi
    rm -rf "${model_output_dir}/models/svd_rank8_clean"
    unset -f create_lowrank_model
    if task_create_edit "${model_name}" 0 "lowrank_svd:8:attn" clean "${out}" "${log_file}"; then
        t_fail "expected missing create_lowrank_model to fail"
    fi

    # Batch edits: baseline missing + python exit_code branches.
    if task_create_edits_batch "${model_name}" 0 "[]" "${TEST_TMPDIR}/nope" "${log_file}"; then
        t_fail "expected batch edits to fail without baseline"
    fi
    fixture_write "python3.rc" "0"
    task_create_edits_batch "${model_name}" 0 "[]" "${out}" "${log_file}"
    fixture_write "python3.rc" "1"
    if task_create_edits_batch "${model_name}" 0 "[]" "${out}" "${log_file}"; then
        t_fail "expected batch edits to propagate python failure"
    fi
}

test_task_eval_edit_and_single_benchmark_cover_mapping_overrides_results_and_warnings() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"
    stub_resolve_edit_params

    fixture_write "python3.stub" ""
    fixture_write "python3.rc" "0"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "$(dirname "${log_file}")" "${model_output_dir}/models"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "meta-llama/Llama-2-40b-hf" > "${model_output_dir}/.model_id"
    : > "${log_file}"

    # Missing edit model errors.
    if task_eval_edit "${model_name}" 0 "quant_rtn:4:32:attn" "${out}" "${log_file}"; then
        t_fail "expected eval_edit to fail when edit path is missing"
    fi

    # Create edit dirs for each mapping arm.
    mkdir -p "${model_output_dir}/models/quant_4bit_clean"
    mkdir -p "${model_output_dir}/models/fp8_e4m3fn_clean"
    mkdir -p "${model_output_dir}/models/prune_10pct_clean"
    mkdir -p "${model_output_dir}/models/svd_rank8_clean"

    export TASK_ID="eval_edit"
    export TASK_PARAMS='{"batch_size":"auto:2"}'
    _estimate_model_size() { echo "7"; }
    local tmp_eval_dir="${model_output_dir}/evals/.tmp/${TASK_ID}"
    mkdir -p "${tmp_eval_dir}"
    echo "{}" > "${tmp_eval_dir}/results.json"
    task_eval_edit "${model_name}" 0 "quant_rtn:4:32:attn" "${out}" "${log_file}"

    # Error path when results are missing.
    rm -f "${model_output_dir}/evals/"*.json 2>/dev/null || true
    rm -rf "${tmp_eval_dir}"
    mkdir -p "${tmp_eval_dir}"
    run task_eval_edit "${model_name}" 0 "fp8_quant:e4m3fn:ffn" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "missing results returns non-zero"

    # Edit type mapping arms in task_eval_edit.
    mkdir -p "${tmp_eval_dir}"
    echo "{}" > "${tmp_eval_dir}/results.json"
    task_eval_edit "${model_name}" 0 "magnitude_prune:0.1:ffn" "${out}" "${log_file}"
    mkdir -p "${tmp_eval_dir}"
    echo "{}" > "${tmp_eval_dir}/results.json"
    task_eval_edit "${model_name}" 0 "lowrank_svd:8:attn" "${out}" "${log_file}"

    # task_eval_edit skip branch when results exist.
    local edit_name="quant_4bit_clean"
    mkdir -p "${model_output_dir}/evals"
    echo "{}" > "${model_output_dir}/evals/${edit_name}_results.json"
    task_eval_edit "${model_name}" 0 "quant_rtn:4:32:attn" "${out}" "${log_file}"

    # Single benchmark mapping arms + default branch.
    export TASK_ID="eval_one"
    tmp_eval_dir="${model_output_dir}/evals/.tmp/${TASK_ID}"
    mkdir -p "${tmp_eval_dir}"
    echo "{}" > "${tmp_eval_dir}/results.json"
    task_eval_single_benchmark "${model_name}" 0 "quant_rtn:4:32:attn" mmlu "${out}" "${log_file}"
    mkdir -p "${tmp_eval_dir}"
    echo "{}" > "${tmp_eval_dir}/results.json"
    task_eval_single_benchmark "${model_name}" 0 "quant_rtn:4:32:attn" hellaswag "${out}" "${log_file}"
    mkdir -p "${tmp_eval_dir}"
    echo "{}" > "${tmp_eval_dir}/results.json"
    task_eval_single_benchmark "${model_name}" 0 "quant_rtn:4:32:attn" arc "${out}" "${log_file}"
    mkdir -p "${tmp_eval_dir}"
    echo "{}" > "${tmp_eval_dir}/results.json"
    task_eval_single_benchmark "${model_name}" 0 "quant_rtn:4:32:attn" winogrande "${out}" "${log_file}"
    mkdir -p "${tmp_eval_dir}"
    echo "{}" > "${tmp_eval_dir}/results.json"
    task_eval_single_benchmark "${model_name}" 0 "quant_rtn:4:32:attn" custom "${out}" "${log_file}"

    # task_eval_single_benchmark edit mapping arms.
    mkdir -p "${tmp_eval_dir}"
    echo "{}" > "${tmp_eval_dir}/results.json"
    task_eval_single_benchmark "${model_name}" 0 "fp8_quant:e4m3fn:ffn" mmlu "${out}" "${log_file}"
    mkdir -p "${tmp_eval_dir}"
    echo "{}" > "${tmp_eval_dir}/results.json"
    task_eval_single_benchmark "${model_name}" 0 "magnitude_prune:0.1:ffn" mmlu "${out}" "${log_file}"
    mkdir -p "${tmp_eval_dir}"
    echo "{}" > "${tmp_eval_dir}/results.json"
    task_eval_single_benchmark "${model_name}" 0 "lowrank_svd:8:attn" mmlu "${out}" "${log_file}"

    # Error branch when edit path is missing.
    if task_eval_single_benchmark "${model_name}" 0 "quant_rtn:99:32:attn" mmlu "${out}" "${log_file}"; then
        t_fail "expected eval_single_benchmark to fail when edit path is missing"
    fi

    # Skip branch when results already exist.
    echo "{}" > "${model_output_dir}/evals/${edit_name}_mmlu_results.json"
    task_eval_single_benchmark "${model_name}" 0 "quant_rtn:4:32:attn" mmlu "${out}" "${log_file}"

    # Error branch when no results are found.
    rm -f "${model_output_dir}/evals/${edit_name}_arc_results.json"
    mkdir -p "${tmp_eval_dir}"
    run task_eval_single_benchmark "${model_name}" 0 "quant_rtn:4:32:attn" arc "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "missing results returns non-zero"
}

test_task_eval_edit_returns_nonzero_when_results_move_fails() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"
    stub_resolve_edit_params

    fixture_write "python3.stub" ""
    fixture_write "python3.rc" "0"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "$(dirname "${log_file}")" "${model_output_dir}/models"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "meta-llama/Llama-2-40b-hf" > "${model_output_dir}/.model_id"
    : > "${log_file}"

    mkdir -p "${model_output_dir}/models/quant_4bit_clean"

    export TASK_ID="eval_edit_mv_fail"
    local tmp_eval_dir="${model_output_dir}/evals/.tmp/${TASK_ID}"
    mkdir -p "${tmp_eval_dir}"
    echo "{}" > "${tmp_eval_dir}/results.json"

    mv() {
        if [[ "${2:-}" == "${model_output_dir}/evals/quant_4bit_clean_results.json" ]]; then
            return 1
        fi
        command mv "$@"
    }

    run task_eval_edit "${model_name}" 0 "quant_rtn:4:32:attn" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "move failure triggers non-zero"
    unset -f mv
}

test_task_eval_single_benchmark_returns_nonzero_when_results_move_fails() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"
    stub_resolve_edit_params

    fixture_write "python3.stub" ""
    fixture_write "python3.rc" "0"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "$(dirname "${log_file}")" "${model_output_dir}/models"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "meta-llama/Llama-2-40b-hf" > "${model_output_dir}/.model_id"
    : > "${log_file}"

    mkdir -p "${model_output_dir}/models/quant_4bit_clean"

    export TASK_ID="eval_one_mv_fail"
    local tmp_eval_dir="${model_output_dir}/evals/.tmp/${TASK_ID}"
    mkdir -p "${tmp_eval_dir}"
    echo "{}" > "${tmp_eval_dir}/results.json"

    mv() {
        if [[ "${2:-}" == "${model_output_dir}/evals/quant_4bit_clean_mmlu_results.json" ]]; then
            return 1
        fi
        command mv "$@"
    }

    run task_eval_single_benchmark "${model_name}" 0 "quant_rtn:4:32:attn" mmlu "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "move failure triggers non-zero"
    unset -f mv
}

test_task_certify_edit_and_error_cover_preset_discovery_overrides_and_certificate_copy_paths() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"
    stub_resolve_edit_params

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "$(dirname "${log_file}")" "${model_output_dir}/models"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "meta-llama/Llama-2-40b-hf" > "${model_output_dir}/.model_id"
    : > "${log_file}"

    # Baseline missing error.
    if task_certify_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean 1 "${TEST_TMPDIR}/nope" "${log_file}"; then
        t_fail "expected certify_edit to fail without baseline"
    fi

    # Case arms for edit dir name mapping + missing edit path error.
    if task_certify_edit "${model_name}" 0 "fp8_quant:e4m3fn:ffn" clean 1 "${out}" "${log_file}"; then :; fi
    if task_certify_edit "${model_name}" 0 "magnitude_prune:0.1:ffn" clean 1 "${out}" "${log_file}"; then :; fi
    if task_certify_edit "${model_name}" 0 "lowrank_svd:8:attn" clean 1 "${out}" "${log_file}"; then :; fi

    # Full certify flow for quant_rtn with overrides and certificate copy.
    mkdir -p "${model_output_dir}/models/quant_4bit_clean"
    local cert_dir="${model_output_dir}/certificates/quant_4bit_clean/run_1"
    mkdir -p "${cert_dir}/nested"
    echo "{}" > "${cert_dir}/nested/evaluation.cert.json"

    export TASK_PARAMS='{"seq_len":100,"stride":200}'
    export INVARLOCK_BOOTSTRAP_N="1234"
    _estimate_model_size() { echo "7"; }

    task_certify_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean 1 "${out}" "${log_file}"
    # Skip branch when cert already exists.
    task_certify_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean 1 "${out}" "${log_file}"

    # Preset discovery branch when preset exists.
    mkdir -p "${out}/presets"
    echo "{}" > "${out}/presets/calibrated_preset_${model_name}.yaml"
    task_certify_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean 2 "${out}" "${log_file}"

    # Error model certify mirrors certify_edit branches.
    local error_path="${model_output_dir}/models/error_cuda_assert"
    mkdir -p "${error_path}"
    echo "{}" > "${error_path}/config.json"
    cert_dir="${model_output_dir}/certificates/errors/cuda_assert"
    mkdir -p "${cert_dir}/nested"
    echo "{}" > "${cert_dir}/nested/evaluation.cert.json"
    task_certify_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"
}

test_task_certify_edit_exits_when_workdir_cd_fails() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"
    stub_resolve_edit_params

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local edit_dir="${model_output_dir}/models/quant_4bit_clean"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "${edit_dir}" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "{}" > "${edit_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    : > "${log_file}"

    _estimate_model_size() { echo "7"; }
    cd() {
        if [[ $# -gt 0 && "${1}" == *"/.workdir" ]]; then
            return 1
        fi
        builtin cd "$@"
    }

    run task_certify_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean 1 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "cd failure exits subshell and propagates non-zero"
}

test_task_certify_error_exits_when_workdir_cd_fails() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local error_dir="${model_output_dir}/models/error_cuda_assert"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "${error_dir}" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "{}" > "${error_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    : > "${log_file}"

    _estimate_model_size() { echo "7"; }
    cd() {
        if [[ $# -gt 0 && "${1}" == *"/.workdir" ]]; then
            return 1
        fi
        builtin cd "$@"
    }

    run task_certify_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "cd failure exits subshell and propagates non-zero"
}

test_task_certify_error_missing_baseline_missing_error_model_skip_and_preset_missing_branches() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "$(dirname "${log_file}")"
    : > "${log_file}"

    # Baseline missing.
    if task_certify_error "${model_name}" 0 cuda_assert "${TEST_TMPDIR}/nope" "${log_file}"; then
        t_fail "expected certify_error to fail without baseline"
    fi

    # Baseline present, error model missing.
    mkdir -p "${baseline_dir}"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    if task_certify_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"; then
        t_fail "expected certify_error to fail without error model"
    fi

    # Error model present, cert exists skip.
    local error_path="${model_output_dir}/models/error_cuda_assert"
    mkdir -p "${error_path}"
    echo "{}" > "${error_path}/config.json"
    local cert_dir="${model_output_dir}/certificates/errors/cuda_assert"
    mkdir -p "${cert_dir}"
    echo "{}" > "${cert_dir}/evaluation.cert.json"
    task_certify_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"

    # Preset missing branch creates a minimal preset.
    rm -f "${cert_dir}/evaluation.cert.json"
    rm -rf "${out}/presets"
    task_certify_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"
}

test_task_create_error_branches_cover_skip_missing_function_and_verify_paths() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "$(dirname "${log_file}")" "${model_output_dir}/models"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    : > "${log_file}"

    # Baseline missing error.
    if task_create_error "${model_name}" 0 cuda_assert "${TEST_TMPDIR}/nope" "${log_file}"; then
        t_fail "expected create_error to fail without baseline"
    fi

    # Missing create_error_model implementation.
    if task_create_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"; then
        t_fail "expected create_error to fail when create_error_model is missing"
    fi

    create_error_model() { mkdir -p "$2"; echo "{}" > "$2/config.json"; }
    task_create_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"
    task_create_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"

    # Verify failure branch.
    rm -rf "${model_output_dir}/models/error_cuda_assert"
    create_error_model() { mkdir -p "$2"; }
    if task_create_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"; then
        t_fail "expected create_error verification failure"
    fi
}
