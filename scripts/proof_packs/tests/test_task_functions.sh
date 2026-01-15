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
    local override_preset="${cert_dir}/oom_override_preset.yaml"
    assert_file_exists "${override_preset}" "override preset created"
    local override_contents
    override_contents="$(cat "${override_preset}")"
    assert_match "seq_len: 100" "${override_contents}" "override preset seq_len"
    assert_match "stride: 100" "${override_contents}" "override preset stride uses pairing"
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

test_task_helpers_cover_fallback_branches() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    assert_eq "moe" "$(_get_model_size_from_name "Mixtral-8x7B")" "moe detection"
    assert_eq "13" "$(_get_model_size_from_name "llama-13b")" "13B detection"

    assert_eq "1536:1536:192:192:64" "$(_get_model_invarlock_config_fallback "13")" "13B config"
    assert_eq "1024:1024:192:192:48" "$(_get_model_invarlock_config_fallback "30")" "30B config"
    assert_eq "1024:1024:192:192:24" "$(_get_model_invarlock_config_fallback "moe")" "moe config"
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

    CUDA_VISIBLE_DEVICES="0,1"
    LM_EVAL_PARALLELIZE="true"
    local args
    args="$(_get_lmeval_model_args "/tmp/model")"
    assert_match "parallelize=True" "${args}" "parallelize enabled"
    unset CUDA_VISIBLE_DEVICES LM_EVAL_PARALLELIZE

    _is_large_model "moe" || t_fail "expected moe to be large"
    _is_large_model "llama-30b" || t_fail "expected 30b string to be large"

    assert_eq "${EVAL_BATCH_SIZE_MOE:-auto:6}" "$(_get_eval_batch_size "moe")" "moe batch"
    assert_eq "${EVAL_BATCH_SIZE_LARGE:-auto:4}" "$(_get_eval_batch_size "70")" "large batch"
    assert_eq "${EVAL_BATCH_SIZE_MEDIUM:-auto:8}" "$(_get_eval_batch_size "30")" "medium batch"
    assert_eq "${EVAL_BATCH_SIZE_SMALL:-auto:16}" "$(_get_eval_batch_size "7")" "small batch"
    assert_eq "${EVAL_BATCH_SIZE_SMALL:-auto:16}" "$(_get_eval_batch_size "foo")" "default batch"
}

test_task_timeout_and_profile_helpers() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

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
}

test_execute_task_dispatches_all_task_types() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}"
    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}/running"

    TASK_TIMEOUT_DEFAULT=""
    set +m

    task_setup_baseline() { :; }
    task_eval_baseline() { :; }
    task_calibrate_clean_edits() { :; }
    task_calibration_run() { :; }
    task_create_edit() { :; }
    task_create_edits_batch() { :; }
    task_eval_edit() { :; }
    task_eval_single_benchmark() { :; }
    task_certify_edit() { :; }
    task_create_error() { :; }
    task_certify_error() { :; }
    task_generate_preset() { :; }

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

    local types=(SETUP_BASELINE EVAL_BASELINE CALIBRATE_CLEAN CALIBRATION_RUN CREATE_EDIT CREATE_EDITS_BATCH EVAL_EDIT EVAL_MMLU EVAL_HELLASWAG EVAL_ARC EVAL_WINOGRANDE CERTIFY_EDIT CREATE_ERROR CERTIFY_ERROR GENERATE_PRESET)
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
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}"
    export QUEUE_DIR="${TEST_TMPDIR}/queue"
    mkdir -p "${QUEUE_DIR}/running"

    task_eval_baseline() { :; }

    jq -n '{task_id:"t1", task_type:"EVAL_BASELINE", model_id:"m", model_name:"model", status:"pending", assigned_gpus:null, params:{}}' \
        > "${TEST_TMPDIR}/t1.task"

    set -m
    run execute_task "${TEST_TMPDIR}/t1.task" 0 "${out}"
    assert_rc "0" "${RUN_RC}" "execute_task succeeds with job control enabled"
    set +m
}

test_execute_task_timeout_triggers_marker() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

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
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

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
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

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

test_task_eval_edit_clean_resolution_errors() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

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
        jq -n '{status:"skipped", edit_type:"quant_rtn", param1:"clean", param2:"", scope:"ffn", edit_dir_name:"quant_8bit_clean"}'
    }
    task_eval_edit "${model_name}" 0 "quant_rtn:clean:ffn" "${out}" "${log_file}"

    resolve_edit_params() {
        jq -n '{status:"invalid", edit_type:"quant_rtn", param1:"clean", param2:"", scope:"ffn", edit_dir_name:""}'
    }
    run task_eval_edit "${model_name}" 0 "quant_rtn:clean:ffn" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "invalid clean resolution errors"

    _cmd_python() { return 0; }
    resolve_edit_params() {
        jq -n '{status:"selected", edit_type:"quant_rtn", param1:"clean", param2:"", scope:"ffn", edit_dir_name:"quant_8bit_clean"}'
    }
    TASK_ID="eval_clean"
    local tmp_eval_dir="${model_output_dir}/evals/.tmp/${TASK_ID}"
    mkdir -p "${tmp_eval_dir}" "${model_output_dir}/models/quant_8bit_clean"
    echo "{}" > "${tmp_eval_dir}/results.json"
    task_eval_edit "${model_name}" 0 "quant_rtn:clean:ffn" "${out}" "${log_file}"

    resolve_edit_params() {
        local version="$3"
        if [[ "${version}" == "clean" ]]; then
            jq -n '{status:"invalid", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:""}'
        else
            jq -n '{status:"selected", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:"quant_4bit_stress"}'
        fi
    }
    run task_eval_edit "${model_name}" 0 "quant_rtn:4:32:ffn" "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "non-selected resolution continues"
}

test_task_eval_single_benchmark_clean_resolution_errors() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

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
        jq -n '{status:"skipped", edit_type:"quant_rtn", param1:"clean", param2:"", scope:"ffn", edit_dir_name:"quant_8bit_clean"}'
    }
    task_eval_single_benchmark "${model_name}" 0 "quant_rtn:clean:ffn" mmlu "${out}" "${log_file}"

    resolve_edit_params() {
        jq -n '{status:"invalid", edit_type:"quant_rtn", param1:"clean", param2:"", scope:"ffn", edit_dir_name:""}'
    }
    run task_eval_single_benchmark "${model_name}" 0 "quant_rtn:clean:ffn" mmlu "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "invalid clean resolution errors"

    _cmd_python() { return 0; }
    resolve_edit_params() {
        jq -n '{status:"selected", edit_type:"quant_rtn", param1:"clean", param2:"", scope:"ffn", edit_dir_name:"quant_8bit_clean"}'
    }
    TASK_ID="eval_clean_one"
    local tmp_eval_dir="${model_output_dir}/evals/.tmp/${TASK_ID}"
    mkdir -p "${tmp_eval_dir}" "${model_output_dir}/models/quant_8bit_clean"
    echo "{}" > "${tmp_eval_dir}/results.json"
    task_eval_single_benchmark "${model_name}" 0 "quant_rtn:clean:ffn" mmlu "${out}" "${log_file}"

    resolve_edit_params() {
        local version="$3"
        if [[ "${version}" == "clean" ]]; then
            jq -n '{status:"invalid", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:""}'
        else
            jq -n '{status:"selected", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:"quant_4bit_stress"}'
        fi
    }
    run task_eval_single_benchmark "${model_name}" 0 "quant_rtn:4:32:ffn" mmlu "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "non-selected resolution continues"
}

test_task_certify_edit_skip_and_invalid() {
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

    resolve_edit_params() {
        jq -n '{status:"skipped", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:"quant_4bit_clean"}'
    }
    task_certify_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean 1 "${out}" "${log_file}"

    resolve_edit_params() {
        jq -n '{status:"invalid", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:"quant_4bit_clean"}'
    }
    run task_certify_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean 1 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "invalid resolution errors"
}

test_task_calibrate_clean_edits_early_exits() {
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

    CALIBRATE_CLEAN_EDITS="false"
    task_calibrate_clean_edits "${model_name}" 0 "${out}" "${log_file}"

    CALIBRATE_CLEAN_EDITS="true"
    mkdir -p "${model_output_dir}/state"
    echo "{}" > "${model_output_dir}/state/clean_edit_params.json"
    task_calibrate_clean_edits "${model_name}" 0 "${out}" "${log_file}"

    rm -rf "${model_output_dir}"
    mkdir -p "${model_output_dir}/evals"
    if task_calibrate_clean_edits "${model_name}" 0 "${out}" "${log_file}"; then
        t_fail "expected missing baseline to fail"
    fi

    mkdir -p "${baseline_dir}"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    if task_calibrate_clean_edits "${model_name}" 0 "${out}" "${log_file}"; then
        t_fail "expected missing baseline eval to fail"
    fi

    mkdir -p "${model_output_dir}/evals"
    echo "{}" > "${model_output_dir}/evals/baseline_results.json"
    mkdir -p "${model_output_dir}/state/clean_edit_cal.lock"
    echo "{}" > "${model_output_dir}/state/clean_edit_params.json"
    task_calibrate_clean_edits "${model_name}" 0 "${out}" "${log_file}"
}


test_task_calibrate_clean_edits_waits_on_lock_and_detects_params() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "${model_output_dir}/evals" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "{}" > "${model_output_dir}/evals/baseline_results.json"
    : > "${log_file}"

    local lock_dir="${model_output_dir}/state/clean_edit_cal.lock"
    mkdir -p "${lock_dir}"

    local slept=0
    _sleep() {
        slept=$((slept + 1))
        if [[ ${slept} -eq 1 ]]; then
            mkdir -p "${model_output_dir}/state"
            echo "{}" > "${model_output_dir}/state/clean_edit_params.json"
        fi
    }

    task_calibrate_clean_edits "${model_name}" 0 "${out}" "${log_file}"
    assert_file_exists "${model_output_dir}/state/clean_edit_params.json" "params detected after lock wait"
}


test_task_calibrate_clean_edits_lock_timeout_errors() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "${model_output_dir}/evals" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "{}" > "${model_output_dir}/evals/baseline_results.json"
    : > "${log_file}"

    mkdir -p "${model_output_dir}/state/clean_edit_cal.lock"
    _sleep() { :; }

    run task_calibrate_clean_edits "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "lock timeout returns non-zero"
}


test_task_calibrate_clean_edits_baseline_lmeval_missing_results() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "${model_output_dir}/evals" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "id" > "${model_output_dir}/.model_id"
    echo "{}" > "${model_output_dir}/evals/baseline_results.json"
    : > "${log_file}"

    _cmd_python() {
        if [[ "$*" == *"-m"*"lm_eval"* ]]; then
            local out_dir=""
            local args=("$@");
            local i=0
            while [[ ${i} -lt ${#args[@]} ]]; do
                if [[ "${args[$i]}" == "--output_path" ]]; then
                    out_dir="${args[$((i + 1))]}"
                    break
                fi
                i=$((i + 1))
            done
            mkdir -p "${out_dir}"
            return 0
        fi
        return 0
    }

    run task_calibrate_clean_edits "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "baseline lm-eval missing results returns non-zero"
}

test_task_calibrate_clean_edits_baseline_failure() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    fixture_write "python3.stub" ""
    fixture_write "python3.rc" "1"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "${model_output_dir}/evals" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "id" > "${model_output_dir}/.model_id"
    : > "${log_file}"

    run task_calibrate_clean_edits "${model_name}" 0 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "baseline calibration failure returns non-zero"
}

test_task_calibrate_clean_edits_candidate_selection_branches() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "${model_output_dir}/evals" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "meta-llama/Llama-2-13b-hf" > "${model_output_dir}/.model_id"
    echo "{}" > "${model_output_dir}/evals/baseline_results.json"
    : > "${log_file}"

    CLEAN_QUANT_BITS=8
    CLEAN_QUANT_GROUP_SIZES="128,64"
    CLEAN_FP8_FORMATS="e4m3fn"
    CLEAN_PRUNE_LEVELS="0.1"
    CLEAN_SVD_RANK_RATIOS="0.001"
    CLEAN_EVAL_LIMIT=200
    TASK_ID="calib"

    local baseline_path="${baseline_dir}"
    _cmd_python() {
        if [[ "$*" == *"-m"*"lm_eval"* ]]; then
            local out_dir=""
            local args=("$@");
            local i=0
            while [[ ${i} -lt ${#args[@]} ]]; do
                if [[ "${args[$i]}" == "--output_path" ]]; then
                    out_dir="${args[$((i + 1))]}"
                    break
                fi
                i=$((i + 1))
            done
            mkdir -p "${out_dir}"
            echo '{"results": {"mmlu": {"acc": 0.5}}}' > "${out_dir}/results.json"
            return 0
        fi
        if [[ "${1:-}" == "-" && "${2:-}" == "${baseline_path}" ]]; then
            echo "4"
            return 0
        fi
        if [[ "${1:-}" == "-" && "${2:-}" == "${model_output_dir}/state/clean_edit_params.jsonl" ]]; then
            local output_path="${3:-}"
            if [[ -n "${output_path}" ]]; then
                mkdir -p "$(dirname "${output_path}")"
                echo '{}' > "${output_path}"
            fi
            return 0
        fi
        return 0
    }

    local calib_tmp_dir="${model_output_dir}/evals/.clean_calib"
    mkdir -p "${calib_tmp_dir}"
    echo "{}" > "${calib_tmp_dir}/quant_8bit_clean_calib.json"

    task_calibrate_clean_edits "${model_name}" 0 "${out}" "${log_file}"
}


test_task_calibrate_clean_edits_candidate_failure_and_rejection_branches() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "${model_output_dir}/evals" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "meta-llama/Llama-2-13b-hf" > "${model_output_dir}/.model_id"
    echo "{}" > "${model_output_dir}/evals/baseline_results.json"
    : > "${log_file}"

    CLEAN_QUANT_BITS=8
    CLEAN_QUANT_GROUP_SIZES="128,64"
    CLEAN_FP8_FORMATS="e4m3fn"
    CLEAN_PRUNE_LEVELS="0.1"
    CLEAN_SVD_RANK_RATIOS="0.25"
    CLEAN_EVAL_LIMIT=200
    TASK_ID="calib"

    create_edited_model() { mkdir -p "$2"; }
    create_fp8_model() { mkdir -p "$2"; }
    create_pruned_model() { mkdir -p "$2"; }
    create_lowrank_model() { mkdir -p "$2"; }
    check_no_regression() { return 1; }

    local eval_calls=0
    _cmd_python() {
        if [[ "$*" == *"-m"*"lm_eval"* ]]; then
            local out_dir=""
            local args=("$@");
            local i=0
            while [[ ${i} -lt ${#args[@]} ]]; do
                if [[ "${args[$i]}" == "--output_path" ]]; then
                    out_dir="${args[$((i + 1))]}"
                    break
                fi
                i=$((i + 1))
            done
            mkdir -p "${out_dir}"
            if [[ "${out_dir}" == *"baseline_calib"* ]]; then
                echo '{"results": {"mmlu": {"acc": 0.5}}}' > "${out_dir}/results.json"
                return 0
            fi
            eval_calls=$((eval_calls + 1))
            if [[ ${eval_calls} -eq 1 ]]; then
                return 1
            fi
            echo '{"results": {"mmlu": {"acc": 0.5}}}' > "${out_dir}/results.json"
            return 0
        fi
        if [[ "${1:-}" == "-" && "${2:-}" == "${baseline_dir}" ]]; then
            echo "4"
            return 0
        fi
        if [[ "${1:-}" == "-" && "${2:-}" == *"baseline_calibration_results.json" ]]; then
            return 1
        fi
        if [[ "${1:-}" == "-" && "${2:-}" == "${model_output_dir}/state/clean_edit_params.jsonl" ]]; then
            echo "{}" > "${3}"
            return 0
        fi
        return 0
    }

    task_calibrate_clean_edits "${model_name}" 0 "${out}" "${log_file}"
    assert_file_exists "${model_output_dir}/state/clean_edit_params.json" "clean params written"
}


test_task_calibrate_clean_edits_selects_all_families_and_unknown_family_branch() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "${model_output_dir}/evals" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "meta-llama/Llama-2-13b-hf" > "${model_output_dir}/.model_id"
    echo "{}" > "${model_output_dir}/evals/baseline_results.json"
    : > "${log_file}"

    CLEAN_QUANT_BITS=8
    CLEAN_QUANT_GROUP_SIZES="128"
    CLEAN_FP8_FORMATS="e4m3fn"
    CLEAN_PRUNE_LEVELS="0.1"
    CLEAN_SVD_RANK_RATIOS="0.25"
    CLEAN_EVAL_LIMIT=200
    TASK_ID="calib"

    create_edited_model() { mkdir -p "$2"; }
    create_fp8_model() { mkdir -p "$2"; }
    create_pruned_model() { mkdir -p "$2"; }
    create_lowrank_model() { mkdir -p "$2"; }

    local calib_tmp_dir="${model_output_dir}/evals/.clean_calib"
    mkdir -p "${calib_tmp_dir}"
    echo '{"results": {"mmlu": {"acc": 0.5}}}' > "${calib_tmp_dir}/quant_8bit_clean_calib.json"

    _cmd_python() {
        if [[ "$*" == *"-m"*"lm_eval"* ]]; then
            local out_dir=""
            local args=("$@");
            local i=0
            while [[ ${i} -lt ${#args[@]} ]]; do
                if [[ "${args[$i]}" == "--output_path" ]]; then
                    out_dir="${args[$((i + 1))]}"
                    break
                fi
                i=$((i + 1))
            done
            mkdir -p "${out_dir}"
            echo '{"results": {"mmlu": {"acc": 0.5}}}' > "${out_dir}/results.json"
            return 0
        fi
        if [[ "${1:-}" == "-" && "${2:-}" == "${baseline_dir}" ]]; then
            echo "4"
            return 0
        fi
        if [[ "${1:-}" == "-" && "${2:-}" == *"baseline_calibration_results.json" ]]; then
            return 0
        fi
        if [[ "${1:-}" == "-" && "${2:-}" == "${model_output_dir}/state/clean_edit_params.jsonl" ]]; then
            echo "{}" > "${3}"
            return 0
        fi
        return 0
    }

    task_calibrate_clean_edits "${model_name}" 0 "${out}" "${log_file}"
    assert_file_exists "${model_output_dir}/state/clean_edit_params.json" "clean params written"

    local unknown_params="${TEST_TMPDIR}/unknown.jsonl"
    : > "${unknown_params}"
    params_jsonl="${unknown_params}"
    select_candidate "mystery" "ffn" "noop"
    assert_match "skipped" "$(cat "${unknown_params}")" "unknown family recorded"
}

test_task_calibrate_clean_edits_skips_when_no_candidate_selected() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "${model_output_dir}/evals" "$(dirname "${log_file}")"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    echo "meta-llama/Llama-2-13b-hf" > "${model_output_dir}/.model_id"
    echo "{}" > "${model_output_dir}/evals/baseline_results.json"
    : > "${log_file}"

    CLEAN_QUANT_BITS=8
    CLEAN_QUANT_GROUP_SIZES="128"
    CLEAN_FP8_FORMATS="e4m3fn"
    CLEAN_PRUNE_LEVELS="0.1"
    CLEAN_SVD_RANK_RATIOS="0.25"
    CLEAN_EVAL_LIMIT=200

    _cmd_python() {
        if [[ "$*" == *"-m"*"lm_eval"* ]]; then
            local out_dir=""
            local args=("$@");
            local i=0
            while [[ ${i} -lt ${#args[@]} ]]; do
                if [[ "${args[$i]}" == "--output_path" ]]; then
                    out_dir="${args[$((i + 1))]}"
                    break
                fi
                i=$((i + 1))
            done
            mkdir -p "${out_dir}"
            echo '{"results": {"mmlu": {"acc": 0.5}}}' > "${out_dir}/results.json"
            return 0
        fi
        if [[ "${1:-}" == "-" && "${2:-}" == "${baseline_dir}" ]]; then
            echo "4"
            return 0
        fi
        if [[ "${1:-}" == "-" && "${2:-}" == *"baseline_calibration_results.json" ]]; then
            return 1
        fi
        if [[ "${1:-}" == "-" && "${2:-}" == "${model_output_dir}/state/clean_edit_params.jsonl" ]]; then
            echo "{}" > "${3}"
            return 0
        fi
        return 0
    }

    task_calibrate_clean_edits "${model_name}" 0 "${out}" "${log_file}"
}


test_task_calibration_run_guard_order_branches() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

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

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"
    PATH="${bin_dir}:${PATH}"
    export PATH

    PACK_GUARDS_ORDER="variance"
    task_calibration_run "${model_name}" 0 "1" "42" "${out}" "${log_file}"
    assert_match "variance" "$(cat "${model_output_dir}/certificates/calibration/run_1/calibration_config.yaml")" "explicit guard order used"

    PACK_GUARDS_ORDER=" , "
    task_calibration_run "${model_name}" 0 "2" "43" "${out}" "${log_file}"
    assert_match "invariants" "$(cat "${model_output_dir}/certificates/calibration/run_2/calibration_config.yaml")" "default guard order used"
}
