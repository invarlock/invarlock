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
    export INVARLOCK_CERT_MIN_WINDOWS="256"
    export TASK_ID="cal1"
    export TASK_PARAMS='{"seq_len":100,"stride":200,"batch_size":16}'
    _estimate_model_size() { echo "7"; }

    mkdir -p "${run_dir}"
    echo "{}" > "${run_dir}/report.json"
    task_calibration_run "${model_name}" 0 1 42 "${out}" "${log_file}"
    assert_match "CI window override: preview=256, final=256" "$(cat "${log_file}")" "ci window override applied"

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

    mkdir -p "${out}/presets"
    cat > "${out}/presets/calibrated_preset_${model_name}__quant_rtn.yaml" <<'YAML'
dataset:
  provider: wikitext2
  split: validation
  seq_len: 2048
  stride: 1024
guards:
  spectral:
    max_caps: 15
YAML

    task_certify_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean 1 "${out}" "${log_file}"
    local profile_yaml="${cert_dir}/config_root/runtime/profiles/ci.yaml"
    assert_file_exists "${profile_yaml}" "profile override created"
    local profile_contents
    profile_contents="$(cat "${profile_yaml}")"
    assert_match "seq_len: 100" "${profile_contents}" "profile override seq_len"
    assert_match "stride: 100" "${profile_contents}" "profile override stride uses pairing"
    assert_match "preview_n: 192" "${profile_contents}" "profile override preview_n"
    assert_match "final_n: 192" "${profile_contents}" "profile override final_n"

    local calls
    calls="$(cat "${TEST_TMPDIR}/fixtures/invarlock.calls")"
    assert_match "calibrated_preset_${model_name}__quant_rtn\\.yaml" "${calls}" "uses edit-type preset"
    if [[ "${calls}" =~ oom_override_preset\.yaml ]]; then
        t_fail "expected certify to avoid override preset file"
    fi
    # Skip branch when cert already exists.
    task_certify_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean 1 "${out}" "${log_file}"

    # Preset discovery branch when preset exists.
    rm -f "${out}/presets/calibrated_preset_${model_name}__quant_rtn.yaml"
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
    assert_eq "70" "$(_get_model_size_from_name "Qwen1.5-72B")" "70B detection"
    assert_eq "30" "$(_get_model_size_from_name "Qwen2.5-32B")" "30B detection"

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

    _is_large_model "moe" || t_fail "expected moe to be large"
    _is_large_model "llama-30b" || t_fail "expected 30b string to be large"
}

test_task_create_model_variant_dispatch_and_fallback_errors() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    create_model_variant() { echo "main:$*"; return 0; }
    run _task_create_model_variant "/b" "/o" "quant_rtn" "8" "128" "ffn" "0"
    assert_rc "0" "${RUN_RC}" "dispatches to create_model_variant when available"
    assert_match "^main:" "${RUN_OUT}" "create_model_variant called"

    unset -f create_model_variant
    unset -f create_edited_model create_fp8_model create_pruned_model create_lowrank_model || true

    run _task_create_model_variant "/b" "/o" "quant_rtn" "8" "128" "ffn" "0"
    assert_rc "1" "${RUN_RC}" "quant_rtn without create_edited_model returns non-zero"

    run _task_create_model_variant "/b" "/o" "fp8_quant" "e4m3fn" "" "ffn" "0"
    assert_rc "1" "${RUN_RC}" "fp8_quant without create_fp8_model returns non-zero"

    run _task_create_model_variant "/b" "/o" "magnitude_prune" "0.1" "" "ffn" "0"
    assert_rc "1" "${RUN_RC}" "magnitude_prune without create_pruned_model returns non-zero"

    run _task_create_model_variant "/b" "/o" "lowrank_svd" "8" "" "ffn" "0"
    assert_rc "1" "${RUN_RC}" "lowrank_svd without create_lowrank_model returns non-zero"

    run _task_create_model_variant "/b" "/o" "nope" "" "" "" "0"
    assert_rc "1" "${RUN_RC}" "unknown edit type returns non-zero"
}

test_task_create_model_variant_fallback_success_paths() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    unset -f create_model_variant || true

    create_edited_model() { echo "edited:$*"; return 0; }
    create_fp8_model() { echo "fp8:$*"; return 0; }
    create_pruned_model() { echo "pruned:$*"; return 0; }
    create_lowrank_model() { echo "lowrank:$*"; return 0; }

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
}

test_task_baseline_report_helpers_cover_reuse_lock_race_and_wait_paths() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    run _resolve_invarlock_adapter ""
    assert_ne "0" "${RUN_RC}" "empty adapter input returns non-zero"

    run _validate_certify_baseline_report "" "hf" "ci" "balanced"
    assert_ne "0" "${RUN_RC}" "missing baseline report returns non-zero"

    local baseline_root="${TEST_TMPDIR}/baseline_root"
    mkdir -p "${baseline_root}"
    local baseline_report="${baseline_root}/baseline_report.json"

    _resolve_invarlock_adapter() { echo "hf_test"; }
    _validate_certify_baseline_report() { return 0; }

    echo "{}" > "${baseline_report}"
    local reuse
    reuse="$(_ensure_certify_baseline_report "${baseline_root}" "/abs/base" "ci" "balanced" 128 128 1 1 1 10 "7" "${TEST_TMPDIR}/log.txt")"
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
    raced="$(_ensure_certify_baseline_report "${baseline_root}" "/abs/base" "ci" "balanced" 128 128 1 1 1 10 "7" "${TEST_TMPDIR}/log.txt")"
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
    if _ensure_certify_baseline_report "${baseline_root}" "/abs/base" "ci" "balanced" 128 128 1 1 1 10 "7" "${TEST_TMPDIR}/log.txt" > "${waited_file}"; then
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
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local calls="${TEST_TMPDIR}/python.calls"
    _cmd_python() {
        echo "python $*" >> "${calls}"
        cat >/dev/null || true
        if [[ $# -eq 2 ]]; then
            echo "hf_auto"
        fi
        return 0
    }

    run _resolve_invarlock_adapter "org/model"
    assert_rc "0" "${RUN_RC}" "adapter resolver runs python wrapper"
    assert_eq "hf_auto" "${RUN_OUT}" "adapter output forwarded"

    local report="${TEST_TMPDIR}/baseline_report.json"
    echo "{}" > "${report}"

    run _validate_certify_baseline_report "${report}" "hf_auto" "ci" "balanced"
    assert_rc "0" "${RUN_RC}" "baseline report validation runs python wrapper"
    assert_file_exists "${calls}" "python stub invoked"
}

test_task_baseline_report_helpers_cover_generate_baseline_report_path() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local baseline_root="${TEST_TMPDIR}/baseline_root_generate"
    mkdir -p "${baseline_root}"
    local baseline_report="${baseline_root}/baseline_report.json"
    rm -f "${baseline_report}"

    local log_file="${TEST_TMPDIR}/baseline_generate.log"
    : > "${log_file}"

    _resolve_invarlock_adapter() { echo "hf_test"; }
    _validate_certify_baseline_report() { return 0; }

    export PACK_GUARDS_ORDER="invariants, spectral , rmt"
    fixture_write "invarlock.create_report_nested" ""

    local generated
    generated="$(_ensure_certify_baseline_report "${baseline_root}" "/abs/base" "ci" "balanced" 128 128 1 1 1 10 "7" "${log_file}")"

    assert_eq "${baseline_report}" "${generated}" "baseline report path returned"
    assert_file_exists "${baseline_report}" "baseline report generated"
    assert_match "Generating reusable baseline report" "$(cat "${log_file}")" "generation logged"
}

test_task_baseline_report_helpers_remove_invalid_baseline_report_and_timeout_wait() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local baseline_root="${TEST_TMPDIR}/baseline_root_timeout"
    mkdir -p "${baseline_root}"
    local baseline_report="${baseline_root}/baseline_report.json"
    echo "{}" > "${baseline_report}"

    _resolve_invarlock_adapter() { echo "hf_test"; }
    _validate_certify_baseline_report() { return 1; }

    mkdir -p "${baseline_root}/.baseline_lock"
    _sleep() { return 0; }

    local log_file="${TEST_TMPDIR}/baseline_timeout.log"
    : > "${log_file}"

    local rc=0
    ( _ensure_certify_baseline_report "${baseline_root}" "/abs/base" "ci" "balanced" 128 128 1 1 1 10 "7" "${log_file}" ) || rc=$?
    assert_ne "0" "${rc}" "timeout wait returns non-zero"
    [[ ! -f "${baseline_report}" ]] || t_fail "invalid baseline report should be removed"
}

test_task_baseline_report_helpers_remove_invalid_baseline_report_after_lock_acquired() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local baseline_root="${TEST_TMPDIR}/baseline_root_lock_rm"
    mkdir -p "${baseline_root}"
    local baseline_report="${baseline_root}/baseline_report.json"
    rm -f "${baseline_report}"

    _resolve_invarlock_adapter() { echo "hf_test"; }
    _validate_certify_baseline_report() { return 1; }

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
    ( _ensure_certify_baseline_report "${baseline_root}" "/abs/base" "ci" "balanced" 128 128 1 1 1 10 "7" "${log_file}" ) || rc=$?
    assert_ne "0" "${rc}" "invalid baseline report triggers error path"
    unset -f mkdir
}

test_task_certify_edit_reuses_baseline_report_applies_ci_override_and_falls_back_label() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    fixture_write "invarlock.create_cert" ""

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

    # Force CI window override by returning tiny preview/final windows.
    _get_invarlock_config() { echo "128:128:1:1:1"; }
    export INVARLOCK_CERT_MIN_WINDOWS="192"

    local baseline_report="${TEST_TMPDIR}/baseline_report.json"
    echo '{"evaluation_windows":{"preview":{"window_ids":[1],"input_ids":[[1]]},"final":{"window_ids":[1],"input_ids":[[1]]}},"edit":{"name":"noop"}}' > "${baseline_report}"
    _ensure_certify_baseline_report() { echo "${baseline_report}"; }

    resolve_edit_params() {
        jq -n '{status:"selected", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:"_clean"}'
    }
    local edit_dir="${model_output_dir}/models/_clean"
    mkdir -p "${edit_dir}"
    echo "{}" > "${edit_dir}/config.json"

    mkdir -p "${out}/presets"
    echo "{}" > "${out}/presets/calibrated_preset_${model_name}.yaml"

    task_certify_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean 1 "${out}" "${log_file}"

    assert_match "CI window override" "$(cat "${log_file}")" "CI window override applied"
    assert_match "Reusing baseline report" "$(cat "${log_file}")" "baseline report reused"

    local calls
    calls="$(cat "${TEST_TMPDIR}/fixtures/invarlock.calls")"
    assert_match "--baseline-report" "${calls}" "baseline report forwarded to invarlock certify"
    assert_match "--edit-label custom" "${calls}" "empty edit label falls back to custom"
}

test_task_certify_error_reuses_baseline_report_and_applies_ci_override() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

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
    _ensure_certify_baseline_report() { echo "${baseline_report}"; }

    mkdir -p "${out}/presets"
    echo "{}" > "${out}/presets/calibrated_preset_${model_name}.yaml"

    task_certify_error "${model_name}" 0 nan_injection "${out}" "${log_file}"

    assert_match "CI window override" "$(cat "${log_file}")" "CI window override applied"
    assert_match "Reusing baseline report" "$(cat "${log_file}")" "baseline report reused"
    assert_file_exists "${model_output_dir}/certificates/errors/nan_injection/evaluation.cert.json" "error cert written"
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
    task_calibration_run() { :; }
    task_create_edit() { :; }
    task_create_edits_batch() { :; }
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

    local types=(SETUP_BASELINE CALIBRATION_RUN CREATE_EDIT CREATE_EDITS_BATCH CERTIFY_EDIT CREATE_ERROR CERTIFY_ERROR GENERATE_PRESET)
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

test_resolve_edit_params_uses_tuned_presets() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

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

    PACK_TUNED_EDIT_PARAMS_FILE="${TEST_TMPDIR}/missing.json"
    export PACK_TUNED_EDIT_PARAMS_FILE
    resolved=$(resolve_edit_params "${model_output_dir}" "lowrank_svd:clean:ffn" "clean")
    assert_eq "missing" "$(echo "${resolved}" | jq -r '.status')" "missing tuned params file returns missing"
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
    assert_match "spectral" "$(cat "${model_output_dir}/certificates/calibration/run_2/calibration_config.yaml")" "default guard order used"
}
