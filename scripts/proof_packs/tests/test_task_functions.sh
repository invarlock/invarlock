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

test_default_ci_min_windows_accounts_for_padding() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    unset INVARLOCK_CERT_MIN_WINDOWS
    unset INVARLOCK_DATASET
    unset INVARLOCK_TIER
    assert_eq "400" "$(_default_ci_min_windows "512")" "default proof-pack dataset is WikiText-2 with the higher balanced floor"
    assert_eq "352" "$(_default_ci_min_windows "128")" "small seq_len uses higher default to meet token floors"

    INVARLOCK_DATASET="wikitext2"
    INVARLOCK_TIER="balanced"
    assert_eq "400" "$(_default_ci_min_windows "1536")" "balanced WikiText-2 uses higher window floor for long sequences"
    assert_eq "400" "$(_default_ci_min_windows "512")" "balanced WikiText-2 uses higher window floor at 512 seq_len"
    assert_eq "352" "$(_default_ci_min_windows "128")" "tiny seq_len still uses the padding-aware floor"

    INVARLOCK_DATASET="hf_text"
    assert_eq "256" "$(_default_ci_min_windows "1536")" "non-WikiText datasets keep the generic long-sequence floor"

    INVARLOCK_CERT_MIN_WINDOWS="300"
    assert_eq "300" "$(_default_ci_min_windows "128")" "env override wins"
}

test_effective_ci_schedule_selects_and_logs_viable_candidate() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local model_dir="${TEST_TMPDIR}/model"
    mkdir -p "${model_dir}"
    echo "{}" > "${model_dir}/config.json"

    local planner_calls="${TEST_TMPDIR}/planner.calls"
    _cmd_python() {
        echo "$*" > "${planner_calls}"
        cat <<'EOF'
{"status":"selected","min_tokens_target":50000,"effective_min_tokens":52500,"candidates":[{"seq_len":512,"stride":512,"requested_preview":400,"requested_final":400,"actual_preview":360,"actual_final":360,"total_tokens":53000,"effective_min_tokens":52500,"tokens_floor_met":true,"reason":"selected"}],"selected":{"seq_len":512,"stride":512,"requested_preview":400,"requested_final":400,"actual_preview":360,"actual_final":360,"total_tokens":53000,"effective_min_tokens":52500,"tokens_floor_met":true,"reason":"selected"}}
EOF
    }

    local plan_json
    plan_json="$(_plan_effective_ci_schedule "${model_dir}" "13" "balanced" "wikitext2" "validation" "42")"
    assert_match '"status":"selected"' "${plan_json}" "planner result returned"
    assert_match "--candidate 512:400:400" "$(cat "${planner_calls}")" "planner considers 512 candidate first"
    assert_match "--candidate 1536:400:400" "$(cat "${planner_calls}")" "planner keeps long-sequence candidate available"

    local log_file="${TEST_TMPDIR}/plan.log"
    : > "${log_file}"
    local selected
    selected="$(_apply_effective_ci_schedule "${plan_json}" "${log_file}")"
    assert_eq "512:512:360:360" "${selected}" "selected schedule uses effective post-dedupe counts"
    local log_text
    log_text="$(cat "${log_file}")"
    assert_match "Effective CI planning target: min_tokens=50000, with_headroom=52500" "${log_text}" "token target logged"
    assert_match "Candidate: seq=512 stride=512 requested=400\\+400 actual=360\\+360 tokens=53000 floor=52500 floor_met=true reason=selected" "${log_text}" "candidate summary logged"
    assert_match "Selected effective CI schedule: 512:512:360:360" "${log_text}" "selected schedule logged"
}

test_effective_ci_schedule_fails_fast_when_no_candidate_clears_floor() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local log_file="${TEST_TMPDIR}/plan_fail.log"
    : > "${log_file}"
    local rc=0
    if _apply_effective_ci_schedule \
        '{"status":"no_candidate","min_tokens_target":50000,"effective_min_tokens":52500,"candidates":[{"seq_len":512,"stride":512,"requested_preview":400,"requested_final":400,"actual_preview":220,"actual_final":220,"total_tokens":45723,"effective_min_tokens":52500,"tokens_floor_met":false,"reason":"below_token_floor"}]}' \
        "${log_file}" >/dev/null; then
        rc=0
    else
        rc=$?
    fi
    assert_ne "0" "${rc}" "planner should fail when every candidate misses the floor"
    assert_match "Switch dataset provider" "$(cat "${log_file}")" "failure message recommends dataset switch"
}

test_large_model_threshold_covers_14b_dense_checkpoints() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    _is_large_model "13" || t_fail "expected 13B-class model sizes to skip overhead check"
    _is_large_model "14" || t_fail "expected 14B-class model sizes to skip overhead check"
    if _is_large_model "7"; then
        t_fail "expected 7B-class model sizes to keep overhead check enabled"
    fi
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

test_ensure_evaluate_baseline_report_falls_back_to_hf_causal_adapter_when_resolver_empty() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    _resolve_invarlock_adapter() { echo ""; }
    _validate_evaluate_baseline_report() {
        printf '%s\n' "$*" > "${TEST_TMPDIR}/validate.calls"
        return 0
    }

    local baseline_root="${TEST_TMPDIR}/baseline"
    mkdir -p "${baseline_root}"
    echo "{}" > "${baseline_root}/baseline_report.json"

    local log_file="${TEST_TMPDIR}/log.txt"
    : > "${log_file}"

    local result
    result="$(
        _ensure_evaluate_baseline_report \
            "${baseline_root}" \
            "${TEST_TMPDIR}/baseline_path" \
            "ci" \
            "ci" \
            "128" \
            "64" \
            "10" \
            "20" \
            "1" \
            "100" \
            "7" \
            "${log_file}"
    )"

    assert_match "baseline_report\\.json" "${result}" "baseline report path returned"
    assert_match "hf_causal" "$(cat "${TEST_TMPDIR}/validate.calls")" "adapter falls back to hf_causal"
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
    echo "Qwen/Qwen2.5-32B" > "${model_output_dir}/.model_id"
    : > "${log_file}"

    # Baseline missing error.
    if task_calibration_run "${model_name}" 0 1 42 "${TEST_TMPDIR}/nope" "${log_file}"; then
        t_fail "expected calibration to fail with missing baseline"
    fi

    # Already done skip.
    local run_dir="${model_output_dir}/reports/calibration/run_1"
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

test_task_evaluate_edit_and_error_cover_preset_discovery_overrides_and_report_copy_paths() {
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
    echo "Qwen/Qwen2.5-32B" > "${model_output_dir}/.model_id"
    : > "${log_file}"

    # Baseline missing error.
    if task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean 1 "${TEST_TMPDIR}/nope" "${log_file}"; then
        t_fail "expected evaluate_edit to fail without baseline"
    fi

    # Case arms for edit dir name mapping + missing edit path error.
    if task_evaluate_edit "${model_name}" 0 "fp8_quant:e4m3fn:ffn" clean 1 "${out}" "${log_file}"; then :; fi
    if task_evaluate_edit "${model_name}" 0 "magnitude_prune:0.1:ffn" clean 1 "${out}" "${log_file}"; then :; fi
    if task_evaluate_edit "${model_name}" 0 "lowrank_svd:8:attn" clean 1 "${out}" "${log_file}"; then :; fi

    # Full evaluate flow for quant_rtn with overrides and report copy.
    mkdir -p "${model_output_dir}/models/quant_4bit_clean"
    local cert_dir="${model_output_dir}/reports/quant_4bit_clean/run_1"
    mkdir -p "${cert_dir}/nested"
    echo "{}" > "${cert_dir}/nested/evaluation.report.json"

    export TASK_PARAMS='{"seq_len":100,"stride":200}'
    export INVARLOCK_BOOTSTRAP_N="1234"
    export INVARLOCK_CERT_MIN_WINDOWS="256"
    _estimate_model_size() { echo "13"; }

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

    task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean 1 "${out}" "${log_file}"
    local profile_yaml="${cert_dir}/config_root/runtime/profiles/ci.yaml"
    assert_file_exists "${profile_yaml}" "profile override created"
    local profile_contents
    profile_contents="$(cat "${profile_yaml}")"
    assert_match "seq_len: 100" "${profile_contents}" "profile override seq_len"
    assert_match "stride: 100" "${profile_contents}" "profile override stride uses pairing"
    assert_match "preview_n: 256" "${profile_contents}" "profile override preview_n"
    assert_match "final_n: 256" "${profile_contents}" "profile override final_n"
    assert_match "skip_overhead_check: true" "${profile_contents}" "large-model profile disables overhead check via config"

    local calls
    calls="$(cat "${TEST_TMPDIR}/fixtures/invarlock.calls")"
    assert_match "calibrated_preset_${model_name}__quant_rtn\\.yaml" "${calls}" "uses edit-type preset"
    if [[ "${calls}" =~ oom_override_preset\.yaml ]]; then
        t_fail "expected evaluate to avoid override preset file"
    fi
    local staged_quant_preset
    staged_quant_preset="$(cat "${cert_dir}/runtime_inputs/calibrated_preset_${model_name}__quant_rtn.yaml")"
    assert_match "skip_overhead_check: true" "${staged_quant_preset}" "staged preset carries skip_overhead policy"
    # Skip branch when cert already exists.
    task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean 1 "${out}" "${log_file}"

    # Preset discovery branch when preset exists.
    rm -f "${out}/presets/calibrated_preset_${model_name}__quant_rtn.yaml"
    echo "{}" > "${out}/presets/calibrated_preset_${model_name}.yaml"
    task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean 2 "${out}" "${log_file}"

    # Error model evaluate mirrors evaluate_edit branches.
    local error_path="${model_output_dir}/models/error_cuda_assert"
    mkdir -p "${error_path}"
    echo "{}" > "${error_path}/config.json"
    cert_dir="${model_output_dir}/reports/errors/cuda_assert"
    mkdir -p "${cert_dir}/nested"
    echo "{}" > "${cert_dir}/nested/evaluation.report.json"
    task_evaluate_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"
}

test_task_evaluate_edit_exits_when_workdir_cd_fails() {
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

    run task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean 1 "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "cd failure exits subshell and propagates non-zero"
}

test_task_evaluate_error_exits_when_workdir_cd_fails() {
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

    run task_evaluate_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "cd failure exits subshell and propagates non-zero"
}

test_task_evaluate_error_missing_baseline_missing_error_model_skip_and_preset_missing_branches() {
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
    if task_evaluate_error "${model_name}" 0 cuda_assert "${TEST_TMPDIR}/nope" "${log_file}"; then
        t_fail "expected evaluate_error to fail without baseline"
    fi

    # Baseline present, error model missing.
    mkdir -p "${baseline_dir}"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
    if task_evaluate_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"; then
        t_fail "expected evaluate_error to fail without error model"
    fi

    # Error model present, cert exists skip.
    local error_path="${model_output_dir}/models/error_cuda_assert"
    mkdir -p "${error_path}"
    echo "{}" > "${error_path}/config.json"
    local cert_dir="${model_output_dir}/reports/errors/cuda_assert"
    mkdir -p "${cert_dir}"
    echo "{}" > "${cert_dir}/evaluation.report.json"
    task_evaluate_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"

    # Preset missing branch creates a minimal preset.
    rm -f "${cert_dir}/evaluation.report.json"
    rm -rf "${out}/presets"
    task_evaluate_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"
}

test_task_evaluate_tasks_treat_nonzero_cli_rc_as_success_when_report_written() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"
    stub_resolve_edit_params

    fixture_write "invarlock.create_cert" ""
    fixture_write "invarlock.rc" "3"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "$(dirname "${log_file}")"
    : > "${log_file}"

    _estimate_model_size() { echo "7"; }
    _ensure_evaluate_baseline_report() { echo "${TEST_TMPDIR}/baseline_report.json"; }
    echo "{}" > "${TEST_TMPDIR}/baseline_report.json"

    # evaluate_EDIT: rc!=0 but report exists -> treat as success
    local edit_dir="${model_output_dir}/models/quant_4bit_clean"
    mkdir -p "${baseline_dir}" "${edit_dir}"
    echo "{}" > "${baseline_dir}/config.json"
    echo "{}" > "${edit_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"

    run task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean 1 "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "non-zero invarlock rc is ignored when report exists (edit)"
    assert_file_exists "${model_output_dir}/reports/quant_4bit_clean/run_1/evaluation.report.json" "edit report written"

    # evaluate_ERROR: rc!=0 but report exists -> treat as success
    local error_dir="${model_output_dir}/models/error_cuda_assert"
    mkdir -p "${error_dir}"
    echo "{}" > "${error_dir}/config.json"

    run task_evaluate_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "non-zero invarlock rc is ignored when report exists (error)"
    assert_file_exists "${model_output_dir}/reports/errors/cuda_assert/evaluation.report.json" "error report written"
}

test_task_evaluate_tasks_generate_evaluation_report_when_only_report_json_written() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"
    stub_resolve_edit_params

    fixture_write "invarlock.create_report_for_evaluate" ""
    fixture_write "invarlock.rc" "1"

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "$(dirname "${log_file}")"
    : > "${log_file}"

    _estimate_model_size() { echo "7"; }
    _ensure_evaluate_baseline_report() { echo "${TEST_TMPDIR}/baseline_report.json"; }
    echo "{}" > "${TEST_TMPDIR}/baseline_report.json"

    _cmd_python() {
        local script="$1"
        shift || true
        if [[ "${script}" == *"evaluation_report_from_report.py" ]]; then
            local out_path=""
            while [[ $# -gt 0 ]]; do
                if [[ "${1}" == "--out" ]]; then
                    out_path="${2:-}"
                    break
                fi
                shift
            done
            if [[ -n "${out_path}" ]]; then
                mkdir -p "$(dirname "${out_path}")"
                echo "{}" > "${out_path}"
            fi
            return 0
        fi
        return 0
    }

    # evaluate_EDIT: no evaluation.report.json produced by CLI, but report.json exists -> conversion creates cert.
    local edit_dir="${model_output_dir}/models/quant_4bit_clean"
    mkdir -p "${baseline_dir}" "${edit_dir}"
    echo "{}" > "${baseline_dir}/config.json"
    echo "{}" > "${edit_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"

    run task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean 1 "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "report.json conversion makes evaluate_EDIT succeed"
    assert_file_exists "${model_output_dir}/reports/quant_4bit_clean/run_1/evaluation.report.json" "converted edit report exists"

    # evaluate_ERROR: same behavior.
    local error_dir="${model_output_dir}/models/error_cuda_assert"
    mkdir -p "${error_dir}"
    echo "{}" > "${error_dir}/config.json"

    run task_evaluate_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"
    assert_rc "0" "${RUN_RC}" "report.json conversion makes evaluate_ERROR succeed"
    assert_file_exists "${model_output_dir}/reports/errors/cuda_assert/evaluation.report.json" "converted error report exists"
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
    if task_create_error "${model_name}" 0 cuda_assert '{}' "${TEST_TMPDIR}/nope" "${log_file}"; then
        t_fail "expected create_error to fail without baseline"
    fi

    # Missing create_error_model implementation.
    if task_create_error "${model_name}" 0 cuda_assert '{}' "${out}" "${log_file}"; then
        t_fail "expected create_error to fail when create_error_model is missing"
    fi

    local captured_mode="${TEST_TMPDIR}/captured_mode.txt"
    create_error_model() {
        echo "${INVARLOCK_RMT_PROBE_MODE:-}" > "${captured_mode}"
        mkdir -p "$2"
        echo "{}" > "$2/config.json"
        # task_create_error treats error models as cached only when the injector
        # completed (signaled by error_metadata.json).
        echo "{}" > "$2/error_metadata.json"
    }
    task_create_error "${model_name}" 0 cuda_assert '{"INVARLOCK_RMT_PROBE_MODE":"anisotropy"}' "${out}" "${log_file}"
    assert_eq "anisotropy" "$(cat "${captured_mode}")" "injector env propagated to create_error_model"

    task_create_error "${model_name}" 0 cuda_assert '{}' "${out}" "${log_file}"
    task_create_error "${model_name}" 0 cuda_assert '{}' "${out}" "${log_file}"

    # Verify failure branch.
    rm -rf "${model_output_dir}/models/error_cuda_assert"
    create_error_model() { mkdir -p "$2"; }
    if task_create_error "${model_name}" 0 cuda_assert '{}' "${out}" "${log_file}"; then
        t_fail "expected create_error verification failure"
    fi
}

test_task_create_error_recreates_incomplete_models_and_propagates_failures() {
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

    local error_dir="${model_output_dir}/models/error_cuda_assert"
    mkdir -p "${error_dir}"
    echo "{}" > "${error_dir}/config.json"

    local env_capture="${TEST_TMPDIR}/env.capture"
    export INVARLOCK_SOMETHING="previous"
    create_error_model() {
        printf '%s\n' "${INVARLOCK_SOMETHING-}" > "${env_capture}"
        mkdir -p "$2"
        echo "{}" > "$2/config.json"
        echo "{}" > "$2/error_metadata.json"
    }

    task_create_error "${model_name}" 0 cuda_assert '[]' "${out}" "${log_file}"
    assert_eq "previous" "$(cat "${env_capture}")" "non-object env payloads are ignored"
    assert_match "missing error_metadata" "$(cat "${log_file}")" "incomplete error model warning logged"

    rm -rf "${error_dir}"
    task_create_error "${model_name}" 0 cuda_assert '{"INVARLOCK_SOMETHING":"override"}' "${out}" "${log_file}"
    assert_eq "override" "$(cat "${env_capture}")" "existing env values are overridden during injector execution"
    assert_eq "previous" "${INVARLOCK_SOMETHING}" "injector env values are restored after create_error_model"

    rm -rf "${error_dir}"
    create_error_model() { return 4; }
    run task_create_error "${model_name}" 0 cuda_assert '{}' "${out}" "${log_file}"
    assert_rc "1" "${RUN_RC}" "create_error_model failure returns non-zero"
    assert_match "create_error_model failed" "${RUN_OUT}${RUN_ERR}$(cat "${log_file}")" "failure logged"
}

test_task_cleanup_edit_and_error_cover_guard_paths() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

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

    _estimate_model_size() { echo "7"; }
    _ensure_evaluate_baseline_report() { echo "${TEST_TMPDIR}/baseline_report.json"; }
    echo "{}" > "${TEST_TMPDIR}/baseline_report.json"

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

    echo "{}" > "${TEST_TMPDIR}/baseline_report.json"
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
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    assert_eq "moe" "$(_get_model_size_from_name "Mixtral-8x7B")" "moe detection"
    assert_eq "13" "$(_get_model_size_from_name "model-13b")" "13B detection"
    assert_eq "70" "$(_get_model_size_from_name "Qwen1.5-72B")" "70B detection"
    assert_eq "40" "$(_get_model_size_from_name "01-ai/Yi-34B")" "40B detection"
    assert_eq "30" "$(_get_model_size_from_name "Qwen2.5-32B")" "30B detection"

    assert_eq "512:512:192:192:64" "$(_get_model_invarlock_config_fallback "13")" "13B config"
    assert_eq "1024:1024:192:192:48" "$(_get_model_invarlock_config_fallback "30")" "30B config"
    assert_eq "1024:1024:192:192:32" "$(_get_model_invarlock_config_fallback "40")" "40B config"
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
    _is_large_model "model-30b" || t_fail "expected 30b string to be large"
}

test_task_helpers_cover_bootstrap_replicates_floor_logic() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

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
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

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
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

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
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

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
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

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
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

    local calls="${TEST_TMPDIR}/python.calls"
    _cmd_python() {
        echo "python $*" >> "${calls}"
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

    run _validate_evaluate_baseline_report "${report}" "hf_auto" "ci" "balanced"
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
    _validate_evaluate_baseline_report() { return 0; }

    export PACK_GUARDS_ORDER="invariants, spectral , rmt"
    export HF_HOME="${TEST_TMPDIR}/hf-home"
    mkdir -p "${HF_HOME}"
    fixture_write "invarlock.create_report_nested" ""
    cat > "${TEST_TMPDIR}/fixtures/invarlock.capture_env_keys" <<'EOF'
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
    assert_match "Generating reusable baseline report" "$(cat "${log_file}")" "generation logged"
    local env_capture
    env_capture="$(cat "${TEST_TMPDIR}/fixtures/invarlock.env")"
    assert_match "PYTHONPATH=${PACK_REPO_PYTHONPATH}" "${env_capture}" "baseline run forwards repo pythonpath"
    assert_match "INVARLOCK_CONFIG_ROOT=${baseline_root}/config_root" "${env_capture}" "baseline run forwards config root"
    assert_match "INVARLOCK_STORE_EVAL_WINDOWS=1" "${env_capture}" "baseline run stores eval windows"
    assert_match "INVARLOCK_SKIP_OVERHEAD_CHECK=1" "${env_capture}" "large-model baseline run skips overhead check"
    assert_match "HF_HOME=${HF_HOME}" "${env_capture}" "baseline run preserves inherited HF cache root"
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
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

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

test_task_evaluate_edit_reuses_baseline_report_applies_ci_override_and_falls_back_label() {
    mock_reset
    push_active_python_bin
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

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
    export INVARLOCK_CERT_MIN_WINDOWS="192"

    local baseline_report="${TEST_TMPDIR}/baseline_report.json"
    echo '{"evaluation_windows":{"preview":{"window_ids":[1],"input_ids":[[1]]},"final":{"window_ids":[1],"input_ids":[[1]]}},"edit":{"name":"noop"}}' > "${baseline_report}"
    _ensure_evaluate_baseline_report() { echo "${baseline_report}"; }

    resolve_edit_params() {
        jq -n '{status:"selected", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:"_clean"}'
    }
    local edit_dir="${model_output_dir}/models/_clean"
    mkdir -p "${edit_dir}"
    echo "{}" > "${edit_dir}/config.json"

    mkdir -p "${out}/presets"
    echo "{}" > "${out}/presets/calibrated_preset_${model_name}.yaml"

    task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean 1 "${out}" "${log_file}"

    assert_match "CI window override" "$(cat "${log_file}")" "CI window override applied"
    assert_match "Reusing baseline report" "$(cat "${log_file}")" "baseline report reused"
    assert_match "Staged baseline report for evaluate runtime" "$(cat "${log_file}")" "baseline report staged into cert dir"
    assert_match "Staged preset for evaluate runtime" "$(cat "${log_file}")" "preset staged into cert dir"
    assert_match "Normalized staged preset dataset for evaluate runtime: seq=128, stride=128, preview=192, final=192" "$(cat "${log_file}")" "preset dataset normalized for evaluate edit"
    assert_file_exists "${model_output_dir}/reports/_clean/run_1/runtime_inputs/baseline_report.json" "staged baseline report exists for evaluate edit"
    assert_file_exists "${model_output_dir}/reports/_clean/run_1/runtime_inputs/calibrated_preset_${model_name}.yaml" "staged preset exists for evaluate edit"
    local staged_preset_contents
    staged_preset_contents="$(cat "${model_output_dir}/reports/_clean/run_1/runtime_inputs/calibrated_preset_${model_name}.yaml")"
    assert_match "seq_len: 128" "${staged_preset_contents}" "staged preset seq_len normalized for evaluate edit"
    assert_match "stride: 128" "${staged_preset_contents}" "staged preset stride normalized for evaluate edit"
    assert_match "preview_n: 192" "${staged_preset_contents}" "staged preset preview_n normalized for evaluate edit"
    assert_match "final_n: 192" "${staged_preset_contents}" "staged preset final_n normalized for evaluate edit"

    local calls
    calls="$(cat "${TEST_TMPDIR}/fixtures/invarlock.calls")"
    assert_match "PYTHONPATH=${PACK_REPO_PYTHONPATH}" "${calls}" "absolute repo PYTHONPATH forwarded to evaluate"
    assert_match "INVARLOCK_CONFIG_ROOT=${model_output_dir}/reports/_clean/run_1/config_root" "${calls}" "evaluate forwards config root"
    assert_match "INVARLOCK_STORE_EVAL_WINDOWS=1" "${calls}" "evaluate enables stored windows"
    assert_match "HF_HOME=${HF_HOME}" "${calls}" "evaluate preserves inherited HF cache root"
    assert_match "--baseline-report" "${calls}" "baseline report forwarded to invarlock evaluate"
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
    source "${TEST_ROOT}/scripts/proof_packs/lib/task_functions.sh"

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
    pop_active_python_bin
}

test_task_evaluate_error_reuses_baseline_report_and_applies_ci_override() {
    mock_reset
    push_active_python_bin
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
    _ensure_evaluate_baseline_report() { echo "${baseline_report}"; }

    mkdir -p "${out}/presets"
    echo "{}" > "${out}/presets/calibrated_preset_${model_name}.yaml"

    task_evaluate_error "${model_name}" 0 nan_injection "${out}" "${log_file}"

    assert_match "CI window override" "$(cat "${log_file}")" "CI window override applied"
    assert_match "Reusing baseline report" "$(cat "${log_file}")" "baseline report reused"
    assert_match "Staged baseline report for evaluate runtime" "$(cat "${log_file}")" "baseline report staged into error cert dir"
    assert_match "Staged preset for evaluate runtime" "$(cat "${log_file}")" "preset staged into error cert dir"
    assert_match "Normalized staged preset dataset for evaluate runtime: seq=128, stride=128, preview=192, final=192" "$(cat "${log_file}")" "preset dataset normalized for evaluate error"
    assert_file_exists "${model_output_dir}/reports/errors/nan_injection/evaluation.report.json" "error cert written"
    assert_file_exists "${model_output_dir}/reports/errors/nan_injection/runtime_inputs/baseline_report.json" "staged baseline report exists for evaluate error"
    assert_file_exists "${model_output_dir}/reports/errors/nan_injection/runtime_inputs/calibrated_preset_${model_name}.yaml" "staged preset exists for evaluate error"
    local staged_error_preset_contents
    staged_error_preset_contents="$(cat "${model_output_dir}/reports/errors/nan_injection/runtime_inputs/calibrated_preset_${model_name}.yaml")"
    assert_match "seq_len: 128" "${staged_error_preset_contents}" "staged preset seq_len normalized for evaluate error"
    assert_match "stride: 128" "${staged_error_preset_contents}" "staged preset stride normalized for evaluate error"
    assert_match "preview_n: 192" "${staged_error_preset_contents}" "staged preset preview_n normalized for evaluate error"
    assert_match "final_n: 192" "${staged_error_preset_contents}" "staged preset final_n normalized for evaluate error"
    pop_active_python_bin
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
    task_evaluate_edit() { :; }
    task_cleanup_edit() { :; }
    task_create_error() { :; }
    task_evaluate_error() { :; }
    task_cleanup_error() { :; }
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

    local types=(SETUP_BASELINE CALIBRATION_RUN CREATE_EDIT CREATE_EDITS_BATCH evaluate_EDIT CLEANUP_EDIT CREATE_ERROR evaluate_ERROR CLEANUP_ERROR GENERATE_PRESET)
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

test_task_evaluate_edit_skip_and_invalid() {
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
    task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean 1 "${out}" "${log_file}"

    resolve_edit_params() {
        jq -n '{status:"invalid", edit_type:"quant_rtn", param1:"4", param2:"32", scope:"ffn", edit_dir_name:"quant_4bit_clean"}'
    }
    run task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:ffn" clean 1 "${out}" "${log_file}"
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
    local captured_py="${TEST_TMPDIR}/captured_pythonpath.txt"
    local captured_cfg="${TEST_TMPDIR}/captured_config_root.txt"
    local captured_tmp="${TEST_TMPDIR}/captured_tmpdir.txt"
    local captured_hf="${TEST_TMPDIR}/captured_hf_home.txt"
    cat > "${bin_dir}/invarlock" <<EOF
#!/usr/bin/env bash
printf '%s\n' "\${PYTHONPATH:-}" > "${captured_py}"
printf '%s\n' "\${INVARLOCK_CONFIG_ROOT:-}" > "${captured_cfg}"
printf '%s\n' "\${TMPDIR:-}" > "${captured_tmp}"
printf '%s\n' "\${HF_HOME:-}" > "${captured_hf}"
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"
    PATH="${bin_dir}:${PATH}"
    export PATH
    export TMPDIR="${TEST_TMPDIR}/tmpdir"
    export HF_HOME="${TEST_TMPDIR}/hf-home"
    mkdir -p "${TMPDIR}" "${HF_HOME}"

    PACK_GUARDS_ORDER="variance"
    task_calibration_run "${model_name}" 0 "1" "42" "${out}" "${log_file}"
    assert_match "variance" "$(cat "${model_output_dir}/reports/calibration/run_1/calibration_config.yaml")" "explicit guard order used"
    assert_eq "${PACK_REPO_PYTHONPATH}" "$(cat "${captured_py}")" "calibration injects repo pythonpath"
    assert_eq "${model_output_dir}/reports/calibration/run_1/config_root" "$(cat "${captured_cfg}")" "calibration injects config root"
    assert_eq "${TMPDIR}" "$(cat "${captured_tmp}")" "calibration preserves tmpdir override"
    assert_eq "${HF_HOME}" "$(cat "${captured_hf}")" "calibration preserves HF cache root"

    PACK_GUARDS_ORDER=" , "
    task_calibration_run "${model_name}" 0 "2" "43" "${out}" "${log_file}"
    assert_match "spectral" "$(cat "${model_output_dir}/reports/calibration/run_2/calibration_config.yaml")" "default guard order used"
}
