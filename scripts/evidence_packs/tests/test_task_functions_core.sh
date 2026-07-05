#!/usr/bin/env bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/task_functions_test_helpers.sh"

test_default_ci_min_windows_accounts_for_padding() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    unset INVARLOCK_CERT_MIN_WINDOWS
    unset INVARLOCK_DATASET
    unset INVARLOCK_TIER
    assert_eq "400" "$(_default_ci_min_windows "512")" "default evidence-pack dataset is WikiText-2 with the higher balanced floor"
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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    _is_large_model "13" || t_fail "expected 13B-class model sizes to skip overhead check"
    _is_large_model "14" || t_fail "expected 14B-class model sizes to skip overhead check"
    if _is_large_model "7"; then
        t_fail "expected 7B-class model sizes to keep overhead check enabled"
    fi
}

test_baseline_report_wait_budget_scales_for_heavy_7b_windows() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    run _baseline_report_wait_secs "7" "128" "128"
    assert_rc "0" "${RUN_RC}" "default wait helper succeeds"
    assert_eq "240" "${RUN_OUT}" "non-heavy 7B windows keep the short wait budget"

    run _baseline_report_wait_secs "7" "400" "400"
    assert_rc "0" "${RUN_RC}" "heavy-window wait helper succeeds"
    assert_eq "1800" "${RUN_OUT}" "heavy 7B windows use the long wait budget"

    export PACK_BASELINE_REPORT_WAIT_SECS_HEAVY_WINDOWS="900"
    run _baseline_report_wait_secs "7" "400" "400"
    assert_rc "0" "${RUN_RC}" "heavy-window override succeeds"
    assert_eq "900" "${RUN_OUT}" "heavy-window override is honored"

    export PACK_BASELINE_REPORT_WAIT_SECS_LARGE="1200"
    run _baseline_report_wait_secs "14" "400" "400"
    assert_rc "0" "${RUN_RC}" "large-model wait helper succeeds"
    assert_eq "1200" "${RUN_OUT}" "large-model override still takes precedence"

    export PACK_BASELINE_REPORT_WAIT_HEAVY_WINDOW_TOTAL_MIN="bad"
    run _baseline_report_wait_secs "7" "400" "400"
    assert_rc "0" "${RUN_RC}" "invalid heavy-window floor is sanitized"
    assert_eq "900" "${RUN_OUT}" "invalid heavy-window floor still honors heavy-window wait override"

    unset PACK_BASELINE_REPORT_WAIT_SECS_HEAVY_WINDOWS PACK_BASELINE_REPORT_WAIT_SECS_LARGE PACK_BASELINE_REPORT_WAIT_HEAVY_WINDOW_TOTAL_MIN
}

test_model_size_and_eval_batch_selection() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    mock_python3_stub_enable

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

    mock_python3_stub_enable
    fixture_write "python3.rc" "0"
    export PYTHON_BIN="$(command -v python3)"
    _cmd_python() { command "${PYTHON_BIN}" "$@"; }

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"
    stub_resolve_edit_params

    mock_python3_stub_enable
    fixture_write "python3.rc" "0"
    mock_python3_stub_allow_real_script "validate_artifact.py"

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

    _write_complete_edit_artifact() {
        write_minimal_validation_edit_artifact "$1"
    }

    # Create function stubs that materialize complete edit artifacts for verification.
    create_edited_model() { _write_complete_edit_artifact "$2"; }
    create_fp8_model() { _write_complete_edit_artifact "$2"; }
    create_pruned_model() { _write_complete_edit_artifact "$2"; }
    create_lowrank_model() { _write_complete_edit_artifact "$2"; }

    task_create_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean "${out}" "${log_file}"
    task_create_edit "${model_name}" 0 "fp8_quant:e4m3fn:ffn" clean "${out}" "${log_file}"
    task_create_edit "${model_name}" 0 "magnitude_prune:0.1:ffn" clean "${out}" "${log_file}"
    task_create_edit "${model_name}" 0 "lowrank_svd:8:attn" clean "${out}" "${log_file}"

    # Existing edit skips.
    task_create_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean "${out}" "${log_file}"

    # Partial artifacts must not be treated as complete.
    rm -f "${model_output_dir}/models/quant_4bit_clean/pytorch_model.bin"
    create_edited_model() { _write_complete_edit_artifact "$2"; echo "recreated" > "$2/recreated.marker"; }
    task_create_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean "${out}" "${log_file}"
    [[ -f "${model_output_dir}/models/quant_4bit_clean/recreated.marker" ]] || t_fail "expected partial artifact to be recreated"

    # Corrupt safetensors artifacts must not be treated as complete.
    local corrupt_dir="${model_output_dir}/models/corrupt_safe"
    mkdir -p "${corrupt_dir}"
    echo "{}" > "${corrupt_dir}/config.json"
    echo "{}" > "${corrupt_dir}/tokenizer_config.json"
    echo "not-a-valid-safetensors-file" > "${corrupt_dir}/model.safetensors"
    if _edit_artifact_complete "${corrupt_dir}"; then
        t_fail "expected corrupt safetensors artifact to fail completeness validation"
    fi

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"
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
    write_minimal_validation_edit_artifact "${model_output_dir}/models/quant_4bit_clean" "quant_rtn"
    local cert_dir="${model_output_dir}/reports/quant_4bit_clean/run_1"
    mkdir -p "${cert_dir}/nested"
    echo "{}" > "${cert_dir}/nested/evaluation.report.json"

    export TASK_PARAMS='{"seq_len":100,"stride":200}'
    export INVARLOCK_BOOTSTRAP_N="1234"
    export INVARLOCK_CERT_MIN_WINDOWS="256"
    export PACK_DEFER_REPORT_RENDERING="1"
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
    assert_match "--defer-report-rendering" "${calls}" "deferred optional report rendering flag forwarded"
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
    unset PACK_DEFER_REPORT_RENDERING
}

test_task_evaluate_edit_exits_when_workdir_cd_fails() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"
    stub_resolve_edit_params

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local edit_dir="${model_output_dir}/models/quant_4bit_clean"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "${baseline_dir}" "$(dirname "${log_file}")"
    write_minimal_validation_edit_artifact "${edit_dir}" "quant_rtn"
    echo "{}" > "${baseline_dir}/config.json"
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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"

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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"
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
    write_minimal_evaluate_baseline_report "${TEST_TMPDIR}/baseline_report.json"

    # evaluate_EDIT: rc!=0 but report exists -> treat as success
    local edit_dir="${model_output_dir}/models/quant_4bit_clean"
    mkdir -p "${baseline_dir}"
    write_minimal_validation_edit_artifact "${edit_dir}" "quant_rtn"
    echo "{}" > "${baseline_dir}/config.json"
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
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"
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
    write_minimal_evaluate_baseline_report "${TEST_TMPDIR}/baseline_report.json"

    _cmd_python() {
        local script="$1"
        shift || true
        if [[ "${script}" == *"task_tools.py" && "${1:-}" == "baseline-report-schedule" ]]; then
            echo "128:128:192:192"
            return 0
        fi
        if [[ "${script}" == *"task_tools.py" && "${1:-}" == "evaluation-report" ]]; then
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
    mkdir -p "${baseline_dir}"
    write_minimal_validation_edit_artifact "${edit_dir}" "quant_rtn"
    echo "{}" > "${baseline_dir}/config.json"
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

test_task_evaluate_tasks_return_conversion_failure_when_report_generation_fails() {
    mock_reset
    # shellcheck source=../task_functions.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/tasks/task_functions.sh"
    stub_resolve_edit_params

    fixture_write "invarlock.create_report_for_evaluate" ""

    local out="${TEST_TMPDIR}/out"
    local model_name="m"
    local model_output_dir="${out}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"
    local log_file="${TEST_TMPDIR}/log.txt"
    mkdir -p "$(dirname "${log_file}")"
    : > "${log_file}"

    _estimate_model_size() { echo "7"; }
    _ensure_evaluate_baseline_report() { echo "${TEST_TMPDIR}/baseline_report.json"; }
    write_minimal_evaluate_baseline_report "${TEST_TMPDIR}/baseline_report.json"
    _cmd_python() {
        local script="$1"
        shift || true
        if [[ "${script}" == *"task_tools.py" && "${1:-}" == "baseline-report-schedule" ]]; then
            echo "128:128:192:192"
            return 0
        fi
        if [[ "${script}" == *"task_tools.py" && "${1:-}" == "evaluation-report" ]]; then
            return 9
        fi
        return 0
    }

    local edit_dir="${model_output_dir}/models/quant_4bit_clean"
    mkdir -p "${baseline_dir}"
    write_minimal_validation_edit_artifact "${edit_dir}" "quant_rtn"
    echo "{}" > "${baseline_dir}/config.json"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"

    run task_evaluate_edit "${model_name}" 0 "quant_rtn:4:32:attn" clean 1 "${out}" "${log_file}"
    assert_rc "9" "${RUN_RC}" "evaluate_EDIT returns report conversion failure"
    assert_match "failed to generate evaluation\\.report\\.json" "$(cat "${log_file}")" "edit conversion failure is logged"

    local error_dir="${model_output_dir}/models/error_cuda_assert"
    mkdir -p "${error_dir}"
    echo "{}" > "${error_dir}/config.json"

    run task_evaluate_error "${model_name}" 0 cuda_assert "${out}" "${log_file}"
    assert_rc "9" "${RUN_RC}" "evaluate_ERROR returns report conversion failure"
}

test_task_create_error_branches_cover_skip_missing_function_and_verify_paths() {
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
