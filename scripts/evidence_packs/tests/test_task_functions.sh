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
            lora_merge)
                edit_dir_name="lora_rank${param1}_${version}"
                ;;
            fine_tune)
                edit_dir_name="fine_tune_step${param2}_${version}"
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

write_validation_edit_metadata() {
    local edit_path="$1"
    local edit_type="${2:-quant_rtn}"
    local storage_format="float_dequantized"
    case "${edit_type}" in
        magnitude_prune)
            storage_format="dense_float_with_zeros"
            ;;
        lowrank_svd)
            storage_format="dense_float_lowrank_approximated"
            ;;
        lora_merge)
            storage_format="merged_dense_checkpoint"
            ;;
        fine_tune)
            storage_format="fine_tuned_dense_checkpoint"
            ;;
    esac
    cat > "${edit_path}/edit_metadata.json" <<JSON
{
  "schema": "invarlock/evidence-pack-edit-metadata-v1",
  "artifact_class": "validation_subject_checkpoint",
  "edit_type": "${edit_type}",
  "edit_semantics": "external_subject_validation_edit",
  "deployable_as_hf_checkpoint": true,
  "optimized_deployment_backend": false,
  "backend": null,
  "storage_format": "${storage_format}",
  "actual_storage_format": "${storage_format}",
  "packed_quantized_storage": false,
  "runtime_memory_reduction": false,
  "scope": "ffn",
  "parameters": {},
  "coverage": {
    "edited_tensors": 1,
    "edited_params": 1,
    "total_params": 1,
    "coverage_ratio": 1.0
  }
}
JSON
}

write_minimal_validation_edit_artifact() {
    local edit_path="$1"
    local edit_type="${2:-quant_rtn}"
    mkdir -p "${edit_path}"
    echo "{}" > "${edit_path}/config.json"
    echo "weights" > "${edit_path}/pytorch_model.bin"
    echo "{}" > "${edit_path}/tokenizer_config.json"
    write_validation_edit_metadata "${edit_path}" "${edit_type}"
}

write_minimal_evaluate_baseline_report() {
    local report_path="$1"
    local seq_len="${2:-128}"
    local stride="${3:-128}"
    local preview_n="${4:-192}"
    local final_n="${5:-192}"
    cat > "${report_path}" <<JSON
{
  "data": {
    "seq_len": ${seq_len},
    "stride": ${stride},
    "preview_n": ${preview_n},
    "final_n": ${final_n}
  },
  "evaluation_windows": {
    "preview": {"window_ids": [1], "input_ids": [[1]]},
    "final": {"window_ids": [1], "input_ids": [[1]]}
  },
  "edit": {"name": "noop"}
}
JSON
}

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
