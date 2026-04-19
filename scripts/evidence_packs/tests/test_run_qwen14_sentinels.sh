#!/usr/bin/env bash

test_run_qwen14_sentinels_helper_functions_cover_resolution_and_error_paths() {
    mock_reset

    source ./scripts/evidence_packs/run_qwen14_sentinels.sh

    run require_dir "${TEST_TMPDIR}/missing-dir" "run directory"
    assert_rc "1" "${RUN_RC}" "missing directories fail directly"

    run require_file "${TEST_TMPDIR}/missing-file" "preset"
    assert_rc "1" "${RUN_RC}" "missing files fail directly"

    run stage_runtime_input "${TEST_TMPDIR}/missing-file" "${TEST_TMPDIR}/staged-missing" "preset"
    assert_rc "1" "${RUN_RC}" "stage_runtime_input fails when the source file is missing"

    local staged_source="${TEST_TMPDIR}/preset.yaml"
    printf 'dataset:\n  seq_len: 64\n' > "${staged_source}"
    local staged_path
    staged_path="$(stage_runtime_input "${staged_source}" "${TEST_TMPDIR}/staged" "preset")"
    assert_eq "${TEST_TMPDIR}/staged/preset.yaml" "${staged_path}" "stage_runtime_input copies files into runtime_inputs"
    assert_file_exists "${staged_path}" "staged file created"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/pycustom" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    cat > "${bin_dir}/python" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    cat > "${bin_dir}/python3" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    chmod +x "${bin_dir}/pycustom" "${bin_dir}/python" "${bin_dir}/python3"

    PYTHON_BIN="${bin_dir}/pycustom"
    assert_eq "${bin_dir}/pycustom" "$(resolve_python_bin)" "explicit PYTHON_BIN wins when executable"

    unset PYTHON_BIN
    PATH="${bin_dir}"
    assert_eq "python" "$(resolve_python_bin)" "python is used before python3 when present"

    /bin/rm -f "${bin_dir}/python"
    assert_eq "python3" "$(resolve_python_bin)" "python3 is used when python is absent"

    PATH=""
    local rc=0
    local err_file="${TEST_TMPDIR}/resolve_python_bin.err"
    if resolve_python_bin > /dev/null 2> "${err_file}"; then
        rc=0
    else
        rc=$?
    fi
    assert_eq "1" "${rc}" "resolve_python_bin fails when no interpreter can be found"

    PATH="/bin:/usr/bin"

    local run_dir="${TEST_TMPDIR}/run"
    local model_name="qwen__qwen2.5-14b"
    local model_dir="${run_dir}/${model_name}"
    mkdir -p "${model_dir}/models/baseline" "${run_dir}/presets" "${model_dir}/baseline_reports/report_a" "${model_dir}/baseline_reports/report_z"
    printf '%s\n' "${model_dir}/models/baseline" > "${model_dir}/.baseline_path"
    printf '%s\n' '{}' > "${model_dir}/baseline_reports/report_a/baseline_report.json"
    printf '%s\n' '{}' > "${model_dir}/baseline_reports/report_z/baseline_report.json"

    assert_eq "${model_dir}/models/baseline" "$(resolve_baseline_path "${run_dir}" "${model_name}")" "valid baseline hint is used"

    printf '%s\n' "${TEST_TMPDIR}/invalid-baseline" > "${model_dir}/.baseline_path"
    assert_eq "${model_dir}/models/baseline" "$(resolve_baseline_path "${run_dir}" "${model_name}")" "invalid baseline hint falls back to baseline model dir"
    assert_eq "${model_dir}/baseline_reports/report_z/baseline_report.json" "$(resolve_baseline_report "${run_dir}" "${model_name}")" "lexically latest baseline report is selected"

    printf 'preset: quant\n' > "${run_dir}/presets/calibrated_preset_${model_name}__quant_rtn.json"
    assert_eq "${run_dir}/presets/calibrated_preset_${model_name}__quant_rtn.json" "$(resolve_preset_path "${run_dir}" "${model_name}" "quant_rtn")" "quant-specific json preset is accepted"
    rm -f "${run_dir}/presets/calibrated_preset_${model_name}__quant_rtn.json"
    printf 'preset: base\n' > "${run_dir}/presets/calibrated_preset_${model_name}.json"
    assert_eq "${run_dir}/presets/calibrated_preset_${model_name}.json" "$(resolve_preset_path "${run_dir}" "${model_name}" "quant_rtn")" "base json preset is accepted as final fallback"
}

test_run_qwen14_sentinels_main_is_sourceable_and_covers_mode_dispatch() {
    mock_reset

    source ./scripts/evidence_packs/run_qwen14_sentinels.sh

    local run_dir="${TEST_TMPDIR}/run"
    local model_name="qwen__qwen2.5-14b"
    mkdir -p "${run_dir}/${model_name}/models/quant_4bit_clean" "${run_dir}/${model_name}/models/prune_clean"

    local calls_file="${TEST_TMPDIR}/main.calls"
    require_dir() { return 0; }
    resolve_baseline_path() { printf '%s\n' "${TEST_TMPDIR}/baseline"; }
    resolve_baseline_report() { printf '%s\n' "${TEST_TMPDIR}/baseline_report.json"; }
    resolve_preset_path() { printf '%s\n' "${TEST_TMPDIR}/preset_${3}.yaml"; }
    run_evaluate_sentinel() {
        printf 'evaluate\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$8" >> "${calls_file}"
        return 0
    }
    run_public_quant_verify() {
        printf 'verify\t%s\t%s\n' "$1" "$2" >> "${calls_file}"
        return 0
    }
    find() { return 0; }

    run main --help
    assert_rc "0" "${RUN_RC}" "help returns zero from sourced main"
    assert_match "Usage" "${RUN_OUT}" "help prints usage"

    run main --bogus
    assert_rc "2" "${RUN_RC}" "unknown args return usage error"

    run main --model-name "${model_name}"
    assert_rc "2" "${RUN_RC}" "missing run-dir is rejected"

    run main --run-dir "${run_dir}"
    assert_rc "2" "${RUN_RC}" "missing model-name is rejected"

    run main --run-dir "${run_dir}" --model-name "${model_name}" --mode nope
    assert_rc "2" "${RUN_RC}" "invalid mode is rejected"

    : > "${calls_file}"
    run main --run-dir "${run_dir}" --model-name "${model_name}" --mode public-quant --device cpu --profile smoke --adapter test
    assert_rc "0" "${RUN_RC}" "public-quant mode succeeds"
    assert_match "evaluate.*quant_4bit_clean.*preset_quant_rtn\\.yaml.*${run_dir}/sentinels/qwen14/quant_4bit_clean.*cpu" "$(cat "${calls_file}")" "public-quant mode evaluates the quant subject and uses default out dir"
    assert_match "verify.*quant_4bit_clean/evaluation\\.report\\.json.*${run_dir}/sentinels/qwen14/quant_4bit_clean" "$(cat "${calls_file}")" "public-quant mode verifies the quant report"

    : > "${calls_file}"
    run main --run-dir "${run_dir}" --model-name "${model_name}" --mode saved-model --out "${TEST_TMPDIR}/sentinels"
    assert_rc "0" "${RUN_RC}" "saved-model mode succeeds"
    assert_match "evaluate.*quant_4bit_clean" "$(cat "${calls_file}")" "saved-model mode still runs the quant sentinel"
    assert_match "evaluate.*prune_clean.*preset_magnitude_prune\\.yaml.*${TEST_TMPDIR}/sentinels/prune_clean" "$(cat "${calls_file}")" "saved-model mode runs the prune sentinel"
}

test_run_qwen14_sentinels_runs_saved_model_and_public_quant_modes() {
    mock_reset

    local run_dir="${TEST_TMPDIR}/run"
    local model_name="qwen__qwen2.5-14b"
    local model_dir="${run_dir}/${model_name}"
    mkdir -p \
        "${model_dir}/models/baseline" \
        "${model_dir}/models/quant_4bit_clean" \
        "${model_dir}/models/prune_clean" \
        "${model_dir}/baseline_reports/ci_balanced_seq1536_pv48_fn48" \
        "${run_dir}/presets"
    printf '%s\n' "${model_dir}/models/baseline" > "${model_dir}/.baseline_path"
    cat > "${model_dir}/baseline_reports/ci_balanced_seq1536_pv48_fn48/baseline_report.json" <<'EOF'
{"data":{"seq_len":1536,"stride":1536,"preview_n":48,"final_n":48}}
EOF
    printf 'dataset:\n  seq_len: 1536\n' > "${run_dir}/presets/calibrated_preset_${model_name}__quant_rtn.yaml"
    printf 'dataset:\n  seq_len: 1536\n' > "${run_dir}/presets/calibrated_preset_${model_name}.yaml"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
calls_file="${TEST_QWEN14_SENTINEL_CALLS:?}"
cmd="${1:-}"
shift || true
case "${cmd}" in
    evaluate)
        subject=""
        preset=""
        baseline=""
        baseline_report=""
        report_out=""
        out=""
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --subject) subject="$2"; shift 2 ;;
                --preset) preset="$2"; shift 2 ;;
                --baseline) baseline="$2"; shift 2 ;;
                --baseline-report) baseline_report="$2"; shift 2 ;;
                --report-out) report_out="$2"; shift 2 ;;
                --out) out="$2"; shift 2 ;;
                *) shift ;;
            esac
        done
        printf 'evaluate\t%s\t%s\t%s\t%s\t%s\t%s\n' "${subject}" "${preset}" "${baseline}" "${baseline_report}" "${report_out}" "${out}" >> "${calls_file}"
        mkdir -p "${report_out}"
        cp "${preset}" "${report_out}/observed_preset.yaml"
        cp "${baseline_report}" "${report_out}/observed_baseline_report.json"
        printf '{"ok":true}\n' > "${report_out}/evaluation.report.json"
        printf '{"report":"ok"}\n' > "${out}"
        if [[ "${subject}" == *"quant_4bit_clean" ]]; then
            exit 13
        fi
        exit 0
        ;;
    verify)
        report=""
        profile=""
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --json)
                    shift
                    ;;
                --profile)
                    profile="$2"
                    shift 2
                    ;;
                *)
                    report="$1"
                    shift
                    ;;
            esac
        done
        printf 'verify\t%s\t%s\n' "${profile}" "${report}" >> "${calls_file}"
        printf '{"ok":true}\n'
        exit 17
        ;;
    *)
        printf 'unexpected\t%s\n' "${cmd}" >> "${calls_file}"
        exit 99
        ;;
esac
EOF
    chmod +x "${bin_dir}/invarlock"
    export PATH="${bin_dir}:${PATH}"
    export TEST_QWEN14_SENTINEL_CALLS="${TEST_TMPDIR}/sentinel.calls"

    run bash -x ./scripts/evidence_packs/run_qwen14_sentinels.sh \
        --run-dir "${run_dir}" \
        --model-name "${model_name}" \
        --out "${TEST_TMPDIR}/sentinels"
    assert_rc "0" "${RUN_RC}" "sentinel script succeeds when reports are written"
    assert_file_exists "${TEST_TMPDIR}/sentinels/quant_4bit_clean/evaluation.report.json" "quant report produced"
    assert_file_exists "${TEST_TMPDIR}/sentinels/quant_4bit_clean/verify.json" "quant verify output captured"
    assert_file_exists "${TEST_TMPDIR}/sentinels/prune_clean/evaluation.report.json" "prune report produced"

    local calls
    calls="$(cat "${TEST_QWEN14_SENTINEL_CALLS}")"
    assert_match "quant_4bit_clean.*runtime_inputs/calibrated_preset_${model_name}__quant_rtn\\.yaml" "${calls}" "quant sentinel stages the quant-specific preset"
    assert_match "prune_clean.*runtime_inputs/calibrated_preset_${model_name}\\.yaml" "${calls}" "prune sentinel stages the base preset"
    assert_match "runtime_inputs/baseline_report\\.json" "${calls}" "sentinel stages the baseline report"
    assert_match "verify.*quant_4bit_clean/evaluation\\.report\\.json" "${calls}" "public quant verify runs"

    local quant_preset
    quant_preset="$(cat "${TEST_TMPDIR}/sentinels/quant_4bit_clean/observed_preset.yaml")"
    assert_match "seq_len: 1536" "${quant_preset}" "quant preset uses baseline seq_len"
    assert_match "stride: 1536" "${quant_preset}" "quant preset uses baseline stride"
    assert_match "preview_n: 48" "${quant_preset}" "quant preset uses baseline preview count"
    assert_match "final_n: 48" "${quant_preset}" "quant preset uses baseline final count"
    assert_match "skip_overhead_check: true" "${quant_preset}" "quant preset injects skip_overhead_check"
}

test_run_qwen14_sentinels_requires_inputs_and_rejects_bad_mode() {
    mock_reset

    run bash -x ./scripts/evidence_packs/run_qwen14_sentinels.sh --model-name qwen
    assert_rc "2" "${RUN_RC}" "missing run-dir returns usage error"

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}"
    run bash -x ./scripts/evidence_packs/run_qwen14_sentinels.sh --run-dir "${run_dir}" --model-name qwen --mode nope
    assert_rc "2" "${RUN_RC}" "invalid mode returns usage error"
}

test_run_qwen14_sentinels_evaluate_and_verify_warning_paths() {
    mock_reset

    source ./scripts/evidence_packs/run_qwen14_sentinels.sh

    local preset="${TEST_TMPDIR}/preset.yaml"
    local baseline_report="${TEST_TMPDIR}/baseline_report.json"
    local baseline_dir="${TEST_TMPDIR}/baseline"
    local subject_dir="${TEST_TMPDIR}/subject"
    mkdir -p "${baseline_dir}" "${subject_dir}"
    printf 'dataset:\n  seq_len: 128\n' > "${preset}"
    printf '%s\n' '{"evaluation_windows":{"preview":{"window_ids":[1],"input_ids":[[1]]},"final":{"window_ids":[1],"input_ids":[[1]]}}}' > "${baseline_report}"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cmd="${1:-}"
shift || true
case "${cmd}" in
    evaluate)
        report_out=""
        out=""
        while [[ $# -gt 0 ]]; do
            case "${1}" in
                --report-out) report_out="${2:-}"; shift 2 ;;
                --out) out="${2:-}"; shift 2 ;;
                *) shift ;;
            esac
        done
        mkdir -p "${report_out}"
        printf '{"ok":true}\n' > "${report_out}/evaluation.report.json"
        printf '{"report":"ok"}\n' > "${out}"
        exit 13
        ;;
    verify)
        printf '{"ok":true}\n'
        exit 17
        ;;
esac
exit 99
EOF
    chmod +x "${bin_dir}/invarlock"
    PATH="${bin_dir}:/bin:/usr/bin"
    export PATH
    normalize_staged_preset_for_baseline_report() { :; }

    local out_dir="${TEST_TMPDIR}/sentinel"
    run run_evaluate_sentinel "${baseline_dir}" "${baseline_report}" "${subject_dir}" "${preset}" "${out_dir}" "auto" "ci" "cpu"
    assert_rc "0" "${RUN_RC}" "evaluate sentinel treats non-zero exit with written report as success"
    assert_match "treating sentinel as load-path success" "${RUN_ERR}" "evaluate warning is surfaced"

    run run_public_quant_verify "${out_dir}/evaluation.report.json" "${out_dir}" "ci"
    assert_rc "0" "${RUN_RC}" "verify sentinel treats non-zero exit with written summary as success"
    assert_match "treating sentinel as load-path success" "${RUN_ERR}" "verify warning is surfaced"
}

test_run_qwen14_sentinels_verify_success_path_writes_summary_without_warning() {
    mock_reset

    source ./scripts/evidence_packs/run_qwen14_sentinels.sh

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '{"ok":true,"reason":"ok"}\n'
EOF
    chmod +x "${bin_dir}/invarlock"
    PATH="${bin_dir}:/bin:/usr/bin"
    export PATH

    local out_dir="${TEST_TMPDIR}/sentinel"
    mkdir -p "${out_dir}"
    printf '{"ok":true}\n' > "${out_dir}/evaluation.report.json"

    run run_public_quant_verify "${out_dir}/evaluation.report.json" "${out_dir}" "ci"
    assert_rc "0" "${RUN_RC}" "verify success path returns zero"
    assert_file_exists "${out_dir}/verify.json" "verify summary is written"
    if [[ "${RUN_ERR}" == *"treating sentinel as load-path success"* ]]; then
        fail_test "verify success path should not emit a warning"
    fi
}
