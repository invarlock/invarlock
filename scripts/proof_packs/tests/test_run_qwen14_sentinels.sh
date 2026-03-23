#!/usr/bin/env bash

test_run_qwen14_sentinels_runs_saved_model_and_public_quant_modes() {
    mock_reset

    local run_dir="${TEST_TMPDIR}/run"
    local model_name="qwen__qwen2.5-14b"
    local model_dir="${run_dir}/${model_name}"
    mkdir -p \
        "${model_dir}/models/baseline" \
        "${model_dir}/models/quant_4bit_clean" \
        "${model_dir}/models/prune_12pct_clean" \
        "${model_dir}/baseline_reports/ci_balanced_seq1536_pv48_fn48" \
        "${run_dir}/presets"
    printf '%s\n' "${model_dir}/models/baseline" > "${model_dir}/.baseline_path"
    printf '{"windows":"ok"}\n' > "${model_dir}/baseline_reports/ci_balanced_seq1536_pv48_fn48/baseline_report.json"
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

    run bash ./scripts/proof_packs/run_qwen14_sentinels.sh \
        --run-dir "${run_dir}" \
        --model-name "${model_name}" \
        --out "${TEST_TMPDIR}/sentinels"
    assert_rc "0" "${RUN_RC}" "sentinel script succeeds when reports are written"
    assert_file_exists "${TEST_TMPDIR}/sentinels/quant_4bit_clean/evaluation.report.json" "quant report produced"
    assert_file_exists "${TEST_TMPDIR}/sentinels/quant_4bit_clean/verify.json" "quant verify output captured"
    assert_file_exists "${TEST_TMPDIR}/sentinels/prune_12pct_clean/evaluation.report.json" "prune report produced"

    local calls
    calls="$(cat "${TEST_QWEN14_SENTINEL_CALLS}")"
    assert_match "quant_4bit_clean.*calibrated_preset_${model_name}__quant_rtn\\.yaml" "${calls}" "quant sentinel uses quant-specific preset"
    assert_match "prune_12pct_clean.*calibrated_preset_${model_name}\\.yaml" "${calls}" "prune sentinel falls back to base preset"
    assert_match "verify.*quant_4bit_clean/evaluation\\.report\\.json" "${calls}" "public quant verify runs"
}

test_run_qwen14_sentinels_requires_inputs_and_rejects_bad_mode() {
    mock_reset

    run bash ./scripts/proof_packs/run_qwen14_sentinels.sh --model-name qwen
    assert_rc "2" "${RUN_RC}" "missing run-dir returns usage error"

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}"
    run bash ./scripts/proof_packs/run_qwen14_sentinels.sh --run-dir "${run_dir}" --model-name qwen --mode nope
    assert_rc "2" "${RUN_RC}" "invalid mode returns usage error"
}
