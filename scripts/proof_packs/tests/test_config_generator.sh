#!/usr/bin/env bash

test_config_generator_run_single_calibration_large_model_emits_log_and_captures_report() {
    mock_reset

    # shellcheck source=../config_generator.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/config_generator.sh"

    INVARLOCK_DATASET="wikitext2"
    INVARLOCK_TIER="balanced"
    FLASH_ATTENTION_AVAILABLE="false"
    PACK_DETERMINISM="throughput"
    export INVARLOCK_DATASET INVARLOCK_TIER FLASH_ATTENTION_AVAILABLE PACK_DETERMINISM

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"

    cat > "${bin_dir}/python3" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exit 0
EOF
    chmod +x "${bin_dir}/python3"

    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cmd="${1:-}"
shift || true
case "${cmd}" in
    run)
        out=""
        while [[ $# -gt 0 ]]; do
            case "${1}" in
                --out)
                    out="${2:-}"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done
        mkdir -p "${out}"
        echo '{}' > "${out}/report.json"
        exit 0
        ;;
    certify)
        cert_out=""
        while [[ $# -gt 0 ]]; do
            case "${1}" in
                --cert-out)
                    cert_out="${2:-}"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done
        mkdir -p "${cert_out}"
        echo '{}' > "${cert_out}/evaluation.cert.json"
        exit 0
        ;;
esac
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"

    PATH="${bin_dir}:${PATH}"
    export PATH

    estimate_model_params() { echo "70"; }

    local out="${TEST_TMPDIR}/out"
    mkdir -p "${out}/logs"
    OUTPUT_DIR="${out}"
    export OUTPUT_DIR

    local run_dir="${TEST_TMPDIR}/calibration/run_1"
    local log_file="${TEST_TMPDIR}/calibration.log"
    mkdir -p "$(dirname "${run_dir}")"
    : > "${log_file}"

    run_single_calibration "${TEST_TMPDIR}/model" "${run_dir}" 42 2 2 10 "${log_file}" 0 128 128 1

    assert_match "Large model \\(70\\)" "$(cat "${log_file}")" "large model branch logged"
    assert_file_exists "${run_dir}/baseline_report.json" "baseline report copied"
}

test_config_generator_run_invarlock_calibration_logs_moe_and_all_runs_failed() {
    mock_reset

    # shellcheck source=../config_generator.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/config_generator.sh"

    INVARLOCK_DATASET="wikitext2"
    INVARLOCK_TIER="balanced"
    export INVARLOCK_DATASET INVARLOCK_TIER

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/python3" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exit 0
EOF
    chmod +x "${bin_dir}/python3"
    PATH="${bin_dir}:${PATH}"
    export PATH

    log() { echo "$*" >> "${TEST_TMPDIR}/log.txt"; }

    get_model_invarlock_config() { echo "128:128:1:1:1"; }

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    export OUTPUT_DIR
    mkdir -p "${OUTPUT_DIR}/logs"

    estimate_model_params() { echo "moe"; }
    run_single_calibration() { return 1; }

    run run_invarlock_calibration "${TEST_TMPDIR}/model" "m" "${TEST_TMPDIR}/calibration" 1 "${TEST_TMPDIR}/presets" 0
    assert_rc "1" "${RUN_RC}" "all calibration runs failed returns non-zero"
    assert_match "MoE architecture" "$(cat "${TEST_TMPDIR}/log.txt")" "moe log branch"

    : > "${TEST_TMPDIR}/log.txt"
    estimate_model_params() { echo "7"; }
    run_single_calibration() { return 0; }
    run run_invarlock_calibration "${TEST_TMPDIR}/model" "m" "${TEST_TMPDIR}/calibration_ok" 1 "${TEST_TMPDIR}/presets" 0
    assert_rc "0" "${RUN_RC}" "successful calibration returns zero"
    assert_match "\\(7B params\\)" "$(cat "${TEST_TMPDIR}/log.txt")" "non-moe log branch"
}

test_config_generator_run_invarlock_certify_preset_and_cert_copy_branches() {
    mock_reset

    # shellcheck source=../config_generator.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/config_generator.sh"

    INVARLOCK_TIER="balanced"
    export INVARLOCK_TIER

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"

    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cmd="${1:-}"
shift || true
if [[ "${cmd}" != "certify" ]]; then
  exit 0
fi

cert_out=""
mode="canonical"
while [[ $# -gt 0 ]]; do
  case "${1}" in
    --cert-out)
      cert_out="${2:-}"
      shift 2
      ;;
    --preset)
      # Presence indicates calibrated_preset path was detected.
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

mkdir -p "${cert_out}"
if [[ "${mode}" == "canonical" ]]; then
  echo '{}' > "${cert_out}/evaluation.cert.json"
else
  mkdir -p "${cert_out}/nested"
  echo '{}' > "${cert_out}/nested/evaluation.cert.json"
fi
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"
    PATH="${bin_dir}:${PATH}"
    export PATH

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    export OUTPUT_DIR
    mkdir -p "${OUTPUT_DIR}/logs"

    local preset_dir="${TEST_TMPDIR}/presets"
    mkdir -p "${preset_dir}"
    echo '{}' > "${preset_dir}/calibrated_preset_model.yaml"

    estimate_model_params() { echo "70"; }
    local out_dir="${TEST_TMPDIR}/certs"
    mkdir -p "${out_dir}"
    run_invarlock_certify "${TEST_TMPDIR}/subject" "${TEST_TMPDIR}/baseline" "${out_dir}" "run_large" "${preset_dir}" "model" 0
    assert_file_exists "${out_dir}/run_large/evaluation.cert.json" "canonical cert copied"

    # Alt cert lookup branch: remove canonical and provide nested evaluation.cert.json.
    estimate_model_params() { echo "7"; }
    pack_run_cmd() { :; }
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cmd="${1:-}"
shift || true
if [[ "${cmd}" != "certify" ]]; then
  exit 0
fi

cert_out=""
while [[ $# -gt 0 ]]; do
  case "${1}" in
    --cert-out)
      cert_out="${2:-}"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

mkdir -p "${cert_out}/nested"
echo '{}' > "${cert_out}/nested/evaluation.cert.json"
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"

    run_invarlock_certify "${TEST_TMPDIR}/subject" "${TEST_TMPDIR}/baseline" "${out_dir}" "run_small" "${preset_dir}" "model" 0
    assert_file_exists "${out_dir}/run_small/evaluation.cert.json" "nested cert copied"
}
