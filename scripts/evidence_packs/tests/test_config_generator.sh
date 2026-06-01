#!/usr/bin/env bash

test_config_generator_run_single_calibration_large_model_emits_log_and_captures_report() {
    mock_reset

    # shellcheck source=../config_generator.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/config_generator.sh"

    INVARLOCK_DATASET="wikitext2"
    INVARLOCK_TIER="balanced"
    FLASH_ATTENTION_AVAILABLE="false"
    PACK_DETERMINISM="throughput"
    export INVARLOCK_DATASET INVARLOCK_TIER FLASH_ATTENTION_AVAILABLE PACK_DETERMINISM

    local run_calls="${TEST_TMPDIR}/config_runner.calls"
    _pack_run_from_config() {
        printf '%s\n' "$*" > "${run_calls}"
        local out=""
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
        return 0
    }
    _cmd_python() { return 0; }

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
    assert_match "--config" "$(cat "${run_calls}")" "calibration forwards config path through repo config runner"
    assert_match "--out" "$(cat "${run_calls}")" "calibration forwards output path through repo config runner"
    assert_match "skip_overhead_check: true" "$(cat "${run_dir}/calibration_config.yaml")" "calibration config carries skip_overhead policy"
}

test_config_generator_run_single_calibration_returns_runner_failure() {
    mock_reset

    # shellcheck source=../config_generator.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/config_generator.sh"

    INVARLOCK_DATASET="wikitext2"
    INVARLOCK_TIER="balanced"
    FLASH_ATTENTION_AVAILABLE="false"
    PACK_DETERMINISM="throughput"
    export INVARLOCK_DATASET INVARLOCK_TIER FLASH_ATTENTION_AVAILABLE PACK_DETERMINISM

    _pack_run_from_config() { return 7; }
    _cmd_python() { return 0; }
    estimate_model_params() { echo "7"; }

    local run_dir="${TEST_TMPDIR}/calibration_failed/run_1"
    local log_file="${TEST_TMPDIR}/calibration_failed.log"
    mkdir -p "$(dirname "${run_dir}")"
    : > "${log_file}"

    run run_single_calibration "${TEST_TMPDIR}/model" "${run_dir}" 42 2 2 10 "${log_file}" 0 128 128 1
    assert_rc "7" "${RUN_RC}" "calibration propagates config-runner failure"
}

test_config_generator_run_single_calibration_returns_report_conversion_failure() {
    mock_reset

    # shellcheck source=../config_generator.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/config_generator.sh"

    INVARLOCK_DATASET="wikitext2"
    INVARLOCK_TIER="balanced"
    FLASH_ATTENTION_AVAILABLE="false"
    PACK_DETERMINISM="throughput"
    export INVARLOCK_DATASET INVARLOCK_TIER FLASH_ATTENTION_AVAILABLE PACK_DETERMINISM

    _pack_run_from_config() {
        local out=""
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
        return 0
    }
    _cmd_python() { return 12; }
    estimate_model_params() { echo "7"; }

    local run_dir="${TEST_TMPDIR}/calibration_conversion_failed/run_1"
    local log_file="${TEST_TMPDIR}/calibration_conversion_failed.log"
    mkdir -p "$(dirname "${run_dir}")"
    : > "${log_file}"

    run run_single_calibration "${TEST_TMPDIR}/model" "${run_dir}" 42 2 2 10 "${log_file}" 0 128 128 1
    assert_rc "12" "${RUN_RC}" "calibration propagates report conversion failure"
    assert_match "failed to generate evaluation\\.report\\.json" "$(cat "${log_file}")" "conversion failure is logged"
}

test_config_generator_run_invarlock_calibration_logs_moe_and_all_runs_failed() {
    mock_reset

    # shellcheck source=../config_generator.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/config_generator.sh"

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

test_config_generator_run_invarlock_evaluate_preset_and_cert_copy_branches() {
    mock_reset

    # shellcheck source=../config_generator.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/config_generator.sh"

    INVARLOCK_TIER="balanced"
    export INVARLOCK_TIER

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
mode="canonical"
while [[ $# -gt 0 ]]; do
  case "${1}" in
    --report-out)
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
  echo '{}' > "${cert_out}/evaluation.report.json"
else
  mkdir -p "${cert_out}/nested"
  echo '{}' > "${cert_out}/nested/evaluation.report.json"
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
    run_invarlock_evaluate "${TEST_TMPDIR}/subject" "${TEST_TMPDIR}/baseline" "${out_dir}" "run_large" "${preset_dir}" "model" 0
    assert_file_exists "${out_dir}/run_large/evaluation.report.json" "canonical cert copied"

    # Alt cert lookup branch: remove canonical and provide nested evaluation.report.json.
    estimate_model_params() { echo "7"; }
    pack_run_cmd() { :; }
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
    --report-out)
      cert_out="${2:-}"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

mkdir -p "${cert_out}/nested"
echo '{}' > "${cert_out}/nested/evaluation.report.json"
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"

    run_invarlock_evaluate "${TEST_TMPDIR}/subject" "${TEST_TMPDIR}/baseline" "${out_dir}" "run_small" "${preset_dir}" "model" 0
    assert_file_exists "${out_dir}/run_small/evaluation.report.json" "nested cert copied"
}

test_config_generator_generate_invarlock_config_writes_to_stdout_when_requested() {
    mock_reset

    # shellcheck source=../config_generator.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/config_generator.sh"

    INVARLOCK_PREVIEW_WINDOWS="128"
    INVARLOCK_FINAL_WINDOWS="128"
    INVARLOCK_SEQ_LEN="256"
    INVARLOCK_STRIDE="128"
    INVARLOCK_EVAL_BATCH="1"
    INVARLOCK_DATASET="wikitext2"
    INVARLOCK_TIER="balanced"
    FLASH_ATTENTION_AVAILABLE="false"
    PACK_DETERMINISM="throughput"
    export \
        INVARLOCK_PREVIEW_WINDOWS \
        INVARLOCK_FINAL_WINDOWS \
        INVARLOCK_SEQ_LEN \
        INVARLOCK_STRIDE \
        INVARLOCK_EVAL_BATCH \
        INVARLOCK_DATASET \
        INVARLOCK_TIER \
        FLASH_ATTENTION_AVAILABLE \
        PACK_DETERMINISM

    local out
    out="$(generate_invarlock_config "demo/model" "/dev/stdout" "edit")"
    assert_match $'\nmodel:' "${out}" "config emitted to stdout"
    assert_match 'adapter:' "${out}" "config payload rendered"
    assert_match 'trust_remote_code: false' "${out}" "remote code disabled by default"

    INVARLOCK_ALLOW_REMOTE_CODE="1"
    export INVARLOCK_ALLOW_REMOTE_CODE
    out="$(generate_invarlock_config "demo/model" "/dev/stdout" "edit")"
    assert_match 'trust_remote_code: true' "${out}" "remote code emitted only with explicit allow"

    INVARLOCK_SKIP_OVERHEAD_CHECK="1"
    export INVARLOCK_SKIP_OVERHEAD_CHECK
    out="$(generate_invarlock_config "demo/model" "/dev/stdout" "edit")"
    assert_match 'skip_overhead_check: true' "${out}" "skip-overhead policy emitted when requested"
}

test_config_generator_run_single_calibration_exports_remote_code_allowance() {
    mock_reset

    # shellcheck source=../config_generator.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/config_generator.sh"

    INVARLOCK_DATASET="wikitext2"
    INVARLOCK_TIER="balanced"
    FLASH_ATTENTION_AVAILABLE="false"
    PACK_DETERMINISM="throughput"
    export INVARLOCK_DATASET INVARLOCK_TIER FLASH_ATTENTION_AVAILABLE PACK_DETERMINISM

    local env_log="${TEST_TMPDIR}/run.env"
    _pack_run_from_config() {
        printf '%s\n' "${INVARLOCK_ALLOW_REMOTE_CODE-}" > "${env_log}"
        local out=""
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
        return 0
    }
    _cmd_python() { return 0; }
    estimate_model_params() { echo "7"; }
    pack_remote_code_allowed() { return 0; }

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    export OUTPUT_DIR
    mkdir -p "${OUTPUT_DIR}/logs"

    local run_dir="${TEST_TMPDIR}/calibration/run_1"
    local log_file="${TEST_TMPDIR}/calibration.log"
    mkdir -p "$(dirname "${run_dir}")"
    : > "${log_file}"

    run_single_calibration "${TEST_TMPDIR}/model" "${run_dir}" 42 2 2 10 "${log_file}" 0 128 128 1
    assert_eq "1" "$(cat "${env_log}")" "remote code allowance exported into calibration runner env"
}
