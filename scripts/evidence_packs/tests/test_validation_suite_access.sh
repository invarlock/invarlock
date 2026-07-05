#!/usr/bin/env bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/validation_suite_test_helpers.sh"

test_pack_apply_network_mode_sets_env_flags() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    PACK_NET="0"
    pack_apply_network_mode "1"
    assert_eq "1" "${PACK_NET}" "net mode sets PACK_NET"
    assert_eq "1" "${INVARLOCK_ALLOW_NETWORK}" "network allowed"
    assert_eq "0" "${HF_DATASETS_OFFLINE}" "datasets online"
    assert_eq "0" "${TRANSFORMERS_OFFLINE}" "transformers online"
    assert_eq "0" "${HF_HUB_OFFLINE}" "hub online"

    pack_apply_network_mode "0"
    assert_eq "0" "${PACK_NET}" "offline mode sets PACK_NET"
    assert_eq "0" "${INVARLOCK_ALLOW_NETWORK}" "network disabled"
    assert_eq "1" "${HF_DATASETS_OFFLINE}" "datasets offline"
    assert_eq "1" "${TRANSFORMERS_OFFLINE}" "transformers offline"
    assert_eq "1" "${HF_HUB_OFFLINE}" "hub offline"
}

test_pack_configure_hf_access_noop_when_offline() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    PACK_NET="0"
    unset HF_ENDPOINT HF_HUB_TIMEOUT HF_HUB_ETAG_TIMEOUT HF_HUB_DOWNLOAD_TIMEOUT HF_HUB_MAX_RETRIES
    unset HF_PRIMARY_ENDPOINT HF_MIRROR_ENDPOINT

    pack_configure_hf_access
    assert_eq "" "${HF_ENDPOINT:-}" "offline mode does not set HF_ENDPOINT"
    assert_eq "" "${HF_HUB_TIMEOUT:-}" "offline mode does not set HF_HUB_TIMEOUT"
    assert_eq "" "${HF_HUB_ETAG_TIMEOUT:-}" "offline mode does not set HF_HUB_ETAG_TIMEOUT"
    assert_eq "" "${HF_HUB_DOWNLOAD_TIMEOUT:-}" "offline mode does not set HF_HUB_DOWNLOAD_TIMEOUT"
    assert_eq "" "${HF_HUB_MAX_RETRIES:-}" "offline mode does not set HF_HUB_MAX_RETRIES"
}

test_pack_configure_hf_access_sets_timeouts_and_chooses_mirror_when_primary_fails() {
    mock_reset

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

url="${!#}"
echo "${url}" >> "${TEST_TMPDIR}/curl.calls"

if [[ "${url}" == "https://huggingface.co/api/whoami-v2" ]]; then
    exit 1
fi

if [[ "${url}" == "https://hf-mirror.com/api/whoami-v2" ]]; then
    exit 0
fi

exit 1
EOF
    chmod +x "${bin_dir}/curl"
    export PATH="${bin_dir}:$PATH"
    hash -r 2>/dev/null || true

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    PACK_NET="1"
    unset HF_ENDPOINT HF_HUB_TIMEOUT HF_HUB_ETAG_TIMEOUT HF_HUB_DOWNLOAD_TIMEOUT HF_HUB_MAX_RETRIES
    unset HF_PRIMARY_ENDPOINT HF_MIRROR_ENDPOINT HF_ENDPOINT_TEST_PATH
    export HF_ENDPOINT_TEST_PATH="/api/whoami-v2"

    pack_configure_hf_access
    assert_eq "60" "${HF_HUB_TIMEOUT}" "HF_HUB_TIMEOUT default set"
    assert_eq "60" "${HF_HUB_ETAG_TIMEOUT}" "HF_HUB_ETAG_TIMEOUT default set"
    assert_eq "300" "${HF_HUB_DOWNLOAD_TIMEOUT}" "HF_HUB_DOWNLOAD_TIMEOUT default set"
    assert_eq "10" "${HF_HUB_MAX_RETRIES}" "HF_HUB_MAX_RETRIES default set"
    assert_eq "https://hf-mirror.com" "${HF_ENDPOINT}" "mirror endpoint chosen when primary fails"

    assert_file_exists "${TEST_TMPDIR}/curl.calls" "curl invoked for endpoint probe"
}

test_pack_configure_hf_access_chooses_primary_when_primary_succeeds() {
    mock_reset

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

url="${!#}"
echo "${url}" >> "${TEST_TMPDIR}/curl.calls"

if [[ "${url}" == "https://huggingface.co/api/whoami-v2" ]]; then
    exit 0
fi

exit 1
EOF
    chmod +x "${bin_dir}/curl"
    export PATH="${bin_dir}:$PATH"
    hash -r 2>/dev/null || true

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    PACK_NET="1"
    unset HF_ENDPOINT HF_PRIMARY_ENDPOINT HF_MIRROR_ENDPOINT HF_ENDPOINT_TEST_PATH
    export HF_ENDPOINT_TEST_PATH="/api/whoami-v2"

    pack_configure_hf_access
    assert_eq "https://huggingface.co" "${HF_ENDPOINT}" "primary endpoint chosen when probe succeeds"
    assert_file_exists "${TEST_TMPDIR}/curl.calls" "curl invoked for endpoint probe"
}

test_pack_configure_hf_access_falls_back_to_primary_when_both_endpoints_fail() {
    mock_reset

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

url="${!#}"
echo "${url}" >> "${TEST_TMPDIR}/curl.calls"
exit 1
EOF
    chmod +x "${bin_dir}/curl"
    export PATH="${bin_dir}:$PATH"
    hash -r 2>/dev/null || true

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    PACK_NET="1"
    unset HF_ENDPOINT HF_PRIMARY_ENDPOINT HF_MIRROR_ENDPOINT HF_ENDPOINT_TEST_PATH
    export HF_ENDPOINT_TEST_PATH="/api/whoami-v2"

    pack_configure_hf_access
    assert_eq "https://huggingface.co" "${HF_ENDPOINT}" "defaults to primary when both probes fail"
    assert_file_exists "${TEST_TMPDIR}/curl.calls" "curl invoked for endpoint probe"
}

test_pack_configure_hf_access_falls_back_to_primary_when_curl_missing() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    PACK_NET="1"
    unset HF_ENDPOINT HF_PRIMARY_ENDPOINT HF_MIRROR_ENDPOINT HF_ENDPOINT_TEST_PATH
    export HF_ENDPOINT_TEST_PATH="/api/whoami-v2"

    local empty_bin="${TEST_TMPDIR}/emptybin"
    mkdir -p "${empty_bin}"
    export PATH="${empty_bin}"
    hash -r 2>/dev/null || true

    pack_configure_hf_access
    assert_eq "https://huggingface.co" "${HF_ENDPOINT}" "defaults to primary when curl is missing"
}

test_pack_configure_hf_access_respects_existing_hf_endpoint() {
    mock_reset

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "unexpected curl call" >> "${TEST_TMPDIR}/curl.calls"
exit 0
EOF
    chmod +x "${bin_dir}/curl"
    export PATH="${bin_dir}:$PATH"
    hash -r 2>/dev/null || true

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    PACK_NET="1"
    HF_ENDPOINT="https://example.invalid"
    export HF_ENDPOINT
    rm -f "${TEST_TMPDIR}/curl.calls"

    pack_configure_hf_access
    assert_eq "https://example.invalid" "${HF_ENDPOINT}" "existing HF_ENDPOINT preserved"
    if [[ -f "${TEST_TMPDIR}/curl.calls" ]]; then
        t_fail "curl should not be invoked when HF_ENDPOINT is already set"
    fi
}

test_pack_prepare_tuned_edit_params_resolves_default_from_scripts_dir_and_copies_into_state() {
    mock_reset

    local fake_repo
    fake_repo="$(mktemp -d "${TEST_TMPDIR}/fake_repo.XXXXXX")"
    mkdir -p "${fake_repo}/scripts/evidence_packs/lib"
    cp -R "${TEST_ROOT}/scripts/evidence_packs/lib/." "${fake_repo}/scripts/evidence_packs/lib/"
    mkdir -p "${fake_repo}/scripts/evidence_packs/python"
    mkdir -p "${fake_repo}/scripts/evidence_packs"

    cat > "${fake_repo}/scripts/evidence_packs/tuned_edit_params.json" <<'JSON'
{
  "_meta": {"schema": "tuned_edit_params_v1"},
  "models": {}
}
JSON

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    CLEAN_EDIT_RUNS="1"
    unset PACK_TUNED_EDIT_PARAMS_FILE

    # shellcheck source=../lib/validation/validation_suite.sh
    source "${fake_repo}/scripts/evidence_packs/lib/validation/validation_suite.sh"
    pack_setup_output_dirs

    pack_prepare_tuned_edit_params

    assert_file_exists "${OUTPUT_DIR}/state/tuned_edit_params.json" "tuned file copied into run state"
    assert_match "\"tuned_edit_params_v1\"" "$(cat "${OUTPUT_DIR}/state/tuned_edit_params.json")" "copied content preserved"
    assert_eq "${OUTPUT_DIR}/state/tuned_edit_params.json" "${PACK_TUNED_EDIT_PARAMS_FILE}" "env updated to copied path"
}

test_pack_resolve_tuned_edit_params_file_returns_early_when_env_set() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    PACK_TUNED_EDIT_PARAMS_FILE="/tmp/already_set.json"
    export PACK_TUNED_EDIT_PARAMS_FILE
    run pack_resolve_tuned_edit_params_file
    assert_rc "0" "${RUN_RC}" "returns zero when PACK_TUNED_EDIT_PARAMS_FILE already set"
    assert_eq "/tmp/already_set.json" "${PACK_TUNED_EDIT_PARAMS_FILE}" "env preserved"
}

test_pack_prepare_tuned_edit_params_skips_when_clean_edit_runs_zero() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    CLEAN_EDIT_RUNS="0"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    run pack_prepare_tuned_edit_params
    assert_rc "0" "${RUN_RC}" "clean presets skipped when CLEAN_EDIT_RUNS=0"
}

test_pack_prepare_tuned_edit_params_uses_repo_root_override_and_copies_to_state() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    CLEAN_EDIT_RUNS="1"
    unset PACK_TUNED_EDIT_PARAMS_FILE

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    local fake_root="${TEST_TMPDIR}/fake_root"
    mkdir -p "${fake_root}/scripts/evidence_packs/lib" "${fake_root}/scripts/evidence_packs"
    cat > "${fake_root}/scripts/evidence_packs/tuned_edit_params.json" <<'JSON'
{"defaults":{"quant_rtn":{"status":"selected"}},"models":{}}
JSON

    _PACK_VALIDATION_LIB_DIR="${fake_root}/scripts/evidence_packs/lib"
    pack_prepare_tuned_edit_params

    assert_file_exists "${OUTPUT_DIR}/state/tuned_edit_params.json" "tuned file copied into run state"
    assert_eq "${OUTPUT_DIR}/state/tuned_edit_params.json" "${PACK_TUNED_EDIT_PARAMS_FILE}" "env updated to copied path"
}

test_pack_prepare_tuned_edit_params_errors_when_missing_preset_file() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    CLEAN_EDIT_RUNS="1"
    unset PACK_TUNED_EDIT_PARAMS_FILE

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    local fake_root="${TEST_TMPDIR}/fake_root_missing"
    mkdir -p "${fake_root}/scripts/evidence_packs/lib" "${fake_root}/scripts/evidence_packs"
    _PACK_VALIDATION_LIB_DIR="${fake_root}/scripts/evidence_packs/lib"

    local rc=0
    ( pack_prepare_tuned_edit_params ) || rc=$?
    assert_ne "0" "${rc}" "missing tuned preset file triggers failure"
    assert_match "Missing PACK_TUNED_EDIT_PARAMS_FILE" "$(cat "${OUTPUT_DIR}/logs/main.log")" "error logged"
}

test_pack_prepare_tuned_edit_params_errors_when_file_missing() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    CLEAN_EDIT_RUNS="1"
    PACK_TUNED_EDIT_PARAMS_FILE="${TEST_TMPDIR}/does_not_exist.json"
    export PACK_TUNED_EDIT_PARAMS_FILE

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    local rc=0
    ( pack_prepare_tuned_edit_params ) || rc=$?
    assert_ne "0" "${rc}" "missing tuned preset file triggers error_exit"
    assert_match "Tuned edit preset file not found" "$(cat "${OUTPUT_DIR}/logs/main.log")" "error logged"
}

test_pack_validate_tuned_edit_params_skips_when_clean_edit_runs_zero() {
    mock_reset

    CLEAN_EDIT_RUNS="0"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    run pack_validate_tuned_edit_params
    assert_rc "0" "${RUN_RC}" "validation skipped when CLEAN_EDIT_RUNS=0"
}

test_pack_validate_tuned_edit_params_builds_model_names_csv_and_succeeds() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    CLEAN_EDIT_RUNS="1"
    EDIT_TYPES_CLEAN=("quant_rtn:clean:ffn")
    PACK_MODEL_LIST=("org/model1" "org/model2")

    PACK_TUNED_EDIT_PARAMS_FILE="${TEST_TMPDIR}/tuned.json"
    export PACK_TUNED_EDIT_PARAMS_FILE
    cat > "${PACK_TUNED_EDIT_PARAMS_FILE}" <<'JSON'
{"defaults":{"quant_rtn":{"status":"selected"}},"models":{}}
JSON

    run pack_validate_tuned_edit_params
    assert_rc "0" "${RUN_RC}" "tuned edit params validated"
}

test_pack_validate_tuned_edit_params_uses_presets_canonical_fallback() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    CLEAN_EDIT_RUNS="1"
    EDIT_TYPES_CLEAN=("quant_rtn:clean:ffn")
    PACK_MODEL_LIST=("org/model1")

    local fake_root="${TEST_TMPDIR}/fake-root"
    mkdir -p \
        "${fake_root}/scripts/evidence_packs/lib" \
        "${fake_root}/scripts/evidence_packs/python" \
        "${fake_root}/scripts/evidence_packs/presets"
    cat > "${fake_root}/scripts/evidence_packs/python/validation_state.py" <<'PY'
import sys
sys.exit(0)
PY
    printf '{}\n' > "${fake_root}/scripts/evidence_packs/presets/tuned_edit_params.json"

    PACK_TUNED_EDIT_PARAMS_FILE="${TEST_TMPDIR}/tuned.json"
    export PACK_TUNED_EDIT_PARAMS_FILE
    printf '{"defaults":{"quant_rtn":{"status":"selected"}},"models":{}}\n' > "${PACK_TUNED_EDIT_PARAMS_FILE}"
    _PACK_VALIDATION_LIB_DIR="${fake_root}/scripts/evidence_packs/lib"

    run pack_validate_tuned_edit_params
    assert_rc "0" "${RUN_RC}" "presets/tuned_edit_params.json is accepted as canonical fallback"
}

test_pack_validate_tuned_edit_params_rejects_noncanonical_selected_entries() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    CLEAN_EDIT_RUNS="1"
    EDIT_TYPES_CLEAN=("lowrank_svd:clean:ffn")
    PACK_MODEL_LIST=("Qwen/Qwen3-8B")

    PACK_TUNED_EDIT_PARAMS_FILE="${TEST_TMPDIR}/tuned.json"
    export PACK_TUNED_EDIT_PARAMS_FILE
    cat > "${PACK_TUNED_EDIT_PARAMS_FILE}" <<'JSON'
{
  "models": {
    "Qwen/Qwen3-8B": {
      "lowrank_svd": {
        "edit_dir_name": "svd_rank32_clean",
        "rank": 32,
        "reason": "trial_on_h100:qwen3_8b_rank32_ffn_layer17",
        "scope": "ffn@layer=17",
        "status": "selected"
      }
    }
  }
}
JSON

    run pack_validate_tuned_edit_params
    assert_ne "0" "${RUN_RC}" "noncanonical tuned params rejected"
    assert_match "Noncanonical tuned edit presets" "${RUN_ERR}" "error labels canonical mismatch"
    assert_match "Qwen/Qwen3-8B:lowrank_svd" "${RUN_ERR}" "error names mismatched model"
}

test_pack_validate_tuned_edit_params_allows_noncanonical_override_when_opted_in() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    CLEAN_EDIT_RUNS="1"
    EDIT_TYPES_CLEAN=("lowrank_svd:clean:ffn")
    PACK_MODEL_LIST=("Qwen/Qwen3-8B")
    PACK_ALLOW_NONCANONICAL_TUNED_EDIT_PARAMS="1"
    export PACK_ALLOW_NONCANONICAL_TUNED_EDIT_PARAMS

    PACK_TUNED_EDIT_PARAMS_FILE="${TEST_TMPDIR}/tuned.json"
    export PACK_TUNED_EDIT_PARAMS_FILE
    cat > "${PACK_TUNED_EDIT_PARAMS_FILE}" <<'JSON'
{
  "models": {
    "Qwen/Qwen3-8B": {
      "lowrank_svd": {
        "edit_dir_name": "svd_rank32_clean",
        "rank": 32,
        "reason": "trial_on_h100:qwen3_8b_rank32_ffn_layer17",
        "scope": "ffn@layer=17",
        "status": "selected"
      }
    }
  }
}
JSON

    run pack_validate_tuned_edit_params
    assert_rc "0" "${RUN_RC}" "explicit noncanonical override succeeds"
    unset PACK_ALLOW_NONCANONICAL_TUNED_EDIT_PARAMS
}

test_pack_validate_tuned_edit_params_returns_nonzero_when_python_fails() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    CLEAN_EDIT_RUNS="1"
    EDIT_TYPES_CLEAN=("quant_rtn:clean:ffn")
    PACK_MODEL_LIST=("org/model1")

    PACK_TUNED_EDIT_PARAMS_FILE="${TEST_TMPDIR}/tuned.json"
    export PACK_TUNED_EDIT_PARAMS_FILE
    echo '{"defaults":{"quant_rtn":{"status":"selected"}},"models":{}}' > "${PACK_TUNED_EDIT_PARAMS_FILE}"

    mock_python3_stub_enable
    fixture_write "python3.rc" "1"

    local rc=0
    if pack_validate_tuned_edit_params; then
        rc=0
    else
        rc=$?
    fi
    assert_ne "0" "${rc}" "python failure returns non-zero"
}

test_pack_validate_runtime_provenance_propagates_python_failure() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    mock_python3_stub_enable
    fixture_write "python3.rc" "1"

    run pack_validate_runtime_provenance
    assert_rc "1" "${RUN_RC}" "runtime provenance validation propagates helper failure"
}

test_pack_prepare_calibration_presets_skips_when_no_preset_dir_or_file() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    PACK_MODEL_LIST=("org/model")
    unset PACK_CALIBRATION_PRESET_DIR PACK_CALIBRATION_PRESET_FILE

    run pack_prepare_calibration_presets
    assert_rc "0" "${RUN_RC}" "calibration presets skipped when unset"
}

test_pack_prepare_calibration_presets_errors_when_preset_file_missing() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    PACK_MODEL_LIST=("org/model")
    PACK_CALIBRATION_PRESET_FILE="${TEST_TMPDIR}/missing_preset.yaml"
    export PACK_CALIBRATION_PRESET_FILE

    local rc=0
    ( pack_prepare_calibration_presets ) || rc=$?
    assert_ne "0" "${rc}" "missing PACK_CALIBRATION_PRESET_FILE triggers error_exit"
    assert_match "Calibration preset file not found" "$(cat "${OUTPUT_DIR}/logs/main.log")" "error logged"
}

test_pack_prepare_calibration_presets_uses_preset_file_for_all_models() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    PACK_MODEL_LIST=("org/model")
    local preset_file="${TEST_TMPDIR}/preset.yaml"
    echo "guards: {}" > "${preset_file}"
    PACK_CALIBRATION_PRESET_FILE="${preset_file}"
    export PACK_CALIBRATION_PRESET_FILE
    unset PACK_CALIBRATION_PRESET_DIR

    pack_prepare_calibration_presets

    assert_file_exists "${OUTPUT_DIR}/presets/calibrated_preset_org__model.yaml" "preset copied to per-model output dir"
    assert_eq "true" "${PACK_PRESET_READY}" "PACK_PRESET_READY set"
    assert_eq "0" "${DRIFT_CALIBRATION_RUNS}" "DRIFT_CALIBRATION_RUNS disabled when presets reused"
}

test_pack_prepare_calibration_presets_uses_preset_dir_candidates() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    PACK_MODEL_LIST=("org/model")
    local preset_dir="${TEST_TMPDIR}/preset_dir"
    mkdir -p "${preset_dir}"
    echo "guards: {}" > "${preset_dir}/calibrated_preset_org__model.yaml"
    PACK_CALIBRATION_PRESET_DIR="${preset_dir}"
    export PACK_CALIBRATION_PRESET_DIR
    unset PACK_CALIBRATION_PRESET_FILE

    pack_prepare_calibration_presets

    assert_file_exists "${OUTPUT_DIR}/presets/calibrated_preset_org__model.yaml" "preset copied from dir candidate"
    assert_eq "true" "${PACK_PRESET_READY}" "PACK_PRESET_READY set"
    assert_eq "0" "${DRIFT_CALIBRATION_RUNS}" "DRIFT_CALIBRATION_RUNS disabled when presets reused"
}

test_pack_prepare_calibration_presets_copies_edit_type_presets_from_dir() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    PACK_MODEL_LIST=("org/model")
    local preset_dir="${TEST_TMPDIR}/preset_dir"
    mkdir -p "${preset_dir}"
    echo "dataset: {seq_len: 512}" > "${preset_dir}/calibrated_preset_org__model__quant_rtn.yaml"
    echo '{"dataset":{"seq_len":512}}' > "${preset_dir}/calibrated_preset_org__model__magnitude_prune.json"
    PACK_CALIBRATION_PRESET_DIR="${preset_dir}"
    export PACK_CALIBRATION_PRESET_DIR
    unset PACK_CALIBRATION_PRESET_FILE

    pack_prepare_calibration_presets

    assert_file_exists "${OUTPUT_DIR}/presets/calibrated_preset_org__model__quant_rtn.yaml" "edit-type yaml preset copied from dir"
    assert_file_exists "${OUTPUT_DIR}/presets/calibrated_preset_org__model__magnitude_prune.json" "edit-type json preset copied from dir"
    assert_eq "true" "${PACK_PRESET_READY}" "PACK_PRESET_READY set"
    assert_eq "0" "${DRIFT_CALIBRATION_RUNS}" "DRIFT_CALIBRATION_RUNS disabled when presets reused"
}

test_pack_prepare_calibration_presets_errors_when_candidate_missing() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    PACK_MODEL_LIST=("org/model")
    local preset_dir="${TEST_TMPDIR}/preset_dir"
    mkdir -p "${preset_dir}"
    PACK_CALIBRATION_PRESET_DIR="${preset_dir}"
    export PACK_CALIBRATION_PRESET_DIR
    unset PACK_CALIBRATION_PRESET_FILE

    local rc=0
    ( pack_prepare_calibration_presets ) || rc=$?
    assert_ne "0" "${rc}" "missing candidate preset triggers error_exit"
    assert_match "Missing calibration preset" "$(cat "${OUTPUT_DIR}/logs/main.log")" "error logged"
}

test_pack_validate_guard_calibration_sanitizes_non_numeric_runs() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    DRIFT_CALIBRATION_RUNS="not-a-number"
    unset PACK_CALIBRATION_PRESET_DIR PACK_CALIBRATION_PRESET_FILE

    run pack_validate_guard_calibration
    assert_rc "0" "${RUN_RC}" "non-numeric DRIFT_CALIBRATION_RUNS coerces to default"
}
