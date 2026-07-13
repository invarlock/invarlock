#!/usr/bin/env bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pack_manifest_test_helpers.sh"

test_verify_pack_baseline_material_helper_forwards_assurance() {
    mock_reset
    source ./scripts/evidence_packs/verify_pack.sh

    local calls="${TEST_TMPDIR}/baseline-material.calls"
    _cmd_python() {
        printf '%s\n' "$*" > "${calls}"
    }

    run pack_verify_baseline_materials "${TEST_TMPDIR}/pack" strict

    assert_rc "0" "${RUN_RC}" "baseline material helper succeeds"
    assert_match "baseline-materials ${TEST_TMPDIR}/pack --report-assurance strict" "$(cat "${calls}")" "baseline assurance is forwarded"
}

test_verify_pack_strict_rejects_missing_signed_baseline_material() {
    mock_reset
    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    printf '%s\n' \
        '{"format":"evidence-pack-v1","verification":{"report_assurance":"strict"}}' \
        > "${pack_dir}/manifest.json"

    run pack_verify_baseline_materials "${pack_dir}" strict

    assert_rc "1" "${RUN_RC}" "strict baseline declaration is mandatory"
    assert_match "requires signed verification_baselines" "${RUN_ERR}" "missing baseline declaration is explicit"
}

test_verify_pack_final_verdict_binding_matches_report() {
    mock_reset
    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    local report_rel="reports/report-001/evaluation.report.json"
    mkdir -p "${pack_dir}/results" "${pack_dir}/reports/report-001"
    printf '%s\n' '{"run_id":"run-1","ok":true}' > "${pack_dir}/${report_rel}"
    local report_sha
    report_sha="$(pack_file_sha256 "${pack_dir}" "${report_rel}")"
    printf '%s\n' \
        "{\"verdict\":\"PASS\",\"report_sha256\":\"${report_sha}\",\"run_id\":\"run-1\"}" \
        > "${pack_dir}/results/final_verdict.json"

    run pack_verify_final_verdict_binding "${pack_dir}" 1
    assert_rc "0" "${RUN_RC}" "matching final-verdict report binding passes"
}

test_verify_pack_final_verdict_binding_rejects_stale_hash() {
    mock_reset
    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/results" "${pack_dir}/reports/report-001"
    printf '%s\n' '{"run_id":"run-1","ok":true}' \
        > "${pack_dir}/reports/report-001/evaluation.report.json"
    printf '%s\n' \
        "{\"verdict\":\"PASS\",\"report_sha256\":\"$(printf '0%.0s' {1..64})\",\"run_id\":\"run-1\"}" \
        > "${pack_dir}/results/final_verdict.json"

    run pack_verify_final_verdict_binding "${pack_dir}" 1
    assert_rc "1" "${RUN_RC}" "stale final-verdict report hash fails"
    assert_match "report_sha256 does not match" "${RUN_ERR}" "hash mismatch is explicit"
}

test_verify_pack_rejects_report_symlink_before_checksum_read() {
    mock_reset
    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    local outside_dir="${TEST_TMPDIR}/outside"
    mkdir -p "${pack_dir}/results" "${pack_dir}/reports/report-001" "${outside_dir}"
    printf '%s\n' '{"run_id":"outside","ok":true}' \
        > "${outside_dir}/evaluation.report.json"
    ln -s "${outside_dir}/evaluation.report.json" \
        "${pack_dir}/reports/report-001/evaluation.report.json"
    printf '%s\n' \
        "{\"verdict\":\"PASS\",\"report_sha256\":\"$(printf '0%.0s' {1..64})\"}" \
        > "${pack_dir}/results/final_verdict.json"
    printf '%s\n' '{}' > "${pack_dir}/manifest.json"
    : > "${pack_dir}/checksums.sha256"

    pack_validate_manifest_schema() { return 0; }
    pack_verify_signature() { return 0; }
    pack_verify_manifest_binds_checksums() { return 0; }
    pack_verify_checksums() {
        touch "${TEST_TMPDIR}/checksum-called"
        return 0
    }

    run pack_verify_pack --pack "${pack_dir}" --skip-verify --strict
    assert_rc "${PACK_VERIFY_INTEGRITY}" "${RUN_RC}" "report symlink is an integrity failure"
    assert_match "must not contain symlinks" "${RUN_ERR}" "symlink rejection is explicit"
    [[ ! -e "${TEST_TMPDIR}/checksum-called" ]] \
        || t_fail "checksum verifier must not run before report symlink rejection"
}

test_verify_pack_strict_rejects_extra_files() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"
    echo "extra" > "${pack_dir}/extra.txt"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} payload.txt > checksums.sha256
    )

    local checksums_digest
    checksums_digest="$(cd "${pack_dir}" && python3 -c 'import hashlib;print(hashlib.sha256(open("checksums.sha256","rb").read()).hexdigest())' < /dev/null)"
    printf '%s\n' "{\"format\":\"evidence-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    pack_test_sign_manifest "${pack_dir}"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify --strict
    assert_rc "6" "${RUN_RC}" "strict mode rejects extra files"
}

test_verify_pack_warns_on_extra_files_in_non_strict_mode() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"
    echo "extra" > "${pack_dir}/extra.txt"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} payload.txt > checksums.sha256
    )

    local checksums_digest
    checksums_digest="$(cd "${pack_dir}" && python3 -c 'import hashlib;print(hashlib.sha256(open("checksums.sha256","rb").read()).hexdigest())' < /dev/null)"
    printf '%s\n' "{\"format\":\"evidence-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "0" "${RUN_RC}" "non-strict mode warns but succeeds"
    assert_match "Pack contains extra files not covered by checksums\\.sha256" "${RUN_ERR}" "warns on extra files"
    assert_match "extra\\.txt" "${RUN_ERR}" "lists extra file"
}

test_verify_pack_ignores_macos_transport_artifacts_in_strict_mode() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh
    pack_verify_signature_helper() {
        printf '%s\n' "sha256:$(printf 'a%.0s' {1..64})"
    }
    pack_validate_manifest_schema() {
        return 0
    }
    pack_verify_manifest_provenance() {
        return 0
    }

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/__MACOSX"
    echo "payload" > "${pack_dir}/payload.txt"
    echo "junk" > "${pack_dir}/._payload.txt"
    echo "junk" > "${pack_dir}/.DS_Store"
    echo "junk" > "${pack_dir}/__MACOSX/junk.txt"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} payload.txt > checksums.sha256
    )

    local checksums_digest
    checksums_digest="$(cd "${pack_dir}" && python3 -c 'import hashlib;print(hashlib.sha256(open("checksums.sha256","rb").read()).hexdigest())' < /dev/null)"
    printf '%s\n' "{\"format\":\"evidence-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify --strict
    assert_rc "0" "${RUN_RC}" "strict mode ignores macOS transport artifacts"
}

test_verify_pack_rejects_manifest_missing_checksums_digest_field() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} payload.txt > checksums.sha256
    )

    printf '%s\n' '{"format":"evidence-pack-v1","checksums_sha256":"checksums.sha256"}' > "${pack_dir}/manifest.json"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "4" "${RUN_RC}" "missing digest must fail schema validation"
    assert_match "checksums_sha256_digest" "${RUN_ERR}" "missing digest error mentions field"
    assert_match "required property" "${RUN_ERR}" "missing digest error comes from schema validation"
}

test_verify_pack_rejects_manifest_with_empty_checksums_digest_field() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} payload.txt > checksums.sha256
    )

    printf '%s\n' '{"format":"evidence-pack-v1","checksums_sha256":"checksums.sha256","checksums_sha256_digest":""}' > "${pack_dir}/manifest.json"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "4" "${RUN_RC}" "empty digest must fail schema validation"
    assert_match "checksums_sha256_digest" "${RUN_ERR}" "empty digest error mentions field"
    assert_match "too short" "${RUN_ERR}" "empty digest rejected by schema length validation"
}

test_verify_pack_rejects_manifest_with_wrong_format() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} payload.txt > checksums.sha256
    )

    local checksums_digest
    checksums_digest="$(cd "${pack_dir}" && python3 -c 'import hashlib;print(hashlib.sha256(open("checksums.sha256","rb").read()).hexdigest())' < /dev/null)"
    printf '%s\n' "{\"format\":\"evidence-pack-v0\",\"generated_at\":\"2026-03-12T00:00:00Z\",\"suite\":\"subset\",\"network_mode\":\"offline\",\"determinism\":\"strict\",\"repeats\":0,\"run_dir\":\"runs/example\",\"artifacts\":[\"payload.txt\"],\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "4" "${RUN_RC}" "bad manifest format fails"
    assert_match "evidence-pack-v1" "${RUN_ERR}" "error mentions required format"
}

test_verify_pack_rejects_manifest_with_bad_checksums_pointer() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} payload.txt > checksums.sha256
    )

    local checksums_digest
    checksums_digest="$(cd "${pack_dir}" && python3 -c 'import hashlib;print(hashlib.sha256(open("checksums.sha256","rb").read()).hexdigest())' < /dev/null)"
    printf '%s\n' "{\"format\":\"evidence-pack-v1\",\"checksums_sha256\":\"manifest.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "4" "${RUN_RC}" "bad checksum pointer fails schema validation"
    assert_match "checksums_sha256" "${RUN_ERR}" "error mentions checksums pointer field"
    assert_match "checksums\\.sha256" "${RUN_ERR}" "error mentions required checksums pointer"
}

test_verify_pack_rejects_manifest_when_checksums_digest_computation_is_empty() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"
    echo "junk" > "${pack_dir}/checksums.sha256"
    echo '{"format":"evidence-pack-v1","checksums_sha256":"checksums.sha256","checksums_sha256_digest":"0000000000000000000000000000000000000000000000000000000000000000"}' > "${pack_dir}/manifest.json"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/sha256sum" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    chmod +x "${bin_dir}/sha256sum"

    local original_path="${PATH}"
    PATH="${bin_dir}:${PATH}"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "6" "${RUN_RC}" "empty computed digest must fail"
    assert_match "Failed to compute sha256 for checksums\\.sha256" "${RUN_ERR}" "digest computation error"

    PATH="${original_path}"
}

test_verify_pack_rejects_signature_helper_errors() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"
    echo "{}" > "${pack_dir}/checksums.sha256"
    printf '%s\n' '{"format":"evidence-pack-v1","checksums_sha256":"checksums.sha256","checksums_sha256_digest":"0000000000000000000000000000000000000000000000000000000000000000"}' > "${pack_dir}/manifest.json"

    pack_verify_signature_helper() {
        echo "signature helper exploded" >&2
        return 1
    }

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "5" "${RUN_RC}" "signature helper failures surface as signature errors"
    assert_match "signature helper exploded" "${RUN_ERR}" "helper stderr surfaced"
}

test_verify_pack_rejects_bad_manifest_signature() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"
    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} payload.txt > checksums.sha256
    )
    local checksums_digest
    checksums_digest="$(cd "${pack_dir}" && python3 -c 'import hashlib;print(hashlib.sha256(open("checksums.sha256","rb").read()).hexdigest())' < /dev/null)"
    printf '%s\n' "{\"format\":\"evidence-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"
    pack_test_sign_manifest "${pack_dir}"
    python3 - "${pack_dir}/manifest.json" <<'PY'
import json
import sys

path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
payload["checksums_sha256_digest"] = "0" * 64
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
PY

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "5" "${RUN_RC}" "invalid signature must fail"
    assert_match "manifest signature verification failed" "${RUN_ERR}" "signature failure surfaced"
}

test_verify_pack_rejects_signature_when_manifest_records_mismatched_fingerprint() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"
    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} payload.txt > checksums.sha256
    )
    local checksums_digest
    checksums_digest="$(cd "${pack_dir}" && python3 -c 'import hashlib;print(hashlib.sha256(open("checksums.sha256","rb").read()).hexdigest())' < /dev/null)"
    printf '%s\n' "{\"format\":\"evidence-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\",\"signing_key_fingerprint\":\"sha256:$(printf '0%.0s' {1..64})\"}" > "${pack_dir}/manifest.json"
    local signing_key="${TEST_TMPDIR}/evidence-pack-signing-key.pem"
    local public_key="${TEST_TMPDIR}/evidence-pack-signing-key.pub.pem"
    PYTHONPATH=src python3 - "${signing_key}" "${public_key}" "${pack_dir}/manifest.json" <<'PY'
import sys
from pathlib import Path

from tests._support_evidence_pack_signing import generate_signing_keypair, sign_manifest

private_key = Path(sys.argv[1])
public_key = Path(sys.argv[2])
manifest = Path(sys.argv[3])
generate_signing_keypair(private_key, public_key_path=public_key)
sign_manifest(manifest, signing_key_path=private_key)
PY

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "5" "${RUN_RC}" "mismatched fingerprint must fail"
    assert_match "does not match signature key" "${RUN_ERR}" "fingerprint mismatch surfaced"
}

test_verify_pack_rejects_unsignaled_error_injection_failure() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/metadata" "${pack_dir}/reports/modelA/edit/run_1" "${pack_dir}/reports/modelA/errors/nan_injection"
    cat > "${pack_dir}/metadata/scenarios.json" <<'JSON'
{
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {"id": "edit", "strictness": "must_pass"},
    {
      "id": "nan_injection",
      "strictness": "must_fail",
      "requirements": {
        "detectors_any_of": [
          {
            "kind": "validation_flag",
            "flag": "primary_metric_acceptable",
            "expected": false
          }
        ]
      }
    }
  ]
}
JSON
    echo "{}" > "${pack_dir}/reports/modelA/edit/run_1/evaluation.report.json"
    echo "{}" > "${pack_dir}/reports/modelA/errors/nan_injection/evaluation.report.json"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "${TEST_TMPDIR}/invarlock.calls"
for arg in "$@"; do
    if [[ "${arg}" == */errors/*/evaluation.report.json ]]; then
        echo '{"summary":{"ok":false,"reason":"policy_fail"},"results":[{"ok":false,"reason":"policy_fail","verification":{"runtime_provenance":{"binding_verified":true,"expected_digest_matched":false,"trust_status":"manifest_bound"}}}]}'
        exit 1
    fi
done
echo '{"summary":{"ok":true,"reason":"ok"},"results":[{"ok":true,"reason":"ok"}]}'
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"

    local original_path="${PATH}"
    PATH="${bin_dir}:${PATH}"

    run pack_verify_reports "${pack_dir}" ""
    assert_rc "1" "${RUN_RC}" "unsignaled error-injection failure is rejected"
    assert_match "lacks its intended report-local failure signal" "${RUN_ERR}" "missing report signal is explicit"
    assert_match "errors/nan_injection/evaluation\\.report\\.json" "$(cat "${TEST_TMPDIR}/invarlock.calls")" "attempts error-injection reports"

    PATH="${original_path}"
}

test_verify_pack_verify_reports_accepts_error_injection_report_failure_signal() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/metadata" "${pack_dir}/reports/modelA/edit/run_1" "${pack_dir}/reports/modelA/errors/scale_explosion"
    cat > "${pack_dir}/metadata/scenarios.json" <<'JSON'
{
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {"id": "edit", "strictness": "must_pass"},
    {
      "id": "scale_explosion",
      "strictness": "must_fail",
      "requirements": {
        "detectors_any_of": [
          {
            "kind": "validation_flag",
            "flag": "primary_metric_acceptable",
            "expected": false
          }
        ]
      }
    }
  ]
}
JSON
    echo '{"validation":{"primary_metric_acceptable":true,"preview_final_drift_acceptable":true,"invariants_pass":true,"spectral_stable":true,"rmt_stable":true,"guard_metric_impact_acceptable":true}}' > "${pack_dir}/reports/modelA/edit/run_1/evaluation.report.json"
    cat > "${pack_dir}/reports/modelA/errors/scale_explosion/evaluation.report.json" <<'JSON'
{
  "validation": {
    "primary_metric_acceptable": false,
    "preview_final_drift_acceptable": true,
    "invariants_pass": true,
    "spectral_stable": true,
    "rmt_stable": true,
    "guard_metric_impact_acceptable": true
  }
}
JSON

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "${TEST_TMPDIR}/invarlock.calls"
for arg in "$@"; do
    if [[ "${arg}" == */errors/*/evaluation.report.json ]]; then
        echo '{"summary":{"ok":false,"reason":"policy_fail"},"results":[{"ok":false,"reason":"policy_fail","verification":{"runtime_provenance":{"binding_verified":true,"expected_digest_matched":false,"trust_status":"manifest_bound"}}}]}'
        exit 1
    fi
done
echo '{"summary":{"ok":true,"reason":"ok"},"results":[{"ok":true,"reason":"ok"}]}'
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"

    local original_path="${PATH}"
    PATH="${bin_dir}:${PATH}"

    run pack_verify_reports "${pack_dir}" ""
    assert_rc "0" "${RUN_RC}" "verify accepts expected-failure reports with report failure signal"
    assert_match "edit/run_1/evaluation\\.report\\.json" "$(cat "${TEST_TMPDIR}/invarlock.calls")" "expected-pass report is verified"
    assert_match "errors/scale_explosion/evaluation\\.report\\.json" "$(cat "${TEST_TMPDIR}/invarlock.calls")" "error-injection report is verified"

    PATH="${original_path}"
}

test_verify_pack_verify_reports_accepts_scenario_expected_failures() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/metadata"
    mkdir -p "${pack_dir}/reports/modelA/quant_4bit_clean/run_1"
    mkdir -p "${pack_dir}/reports/modelA/prune_50pct_stress/run_1"
    cat > "${pack_dir}/metadata/scenarios.json" <<'JSON'
{
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {"id": "quant_4bit_clean", "strictness": "must_pass"},
    {
      "id": "prune_50pct_stress",
      "strictness": "must_fail",
      "primary_guard": "primary_metric",
      "requirements": {
        "detectors_any_of": [
          {
            "kind": "validation_flag",
            "flag": "primary_metric_acceptable",
            "expected": false
          }
        ]
      }
    }
  ]
}
JSON
    echo "{}" > "${pack_dir}/reports/modelA/quant_4bit_clean/run_1/evaluation.report.json"
    echo '{"validation":{"primary_metric_acceptable":false}}' > "${pack_dir}/reports/modelA/prune_50pct_stress/run_1/evaluation.report.json"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "${TEST_TMPDIR}/invarlock.calls"
for arg in "$@"; do
    if [[ "${arg}" == */prune_50pct_stress/*/evaluation.report.json ]]; then
        echo '{"summary":{"ok":false,"reason":"policy_fail"},"results":[{"ok":false,"reason":"policy_fail","verification":{"runtime_provenance":{"binding_verified":true,"expected_digest_matched":false,"trust_status":"manifest_bound"}}}]}'
        exit 1
    fi
done
echo '{"summary":{"ok":true,"reason":"ok"},"results":[{"ok":true,"reason":"ok"}]}'
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"

    local original_path="${PATH}"
    PATH="${bin_dir}:${PATH}"

    run pack_verify_reports "${pack_dir}" ""
    assert_rc "0" "${RUN_RC}" "verify succeeds when scenario-declared must_fail reports fail"
    assert_match "quant_4bit_clean/run_1/evaluation\\.report\\.json" "$(cat "${TEST_TMPDIR}/invarlock.calls")" "verifies expected-pass report"
    assert_match "prune_50pct_stress/run_1/evaluation\\.report\\.json" "$(cat "${TEST_TMPDIR}/invarlock.calls")" "attempts scenario expected-failure report"

    PATH="${original_path}"
}

test_verify_pack_verify_reports_accepts_informational_error_probe_that_verifies_clean() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/metadata"
    mkdir -p "${pack_dir}/reports/modelA/quant_4bit_clean/run_1"
    mkdir -p "${pack_dir}/reports/modelA/errors/spectral_moderate_scale_mlp_l31_up_s112"
    mkdir -p "${pack_dir}/reports/modelA/errors/rmt_norm_noise_probe"
    cat > "${pack_dir}/metadata/scenarios.json" <<'JSON'
{
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {"id": "quant_4bit_clean", "strictness": "must_pass"},
    {"id": "spectral_moderate_scale_mlp_l31_up_s112", "strictness": "must_detect"},
    {"id": "rmt_norm_noise_probe", "strictness": "informational"}
  ]
}
JSON
    echo "{}" > "${pack_dir}/reports/modelA/quant_4bit_clean/run_1/evaluation.report.json"
    echo "{}" > "${pack_dir}/reports/modelA/errors/spectral_moderate_scale_mlp_l31_up_s112/evaluation.report.json"
    echo "{}" > "${pack_dir}/reports/modelA/errors/rmt_norm_noise_probe/evaluation.report.json"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "${TEST_TMPDIR}/invarlock.calls"
printf '%s\n' '{"summary":{"ok":true,"reason":"ok"},"results":[{"ok":true,"reason":"ok"}]}'
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"

    local original_path="${PATH}"
    PATH="${bin_dir}:${PATH}"

    run pack_verify_reports "${pack_dir}" ""
    assert_rc "0" "${RUN_RC}" "informational error probe may verify clean"
    assert_match "quant_4bit_clean/run_1/evaluation\\.report\\.json" "$(cat "${TEST_TMPDIR}/invarlock.calls")" "verifies expected-pass report"
    assert_match "errors/spectral_moderate_scale_mlp_l31_up_s112/evaluation\\.report\\.json" "$(cat "${TEST_TMPDIR}/invarlock.calls")" "verifies must_detect probe report"
    assert_match "errors/rmt_norm_noise_probe/evaluation\\.report\\.json" "$(cat "${TEST_TMPDIR}/invarlock.calls")" "verifies informational probe report"
    [[ "${RUN_ERR}" != *"Expected verify failure verified as passing"* ]] || t_fail "informational probe should not be expected to fail verification"

    PATH="${original_path}"
}

test_verify_pack_verify_reports_rejects_scenario_expected_failure_that_passes() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/metadata"
    mkdir -p "${pack_dir}/reports/modelA/quant_4bit_clean/run_1"
    mkdir -p "${pack_dir}/reports/modelA/prune_50pct_stress/run_1"
    cat > "${pack_dir}/metadata/scenarios.json" <<'JSON'
{
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {"id": "quant_4bit_clean", "strictness": "must_pass"},
    {"id": "prune_50pct_stress", "strictness": "must_fail"}
  ]
}
JSON
    echo "{}" > "${pack_dir}/reports/modelA/quant_4bit_clean/run_1/evaluation.report.json"
    echo "{}" > "${pack_dir}/reports/modelA/prune_50pct_stress/run_1/evaluation.report.json"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' '{"summary":{"ok":true,"reason":"ok"},"results":[{"ok":true,"reason":"ok"}]}'
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"

    local original_path="${PATH}"
    PATH="${bin_dir}:${PATH}"

    run pack_verify_reports "${pack_dir}" ""
    assert_rc "1" "${RUN_RC}" "verify fails when scenario-declared must_fail report verifies clean"
    assert_match "Expected verify failure verified as passing" "${RUN_ERR}" "unexpected expected-failure pass is explicit"

    PATH="${original_path}"
}

_write_expected_failure_contract_pack() {
    local pack_dir="$1"
    mkdir -p "${pack_dir}/metadata"
    mkdir -p "${pack_dir}/reports/modelA/clean/run_1"
    mkdir -p "${pack_dir}/reports/modelA/stress/run_1"
    cat > "${pack_dir}/metadata/scenarios.json" <<'JSON'
{
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {"id": "clean", "strictness": "must_pass"},
    {
      "id": "stress",
      "strictness": "must_fail",
      "primary_guard": "primary_metric",
      "requirements": {
        "detectors_any_of": [
          {
            "kind": "validation_flag",
            "flag": "primary_metric_acceptable",
            "expected": false
          }
        ]
      }
    }
  ]
}
JSON
    echo '{}' > "${pack_dir}/reports/modelA/clean/run_1/evaluation.report.json"
    echo '{"validation":{"primary_metric_acceptable":false}}' > "${pack_dir}/reports/modelA/stress/run_1/evaluation.report.json"
}

test_verify_pack_rejects_malformed_result_as_expected_failure() {
    mock_reset
    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    _write_expected_failure_contract_pack "${pack_dir}"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
for arg in "$@"; do
    if [[ "${arg}" == */stress/*/evaluation.report.json ]]; then
        echo '{"summary":{"ok":false,"reason":"malformed"},"results":[{"ok":false,"reason":"malformed","verification":{"runtime_provenance":{"binding_verified":true,"expected_digest_matched":false,"trust_status":"manifest_bound"}}}]}'
        exit 2
    fi
done
echo '{"summary":{"ok":true,"reason":"ok"},"results":[{"ok":true,"reason":"ok"}]}'
EOF
    chmod +x "${bin_dir}/invarlock"

    local original_path="${PATH}"
    PATH="${bin_dir}:${PATH}"
    run pack_verify_reports "${pack_dir}" ""
    assert_rc "1" "${RUN_RC}" "malformed report cannot satisfy must_fail"
    assert_match "must produce POLICY_FAIL, not malformed" "${RUN_ERR}" "malformed outcome is explicit"
    PATH="${original_path}"
}

test_verify_pack_rejects_runtime_only_result_as_expected_failure() {
    mock_reset
    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    _write_expected_failure_contract_pack "${pack_dir}"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
for arg in "$@"; do
    if [[ "${arg}" == */stress/*/evaluation.report.json ]]; then
        echo '{"summary":{"ok":false,"reason":"policy_fail"},"results":[{"ok":false,"reason":"policy_fail","verification":{"runtime_provenance":{"binding_verified":false,"expected_digest_matched":false,"trust_status":"failed"}}}]}'
        exit 1
    fi
done
echo '{"summary":{"ok":true,"reason":"ok"},"results":[{"ok":true,"reason":"ok"}]}'
EOF
    chmod +x "${bin_dir}/invarlock"

    local original_path="${PATH}"
    PATH="${bin_dir}:${PATH}"
    run pack_verify_reports "${pack_dir}" ""
    assert_rc "1" "${RUN_RC}" "runtime-only failure cannot satisfy must_fail"
    assert_match "lacks valid report/runtime binding" "${RUN_ERR}" "runtime binding failure is explicit"
    PATH="${original_path}"
}

test_verify_pack_verify_reports_errors_when_only_error_injection_reports_present() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/metadata" "${pack_dir}/reports/modelA/errors/nan_injection"
    cat > "${pack_dir}/metadata/scenarios.json" <<'JSON'
{
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [{"id": "nan_injection", "strictness": "must_fail"}]
}
JSON
    echo "{}" > "${pack_dir}/reports/modelA/errors/nan_injection/evaluation.report.json"

    run pack_verify_reports "${pack_dir}" ""
    assert_rc "1" "${RUN_RC}" "only error-injection reports must fail"
    assert_match "No reports expected to pass" "${RUN_ERR}" "expected-pass report requirement surfaced"
}

test_verify_pack_rejects_tampered_payload_when_checksums_bound() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} payload.txt > checksums.sha256
    )

    local checksums_digest
    checksums_digest="$(cd "${pack_dir}" && python3 -c 'import hashlib;print(hashlib.sha256(open("checksums.sha256","rb").read()).hexdigest())' < /dev/null)"
    printf '%s\n' "{\"format\":\"evidence-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    echo "tampered" > "${pack_dir}/payload.txt"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "6" "${RUN_RC}" "tampered payload must fail checksum verification"
}

test_verify_pack_returns_integrity_error_when_manifest_provenance_fails() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    printf '%s\n' '{}' > "${pack_dir}/manifest.json"
    printf '%s\n' 'hash  payload' > "${pack_dir}/checksums.sha256"

    pack_validate_manifest_schema() { return 0; }
    pack_verify_signature() { return 0; }
    pack_verify_manifest_binds_checksums() { return 0; }
    pack_verify_baseline_materials() { return 0; }
    pack_verify_checksums() { return 0; }
    pack_verify_manifest_provenance() { return 1; }
    pack_verify_no_extra_files() { return 0; }
    pack_verify_reports() { return 0; }

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "${PACK_VERIFY_INTEGRITY}" "${RUN_RC}" "manifest provenance failure maps to integrity exit code"
}

test_verify_pack_exit_code_mappings_direct_branches() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    local policy_pack="${TEST_TMPDIR}/acceptance-policy.json"
    mkdir -p "${pack_dir}"
    echo "{}" > "${pack_dir}/manifest.json"
    echo "hash payload" > "${pack_dir}/checksums.sha256"
    echo "{}" > "${policy_pack}"

    pack_validate_manifest_schema() { return 0; }
    pack_verify_signature() { return 0; }
    pack_verify_manifest_binds_checksums() { return 0; }
    pack_verify_baseline_materials() { return 0; }
    pack_verify_policy_materials() { return 0; }
    pack_verify_checksums() { return 0; }
    pack_verify_manifest_provenance() { return 0; }
    pack_verify_no_extra_files() { return 0; }
    pack_verify_reports() { return 0; }

    run pack_verify_pack \
        --pack "${pack_dir}" \
        --skip-verify \
        --report-assurance strict \
        --policy-pack "${policy_pack}" \
        --expected-runtime-image-digest \
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert_rc "${PACK_VERIFY_OK}" "${RUN_RC}" "strict report assurance parses and succeeds"

    pack_verify_checksums() { return 1; }
    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "${PACK_VERIFY_INTEGRITY}" "${RUN_RC}" "checksum verification failure maps to integrity"
    pack_verify_checksums() { return 0; }

    pack_verify_manifest_provenance() { return 1; }
    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "${PACK_VERIFY_INTEGRITY}" "${RUN_RC}" "manifest provenance failure maps to integrity"
    pack_verify_manifest_provenance() { return 0; }

    pack_verify_no_extra_files() { return 1; }
    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "${PACK_VERIFY_INTEGRITY}" "${RUN_RC}" "extra-file verification failure maps to integrity"
    pack_verify_no_extra_files() { return 0; }

    pack_verify_reports() { return 1; }
    run pack_verify_pack --pack "${pack_dir}"
    assert_rc "${PACK_VERIFY_REPORTS}" "${RUN_RC}" "report verification failure maps to report exit code"
}

test_verify_pack_forwards_runtime_pin_to_all_nested_verify_calls() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/metadata"
    mkdir -p "${pack_dir}/reports/modelA/clean/run_1"
    mkdir -p "${pack_dir}/reports/modelA/stress/run_1"
    cat > "${pack_dir}/metadata/scenarios.json" <<'JSON'
{
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {"id": "clean", "strictness": "must_pass"},
    {
      "id": "stress",
      "strictness": "must_fail",
      "primary_guard": "primary_metric",
      "requirements": {
        "detectors_any_of": [
          {
            "kind": "validation_flag",
            "flag": "primary_metric_acceptable",
            "expected": false
          }
        ]
      }
    }
  ]
}
JSON
    echo "{}" > "${pack_dir}/reports/modelA/clean/run_1/evaluation.report.json"
    echo '{"validation":{"primary_metric_acceptable":false}}' > "${pack_dir}/reports/modelA/stress/run_1/evaluation.report.json"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "${TEST_TMPDIR}/invarlock.calls"
for arg in "$@"; do
    if [[ "${arg}" == */stress/*/evaluation.report.json ]]; then
        echo '{"summary":{"ok":false,"reason":"policy_fail"},"results":[{"ok":false,"reason":"policy_fail","verification":{"runtime_provenance":{"binding_verified":true,"expected_digest_matched":true,"trust_status":"expected_image_digest_matched"}}}]}'
        exit 1
    fi
done
echo '{"summary":{"ok":true,"reason":"ok"},"results":[{"ok":true,"reason":"ok"}]}'
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"

    local original_path="${PATH}"
    local expected_digest="sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    PATH="${bin_dir}:${PATH}"
    PACK_REPORT_ASSURANCE=report
    PACK_EXPECTED_RUNTIME_IMAGE_DIGEST="${expected_digest}"
    export PACK_REPORT_ASSURANCE PACK_EXPECTED_RUNTIME_IMAGE_DIGEST

    run pack_verify_reports "${pack_dir}" ""
    assert_rc "0" "${RUN_RC}" "verification preserves expected-failure handling"
    local calls
    calls="$(cat "${TEST_TMPDIR}/invarlock.calls")"
    assert_match "clean/run_1/evaluation\\.report\\.json" "${calls}" "expected-pass report is verified"
    assert_match "stress/run_1/evaluation\\.report\\.json" "${calls}" "expected-failure report is verified"
    local digest_count
    digest_count="$(grep -c -- "--expected-runtime-image-digest ${expected_digest}" "${TEST_TMPDIR}/invarlock.calls")"
    assert_eq "2" "${digest_count}" "runtime image digest is forwarded to every nested verifier"

    unset PACK_REPORT_ASSURANCE PACK_EXPECTED_RUNTIME_IMAGE_DIGEST
    PATH="${original_path}"
}
