#!/usr/bin/env bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pack_manifest_test_helpers.sh"

test_verify_pack_validates_checksums_and_reports() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/reports" "${pack_dir}/results"
    echo "{}" > "${pack_dir}/reports/evaluation.report.json"
    local report_digest
    report_digest="$(pack_file_sha256 "${pack_dir}" "reports/evaluation.report.json")"
    printf '%s\n' \
        "{\"verdict\":\"PASS\",\"report_sha256\":\"${report_digest}\"}" \
        > "${pack_dir}/results/final_verdict.json"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} reports/evaluation.report.json results/final_verdict.json > checksums.sha256
    )

    local checksums_digest
    checksums_digest="$(cd "${pack_dir}" && python3 -c 'import hashlib;print(hashlib.sha256(open("checksums.sha256","rb").read()).hexdigest())' < /dev/null)"
    printf '%s\n' "{\"format\":\"evidence-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
for arg in "$@"; do
    [[ "${arg}" != "--allow-unverified-provenance" ]] || exit 97
done
echo '{"ok": true}'
EOF
    chmod +x "${bin_dir}/invarlock"
    export PATH="${bin_dir}:${PATH}"

    local verify_out="${TEST_TMPDIR}/verify.json"
    pack_verify_pack --pack "${pack_dir}" --json-out "${verify_out}"

    assert_file_exists "${verify_out}" "verify output written"
}

test_verify_pack_report_assurance_off_still_invokes_report_verify() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/reports" "${pack_dir}/results"
    echo "{}" > "${pack_dir}/reports/evaluation.report.json"
    local report_digest
    report_digest="$(pack_file_sha256 "${pack_dir}" "reports/evaluation.report.json")"
    printf '%s\n' \
        "{\"verdict\":\"PASS\",\"report_sha256\":\"${report_digest}\"}" \
        > "${pack_dir}/results/final_verdict.json"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} reports/evaluation.report.json results/final_verdict.json > checksums.sha256
    )

    local checksums_digest
    checksums_digest="$(cd "${pack_dir}" && python3 -c 'import hashlib;print(hashlib.sha256(open("checksums.sha256","rb").read()).hexdigest())' < /dev/null)"
    printf '%s\n' "{\"format\":\"evidence-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    local verify_out="${TEST_TMPDIR}/verify-off.json"
    local bin_dir
    bin_dir="$(mock_install_bin_dir)"
    rm -f "${bin_dir}/invarlock"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "invarlock $*" >> "${TEST_TMPDIR}/fixtures/invarlock.calls"
printf '%s\n' '{"summary":{"ok":true,"reason":"ok"},"results":[{"ok":true,"reason":"ok"}]}'
EOF
    chmod +x "${bin_dir}/invarlock"
    run pack_verify_pack --pack "${pack_dir}" --report-assurance off --json-out "${verify_out}"

    assert_rc "0" "${RUN_RC}" "report-assurance off still verifies report files"
    assert_file_exists "${verify_out}" "verify output path is still used"
    assert_match "--assurance off" "$(cat "${TEST_TMPDIR}/fixtures/invarlock.calls")" "nested verify uses assurance off"

}

test_verify_pack_errors_on_missing_args() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    run pack_verify_pack
    assert_rc "2" "${RUN_RC}" "pack argument required"

    run pack_verify_pack --pack
    assert_rc "2" "${RUN_RC}" "missing pack value"

    run pack_verify_pack --pack "${TEST_TMPDIR}/pack" --json-out
    assert_rc "2" "${RUN_RC}" "missing json-out value"

    run pack_verify_pack --pack "${TEST_TMPDIR}/pack" --expected-fingerprint
    assert_rc "2" "${RUN_RC}" "missing expected-fingerprint value"

    run pack_verify_pack --pack "${TEST_TMPDIR}/pack" --expected-runtime-image-digest
    assert_rc "2" "${RUN_RC}" "missing expected runtime image digest value"

    run pack_verify_pack \
        --pack "${TEST_TMPDIR}/pack" \
        --expected-runtime-image-digest "sha256:not-a-digest"
    assert_rc "2" "${RUN_RC}" "malformed expected runtime image digest is rejected"

    run pack_verify_pack --nope
    assert_rc "2" "${RUN_RC}" "unknown arg returns 2"

    run pack_verify_pack --pack "${TEST_TMPDIR}/pack" --report-assurance
    assert_rc "2" "${RUN_RC}" "missing report-assurance value is rejected"

    run pack_verify_pack --pack "${TEST_TMPDIR}/pack" --report-assurance weak
    assert_rc "2" "${RUN_RC}" "invalid report-assurance value is rejected"

    run pack_verify_pack --pack "${TEST_TMPDIR}/pack" --report-assurance strict
    assert_rc "2" "${RUN_RC}" "strict report assurance requires an external runtime image digest"
}

test_verify_pack_source_selects_python_from_path_without_test_real_python() {
    mock_reset

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/python" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    chmod +x "${bin_dir}/python"

    run bash -x -c 'unset PYTHON_BIN TEST_REAL_PYTHON3; PATH="$1:/usr/bin:/bin"; source ./scripts/evidence_packs/verify_pack.sh; printf "%s\n" "${PYTHON_BIN}"' _ "${bin_dir}"
    assert_rc "0" "${RUN_RC}" "verify_pack source selects python from PATH"
    assert_eq "${bin_dir}/python" "${RUN_OUT}" "python path exported from PATH"
}

test_verify_pack_source_selects_python_from_path_in_current_shell() {
    mock_reset

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/python" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    chmod +x "${bin_dir}/python"

    unset PYTHON_BIN TEST_REAL_PYTHON3
    PATH="${bin_dir}:/usr/bin:/bin"
    source ./scripts/evidence_packs/verify_pack.sh

    assert_eq "${bin_dir}/python" "${PYTHON_BIN}" "verify_pack source selects python from PATH in the current shell"
}


test_verify_pack_rejects_json_output_inside_pack() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/reports"
    echo "{}" > "${pack_dir}/reports/evaluation.report.json"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} reports/evaluation.report.json > checksums.sha256
    )

    local checksums_digest
    checksums_digest="$(cd "${pack_dir}" && python3 -c 'import hashlib;print(hashlib.sha256(open("checksums.sha256","rb").read()).hexdigest())' < /dev/null)"
    printf '%s\n' "{\"format\":\"evidence-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    run pack_verify_pack --pack "${pack_dir}" --json-out "${pack_dir}/verify.json"
    assert_rc "2" "${RUN_RC}" "json output inside pack is rejected"
    assert_match "--json-out must point outside the pack directory" "${RUN_ERR}" "error explains path constraint"
}


test_verify_pack_help_and_main_entrypoint() {
    mock_reset

    run bash -x ./scripts/evidence_packs/verify_pack.sh --help
    assert_rc "0" "${RUN_RC}" "help returns 0"
    assert_match "Usage" "${RUN_OUT}" "usage printed"
}


test_verify_pack_double_dash_terminator() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    run pack_verify_pack -- --pack "${TEST_TMPDIR}/pack"
    assert_rc "2" "${RUN_RC}" "terminator stops parsing"
}

test_verify_pack_manifest_digest_validation_reports_missing_and_empty_fields_directly() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"

    printf '%s\n' '{"format":"evidence-pack-v1","checksums_sha256":"checksums.sha256"}' > "${pack_dir}/manifest.json"
    run pack_verify_manifest_binds_checksums "${pack_dir}" "false"
    assert_rc "1" "${RUN_RC}" "missing digest field fails"
    assert_match "missing checksums_sha256_digest" "${RUN_ERR}" "missing digest error is direct"

    printf '%s\n' '{"format":"evidence-pack-v1","checksums_sha256":"checksums.sha256","checksums_sha256_digest":""}' > "${pack_dir}/manifest.json"
    run pack_verify_manifest_binds_checksums "${pack_dir}" "false"
    assert_rc "1" "${RUN_RC}" "empty digest field fails"
    assert_match "checksums_sha256_digest is empty" "${RUN_ERR}" "empty digest error is direct"
}

test_verify_pack_manifest_digest_mismatch_direct_branch() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    pack_manifest_field() { echo "expected-digest"; }
    pack_file_sha256() { echo "actual-digest"; }

    run pack_verify_manifest_binds_checksums "${TEST_TMPDIR}/pack" "0"
    assert_rc "1" "${RUN_RC}" "manifest digest mismatch fails directly"
    assert_match "checksums\\.sha256 digest mismatch" "${RUN_ERR}" "digest mismatch error is explicit"
}

test_verify_pack_manifest_provenance_accepts_digest_backed_refs() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/results/verdicts" "${pack_dir}/metadata"
    echo '{"verdict":"PASS"}' > "${pack_dir}/results/verdicts/final_verdict.json"
    echo '{"commit":"abc"}' > "${pack_dir}/metadata/source_repo.json"
    echo '{"models":{"org/model":{"revision":"abc"}}}' > "${pack_dir}/metadata/model_revisions.json"

    local subject_digest config_digest materials_digest
    subject_digest="$(python3 -c 'import hashlib,sys; print("sha256:"+hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${pack_dir}/results/verdicts/final_verdict.json")"
    config_digest="$(python3 -c 'import hashlib,sys; print("sha256:"+hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${pack_dir}/metadata/source_repo.json")"
    materials_digest="$(python3 -c 'import hashlib,sys; print("sha256:"+hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${pack_dir}/metadata/model_revisions.json")"

    cat > "${pack_dir}/manifest.json" <<EOF
{"format":"evidence-pack-v1","checksums_sha256":"checksums.sha256","checksums_sha256_digest":"0000000000000000000000000000000000000000000000000000000000000000","subject":{"name":"final_verdict","path":"results/verdicts/final_verdict.json","digest":"${subject_digest}"},"invocation":{"config_source":{"path":"metadata/source_repo.json","digest":"${config_digest}"}},"materials":[{"name":"model_revisions","path":"metadata/model_revisions.json","digest":"${materials_digest}"}]}
EOF

    run pack_verify_manifest_provenance "${pack_dir}"
    assert_rc "0" "${RUN_RC}" "digest-backed provenance references verify"
}

test_verify_pack_manifest_provenance_rejects_digest_mismatch() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/results/verdicts"
    echo '{"verdict":"PASS"}' > "${pack_dir}/results/verdicts/final_verdict.json"
    printf '%s\n' '{"format":"evidence-pack-v1","checksums_sha256":"checksums.sha256","checksums_sha256_digest":"0000000000000000000000000000000000000000000000000000000000000000","subject":{"name":"final_verdict","path":"results/verdicts/final_verdict.json","digest":"sha256:0000000000000000000000000000000000000000000000000000000000000000"}}' > "${pack_dir}/manifest.json"

    run pack_verify_manifest_provenance "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "subject digest mismatch fails provenance verification"
    assert_match "digest mismatch" "${RUN_ERR}" "digest mismatch error reported"
}


test_verify_pack_verify_reports_without_json_out() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/reports"
    echo "{}" > "${pack_dir}/reports/evaluation.report.json"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
echo '{"ok": true}'
EOF
    chmod +x "${bin_dir}/invarlock"
    PATH="${bin_dir}:${PATH}"

    run pack_verify_reports "${pack_dir}" ""
    assert_rc "0" "${RUN_RC}" "verify without json_out succeeds"
}

test_verify_pack_direct_helper_argument_branches() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local calls="${TEST_TMPDIR}/python.calls"
    _cmd_python() {
        printf '%s\n' "$*" > "${calls}"
    }

    run pack_warn "direct warning"
    assert_rc "0" "${RUN_RC}" "pack_warn returns success"
    assert_match "WARNING: direct warning" "${RUN_ERR}" "pack_warn writes to stderr"

    run pack_verify_signature_helper "${TEST_TMPDIR}/pack" "1" "sha256:abc"
    assert_rc "0" "${RUN_RC}" "signature helper accepts strict and expected fingerprint"
    assert_match "signature --strict --expected-fingerprint sha256:abc ${TEST_TMPDIR}/pack" "$(cat "${calls}")" "signature helper forwards strict and expected fingerprint args"

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/metadata"
    run pack_scenario_strictness "${pack_dir}" "missing"
    assert_rc "1" "${RUN_RC}" "scenario strictness rejects missing metadata in verify path"

    mkdir -p "${pack_dir}/reports/model/clean"
    printf '%s\n' '{"scenarios":[{"id":"clean","strictness":"must_pass"}]}' > "${pack_dir}/metadata/scenarios.json"
    run pack_report_scenario_id "${pack_dir}" "${pack_dir}/reports/model/clean/evaluation.report.json"
    assert_rc "0" "${RUN_RC}" "report scenario id helper forwards report path"
    assert_match "report-scenario-id ${pack_dir} ${pack_dir}/reports/model/clean/evaluation\\.report\\.json" "$(cat "${calls}")" "report scenario id command forwarded"

    run pack_scenario_strictness "${pack_dir}" "clean"
    assert_rc "0" "${RUN_RC}" "scenario strictness helper forwards metadata path"
    assert_match "scenario-strictness ${pack_dir}/metadata/scenarios\\.json clean" "$(cat "${calls}")" "scenario strictness command forwarded"

    run pack_report_expects_verify_failure "${pack_dir}" "${pack_dir}/reports/model/clean/evaluation.report.json"
    assert_rc "0" "${RUN_RC}" "report expected-failure helper forwards report path"
    assert_match "report-expects-verify-failure ${pack_dir} ${pack_dir}/reports/model/clean/evaluation\\.report\\.json" "$(cat "${calls}")" "report expected-failure command forwarded"

    run pack_verify_reports "${pack_dir}" "${TEST_TMPDIR}/verify.json"
    assert_rc "0" "${RUN_RC}" "verify reports accepts json_out"
    assert_match "verify-reports ${pack_dir} --profile dev --report-assurance report --require-clean --json-out ${TEST_TMPDIR}/verify\\.json" "$(cat "${calls}")" "verify reports forwards json_out"

    PACK_EXPECTED_RUNTIME_IMAGE_DIGEST="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    export PACK_EXPECTED_RUNTIME_IMAGE_DIGEST
    run pack_verify_reports "${pack_dir}" "${TEST_TMPDIR}/verify.json"
    assert_rc "0" "${RUN_RC}" "verify reports accepts an expected runtime image digest"
    assert_match "--expected-runtime-image-digest ${PACK_EXPECTED_RUNTIME_IMAGE_DIGEST}" "$(cat "${calls}")" "verify reports forwards expected runtime image digest"
    unset PACK_EXPECTED_RUNTIME_IMAGE_DIGEST
}

test_verify_pack_manifest_field_reads_values() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local manifest_path="${TEST_TMPDIR}/manifest.json"
    printf '%s\n' '{"foo":"bar","num":1}' > "${manifest_path}"

    run pack_manifest_field "${manifest_path}" "foo"
    assert_rc "0" "${RUN_RC}" "string field lookup succeeds"
    assert_eq "bar" "${RUN_OUT}" "string field returned"

    run pack_manifest_field "${manifest_path}" "num"
    assert_rc "0" "${RUN_RC}" "numeric field lookup succeeds"
    assert_eq "1" "${RUN_OUT}" "numeric field returned"
}

test_verify_pack_reports_missing_pack_dir_and_files() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    run pack_verify_pack --pack "${TEST_TMPDIR}/missing"
    assert_rc "3" "${RUN_RC}" "missing pack dir fails"

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    run pack_verify_pack --pack "${pack_dir}"
    assert_rc "3" "${RUN_RC}" "missing manifest fails"

    echo "{}" > "${pack_dir}/manifest.json"
    run pack_verify_pack --pack "${pack_dir}"
    assert_rc "3" "${RUN_RC}" "missing checksums fails"
}

test_verify_pack_sha256_cmd_fallback_and_no_reports() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/shasum" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
python3 - "$@" <<'PY'
from __future__ import annotations

import hashlib
import pathlib
import sys

args = sys.argv[1:]
check_file = None
files: list[str] = []
i = 0
while i < len(args):
    if args[i] == "-a":
        i += 2
    elif args[i] == "-c":
        check_file = args[i + 1] if i + 1 < len(args) else ""
        i += 2
    else:
        files.append(args[i])
        i += 1


def sha256(path: str) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


if check_file:
    ok = True
    for line in pathlib.Path(check_file).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if sha256(parts[-1]) != parts[0]:
            ok = False
    raise SystemExit(0 if ok else 1)

for filename in files:
    print(f"{sha256(filename)}  {filename}")
PY
EOF
    chmod +x "${bin_dir}/shasum"

    local original_path="${PATH}"
    PATH="${bin_dir}"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    assert_match "shasum" "${sha_cmd}" "fallback to shasum when sha256sum missing"

    PATH="${bin_dir}:${original_path}"

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"
    (
        cd "${pack_dir}"
        ${sha_cmd} payload.txt > checksums.sha256
    )

    local checksums_digest
    checksums_digest="$(cd "${pack_dir}" && python3 -c 'import hashlib;print(hashlib.sha256(open("checksums.sha256","rb").read()).hexdigest())' < /dev/null)"
    printf '%s\n' "{\"format\":\"evidence-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    run pack_verify_pack --pack "${pack_dir}"
    assert_rc "7" "${RUN_RC}" "missing reports fails"

    PATH="${original_path}"
}

test_verify_pack_skip_verify_and_unsigned_warning() {
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

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "0" "${RUN_RC}" "skip-verify succeeds"
    assert_match "manifest\\.signature\\.json missing; pack is unsigned" "${RUN_ERR}" "warns when signature bundle is missing"
}


test_verify_pack_signed_manifest_verifies_signature() {
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

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "0" "${RUN_RC}" "verify succeeds with package-native signature present"
}

test_verify_pack_signed_manifest_accepts_expected_fingerprint() {
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

    local expected_fingerprint
    expected_fingerprint="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["signing_key_fingerprint"])' "${pack_dir}/manifest.json")"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify --expected-fingerprint "${expected_fingerprint}"
    assert_rc "0" "${RUN_RC}" "verify succeeds with pinned package-native signature"
}

test_verify_pack_signed_manifest_rejects_unexpected_fingerprint() {
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

    run pack_verify_pack --pack "${pack_dir}" --skip-verify --expected-fingerprint "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    assert_rc "5" "${RUN_RC}" "verify rejects an unexpected signature signer"
    assert_match "signer mismatch" "${RUN_ERR}" "mismatch error names signer mismatch"
}


test_verify_pack_rejects_tampered_checksums_when_manifest_binds_digest() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "a" > "${pack_dir}/a.txt"
    echo "b" > "${pack_dir}/b.txt"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} a.txt b.txt > checksums.sha256
    )

    local checksums_digest
    checksums_digest="$(cd "${pack_dir}" && python3 -c 'import hashlib;print(hashlib.sha256(open("checksums.sha256","rb").read()).hexdigest())' < /dev/null)"
    printf '%s\n' "{\"format\":\"evidence-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "0" "${RUN_RC}" "baseline verification succeeds"

    # Reorder checksums lines (valid checksums, but different file digest).
    (
        cd "${pack_dir}"
        tail -n 1 checksums.sha256 > checksums.sha256.tmp
        head -n 1 checksums.sha256 >> checksums.sha256.tmp
        mv checksums.sha256.tmp checksums.sha256
    )

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "6" "${RUN_RC}" "tampered checksums must fail when digest is bound"
}


test_verify_pack_strict_requires_manifest_signature() {
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

    run pack_verify_pack --pack "${pack_dir}" --skip-verify --strict
    assert_rc "5" "${RUN_RC}" "strict mode requires a manifest signature"
}
