#!/usr/bin/env bash

pack_test_sign_manifest() {
    local pack_dir="$1"
    local repo_root="${TEST_ROOT:-$(pwd)}"
    python3 "${repo_root}/scripts/evidence_packs/python/sign_manifest.py" \
        --manifest "${pack_dir}/manifest.json" \
        --generate-ephemeral \
        >/dev/null
}

test_verify_pack_validates_checksums_and_reports() {
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

    local verify_out="${TEST_TMPDIR}/verify-off.json"
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

    run pack_verify_pack --nope
    assert_rc "2" "${RUN_RC}" "unknown arg returns 2"
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
    local repo_root
    repo_root="$(pwd)"
    cat > "${bin_dir}/shasum" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec python3 "${repo_root}/scripts/evidence_packs/python/shasum_mock.py" "\$@"
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

from invarlock.evidence_pack_integrity import generate_signing_keypair, sign_manifest

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

test_verify_pack_verify_reports_attempts_error_injection_reports_best_effort() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/reports/modelA/edit/run_1" "${pack_dir}/reports/modelA/errors/nan_injection"
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
        exit 1
    fi
done
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"

    local original_path="${PATH}"
    PATH="${bin_dir}:${PATH}"

    run pack_verify_reports "${pack_dir}" ""
    assert_rc "0" "${RUN_RC}" "verify succeeds when error-injection reports fail"
    assert_match "errors/nan_injection/evaluation\\.report\\.json" "$(cat "${TEST_TMPDIR}/invarlock.calls")" "attempts error-injection reports"

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
echo "$*" >> "${TEST_TMPDIR}/invarlock.calls"
for arg in "$@"; do
    if [[ "${arg}" == */prune_50pct_stress/*/evaluation.report.json ]]; then
        exit 1
    fi
done
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
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"

    local original_path="${PATH}"
    PATH="${bin_dir}:${PATH}"

    run pack_verify_reports "${pack_dir}" ""
    assert_rc "1" "${RUN_RC}" "verify fails when scenario-declared must_fail report verifies clean"
    assert_match "Expected verify failure passed" "${RUN_ERR}" "unexpected expected-failure pass is explicit"

    PATH="${original_path}"
}

test_verify_pack_verify_reports_errors_when_only_error_injection_reports_present() {
    mock_reset

    source ./scripts/evidence_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/reports/modelA/errors/nan_injection"
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
    pack_verify_checksums() { return 0; }
    pack_verify_manifest_provenance() { return 1; }
    pack_verify_no_extra_files() { return 0; }
    pack_verify_reports() { return 0; }

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "${PACK_VERIFY_INTEGRITY}" "${RUN_RC}" "manifest provenance failure maps to integrity exit code"
}
