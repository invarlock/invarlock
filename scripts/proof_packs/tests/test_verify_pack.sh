#!/usr/bin/env bash

test_verify_pack_validates_checksums_and_certs() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/certs"
    echo "{}" > "${pack_dir}/certs/evaluation.report.json"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} certs/evaluation.report.json > checksums.sha256
    )

    local checksums_digest
    checksums_digest="$(cd "${pack_dir}" && python3 -c 'import hashlib;print(hashlib.sha256(open("checksums.sha256","rb").read()).hexdigest())' < /dev/null)"
    printf '%s\n' "{\"format\":\"proof-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo '{"ok": true}'
EOF
    chmod +x "${bin_dir}/invarlock"
    export PATH="${bin_dir}:${PATH}"

    local verify_out="${TEST_TMPDIR}/verify.json"
    pack_verify_pack --pack "${pack_dir}" --json-out "${verify_out}"

    assert_file_exists "${verify_out}" "verify output written"
}

test_verify_pack_errors_on_missing_args() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

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

    source ./scripts/proof_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/certs"
    echo "{}" > "${pack_dir}/certs/evaluation.report.json"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} certs/evaluation.report.json > checksums.sha256
    )

    local checksums_digest
    checksums_digest="$(cd "${pack_dir}" && python3 -c 'import hashlib;print(hashlib.sha256(open("checksums.sha256","rb").read()).hexdigest())' < /dev/null)"
    printf '%s\n' "{\"format\":\"proof-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    run pack_verify_pack --pack "${pack_dir}" --json-out "${pack_dir}/verify.json"
    assert_rc "2" "${RUN_RC}" "json output inside pack is rejected"
    assert_match "--json-out must point outside the pack directory" "${RUN_ERR}" "error explains path constraint"
}


test_verify_pack_help_and_main_entrypoint() {
    mock_reset

    run bash -x ./scripts/proof_packs/verify_pack.sh --help
    assert_rc "0" "${RUN_RC}" "help returns 0"
    assert_match "Usage" "${RUN_OUT}" "usage printed"
}


test_verify_pack_double_dash_terminator() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    run pack_verify_pack -- --pack "${TEST_TMPDIR}/pack"
    assert_rc "2" "${RUN_RC}" "terminator stops parsing"
}

test_verify_pack_manifest_digest_validation_reports_missing_and_empty_fields_directly() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"

    printf '%s\n' '{"format":"proof-pack-v1","checksums_sha256":"checksums.sha256"}' > "${pack_dir}/manifest.json"
    run pack_verify_manifest_binds_checksums "${pack_dir}" "false"
    assert_rc "1" "${RUN_RC}" "missing digest field fails"
    assert_match "missing checksums_sha256_digest" "${RUN_ERR}" "missing digest error is direct"

    printf '%s\n' '{"format":"proof-pack-v1","checksums_sha256":"checksums.sha256","checksums_sha256_digest":""}' > "${pack_dir}/manifest.json"
    run pack_verify_manifest_binds_checksums "${pack_dir}" "false"
    assert_rc "1" "${RUN_RC}" "empty digest field fails"
    assert_match "checksums_sha256_digest is empty" "${RUN_ERR}" "empty digest error is direct"
}

test_verify_pack_manifest_attestation_accepts_digest_backed_refs() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

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
{"format":"proof-pack-v1","checksums_sha256":"checksums.sha256","checksums_sha256_digest":"0000000000000000000000000000000000000000000000000000000000000000","subject":{"name":"final_verdict","path":"results/verdicts/final_verdict.json","digest":"${subject_digest}"},"invocation":{"config_source":{"path":"metadata/source_repo.json","digest":"${config_digest}"}},"materials":[{"name":"model_revisions","path":"metadata/model_revisions.json","digest":"${materials_digest}"}]}
EOF

    run pack_verify_manifest_attestation "${pack_dir}"
    assert_rc "0" "${RUN_RC}" "digest-backed attestation references verify"
}

test_verify_pack_manifest_attestation_rejects_digest_mismatch() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/results/verdicts"
    echo '{"verdict":"PASS"}' > "${pack_dir}/results/verdicts/final_verdict.json"
    printf '%s\n' '{"format":"proof-pack-v1","checksums_sha256":"checksums.sha256","checksums_sha256_digest":"0000000000000000000000000000000000000000000000000000000000000000","subject":{"name":"final_verdict","path":"results/verdicts/final_verdict.json","digest":"sha256:0000000000000000000000000000000000000000000000000000000000000000"}}' > "${pack_dir}/manifest.json"

    run pack_verify_manifest_attestation "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "subject digest mismatch fails attestation verification"
    assert_match "digest mismatch" "${RUN_ERR}" "digest mismatch error reported"
}


test_verify_pack_verify_certs_without_json_out() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/certs"
    echo "{}" > "${pack_dir}/certs/evaluation.report.json"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
echo '{"ok": true}'
EOF
    chmod +x "${bin_dir}/invarlock"
    PATH="${bin_dir}:${PATH}"

    run pack_verify_certs "${pack_dir}" ""
    assert_rc "0" "${RUN_RC}" "verify without json_out succeeds"
}

test_verify_pack_manifest_field_reads_values() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

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

    source ./scripts/proof_packs/verify_pack.sh

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

test_verify_pack_sha256_cmd_fallback_and_no_certs() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    local repo_root
    repo_root="$(pwd)"
    cat > "${bin_dir}/shasum" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec python3 "${repo_root}/scripts/proof_packs/python/shasum_mock.py" "\$@"
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
    printf '%s\n' "{\"format\":\"proof-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    run pack_verify_pack --pack "${pack_dir}"
    assert_rc "7" "${RUN_RC}" "missing certs fails"

    PATH="${original_path}"
}

test_verify_pack_skip_verify_and_gpg_warning() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"
    echo "sig" > "${pack_dir}/manifest.json.asc"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} payload.txt > checksums.sha256
    )

    local checksums_digest
    checksums_digest="$(cd "${pack_dir}" && python3 -c 'import hashlib;print(hashlib.sha256(open("checksums.sha256","rb").read()).hexdigest())' < /dev/null)"
    printf '%s\n' "{\"format\":\"proof-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    mkdir -p "${TEST_TMPDIR}/bin"
    local repo_root
    repo_root="$(pwd)"
    cat > "${TEST_TMPDIR}/bin/shasum" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec python3 "${repo_root}/scripts/proof_packs/python/shasum_mock.py" "\$@"
EOF
    chmod +x "${TEST_TMPDIR}/bin/shasum"

    local original_path="${PATH}"
    PATH="${TEST_TMPDIR}/bin:${original_path}"

    command() {
        if [[ "${1:-}" == "-v" && "${2:-}" == "gpg" ]]; then
            return 1
        fi
        builtin command "$@"
    }

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "0" "${RUN_RC}" "skip-verify succeeds"
    assert_match "gpg not found" "${RUN_ERR}" "warn when gpg missing"

    unset -f command
    PATH="${original_path}"
}


test_verify_pack_gpg_present_verifies_signature() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"
    echo "sig" > "${pack_dir}/manifest.json.asc"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} payload.txt > checksums.sha256
    )

    local checksums_digest
    checksums_digest="$(cd "${pack_dir}" && python3 -c 'import hashlib;print(hashlib.sha256(open("checksums.sha256","rb").read()).hexdigest())' < /dev/null)"
    printf '%s\n' "{\"format\":\"proof-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/gpg" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "${TEST_TMPDIR}/gpg.calls"
exit 0
EOF
    chmod +x "${bin_dir}/gpg"

    local original_path="${PATH}"
    PATH="${bin_dir}:${PATH}"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "0" "${RUN_RC}" "verify succeeds with gpg present"
    assert_file_exists "${TEST_TMPDIR}/gpg.calls" "gpg invoked"

    PATH="${original_path}"
}


test_verify_pack_rejects_tampered_checksums_when_manifest_binds_digest() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

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
    printf '%s\n' "{\"format\":\"proof-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

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

    source ./scripts/proof_packs/verify_pack.sh

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
    printf '%s\n' "{\"format\":\"proof-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify --strict
    assert_rc "5" "${RUN_RC}" "strict mode requires a manifest signature"
}


test_verify_pack_strict_rejects_extra_files() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

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
    printf '%s\n' "{\"format\":\"proof-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    echo "sig" > "${pack_dir}/manifest.json.asc"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/gpg" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exit 0
EOF
    chmod +x "${bin_dir}/gpg"

    local original_path="${PATH}"
    PATH="${bin_dir}:${PATH}"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify --strict
    assert_rc "6" "${RUN_RC}" "strict mode rejects extra files"

    PATH="${original_path}"
}

test_verify_pack_warns_on_extra_files_in_non_strict_mode() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

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
    printf '%s\n' "{\"format\":\"proof-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "0" "${RUN_RC}" "non-strict mode warns but succeeds"
    assert_match "Pack contains extra files not covered by checksums\\.sha256" "${RUN_ERR}" "warns on extra files"
    assert_match "extra\\.txt" "${RUN_ERR}" "lists extra file"
}

test_verify_pack_rejects_manifest_missing_checksums_digest_field() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} payload.txt > checksums.sha256
    )

    printf '%s\n' '{"format":"proof-pack-v1","checksums_sha256":"checksums.sha256"}' > "${pack_dir}/manifest.json"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "4" "${RUN_RC}" "missing digest must fail schema validation"
    assert_match "checksums_sha256_digest" "${RUN_ERR}" "missing digest error mentions field"
    assert_match "required property" "${RUN_ERR}" "missing digest error comes from schema validation"
}

test_verify_pack_rejects_manifest_with_empty_checksums_digest_field() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} payload.txt > checksums.sha256
    )

    printf '%s\n' '{"format":"proof-pack-v1","checksums_sha256":"checksums.sha256","checksums_sha256_digest":""}' > "${pack_dir}/manifest.json"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "4" "${RUN_RC}" "empty digest must fail schema validation"
    assert_match "checksums_sha256_digest" "${RUN_ERR}" "empty digest error mentions field"
    assert_match "too short" "${RUN_ERR}" "empty digest rejected by schema length validation"
}

test_verify_pack_rejects_manifest_with_wrong_format() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

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
    printf '%s\n' "{\"format\":\"proof-pack-v0\",\"generated_at\":\"2026-03-12T00:00:00Z\",\"suite\":\"subset\",\"network_mode\":\"offline\",\"determinism\":\"strict\",\"repeats\":0,\"run_dir\":\"runs/example\",\"artifacts\":[\"payload.txt\"],\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "4" "${RUN_RC}" "bad manifest format fails"
    assert_match "proof-pack-v1" "${RUN_ERR}" "error mentions required format"
}

test_verify_pack_rejects_manifest_with_bad_checksums_pointer() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

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
    printf '%s\n' "{\"format\":\"proof-pack-v1\",\"checksums_sha256\":\"manifest.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "4" "${RUN_RC}" "bad checksum pointer fails schema validation"
    assert_match "checksums_sha256" "${RUN_ERR}" "error mentions checksums pointer field"
    assert_match "checksums\\.sha256" "${RUN_ERR}" "error mentions required checksums pointer"
}

test_verify_pack_rejects_manifest_when_checksums_digest_computation_is_empty() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"
    echo "junk" > "${pack_dir}/checksums.sha256"
    echo '{"format":"proof-pack-v1","checksums_sha256":"checksums.sha256","checksums_sha256_digest":"0000000000000000000000000000000000000000000000000000000000000000"}' > "${pack_dir}/manifest.json"

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

test_verify_pack_strict_requires_gpg_when_signature_present() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"
    echo "sig" > "${pack_dir}/manifest.json.asc"
    echo "{}" > "${pack_dir}/checksums.sha256"
    printf '%s\n' '{"format":"proof-pack-v1","checksums_sha256":"checksums.sha256","checksums_sha256_digest":"0000000000000000000000000000000000000000000000000000000000000000"}' > "${pack_dir}/manifest.json"

    command() {
        if [[ "${1:-}" == "-v" && "${2:-}" == "gpg" ]]; then
            return 1
        fi
        builtin command "$@"
    }

    run pack_verify_pack --pack "${pack_dir}" --skip-verify --strict
    assert_rc "5" "${RUN_RC}" "strict verification fails when gpg missing"
    assert_match "gpg not found \\(strict mode requires signature verification\\)" "${RUN_ERR}" "strict gpg missing error"
}

test_verify_pack_rejects_bad_manifest_signature() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"
    echo "sig" > "${pack_dir}/manifest.json.asc"
    echo "{}" > "${pack_dir}/checksums.sha256"
    printf '%s\n' '{"format":"proof-pack-v1","checksums_sha256":"checksums.sha256","checksums_sha256_digest":"0000000000000000000000000000000000000000000000000000000000000000"}' > "${pack_dir}/manifest.json"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/gpg" <<'EOF'
#!/usr/bin/env bash
echo "BAD SIG" >&2
exit 2
EOF
    chmod +x "${bin_dir}/gpg"

    local original_path="${PATH}"
    PATH="${bin_dir}:${PATH}"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "5" "${RUN_RC}" "invalid signature must fail"
    assert_match "manifest signature verification failed" "${RUN_ERR}" "signature failure surfaced"

    PATH="${original_path}"
}

test_verify_pack_rejects_signature_when_manifest_records_mismatched_fingerprint() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/payload.txt"
    echo "sig" > "${pack_dir}/manifest.json.asc"
    echo "{}" > "${pack_dir}/checksums.sha256"
    printf '%s\n' '{"format":"proof-pack-v1","checksums_sha256":"checksums.sha256","checksums_sha256_digest":"0000000000000000000000000000000000000000000000000000000000000000","signing_key_fingerprint":"DEADBEEFDEADBEEF"}' > "${pack_dir}/manifest.json"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/gpg" <<'EOF'
#!/usr/bin/env bash
printf '[GNUPG:] VALIDSIG %s 20240101T000000 0 0 0 0 0 0 0 0\n' "0123456789ABCDEF0123456789ABCDEF01234567"
exit 0
EOF
    chmod +x "${bin_dir}/gpg"

    local original_path="${PATH}"
    PATH="${bin_dir}:${PATH}"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "5" "${RUN_RC}" "mismatched fingerprint must fail"
    assert_match "does not match signature key" "${RUN_ERR}" "fingerprint mismatch surfaced"

    PATH="${original_path}"
}

test_verify_pack_verify_certs_attempts_error_injection_reports_best_effort() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/certs/modelA/edit/run_1" "${pack_dir}/certs/modelA/errors/nan_injection"
    echo "{}" > "${pack_dir}/certs/modelA/edit/run_1/evaluation.report.json"
    echo "{}" > "${pack_dir}/certs/modelA/errors/nan_injection/evaluation.report.json"

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

    run pack_verify_certs "${pack_dir}" ""
    assert_rc "0" "${RUN_RC}" "verify succeeds when error-injection certs fail"
    assert_match "errors/nan_injection/evaluation\\.report\\.json" "$(cat "${TEST_TMPDIR}/invarlock.calls")" "attempts error-injection reports"

    PATH="${original_path}"
}

test_verify_pack_verify_certs_errors_when_only_error_injection_reports_present() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/certs/modelA/errors/nan_injection"
    echo "{}" > "${pack_dir}/certs/modelA/errors/nan_injection/evaluation.report.json"

    run pack_verify_certs "${pack_dir}" ""
    assert_rc "1" "${RUN_RC}" "only error-injection reports must fail"
    assert_match "No clean reports found" "${RUN_ERR}" "clean report requirement surfaced"
}

test_verify_pack_rejects_tampered_payload_when_checksums_bound() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

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
    printf '%s\n' "{\"format\":\"proof-pack-v1\",\"checksums_sha256\":\"checksums.sha256\",\"checksums_sha256_digest\":\"${checksums_digest}\"}" > "${pack_dir}/manifest.json"

    echo "tampered" > "${pack_dir}/payload.txt"

    run pack_verify_pack --pack "${pack_dir}" --skip-verify
    assert_rc "6" "${RUN_RC}" "tampered payload must fail checksum verification"
}
