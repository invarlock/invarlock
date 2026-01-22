#!/usr/bin/env bash

test_verify_pack_validates_checksums_and_certs() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/certs"
    echo "{}" > "${pack_dir}/manifest.json"
    echo "{}" > "${pack_dir}/certs/evaluation.cert.json"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} manifest.json certs/evaluation.cert.json > checksums.sha256
    )

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


test_verify_pack_verify_certs_without_json_out() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/certs"
    echo "{}" > "${pack_dir}/certs/evaluation.cert.json"

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

test_verify_pack_reports_missing_pack_dir_and_files() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    run pack_verify_pack --pack "${TEST_TMPDIR}/missing"
    assert_rc "1" "${RUN_RC}" "missing pack dir fails"

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    run pack_verify_pack --pack "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "missing manifest fails"

    echo "{}" > "${pack_dir}/manifest.json"
    run pack_verify_pack --pack "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "missing checksums fails"
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
    echo "{}" > "${pack_dir}/manifest.json"
    (
        cd "${pack_dir}"
        ${sha_cmd} manifest.json > checksums.sha256
    )

    run pack_verify_pack --pack "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "missing certs fails"

    PATH="${original_path}"
}

test_verify_pack_skip_verify_and_gpg_warning() {
    mock_reset

    source ./scripts/proof_packs/verify_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "{}" > "${pack_dir}/manifest.json"
    echo "sig" > "${pack_dir}/manifest.json.asc"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} manifest.json manifest.json.asc > checksums.sha256
    )

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
    echo "{}" > "${pack_dir}/manifest.json"
    echo "sig" > "${pack_dir}/manifest.json.asc"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} manifest.json manifest.json.asc > checksums.sha256
    )

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
