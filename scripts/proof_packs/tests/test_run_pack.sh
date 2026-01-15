#!/usr/bin/env bash

test_run_pack_build_pack_collects_artifacts() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/analysis" "${run_dir}/state"
    mkdir -p "${run_dir}/modelA/certificates/edit/run_1"

    echo "verdict" > "${run_dir}/reports/final_verdict.txt"
    echo "{}" > "${run_dir}/reports/final_verdict.json"
    echo "model,score" > "${run_dir}/analysis/eval_results.csv"
    echo "model,metric" > "${run_dir}/analysis/guard_sensitivity_matrix.csv"
    echo '{"model_list": ["org/model"], "models": {"org/model": {"revision": "abc"}}}' > "${run_dir}/state/model_revisions.json"
    echo "{}" > "${run_dir}/modelA/certificates/edit/run_1/evaluation.cert.json"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cmd="${1:-}"
shift || true
case "${cmd}" in
    report)
        sub="${1:-}"
        if [[ "${sub}" == "html" ]]; then
            out=""
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --output|-o)
                        out="$2"
                        shift 2
                        ;;
                    *)
                        shift
                        ;;
                esac
            done
            mkdir -p "$(dirname "${out}")"
            printf '<html>ok</html>\n' > "${out}"
            exit 0
        fi
        ;;
    verify)
        echo '{"ok": true}'
        exit 0
        ;;
esac
echo '{}'
EOF
    chmod +x "${bin_dir}/invarlock"
    export PATH="${bin_dir}:${PATH}"

    PACK_GPG_SIGN=0

    local pack_dir="${TEST_TMPDIR}/pack"
    pack_build_pack "${run_dir}" "${pack_dir}"

    assert_file_exists "${pack_dir}/results/final_verdict.txt" "verdict copied"
    assert_file_exists "${pack_dir}/results/eval_results.csv" "eval results copied"
    assert_file_exists "${pack_dir}/certs/modelA/edit/run_1/evaluation.cert.json" "cert copied"
    assert_file_exists "${pack_dir}/certs/modelA/edit/run_1/verify.json" "verify output captured"
    assert_file_exists "${pack_dir}/manifest.json" "manifest written"
    assert_file_exists "${pack_dir}/checksums.sha256" "checksums written"
    assert_file_exists "${pack_dir}/certs/modelA/edit/run_1/evaluation.html" "html rendered"
    assert_file_exists "${pack_dir}/README.md" "readme written"
}


test_run_pack_checksums_include_files() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/results"
    echo "verdict" > "${pack_dir}/results/final_verdict.txt"
    echo "{}" > "${pack_dir}/manifest.json"

    pack_write_checksums "${pack_dir}"

    assert_file_exists "${pack_dir}/checksums.sha256" "checksums written"

    local checksums
    checksums="$(cat "${pack_dir}/checksums.sha256")"
    assert_match "results/final_verdict.txt" "${checksums}" "checksums include results"
    assert_match "manifest.json" "${checksums}" "checksums include manifest"
}


test_run_pack_helpers_cover_error_paths() {

    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    run pack_require_cmd definitely_missing_cmd
    assert_rc "1" "${RUN_RC}" "missing command returns non-zero"

    run pack_copy_file "${TEST_TMPDIR}/missing.txt" "${TEST_TMPDIR}/dest.txt"
    assert_rc "1" "${RUN_RC}" "missing artifact returns non-zero"

    run pack_cert_rel_path "${TEST_TMPDIR}/run" "${TEST_TMPDIR}/nope"
    assert_rc "1" "${RUN_RC}" "invalid cert path returns non-zero"

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/certs"
    run pack_verify_certs "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "missing certs returns non-zero"
}

test_run_pack_sha256_cmd_fallback_and_sign_warning() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/shasum" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-a" ]]; then
    shift 2
fi

if [[ "${1:-}" == "-c" ]]; then
    shift
    python3 - "$1" <<'PY'
import hashlib
import sys
from pathlib import Path

checksums = Path(sys.argv[1]).read_text().splitlines()
ok = True
for line in checksums:
    if not line.strip():
        continue
    parts = line.split()
    expected = parts[0]
    filename = parts[-1]
    actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    if actual != expected:
        ok = False
if not ok:
    sys.exit(1)
PY
    exit $?
fi

python3 - "$@" <<'PY'
import hashlib
import sys
from pathlib import Path

for filename in sys.argv[1:]:
    digest = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    print(f"{digest}  {filename}")
PY
EOF
    chmod +x "${bin_dir}/shasum"

    local original_path="${PATH}"
    PATH="${bin_dir}"

    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    assert_match "shasum" "${sha_cmd}" "sha fallback uses shasum"

    PATH="${bin_dir}:${original_path}"

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "{}" > "${pack_dir}/manifest.json"

    command() {
        if [[ "${1:-}" == "-v" && "${2:-}" == "gpg" ]]; then
            return 1
        fi
        builtin command "$@"
    }

    run pack_sign_manifest "${pack_dir}"
    assert_rc "0" "${RUN_RC}" "sign manifest returns 0 when gpg missing"
    assert_match "gpg not found" "${RUN_ERR}" "warns when gpg missing"

    unset -f command

    PATH="${original_path}"
}


test_run_pack_sign_manifest_with_gpg() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "{}" > "${pack_dir}/manifest.json"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/gpg" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
out=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            out="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done
if [[ -n "${out}" ]]; then
    printf 'sig' > "${out}"
fi
EOF
    chmod +x "${bin_dir}/gpg"

    local original_path="${PATH}"
    PATH="${bin_dir}:${PATH}"

    pack_sign_manifest "${pack_dir}"
    assert_file_exists "${pack_dir}/manifest.json.asc" "gpg signature created"

    PATH="${original_path}"
}

test_run_pack_build_pack_error_conditions() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    run pack_build_pack "" ""
    assert_rc "1" "${RUN_RC}" "missing args returns non-zero"

    run pack_build_pack "${TEST_TMPDIR}/missing" "${TEST_TMPDIR}/pack"
    assert_rc "1" "${RUN_RC}" "missing run dir returns non-zero"

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}"
    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "x" > "${pack_dir}/existing"
    run pack_build_pack "${run_dir}" "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "non-empty pack dir returns non-zero"
}

test_run_pack_entrypoint_errors_on_invalid_args() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    pack_entrypoint() { return 0; }
    pack_build_pack() { return 0; }

    run pack_run_pack --suite
    assert_rc "2" "${RUN_RC}" "missing suite value"

    run pack_run_pack --net
    assert_rc "2" "${RUN_RC}" "missing net value"

    run pack_run_pack --out
    assert_rc "2" "${RUN_RC}" "missing out value"

    run pack_run_pack --pack-dir
    assert_rc "2" "${RUN_RC}" "missing pack-dir value"

    run pack_run_pack --determinism
    assert_rc "2" "${RUN_RC}" "missing determinism value"

    run pack_run_pack --repeats nope
    assert_rc "2" "${RUN_RC}" "invalid repeats value"

    run pack_run_pack --net 9 --out "${TEST_TMPDIR}/out"
    assert_rc "2" "${RUN_RC}" "invalid net value"

    run pack_run_pack --nope
    assert_rc "2" "${RUN_RC}" "unknown arg returns 2"
}


test_run_pack_entrypoint_parses_suite_determinism_and_repeats() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    pack_entrypoint() { printf '%s\n' "$@" > "${TEST_TMPDIR}/run.args"; }
    pack_build_pack() { :; }

    pack_run_pack --suite full --net 1 --determinism strict --repeats 2 --out "${TEST_TMPDIR}/out"

    assert_match "--suite[[:space:]]+full" "$(cat "${TEST_TMPDIR}/run.args")" "suite forwarded"
    assert_match "--determinism[[:space:]]+strict" "$(cat "${TEST_TMPDIR}/run.args")" "determinism forwarded"
    assert_match "--repeats[[:space:]]+2" "$(cat "${TEST_TMPDIR}/run.args")" "repeats forwarded"
}


test_run_pack_help_and_main_entrypoint() {
    mock_reset

    run bash -x ./scripts/proof_packs/run_pack.sh --help
    assert_rc "0" "${RUN_RC}" "help returns 0"
    assert_match "Usage" "${RUN_OUT}" "usage printed"
}


test_run_pack_double_dash_defaults_out_and_pack_dir() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    pack_entrypoint() { printf '%s\n' "$@" > "${TEST_TMPDIR}/run.args"; }
    pack_build_pack() { printf '%s|%s' "$1" "$2" > "${TEST_TMPDIR}/pack.args"; }
    date() { echo "20240101_000000"; }

    pack_run_pack --

    assert_match "--out[[:space:]]+./proof_pack_runs/subset_20240101_000000" "$(cat "${TEST_TMPDIR}/run.args")" "default output dir used"
    assert_eq "./proof_pack_runs/subset_20240101_000000|./proof_pack_runs/subset_20240101_000000/proof_pack" "$(cat "${TEST_TMPDIR}/pack.args")" "default pack dir used"
}

test_run_pack_entrypoint_builds_run_args_for_modes() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    pack_entrypoint() { printf '%s\n' "$@" > "${TEST_TMPDIR}/run.args"; }
    pack_build_pack() { printf '%s|%s' "$1" "$2" > "${TEST_TMPDIR}/pack.args"; }

    pack_run_pack --calibrate-only --out "${TEST_TMPDIR}/out1"
    assert_match "--calibrate-only" "$(cat "${TEST_TMPDIR}/run.args")" "calibrate-only forwarded"
    assert_eq "${TEST_TMPDIR}/out1|${TEST_TMPDIR}/out1/proof_pack" "$(cat "${TEST_TMPDIR}/pack.args")" "default pack dir used"

    pack_run_pack --run-only --out "${TEST_TMPDIR}/out2"
    assert_match "--run-only" "$(cat "${TEST_TMPDIR}/run.args")" "run-only forwarded"

    pack_run_pack --resume --pack-dir "${TEST_TMPDIR}/pack3" --out "${TEST_TMPDIR}/out3"
    assert_match "--resume" "$(cat "${TEST_TMPDIR}/run.args")" "resume forwarded"
    assert_eq "${TEST_TMPDIR}/out3|${TEST_TMPDIR}/pack3" "$(cat "${TEST_TMPDIR}/pack.args")" "custom pack dir used"
}

