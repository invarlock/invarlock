#!/usr/bin/env bash

pack_test_sign_manifest() {
    local pack_dir="$1"
    local repo_root="${TEST_ROOT:-$(pwd)}"
    python3 "${repo_root}/scripts/proof_packs/python/sign_manifest.py" \
        --manifest "${pack_dir}/manifest.json" \
        --generate-ephemeral \
        >/dev/null
}

test_run_pack_build_pack_collects_artifacts() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/analysis" "${run_dir}/state"
    mkdir -p "${run_dir}/modelA/reports/edit/run_1"

    echo "verdict" > "${run_dir}/reports/final_verdict.txt"
    echo "{}" > "${run_dir}/reports/final_verdict.json"
    echo "{}" > "${run_dir}/reports/guard_intervention_summary.json"
    echo '{"model_list": ["org/model"], "models": {"org/model": {"revision": "abc"}}}' > "${run_dir}/state/model_revisions.json"
    echo '{"schema":"proof_pack_scenarios_v1","schema_version":1,"scenarios":[]}' > "${run_dir}/state/scenarios.json"
    echo '{"org/model":{"quant_rtn":{"bits":4}}}' > "${run_dir}/state/tuned_edit_params.json"
    echo "{}" > "${run_dir}/modelA/reports/edit/run_1/evaluation.report.json"
    echo '{"probe":"rmt_cross_model_v1","stable":false}' > "${run_dir}/modelA/reports/edit/run_1/rmt_probe.json"
    echo '{"probe":"ve_probe_v1","signal":true}' > "${run_dir}/modelA/reports/edit/run_1/ve_probe.json"

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
        for arg in "$@"; do
            [[ "${arg}" != "--allow-unattested-artifacts" ]] || exit 97
        done
        echo '{"ok": true}'
        exit 0
        ;;
esac
echo '{}'
EOF
    chmod +x "${bin_dir}/invarlock"
    export PATH="${bin_dir}:${PATH}"

    PACK_SIGN_MANIFEST=0
    # run_suite.sh/validation_suite.sh may clobber SCRIPT_DIR at runtime; ensure
    # run_pack.sh packaging does not depend on it.
    SCRIPT_DIR="${TEST_TMPDIR}/bogus"
    export SCRIPT_DIR

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    pack_build_pack "${run_dir}" "${pack_dir}"

    assert_file_exists "${pack_dir}/results/verdicts/final_verdict.txt" "verdict copied"
    assert_file_exists "${pack_dir}/metadata/model_revisions.json" "revisions copied"
    assert_file_exists "${pack_dir}/metadata/scenarios.json" "scenarios manifest copied"
    assert_file_exists "${pack_dir}/reports/modelA/edit/run_1/evaluation.report.json" "report copied"
    assert_file_exists "${pack_dir}/reports/modelA/edit/run_1/rmt_probe.json" "probe sidecar copied"
    assert_file_exists "${pack_dir}/reports/modelA/edit/run_1/ve_probe.json" "ve probe sidecar copied"
    assert_file_exists "${pack_dir}/reports/modelA/edit/run_1/verify.json" "verify output captured"
    assert_file_exists "${pack_dir}/results/verification_summary.json" "verification summary written"
    run python3 -c 'import json,sys; json.load(open(sys.argv[1], encoding="utf-8"))' "${pack_dir}/results/verification_summary.json"
    assert_rc "0" "${RUN_RC}" "verification summary is valid JSON"
    assert_file_exists "${pack_dir}/manifest.json" "manifest written"
    assert_file_exists "${pack_dir}/checksums.sha256" "checksums written"
    assert_file_exists "${pack_dir}/reports/modelA/edit/run_1/evaluation.html" "html rendered"
    assert_file_exists "${pack_dir}/README.md" "readme written"
    assert_match "signed manifest, strict verification, and a PASS final verdict" "$(cat "${pack_dir}/README.md")" "README documents proof-grade triad"
    assert_match "invarlock advanced proof-pack verify" "$(cat "${pack_dir}/README.md")" "README points to advanced proof-pack verify"
    assert_file_exists "${pack_dir}/results/analysis/guard_intervention_summary.json" "intervention summary copied"
    assert_file_exists "${pack_dir}/metadata/source_repo.json" "source repo metadata written"
    assert_file_exists "${pack_dir}/metadata/environment.json" "environment metadata written"
    assert_file_exists "${pack_dir}/metadata/tuned_edit_params.json" "tuned edit params copied"
    run python3 -c 'import json,sys; payload=json.load(open(sys.argv[1], encoding="utf-8")); assert payload["builder"]["id"]=="invarlock/proof-pack@v1"; assert payload["subject"]["path"]=="results/verdicts/final_verdict.json"; assert payload["invocation"]["config_source"]["path"]=="metadata/source_repo.json"; assert payload["environment"]["path"]=="metadata/environment.json"; assert any(item["path"]=="metadata/model_revisions.json" for item in payload["materials"]); assert any(item["path"]=="metadata/scenarios.json" for item in payload["materials"]); assert any(item["path"]=="metadata/tuned_edit_params.json" for item in payload["materials"])' "${pack_dir}/manifest.json"
    assert_rc "0" "${RUN_RC}" "manifest carries attestation metadata"
}

test_run_pack_build_pack_fails_when_source_repo_metadata_fails() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/analysis" "${run_dir}/state"
    echo "verdict" > "${run_dir}/reports/final_verdict.txt"
    echo '{"verdict":"PASS"}' > "${run_dir}/reports/final_verdict.json"

    pack_write_source_repo_metadata() {
        local dest="$1"
        echo "ERROR: git is required to collect proof-pack source provenance." >&2
        return 1
    }

    run pack_build_pack "${run_dir}" "${TEST_TMPDIR}/pack"
    assert_rc "1" "${RUN_RC}" "pack build fails when source repo metadata cannot be written"
    assert_match "git is required to collect proof-pack source provenance" "${RUN_ERR}" "source provenance failure is surfaced"
}

test_run_pack_build_pack_rejects_failed_final_verdict() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports"
    echo "FAIL" > "${run_dir}/reports/final_verdict.txt"
    echo '{"verdict":"FAIL"}' > "${run_dir}/reports/final_verdict.json"

    run pack_build_pack "${run_dir}" "${TEST_TMPDIR}/pack"
    assert_rc "1" "${RUN_RC}" "pack build fails when run verdict is FAIL"
    assert_match "refusing to build a distributable pack" "${RUN_ERR}" "rejects failed run verdict"
}

test_run_pack_build_pack_layout_v2_nests_results_and_metadata() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/analysis" "${run_dir}/state"
    mkdir -p "${run_dir}/modelA/reports/edit/run_1"

    echo "verdict" > "${run_dir}/reports/final_verdict.txt"
    echo "{}" > "${run_dir}/reports/final_verdict.json"
    echo "{}" > "${run_dir}/reports/guard_intervention_summary.json"
    echo '{"model_list": ["org/model"], "models": {"org/model": {"revision": "abc"}}}' > "${run_dir}/state/model_revisions.json"
    echo '{"schema":"proof_pack_scenarios_v1","schema_version":1,"scenarios":[]}' > "${run_dir}/state/scenarios.json"
    echo "{}" > "${run_dir}/modelA/reports/edit/run_1/evaluation.report.json"

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
        for arg in "$@"; do
            [[ "${arg}" != "--allow-unattested-artifacts" ]] || exit 97
        done
        echo '{"ok": true}'
        exit 0
        ;;
esac
echo '{}'
EOF
    chmod +x "${bin_dir}/invarlock"
    export PATH="${bin_dir}:${PATH}"

    PACK_SIGN_MANIFEST=0
    PACK_PACK_LAYOUT="v2"
    pack_sign_manifest() {
        local pack_dir="$1"
        echo "sig" > "${pack_dir}/manifest.signature.json"
    }

    local pack_dir="${TEST_TMPDIR}/pack"
    pack_build_pack "${run_dir}" "${pack_dir}"

    assert_file_exists "${pack_dir}/results/verdicts/final_verdict.txt" "verdict nested"
    assert_file_exists "${pack_dir}/metadata/model_revisions.json" "revisions moved to metadata"
    assert_file_exists "${pack_dir}/metadata/scenarios.json" "scenarios manifest moved to metadata"
    assert_file_exists "${pack_dir}/metadata/source_repo.json" "source repo metadata written"
    assert_file_exists "${pack_dir}/metadata/environment.json" "environment metadata written"
    assert_file_exists "${pack_dir}/metadata/manifest.json" "manifest copied to metadata"
    assert_file_exists "${pack_dir}/metadata/manifest.signature.json" "manifest signature copied to metadata"
    assert_file_exists "${pack_dir}/metadata/checksums.sha256" "checksums copied to metadata"
    assert_file_exists "${pack_dir}/results/analysis/guard_intervention_summary.json" "intervention summary nested"
    [[ ! -f "${pack_dir}/results/final_verdict.txt" ]] || t_fail "legacy verdict path should not exist under v2 layout"
}

test_run_pack_build_pack_rejects_unknown_layout() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"
    PATH="${bin_dir}:${PATH}"
    export PATH

    PACK_PACK_LAYOUT="nope"
    run pack_build_pack "${run_dir}" "${TEST_TMPDIR}/pack"
    assert_rc "2" "${RUN_RC}" "unknown layout returns 2"
}

test_run_pack_build_pack_rejects_legacy_layouts() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    chmod +x "${bin_dir}/invarlock"
    PATH="${bin_dir}:${PATH}"
    export PATH

    local layout
    for layout in v1 flat legacy; do
        PACK_PACK_LAYOUT="${layout}"
        run pack_build_pack "${run_dir}" "${TEST_TMPDIR}/pack_${layout}"
        assert_rc "2" "${RUN_RC}" "legacy layout ${layout} returns 2"
        assert_match "no longer supported; use v2" "${RUN_ERR}" "legacy layout ${layout} error"
    done
}

test_run_pack_build_pack_ignores_error_injection_verify_failures() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/analysis" "${run_dir}/state"
    mkdir -p "${run_dir}/modelA/reports/edit/run_1"
    mkdir -p "${run_dir}/modelA/reports/errors/nan_injection"

    echo "verdict" > "${run_dir}/reports/final_verdict.txt"
    echo "{}" > "${run_dir}/reports/final_verdict.json"
    echo "model,score" > "${run_dir}/analysis/eval_results.csv"
    echo "{}" > "${run_dir}/modelA/reports/edit/run_1/evaluation.report.json"
    echo "{}" > "${run_dir}/modelA/reports/errors/nan_injection/evaluation.report.json"

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
        for arg in "$@"; do
            [[ "${arg}" != "--allow-unattested-artifacts" ]] || exit 97
        done
        report="${@: -1}"
        echo '{"ok": false}'
        if [[ "${report}" == */errors/*/evaluation.report.json ]]; then
            exit 1
        fi
        exit 0
        ;;
esac
echo '{}'
EOF
    chmod +x "${bin_dir}/invarlock"
    export PATH="${bin_dir}:${PATH}"

    PACK_SIGN_MANIFEST=0

    local pack_dir="${TEST_TMPDIR}/pack"
    pack_build_pack "${run_dir}" "${pack_dir}"

    assert_file_exists "${pack_dir}/reports/modelA/edit/run_1/verify.json" "clean verify output captured"
    assert_file_exists "${pack_dir}/reports/modelA/errors/nan_injection/verify.json" "error injection verify output captured"
    assert_file_exists "${pack_dir}/results/verification_summary.json" "verification summary written"
    run python3 -c 'import json,sys; json.load(open(sys.argv[1], encoding="utf-8"))' "${pack_dir}/results/verification_summary.json"
    assert_rc "0" "${RUN_RC}" "verification summary is valid JSON"
    assert_match "\"clean_reports\": 1" "$(cat "${pack_dir}/results/verification_summary.json")" "clean count recorded"
    assert_match "\"error_injection_reports\": 1" "$(cat "${pack_dir}/results/verification_summary.json")" "error injection count recorded"
    assert_match "\"failed_reports\": 0" "$(cat "${pack_dir}/results/verification_summary.json")" "failed count recorded"
}

test_run_pack_build_pack_continues_when_html_report_fails() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/analysis" "${run_dir}/state"
    mkdir -p "${run_dir}/modelA/reports/edit/run_1"

    echo "verdict" > "${run_dir}/reports/final_verdict.txt"
    echo "{}" > "${run_dir}/reports/final_verdict.json"
    echo '{"model_list": ["org/model"], "models": {"org/model": {"revision": "abc"}}}' > "${run_dir}/state/model_revisions.json"
    echo '{"schema":"proof_pack_scenarios_v1","schema_version":1,"scenarios":[]}' > "${run_dir}/state/scenarios.json"
    echo "{}" > "${run_dir}/modelA/reports/edit/run_1/evaluation.report.json"

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
            exit 1
        fi
        ;;
    verify)
        for arg in "$@"; do
            [[ "${arg}" != "--allow-unattested-artifacts" ]] || exit 97
        done
        echo '{"ok": true}'
        exit 0
        ;;
esac
echo '{}'
EOF
    chmod +x "${bin_dir}/invarlock"
    export PATH="${bin_dir}:${PATH}"

    PACK_SIGN_MANIFEST=0

    local pack_dir="${TEST_TMPDIR}/pack"
    run pack_build_pack "${run_dir}" "${pack_dir}"
    assert_rc "0" "${RUN_RC}" "pack build succeeds when html render fails"
    assert_match "Failed to render HTML" "${RUN_ERR}" "warns when html render fails"
    assert_file_exists "${pack_dir}/manifest.json" "manifest still written"
    assert_file_exists "${pack_dir}/checksums.sha256" "checksums still written"
}

test_run_pack_build_pack_writes_pack_files_on_unexpected_verify_failure() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/analysis"
    mkdir -p "${run_dir}/modelA/reports/edit/run_1"

    echo "verdict" > "${run_dir}/reports/final_verdict.txt"
    echo "{}" > "${run_dir}/reports/final_verdict.json"
    echo "model,score" > "${run_dir}/analysis/eval_results.csv"
    echo "{}" > "${run_dir}/modelA/reports/edit/run_1/evaluation.report.json"

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
        for arg in "$@"; do
            [[ "${arg}" != "--allow-unattested-artifacts" ]] || exit 97
        done
        echo '{"ok": false}'
        exit 1
        ;;
esac
echo '{}'
EOF
    chmod +x "${bin_dir}/invarlock"
    export PATH="${bin_dir}:${PATH}"

    PACK_SIGN_MANIFEST=0

    local pack_dir="${TEST_TMPDIR}/pack"
    run pack_build_pack "${run_dir}" "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "unexpected verify failure returns non-zero"
    [[ ! -e "${pack_dir}" ]] || t_fail "failed pack build should not leave a final pack behind path='${pack_dir}'"
    assert_eq "0" "$(find "${TEST_TMPDIR}" -maxdepth 1 -type d -name '.pack.tmp.*' | wc -l | tr -d ' ')" "failed pack build cleans staging directories"
}


test_run_pack_checksums_include_files() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/results/verdicts"
    echo "verdict" > "${pack_dir}/results/verdicts/final_verdict.txt"
    echo "{}" > "${pack_dir}/manifest.json"
    mkdir -p "${pack_dir}/metadata" "${pack_dir}/__MACOSX"
    echo "{}" > "${pack_dir}/metadata/manifest.json"
    echo "sig" > "${pack_dir}/metadata/manifest.signature.json"
    echo "x" > "${pack_dir}/metadata/checksums.sha256"
    echo "junk" > "${pack_dir}/.DS_Store"
    echo "junk" > "${pack_dir}/__MACOSX/junk.txt"

    pack_write_checksums "${pack_dir}"

    assert_file_exists "${pack_dir}/checksums.sha256" "checksums written"

    local checksums
    checksums="$(cat "${pack_dir}/checksums.sha256")"
    assert_match "results/verdicts/final_verdict.txt" "${checksums}" "checksums include results"
    if [[ "${checksums}" == *manifest.json* ]]; then
        t_fail "checksums must not include manifest.json to avoid signature cycles"
    fi
    if [[ "${checksums}" == *metadata/manifest.json* ]]; then
        t_fail "checksums must not include metadata/manifest.json to avoid signature cycles"
    fi
    if [[ "${checksums}" == *.DS_Store* ]]; then
        t_fail "checksums must ignore .DS_Store artifacts"
    fi
    if [[ "${checksums}" == *__MACOSX* ]]; then
        t_fail "checksums must ignore __MACOSX artifacts"
    fi
}


test_run_pack_helpers_cover_error_paths() {

    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    run pack_require_cmd definitely_missing_cmd
    assert_rc "1" "${RUN_RC}" "missing command returns non-zero"

    run pack_copy_file "${TEST_TMPDIR}/missing.txt" "${TEST_TMPDIR}/dest.txt"
    assert_rc "1" "${RUN_RC}" "missing artifact returns non-zero"

    run pack_report_rel_path "${TEST_TMPDIR}/run" "${TEST_TMPDIR}/nope"
    assert_rc "1" "${RUN_RC}" "invalid report path returns non-zero"

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/reports"
    run pack_verify_reports "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "missing reports returns non-zero"
}

test_run_pack_sha256_cmd_fallback_and_sign_toggle() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

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
    assert_match "shasum" "${sha_cmd}" "sha fallback uses shasum"

    PATH="${bin_dir}:${original_path}"

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "{}" > "${pack_dir}/manifest.json"

    PACK_SIGN_MANIFEST=0
    run pack_sign_manifest "${pack_dir}"
    assert_rc "0" "${RUN_RC}" "sign manifest returns 0 when signing is disabled"
    [[ ! -f "${pack_dir}/manifest.signature.json" ]] || t_fail "signature bundle should not be written when signing is disabled"

    PATH="${original_path}"
}


test_run_pack_sign_manifest_writes_package_native_signature() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo '{"format":"proof-pack-v1"}' > "${pack_dir}/manifest.json"

    run pack_sign_manifest "${pack_dir}"
    assert_rc "0" "${RUN_RC}" "sign manifest succeeds"
    assert_file_exists "${pack_dir}/manifest.signature.json" "signature bundle created"
    assert_eq \
        "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["signing_key_fingerprint"])' "${pack_dir}/manifest.json" < /dev/null)" \
        "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["signing_key_fingerprint"])' "${pack_dir}/manifest.signature.json" < /dev/null)" \
        "manifest and signature bundle record the same fingerprint"
}

test_run_pack_sign_manifest_errors_and_cleans_when_helper_fails() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo "{}" > "${pack_dir}/manifest.json"

    pack_sign_manifest_helper() {
        local manifest_path="$1"
        printf 'partial' > "$(dirname "${manifest_path}")/manifest.signature.json"
        echo "helper failed" >&2
        return 1
    }

    run pack_sign_manifest "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "sign manifest fails when helper fails"
    [[ ! -f "${pack_dir}/manifest.signature.json" ]] || t_fail "failed signature bundle should be removed"
    assert_match "manifest signing failed" "${RUN_ERR}" "signing error surfaced"
    assert_match "helper failed" "${RUN_ERR}" "helper stderr surfaced"
}

test_run_pack_sign_manifest_accepts_explicit_signing_key() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo '{"format":"proof-pack-v1"}' > "${pack_dir}/manifest.json"
    local signing_key="${TEST_TMPDIR}/proof-pack-signing-key.pem"
    local public_key="${TEST_TMPDIR}/proof-pack-signing-key.pub.pem"

    PYTHONPATH=src python3 - "${signing_key}" "${public_key}" <<'PY'
import sys
from pathlib import Path

from invarlock.proof_pack_integrity import generate_signing_keypair

generate_signing_keypair(
    Path(sys.argv[1]),
    public_key_path=Path(sys.argv[2]),
)
PY

    PACK_SIGNING_KEY="${signing_key}"
    run pack_sign_manifest "${pack_dir}"
    assert_rc "0" "${RUN_RC}" "sign manifest succeeds with explicit signing key"
    assert_file_exists "${pack_dir}/manifest.signature.json" "signature bundle created with explicit key"
}

test_run_pack_sign_manifest_error_paths() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack_missing_manifest"
    mkdir -p "${pack_dir}"

    run pack_sign_manifest "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "sign fails when manifest missing"
    assert_match "manifest\\.json missing" "${RUN_ERR}" "missing manifest error"

    local invalid_key_dir="${TEST_TMPDIR}/pack_invalid_key"
    mkdir -p "${invalid_key_dir}"
    echo "{}" > "${invalid_key_dir}/manifest.json"

    PACK_SIGNING_KEY="${TEST_TMPDIR}/missing-signing-key.pem"
    run pack_sign_manifest "${invalid_key_dir}"
    assert_rc "1" "${RUN_RC}" "sign fails when explicit signing key is missing"
    assert_match "manifest signing failed" "${RUN_ERR}" "explicit signing key error surfaced"
}

test_run_pack_build_pack_fails_when_signing_fails() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/analysis" "${run_dir}/state"
    mkdir -p "${run_dir}/modelA/reports/edit/run_1"

    echo "verdict" > "${run_dir}/reports/final_verdict.txt"
    echo "{}" > "${run_dir}/reports/final_verdict.json"
    echo "{}" > "${run_dir}/modelA/reports/edit/run_1/evaluation.report.json"

    PACK_SKIP_HTML=1
    pack_sign_manifest_helper() {
        echo "signature helper failed" >&2
        return 1
    }

    local pack_dir="${TEST_TMPDIR}/pack"
    run pack_build_pack "${run_dir}" "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "pack build fails when signing helper fails"
    assert_match "manifest signing failed" "${RUN_ERR}" "signing failure surfaced"
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

    local pack_file="${TEST_TMPDIR}/pack.file"
    : > "${pack_file}"
    run pack_build_pack "${run_dir}" "${pack_file}"
    assert_rc "1" "${RUN_RC}" "non-directory pack target returns non-zero"
}

test_run_pack_atomic_helpers_cover_finalize_and_cleanup_paths() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    local staging_dir
    staging_dir="$(pack_prepare_staging_dir "${pack_dir}")"
    assert_match "/\\.pack\\.tmp\\." "${staging_dir}" "staging dir uses hidden tmp naming"
    echo "payload" > "${staging_dir}/payload.txt"

    mkdir -p "${pack_dir}"
    run pack_finalize_staging_dir "${staging_dir}" "${pack_dir}"
    assert_rc "0" "${RUN_RC}" "finalize replaces empty target atomically"
    assert_file_exists "${pack_dir}/payload.txt" "staged payload moved into final pack"

    local file_target="${TEST_TMPDIR}/pack.file"
    : > "${file_target}"
    local staging_file
    staging_file="$(pack_prepare_staging_dir "${file_target}")"
    run pack_finalize_staging_dir "${staging_file}" "${file_target}"
    assert_rc "1" "${RUN_RC}" "finalize rejects non-directory targets"
    pack_cleanup_staging_dir "${staging_file}"
    [[ ! -e "${staging_file}" ]] || t_fail "cleanup helper removes staging dir path='${staging_file}'"
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

    run pack_run_pack --models
    assert_rc "2" "${RUN_RC}" "missing models value"

    run pack_run_pack --out
    assert_rc "2" "${RUN_RC}" "missing out value"

    run pack_run_pack --pack-dir
    assert_rc "2" "${RUN_RC}" "missing pack-dir value"

    run pack_run_pack --layout
    assert_rc "2" "${RUN_RC}" "missing layout value"

    run pack_run_pack --determinism
    assert_rc "2" "${RUN_RC}" "missing determinism value"

    run pack_run_pack --scenario-ids
    assert_rc "2" "${RUN_RC}" "missing scenario-ids value"

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

    pack_run_pack --suite full --models "org/modelA" --net 1 --layout v2 --determinism strict --repeats 2 --scenario-ids "x,y" --out "${TEST_TMPDIR}/out"

    assert_match "--suite[[:space:]]+full" "$(cat "${TEST_TMPDIR}/run.args")" "suite forwarded"
    assert_match "--models[[:space:]]+org/modelA" "$(cat "${TEST_TMPDIR}/run.args")" "models forwarded"
    assert_eq "v2" "${PACK_PACK_LAYOUT}" "layout forwarded"
    assert_match "--determinism[[:space:]]+strict" "$(cat "${TEST_TMPDIR}/run.args")" "determinism forwarded"
    assert_match "--repeats[[:space:]]+2" "$(cat "${TEST_TMPDIR}/run.args")" "repeats forwarded"
    assert_match "--scenario-ids[[:space:]]+x,y" "$(cat "${TEST_TMPDIR}/run.args")" "scenario ids forwarded"
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

    pack_run_pack --errors-only --out "${TEST_TMPDIR}/out_err"
    assert_match "--errors-only" "$(cat "${TEST_TMPDIR}/run.args")" "errors-only forwarded"

    pack_run_pack --resume --pack-dir "${TEST_TMPDIR}/pack3" --out "${TEST_TMPDIR}/out3"
    assert_match "--resume" "$(cat "${TEST_TMPDIR}/run.args")" "resume forwarded"
    assert_eq "${TEST_TMPDIR}/out3|${TEST_TMPDIR}/pack3" "$(cat "${TEST_TMPDIR}/pack.args")" "custom pack dir used"
}

test_run_pack_require_passing_run_verdict_falls_back_without_helper() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    unset -f pack_read_final_verdict 2>/dev/null || true

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports"
    printf '%s\n' '{"verdict":"PASS"}' > "${run_dir}/reports/final_verdict.json"

    pack_require_passing_run_verdict "${run_dir}"

    printf '%s\n' '{"verdict":"FAIL"}' > "${run_dir}/reports/final_verdict.json"
    run pack_require_passing_run_verdict "${run_dir}"
    assert_rc "1" "${RUN_RC}" "fallback parser rejects failed verdicts"
    assert_match "refusing to build a distributable pack" "${RUN_ERR}" "fallback failure explains requirement"
}

test_run_pack_entrypoint_propagates_layout_normalization_failures() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    pack_entrypoint() { :; }
    pack_build_pack() { t_fail "pack_build_pack should not run when layout normalization fails"; }
    pack_normalize_layout() { return 7; }

    run pack_run_pack --out "${TEST_TMPDIR}/out"
    assert_rc "7" "${RUN_RC}" "layout normalization failure propagates"
}
