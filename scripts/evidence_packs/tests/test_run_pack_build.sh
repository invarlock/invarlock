#!/usr/bin/env bash

test_run_pack_build_pack_layout_v2_nests_results_and_metadata() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/analysis" "${run_dir}/state"
    mkdir -p "${run_dir}/modelA/reports/edit/run_1"

    echo "verdict" > "${run_dir}/reports/final_verdict.txt"
    echo "{}" > "${run_dir}/reports/final_verdict.json"
    echo "{}" > "${run_dir}/reports/guard_intervention_summary.json"
    echo '{"model_list": ["org/model"], "models": {"org/model": {"revision": "abc"}}}' > "${run_dir}/state/model_revisions.json"
    echo '{"schema":"evidence_pack_scenarios_v1","schema_version":1,"scenarios":[]}' > "${run_dir}/state/scenarios.json"
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
            [[ "${arg}" != "--allow-unverified-provenance" ]] || exit 97
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

    source ./scripts/evidence_packs/run_pack.sh

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

    source ./scripts/evidence_packs/run_pack.sh

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

    source ./scripts/evidence_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/analysis" "${run_dir}/state"
    mkdir -p "${run_dir}/modelA/reports/edit/run_1"
    mkdir -p "${run_dir}/modelA/reports/errors/nan_injection"

    echo "verdict" > "${run_dir}/reports/final_verdict.txt"
    echo "{}" > "${run_dir}/reports/final_verdict.json"
    echo "model,score" > "${run_dir}/analysis/eval_results.csv"
    echo "{}" > "${run_dir}/modelA/reports/edit/run_1/evaluation.report.json"
    echo "{}" > "${run_dir}/modelA/reports/errors/nan_injection/evaluation.report.json"
    mkdir -p "${run_dir}/modelA/reports/errors/nan_injection/source" "${run_dir}/modelA/reports/errors/nan_injection/edited"
    echo "{}" > "${run_dir}/modelA/reports/errors/nan_injection/source/report.json"
    echo "{}" > "${run_dir}/modelA/reports/errors/nan_injection/edited/report.json"

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
            [[ "${arg}" != "--allow-unverified-provenance" ]] || exit 97
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
    assert_file_exists "${pack_dir}/reports/modelA/errors/nan_injection/source/report.json" "error source evidence copied"
    assert_file_exists "${pack_dir}/reports/modelA/errors/nan_injection/edited/report.json" "error edited evidence copied"
    assert_file_exists "${pack_dir}/results/verification_summary.json" "verification summary written"
    run python3 -c 'import json,sys; json.load(open(sys.argv[1], encoding="utf-8"))' "${pack_dir}/results/verification_summary.json"
    assert_rc "0" "${RUN_RC}" "verification summary is valid JSON"
    assert_match "\"clean_reports\": 1" "$(cat "${pack_dir}/results/verification_summary.json")" "clean count recorded"
    assert_match "\"error_injection_reports\": 1" "$(cat "${pack_dir}/results/verification_summary.json")" "error injection count recorded"
    assert_match "\"failed_reports\": 0" "$(cat "${pack_dir}/results/verification_summary.json")" "failed count recorded"
}

test_run_pack_build_pack_accepts_scenario_expected_verify_failures() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/analysis" "${run_dir}/state"
    mkdir -p "${run_dir}/modelA/reports/quant_4bit_clean/run_1"
    mkdir -p "${run_dir}/modelA/reports/prune_50pct_stress/run_1"

    echo "verdict" > "${run_dir}/reports/final_verdict.txt"
    echo '{"verdict":"PASS"}' > "${run_dir}/reports/final_verdict.json"
    echo '{"model_list": ["org/model"], "models": {"org/model": {"revision": "abc"}}}' > "${run_dir}/state/model_revisions.json"
    cat > "${run_dir}/state/scenarios.json" <<'JSON'
{
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {"id": "quant_4bit_clean", "strictness": "must_pass"},
    {"id": "prune_50pct_stress", "strictness": "must_fail"}
  ]
}
JSON
    echo "{}" > "${run_dir}/modelA/reports/quant_4bit_clean/run_1/evaluation.report.json"
    echo "{}" > "${run_dir}/modelA/reports/prune_50pct_stress/run_1/evaluation.report.json"

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
        report="${@: -1}"
        echo '{"ok": false}'
        if [[ "${report}" == */prune_50pct_stress/*/evaluation.report.json ]]; then
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

    assert_file_exists "${pack_dir}/reports/modelA/quant_4bit_clean/run_1/verify.json" "expected-pass verify output captured"
    assert_file_exists "${pack_dir}/reports/modelA/prune_50pct_stress/run_1/verify.json" "expected-fail verify output captured"
    assert_match "\"clean_reports\": 1" "$(cat "${pack_dir}/results/verification_summary.json")" "expected-pass count recorded"
    assert_match "\"expected_failure_reports\": 1" "$(cat "${pack_dir}/results/verification_summary.json")" "scenario expected-failure count recorded"
    assert_match "\"failed_reports\": 0" "$(cat "${pack_dir}/results/verification_summary.json")" "failed count recorded"
}

test_run_pack_build_pack_accepts_informational_error_probe_that_verifies_clean() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/analysis" "${run_dir}/state"
    mkdir -p "${run_dir}/modelA/reports/quant_4bit_clean/run_1"
    mkdir -p "${run_dir}/modelA/reports/errors/rmt_norm_noise_probe"

    echo "verdict" > "${run_dir}/reports/final_verdict.txt"
    echo '{"verdict":"PASS"}' > "${run_dir}/reports/final_verdict.json"
    echo '{"model_list": ["org/model"], "models": {"org/model": {"revision": "abc"}}}' > "${run_dir}/state/model_revisions.json"
    cat > "${run_dir}/state/scenarios.json" <<'JSON'
{
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {"id": "quant_4bit_clean", "strictness": "must_pass"},
    {"id": "rmt_norm_noise_probe", "strictness": "informational"}
  ]
}
JSON
    echo "{}" > "${run_dir}/modelA/reports/quant_4bit_clean/run_1/evaluation.report.json"
    echo "{}" > "${run_dir}/modelA/reports/errors/rmt_norm_noise_probe/evaluation.report.json"

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

    PACK_SIGN_MANIFEST=0

    local pack_dir="${TEST_TMPDIR}/pack"
    pack_build_pack "${run_dir}" "${pack_dir}"

    assert_file_exists "${pack_dir}/reports/modelA/quant_4bit_clean/run_1/verify.json" "expected-pass verify output captured"
    assert_file_exists "${pack_dir}/reports/modelA/errors/rmt_norm_noise_probe/verify.json" "informational probe verify output captured"
    assert_match "\"clean_reports\": 1" "$(cat "${pack_dir}/results/verification_summary.json")" "expected-pass count recorded"
    assert_match "\"error_injection_reports\": 1" "$(cat "${pack_dir}/results/verification_summary.json")" "informational probe is not counted as clean"
    assert_match "\"failed_reports\": 0" "$(cat "${pack_dir}/results/verification_summary.json")" "failed count recorded"
}

test_run_pack_build_pack_rejects_expected_failure_report_that_verifies_clean() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/analysis" "${run_dir}/state"
    mkdir -p "${run_dir}/modelA/reports/prune_50pct_stress/run_1"

    echo "verdict" > "${run_dir}/reports/final_verdict.txt"
    echo '{"verdict":"PASS"}' > "${run_dir}/reports/final_verdict.json"
    echo '{"model_list": ["org/model"], "models": {"org/model": {"revision": "abc"}}}' > "${run_dir}/state/model_revisions.json"
    cat > "${run_dir}/state/scenarios.json" <<'JSON'
{
  "schema": "evidence_pack_scenarios_v1",
  "schema_version": 1,
  "scenarios": [
    {"id": "prune_50pct_stress", "strictness": "must_fail"}
  ]
}
JSON
    echo "{}" > "${run_dir}/modelA/reports/prune_50pct_stress/run_1/evaluation.report.json"

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

    PACK_SIGN_MANIFEST=0

    local pack_dir="${TEST_TMPDIR}/pack"
    run pack_build_pack "${run_dir}" "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "expected-failure report verifying clean fails pack build"
    assert_match "Expected verify failure verified as passing" "${RUN_ERR}" "unexpected expected-failure pass is explicit"
}

test_run_pack_build_pack_continues_when_html_report_fails() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/analysis" "${run_dir}/state"
    mkdir -p "${run_dir}/modelA/reports/edit/run_1"

    echo "verdict" > "${run_dir}/reports/final_verdict.txt"
    echo "{}" > "${run_dir}/reports/final_verdict.json"
    echo '{"model_list": ["org/model"], "models": {"org/model": {"revision": "abc"}}}' > "${run_dir}/state/model_revisions.json"
    echo '{"schema":"evidence_pack_scenarios_v1","schema_version":1,"scenarios":[]}' > "${run_dir}/state/scenarios.json"
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
            [[ "${arg}" != "--allow-unverified-provenance" ]] || exit 97
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

    source ./scripts/evidence_packs/run_pack.sh

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
            [[ "${arg}" != "--allow-unverified-provenance" ]] || exit 97
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
