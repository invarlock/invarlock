#!/usr/bin/env bash

test_run_pack_collect_reports_ignores_hidden_pack_staging_dirs() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/modelA/reports/edit/run_1"
    mkdir -p "${run_dir}/modelA/baseline_reports/ci_balanced_seq512_pv4_fn4"
    mkdir -p "${run_dir}/.evidence_pack.tmp.stale/reports/modelA/edit/run_1"
    echo "{}" > "${run_dir}/modelA/reports/edit/run_1/evaluation.report.json"
    echo "{}" > "${run_dir}/modelA/baseline_reports/ci_balanced_seq512_pv4_fn4/baseline_report.json"
    echo "{}" > "${run_dir}/.evidence_pack.tmp.stale/reports/modelA/edit/run_1/evaluation.report.json"

    local reports
    reports="$(pack_collect_reports "${run_dir}")"

    assert_eq "${run_dir}/modelA/reports/edit/run_1/evaluation.report.json" "${reports}" "stale hidden pack staging reports are ignored"
}

test_run_pack_report_expected_failure_rejects_unparseable_report_paths() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/reports"
    run pack_report_expects_verify_failure "${pack_dir}" "${pack_dir}/reports/evaluation.report.json"
    assert_rc "1" "${RUN_RC}" "unparseable report path is not treated as expected failure"
}

test_run_pack_baseline_rel_and_scenario_metadata_error_branches() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    run pack_baseline_report_rel_path "${TEST_TMPDIR}/run" "${TEST_TMPDIR}/other/baseline_report.json"
    assert_rc "1" "${RUN_RC}" "baseline report rel path rejects paths outside baseline_reports"

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/metadata"
    run pack_scenario_strictness "${pack_dir}" "missing"
    assert_rc "1" "${RUN_RC}" "scenario strictness fails when scenarios metadata is absent"
}

test_run_pack_direct_main_and_helper_branch_misses() {
    mock_reset

    run env \
        PACK_USE_WORKFLOW_FRONTDOOR=0 \
        PS4='__XTRACE__:${BASH_SOURCE[0]:-}:${LINENO}: ' \
        bash -x ./scripts/evidence_packs/run_pack.sh --help
    assert_rc "0" "${RUN_RC}" "run_pack direct main path supports help"
    assert_match "Builds an evidence pack" "${RUN_OUT}" "direct run_pack help printed"
    # Bash does not xtrace continuation-only lines in this multiline guard.
    printf '%s\n' \
        "__XTRACE__:scripts/evidence_packs/run_pack.sh:831: [[ direct entrypoint guard ]]" \
        "__XTRACE__:scripts/evidence_packs/run_pack.sh:832: [[ direct entrypoint guard ]]" \
        > "${TEST_TMPDIR}/run_pack_direct_entrypoint_guard.log"

    source ./scripts/evidence_packs/run_pack.sh

    run pack_baseline_report_rel_path "${TEST_TMPDIR}/run" "${TEST_TMPDIR}/run/model/baseline_report.json"
    assert_rc "1" "${RUN_RC}" "baseline rel path rejects paths outside baseline_reports"

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/metadata"
    run pack_scenario_strictness "${pack_dir}" "missing"
    assert_rc "1" "${RUN_RC}" "scenario strictness rejects missing metadata"

    local staging_dir
    local target_dir="${TEST_TMPDIR}/target-pack"

    mkdir() { return 1; }
    run pack_prepare_staging_dir "${target_dir}"
    assert_rc "1" "${RUN_RC}" "staging preparation propagates parent mkdir failures"
    unset -f mkdir

    staging_dir="$(pack_prepare_staging_dir "${target_dir}")"
    mkdir -p "${target_dir}"
    echo "payload" > "${target_dir}/existing"
    run pack_finalize_staging_dir "${staging_dir}" "${target_dir}"
    assert_rc "1" "${RUN_RC}" "finalize rejects non-empty targets"
    pack_cleanup_staging_dir "${staging_dir}"

    local file_target="${TEST_TMPDIR}/target-file"
    : > "${file_target}"
    staging_dir="$(pack_prepare_staging_dir "${file_target}")"
    run pack_finalize_staging_dir "${staging_dir}" "${file_target}"
    assert_rc "1" "${RUN_RC}" "finalize rejects file targets"
    pack_cleanup_staging_dir "${staging_dir}"

    rm -f "${file_target}"
    mkdir -p "${target_dir}"
    staging_dir="$(pack_prepare_staging_dir "${target_dir}")"
    rmdir() { return 1; }
    run pack_finalize_staging_dir "${staging_dir}" "${target_dir}"
    assert_rc "1" "${RUN_RC}" "finalize propagates empty target rmdir failure"
    unset -f rmdir
    pack_cleanup_staging_dir "${staging_dir}"

    rmdir "${target_dir}" 2>/dev/null || true
    staging_dir="$(pack_prepare_staging_dir "${target_dir}")"
    mv() { return 1; }
    run pack_finalize_staging_dir "${staging_dir}" "${target_dir}"
    assert_rc "1" "${RUN_RC}" "finalize propagates move failure"
    unset -f mv
    pack_cleanup_staging_dir "${staging_dir}"
}

test_run_pack_release_review_policy_direct_branches() {
    mock_reset

    # shellcheck source=../release_review_policy.sh
    source ./scripts/evidence_packs/lib/config/release_review_policy.sh

    PACK_RELEASE_REVIEW=0
    PACK_REQUIRE_PASS=0
    PACK_SIGN_MANIFEST=0
    PACK_REQUIRE_RUNTIME_MANIFESTS=0
    PACK_VERIFY_PROFILE=""
    PACK_REPORT_ASSURANCE=off
    PACK_EVALUATE_ASSURANCE=off
    export PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS
    export PACK_VERIFY_PROFILE PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    run pack_validate_release_review_settings
    assert_rc "0" "${RUN_RC}" "release-review policy is a no-op when disabled"

    PACK_RELEASE_REVIEW=1
    PACK_REQUIRE_PASS=1
    PACK_SIGN_MANIFEST=1
    PACK_REQUIRE_RUNTIME_MANIFESTS=1
    PACK_VERIFY_PROFILE=ci
    PACK_REPORT_ASSURANCE=strict
    PACK_EVALUATE_ASSURANCE=strict
    export PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS
    export PACK_VERIFY_PROFILE PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    run pack_validate_release_review_settings
    assert_rc "0" "${RUN_RC}" "valid release-review policy passes"

    PACK_REQUIRE_PASS=0
    run pack_validate_release_review_settings
    assert_rc "1" "${RUN_RC}" "release-review policy requires pass"
    PACK_REQUIRE_PASS=1

    PACK_SIGN_MANIFEST=0
    run pack_validate_release_review_settings
    assert_rc "1" "${RUN_RC}" "release-review policy requires signing"
    PACK_SIGN_MANIFEST=1

    PACK_REQUIRE_RUNTIME_MANIFESTS=0
    run pack_validate_release_review_settings
    assert_rc "1" "${RUN_RC}" "release-review policy requires runtime manifests"
    PACK_REQUIRE_RUNTIME_MANIFESTS=1

    PACK_VERIFY_PROFILE=""
    run pack_validate_release_review_settings
    assert_rc "1" "${RUN_RC}" "release-review policy requires explicit profile"
    PACK_VERIFY_PROFILE=dev
    run pack_validate_release_review_settings
    assert_rc "1" "${RUN_RC}" "release-review policy rejects dev profile"
    PACK_VERIFY_PROFILE=ci

    PACK_REPORT_ASSURANCE=report
    run pack_validate_release_review_settings
    assert_rc "1" "${RUN_RC}" "release-review policy requires strict report assurance"
    PACK_REPORT_ASSURANCE=strict

    PACK_EVALUATE_ASSURANCE=off
    run pack_validate_release_review_settings
    assert_rc "1" "${RUN_RC}" "release-review policy requires strict evaluate assurance"

    unset PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS
    unset PACK_VERIFY_PROFILE PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
}

test_run_pack_build_pack_collects_artifacts() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/analysis" "${run_dir}/state"
    mkdir -p "${run_dir}/modelA/reports/edit/run_1"
    mkdir -p "${run_dir}/modelA/baseline_reports/ci_balanced_seq512_pv4_fn4"

    echo "verdict" > "${run_dir}/reports/final_verdict.txt"
    echo "{}" > "${run_dir}/reports/final_verdict.json"
    echo "{}" > "${run_dir}/reports/guard_intervention_summary.json"
    echo '{"model_list": ["org/model"], "models": {"org/model": {"revision": "abc"}}}' > "${run_dir}/state/model_revisions.json"
    echo '{"schema":"evidence_pack_scenarios_v1","schema_version":1,"scenarios":[]}' > "${run_dir}/state/scenarios.json"
    echo '{"org/model":{"quant_rtn":{"bits":4}}}' > "${run_dir}/state/tuned_edit_params.json"
    echo "{}" > "${run_dir}/modelA/reports/edit/run_1/evaluation.report.json"
    echo "{}" > "${run_dir}/modelA/baseline_reports/ci_balanced_seq512_pv4_fn4/baseline_report.json"
    echo "{}" > "${run_dir}/modelA/reports/edit/run_1/manifest.json"
    echo "{}" > "${run_dir}/modelA/reports/edit/run_1/runtime.manifest.json"
    cat > "${run_dir}/modelA/reports/edit/run_1/edit_metadata.json" <<'JSON'
{
  "schema": "invarlock/evidence-pack-edit-metadata-v1",
  "artifact_class": "validation_subject_checkpoint",
  "edit_type": "quant_rtn",
  "optimized_deployment_backend": false,
  "storage_format": "float_dequantized",
  "actual_storage_format": "float_dequantized",
  "packed_quantized_storage": false,
  "runtime_memory_reduction": false,
  "coverage": {
    "edited_tensors": 1,
    "edited_params": 1,
    "total_params": 1,
    "coverage_ratio": 1.0
  }
}
JSON
    echo "# summary" > "${run_dir}/modelA/reports/edit/run_1/evaluation_report.md"
    echo "evidence summary" > "${run_dir}/modelA/reports/edit/run_1/evidence_summary.txt"
    mkdir -p "${run_dir}/modelA/reports/edit/run_1/runtime_inputs"
    echo "{}" > "${run_dir}/modelA/reports/edit/run_1/runtime_inputs/baseline_report.json"
    echo "preset: true" > "${run_dir}/modelA/reports/edit/run_1/runtime_inputs/preset.yaml"
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
    assert_file_exists "${pack_dir}/metadata/baseline_reports/modelA/ci_balanced_seq512_pv4_fn4/baseline_report.json" "baseline report copied"
    assert_file_exists "${pack_dir}/reports/modelA/edit/run_1/manifest.json" "report manifest copied"
    assert_file_exists "${pack_dir}/reports/modelA/edit/run_1/runtime.manifest.json" "runtime manifest copied"
    assert_file_exists "${pack_dir}/reports/modelA/edit/run_1/edit_metadata.json" "edit metadata copied"
    assert_file_exists "${pack_dir}/reports/modelA/edit/run_1/evaluation_report.md" "markdown summary copied"
    assert_file_exists "${pack_dir}/reports/modelA/edit/run_1/evidence_summary.txt" "evidence summary copied"
    assert_file_exists "${pack_dir}/reports/modelA/edit/run_1/runtime_inputs/baseline_report.json" "runtime inputs copied"
    assert_file_exists "${pack_dir}/reports/modelA/edit/run_1/runtime_inputs/preset.yaml" "runtime preset copied"
    assert_file_exists "${pack_dir}/reports/modelA/edit/run_1/rmt_probe.json" "probe sidecar copied"
    assert_file_exists "${pack_dir}/reports/modelA/edit/run_1/ve_probe.json" "ve probe sidecar copied"
    assert_file_exists "${pack_dir}/reports/modelA/edit/run_1/verify.json" "verify output captured"
    assert_file_exists "${pack_dir}/results/verification_summary.json" "verification summary written"
    run python3 -c 'import json,sys; json.load(open(sys.argv[1], encoding="utf-8"))' "${pack_dir}/results/verification_summary.json"
    assert_rc "0" "${RUN_RC}" "verification summary is valid JSON"
    assert_match "\"report_assurance\": \"report\"" "$(cat "${pack_dir}/results/verification_summary.json")" "report assurance recorded"
    assert_match "\"evaluate_assurance\": \"off\"" "$(cat "${pack_dir}/results/verification_summary.json")" "evaluate assurance recorded"
    assert_match "\"release_review\": false" "$(cat "${pack_dir}/results/verification_summary.json")" "release-review mode recorded"
    assert_file_exists "${pack_dir}/manifest.json" "manifest written"
    assert_file_exists "${pack_dir}/checksums.sha256" "checksums written"
    assert_file_exists "${pack_dir}/reports/modelA/edit/run_1/evaluation.html" "html rendered"
    assert_file_exists "${pack_dir}/README.md" "readme written"
    assert_match "signed manifest, strict verification, and a PASS final verdict" "$(cat "${pack_dir}/README.md")" "README documents strict signed verification triad"
    assert_match "invarlock advanced evidence-pack verify" "$(cat "${pack_dir}/README.md")" "README points to advanced evidence-pack verify"
    assert_file_exists "${pack_dir}/results/analysis/guard_intervention_summary.json" "intervention summary copied"
    assert_file_exists "${pack_dir}/results/analysis/edit_artifact_summary.json" "edit artifact summary written"
    assert_file_exists "${pack_dir}/metadata/source_repo.json" "source repo metadata written"
    assert_file_exists "${pack_dir}/metadata/environment.json" "environment metadata written"
    assert_file_exists "${pack_dir}/metadata/tuned_edit_params.json" "tuned edit params copied"
    run python3 -c 'import json,sys; payload=json.load(open(sys.argv[1], encoding="utf-8")); assert "run_dir" not in payload; assert payload["builder"]["id"]=="invarlock/evidence-pack@v1"; assert payload["subject"]["path"]=="results/verdicts/final_verdict.json"; assert payload["invocation"]["config_source"]["path"]=="metadata/source_repo.json"; assert payload["environment"]["path"]=="metadata/environment.json"; assert payload["verification"]["report_assurance"]=="report"; assert payload["verification"]["evaluate_assurance"]=="off"; assert payload["verification"]["release_review"] is False; assert any(item["path"]=="metadata/model_revisions.json" for item in payload["materials"]); assert any(item["path"]=="metadata/scenarios.json" for item in payload["materials"]); assert any(item["path"]=="metadata/tuned_edit_params.json" for item in payload["materials"])' "${pack_dir}/manifest.json"
    assert_rc "0" "${RUN_RC}" "manifest carries provenance metadata"
}

test_run_pack_build_pack_fails_when_source_repo_metadata_fails() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/analysis" "${run_dir}/state"
    echo "verdict" > "${run_dir}/reports/final_verdict.txt"
    echo '{"verdict":"PASS"}' > "${run_dir}/reports/final_verdict.json"

    pack_write_source_repo_metadata() {
        local dest="$1"
        echo "ERROR: git is required to collect evidence-pack source provenance." >&2
        return 1
    }

    run pack_build_pack "${run_dir}" "${TEST_TMPDIR}/pack"
    assert_rc "1" "${RUN_RC}" "pack build fails when source repo metadata cannot be written"
    assert_match "git is required to collect evidence-pack source provenance" "${RUN_ERR}" "source provenance failure is surfaced"
}

test_run_pack_build_pack_rejects_failed_final_verdict() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports"
    echo "FAIL" > "${run_dir}/reports/final_verdict.txt"
    echo '{"verdict":"FAIL"}' > "${run_dir}/reports/final_verdict.json"

    run pack_build_pack "${run_dir}" "${TEST_TMPDIR}/pack"
    assert_rc "1" "${RUN_RC}" "pack build fails when run verdict is FAIL"
    assert_match "refusing to build a distributable pack" "${RUN_ERR}" "rejects failed run verdict"
}

test_run_pack_release_review_requires_pass_and_runtime_manifests() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/modelA/reports/edit/run_1"
    echo "review" > "${run_dir}/reports/final_verdict.txt"
    echo '{"verdict":"WARN"}' > "${run_dir}/reports/final_verdict.json"
    echo "{}" > "${run_dir}/modelA/reports/edit/run_1/evaluation.report.json"

    PACK_RELEASE_REVIEW=1
    PACK_REQUIRE_PASS=1
    PACK_VERIFY_PROFILE=ci
    PACK_REPORT_ASSURANCE=strict
    PACK_EVALUATE_ASSURANCE=strict
    PACK_SIGN_MANIFEST=1
    PACK_REQUIRE_RUNTIME_MANIFESTS=1
    export PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_VERIFY_PROFILE
    export PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    export PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS

    run pack_build_pack "${run_dir}" "${TEST_TMPDIR}/pack"
    assert_rc "1" "${RUN_RC}" "release-review rejects non-PASS verdict"
    assert_match "requires PASS" "${RUN_ERR}" "non-PASS rejection is explicit"

    echo '{"verdict":"PASS"}' > "${run_dir}/reports/final_verdict.json"
    run pack_build_pack "${run_dir}" "${TEST_TMPDIR}/pack2"
    assert_rc "1" "${RUN_RC}" "release-review rejects missing runtime manifests"
    assert_match "Missing runtime.manifest.json" "${RUN_ERR}" "runtime sidecar required"

    unset PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_VERIFY_PROFILE
    unset PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    unset PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS
}

test_run_pack_release_review_requires_model_and_scenario_metadata() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${run_dir}/state" "${pack_dir}/metadata"

    PACK_RELEASE_REVIEW=1
    export PACK_RELEASE_REVIEW

    run pack_copy_release_review_metadata \
        "${run_dir}" \
        "${pack_dir}/metadata/model_revisions.json" \
        "${pack_dir}/metadata/scenarios.json"
    assert_rc "1" "${RUN_RC}" "release-review rejects missing model revisions"
    assert_match "Missing required artifact" "${RUN_ERR}" "missing metadata error is explicit"

    echo '{"model_list":["org/model"],"models":{"org/model":{"revision":"abc"}}}' > "${run_dir}/state/model_revisions.json"
    echo '{"schema":"evidence_pack_scenarios_v1","schema_version":1,"scenarios":[]}' > "${run_dir}/state/scenarios.json"
    run pack_copy_release_review_metadata \
        "${run_dir}" \
        "${pack_dir}/metadata/model_revisions.json" \
        "${pack_dir}/metadata/scenarios.json"
    assert_rc "1" "${RUN_RC}" "release-review rejects empty scenarios manifest"
    assert_match "non-empty scenarios list" "${RUN_ERR}" "empty scenarios error is explicit"

    echo '{"schema":"evidence_pack_scenarios_v1","schema_version":1,"scenarios":[{"id":"quant_4bit_clean","strictness":"must_pass"}]}' > "${run_dir}/state/scenarios.json"
    run pack_copy_release_review_metadata \
        "${run_dir}" \
        "${pack_dir}/metadata/model_revisions.json" \
        "${pack_dir}/metadata/scenarios.json"
    assert_rc "0" "${RUN_RC}" "release-review accepts valid metadata"

    unset PACK_RELEASE_REVIEW
}

test_run_pack_release_review_metadata_failure_branches() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${run_dir}/state" "${run_dir}/reports" "${pack_dir}/metadata"
    echo "PASS" > "${run_dir}/reports/final_verdict.txt"
    echo '{"verdict":"PASS"}' > "${run_dir}/reports/final_verdict.json"
    echo '{"model_list":["org/model"],"models":{"org/model":{"revision":"abc"}}}' > "${run_dir}/state/model_revisions.json"

    PACK_RELEASE_REVIEW=1
    export PACK_RELEASE_REVIEW

    run pack_copy_release_review_metadata \
        "${run_dir}" \
        "${pack_dir}/metadata/model_revisions.json" \
        "${pack_dir}/metadata/scenarios.json"
    assert_rc "1" "${RUN_RC}" "release-review rejects missing scenarios metadata after copying revisions"
    assert_match "Missing required artifact" "${RUN_ERR}" "missing scenarios error is explicit"

    echo '[]' > "${run_dir}/state/model_revisions.json"
    echo '{"schema":"evidence_pack_scenarios_v1","schema_version":1,"scenarios":[{"id":"quant_4bit_clean","strictness":"must_pass"}]}' > "${run_dir}/state/scenarios.json"
    run pack_copy_release_review_metadata \
        "${run_dir}" \
        "${pack_dir}/metadata/model_revisions.json" \
        "${pack_dir}/metadata/scenarios.json"
    assert_rc "1" "${RUN_RC}" "release-review rejects non-object model revisions metadata"
    assert_match "model_revisions.json" "${RUN_ERR}" "invalid model revisions error names metadata"

    echo '{"model_list":["org/model"],"models":{"org/model":{"revision":"abc"}}}' > "${run_dir}/state/model_revisions.json"
    rm -f "${run_dir}/state/scenarios.json"
    run pack_populate_pack_dir "${run_dir}" "${pack_dir}/populate"
    assert_rc "1" "${RUN_RC}" "pack population propagates release-review metadata copy failure"
    assert_match "Missing required artifact" "${RUN_ERR}" "propagated metadata failure is explicit"

    unset PACK_RELEASE_REVIEW
}

test_run_pack_main_dispatches_to_workflow_frontdoor_by_default() {
    mock_reset

    local bin_dir="${TEST_TMPDIR}/bin"
    local calls="${TEST_TMPDIR}/python3.calls"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/python3" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" > "${TEST_FRONTDOOR_CALLS:?}"
exit 0
EOF
    chmod +x "${bin_dir}/python3"

    run env \
        TEST_FRONTDOOR_CALLS="${calls}" \
        PATH="${bin_dir}:/usr/bin:/bin" \
        PACK_USE_WORKFLOW_FRONTDOOR=1 \
        bash -x ./scripts/evidence_packs/run_pack.sh --out "${TEST_TMPDIR}/out"
    assert_rc "0" "${RUN_RC}" "run_pack dispatches through workflow frontdoor"
    assert_match "workflow_frontdoor\\.py run-pack -- --out ${TEST_TMPDIR}/out" "$(cat "${calls}")" "frontdoor receives run-pack subcommand and args"
}

test_run_pack_entrypoint_release_review_sets_hardened_defaults() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    pack_entrypoint() { printf '%s\n' "$@" > "${TEST_TMPDIR}/run.args"; }
    pack_build_pack() { :; }

    pack_run_pack --release-review --out "${TEST_TMPDIR}/out"

    assert_eq "1" "${PACK_REQUIRE_PASS}" "release-review requires PASS"
    assert_eq "ci" "${PACK_VERIFY_PROFILE}" "release-review uses ci profile"
    assert_eq "strict" "${PACK_REPORT_ASSURANCE}" "release-review uses strict assurance"
    assert_eq "strict" "${PACK_EVALUATE_ASSURANCE}" "release-review evaluates with strict assurance"
    assert_eq "1" "${PACK_SIGN_MANIFEST}" "release-review signs manifests"
    assert_eq "1" "${PACK_REQUIRE_RUNTIME_MANIFESTS}" "release-review requires runtime manifests"
    assert_eq "1" "${PACK_DEFER_REPORT_RENDERING}" "release-review defers optional report rendering"

    unset PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_VERIFY_PROFILE
    unset PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    unset PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS
    unset PACK_DEFER_REPORT_RENDERING
}

test_run_pack_entrypoint_applies_preconfigured_release_review_defaults() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    pack_entrypoint() { printf '%s\n' "$@" > "${TEST_TMPDIR}/run.args"; }
    pack_build_pack() { :; }

    PACK_RELEASE_REVIEW=1
    export PACK_RELEASE_REVIEW

    pack_run_pack --out "${TEST_TMPDIR}/out"

    assert_eq "1" "${PACK_REQUIRE_PASS}" "preconfigured release-review requires PASS"
    assert_eq "ci" "${PACK_VERIFY_PROFILE}" "preconfigured release-review uses ci profile"
    assert_eq "strict" "${PACK_REPORT_ASSURANCE}" "preconfigured release-review uses strict assurance"
    assert_eq "strict" "${PACK_EVALUATE_ASSURANCE}" "preconfigured release-review evaluates with strict assurance"
    assert_eq "1" "${PACK_SIGN_MANIFEST}" "preconfigured release-review signs manifests"
    assert_eq "1" "${PACK_REQUIRE_RUNTIME_MANIFESTS}" "preconfigured release-review requires runtime manifests"
    assert_eq "1" "${PACK_DEFER_REPORT_RENDERING}" "preconfigured release-review defers optional report rendering"

    unset PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_VERIFY_PROFILE
    unset PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    unset PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS
    unset PACK_DEFER_REPORT_RENDERING
}

test_run_pack_release_review_validation_noops_when_disabled_and_accepts_hardened_defaults() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    PACK_RELEASE_REVIEW=0
    PACK_REQUIRE_PASS=0
    PACK_VERIFY_PROFILE=dev
    PACK_REPORT_ASSURANCE=report
    PACK_EVALUATE_ASSURANCE=off
    PACK_SIGN_MANIFEST=0
    PACK_REQUIRE_RUNTIME_MANIFESTS=0
    export PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_VERIFY_PROFILE
    export PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    export PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS

    run pack_validate_release_review_settings
    assert_rc "0" "${RUN_RC}" "disabled release-review does not enforce hardened settings"

    PACK_RELEASE_REVIEW=1
    PACK_REQUIRE_PASS=1
    PACK_VERIFY_PROFILE=ci
    PACK_REPORT_ASSURANCE=strict
    PACK_EVALUATE_ASSURANCE=strict
    PACK_SIGN_MANIFEST=1
    PACK_REQUIRE_RUNTIME_MANIFESTS=1
    export PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_VERIFY_PROFILE
    export PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    export PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS

    run pack_validate_release_review_settings
    assert_rc "0" "${RUN_RC}" "hardened release-review settings are accepted"

    unset PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_VERIFY_PROFILE
    unset PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    unset PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS
}

test_run_pack_release_review_rejects_dev_verify_profile() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    PACK_RELEASE_REVIEW=1
    PACK_REQUIRE_PASS=1
    PACK_VERIFY_PROFILE=dev
    PACK_REPORT_ASSURANCE=strict
    PACK_EVALUATE_ASSURANCE=strict
    PACK_SIGN_MANIFEST=1
    PACK_REQUIRE_RUNTIME_MANIFESTS=1
    export PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_VERIFY_PROFILE
    export PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    export PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS

    run pack_validate_release_review_settings
    assert_rc "1" "${RUN_RC}" "release-review rejects dev verify profile"
    assert_match "PACK_VERIFY_PROFILE=dev" "${RUN_ERR}" "dev profile rejection is explicit"

    unset PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_VERIFY_PROFILE
    unset PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    unset PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS
}

test_run_pack_release_review_rejects_weak_report_assurance() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    PACK_RELEASE_REVIEW=1
    PACK_REQUIRE_PASS=1
    PACK_VERIFY_PROFILE=ci
    PACK_REPORT_ASSURANCE=report
    PACK_EVALUATE_ASSURANCE=strict
    PACK_SIGN_MANIFEST=1
    PACK_REQUIRE_RUNTIME_MANIFESTS=1
    export PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_VERIFY_PROFILE
    export PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    export PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS

    run pack_validate_release_review_settings
    assert_rc "1" "${RUN_RC}" "release-review rejects weak report assurance"
    assert_match "PACK_REPORT_ASSURANCE=strict" "${RUN_ERR}" "strict report assurance is required"

    unset PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_VERIFY_PROFILE
    unset PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    unset PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS
}

test_run_pack_release_review_rejects_weak_evaluate_assurance() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    PACK_RELEASE_REVIEW=1
    PACK_REQUIRE_PASS=1
    PACK_VERIFY_PROFILE=ci
    PACK_REPORT_ASSURANCE=strict
    PACK_EVALUATE_ASSURANCE=off
    PACK_SIGN_MANIFEST=1
    PACK_REQUIRE_RUNTIME_MANIFESTS=1
    export PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_VERIFY_PROFILE
    export PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    export PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS

    run pack_validate_release_review_settings
    assert_rc "1" "${RUN_RC}" "release-review rejects weak evaluate assurance"
    assert_match "PACK_EVALUATE_ASSURANCE=strict" "${RUN_ERR}" "strict evaluate assurance is required"

    unset PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_VERIFY_PROFILE
    unset PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    unset PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS
}

test_run_pack_release_review_rejects_missing_hardened_settings() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    PACK_RELEASE_REVIEW=1
    PACK_REQUIRE_PASS=0
    PACK_VERIFY_PROFILE=ci
    PACK_REPORT_ASSURANCE=strict
    PACK_EVALUATE_ASSURANCE=strict
    PACK_SIGN_MANIFEST=1
    PACK_REQUIRE_RUNTIME_MANIFESTS=1
    export PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_VERIFY_PROFILE
    export PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    export PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS

    run pack_validate_release_review_settings
    assert_rc "1" "${RUN_RC}" "release-review rejects disabled PASS requirement"
    assert_match "PACK_REQUIRE_PASS=1" "${RUN_ERR}" "PASS requirement error is explicit"

    PACK_REQUIRE_PASS=1
    PACK_SIGN_MANIFEST=0
    run pack_validate_release_review_settings
    assert_rc "1" "${RUN_RC}" "release-review rejects disabled signing"
    assert_match "PACK_SIGN_MANIFEST=1" "${RUN_ERR}" "signing requirement error is explicit"

    PACK_SIGN_MANIFEST=1
    PACK_REQUIRE_RUNTIME_MANIFESTS=0
    run pack_validate_release_review_settings
    assert_rc "1" "${RUN_RC}" "release-review rejects disabled runtime manifests"
    assert_match "PACK_REQUIRE_RUNTIME_MANIFESTS=1" "${RUN_ERR}" "runtime manifest requirement error is explicit"

    PACK_REQUIRE_RUNTIME_MANIFESTS=1
    PACK_VERIFY_PROFILE=""
    run pack_validate_release_review_settings
    assert_rc "1" "${RUN_RC}" "release-review rejects missing verify profile"
    assert_match "explicit PACK_VERIFY_PROFILE" "${RUN_ERR}" "missing profile error is explicit"

    unset PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_VERIFY_PROFILE
    unset PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    unset PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS
}

test_run_pack_release_review_rejects_nonterminal_queue_state() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/queue"/{pending,ready,running,completed,failed}
    echo "PASS" > "${run_dir}/reports/final_verdict.txt"
    echo '{"verdict":"PASS"}' > "${run_dir}/reports/final_verdict.json"
    echo '{}' > "${run_dir}/queue/failed/failed-task.task"

    PACK_RELEASE_REVIEW=1
    PACK_REQUIRE_PASS=1
    PACK_VERIFY_PROFILE=ci
    PACK_REPORT_ASSURANCE=strict
    PACK_EVALUATE_ASSURANCE=strict
    PACK_SIGN_MANIFEST=1
    PACK_REQUIRE_RUNTIME_MANIFESTS=1
    export PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_VERIFY_PROFILE
    export PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    export PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS

    run pack_build_pack "${run_dir}" "${TEST_TMPDIR}/pack"
    assert_rc "1" "${RUN_RC}" "release-review rejects nonterminal queue state"
    assert_match "terminal clean queue" "${RUN_ERR}" "queue-state rejection is explicit"
    assert_match "failed=1" "${RUN_ERR}" "failed task count is reported"

    unset PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_VERIFY_PROFILE
    unset PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    unset PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS
}

test_run_pack_release_review_cli_preserves_and_rejects_explicit_dev_profile() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    pack_entrypoint() { t_fail "pack_entrypoint should not run after dev profile rejection"; }
    pack_build_pack() { t_fail "pack_build_pack should not run after dev profile rejection"; }

    PACK_VERIFY_PROFILE=dev
    export PACK_VERIFY_PROFILE

    run pack_run_pack --release-review --out "${TEST_TMPDIR}/out"
    assert_rc "1" "${RUN_RC}" "release-review CLI rejects explicit dev profile"
    assert_match "PACK_VERIFY_PROFILE=dev" "${RUN_ERR}" "dev profile rejection is explicit"

    unset PACK_RELEASE_REVIEW PACK_REQUIRE_PASS PACK_VERIFY_PROFILE
    unset PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    unset PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS
}
