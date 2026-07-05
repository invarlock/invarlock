#!/usr/bin/env bash

test_run_pack_checksums_include_files() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}/results/verdicts"
    echo "verdict" > "${pack_dir}/results/verdicts/final_verdict.txt"
    echo "{}" > "${pack_dir}/manifest.json"
    mkdir -p "${pack_dir}/metadata" "${pack_dir}/__MACOSX"
    echo "{}" > "${pack_dir}/metadata/manifest.json"
    echo "sig" > "${pack_dir}/metadata/manifest.signature.json"
    echo "x" > "${pack_dir}/metadata/checksums.sha256"
    echo "junk" > "${pack_dir}/.DS_Store"
    echo "junk" > "${pack_dir}/._results"
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
    if [[ "${checksums}" == *._results* ]]; then
        t_fail "checksums must ignore AppleDouble artifacts"
    fi
    if [[ "${checksums}" == *__MACOSX* ]]; then
        t_fail "checksums must ignore __MACOSX artifacts"
    fi
}


test_run_pack_helpers_cover_error_paths() {

    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    run pack_require_cmd definitely_missing_cmd
    assert_rc "1" "${RUN_RC}" "missing command returns non-zero"

    run pack_copy_file "${TEST_TMPDIR}/missing.txt" "${TEST_TMPDIR}/dest.txt"
    assert_rc "1" "${RUN_RC}" "missing artifact returns non-zero"

    run pack_report_rel_path "${TEST_TMPDIR}/run" "${TEST_TMPDIR}/nope"
    assert_rc "1" "${RUN_RC}" "invalid report path returns non-zero"

    local pack_dir="${TEST_TMPDIR}/pack"
    run pack_report_scenario_id "${pack_dir}" "${pack_dir}/reports/evaluation.report.json"
    assert_rc "1" "${RUN_RC}" "malformed pack report path has no scenario id"

    mkdir -p "${pack_dir}/metadata" "${pack_dir}/reports/model/clean"
    printf '%s\n' '{"scenarios":[{"id":"clean","strictness":"must_pass"},{"id":"nan_injection","strictness":"must_detect"},{"id":"shape_mismatch","strictness":"must_fail"}]}' > "${pack_dir}/metadata/scenarios.json"
    echo "{}" > "${pack_dir}/reports/model/clean/evaluation.report.json"
    run pack_scenario_strictness "${pack_dir}" "clean"
    assert_rc "0" "${RUN_RC}" "scenario strictness resolves from metadata"
    assert_eq "must_pass" "${RUN_OUT}" "scenario strictness value returned"

    run pack_report_expects_verify_failure "${pack_dir}" "${pack_dir}/outside/evaluation.report.json"
    assert_rc "1" "${RUN_RC}" "malformed report path cannot infer expected failure"

    mkdir -p "${pack_dir}/reports/model/errors/nan_injection"
    run pack_report_expects_verify_failure "${pack_dir}" "${pack_dir}/reports/model/errors/nan_injection/evaluation.report.json"
    assert_rc "1" "${RUN_RC}" "must_detect error-injection report may verify clean"

    mkdir -p "${pack_dir}/reports/model/errors/shape_mismatch"
    run pack_report_expects_verify_failure "${pack_dir}" "${pack_dir}/reports/model/errors/shape_mismatch/evaluation.report.json"
    assert_rc "0" "${RUN_RC}" "must_fail error-injection report expects verify failure"

    local empty_pack_dir="${TEST_TMPDIR}/empty-pack"
    mkdir -p "${empty_pack_dir}/reports"
    run pack_verify_reports "${empty_pack_dir}"
    assert_rc "1" "${RUN_RC}" "missing reports returns non-zero"
}

test_run_pack_sha256_cmd_fallback_and_sign_toggle() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

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

    source ./scripts/evidence_packs/run_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo '{"format":"evidence-pack-v1"}' > "${pack_dir}/manifest.json"

    run pack_sign_manifest "${pack_dir}"
    assert_rc "0" "${RUN_RC}" "sign manifest succeeds"
    assert_file_exists "${pack_dir}/manifest.signature.json" "signature bundle created"
    assert_eq \
        "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["signing_key_fingerprint"])' "${pack_dir}/manifest.json" < /dev/null)" \
        "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["signing_key_fingerprint"])' "${pack_dir}/manifest.signature.json" < /dev/null)" \
        "manifest and signature bundle record the same fingerprint"
}

test_run_pack_sign_manifest_helper_uses_ephemeral_key_by_default() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local manifest="${TEST_TMPDIR}/manifest.json"
    echo "{}" > "${manifest}"
    local calls="${TEST_TMPDIR}/sign_helper.calls"
    _cmd_python() {
        printf '%s\n' "$*" > "${calls}"
        return 0
    }

    run pack_sign_manifest_helper "${manifest}"
    assert_rc "0" "${RUN_RC}" "manifest signing helper succeeds"
    assert_match "manifest_writer\\.py sign" "$(cat "${calls}")" "sign helper invoked"
    assert_match "--generate-ephemeral" "$(cat "${calls}")" "ephemeral signing key path is used by default"
}

test_run_pack_sign_manifest_errors_and_cleans_when_helper_fails() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

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

    source ./scripts/evidence_packs/run_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${pack_dir}"
    echo '{"format":"evidence-pack-v1"}' > "${pack_dir}/manifest.json"
    local signing_key="${TEST_TMPDIR}/evidence-pack-signing-key.pem"
    local public_key="${TEST_TMPDIR}/evidence-pack-signing-key.pub.pem"

    PYTHONPATH=src python3 - "${signing_key}" "${public_key}" <<'PY'
import sys
from pathlib import Path

from invarlock.evidence_pack_integrity import generate_signing_keypair

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

    source ./scripts/evidence_packs/run_pack.sh

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

    source ./scripts/evidence_packs/run_pack.sh

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

    source ./scripts/evidence_packs/run_pack.sh

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

    source ./scripts/evidence_packs/run_pack.sh

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

    source ./scripts/evidence_packs/run_pack.sh

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

    source ./scripts/evidence_packs/run_pack.sh

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

    run bash -x ./scripts/evidence_packs/run_pack.sh --help
    assert_rc "0" "${RUN_RC}" "help returns 0"
    assert_match "Usage" "${RUN_OUT}" "usage printed"
}


test_run_pack_double_dash_defaults_out_and_pack_dir() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    pack_entrypoint() { printf '%s\n' "$@" > "${TEST_TMPDIR}/run.args"; }
    pack_build_pack() { printf '%s|%s' "$1" "$2" > "${TEST_TMPDIR}/pack.args"; }
    date() { echo "20240101_000000"; }

    pack_run_pack --

    assert_match "--out[[:space:]]+./evidence_pack_runs/subset_20240101_000000" "$(cat "${TEST_TMPDIR}/run.args")" "default output dir used"
    assert_eq "./evidence_pack_runs/subset_20240101_000000|./evidence_pack_runs/subset_20240101_000000/evidence_pack" "$(cat "${TEST_TMPDIR}/pack.args")" "default pack dir used"
}

test_run_pack_entrypoint_builds_run_args_for_modes() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    pack_entrypoint() { printf '%s\n' "$@" > "${TEST_TMPDIR}/run.args"; }
    pack_build_pack() { printf '%s|%s' "$1" "$2" > "${TEST_TMPDIR}/pack.args"; }

    pack_run_pack --calibrate-only --out "${TEST_TMPDIR}/out1"
    assert_match "--calibrate-only" "$(cat "${TEST_TMPDIR}/run.args")" "calibrate-only forwarded"
    assert_eq "${TEST_TMPDIR}/out1|${TEST_TMPDIR}/out1/evidence_pack" "$(cat "${TEST_TMPDIR}/pack.args")" "default pack dir used"

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

    source ./scripts/evidence_packs/run_pack.sh

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

    source ./scripts/evidence_packs/run_pack.sh

    pack_entrypoint() { :; }
    pack_build_pack() { t_fail "pack_build_pack should not run when layout normalization fails"; }
    pack_normalize_layout() { return 7; }

    run pack_run_pack --out "${TEST_TMPDIR}/out"
    assert_rc "7" "${RUN_RC}" "layout normalization failure propagates"
}

test_run_pack_atomic_helpers_cover_error_paths() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local pack_dir="${TEST_TMPDIR}/pack"
    local staging_dir

    mkdir() { return 1; }
    run pack_prepare_staging_dir "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "prepare staging dir fails when parent mkdir fails"
    unset -f mkdir

    staging_dir="$(pack_prepare_staging_dir "${pack_dir}")"
    mkdir -p "${pack_dir}"
    echo "payload" > "${pack_dir}/existing"
    run pack_finalize_staging_dir "${staging_dir}" "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "finalize rejects non-empty target directories"
    pack_cleanup_staging_dir "${staging_dir}"

    staging_dir="$(pack_prepare_staging_dir "${pack_dir}")"
    rm -f "${pack_dir}/existing"
    rmdir() { return 1; }
    run pack_finalize_staging_dir "${staging_dir}" "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "finalize fails when empty target cannot be removed atomically"
    unset -f rmdir
    pack_cleanup_staging_dir "${staging_dir}"

    staging_dir="$(pack_prepare_staging_dir "${pack_dir}")"
    mv() { return 1; }
    run pack_finalize_staging_dir "${staging_dir}" "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "finalize fails when staged directory cannot be moved into place"
    unset -f mv
    pack_cleanup_staging_dir "${staging_dir}"
}

test_run_pack_build_pack_propagates_staging_and_finalize_failures() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}"

    pack_require_passing_run_verdict() { return 0; }
    pack_require_cmd() { return 0; }
    pack_prepare_staging_dir() { return 1; }

    run pack_build_pack "${run_dir}" "${TEST_TMPDIR}/pack"
    assert_rc "1" "${RUN_RC}" "pack build fails when staging dir cannot be prepared"

    pack_prepare_staging_dir() {
        mkdir -p "${TEST_TMPDIR}/staging"
        printf '%s\n' "${TEST_TMPDIR}/staging"
    }
    pack_populate_pack_dir() { return 0; }
    pack_finalize_staging_dir() { return 1; }

    run pack_build_pack "${run_dir}" "${TEST_TMPDIR}/pack"
    assert_rc "1" "${RUN_RC}" "pack build fails when atomic finalize fails"
    [[ ! -d "${TEST_TMPDIR}/staging" ]] || t_fail "staging directory should be cleaned on finalize failure"
}

test_run_pack_build_pack_propagates_release_review_validation_failure() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}"

    pack_require_passing_run_verdict() { return 0; }
    pack_validate_release_review_settings() { return 1; }

    run pack_build_pack "${run_dir}" "${TEST_TMPDIR}/pack"
    assert_rc "1" "${RUN_RC}" "pack build fails when release-review validation fails"
}

test_run_pack_populate_pack_dir_propagates_environment_and_manifest_write_failures() {
    mock_reset

    source ./scripts/evidence_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    local pack_dir="${TEST_TMPDIR}/pack"
    mkdir -p "${run_dir}/reports" "${pack_dir}"
    echo "verdict" > "${run_dir}/reports/final_verdict.txt"
    echo '{"verdict":"PASS"}' > "${run_dir}/reports/final_verdict.json"

    pack_normalize_layout() { echo "v2"; }
    pack_copy_file() { return 0; }
    pack_copy_optional() { return 0; }
    pack_write_source_repo_metadata() { return 0; }
    pack_collect_reports() { :; }
    pack_verify_reports() { return 0; }
    pack_generate_html() { return 0; }
    pack_sign_manifest() { return 0; }

    pack_write_environment_metadata() { return 1; }
    run pack_populate_pack_dir "${run_dir}" "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "populate pack dir fails when environment metadata write fails"

    pack_write_environment_metadata() { return 0; }
    pack_write_edit_artifact_summary() { return 1; }
    run pack_populate_pack_dir "${run_dir}" "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "populate pack dir fails when edit artifact summary write fails"

    pack_write_edit_artifact_summary() { return 0; }
    pack_write_readme() { return 1; }
    run pack_populate_pack_dir "${run_dir}" "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "populate pack dir fails when README write fails"

    pack_write_readme() { return 0; }
    pack_write_checksums() { return 1; }
    run pack_populate_pack_dir "${run_dir}" "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "populate pack dir fails when checksum write fails"

    pack_write_checksums() { return 0; }
    pack_write_manifest() { return 1; }
    run pack_populate_pack_dir "${run_dir}" "${pack_dir}"
    assert_rc "1" "${RUN_RC}" "populate pack dir fails when manifest write fails"
}

test_pack_build_pack_and_verify_pack_end_to_end_v2_layout() {
    mock_reset

    PACK_SKIP_HTML=1
    PACK_PACK_LAYOUT=v2
    PACK_SUITE=subset
    PACK_NET=0
    PACK_DETERMINISM=throughput
    PACK_REPEATS=0

    local run_dir="${TEST_TMPDIR}/run"
    local pack_dir="${TEST_TMPDIR}/pack"

    mkdir -p "${run_dir}/reports" "${run_dir}/analysis" "${run_dir}/state"
    echo "PASS" > "${run_dir}/reports/final_verdict.txt"
    echo '{"ok":true}' > "${run_dir}/reports/final_verdict.json"
    echo 'model,edit' > "${run_dir}/analysis/eval_results.csv"
    echo 'm,quant_rtn' >> "${run_dir}/analysis/eval_results.csv"
    echo '{"model_list":["m"],"models":{"m":{"revision":"rev"}}}' > "${run_dir}/state/model_revisions.json"

    mkdir -p "${run_dir}/m/reports/clean/quant_rtn"
    echo '{}' > "${run_dir}/m/reports/clean/quant_rtn/evaluation.report.json"

    source "${TEST_ROOT}/scripts/evidence_packs/run_pack.sh"

    run pack_build_pack "${run_dir}" "${pack_dir}"
    assert_rc "0" "${RUN_RC}" "pack_build_pack succeeds"

    assert_file_exists "${pack_dir}/manifest.json" "manifest written"
    assert_file_exists "${pack_dir}/checksums.sha256" "checksums written"
    assert_file_exists "${pack_dir}/README.md" "readme written"
    assert_file_exists "${pack_dir}/results/verification_summary.json" "verification summary written"
    assert_file_exists "${pack_dir}/reports/m/clean/quant_rtn/evaluation.report.json" "report copied"
    assert_file_exists "${pack_dir}/manifest.signature.json" "signature bundle written"

    assert_file_exists "${pack_dir}/metadata/manifest.json" "manifest copied to metadata"
    assert_file_exists "${pack_dir}/metadata/manifest.signature.json" "signature copied to metadata"
    assert_file_exists "${pack_dir}/metadata/checksums.sha256" "checksum copied to metadata"
    assert_file_exists "${pack_dir}/metadata/source_repo.json" "source repo metadata copied"
    assert_file_exists "${pack_dir}/metadata/environment.json" "environment metadata copied"

    local verify_json="${TEST_TMPDIR}/verify.json"
    run bash "${TEST_ROOT}/scripts/evidence_packs/verify_pack.sh" --pack "${pack_dir}" --json-out "${verify_json}"
    assert_rc "0" "${RUN_RC}" "verify_pack succeeds"
    assert_file_exists "${verify_json}" "verify json written"
}
