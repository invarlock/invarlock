#!/usr/bin/env bash
# run_pack.sh - Run an evidence pack suite and package artifacts.

RUN_PACK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=run_suite.sh
source "${RUN_PACK_SCRIPT_DIR}/run_suite.sh"

pack_usage() {
    cat <<'EOF'
Usage: scripts/evidence_packs/run_pack.sh [options]

Builds an evidence pack from a completed suite run.
For strong distributable evidence, require a signed manifest, strict verification,
and a PASS final verdict.
Deployable edit scenarios are opt-in; set PACK_INCLUDE_DEPLOYABLE_EDITS=1,
PACK_DEPLOY_BACKENDS=bitsandbytes,gptq,awq, and select explicit deployable
scenario IDs only in backend-ready environments.
For containerized quant adapter evidence on a CUDA remote, set
PACK_RUNTIME_IMAGE_FLAVOR=quant during remote setup or explicitly set
INVARLOCK_RUNTIME_IMAGE=invarlock-runtime:cuda-quant with a provenance digest.

Verify a completed pack with:
  invarlock advanced evidence-pack verify <pack-dir> --strict --report-assurance strict

Options:
  --suite NAME         Suite name (subset|showcase|workshop3|full)
  --models CSV         Comma-separated model IDs to run (overrides suite defaults)
  --net 1|0            Enable network access for preflight/downloads (default: 0)
  --out DIR            Output directory for the run (default: ./evidence_pack_runs/<suite>_<timestamp>)
  --pack-dir DIR       Output directory for the evidence pack (default: <out>/evidence_pack)
  --layout NAME        Pack layout (v2 only) (default: v2)
  --determinism MODE   Determinism mode (strict|throughput)
  --repeats N          Determinism repeat count metadata (default: 0)
  --scenario-ids IDS   Comma-separated scenario IDs to include (filters scenarios.json before queue generation)
  --release-review     Hardened release-review mode: require PASS, signed pack,
                       runtime manifests, ci profile, and strict report assurance
  --calibrate-only     Only run calibration tasks (implies PACK_SUITE_MODE=calibrate-only)
  --errors-only        Only run error injection scenarios (still performs calibration unless presets are provided)
  --run-only           Run edits/reports only (implies resume)
  --resume             Resume an existing run directory
  --help               Show this help message
EOF
}

pack_require_cmd() {
    local cmd="$1"
    command -v "${cmd}" >/dev/null 2>&1 || {
        echo "ERROR: Required command not found: ${cmd}" >&2
        return 1
    }
}

pack_sha256_cmd() {
    if command -v sha256sum >/dev/null 2>&1; then
        echo "sha256sum"
    else
        echo "shasum -a 256"
    fi
}

pack_normalize_layout() {
    local layout="${1:-v2}"
    case "${layout}" in
        ""|v2)
            echo "v2"
            return 0
            ;;
        v1|flat|legacy)
            echo "ERROR: Legacy pack layout '${layout}' is no longer supported; use v2." >&2
            return 2
            ;;
        *)
            echo "ERROR: Unknown pack layout: ${layout} (expected v2)" >&2
            return 2
            ;;
    esac
}

pack_copy_file() {
    local src="$1"
    local dest="$2"
    if [[ ! -f "${src}" ]]; then
        echo "ERROR: Missing required artifact: ${src}" >&2
        return 1
    fi
    mkdir -p "$(dirname "${dest}")"
    cp "${src}" "${dest}"
}

pack_copy_optional() {
    local src="$1"
    local dest="$2"
    if [[ -f "${src}" ]]; then
        mkdir -p "$(dirname "${dest}")"
        cp "${src}" "${dest}"
    fi
}

pack_copy_optional_dir() {
    local src="$1"
    local dest="$2"
    if [[ -d "${src}" ]]; then
        mkdir -p "$(dirname "${dest}")"
        cp -a "${src}" "${dest}"
    fi
}

pack_copy_report_sidecars() {
    local report_dir="$1"
    local dest_dir="$2"

    pack_copy_optional "${report_dir}/manifest.json" "${dest_dir}/manifest.json"
    pack_copy_optional "${report_dir}/runtime.manifest.json" "${dest_dir}/runtime.manifest.json"
    pack_copy_optional "${report_dir}/evaluation_report.md" "${dest_dir}/evaluation_report.md"
    pack_copy_optional "${report_dir}/reviewer_summary.txt" "${dest_dir}/reviewer_summary.txt"
    pack_copy_optional "${report_dir}/edit_metadata.json" "${dest_dir}/edit_metadata.json"
    pack_copy_optional "${report_dir}/deployable_artifact_validation.json" "${dest_dir}/deployable_artifact_validation.json"
    pack_copy_optional "${report_dir}/backend_inventory.json" "${dest_dir}/backend_inventory.json"
    pack_copy_optional "${report_dir}/memory_report.json" "${dest_dir}/memory_report.json"
    pack_copy_optional "${report_dir}/load_smoke.json" "${dest_dir}/load_smoke.json"
    pack_copy_optional "${report_dir}/inference_smoke.json" "${dest_dir}/inference_smoke.json"
    pack_copy_optional_dir "${report_dir}/runtime_inputs" "${dest_dir}/runtime_inputs"
    pack_copy_optional_dir "${report_dir}/source" "${dest_dir}/source"
    pack_copy_optional_dir "${report_dir}/edited" "${dest_dir}/edited"
}

pack_collect_reports() {
    local run_dir="$1"
    find "${run_dir}" \
        -type d -name ".*.tmp.*" -prune \
        -o -type f -name "evaluation.report.json" -path "*/reports/*" -print | sort
}

pack_report_rel_path() {
    local run_dir="$1"
    local report_path="$2"
    local rel="${report_path#"${run_dir}/"}"
    local model="${rel%%/*}"
    local remainder="${rel#*/reports/}"
    remainder="${remainder%/evaluation.report.json}"
    if [[ -z "${model}" || "${remainder}" == "${rel}" ]]; then
        return 1
    fi
    printf '%s/%s\n' "${model}" "${remainder}"
}

pack_generate_html() {
    local pack_dir="$1"
    local report
    while IFS= read -r report; do
        [[ -n "${report}" ]] || continue
        local html="${report%.report.json}.html"
        if ! invarlock report html --input "${report}" --output "${html}" --force >/dev/null; then
            echo "WARNING: Failed to render HTML report for ${report}" >&2
        fi
    done < <(find "${pack_dir}/reports" -type f -name "evaluation.report.json" | sort)
}

pack_report_scenario_id() {
    local pack_dir="$1"
    local report="$2"
    python3 "${RUN_PACK_SCRIPT_DIR}/python/verify_pack_checks.py" report-scenario-id "${pack_dir}" "${report}"
}

pack_scenario_strictness() {
    local pack_dir="$1"
    local scenario_id="$2"
    local scenarios_path="${pack_dir}/metadata/scenarios.json"
    if [[ ! -f "${scenarios_path}" ]]; then
        return 1
    fi
    python3 "${RUN_PACK_SCRIPT_DIR}/python/verify_pack_checks.py" scenario-strictness "${scenarios_path}" "${scenario_id}"
}

pack_report_expects_verify_failure() {
    local pack_dir="$1"
    local report="$2"
    python3 "${RUN_PACK_SCRIPT_DIR}/python/verify_pack_checks.py" report-expects-verify-failure "${pack_dir}" "${report}"
}

pack_verify_reports() {
    local pack_dir="$1"
    local profile="${PACK_VERIFY_PROFILE:-dev}"
    local report_assurance="${PACK_REPORT_ASSURANCE:-report}"
    PACK_VERIFY_PROFILE_USED="${profile}"
    PACK_REPORT_ASSURANCE_USED="${report_assurance}"
    export PACK_VERIFY_PROFILE_USED PACK_REPORT_ASSURANCE_USED

    local results_dir="${pack_dir}/results"
    mkdir -p "${results_dir}"
    python3 "${RUN_PACK_SCRIPT_DIR}/python/verify_pack_checks.py" \
        verify-reports \
        "${pack_dir}" \
        --profile "${profile}" \
        --report-assurance "${report_assurance}" \
        --write-sidecars \
        --summary-out "${results_dir}/verification_summary.json"
}

pack_write_source_repo_metadata() {
    local dest="$1"
    python3 "${RUN_PACK_SCRIPT_DIR}/python/write_source_repo_metadata.py" --out "${dest}"
}

pack_write_environment_metadata() {
    local run_dir="$1"
    local dest="$2"
    python3 "${RUN_PACK_SCRIPT_DIR}/python/write_environment_metadata.py" \
        --run-dir "${run_dir}" \
        --out "${dest}"
}

pack_write_manifest() {
    local pack_dir="$1"
    local run_dir="$2"
    local suite="$3"
    local net="$4"
    local determinism="$5"
    local repeats="$6"

    python3 "${RUN_PACK_SCRIPT_DIR}/python/manifest_writer.py" \
        --pack-dir "${pack_dir}" \
        --run-dir "${run_dir}" \
        --suite "${suite}" \
        --net "${net}" \
        --determinism "${determinism}" \
        --repeats "${repeats}"
}

pack_write_edit_artifact_summary() {
    local pack_dir="$1"
    local scenarios_path="${pack_dir}/metadata/scenarios.json"
    local summary_path="${pack_dir}/results/analysis/edit_artifact_summary.json"
    if [[ ! -f "${scenarios_path}" ]]; then
        return 0
    fi
    python3 "${RUN_PACK_SCRIPT_DIR}/python/edit_artifact_summary.py" \
        --pack-dir "${pack_dir}" \
        --scenarios "${scenarios_path}" \
        --out "${summary_path}"
}

pack_sign_manifest_helper() {
    local manifest_path="$1"
    local signing_key_path="${2:-}"
    local helper="${RUN_PACK_SCRIPT_DIR}/python/sign_manifest.py"

    if [[ -n "${signing_key_path}" ]]; then
        _cmd_python "${helper}" --manifest "${manifest_path}" --signing-key "${signing_key_path}"
        return
    fi
    _cmd_python "${helper}" --manifest "${manifest_path}" --generate-ephemeral
}

pack_require_passing_run_verdict() {
    local run_dir="$1"
    local verdict_file="${run_dir}/reports/final_verdict.json"
    local verdict_status="MISSING"

    if type pack_read_final_verdict >/dev/null 2>&1; then
        verdict_status="$(pack_read_final_verdict "${verdict_file}")"
    else
        verdict_status="$(python3 - "${verdict_file}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("MISSING")
    raise SystemExit(0)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    print("INVALID")
    raise SystemExit(0)
value = payload.get("verdict")
if isinstance(value, str):
    print(value.strip().upper())
else:
    print("MISSING")
PY
)"
    fi

    if [[ "${verdict_status}" == "FAIL" ]]; then
        echo "ERROR: Run final verdict is FAIL; refusing to build a distributable pack." >&2
        return 1
    fi
    if [[ "${verdict_status}" != "PASS" ]]; then
        if [[ "${PACK_REQUIRE_PASS:-0}" == "1" ]]; then
            echo "ERROR: Run final verdict status is ${verdict_status}; release-review mode requires PASS." >&2
            return 1
        fi
        echo "WARNING: Run final verdict status is ${verdict_status}; proceeding with pack build." >&2
    fi
}

pack_require_runtime_manifests() {
    local run_dir="$1"
    local missing=0
    local report
    while IFS= read -r report; do
        [[ -n "${report}" ]] || continue
        if [[ ! -f "$(dirname "${report}")/runtime.manifest.json" ]]; then
            echo "ERROR: Missing runtime.manifest.json for ${report}" >&2
            missing=$((missing + 1))
        fi
    done < <(pack_collect_reports "${run_dir}")
    if [[ "${missing}" -gt 0 ]]; then
        return 1
    fi
}

pack_apply_release_review_defaults() {
    PACK_REQUIRE_PASS="${PACK_REQUIRE_PASS:-1}"
    PACK_VERIFY_PROFILE="${PACK_VERIFY_PROFILE:-ci}"
    PACK_REPORT_ASSURANCE="${PACK_REPORT_ASSURANCE:-strict}"
    PACK_SIGN_MANIFEST="${PACK_SIGN_MANIFEST:-1}"
    PACK_REQUIRE_RUNTIME_MANIFESTS="${PACK_REQUIRE_RUNTIME_MANIFESTS:-1}"
    PACK_DEFER_REPORT_RENDERING="${PACK_DEFER_REPORT_RENDERING:-1}"
    PACK_RELEASE_REVIEW=1
    export PACK_REQUIRE_PASS PACK_VERIFY_PROFILE PACK_REPORT_ASSURANCE
    export PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS
    export PACK_DEFER_REPORT_RENDERING PACK_RELEASE_REVIEW
}

pack_validate_release_review_settings() {
    if [[ "${PACK_RELEASE_REVIEW:-0}" != "1" ]]; then
        return 0
    fi
    if [[ "${PACK_REQUIRE_PASS:-0}" != "1" ]]; then
        echo "ERROR: release-review mode requires PACK_REQUIRE_PASS=1." >&2
        return 1
    fi
    if [[ "${PACK_SIGN_MANIFEST:-1}" == "0" ]]; then
        echo "ERROR: release-review mode requires PACK_SIGN_MANIFEST=1." >&2
        return 1
    fi
    if [[ "${PACK_REQUIRE_RUNTIME_MANIFESTS:-0}" != "1" ]]; then
        echo "ERROR: release-review mode requires PACK_REQUIRE_RUNTIME_MANIFESTS=1." >&2
        return 1
    fi
    if [[ -z "${PACK_VERIFY_PROFILE:-}" ]]; then
        echo "ERROR: release-review mode requires explicit PACK_VERIFY_PROFILE." >&2
        return 1
    fi
    if [[ "${PACK_VERIFY_PROFILE}" == "dev" ]]; then
        echo "ERROR: release-review mode rejects PACK_VERIFY_PROFILE=dev." >&2
        return 1
    fi
    if [[ "${PACK_REPORT_ASSURANCE:-}" != "strict" ]]; then
        echo "ERROR: release-review mode requires PACK_REPORT_ASSURANCE=strict." >&2
        return 1
    fi
}

pack_sign_manifest() {
    local pack_dir="$1"
    if [[ "${PACK_SIGN_MANIFEST:-1}" == "0" ]]; then
        return 0
    fi

    if [[ ! -f "${pack_dir}/manifest.json" ]]; then
        echo "ERROR: manifest.json missing; cannot sign." >&2
        return 1
    fi

    local signing_key_path="${PACK_SIGNING_KEY:-}"
    local tmp_err=""
    local signer_fpr=""

    tmp_err="$(mktemp 2>/dev/null || mktemp -t invarlock_pack_sign.XXXXXXXX)" || return 1
    if signer_fpr="$(pack_sign_manifest_helper "${pack_dir}/manifest.json" "${signing_key_path}" 2>"${tmp_err}")"; then
        rm -f "${tmp_err}"
        if [[ -n "${signer_fpr}" ]]; then
            PACK_SIGNER_FINGERPRINT="${signer_fpr}"
            export PACK_SIGNER_FINGERPRINT
        fi
        return 0
    fi

    echo "ERROR: manifest signing failed." >&2
    if [[ -s "${tmp_err}" ]]; then
        cat "${tmp_err}" >&2
    fi
    rm -f "${tmp_err}"
    rm -f "${pack_dir}/manifest.signature.json"
    return 1
}

pack_write_checksums() {
    local pack_dir="$1"
    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        while IFS= read -r file; do
            [[ -n "${file}" ]] || continue
            case "${file}" in
                ./checksums.sha256|./manifest.json|./manifest.signature.json)
                    continue
                    ;;
                ./metadata/manifest.json|./metadata/manifest.signature.json|./metadata/checksums.sha256)
                    continue
                    ;;
                ./.DS_Store|*/.DS_Store|./._*|*/._*|./__MACOSX/*)
                    continue
                    ;;
            esac
            ${sha_cmd} "${file}"
        done < <(find . -type f -print | sort) > "checksums.sha256"
    )
}

pack_write_readme() {
    local pack_dir="$1"
    echo "[run_pack.sh] Writing README.md to ${pack_dir}" >&2
    python3 "${RUN_PACK_SCRIPT_DIR}/python/write_pack_readme.py" "${pack_dir}"
}

pack_prepare_staging_dir() {
    local pack_dir="$1"
    local parent_dir
    local base_name
    parent_dir="$(dirname "${pack_dir}")"
    base_name="$(basename "${pack_dir}")"
    mkdir -p "${parent_dir}" || return 1
    mktemp -d "${parent_dir}/.${base_name}.tmp.XXXXXX"
}

pack_cleanup_staging_dir() {
    local staging_dir="$1"
    if [[ -n "${staging_dir}" && -d "${staging_dir}" ]]; then
        rm -rf "${staging_dir}"
    fi
}

pack_finalize_staging_dir() {
    local staging_dir="$1"
    local pack_dir="$2"

    if [[ -d "${pack_dir}" && -n "$(ls -A "${pack_dir}" 2>/dev/null)" ]]; then
        echo "ERROR: pack_dir already exists and is not empty: ${pack_dir}" >&2
        return 1
    fi
    if [[ -e "${pack_dir}" && ! -d "${pack_dir}" ]]; then
        echo "ERROR: pack_dir already exists and is not a directory: ${pack_dir}" >&2
        return 1
    fi
    if [[ -d "${pack_dir}" ]]; then
        if ! rmdir "${pack_dir}" 2>/dev/null; then
            echo "ERROR: pack_dir could not be replaced atomically: ${pack_dir}" >&2
            return 1
        fi
    fi
    if ! mv "${staging_dir}" "${pack_dir}"; then
        echo "ERROR: Failed to finalize evidence pack atomically: ${pack_dir}" >&2
        return 1
    fi
    return 0
}

pack_populate_pack_dir() {
    local run_dir="$1"
    local pack_dir="$2"
    local layout
    layout="$(pack_normalize_layout "${PACK_PACK_LAYOUT:-v2}")" || return $?

    local results_dir="${pack_dir}/results"
    local verdicts_dir="${results_dir}/verdicts"
    local analysis_dir="${results_dir}/analysis"
    local metadata_dir="${pack_dir}/metadata"
    local revisions_dest="${pack_dir}/metadata/model_revisions.json"
    local scenarios_dest="${pack_dir}/metadata/scenarios.json"
    local tuned_edit_params_dest="${pack_dir}/metadata/tuned_edit_params.json"
    local source_repo_dest="${pack_dir}/metadata/source_repo.json"
    local environment_dest="${pack_dir}/metadata/environment.json"

    mkdir -p "${results_dir}" "${verdicts_dir}" "${analysis_dir}" "${metadata_dir}"

    pack_copy_file "${run_dir}/reports/final_verdict.txt" "${verdicts_dir}/final_verdict.txt"
    pack_copy_file "${run_dir}/reports/final_verdict.json" "${verdicts_dir}/final_verdict.json"
    pack_copy_optional "${run_dir}/analysis/determinism_repeats.json" "${analysis_dir}/determinism_repeats.json"
    pack_copy_optional "${run_dir}/reports/category_summary.json" "${analysis_dir}/category_summary.json"
    pack_copy_optional "${run_dir}/reports/guard_signal_summary.json" "${analysis_dir}/guard_signal_summary.json"
    pack_copy_optional "${run_dir}/reports/guard_intervention_summary.json" "${analysis_dir}/guard_intervention_summary.json"
    pack_copy_optional "${run_dir}/reports/scenario_signal_summary.json" "${analysis_dir}/scenario_signal_summary.json"
    pack_copy_optional "${run_dir}/results/analysis/evaluation_optimization_summary.json" "${analysis_dir}/evaluation_optimization_summary.json"

    pack_copy_optional "${run_dir}/state/model_revisions.json" "${revisions_dest}"
    pack_copy_optional "${run_dir}/state/scenarios.json" "${scenarios_dest}"
    pack_copy_optional "${run_dir}/state/tuned_edit_params.json" "${tuned_edit_params_dest}"
    pack_write_source_repo_metadata "${source_repo_dest}" || return $?
    pack_write_environment_metadata "${run_dir}" "${environment_dest}" || return $?

    local report
    while IFS= read -r report; do
        [[ -n "${report}" ]] || continue
        local rel
        rel="$(pack_report_rel_path "${run_dir}" "${report}")" || continue
        local dest_dir="${pack_dir}/reports/${rel}"
        mkdir -p "${dest_dir}"
        cp "${report}" "${dest_dir}/evaluation.report.json"
        pack_copy_report_sidecars "$(dirname "${report}")" "${dest_dir}"
        # Optional sidecar artifacts (used by some detectors; safe to omit when absent).
        pack_copy_optional "$(dirname "${report}")/rmt_probe.json" "${dest_dir}/rmt_probe.json"
        pack_copy_optional "$(dirname "${report}")/ve_probe.json" "${dest_dir}/ve_probe.json"
    done < <(pack_collect_reports "${run_dir}")

    local verify_rc=0
    if pack_verify_reports "${pack_dir}"; then
        verify_rc=0
    else
        verify_rc=$?
    fi

    if [[ "${PACK_SKIP_HTML:-0}" != "1" ]]; then
        pack_generate_html "${pack_dir}"
    fi

    pack_write_edit_artifact_summary "${pack_dir}" || return $?
    pack_write_readme "${pack_dir}" || return $?
    pack_write_checksums "${pack_dir}" || return $?
    pack_write_manifest "${pack_dir}" "${run_dir}" "${PACK_SUITE:-}" "${PACK_NET:-0}" "${PACK_DETERMINISM:-}" "${PACK_REPEATS:-0}" || return $?
    local sign_rc=0
    if pack_sign_manifest "${pack_dir}"; then
        sign_rc=0
    else
        sign_rc=$?
    fi
    cp "${pack_dir}/manifest.json" "${pack_dir}/metadata/manifest.json"
    if [[ -f "${pack_dir}/manifest.signature.json" ]]; then
        cp "${pack_dir}/manifest.signature.json" "${pack_dir}/metadata/manifest.signature.json"
    fi
    cp "${pack_dir}/checksums.sha256" "${pack_dir}/metadata/checksums.sha256"

    if [[ "${verify_rc}" -eq 0 && "${sign_rc}" -ne 0 ]]; then
        return "${sign_rc}"
    fi
    return "${verify_rc}"
}

pack_build_pack() {
    local run_dir="$1"
    local pack_dir="$2"
    local staging_dir=""
    local rc=0

    if [[ -z "${run_dir}" || -z "${pack_dir}" ]]; then
        echo "ERROR: pack_build_pack requires run_dir and pack_dir." >&2
        return 1
    fi
    if [[ ! -d "${run_dir}" ]]; then
        echo "ERROR: run_dir not found: ${run_dir}" >&2
        return 1
    fi

    pack_require_passing_run_verdict "${run_dir}" || return 1
    pack_validate_release_review_settings || return 1
    if [[ "${PACK_REQUIRE_RUNTIME_MANIFESTS:-0}" == "1" ]]; then
        pack_require_runtime_manifests "${run_dir}" || return 1
    fi
    pack_require_cmd invarlock

    if [[ -d "${pack_dir}" && -n "$(ls -A "${pack_dir}" 2>/dev/null)" ]]; then
        echo "ERROR: pack_dir already exists and is not empty: ${pack_dir}" >&2
        return 1
    fi
    if [[ -e "${pack_dir}" && ! -d "${pack_dir}" ]]; then
        echo "ERROR: pack_dir already exists and is not a directory: ${pack_dir}" >&2
        return 1
    fi

    staging_dir="$(pack_prepare_staging_dir "${pack_dir}")" || return 1

    pack_populate_pack_dir "${run_dir}" "${staging_dir}"
    rc=$?
    if [[ "${rc}" -ne 0 ]]; then
        pack_cleanup_staging_dir "${staging_dir}"
        return "${rc}"
    fi
    if pack_finalize_staging_dir "${staging_dir}" "${pack_dir}"; then
        :
    else
        rc=$?
        pack_cleanup_staging_dir "${staging_dir}"
        return "${rc}"
    fi
    return 0
}

pack_run_pack() {
    set -euo pipefail

    local suite="${PACK_SUITE:-subset}"
    local net="${PACK_NET:-0}"
    local models_csv="${PACK_MODELS_CSV:-${PACK_MODELS:-}}"
    local out="${PACK_OUTPUT_DIR:-${OUTPUT_DIR:-}}"
    local determinism="${PACK_DETERMINISM:-throughput}"
    local repeats="${PACK_REPEATS:-0}"
    local suite_mode="${PACK_SUITE_MODE:-full}"
    local resume_flag="${RESUME_FLAG:-false}"
    local pack_dir="${PACK_DIR:-}"
    local layout="${PACK_PACK_LAYOUT:-v2}"
    local scenario_ids="${PACK_SCENARIO_IDS:-}"

    if [[ "${PACK_RELEASE_REVIEW:-0}" == "1" ]]; then
        pack_apply_release_review_defaults
    fi

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                pack_usage
                return 0
                ;;
            --suite)
                suite="${2:-}"
                if [[ -z "${suite}" ]]; then
                    echo "ERROR: --suite requires a value" >&2
                    return 2
                fi
                shift 2
                ;;
            --net)
                net="${2:-}"
                if [[ -z "${net}" ]]; then
                    echo "ERROR: --net requires 1 or 0" >&2
                    return 2
                fi
                shift 2
                ;;
            --models)
                models_csv="${2:-}"
                if [[ -z "${models_csv}" ]]; then
                    echo "ERROR: --models requires a value" >&2
                    return 2
                fi
                shift 2
                ;;
            --out)
                out="${2:-}"
                if [[ -z "${out}" ]]; then
                    echo "ERROR: --out requires a value" >&2
                    return 2
                fi
                shift 2
                ;;
            --pack-dir)
                pack_dir="${2:-}"
                if [[ -z "${pack_dir}" ]]; then
                    echo "ERROR: --pack-dir requires a value" >&2
                    return 2
                fi
                shift 2
                ;;
            --layout)
                layout="${2:-}"
                if [[ -z "${layout}" ]]; then
                    echo "ERROR: --layout requires a value" >&2
                    return 2
                fi
                shift 2
                ;;
            --determinism)
                determinism="${2:-}"
                if [[ -z "${determinism}" ]]; then
                    echo "ERROR: --determinism requires a value" >&2
                    return 2
                fi
                shift 2
                ;;
            --scenario-ids)
                scenario_ids="${2:-}"
                if [[ -z "${scenario_ids}" ]]; then
                    echo "ERROR: --scenario-ids requires a value" >&2
                    return 2
                fi
                shift 2
                ;;
            --release-review)
                pack_apply_release_review_defaults
                shift
                ;;
            --repeats)
                repeats="${2:-}"
                if [[ -z "${repeats}" || ! "${repeats}" =~ ^[0-9]+$ ]]; then
                    echo "ERROR: --repeats requires an integer" >&2
                    return 2
                fi
                shift 2
                ;;
            --resume)
                resume_flag="true"
                shift
                ;;
            --calibrate-only)
                suite_mode="calibrate-only"
                resume_flag="false"
                shift
                ;;
            --errors-only)
                suite_mode="errors-only"
                resume_flag="false"
                shift
                ;;
            --run-only)
                suite_mode="run-only"
                resume_flag="true"
                shift
                ;;
            --)
                shift
                break
                ;;
            *)
                echo "Unknown arg: $1" >&2
                pack_usage >&2
                return 2
                ;;
        esac
    done

    if [[ -z "${out}" ]]; then
        local stamp
        stamp="$(date -u +%Y%m%d_%H%M%S)"
        out="./evidence_pack_runs/${suite}_${stamp}"
    fi

    case "${net}" in
        0|1)
            :
            ;;
        *)
            echo "ERROR: --net requires 1 or 0" >&2
            return 2
            ;;
    esac

    local -a run_args
    run_args=("--suite" "${suite}" "--out" "${out}" "--determinism" "${determinism}" "--repeats" "${repeats}" "--net" "${net}")
    if [[ -n "${models_csv}" ]]; then
        run_args+=("--models" "${models_csv}")
    fi
    if [[ "${suite_mode}" == "calibrate-only" ]]; then
        run_args+=("--calibrate-only")
    elif [[ "${suite_mode}" == "errors-only" ]]; then
        run_args+=("--errors-only")
    elif [[ "${suite_mode}" == "run-only" ]]; then
        run_args+=("--run-only")
    elif [[ "${resume_flag}" == "true" ]]; then
        run_args+=("--resume")
    fi
    if [[ -n "${scenario_ids}" ]]; then
        run_args+=("--scenario-ids" "${scenario_ids}")
    fi

    pack_validate_release_review_settings || return 1

    pack_entrypoint "${run_args[@]}"
    layout="$(pack_normalize_layout "${layout}")" || return $?
    PACK_PACK_LAYOUT="${layout}"
    export PACK_PACK_LAYOUT

    if [[ -z "${pack_dir}" ]]; then
        pack_dir="${out}/evidence_pack"
    fi

    pack_build_pack "${out}" "${pack_dir}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    pack_run_pack "$@"
fi
