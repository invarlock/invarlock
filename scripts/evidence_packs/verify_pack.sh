#!/usr/bin/env bash
# verify_pack.sh - Validate evidence pack checksums and reports.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runtime.sh
source "${SCRIPT_DIR}/lib/runtime.sh"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -n "${TEST_REAL_PYTHON3:-}" && -x "${TEST_REAL_PYTHON3}" ]]; then
        PYTHON_BIN="${TEST_REAL_PYTHON3}"
        export PYTHON_BIN
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
        export PYTHON_BIN
    fi
fi
PACK_VERIFY_OK=0
PACK_VERIFY_USAGE=2
PACK_VERIFY_MISSING=3
PACK_VERIFY_FORMAT=4
PACK_VERIFY_SIGNATURE=5
PACK_VERIFY_INTEGRITY=6
PACK_VERIFY_REPORTS=7

pack_usage() {
    cat <<'EOF'
Usage: scripts/evidence_packs/verify_pack.sh --pack DIR [options]

Options:
  --pack DIR          Evidence pack directory to verify
  --json-out FILE     Write verify JSON output to FILE (must be outside the pack)
  --skip-verify       Skip invarlock verify step
  --strict            Fail closed on missing/invalid signatures and pack mismatches
  --expected-fingerprint FPR
                     Require the signed manifest to use this signer fingerprint
  --report-assurance MODE
                     Nested report assurance mode (report|strict|off)
  --help              Show this help message

Notes:
  - Clean reports are re-verified with `invarlock verify` and must satisfy
    runtime.manifest.json provenance checks.
  - Error-injection reports are also rechecked, but their non-zero verify status
    remains expected and does not fail the pack on its own.

Exit codes:
  0  verified
  2  invalid usage / arguments
  3  missing pack or required pack files
  4  manifest format / schema validation failed
  5  signature verification failed
  6  integrity verification failed (checksum binding, checksums, signed provenance refs, or strict extra-file checks)
  7  report verification failed
EOF
}

pack_warn() {
    echo "WARNING: $*" >&2
}

pack_sha256_cmd() {
    if command -v sha256sum >/dev/null 2>&1; then
        echo "sha256sum"
    else
        echo "shasum -a 256"
    fi
}

pack_file_sha256() {
    local pack_dir="$1"
    local rel_path="$2"
    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} "${rel_path}" | awk '{print $1}'
    )
}

pack_manifest_field() {
    local manifest_path="$1"
    local field="$2"
    _cmd_python "${SCRIPT_DIR}/python/verify_pack_checks.py" manifest-field "${manifest_path}" "${field}"
}

pack_path_within_dir() {
    local dir_path="$1"
    local candidate_path="$2"
    _cmd_python "${SCRIPT_DIR}/python/verify_pack_checks.py" path-within "${dir_path}" "${candidate_path}"
}

pack_verify_manifest_binds_checksums() {
    local pack_dir="$1"
    local strict="$2"

    local expected
    if ! expected="$(pack_manifest_field "${pack_dir}/manifest.json" "checksums_sha256_digest" 2>/dev/null)"; then
        echo "ERROR: manifest.json missing checksums_sha256_digest (pack is not tamper-evident)." >&2
        return 1
    fi
    if [[ -z "${expected}" ]]; then
        echo "ERROR: manifest.json checksums_sha256_digest is empty." >&2
        return 1
    fi

    local actual
    actual="$(pack_file_sha256 "${pack_dir}" "checksums.sha256")"
    if [[ -z "${actual}" ]]; then
        echo "ERROR: Failed to compute sha256 for checksums.sha256." >&2
        return 1
    fi

    if [[ "${expected}" != "${actual}" ]]; then
        echo "ERROR: checksums.sha256 digest mismatch (expected ${expected}, got ${actual})." >&2
        return 1
    fi

    return 0
}

pack_validate_manifest_schema() {
    local pack_dir="$1"
    local validator="${SCRIPT_DIR}/python/validate_manifest.py"
    local out

    if ! out="$(_cmd_python "${validator}" "${pack_dir}/manifest.json" 2>&1)"; then
        echo "ERROR: manifest.json failed contract validation." >&2
        printf '%s\n' "${out}" >&2
        return 1
    fi
    return 0
}

pack_verify_manifest_provenance() {
    local pack_dir="$1"
    local verifier="${SCRIPT_DIR}/python/verify_manifest_provenance.py"
    local out

    if ! out="$(_cmd_python "${verifier}" "${pack_dir}" 2>&1)"; then
        echo "ERROR: manifest.json provenance references failed verification." >&2
        printf '%s\n' "${out}" >&2
        return 1
    fi
    return 0
}

pack_verify_checksums() {
    local pack_dir="$1"
    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        ${sha_cmd} -c "checksums.sha256"
    )
}

pack_verify_no_extra_files() {
    local pack_dir="$1"
    local strict="$2"
    local -a args=("${SCRIPT_DIR}/python/verify_pack_checks.py" extra-files "${pack_dir}")
    if [[ "${strict}" == "1" ]]; then
        args+=("--strict")
    fi
    _cmd_python "${args[@]}"
}

pack_verify_signature_helper() {
    local pack_dir="$1"
    local strict="$2"
    local expected_fingerprint="$3"
    local helper="${SCRIPT_DIR}/python/verify_signature.py"
    local -a args=("${helper}")

    if [[ "${strict}" == "1" ]]; then
        args+=("--strict")
    fi
    if [[ -n "${expected_fingerprint}" ]]; then
        args+=("--expected-fingerprint" "${expected_fingerprint}")
    fi
    args+=("${pack_dir}")
    _cmd_python "${args[@]}"
}

pack_verify_signature() {
    local pack_dir="$1"
    local strict="$2"
    local expected_fingerprint="$3"
    local tmp_err=""
    local signer_fpr=""

    tmp_err="$(mktemp 2>/dev/null || mktemp -t invarlock_pack_verify_sig.XXXXXXXX)" || return 1
    if signer_fpr="$(pack_verify_signature_helper "${pack_dir}" "${strict}" "${expected_fingerprint}" 2>"${tmp_err}")"; then
        if [[ -s "${tmp_err}" ]]; then
            cat "${tmp_err}" >&2
        fi
        rm -f "${tmp_err}"
        if [[ -n "${signer_fpr}" ]]; then
            PACK_SIGNER_FINGERPRINT="${signer_fpr}"
            export PACK_SIGNER_FINGERPRINT
        fi
        return 0
    fi
    if [[ -s "${tmp_err}" ]]; then
        cat "${tmp_err}" >&2
    fi
    rm -f "${tmp_err}"
    return 1
}

pack_report_scenario_id() {
    local pack_dir="$1"
    local report="$2"
    _cmd_python "${SCRIPT_DIR}/python/verify_pack_checks.py" report-scenario-id "${pack_dir}" "${report}"
}

pack_scenario_strictness() {
    local pack_dir="$1"
    local scenario_id="$2"
    local scenarios_path="${pack_dir}/metadata/scenarios.json"
    if [[ ! -f "${scenarios_path}" ]]; then
        return 1
    fi
    _cmd_python "${SCRIPT_DIR}/python/verify_pack_checks.py" scenario-strictness "${scenarios_path}" "${scenario_id}"
}

pack_report_expects_verify_failure() {
    local pack_dir="$1"
    local report="$2"
    _cmd_python "${SCRIPT_DIR}/python/verify_pack_checks.py" report-expects-verify-failure "${pack_dir}" "${report}"
}

pack_verify_reports() {
    local pack_dir="$1"
    local json_out="$2"
    local profile="${PACK_VERIFY_PROFILE:-dev}"
    local report_assurance="${PACK_REPORT_ASSURANCE:-report}"
    local -a args=(
        "${SCRIPT_DIR}/python/verify_pack_checks.py"
        verify-reports
        "${pack_dir}"
        --profile
        "${profile}"
        --report-assurance
        "${report_assurance}"
        --require-clean
    )
    if [[ -n "${json_out}" ]]; then
        args+=(--json-out "${json_out}")
    fi
    _cmd_python "${args[@]}"
}

pack_verify_pack() {
    set -euo pipefail

    local pack_dir=""
    local json_out=""
    local skip_verify=0
    local strict="${PACK_STRICT_MODE:-0}"
    local expected_fingerprint="${PACK_EXPECTED_FINGERPRINT:-}"
    local report_assurance="${PACK_REPORT_ASSURANCE:-report}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                pack_usage
                return "${PACK_VERIFY_OK}"
                ;;
            --pack)
                pack_dir="${2:-}"
                if [[ -z "${pack_dir}" ]]; then
                    echo "ERROR: --pack requires a value" >&2
                    return "${PACK_VERIFY_USAGE}"
                fi
                shift 2
                ;;
            --json-out)
                json_out="${2:-}"
                if [[ -z "${json_out}" ]]; then
                    echo "ERROR: --json-out requires a value" >&2
                    return "${PACK_VERIFY_USAGE}"
                fi
                shift 2
                ;;
            --skip-verify)
                skip_verify=1
                shift
                ;;
            --strict)
                strict=1
                shift
                ;;
            --expected-fingerprint)
                expected_fingerprint="${2:-}"
                if [[ -z "${expected_fingerprint}" ]]; then
                    echo "ERROR: --expected-fingerprint requires a value" >&2
                    return "${PACK_VERIFY_USAGE}"
                fi
                shift 2
                ;;
            --report-assurance)
                report_assurance="${2:-}"
                if [[ -z "${report_assurance}" ]]; then
                    echo "ERROR: --report-assurance requires report, strict, or off" >&2
                    return "${PACK_VERIFY_USAGE}"
                fi
                case "${report_assurance}" in
                    report|strict|off)
                        :
                        ;;
                    *)
                        echo "ERROR: --report-assurance requires report, strict, or off" >&2
                        return "${PACK_VERIFY_USAGE}"
                        ;;
                esac
                shift 2
                ;;
            --)
                shift
                break
                ;;
            *)
                echo "Unknown arg: $1" >&2
                pack_usage >&2
                return "${PACK_VERIFY_USAGE}"
                ;;
        esac
    done

    if [[ -z "${pack_dir}" ]]; then
        echo "ERROR: --pack is required" >&2
        pack_usage >&2
        return "${PACK_VERIFY_USAGE}"
    fi
    if [[ ! -d "${pack_dir}" ]]; then
        echo "ERROR: Pack directory not found: ${pack_dir}" >&2
        return "${PACK_VERIFY_MISSING}"
    fi
    if [[ ! -f "${pack_dir}/manifest.json" ]]; then
        echo "ERROR: manifest.json missing in pack." >&2
        return "${PACK_VERIFY_MISSING}"
    fi
    if [[ ! -f "${pack_dir}/checksums.sha256" ]]; then
        echo "ERROR: checksums.sha256 missing in pack." >&2
        return "${PACK_VERIFY_MISSING}"
    fi
    if [[ -n "${json_out}" ]] && pack_path_within_dir "${pack_dir}" "${json_out}"; then
        echo "ERROR: --json-out must point outside the pack directory." >&2
        return "${PACK_VERIFY_USAGE}"
    fi
    PACK_REPORT_ASSURANCE="${report_assurance}"
    export PACK_REPORT_ASSURANCE

    if ! pack_validate_manifest_schema "${pack_dir}"; then
        return "${PACK_VERIFY_FORMAT}"
    fi
    if ! pack_verify_signature "${pack_dir}" "${strict}" "${expected_fingerprint}"; then
        return "${PACK_VERIFY_SIGNATURE}"
    fi
    if ! pack_verify_manifest_binds_checksums "${pack_dir}" "${strict}"; then
        return "${PACK_VERIFY_INTEGRITY}"
    fi
    if ! pack_verify_checksums "${pack_dir}"; then
        return "${PACK_VERIFY_INTEGRITY}"
    fi
    if ! pack_verify_manifest_provenance "${pack_dir}"; then
        return "${PACK_VERIFY_INTEGRITY}"
    fi
    if ! pack_verify_no_extra_files "${pack_dir}" "${strict}"; then
        return "${PACK_VERIFY_INTEGRITY}"
    fi

    if [[ "${skip_verify}" -eq 0 ]]; then
        if ! pack_verify_reports "${pack_dir}" "${json_out}"; then
            return "${PACK_VERIFY_REPORTS}"
        fi
    fi

    return "${PACK_VERIFY_OK}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    pack_verify_pack "$@"
fi
