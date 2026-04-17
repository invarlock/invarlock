#!/usr/bin/env bash
# verify_pack.sh - Validate evidence pack checksums and reports.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runtime.sh
source "${SCRIPT_DIR}/lib/runtime.sh"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
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
    _cmd_python - "${manifest_path}" "${field}" <<'PY'
import json
import sys

path = sys.argv[1]
field = sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
value = payload.get(field)
if value is None:
    raise SystemExit(1)
if isinstance(value, str):
    print(value)
else:
    print(str(value))
PY
}

pack_path_within_dir() {
    local dir_path="$1"
    local candidate_path="$2"
    _cmd_python - "${dir_path}" "${candidate_path}" <<'PY'
from pathlib import Path
import sys

dir_path = Path(sys.argv[1]).resolve()
candidate_path = Path(sys.argv[2]).resolve()

try:
    candidate_path.relative_to(dir_path)
except ValueError:
    raise SystemExit(1)

raise SystemExit(0)
PY
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

    local tmp_dir
    tmp_dir="$(mktemp -d 2>/dev/null || mktemp -d -t invarlock_pack_verify.XXXXXXXX)"
    local expected_file="${tmp_dir}/expected.txt"
    local actual_file="${tmp_dir}/actual.txt"
    local extra_file="${tmp_dir}/extra.txt"

    (
        cd "${pack_dir}"
        # The checksum file uses "hash  filename" (sha256sum) or "hash  filename" (shasum).
        # Normalize to paths without leading "./" and without sha256sum "*" markers.
        awk '{print $NF}' checksums.sha256 | sed 's/^\*//' | sed 's|^\./||' | sort -u > "${expected_file}"

        # Allow pack control files even if they're not in checksums.sha256.
        printf '%s\n' \
            "checksums.sha256" \
            "manifest.json" \
            "manifest.signature.json" \
            "metadata/manifest.json" \
            "metadata/manifest.signature.json" \
            "metadata/checksums.sha256" \
            >> "${expected_file}"
        sort -u -o "${expected_file}" "${expected_file}"

        find . -type f -print \
            | sed 's|^\./||' \
            | grep -v '^\\.DS_Store$' \
            | grep -v '/\\.DS_Store$' \
            | grep -v '^__MACOSX/' \
            | sort -u > "${actual_file}"
    )

    if comm -13 "${expected_file}" "${actual_file}" > "${extra_file}"; then
        if [[ -s "${extra_file}" ]]; then
            if [[ "${strict}" == "1" ]]; then
                echo "ERROR: Pack contains extra files not covered by checksums.sha256:" >&2
                sed 's/^/  - /' "${extra_file}" >&2
                rm -rf "${tmp_dir}"
                return 1
            fi
            pack_warn "Pack contains extra files not covered by checksums.sha256:"
            sed 's/^/  - /' "${extra_file}" >&2
        fi
    fi

    rm -rf "${tmp_dir}"
    return 0
}

pack_verify_signature_helper() {
    local pack_dir="$1"
    local strict="$2"
    local helper="${SCRIPT_DIR}/python/verify_signature.py"

    if [[ "${strict}" == "1" ]]; then
        _cmd_python "${helper}" --strict "${pack_dir}"
        return
    fi
    _cmd_python "${helper}" "${pack_dir}"
}

pack_verify_signature() {
    local pack_dir="$1"
    local strict="$2"
    local tmp_err=""
    local signer_fpr=""

    tmp_err="$(mktemp 2>/dev/null || mktemp -t invarlock_pack_verify_sig.XXXXXXXX)" || return 1
    if signer_fpr="$(pack_verify_signature_helper "${pack_dir}" "${strict}" 2>"${tmp_err}")"; then
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

pack_verify_reports() {
    local pack_dir="$1"
    local json_out="$2"
    local profile="${PACK_VERIFY_PROFILE:-dev}"
    local -a reports=()
    local -a reports_clean=()
    local -a reports_error=()
    while IFS= read -r report; do
        [[ -n "${report}" ]] || continue
        reports+=("${report}")
        if [[ "${report}" == */errors/*/evaluation.report.json ]]; then
            reports_error+=("${report}")
        else
            reports_clean+=("${report}")
        fi
    done < <(find "${pack_dir}/reports" -type f -name "evaluation.report.json" | sort)
    if [[ ${#reports[@]} -eq 0 ]]; then
        echo "ERROR: No reports found in pack." >&2
        return 1
    fi

    if [[ ${#reports_clean[@]} -eq 0 ]]; then
        echo "ERROR: No clean reports found in pack (only error-injection reports present)." >&2
        return 1
    fi

    if [[ -n "${json_out}" ]]; then
        invarlock verify --json --profile "${profile}" "${reports_clean[@]}" > "${json_out}"
    else
        invarlock verify --json --profile "${profile}" "${reports_clean[@]}"
    fi

    if [[ ${#reports_error[@]} -gt 0 ]]; then
        invarlock verify --json --profile "${profile}" "${reports_error[@]}" >/dev/null || true
    fi
}

pack_verify_pack() {
    set -euo pipefail

    local pack_dir=""
    local json_out=""
    local skip_verify=0
    local strict="${PACK_STRICT_MODE:-0}"

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

    if ! pack_validate_manifest_schema "${pack_dir}"; then
        return "${PACK_VERIFY_FORMAT}"
    fi
    if ! pack_verify_signature "${pack_dir}" "${strict}"; then
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
