#!/usr/bin/env bash
# verify_pack.sh - Validate proof pack checksums and reports.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK_VERIFY_OK=0
PACK_VERIFY_USAGE=2
PACK_VERIFY_MISSING=3
PACK_VERIFY_FORMAT=4
PACK_VERIFY_SIGNATURE=5
PACK_VERIFY_INTEGRITY=6
PACK_VERIFY_CERTS=7

pack_usage() {
    cat <<'EOF'
Usage: scripts/proof_packs/verify_pack.sh --pack DIR [options]

Options:
  --pack DIR          Proof pack directory to verify
  --json-out FILE     Write verify JSON output to FILE (must be outside the pack)
  --skip-verify       Skip invarlock verify step
  --strict            Fail closed on missing/invalid signatures and pack mismatches
  --help              Show this help message

Exit codes:
  0  verified
  2  invalid usage / arguments
  3  missing pack or required pack files
  4  manifest format / schema validation failed
  5  signature verification failed
  6  integrity verification failed (checksum binding, checksums, attestation refs, or strict extra-file checks)
  7  cert verification failed
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
    python3 - "${manifest_path}" "${field}" <<'PY'
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
    python3 - "${dir_path}" "${candidate_path}" <<'PY'
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

    if ! out="$(python3 "${validator}" "${pack_dir}/manifest.json" 2>&1)"; then
        echo "ERROR: manifest.json failed contract validation." >&2
        printf '%s\n' "${out}" >&2
        return 1
    fi
    return 0
}

pack_verify_manifest_attestation() {
    local pack_dir="$1"
    local verifier="${SCRIPT_DIR}/python/verify_manifest_attestation.py"
    local out

    if ! out="$(python3 "${verifier}" "${pack_dir}" 2>&1)"; then
        echo "ERROR: manifest.json attestation references failed verification." >&2
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
            "manifest.json.asc" \
            "metadata/manifest.json" \
            "metadata/manifest.json.asc" \
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

pack_verify_gpg() {
    local pack_dir="$1"
    local strict="$2"

    if [[ ! -f "${pack_dir}/manifest.json.asc" ]]; then
        if [[ "${strict}" == "1" ]]; then
            echo "ERROR: manifest.json.asc missing (strict mode requires a signed manifest)." >&2
            return 1
        fi
        pack_warn "manifest.json.asc missing; pack is unsigned."
        return 0
    fi

    if ! command -v gpg >/dev/null 2>&1; then
        if [[ "${strict}" == "1" ]]; then
            echo "ERROR: gpg not found (strict mode requires signature verification)." >&2
            return 1
        fi
        pack_warn "gpg not found; skipping manifest signature verification."
        return 0
    fi

    local out
    if ! out="$(gpg --status-fd 1 --verify "${pack_dir}/manifest.json.asc" "${pack_dir}/manifest.json" 2>&1)"; then
        echo "ERROR: manifest signature verification failed." >&2
        printf '%s\n' "${out}" >&2
        return 1
    fi

    local signer_fpr
    signer_fpr="$(printf '%s\n' "${out}" | awk '/VALIDSIG / {print $3; exit}')"
    if [[ -n "${signer_fpr}" ]]; then
        PACK_SIGNER_FINGERPRINT="${signer_fpr}"
        export PACK_SIGNER_FINGERPRINT

        local recorded
        recorded="$(pack_manifest_field "${pack_dir}/manifest.json" "signing_key_fingerprint" 2>/dev/null || true)"
        if [[ -n "${recorded}" && "${recorded}" != "${signer_fpr}" ]]; then
            echo "ERROR: manifest.json signing_key_fingerprint (${recorded}) does not match signature key (${signer_fpr})." >&2
            return 1
        fi
    fi

    return 0
}

pack_verify_certs() {
    local pack_dir="$1"
    local json_out="$2"
    local profile="${PACK_VERIFY_PROFILE:-dev}"
    local -a certs=()
    local -a certs_clean=()
    local -a certs_error=()
    while IFS= read -r cert; do
        [[ -n "${cert}" ]] || continue
        certs+=("${cert}")
        if [[ "${cert}" == */errors/*/evaluation.report.json ]]; then
            certs_error+=("${cert}")
        else
            certs_clean+=("${cert}")
        fi
    done < <(find "${pack_dir}/certs" -type f -name "evaluation.report.json" | sort)
    if [[ ${#certs[@]} -eq 0 ]]; then
        echo "ERROR: No reports found in pack." >&2
        return 1
    fi

    if [[ ${#certs_clean[@]} -eq 0 ]]; then
        echo "ERROR: No clean reports found in pack (only error-injection reports present)." >&2
        return 1
    fi

    if [[ -n "${json_out}" ]]; then
        invarlock verify --json --profile "${profile}" "${certs_clean[@]}" > "${json_out}"
    else
        invarlock verify --json --profile "${profile}" "${certs_clean[@]}"
    fi

    if [[ ${#certs_error[@]} -gt 0 ]]; then
        invarlock verify --json --profile "${profile}" "${certs_error[@]}" >/dev/null || true
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
    if ! pack_verify_gpg "${pack_dir}" "${strict}"; then
        return "${PACK_VERIFY_SIGNATURE}"
    fi
    if ! pack_verify_manifest_binds_checksums "${pack_dir}" "${strict}"; then
        return "${PACK_VERIFY_INTEGRITY}"
    fi
    if ! pack_verify_checksums "${pack_dir}"; then
        return "${PACK_VERIFY_INTEGRITY}"
    fi
    if ! pack_verify_manifest_attestation "${pack_dir}"; then
        return "${PACK_VERIFY_INTEGRITY}"
    fi
    if ! pack_verify_no_extra_files "${pack_dir}" "${strict}"; then
        return "${PACK_VERIFY_INTEGRITY}"
    fi

    if [[ "${skip_verify}" -eq 0 ]]; then
        if ! pack_verify_certs "${pack_dir}" "${json_out}"; then
            return "${PACK_VERIFY_CERTS}"
        fi
    fi

    return "${PACK_VERIFY_OK}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    pack_verify_pack "$@"
fi
