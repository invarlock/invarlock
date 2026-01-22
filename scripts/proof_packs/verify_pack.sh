#!/usr/bin/env bash
# verify_pack.sh - Validate proof pack checksums and certificates.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pack_usage() {
    cat <<'EOF'
Usage: scripts/proof_packs/verify_pack.sh --pack DIR [options]

Options:
  --pack DIR          Proof pack directory to verify
  --json-out FILE     Write verify JSON output to FILE
  --skip-verify       Skip invarlock verify step
  --help              Show this help message
EOF
}

pack_sha256_cmd() {
    if command -v sha256sum >/dev/null 2>&1; then
        echo "sha256sum"
    else
        echo "shasum -a 256"
    fi
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

pack_verify_gpg() {
    local pack_dir="$1"
    if [[ -f "${pack_dir}/manifest.json.asc" ]]; then
        if command -v gpg >/dev/null 2>&1; then
            gpg --verify "${pack_dir}/manifest.json.asc" "${pack_dir}/manifest.json"
        else
            echo "WARNING: gpg not found; skipping manifest signature verification." >&2
        fi
    fi
}

pack_verify_certs() {
    local pack_dir="$1"
    local json_out="$2"
    local -a certs=()
    while IFS= read -r cert; do
        [[ -n "${cert}" ]] || continue
        certs+=("${cert}")
    done < <(find "${pack_dir}" -type f -name "*.cert.json" | sort)
    if [[ ${#certs[@]} -eq 0 ]]; then
        echo "ERROR: No certificates found in pack." >&2
        return 1
    fi

    if [[ -n "${json_out}" ]]; then
        invarlock verify --json "${certs[@]}" > "${json_out}"
    else
        invarlock verify --json "${certs[@]}"
    fi
}

pack_verify_pack() {
    set -euo pipefail

    local pack_dir=""
    local json_out=""
    local skip_verify=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                pack_usage
                return 0
                ;;
            --pack)
                pack_dir="${2:-}"
                if [[ -z "${pack_dir}" ]]; then
                    echo "ERROR: --pack requires a value" >&2
                    return 2
                fi
                shift 2
                ;;
            --json-out)
                json_out="${2:-}"
                if [[ -z "${json_out}" ]]; then
                    echo "ERROR: --json-out requires a value" >&2
                    return 2
                fi
                shift 2
                ;;
            --skip-verify)
                skip_verify=1
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

    if [[ -z "${pack_dir}" ]]; then
        echo "ERROR: --pack is required" >&2
        pack_usage >&2
        return 2
    fi
    if [[ ! -d "${pack_dir}" ]]; then
        echo "ERROR: Pack directory not found: ${pack_dir}" >&2
        return 1
    fi
    if [[ ! -f "${pack_dir}/manifest.json" ]]; then
        echo "ERROR: manifest.json missing in pack." >&2
        return 1
    fi
    if [[ ! -f "${pack_dir}/checksums.sha256" ]]; then
        echo "ERROR: checksums.sha256 missing in pack." >&2
        return 1
    fi

    pack_verify_checksums "${pack_dir}"
    pack_verify_gpg "${pack_dir}"

    if [[ "${skip_verify}" -eq 0 ]]; then
        pack_verify_certs "${pack_dir}" "${json_out}"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    pack_verify_pack "$@"
fi
