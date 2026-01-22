#!/usr/bin/env bash
# run_pack.sh - Run a proof pack suite and package artifacts.

RUN_PACK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=run_suite.sh
source "${RUN_PACK_SCRIPT_DIR}/run_suite.sh"

pack_usage() {
    cat <<'EOF'
Usage: scripts/proof_packs/run_pack.sh [options]

Options:
  --suite NAME         Suite name (subset|full)
  --net 1|0            Enable network access for preflight/downloads (default: 0)
  --out DIR            Output directory for the run (default: ./proof_pack_runs/<suite>_<timestamp>)
  --pack-dir DIR       Output directory for the proof pack (default: <out>/proof_pack)
  --layout NAME        Pack layout (v1|v2) (default: v1)
  --determinism MODE   Determinism mode (strict|throughput)
  --repeats N          Determinism repeat count metadata (default: 0)
  --calibrate-only     Only run calibration tasks (implies PACK_SUITE_MODE=calibrate-only)
  --run-only           Run edits/certs only (implies resume)
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

pack_collect_certs() {
    local run_dir="$1"
    find "${run_dir}" -type f -name "evaluation.cert.json" -path "*/certificates/*" ! -path "*/cert/*" | sort
}

pack_cert_rel_path() {
    local run_dir="$1"
    local cert_path="$2"
    local rel="${cert_path#"${run_dir}/"}"
    local model="${rel%%/*}"
    local remainder="${rel#*/certificates/}"
    remainder="${remainder%/evaluation.cert.json}"
    if [[ -z "${model}" || "${remainder}" == "${rel}" ]]; then
        return 1
    fi
    printf '%s/%s\n' "${model}" "${remainder}"
}

pack_generate_html() {
    local pack_dir="$1"
    local cert
    while IFS= read -r cert; do
        [[ -n "${cert}" ]] || continue
        local html="${cert%.cert.json}.html"
        if ! invarlock report html --input "${cert}" --output "${html}" --force >/dev/null; then
            echo "WARNING: Failed to render HTML report for ${cert}" >&2
        fi
    done < <(find "${pack_dir}/certs" -type f -name "evaluation.cert.json" | sort)
}

pack_verify_certs() {
    local pack_dir="$1"
    local profile="${PACK_VERIFY_PROFILE:-dev}"
    local count_clean=0
    local count_error=0
    local count_failed=0
    local cert
    while IFS= read -r cert; do
        [[ -n "${cert}" ]] || continue
        local cert_dir
        cert_dir="$(dirname "${cert}")"
        if [[ "${cert}" == */errors/*/evaluation.cert.json ]]; then
            # Error injection certs are expected to fail verify (unsafe edits by design).
            invarlock verify --json --profile "${profile}" "${cert}" > "${cert_dir}/verify.json" || true
            count_error=$((count_error + 1))
            continue
        fi

        if invarlock verify --json --profile "${profile}" "${cert}" > "${cert_dir}/verify.json"; then
            count_clean=$((count_clean + 1))
        else
            echo "ERROR: Unexpected verify failure: ${cert}" >&2
            count_failed=$((count_failed + 1))
        fi
    done < <(find "${pack_dir}/certs" -type f -name "evaluation.cert.json" | sort)

    local total=$((count_clean + count_error + count_failed))
    if [[ ${total} -eq 0 ]]; then
        echo "ERROR: No certificates found to verify." >&2
        return 1
    fi

    PACK_VERIFY_COUNT_CLEAN="${count_clean}"
    PACK_VERIFY_COUNT_ERROR="${count_error}"
    PACK_VERIFY_COUNT_FAILED="${count_failed}"
    PACK_VERIFY_PROFILE_USED="${profile}"
    export PACK_VERIFY_COUNT_CLEAN PACK_VERIFY_COUNT_ERROR PACK_VERIFY_COUNT_FAILED PACK_VERIFY_PROFILE_USED

    local results_dir="${pack_dir}/results"
    mkdir -p "${results_dir}"
    python3 "${RUN_PACK_SCRIPT_DIR}/python/write_verification_summary.py" \
        "${results_dir}/verification_summary.json" \
        "${count_clean}" \
        "${count_error}" \
        "${count_failed}" \
        "${profile}"

    echo "Verified: ${count_clean} clean, ${count_error} error-injection (expected fail), ${count_failed} unexpected failures"

    if [[ ${count_failed} -gt 0 ]]; then
        return 1
    fi
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

pack_sign_manifest() {
    local pack_dir="$1"
    if [[ "${PACK_GPG_SIGN:-1}" == "0" ]]; then
        return 0
    fi
    if command -v gpg >/dev/null 2>&1; then
        if ! gpg --armor --detach-sign --output "${pack_dir}/manifest.json.asc" "${pack_dir}/manifest.json"; then
            rm -f "${pack_dir}/manifest.json.asc"
            echo "WARNING: gpg signing failed; skipping manifest signature." >&2
        fi
    else
        echo "WARNING: gpg not found; skipping manifest signature." >&2
    fi
}

pack_write_checksums() {
    local pack_dir="$1"
    local sha_cmd
    sha_cmd="$(pack_sha256_cmd)"
    (
        cd "${pack_dir}"
        while IFS= read -r file; do
            [[ -n "${file}" ]] || continue
            ${sha_cmd} "${file}"
        done < <(find . -type f ! -name "checksums.sha256" -print | sort) > "checksums.sha256"
    )
}

pack_write_readme() {
    local pack_dir="$1"
    echo "[run_pack.sh] Writing README.md to ${pack_dir}" >&2
    cat > "${pack_dir}/README.md" <<'EOF'
# InvarLock Proof Pack

This proof pack bundles certificates, summary reports, and metadata for offline
verification. No model weights are included.

## Verify

1) Verify the manifest signature (if present):
   gpg --verify manifest.json.asc manifest.json

2) Verify file checksums:
   sha256sum -c checksums.sha256
   # macOS: shasum -a 256 -c checksums.sha256

3) Verify certificate integrity:
   invarlock verify --json certs/**/evaluation.cert.json

Or use:
  scripts/proof_packs/verify_pack.sh --pack <pack-dir>
EOF
}

pack_build_pack() {
    local run_dir="$1"
    local pack_dir="$2"

    if [[ -z "${run_dir}" || -z "${pack_dir}" ]]; then
        echo "ERROR: pack_build_pack requires run_dir and pack_dir." >&2
        return 1
    fi
    if [[ ! -d "${run_dir}" ]]; then
        echo "ERROR: run_dir not found: ${run_dir}" >&2
        return 1
    fi

    if [[ -d "${pack_dir}" && -n "$(ls -A "${pack_dir}" 2>/dev/null)" ]]; then
        echo "ERROR: pack_dir already exists and is not empty: ${pack_dir}" >&2
        return 1
    fi

    pack_require_cmd invarlock

    mkdir -p "${pack_dir}"

    local layout="${PACK_PACK_LAYOUT:-v1}"
    case "${layout}" in
        v2|enhanced)
            layout="v2"
            ;;
        v1|flat|legacy|"")
            layout="v1"
            ;;
        *)
            echo "ERROR: Unknown pack layout: ${layout} (expected v1|v2)" >&2
            return 2
            ;;
    esac

    local results_dir="${pack_dir}/results"
    local verdicts_dir="${results_dir}"
    local analysis_dir="${results_dir}"
    local revisions_dest="${pack_dir}/state/model_revisions.json"
    local scenarios_dest="${pack_dir}/state/scenarios.json"
    if [[ "${layout}" == "v2" ]]; then
        verdicts_dir="${results_dir}/verdicts"
        analysis_dir="${results_dir}/analysis"
        revisions_dest="${pack_dir}/metadata/model_revisions.json"
        scenarios_dest="${pack_dir}/metadata/scenarios.json"
    fi

    mkdir -p "${results_dir}" "${verdicts_dir}" "${analysis_dir}"

    pack_copy_file "${run_dir}/reports/final_verdict.txt" "${verdicts_dir}/final_verdict.txt"
    pack_copy_file "${run_dir}/reports/final_verdict.json" "${verdicts_dir}/final_verdict.json"
    pack_copy_optional "${run_dir}/analysis/determinism_repeats.json" "${analysis_dir}/determinism_repeats.json"

    pack_copy_optional "${run_dir}/state/model_revisions.json" "${revisions_dest}"
    pack_copy_optional "${run_dir}/state/scenarios.json" "${scenarios_dest}"

    local cert
    while IFS= read -r cert; do
        [[ -n "${cert}" ]] || continue
        local rel
        rel="$(pack_cert_rel_path "${run_dir}" "${cert}")" || continue
        local dest_dir="${pack_dir}/certs/${rel}"
        mkdir -p "${dest_dir}"
        cp "${cert}" "${dest_dir}/evaluation.cert.json"
    done < <(pack_collect_certs "${run_dir}")

    local verify_rc=0
    if pack_verify_certs "${pack_dir}"; then
        verify_rc=0
    else
        verify_rc=$?
    fi

    if [[ "${PACK_SKIP_HTML:-0}" != "1" ]]; then
        pack_generate_html "${pack_dir}"
    fi

    pack_write_readme "${pack_dir}"
    pack_write_manifest "${pack_dir}" "${run_dir}" "${PACK_SUITE:-}" "${PACK_NET:-0}" "${PACK_DETERMINISM:-}" "${PACK_REPEATS:-0}"
    pack_sign_manifest "${pack_dir}"
    if [[ "${layout}" == "v2" ]]; then
        mkdir -p "${pack_dir}/metadata"
        cp "${pack_dir}/manifest.json" "${pack_dir}/metadata/manifest.json"
        if [[ -f "${pack_dir}/manifest.json.asc" ]]; then
            cp "${pack_dir}/manifest.json.asc" "${pack_dir}/metadata/manifest.json.asc"
        fi
    fi
    pack_write_checksums "${pack_dir}"
    if [[ "${layout}" == "v2" ]]; then
        cp "${pack_dir}/checksums.sha256" "${pack_dir}/metadata/checksums.sha256"
    fi

    return "${verify_rc}"
}

pack_run_pack() {
    set -euo pipefail

    local suite="${PACK_SUITE:-subset}"
    local net="${PACK_NET:-0}"
    local out="${PACK_OUTPUT_DIR:-${OUTPUT_DIR:-}}"
    local determinism="${PACK_DETERMINISM:-throughput}"
    local repeats="${PACK_REPEATS:-0}"
    local suite_mode="${PACK_SUITE_MODE:-full}"
    local resume_flag="${RESUME_FLAG:-false}"
    local pack_dir="${PACK_DIR:-}"
    local layout="${PACK_PACK_LAYOUT:-v1}"

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
        out="./proof_pack_runs/${suite}_${stamp}"
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
    if [[ "${suite_mode}" == "calibrate-only" ]]; then
        run_args+=("--calibrate-only")
    elif [[ "${suite_mode}" == "run-only" ]]; then
        run_args+=("--run-only")
    elif [[ "${resume_flag}" == "true" ]]; then
        run_args+=("--resume")
    fi

    pack_entrypoint "${run_args[@]}"
    PACK_PACK_LAYOUT="${layout}"
    export PACK_PACK_LAYOUT

    if [[ -z "${pack_dir}" ]]; then
        pack_dir="${out}/proof_pack"
    fi

    pack_build_pack "${out}" "${pack_dir}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    pack_run_pack "$@"
fi
