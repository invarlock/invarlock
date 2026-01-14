#!/usr/bin/env bash
# run_pack.sh - Run a proof pack suite and package artifacts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=run_suite.sh
source "${SCRIPT_DIR}/run_suite.sh"

pack_usage() {
    cat <<'EOF'
Usage: scripts/proof_packs/run_pack.sh [options]

Options:
  --suite NAME         Suite name (subset|full)
  --net 1|0            Enable network access for preflight/downloads (default: 0)
  --out DIR            Output directory for the run (default: ./proof_pack_runs/<suite>_<timestamp>)
  --pack-dir DIR       Output directory for the proof pack (default: <out>/proof_pack)
  --determinism MODE   Determinism mode (strict|throughput)
  --repeats N          Determinism repeat count metadata (default: 0)
  --calibrate-only     Only run calibration tasks (implies PACK_SUITE_MODE=calibrate-only)
  --run-only           Run edits/evals only (implies resume)
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
        invarlock report html --input "${cert}" --output "${html}" --force >/dev/null
    done < <(find "${pack_dir}/certs" -type f -name "evaluation.cert.json" | sort)
}

pack_verify_certs() {
    local pack_dir="$1"
    local profile="${PACK_VERIFY_PROFILE:-dev}"
    local count=0
    local cert
    while IFS= read -r cert; do
        [[ -n "${cert}" ]] || continue
        local cert_dir
        cert_dir="$(dirname "${cert}")"
        invarlock verify --json --profile "${profile}" "${cert}" > "${cert_dir}/verify.json"
        count=$((count + 1))
    done < <(find "${pack_dir}/certs" -type f -name "evaluation.cert.json" | sort)
    if [[ ${count} -eq 0 ]]; then
        echo "ERROR: No certificates found to verify." >&2
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

    python3 - "${pack_dir}" "${run_dir}" "${suite}" "${net}" "${determinism}" "${repeats}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

pack_dir = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
suite = sys.argv[3]
net = sys.argv[4]
determinism = sys.argv[5]
repeats_raw = sys.argv[6]

try:
    repeats = int(repeats_raw)
except Exception:
    repeats = 0

models = []
model_list = []
revisions_path = pack_dir / "state" / "model_revisions.json"
if revisions_path.is_file():
    try:
        data = json.loads(revisions_path.read_text())
    except Exception:
        data = {}
    model_list = data.get("model_list") or []
    for model_id, info in (data.get("models") or {}).items():
        if not isinstance(info, dict):
            info = {}
        models.append({
            "model_id": model_id,
            "revision": info.get("revision") or "",
        })

determinism_repeats = None
det_path = pack_dir / "results" / "determinism_repeats.json"
if det_path.is_file():
    try:
        determinism_repeats = json.loads(det_path.read_text())
    except Exception:
        determinism_repeats = None

artifacts = []
for path in pack_dir.rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(pack_dir)
    if rel.name in {"manifest.json", "manifest.json.asc", "checksums.sha256"}:
        continue
    artifacts.append(str(rel))

try:
    import invarlock
    version = getattr(invarlock, "__version__", "")
except Exception:
    version = ""

payload = {
    "format": "proof-pack-v1",
    "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "suite": suite,
    "network_mode": "online" if str(net) in {"1", "true", "yes", "on"} else "offline",
    "determinism": determinism,
    "repeats": repeats,
    "determinism_repeats": determinism_repeats,
    "run_dir": str(run_dir),
    "invarlock_version": version,
    "model_list": model_list,
    "models": sorted(models, key=lambda item: item.get("model_id", "")),
    "artifacts": sorted(artifacts),
    "checksums_sha256": "checksums.sha256",
}

out_path = pack_dir / "manifest.json"
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

pack_sign_manifest() {
    local pack_dir="$1"
    if [[ "${PACK_GPG_SIGN:-1}" == "0" ]]; then
        return 0
    fi
    if command -v gpg >/dev/null 2>&1; then
        gpg --armor --detach-sign --output "${pack_dir}/manifest.json.asc" "${pack_dir}/manifest.json"
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
        find . -type f ! -name "checksums.sha256" -print | sort | ${sha_cmd} > "checksums.sha256"
    )
}

pack_write_readme() {
    local pack_dir="$1"
    cat > "${pack_dir}/README.md" <<'EOF'
# InvarLock Proof Pack

This proof pack bundles certificates, summary reports, and metadata for offline
verification. No model weights or raw lm-eval logs are included.

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

    local results_dir="${pack_dir}/results"
    mkdir -p "${results_dir}"

    pack_copy_file "${run_dir}/reports/final_verdict.txt" "${results_dir}/final_verdict.txt"
    pack_copy_file "${run_dir}/reports/final_verdict.json" "${results_dir}/final_verdict.json"
    pack_copy_file "${run_dir}/analysis/eval_results.csv" "${results_dir}/eval_results.csv"
    pack_copy_optional "${run_dir}/analysis/guard_sensitivity_matrix.csv" "${results_dir}/guard_sensitivity_matrix.csv"
    pack_copy_optional "${run_dir}/analysis/determinism_repeats.json" "${results_dir}/determinism_repeats.json"

    pack_copy_optional "${run_dir}/state/model_revisions.json" "${pack_dir}/state/model_revisions.json"

    local cert
    while IFS= read -r cert; do
        [[ -n "${cert}" ]] || continue
        local rel
        rel="$(pack_cert_rel_path "${run_dir}" "${cert}")" || continue
        local dest_dir="${pack_dir}/certs/${rel}"
        mkdir -p "${dest_dir}"
        cp "${cert}" "${dest_dir}/evaluation.cert.json"
    done < <(pack_collect_certs "${run_dir}")

    pack_verify_certs "${pack_dir}"

    if [[ "${PACK_SKIP_HTML:-0}" != "1" ]]; then
        pack_generate_html "${pack_dir}"
    fi

    pack_write_readme "${pack_dir}"
    pack_write_manifest "${pack_dir}" "${run_dir}" "${PACK_SUITE:-}" "${PACK_NET:-0}" "${PACK_DETERMINISM:-}" "${PACK_REPEATS:-0}"
    pack_sign_manifest "${pack_dir}"
    pack_write_checksums "${pack_dir}"
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

    if [[ -z "${pack_dir}" ]]; then
        pack_dir="${out}/proof_pack"
    fi

    pack_build_pack "${out}" "${pack_dir}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    pack_run_pack "$@"
fi
