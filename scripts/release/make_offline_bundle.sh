#!/usr/bin/env bash
#
# Assemble a release-side offline verification bundle from existing build outputs.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/release/make_offline_bundle.sh [options]

Required:
  --version VERSION       Release version without leading "v" (for example: 0.3.12)
  --tag TAG               Release tag (for example: v0.3.12)
  --repo OWNER/REPO       GitHub repository slug
  --dist-dir DIR          Directory containing release distributions and Sigstore sidecars
  --sbom PATH             CycloneDX SBOM JSON path
  --provenance-dir DIR    Directory containing GitHub provenance bundle files
  --output-dir DIR        Output directory for the offline bundle tarball

Optional:
  --bundle-name NAME      Bundle base name (default: invarlock-<version>-offline-bundle)
  --issuer URL            Expected OIDC issuer (default: https://token.actions.githubusercontent.com)
  --help                  Show this help message
EOF
}

require_cmd() {
    local cmd="$1"
    command -v "${cmd}" >/dev/null 2>&1 || {
        echo "ERROR: Required command not found: ${cmd}" >&2
        return 1
    }
}

sha256_cmd() {
    if command -v sha256sum >/dev/null 2>&1; then
        echo "sha256sum"
    else
        echo "shasum -a 256"
    fi
}

VERSION=""
TAG=""
REPO=""
DIST_DIR=""
SBOM_PATH=""
PROVENANCE_DIR=""
OUTPUT_DIR=""
BUNDLE_NAME=""
OIDC_ISSUER="https://token.actions.githubusercontent.com"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            VERSION="${2:-}"
            shift 2
            ;;
        --tag)
            TAG="${2:-}"
            shift 2
            ;;
        --repo)
            REPO="${2:-}"
            shift 2
            ;;
        --dist-dir)
            DIST_DIR="${2:-}"
            shift 2
            ;;
        --sbom)
            SBOM_PATH="${2:-}"
            shift 2
            ;;
        --provenance-dir)
            PROVENANCE_DIR="${2:-}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:-}"
            shift 2
            ;;
        --bundle-name)
            BUNDLE_NAME="${2:-}"
            shift 2
            ;;
        --issuer)
            OIDC_ISSUER="${2:-}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${VERSION}" || -z "${TAG}" || -z "${REPO}" || -z "${DIST_DIR}" || -z "${SBOM_PATH}" || -z "${PROVENANCE_DIR}" || -z "${OUTPUT_DIR}" ]]; then
    echo "ERROR: Missing required arguments." >&2
    usage >&2
    exit 2
fi

require_cmd python3
require_cmd tar

if [[ ! -d "${DIST_DIR}" ]]; then
    echo "ERROR: dist directory not found: ${DIST_DIR}" >&2
    exit 1
fi
if [[ ! -f "${SBOM_PATH}" ]]; then
    echo "ERROR: SBOM file not found: ${SBOM_PATH}" >&2
    exit 1
fi
if [[ ! -d "${PROVENANCE_DIR}" ]]; then
    echo "ERROR: provenance directory not found: ${PROVENANCE_DIR}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [[ -z "${BUNDLE_NAME}" ]]; then
    BUNDLE_NAME="invarlock-${VERSION}-offline-bundle"
fi

SBOM_BASENAME="invarlock-${VERSION}-sbom.cdx.json"
BUNDLE_TARBALL="${OUTPUT_DIR}/${BUNDLE_NAME}.tar.gz"
STAGING_DIR="$(mktemp -d "${OUTPUT_DIR}/.${BUNDLE_NAME}.tmp.XXXXXX")"
trap 'rm -rf "${STAGING_DIR}"' EXIT

BUNDLE_ROOT="${STAGING_DIR}/${BUNDLE_NAME}"
mkdir -p "${BUNDLE_ROOT}/dist" "${BUNDLE_ROOT}/provenance"
cp -R "${DIST_DIR}/." "${BUNDLE_ROOT}/dist/"
cp -R "${PROVENANCE_DIR}/." "${BUNDLE_ROOT}/provenance/"
cp "${SBOM_PATH}" "${BUNDLE_ROOT}/${SBOM_BASENAME}"

cat > "${BUNDLE_ROOT}/README.txt" <<EOF
InvarLock Offline Release Bundle

This bundle packages the published distributions, Sigstore verification sidecars,
the GitHub build-provenance bundle, and the CycloneDX SBOM for offline review.

Recommended verification flow:
1. Compare the sha256 for each file in dist/ against release_manifest.json.
2. For each distribution artifact in dist/, verify its Sigstore bundle with:
     cosign verify-blob dist/<artifact> \\
       --bundle dist/<artifact>.sigstore.json \\
       --certificate-identity "repo:${REPO}@refs/tags/${TAG}" \\
       --certificate-oidc-issuer "${OIDC_ISSUER}"
3. Review provenance/* for the GitHub build-provenance attestation.
4. Review ${SBOM_BASENAME} with an offline CycloneDX-capable scanner.

GPG detached signatures are not included in this release bundle.
EOF

cat > "${BUNDLE_ROOT}/public_key_hints.txt" <<EOF
Sigstore OIDC issuer: ${OIDC_ISSUER}
Expected certificate identity: repo:${REPO}@refs/tags/${TAG}
GitHub provenance bundle location: provenance/
SBOM path: ${SBOM_BASENAME}
GPG manifest signature: not included in this release bundle
EOF

python3 - "${BUNDLE_ROOT}" "${BUNDLE_NAME}" "${VERSION}" "${TAG}" "${REPO}" "${OIDC_ISSUER}" "${SBOM_BASENAME}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(root: Path, path: Path, *, kind: str, extra: dict[str, object] | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "kind": kind,
    }
    if extra:
        record.update(extra)
    return record


root = Path(sys.argv[1])
bundle_name = sys.argv[2]
version = sys.argv[3]
tag = sys.argv[4]
repo = sys.argv[5]
issuer = sys.argv[6]
sbom_name = sys.argv[7]

dist_dir = root / "dist"
provenance_dir = root / "provenance"
sbom_path = root / sbom_name

dist_files = sorted(path for path in dist_dir.iterdir() if path.is_file())
if not dist_files:
    raise SystemExit("ERROR: offline bundle requires at least one file under dist/")

primary_dist = [
    path
    for path in dist_files
    if path.name.endswith(".whl") or path.name.endswith(".tar.gz")
]
if not primary_dist:
    raise SystemExit("ERROR: offline bundle requires at least one wheel or sdist artifact")

provenance_files = sorted(path for path in provenance_dir.rglob("*") if path.is_file())
if not provenance_files:
    raise SystemExit("ERROR: offline bundle requires at least one provenance file")

if not sbom_path.is_file():
    raise SystemExit(f"ERROR: offline bundle SBOM missing: {sbom_name}")

allowed_sidecar_suffixes = (
    ".sigstore",
    ".sigstore.json",
    ".crt",
    ".sig",
    ".pem",
)

distributions: list[dict[str, object]] = []
distribution_sidecars: list[dict[str, object]] = []
for artifact in primary_dist:
    sidecars = [
        candidate
        for candidate in dist_files
        if candidate.name.startswith(f"{artifact.name}.")
        and candidate.name != artifact.name
        and any(candidate.name.endswith(suffix) for suffix in allowed_sidecar_suffixes)
    ]
    if not any(sidecar.name.endswith((".sigstore", ".sigstore.json")) for sidecar in sidecars):
        raise SystemExit(
            f"ERROR: missing Sigstore bundle for distribution artifact: {artifact.name}"
        )
    distributions.append(
        file_record(
            root,
            artifact,
            kind="distribution",
            extra={
                "sigstore_sidecars": [
                    sidecar.relative_to(root).as_posix() for sidecar in sorted(sidecars)
                ]
            },
        )
    )
    distribution_sidecars.extend(
        file_record(root, sidecar, kind="distribution_signature")
        for sidecar in sorted(sidecars)
    )

payload = {
    "schema": "invarlock/release-offline-bundle-v1",
    "bundle": {
        "name": bundle_name,
        "version": version,
        "tag": tag,
        "repository": repo,
        "created_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    },
    "verification": {
        "oidc_issuer": issuer,
        "certificate_identity": f"repo:{repo}@refs/tags/{tag}",
    },
    "distributions": distributions,
    "distribution_signatures": distribution_sidecars,
    "sbom": file_record(
        root,
        sbom_path,
        kind="sbom",
        extra={"format": "cyclonedx-1.4"},
    ),
    "provenance_bundles": [
        file_record(root, path, kind="provenance") for path in provenance_files
    ],
    "supporting_files": [
        file_record(root, root / "README.txt", kind="documentation"),
        file_record(root, root / "public_key_hints.txt", kind="verification_hints"),
    ],
}

manifest_path = root / "release_manifest.json"
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

( cd "${STAGING_DIR}" && tar -czf "${BUNDLE_TARBALL}" "${BUNDLE_NAME}" )

echo "Offline release bundle written to ${BUNDLE_TARBALL}"
