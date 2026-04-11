#!/usr/bin/env bash
#
# Generate a CycloneDX SBOM for a Python environment.
# Requires the `cyclonedx-bom` CLI (`pip install cyclonedx-bom`).

set -euo pipefail

SCOPE="environment"
PYTHON_PATH=""
OUTPUT_PATH="artifacts/supply-chain/sbom.json"

usage() {
  cat <<'EOF'
Usage: scripts/generate_sbom.sh [--scope environment|tool-environment|install-surface] [--python PYTHON] [OUTPUT_PATH]

Options:
  --scope SCOPE     Label for the environment being scanned.
  --python PYTHON   Python interpreter or virtual environment to inspect.
  --help            Show this help message.

When --scope is install-surface, --python must point to the installed-artifact
environment that should be scanned.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope)
      SCOPE="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_PATH="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      OUTPUT_PATH="$1"
      shift
      ;;
  esac
done

if [[ -z "${SCOPE}" ]]; then
  echo "ERROR: --scope must not be empty." >&2
  exit 2
fi

if [[ "${SCOPE}" == "install-surface" && -z "${PYTHON_PATH}" ]]; then
  echo "ERROR: --python is required when --scope install-surface is used." >&2
  exit 2
fi

if [[ -z "${PYTHON_PATH}" ]]; then
  PYTHON_PATH="$(command -v python3 || command -v python)"
fi

OUTPUT_DIR="$(dirname "${OUTPUT_PATH}")"

if ! command -v cyclonedx-py >/dev/null 2>&1; then
  echo "ERROR: cyclonedx CLI not found. Install it with 'pip install cyclonedx-bom'." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Generating ${SCOPE} SBOM from ${PYTHON_PATH}" >&2

cyclonedx-py environment "${PYTHON_PATH}" \
  --spec-version 1.4 \
  --output-format JSON \
  --output-file "${OUTPUT_PATH}"

echo "SBOM written to ${OUTPUT_PATH} (${SCOPE})"
