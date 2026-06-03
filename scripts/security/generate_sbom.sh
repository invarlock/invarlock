#!/usr/bin/env bash
#
# Generate a CycloneDX SBOM for a Python environment.
# Requires the `cyclonedx-bom` CLI (`pip install cyclonedx-bom`).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCOPE="environment"
PYTHON_PATH=""
OUTPUT_PATH="artifacts/supply-chain/sbom.json"

usage() {
  cat <<'EOF'
Usage: scripts/security/generate_sbom.sh [--scope environment|tool-environment|install-surface] [--python PYTHON] [OUTPUT_PATH]

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
      if [[ $# -lt 2 || "${2:-}" == --* ]]; then
        echo "ERROR: --scope requires a value." >&2
        exit 2
      fi
      SCOPE="${2:-}"
      shift 2
      ;;
    --python)
      if [[ $# -lt 2 || "${2:-}" == --* ]]; then
        echo "ERROR: --python requires a value." >&2
        exit 2
      fi
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

case "${SCOPE}" in
  environment|tool-environment|install-surface) ;;
  *)
    echo "ERROR: --scope must be environment, tool-environment, or install-surface." >&2
    exit 2
    ;;
esac

if [[ "${SCOPE}" == "install-surface" && -z "${PYTHON_PATH}" ]]; then
  echo "ERROR: --python is required when --scope install-surface is used." >&2
  exit 2
fi

if [[ -z "${PYTHON_PATH}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_PATH="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_PATH="$(command -v python)"
  else
    echo "ERROR: no Python interpreter found on PATH." >&2
    exit 127
  fi
elif [[ "${PYTHON_PATH}" != */* ]]; then
  REQUESTED_PYTHON="${PYTHON_PATH}"
  if ! PYTHON_PATH="$(command -v "${REQUESTED_PYTHON}")"; then
    echo "ERROR: Python interpreter not found on PATH: ${REQUESTED_PYTHON}" >&2
    exit 1
  fi
elif [[ ! -e "${PYTHON_PATH}" ]]; then
  echo "ERROR: Python interpreter or environment not found: ${PYTHON_PATH}" >&2
  exit 1
fi

if [[ "${OUTPUT_PATH}" != /* ]]; then
  OUTPUT_PATH="${REPO_ROOT}/${OUTPUT_PATH}"
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
