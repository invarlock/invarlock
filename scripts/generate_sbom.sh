#!/usr/bin/env bash
#
# Generate a CycloneDX SBOM for the current environment.
# Requires the `cyclonedx-bom` CLI (`pip install cyclonedx-bom`).

set -euo pipefail

OUTPUT_PATH="${1:-artifacts/supply-chain/sbom.json}"
OUTPUT_DIR="$(dirname "${OUTPUT_PATH}")"

if ! command -v cyclonedx-py >/dev/null 2>&1; then
  echo "ERROR: cyclonedx CLI not found. Install it with 'pip install cyclonedx-bom'." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

cyclonedx-py environment \
  --spec-version 1.4 \
  --output-format JSON \
  --output-file "${OUTPUT_PATH}"

echo "SBOM written to ${OUTPUT_PATH}"
