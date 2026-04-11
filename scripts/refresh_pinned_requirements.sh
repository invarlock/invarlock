#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_DIR="${ROOT_DIR}/requirements"
WORKFLOW_DIR="${REQ_DIR}/workflows"
PROOF_PACK_DIR="${REQ_DIR}/proof-packs"

mkdir -p "${WORKFLOW_DIR}" "${PROOF_PACK_DIR}"

compile_pyproject() {
  local output="$1"
  shift
  uv pip compile "${ROOT_DIR}/pyproject.toml" \
    --python-platform x86_64-unknown-linux-gnu \
    --generate-hashes \
    --output-file "${output}" \
    "$@"
}

compile_req_in() {
  local input="$1"
  local output="$2"
  shift 2
  uv pip compile "${input}" \
    --universal \
    --generate-hashes \
    --output-file "${output}" \
    "$@"
}

compile_req_platform() {
  local input="$1"
  local output="$2"
  shift 2
  uv pip compile "${input}" \
    --generate-hashes \
    --output-file "${output}" \
    "$@"
}

compile_pyproject "${WORKFLOW_DIR}/ci-hf-py312.txt" \
  --python-version 3.12 \
  --extra hf \
  --extra ci

compile_pyproject "${WORKFLOW_DIR}/ci-hf-py313.txt" \
  --python-version 3.13 \
  --extra hf \
  --extra ci

compile_pyproject "${WORKFLOW_DIR}/docs-ci-py313.txt" \
  --python-version 3.13 \
  --extra ci \
  --extra docs-ci

compile_pyproject "${WORKFLOW_DIR}/assurance-ci-py313.txt" \
  --python-version 3.13 \
  --extra hf \
  --extra ci \
  --extra docs-ci

compile_pyproject "${WORKFLOW_DIR}/hf-py313.txt" \
  --python-version 3.13 \
  --extra hf

compile_pyproject "${WORKFLOW_DIR}/advanced-py313.txt" \
  --python-version 3.13 \
  --extra advanced

compile_req_platform \
  "${WORKFLOW_DIR}/runtime-image.in" \
  "${WORKFLOW_DIR}/runtime-image-py312.txt" \
  --python-version 3.12 \
  --python-platform x86_64-unknown-linux-gnu \
  --torch-backend cpu

compile_req_platform \
  "${WORKFLOW_DIR}/runtime-image.in" \
  "${WORKFLOW_DIR}/runtime-image-py312-cu128.txt" \
  --python-version 3.12 \
  --python-platform x86_64-unknown-linux-gnu \
  --torch-backend cu128

compile_req_platform \
  "${WORKFLOW_DIR}/runtime-image.in" \
  "${WORKFLOW_DIR}/runtime-image-py312-aarch64.txt" \
  --python-version 3.12 \
  --python-platform aarch64-unknown-linux-gnu \
  --torch-backend cpu

compile_req_platform \
  "${WORKFLOW_DIR}/clusterfuzzlite.in" \
  "${WORKFLOW_DIR}/clusterfuzzlite-py311.txt" \
  --python-version 3.11 \
  --python-platform x86_64-unknown-linux-gnu

compile_pyproject "${WORKFLOW_DIR}/precommit-ci-py313.txt" \
  --python-version 3.13 \
  --extra precommit-ci

compile_pyproject "${WORKFLOW_DIR}/release-security-py313.txt" \
  --python-version 3.13 \
  --extra release-ci \
  --extra security-ci

compile_pyproject "${WORKFLOW_DIR}/security-ci-py313.txt" \
  --python-version 3.13 \
  --extra security-ci

compile_req_in \
  "${PROOF_PACK_DIR}/accelerate.in" \
  "${PROOF_PACK_DIR}/accelerate.txt"

compile_req_in \
  "${PROOF_PACK_DIR}/flash-attn.in" \
  "${PROOF_PACK_DIR}/flash-attn.txt" \
  --no-deps

compile_req_in \
  "${PROOF_PACK_DIR}/huggingface_hub.in" \
  "${PROOF_PACK_DIR}/huggingface_hub.txt"

compile_req_in \
  "${PROOF_PACK_DIR}/protobuf.in" \
  "${PROOF_PACK_DIR}/protobuf.txt"

compile_req_in \
  "${PROOF_PACK_DIR}/pyyaml.in" \
  "${PROOF_PACK_DIR}/pyyaml.txt"

compile_req_in \
  "${PROOF_PACK_DIR}/sentencepiece.in" \
  "${PROOF_PACK_DIR}/sentencepiece.txt"
