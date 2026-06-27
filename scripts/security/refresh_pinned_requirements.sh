#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_REQ_DIR="${ROOT_DIR}/requirements"
MODE="write"
GROUP="all"

usage() {
  cat <<'EOF'
Usage: scripts/security/refresh_pinned_requirements.sh [options]

Options:
  --write              Rewrite checked-in requirement locks (default).
  --check              Compile into a temporary copy without modifying requirements/.
  --group GROUP        all, workflows, or evidence-packs (default: all).
  --help, -h           Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --write)
      MODE="write"
      shift
      ;;
    --check)
      MODE="check"
      shift
      ;;
    --group)
      if [[ $# -lt 2 || "${2:-}" == --* ]]; then
        echo "ERROR: --group requires a value." >&2
        exit 2
      fi
      GROUP="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${GROUP}" in
  all|workflows|evidence-packs) ;;
  *)
    echo "ERROR: --group must be one of: all, workflows, evidence-packs" >&2
    exit 2
    ;;
esac

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required to compile pinned requirements." >&2
  exit 127
fi

REQ_DIR="${SOURCE_REQ_DIR}"
TMP_REQ_DIR=""
if [[ "${MODE}" == "check" ]]; then
  TMP_REQ_DIR="$(mktemp -d "${TMPDIR:-/tmp}/invarlock-req-check.XXXXXX")"
  trap 'rm -rf "${TMP_REQ_DIR}"' EXIT
  cp -R "${SOURCE_REQ_DIR}/." "${TMP_REQ_DIR}/"
  REQ_DIR="${TMP_REQ_DIR}"
fi

WORKFLOW_DIR="${REQ_DIR}/workflows"
EVIDENCE_PACK_DIR="${REQ_DIR}/evidence-packs"

mkdir -p "${WORKFLOW_DIR}" "${EVIDENCE_PACK_DIR}"

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

run_workflow_locks() {
  compile_pyproject "${WORKFLOW_DIR}/ci-hf-py312.txt" \
    --python-version 3.12 \
    --extra hf \
    --extra ci

  compile_pyproject "${WORKFLOW_DIR}/core-py312.txt" \
    --python-version 3.12

  compile_pyproject "${WORKFLOW_DIR}/ci-hf-py313.txt" \
    --python-version 3.13 \
    --extra hf \
    --extra ci

  compile_pyproject "${WORKFLOW_DIR}/core-py313.txt" \
    --python-version 3.13

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
    "${WORKFLOW_DIR}/runtime-image-quant.in" \
    "${WORKFLOW_DIR}/runtime-image-quant-py312-cu128.txt" \
    --python-version 3.12 \
    --python-platform x86_64-unknown-linux-gnu \
    --torch-backend cu128

  compile_req_platform \
    "${WORKFLOW_DIR}/runtime-image.in" \
    "${WORKFLOW_DIR}/runtime-image-py312-aarch64.txt" \
    --python-version 3.12 \
    --python-platform aarch64-unknown-linux-gnu \
    --torch-backend cpu

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
}

run_evidence_pack_locks() {
  compile_req_in \
    "${EVIDENCE_PACK_DIR}/accelerate.in" \
    "${EVIDENCE_PACK_DIR}/accelerate.txt"

  compile_req_in \
    "${EVIDENCE_PACK_DIR}/cuda-nvcc.in" \
    "${EVIDENCE_PACK_DIR}/cuda-nvcc.txt" \
    --no-deps

  compile_req_in \
    "${EVIDENCE_PACK_DIR}/flash-attn.in" \
    "${EVIDENCE_PACK_DIR}/flash-attn.txt" \
    --no-deps

  compile_req_in \
    "${EVIDENCE_PACK_DIR}/huggingface_hub.in" \
    "${EVIDENCE_PACK_DIR}/huggingface_hub.txt"

  compile_req_in \
    "${EVIDENCE_PACK_DIR}/protobuf.in" \
    "${EVIDENCE_PACK_DIR}/protobuf.txt"

  compile_req_in \
    "${EVIDENCE_PACK_DIR}/pyyaml.in" \
    "${EVIDENCE_PACK_DIR}/pyyaml.txt"

  compile_req_in \
    "${EVIDENCE_PACK_DIR}/sentencepiece.in" \
    "${EVIDENCE_PACK_DIR}/sentencepiece.txt"
}

case "${GROUP}" in
  all)
    run_workflow_locks
    run_evidence_pack_locks
    ;;
  workflows)
    run_workflow_locks
    ;;
  evidence-packs)
    run_evidence_pack_locks
    ;;
esac

if [[ "${MODE}" == "check" ]]; then
  echo "Requirement lock compile check completed without modifying requirements/."
fi
