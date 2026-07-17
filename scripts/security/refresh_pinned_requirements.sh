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
  --group GROUP        all or workflows (default: all).
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
  all|workflows) ;;
  *)
    echo "ERROR: --group must be one of: all, workflows" >&2
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

mkdir -p "${WORKFLOW_DIR}"

compile_pyproject() {
  local output="$1"
  local output_arg="$1"
  shift
  if [[ "${output}" == "${ROOT_DIR}/"* ]]; then
    output_arg="${output#${ROOT_DIR}/}"
  fi
  (
    cd "${ROOT_DIR}"
    uv pip compile pyproject.toml \
      --python-platform x86_64-unknown-linux-gnu \
      --generate-hashes \
      --output-file "${output_arg}" \
      "$@"
  )
}

compile_req_platform() {
  local input="$1"
  local output="$2"
  local input_arg="$1"
  local output_arg="$2"
  shift 2
  if [[ "${input}" == "${ROOT_DIR}/"* && "${output}" == "${ROOT_DIR}/"* ]]; then
    input_arg="${input#${ROOT_DIR}/}"
    output_arg="${output#${ROOT_DIR}/}"
  fi
  (
    cd "${ROOT_DIR}"
    uv pip compile "${input_arg}" \
      --generate-hashes \
      --output-file "${output_arg}" \
      "$@"
  )
}

compile_release_install() {
  local output="$1"
  local python_version="$2"
  local python_tag="${python_version/./}"
  local output_arg="$1"
  if [[ "${output}" == "${ROOT_DIR}/"* ]]; then
    output_arg="${output#${ROOT_DIR}/}"
  fi
  (
    cd "${ROOT_DIR}"
    uv pip compile \
      requirements/workflows/release-install.in \
      --python-platform x86_64-unknown-linux-gnu \
      --python-version "${python_version}" \
      --constraints requirements/workflows/release-security-py313.txt \
      --constraints "requirements/workflows/ci-hf-py${python_tag}.txt" \
      --generate-hashes \
      --custom-compile-command "scripts/security/refresh_pinned_requirements.sh --write --group workflows" \
      --output-file "${output_arg}"
  )
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

  compile_pyproject "${WORKFLOW_DIR}/docs-ci-py313.txt" \
    --python-version 3.13 \
    --extra ci \
    --extra docs-ci

  compile_pyproject "${WORKFLOW_DIR}/hf-py313.txt" \
    --python-version 3.13 \
    --extra hf

  compile_req_platform \
    "${WORKFLOW_DIR}/runtime-image.in" \
    "${WORKFLOW_DIR}/runtime-image-py312.txt" \
    --python-version 3.12 \
    --python-platform x86_64-unknown-linux-gnu \
    --torch-backend cpu

  compile_req_platform \
    "${WORKFLOW_DIR}/runtime-image.in" \
    "${WORKFLOW_DIR}/runtime-image-py312-aarch64.txt" \
    --python-version 3.12 \
    --python-platform aarch64-unknown-linux-gnu \
    --torch-backend cpu

  compile_req_platform \
    "${WORKFLOW_DIR}/runtime-image.in" \
    "${WORKFLOW_DIR}/runtime-image-py312-cu128.txt" \
    --python-version 3.12 \
    --python-platform x86_64-unknown-linux-gnu \
    --torch-backend cu128

  compile_req_platform \
    "${WORKFLOW_DIR}/multimodal-runtime.in" \
    "${WORKFLOW_DIR}/multimodal-runtime-py312.txt" \
    --python-version 3.12 \
    --python-platform x86_64-unknown-linux-gnu \
    --torch-backend cu128 \
    --no-deps

  compile_pyproject "${WORKFLOW_DIR}/precommit-ci-py313.txt" \
    --python-version 3.13 \
    --extra precommit-ci

  compile_pyproject "${WORKFLOW_DIR}/release-security-py313.txt" \
    --python-version 3.13 \
    --extra release-ci \
    --extra security-ci

  compile_release_install "${WORKFLOW_DIR}/release-install-py312.txt" 3.12
  compile_release_install "${WORKFLOW_DIR}/release-install-py313.txt" 3.13

  compile_pyproject "${WORKFLOW_DIR}/security-ci-py313.txt" \
    --python-version 3.13 \
    --extra security-ci
}

case "${GROUP}" in
  all)
    run_workflow_locks
    ;;
  workflows)
    run_workflow_locks
    ;;
esac

if [[ "${MODE}" == "check" ]]; then
  echo "Requirement lock compile check completed without modifying requirements/."
fi
