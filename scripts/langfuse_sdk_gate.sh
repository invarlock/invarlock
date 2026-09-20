#!/usr/bin/env bash
# Qualify real SDK exports and separate wheel-only recipients without API calls.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LANGFUSE_PYTHON="${PYTHON:-python3}"
LANGFUSE_TAG="$("${LANGFUSE_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
LANGFUSE_LOCK="${ROOT_DIR}/requirements/workflows/langfuse-sdk-tests-py${LANGFUSE_TAG}.txt"
RECIPIENT_LOCK="${ROOT_DIR}/requirements/workflows/release-install-py${LANGFUSE_TAG}.txt"
test -f "${LANGFUSE_LOCK}"
test -f "${RECIPIENT_LOCK}"
LANGFUSE_ENV="$(mktemp -d "${TMPDIR:-/tmp}/invarlock-langfuse-sdk.XXXXXX")"
trap 'rm -rf "${LANGFUSE_ENV}"' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1
shopt -s nullglob
core_wheels=("${ROOT_DIR}"/dist/invarlock-*.whl)
if [[ ${#core_wheels[@]} -ne 1 ]]; then
  echo "Expected exactly one built core wheel" >&2
  exit 2
fi
for environment in sdk recipient; do
  "${LANGFUSE_PYTHON}" -m venv "${LANGFUSE_ENV}/${environment}"
  LANGFUSE_BIN="${LANGFUSE_ENV}/${environment}/bin/python"
  "${LANGFUSE_BIN}" -m pip install --require-hashes -r "${ROOT_DIR}/requirements/workflows/pip-bootstrap.txt"
  if [[ "${environment}" == sdk ]]; then
    "${LANGFUSE_BIN}" -m pip install --require-hashes -r "${LANGFUSE_LOCK}"
  else
    "${LANGFUSE_BIN}" -m pip install --require-hashes -r "${RECIPIENT_LOCK}"
  fi
  "${LANGFUSE_BIN}" -m pip install --no-index "${core_wheels[0]}"
  "${LANGFUSE_BIN}" -m pip check
  "${LANGFUSE_BIN}" -I -c 'from pathlib import Path; from sysconfig import get_path; import invarlock; assert Path(invarlock.__file__).resolve().is_relative_to(Path(get_path("purelib")).resolve())'
done
"${LANGFUSE_ENV}/recipient/bin/python" -I -c 'import importlib.util; assert importlib.util.find_spec("langfuse") is None'
cd "${LANGFUSE_ENV}"
INVARLOCK_REQUIRE_LANGFUSE_SDK=1 \
INVARLOCK_LANGFUSE_WHEEL_PYTHON="${LANGFUSE_ENV}/recipient/bin/python" \
  "${LANGFUSE_ENV}/sdk/bin/python" -I -m pytest -q -p no:cacheprovider \
  "${ROOT_DIR}/tests/examples/test_langfuse_export.py" \
  "${ROOT_DIR}/tests/integration/test_langfuse_sdk.py"
