#!/usr/bin/env bash
# Qualify the optional SDK using installed release wheels and no provider network.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JUDGE_PYTHON="${PYTHON:-python3}"
JUDGE_TAG="$("${JUDGE_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
JUDGE_LOCK="${ROOT_DIR}/requirements/workflows/inspect-judge-tests-py${JUDGE_TAG}.txt"
test -f "${JUDGE_LOCK}"
JUDGE_ENV="$(mktemp -d "${TMPDIR:-/tmp}/invarlock-judge-sdk.XXXXXX")"
trap 'rm -rf "${JUDGE_ENV}"' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1
"${JUDGE_PYTHON}" -m venv "${JUDGE_ENV}/venv"
JUDGE_BIN="${JUDGE_ENV}/venv/bin/python"
"${JUDGE_BIN}" -m pip install --require-hashes -r "${ROOT_DIR}/requirements/workflows/pip-bootstrap.txt"
"${JUDGE_BIN}" -m pip install --require-hashes -r "${JUDGE_LOCK}"
shopt -s nullglob
core_wheels=("${ROOT_DIR}"/dist/invarlock-*.whl)
judge_wheels=("${ROOT_DIR}"/dist/addins/invarlock_inspect_judge-*.whl)
if [[ ${#core_wheels[@]} -ne 1 || ${#judge_wheels[@]} -ne 1 ]]; then
  echo "Expected exactly one built core wheel and Inspect judge wheel" >&2
  exit 2
fi
# Resolve the real extra against the installed hashed closure. --no-index makes
# an omitted or incompatible dependency fail rather than silently fetching it.
"${JUDGE_BIN}" -m pip install --no-index "${core_wheels[0]}" "${judge_wheels[0]}[inspect]"
"${JUDGE_BIN}" -m pip check
"${JUDGE_BIN}" -c 'from pathlib import Path; from sysconfig import get_path; import invarlock, invarlock_addins.inspect_judge as judge; root=Path(get_path("purelib")).resolve(); assert all(Path(module.__file__).resolve().is_relative_to(root) for module in (invarlock, judge))'
mkdir "${JUDGE_ENV}/tests"
cp "${ROOT_DIR}/addins/inspect_judge/tests/test_live_inspect_sdk.py" "${JUDGE_ENV}/tests/"
cp -R "${ROOT_DIR}/addins/inspect_judge/tests/fixtures" "${JUDGE_ENV}/tests/fixtures"
cd "${JUDGE_ENV}"
INVARLOCK_REQUIRE_INSPECT_SDK=1 "${JUDGE_BIN}" -m pytest -q tests/test_live_inspect_sdk.py
