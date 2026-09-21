#!/usr/bin/env bash
# Replay all native export profiles through isolated installed core processes.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARITY_PYTHON="${PYTHON:-python3}"
PARITY_PYTHON="$("${PARITY_PYTHON}" -c 'import sys; print(sys.executable)')"
PARITY_TAG="$("${PARITY_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
PARITY_LOCK="${ROOT_DIR}/requirements/workflows/release-install-py${PARITY_TAG}.txt"
test -f "${PARITY_LOCK}"
PARITY_TEMP="$(mktemp -d "${TMPDIR:-/tmp}/invarlock-evaluator-parity.XXXXXX")"
trap 'rm -rf "${PARITY_TEMP}"' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1
shopt -s nullglob
parity_wheels=("${ROOT_DIR}"/dist/invarlock-*.whl)
if [[ ${#parity_wheels[@]} -ne 1 ]]; then
  echo "Expected exactly one built core wheel" >&2
  exit 2
fi
"${PARITY_PYTHON}" -m venv "${PARITY_TEMP}/recipient"
PARITY_RECIPIENT="${PARITY_TEMP}/recipient/bin/python"
"${PARITY_RECIPIENT}" -m pip install --require-hashes -r "${ROOT_DIR}/requirements/workflows/pip-bootstrap.txt"
"${PARITY_RECIPIENT}" -m pip install --require-hashes -r "${PARITY_LOCK}"
"${PARITY_RECIPIENT}" -m pip install --no-index "${parity_wheels[0]}"
"${PARITY_RECIPIENT}" -m pip check
cd "${PARITY_TEMP}"
"${PARITY_RECIPIENT}" -I \
  "${ROOT_DIR}/examples/integrations/evaluator-parity/real_judge.py" \
  --recipient-python "${PARITY_RECIPIENT}" \
  --output "${PARITY_TEMP}/retained-judge"
INVARLOCK_EVALUATOR_PARITY_PYTHON="${PARITY_RECIPIENT}" \
  "${PARITY_PYTHON}" -m pytest -q \
  "${ROOT_DIR}/tests/integration/test_evaluator_parity.py" \
  -k test_installed_sdk_free_recipient_signed_journey -n "${PARITY_WORKERS:-4}" \
  --override-ini=addopts= --basetemp "${PARITY_TEMP}/journeys"

# Replay every retained pack from the fresh evaluator campaigns through the
# same isolated candidate recipient. These commands are offline and preserve
# the campaigns' original passing, rejected and insufficient-evidence results.
SENTINEL="${ROOT_DIR}/examples/integrations/evaluator-live/references/mistral-7b-sentinel"
PRIORITY="${ROOT_DIR}/examples/integrations/evaluator-live/references/priority-workflows"
"${PARITY_RECIPIENT}" -I "${SENTINEL}/replay.py"
"${PARITY_RECIPIENT}" -I "${SENTINEL}/judge_replay.py" \
  --output "${PARITY_TEMP}/sentinel-judge"
"${PARITY_RECIPIENT}" -I "${PRIORITY}/replay.py"
"${PARITY_RECIPIENT}" -I "${PRIORITY}/judge_replay.py" \
  --output "${PARITY_TEMP}/priority-judge"
