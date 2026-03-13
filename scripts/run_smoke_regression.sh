#!/usr/bin/env bash
set -euo pipefail

SKIP_RUFF="${SKIP_RUFF:-0}"
PYTHON_BIN="${INVARLOCK_PYTHON:-}"

if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH=src
else
  export PYTHONPATH="src:${PYTHONPATH}"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(bash scripts/select_python.sh)"
fi

"${PYTHON_BIN}" -m pytest -q tests/cli/test_cli_smoke.py tests/cli/test_app_version.py tests/cli/test_verify_json_shape.py
"${PYTHON_BIN}" -m pytest -q tests/reporting/test_report_pm_only.py tests/core/test_default_providers.py
"${PYTHON_BIN}" -m pytest -q tests/guards_property/test_variance_properties.py
"${PYTHON_BIN}" -m pytest -q tests/integration/test_end_to_end_evaluate.py
if [[ "${SKIP_RUFF}" != "1" ]]; then
  "${PYTHON_BIN}" -m ruff check src tests scripts
fi
