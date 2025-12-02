#!/usr/bin/env bash
set -euo pipefail

SKIP_RUFF="${SKIP_RUFF:-0}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate invarlock >/dev/null 2>&1 || true
fi

pytest -q tests/cli/test_cli_smoke.py tests/cli/test_app_version.py tests/cli/test_verify_json_shape.py
pytest -q tests/reporting/test_certificate_pm_only.py tests/core/test_default_providers.py
pytest -q tests/guards_property/test_variance_properties.py
pytest -q tests/integration/test_end_to_end_cert.py
if [[ "${SKIP_RUFF}" != "1" ]]; then
  python -m ruff check src tests scripts
fi
