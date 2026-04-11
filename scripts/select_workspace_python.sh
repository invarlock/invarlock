#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

repo_venv_python="$REPO_ROOT/.venv/bin/python"

if [[ -x "$repo_venv_python" ]] && "$repo_venv_python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' >/dev/null 2>&1; then
  printf '%s\n' "$repo_venv_python"
  exit 0
fi

bash "$REPO_ROOT/scripts/select_python.sh"
