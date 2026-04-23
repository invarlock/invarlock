#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

is_python_312() {
  local candidate="$1"
  "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' >/dev/null 2>&1
}

is_python_312_plus() {
  local candidate="$1"
  "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' >/dev/null 2>&1
}

supports_required_modules() {
  local candidate="$1"
  if [[ -z "${INVARLOCK_SELECT_PYTHON_REQUIRE_MODULES:-}" ]]; then
    return 0
  fi
  "$candidate" -c 'from importlib import metadata as md, util; import os; modules=[m for m in os.environ.get("INVARLOCK_SELECT_PYTHON_REQUIRE_MODULES", "").split(",") if m];
def _has_module(name):
    try:
        md.version(name)
        return True
    except md.PackageNotFoundError:
        return util.find_spec(name) is not None
raise SystemExit(0 if all(_has_module(name) for name in modules) else 1)' >/dev/null 2>&1
}

resolve_command() {
  local candidate="$1"
  if [[ -x "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  command -v "$candidate" 2>/dev/null || return 1
}

prefer_active_python() {
  [[ "${GITHUB_ACTIONS:-}" == "true" ]] || [[ -n "${VIRTUAL_ENV:-}" ]] || [[ -n "${CONDA_PREFIX:-}" ]]
}

print_if_version_matches() {
  local candidate resolved
  candidate="$1"
  resolved="$(resolve_command "$candidate")" || return 1
  if is_python_312 "$resolved" && supports_required_modules "$resolved"; then
    printf '%s\n' "$resolved"
    exit 0
  fi
  return 1
}

print_if_supported() {
  local candidate resolved
  candidate="$1"
  resolved="$(resolve_command "$candidate")" || return 1
  if is_python_312_plus "$resolved" && supports_required_modules "$resolved"; then
    printf '%s\n' "$resolved"
    exit 0
  fi
  return 1
}

repo_venv_python="$REPO_ROOT/.venv/bin/python"
if [[ -x "$repo_venv_python" ]]; then
  print_if_supported "$repo_venv_python" || true
fi

if prefer_active_python && command -v python >/dev/null 2>&1; then
  print_if_supported "python" || true
fi

if [[ -n "${HOME:-}" ]]; then
  print_if_version_matches "$HOME/anaconda3/envs/invarlock-py312/bin/python" || true
  print_if_version_matches "$HOME/miniconda3/envs/invarlock-py312/bin/python" || true
  print_if_version_matches "$HOME/miniforge3/envs/invarlock-py312/bin/python" || true
  print_if_version_matches "$HOME/mambaforge/envs/invarlock-py312/bin/python" || true
fi

if command -v conda >/dev/null 2>&1; then
  conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$conda_base" ]]; then
    print_if_version_matches "$conda_base/envs/invarlock-py312/bin/python" || true
  fi
fi

if command -v python >/dev/null 2>&1; then
  print_if_version_matches "python" || true
fi

print_if_version_matches "python3.12" || true

if command -v python >/dev/null 2>&1; then
  print_if_supported "python" || true
fi

print_if_supported "python3.13" || true
print_if_supported "python3" || true

if command -v python >/dev/null 2>&1; then
  command -v python
  exit 0
fi

printf '%s\n' "python"
