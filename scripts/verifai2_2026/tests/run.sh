#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

FILTER_REGEX=""

usage() {
    cat <<'EOF'
Usage: scripts/verifai2_2026/tests/run.sh [--filter REGEX]

Runs the isolated VerifAI-2 paper-tool test suite under coverage and enforces:
  - 100% line coverage
  - 100% branch coverage

This harness is intentionally separate from the repo's main `tests/` and
`make coverage-enforce` gates.

Options:
  --filter REGEX     Run only tests whose id matches REGEX (id: test_file::test_fn)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter)
            FILTER_REGEX="${2:-}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cd "${ROOT_DIR}"

RCFILE="${SCRIPT_DIR}/coveragerc"

pick_python() {
    local bin
    for bin in python python3; do
        if ! command -v "${bin}" >/dev/null 2>&1; then
            continue
        fi
        if "${bin}" -c 'import coverage, pytest' >/dev/null 2>&1; then
            command -v "${bin}"
            return 0
        fi
    done
    return 1
}

PYTHON_BIN="$(pick_python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "No suitable python found (needs importable: coverage, pytest)." >&2
    exit 2
fi

ARGS=(-q "scripts/verifai2_2026/tests")
if [[ -n "${FILTER_REGEX}" ]]; then
    ARGS=(-q -k "${FILTER_REGEX}" "scripts/verifai2_2026/tests")
fi

"${PYTHON_BIN}" -m coverage erase --rcfile "${RCFILE}"
"${PYTHON_BIN}" -m coverage run --rcfile "${RCFILE}" -m pytest "${ARGS[@]}"
"${PYTHON_BIN}" -m coverage report --rcfile "${RCFILE}"
