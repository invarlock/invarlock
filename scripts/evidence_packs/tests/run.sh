#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
HELPERS_SH="${SCRIPT_DIR}/helpers.sh"

FILTER_REGEX=""
DO_BRANCH_COVERAGE="false"
DO_LINE_COVERAGE="false"
TEST_JOBS="${EVIDENCE_PACK_TEST_JOBS:-1}"
COVERAGE_DIR=""
COVERAGE_RAW_HITS=""
COVERAGE_RAW_PARTS_DIR=""

usage() {
    cat <<'EOF'
Usage: scripts/evidence_packs/tests/run.sh [--filter REGEX] [--coverage] [--line-coverage] [--jobs N]

Options:
  --filter REGEX     Run only tests whose id matches REGEX (id: test_file::test_fn)
  --coverage         Run tests under xtrace and enforce 100% branch coverage for evidence pack bash scripts
  --line-coverage    Run tests under xtrace and enforce 100% executable-line coverage for evidence pack bash scripts
  --jobs N           Run up to N tests in parallel (default: EVIDENCE_PACK_TEST_JOBS or 1)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter)
            FILTER_REGEX="${2:-}"
            shift 2
            ;;
        --coverage)
            DO_BRANCH_COVERAGE="true"
            shift
            ;;
        --line-coverage)
            DO_LINE_COVERAGE="true"
            shift
            ;;
        --jobs)
            TEST_JOBS="${2:-}"
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

if ! [[ "${TEST_JOBS}" =~ ^[0-9]+$ ]] || [[ "${TEST_JOBS}" -lt 1 ]]; then
    echo "ERROR: --jobs must be a positive integer" >&2
    exit 2
fi

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    REAL_PYTHON3="${ROOT_DIR}/.venv/bin/python"
else
    REAL_PYTHON3="$(command -v python3 2>/dev/null || true)"
fi


source "${SCRIPT_DIR}/runner_coverage.sh"
source "${SCRIPT_DIR}/runner_core.sh"

main() {
    if [[ "${DO_BRANCH_COVERAGE}" == "true" || "${DO_LINE_COVERAGE}" == "true" ]]; then
        COVERAGE_DIR="${SCRIPT_DIR}/.coverage"
        rm -rf "${COVERAGE_DIR}"
        mkdir -p "${COVERAGE_DIR}"
        COVERAGE_RAW_HITS="${COVERAGE_DIR}/executed.raw.tsv"
        COVERAGE_RAW_PARTS_DIR="${COVERAGE_DIR}/raw_hits"
        mkdir -p "${COVERAGE_RAW_PARTS_DIR}"
        : >"${COVERAGE_RAW_HITS}"
    fi

    local failures=0
    if [[ "${TEST_JOBS}" -gt 1 ]]; then
        run_selected_tests_parallel || failures=$?
    else
        run_selected_tests_serial || failures=$?
    fi

    if [[ ${failures} -gt 0 ]]; then
        echo "${failures} test(s) failed" >&2
        exit 1
    fi

    if [[ "${DO_BRANCH_COVERAGE}" == "true" ]]; then
        coverage_check
    fi
    if [[ "${DO_LINE_COVERAGE}" == "true" ]]; then
        line_coverage_check
    fi
}

main
