#!/usr/bin/env bash

list_test_files() {
    find "${SCRIPT_DIR}" -maxdepth 1 -type f -name 'test_*.sh' -print | sort
}

list_tests_in_file() {
    local file="$1"
    bash -c '
set -euo pipefail
export TEST_ROOT="$1"
export TEST_TMPDIR="$(mktemp -d 2>/dev/null || mktemp -d -t invarlock_bash_tests.XXXXXXXX)"
trap "rm -rf \"${TEST_TMPDIR}\"" EXIT
source "$2"
source "$3"
declare -F | awk "{print \$3}" | grep "^test_" || true
' -- "${ROOT_DIR}" "${HELPERS_SH}" "${file}"
}

run_one_test() {
    local file="$1"
    local fn="$2"
    local id
    id="$(basename "${file}")::${fn}"

    if [[ -n "${FILTER_REGEX}" ]]; then
        if ! [[ "${id}" =~ ${FILTER_REGEX} ]]; then
            return 0
        fi
    fi

    local tmp_dir
    tmp_dir="$(mktemp -d 2>/dev/null || mktemp -d -t invarlock_bash_tests.XXXXXXXX)"

    local out_file err_file trace_file
    out_file="$(mktemp "${tmp_dir}/stdout.XXXXXX")"
    err_file="$(mktemp "${tmp_dir}/stderr.XXXXXX")"
    trace_file="${tmp_dir}/xtrace.log"
    local rc=0

	if [[ "${DO_BRANCH_COVERAGE}" == "true" || "${DO_LINE_COVERAGE}" == "true" ]]; then
	    bash -c '
	set -euo pipefail
	cd "$1"
export TEST_ROOT="."
export TEST_TMPDIR="$2"
export TEST_REAL_PYTHON3="$3"
source "$4"
export PATH="$(mock_install_bin_dir):$PATH"
source "$5"
# Keep xtrace prefixes short to avoid bash 3.2 truncation on long absolute paths.
# The BASH_SOURCE guard keeps nounset safe for top-level test-function traces.
export COVERAGE_SOURCE_ROOT="$1/"
export PS4="__XTRACE__:\${BASH_SOURCE[0]:+\${BASH_SOURCE[0]#\${COVERAGE_SOURCE_ROOT}}}:\${LINENO}: "
	set -x
	"$6"
	        ' -- "${ROOT_DIR}" "${tmp_dir}" "${REAL_PYTHON3}" "${HELPERS_SH}" "${file}" "${fn}" >"${out_file}" 2>"${err_file}" </dev/null
	    rc=$?
	    if [[ ${rc} -eq 0 ]]; then
            local safe_id trace_copy test_raw_hits previous_raw_hits
            safe_id="$(echo "${id}" | tr -c 'A-Za-z0-9._-' '_')"
            trace_copy="${COVERAGE_DIR}/trace_${safe_id}.log"
            grep -E '^_+XTRACE__:' "${err_file}" >"${trace_copy}" 2>/dev/null || true
            test_raw_hits="${COVERAGE_RAW_PARTS_DIR}/${safe_id}.tsv"
            : >"${test_raw_hits}"
            previous_raw_hits="${COVERAGE_RAW_HITS}"
            COVERAGE_RAW_HITS="${test_raw_hits}"
            coverage_append_trace_hits "${err_file}"
            coverage_append_trace_hits_from_logs "${tmp_dir}"
            COVERAGE_RAW_HITS="${previous_raw_hits}"
            rm -rf "${tmp_dir}"
            echo "ok  ${id}"
            return 0
        fi
	else
	    bash -c '
	set -euo pipefail
	cd "$1"
export TEST_ROOT="."
export TEST_TMPDIR="$2"
export TEST_REAL_PYTHON3="$3"
	source "$4"
export PATH="$(mock_install_bin_dir):$PATH"
	source "$5"
	"$6"
	        ' -- "${ROOT_DIR}" "${tmp_dir}" "${REAL_PYTHON3}" "${HELPERS_SH}" "${file}" "${fn}" >"${out_file}" 2>"${err_file}" </dev/null
	    rc=$?
	    if [[ ${rc} -eq 0 ]]; then
            rm -rf "${tmp_dir}"
            echo "ok  ${id}"
            return 0
        fi
    fi

    echo "not ok  ${id} (rc=${rc})" >&2
    sed 's/^/  | /' "${out_file}" >&2 || true
    sed 's/^/  | /' "${err_file}" >&2 || true
    echo "tmpdir: ${tmp_dir}" >&2
    return 1
}

test_id_selected() {
    local file="$1"
    local fn="$2"
    local id
    id="$(basename "${file}")::${fn}"

    if [[ -n "${FILTER_REGEX}" ]]; then
        [[ "${id}" =~ ${FILTER_REGEX} ]] || return 1
    fi
    return 0
}

active_background_jobs() {
    jobs -pr | wc -l | tr -d ' '
}

run_selected_tests_serial() {
    local failures=0
    local file
    while IFS= read -r file; do
        local fn
        while IFS= read -r fn; do
            [[ -n "${fn}" ]] || continue
            test_id_selected "${file}" "${fn}" || continue
            run_one_test "${file}" "${fn}" || failures=$((failures + 1))
        done < <(list_tests_in_file "${file}")
    done < <(list_test_files)
    return "${failures}"
}

run_selected_tests_parallel() {
    local failures=0
    local -a pids=()
    local file
    while IFS= read -r file; do
        local fn
        while IFS= read -r fn; do
            [[ -n "${fn}" ]] || continue
            test_id_selected "${file}" "${fn}" || continue
            while [[ "$(active_background_jobs)" -ge "${TEST_JOBS}" ]]; do
                sleep 0.1
            done
            run_one_test "${file}" "${fn}" &
            pids+=("$!")
        done < <(list_tests_in_file "${file}")
    done < <(list_test_files)

    local pid
    for pid in "${pids[@]}"; do
        wait "${pid}" || failures=$((failures + 1))
    done
    return "${failures}"
}
