#!/usr/bin/env bash

t_fail() {
    echo "FAIL: $*" >&2
    return 1
}

assert_eq() {
    local expected="$1"
    local actual="$2"
    local msg="${3:-}"
    if [[ "${expected}" != "${actual}" ]]; then
        t_fail "${msg} expected='${expected}' actual='${actual}'"
    fi
}

assert_ne() {
    local not_expected="$1"
    local actual="$2"
    local msg="${3:-}"
    if [[ "${not_expected}" == "${actual}" ]]; then
        t_fail "${msg} not_expected='${not_expected}' actual='${actual}'"
    fi
}

assert_match() {
    local pattern="$1"
    local actual="$2"
    local msg="${3:-}"
    if ! [[ "${actual}" =~ ${pattern} ]]; then
        t_fail "${msg} pattern='${pattern}' actual='${actual}'"
    fi
}

assert_file_exists() {
    local path="$1"
    local msg="${2:-}"
    [[ -f "${path}" ]] || t_fail "${msg} missing_file='${path}'"
}

assert_dir_exists() {
    local path="$1"
    local msg="${2:-}"
    [[ -d "${path}" ]] || t_fail "${msg} missing_dir='${path}'"
}

assert_rc() {
    local expected="$1"
    local actual="$2"
    local msg="${3:-}"
    if [[ "${expected}" != "${actual}" ]]; then
        t_fail "${msg} expected_rc='${expected}' actual_rc='${actual}'"
    fi
}

run() {
    local out_file err_file
    out_file="$(mktemp "${TEST_TMPDIR}/run.out.XXXXXX")"
    err_file="$(mktemp "${TEST_TMPDIR}/run.err.XXXXXX")"
    RUN_RC=0
    if "$@" >"${out_file}" 2>"${err_file}"; then
        RUN_RC=0
    else
        RUN_RC=$?
    fi
    RUN_OUT="$(cat "${out_file}")"
    RUN_ERR="$(cat "${err_file}")"
    case "$-" in
        *x*)
            # When xtrace is enabled (coverage runs), forward captured stderr so the
            # test runner can attribute executed lines inside "$@" to the right files.
            cat "${err_file}" >&2 || true
            ;;
    esac
    rm -f "${out_file}" "${err_file}"
    return 0
}

fixture_write() {
    local rel_path="$1"
    local content="$2"
    local path="${TEST_TMPDIR}/fixtures/${rel_path}"
    mkdir -p "$(dirname "${path}")"
    printf "%s" "${content}" > "${path}"
}

fixture_append() {
    local rel_path="$1"
    local content="$2"
    local path="${TEST_TMPDIR}/fixtures/${rel_path}"
    mkdir -p "$(dirname "${path}")"
    printf "%s" "${content}" >> "${path}"
}

mock_reset() {
    rm -rf "${TEST_TMPDIR}/fixtures"
    mkdir -p "${TEST_TMPDIR}/fixtures"
}

mock_nvidia_smi_set_mem_free_mib() {
    local gpu_id="$1"
    local mib="$2"
    fixture_write "nvidia-smi/memory_free.${gpu_id}" "$(printf '%s\n' "${mib}")"
}

mock_nvidia_smi_set_mem_total_mib() {
    local gpu_id="$1"
    local mib="$2"
    fixture_write "nvidia-smi/memory_total.${gpu_id}" "$(printf '%s\n' "${mib}")"
}

mock_nvidia_smi_set_pids() {
    local gpu_id="$1"
    local pids_text="$2"
    fixture_write "nvidia-smi/compute_pids.${gpu_id}" "${pids_text}"
}

mock_df_set_output() {
    local text="$1"
    fixture_write "df.out" "${text}"
}

mock_ps_set_alive() {
    local pid="$1"
    fixture_append "ps/alive" "$(printf '%s\n' "${pid}")"
}

mock_ps_set_pgid() {
    local pid="$1"
    local pgid="$2"
    fixture_write "ps/pgid.${pid}" "$(printf '%s\n' "${pgid}")"
}
