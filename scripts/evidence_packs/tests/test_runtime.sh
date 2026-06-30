#!/usr/bin/env bash

test_rand_jitter_ms_invalid_input_returns_zero() {
    mock_reset
    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    assert_eq "0" "$(_rand_jitter_ms "nope")" "non-numeric jitter clamps to 0"
    assert_eq "0" "$(_rand_jitter_ms "0")" "zero jitter clamps to 0"
    assert_eq "0" "$(_rand_jitter_ms "-1")" "negative jitter clamps to 0"
}

test_runtime_python_wrapper_invokes_repo_helper() {
    mock_reset
    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    local calls="${TEST_TMPDIR}/python.calls"
    : >"${calls}"
    _cmd_python() {
        printf '%s\n' "$*" > "${calls}"
        echo "ok"
        return 0
    }

    run _runtime_python runtime_tools.py now_iso_plus_seconds 10
    assert_rc "0" "${RUN_RC}" "_runtime_python returns success"
    assert_match "evidence_packs/.*/python/runtime_tools\\.py" "$(cat "${calls}")" "invokes python helper"
    assert_match "now_iso_plus_seconds 10" "$(cat "${calls}")" "forwards args"
    assert_eq "ok" "${RUN_OUT}" "forwards helper stdout"
}

test_pack_run_from_config_invokes_repo_python_entrypoint() {
    mock_reset
    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    local calls="${TEST_TMPDIR}/python.calls"
    : >"${calls}"
    _cmd_python() {
        printf '%s\n' "$*" > "${calls}"
        return 0
    }

    run _pack_run_from_config --config demo.yaml --out runs/demo
    assert_rc "0" "${RUN_RC}" "_pack_run_from_config returns success"
    assert_match "evidence_packs/.*/python/run_from_config\\.py" "$(cat "${calls}")" "invokes repo-only config runner"
    assert_match "--config demo\\.yaml --out runs/demo" "$(cat "${calls}")" "forwards config-run arguments"
}

test_cmd_python_honors_python_bin_override() {
    mock_reset
    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/python-override" <<'EOF'
#!/usr/bin/env bash
printf 'override %s\n' "$*"
EOF
    chmod +x "${bin_dir}/python-override"
    local had_python_bin="0"
    local previous_python_bin="${PYTHON_BIN:-}"
    if [[ "${PYTHON_BIN+x}" == "x" ]]; then
        had_python_bin="1"
    fi
    export PYTHON_BIN="${bin_dir}/python-override"

    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    run _cmd_python --version
    assert_rc "0" "${RUN_RC}" "_cmd_python uses PYTHON_BIN"
    assert_eq "override --version" "${RUN_OUT}" "PYTHON_BIN command receives arguments"

    if [[ "${had_python_bin}" == "1" ]]; then
        export PYTHON_BIN="${previous_python_bin}"
    else
        unset PYTHON_BIN
    fi
}

test_cmd_python_interpreter_ignores_wrappers_and_prefers_helper_python() {
    mock_reset
    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/not-python" <<'EOF'
#!/usr/bin/env bash
exit 9
EOF
    chmod +x "${bin_dir}/not-python"

    local previous_helper="${PACK_HELPER_PYTHON_BIN:-}"
    local previous_python_bin="${PYTHON_BIN:-}"
    local had_helper="0"
    local had_python_bin="0"
    [[ "${PACK_HELPER_PYTHON_BIN+x}" == "x" ]] && had_helper="1"
    [[ "${PYTHON_BIN+x}" == "x" ]] && had_python_bin="1"

    export PACK_HELPER_PYTHON_BIN="${bin_dir}/not-python -m invarlock"
    export PYTHON_BIN="${TEST_REAL_PYTHON3}"
    run _cmd_python_interpreter
    assert_rc "0" "${RUN_RC}" "interpreter resolver skips helper wrappers with arguments"
    assert_eq "${TEST_REAL_PYTHON3}" "${RUN_OUT}" "falls back to valid PYTHON_BIN interpreter"

    export PACK_HELPER_PYTHON_BIN="${TEST_REAL_PYTHON3}"
    export PYTHON_BIN="${bin_dir}/not-python"
    run _cmd_python_interpreter
    assert_rc "0" "${RUN_RC}" "interpreter resolver accepts explicit helper python"
    assert_eq "${TEST_REAL_PYTHON3}" "${RUN_OUT}" "PACK_HELPER_PYTHON_BIN wins when it is a real interpreter"

    if [[ "${had_helper}" == "1" ]]; then
        export PACK_HELPER_PYTHON_BIN="${previous_helper}"
    else
        unset PACK_HELPER_PYTHON_BIN
    fi
    if [[ "${had_python_bin}" == "1" ]]; then
        export PYTHON_BIN="${previous_python_bin}"
    else
        unset PYTHON_BIN
    fi
}

test_cmd_python_interpreter_discovers_python_and_fails_without_candidates() {
    mock_reset
    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    local previous_path="${PATH}"
    local previous_helper="${PACK_HELPER_PYTHON_BIN:-}"
    local previous_python_bin="${PYTHON_BIN:-}"
    local previous_virtual_env="${VIRTUAL_ENV:-}"
    local had_helper="0"
    local had_python_bin="0"
    local had_virtual_env="0"
    [[ "${PACK_HELPER_PYTHON_BIN+x}" == "x" ]] && had_helper="1"
    [[ "${PYTHON_BIN+x}" == "x" ]] && had_python_bin="1"
    [[ "${VIRTUAL_ENV+x}" == "x" ]] && had_virtual_env="1"

    unset PACK_HELPER_PYTHON_BIN
    unset PYTHON_BIN
    unset VIRTUAL_ENV
    run _cmd_python_interpreter
    assert_rc "0" "${RUN_RC}" "interpreter resolver discovers python on PATH"
    assert_match "python" "${RUN_OUT}" "discovered interpreter name contains python"

    local empty_path="${TEST_TMPDIR}/empty-path"
    mkdir -p "${empty_path}"
    local fail_rc=0
    export PATH="${empty_path}"
    export PACK_HELPER_PYTHON_BIN="${TEST_TMPDIR}/missing-helper"
    export PYTHON_BIN="${TEST_TMPDIR}/missing-python"
    export VIRTUAL_ENV="${TEST_TMPDIR}/missing-venv"
    local fail_err="${TEST_TMPDIR}/interpreter_fail.err"
    if _cmd_python_interpreter >/dev/null 2>"${fail_err}"; then
        fail_rc=0
    else
        fail_rc=$?
    fi
    PATH="${previous_path}"
    case "$-" in
        *x*) cat "${fail_err}" >&2 || true ;;
    esac
    assert_rc "1" "${fail_rc}" "interpreter resolver fails when no real python candidate exists"

    if [[ "${had_helper}" == "1" ]]; then
        export PACK_HELPER_PYTHON_BIN="${previous_helper}"
    else
        unset PACK_HELPER_PYTHON_BIN
    fi
    if [[ "${had_python_bin}" == "1" ]]; then
        export PYTHON_BIN="${previous_python_bin}"
    else
        unset PYTHON_BIN
    fi
    if [[ "${had_virtual_env}" == "1" ]]; then
        export VIRTUAL_ENV="${previous_virtual_env}"
    else
        unset VIRTUAL_ENV
    fi
}

test_rand_jitter_ms_positive_returns_value_in_range() {
    mock_reset
    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    local max="7"
    local val
    val="$(_rand_jitter_ms "${max}")"
    [[ "${val}" =~ ^-?[0-9]+$ ]] || t_fail "expected integer jitter got='${val}'"
    if [[ ${val} -lt -${max} || ${val} -gt ${max} ]]; then
        t_fail "jitter out of range val=${val} max=${max}"
    fi
}

test_now_iso_plus_seconds_invalid_input_coerces_to_zero_seconds() {
    mock_reset

    # Provide deterministic clock behavior and ensure python fallback isn't used.
    _cmd_date() { echo "2025-01-01T00:00:00Z"; }
    _cmd_python() { echo "ERROR: python fallback should not be used" >&2; return 1; }

    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    assert_eq "2025-01-01T00:00:00Z" "$(_now_iso_plus_seconds "not-a-number")" "invalid seconds coerced to 0"
}

test_now_iso_plus_seconds_uses_date_v_and_python_fallback_when_date_d_fails() {
    mock_reset

    # Force date -d/-v fallbacks (the date mock fails for -d/-j/-v*), but keep the output deterministic.
    _cmd_python() { echo "2025-01-01T00:00:10Z"; }

    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    assert_eq "2025-01-01T00:00:10Z" "$(_now_iso_plus_seconds "10")" "python fallback used when date flags unavailable"
}

test_pid_is_alive_backend_uses_proc_hook() {
    mock_reset
    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    _has_proc_dir() { return 0; }
    assert_eq "proc" "$(_pid_is_alive_backend)" "forced proc backend"

    _has_proc_dir() { return 1; }
    assert_eq "ps" "$(_pid_is_alive_backend)" "forced ps backend"
}

test_pid_is_alive_proc_checks_proc_pid_dir() {
    mock_reset
    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    local pid="$$"
    local expected_rc="1"
    if [[ -d "/proc/${pid}" ]]; then
        expected_rc="0"
    fi

    run _pid_is_alive_proc "${pid}"
    assert_rc "${expected_rc}" "${RUN_RC}" "proc backend follows /proc/<pid> directory presence"
}

test_pid_is_alive_ps_calls_ps_for_valid_pid() {
    mock_reset

    _cmd_ps() { return 0; }

    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    run _pid_is_alive_ps "123"
    assert_rc "0" "${RUN_RC}" "ps backend uses _cmd_ps for valid pid"
}

test_file_mtime_epoch_falls_back_to_stat_f_format_and_errors_when_unavailable() {
    mock_reset

    _cmd_stat() {
        if [[ "${1:-}" == "-c" ]]; then
            return 1
        fi
        if [[ "${1:-}" == "-f" ]]; then
            echo "1700000001"
            return 0
        fi
        return 1
    }

    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    assert_eq "1700000001" "$(_file_mtime_epoch "${TEST_TMPDIR}")" "stat -f fallback used when -c unsupported"

    _cmd_stat() { return 1; }
    run _file_mtime_epoch "${TEST_TMPDIR}"
    assert_rc "1" "${RUN_RC}" "returns non-zero when stat cannot report mtime"
}

test_file_mtime_epoch_prefers_stat_c_when_available() {
    mock_reset

    local calls="${TEST_TMPDIR}/stat.calls"
    : >"${calls}"

    _cmd_stat() {
        echo "$*" >> "${calls}"
        if [[ "${1:-}" == "-c" ]]; then
            echo "1700000002"
            return 0
        fi
        return 1
    }

    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    assert_eq "1700000002" "$(_file_mtime_epoch "${TEST_TMPDIR}")" "stat -c used when available"
    assert_match '-c %Y' "$(cat "${calls}")" "stat -c called"
}

test_file_mtime_epoch_returns_nonzero_on_non_numeric_output() {
    mock_reset

    _cmd_stat() {
        echo "not-a-number"
        return 0
    }

    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    run _file_mtime_epoch "${TEST_TMPDIR}"
    assert_rc "1" "${RUN_RC}" "non-numeric stat output returns non-zero"
}

test_pid_is_alive_case_arms_call_expected_impl() {
    mock_reset
    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    local calls="${TEST_TMPDIR}/pid.calls"
    : >"${calls}"

    _pid_is_alive_proc() { echo "proc $1" >> "${calls}"; return 0; }
    _pid_is_alive_ps() { echo "ps $1" >> "${calls}"; return 0; }

    _pid_is_alive_backend() { echo "proc"; }
    run _pid_is_alive "123"
    assert_rc "0" "${RUN_RC}" "proc arm ok"

    _pid_is_alive_backend() { echo "ps"; }
    run _pid_is_alive "456"
    assert_rc "0" "${RUN_RC}" "ps arm ok"

    _pid_is_alive_backend() { echo "weird"; }
    run _pid_is_alive "789"
    assert_rc "0" "${RUN_RC}" "default arm falls back to ps"

    local content
    content="$(cat "${calls}")"
    assert_match 'proc 123' "${content}" "proc impl called"
    assert_match 'ps 456' "${content}" "ps impl called"
    assert_match 'ps 789' "${content}" "default impl called"
}

test_iso_to_epoch_python_fallback_parses_iso() {
    mock_reset
    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    # The date mock forces portable fallback; verify the python fallback receives args.
    assert_eq "1735689610" "$(_iso_to_epoch "2025-01-01T00:00:10Z")" "iso parses to epoch seconds"
}

test_iso_to_epoch_empty_and_null_return_zero_without_shelling_out() {
    mock_reset

    _cmd_date() { echo "ERROR: date should not be called" >&2; return 1; }
    _cmd_python() { echo "ERROR: python should not be called" >&2; return 1; }

    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    assert_eq "0" "$(_iso_to_epoch "")" "empty iso returns 0"
    assert_eq "0" "$(_iso_to_epoch "null")" "null iso returns 0"
}

test_iso_to_epoch_uses_date_j_f_in_utc_when_available() {
    mock_reset

    local calls="${TEST_TMPDIR}/date.calls"
    : >"${calls}"

    _cmd_date() {
        echo "$*" >> "${calls}"
        if [[ "${1:-}" == "-d" ]]; then
            return 1
        fi
        if [[ "${1:-}" == "-u" && "${2:-}" == "-j" && "${3:-}" == "-f" ]]; then
            echo "1735689610"
            return 0
        fi
        return 1
    }
    _cmd_python() { echo "ERROR: python fallback should not be used" >&2; return 1; }

    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    assert_eq "1735689610" "$(_iso_to_epoch "2025-01-01T00:00:10Z")" "date -u -j -f path used when available"
    assert_match '-u -j -f' "$(cat "${calls}")" "calls include -u -j -f"
}

test_iso_to_epoch_falls_back_when_date_returns_non_numeric() {
    mock_reset

    _cmd_date() { echo "not-a-number"; return 0; }
    _cmd_python() { echo "1735689610"; }

    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    assert_eq "1735689610" "$(_iso_to_epoch "2025-01-01T00:00:10Z")" "non-numeric date output falls back to python"
}

test_pid_is_alive_ps_invalid_pid_short_circuits() {
    mock_reset
    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    run _pid_is_alive_ps "nope"
    assert_rc "1" "${RUN_RC}" "invalid pid returns non-zero"
}

test_now_iso_plus_seconds_negative_formats_date_args_without_v_plus_minus() {
    mock_reset

    local calls="${TEST_TMPDIR}/date.calls"
    : >"${calls}"

    _cmd_date() {
        echo "$*" >> "${calls}"
        for arg in "$@"; do
            case "${arg}" in
                -d|-v*) return 1 ;;
            esac
        done
        return 1
    }
    _cmd_python() { echo "2024-12-31T23:59:55Z"; }

    # shellcheck source=../runtime.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/core/runtime.sh"

    assert_eq "2024-12-31T23:59:55Z" "$(_now_iso_plus_seconds "-5")" "negative seconds uses python fallback"

    local content
    content="$(cat "${calls}")"
    assert_match '-d -5 seconds' "${content}" "date -d uses negative offset"
    assert_match '-v-5S' "${content}" "date -v uses -v-5S form"
}

test_python3_stub_passthrough_delegates_selected_scripts_to_real_python() {
    mock_reset

    mock_python3_stub_enable
    fixture_write "python3.stdout" "stubbed"
    fixture_write "python3.rc" "0"
    mock_python3_stub_allow_real_script "real_helper.py"

    cat > "${TEST_TMPDIR}/real_helper.py" <<'EOF'
print("real")
EOF

    run python3 "${TEST_TMPDIR}/real_helper.py"
    assert_rc "0" "${RUN_RC}" "selected script should bypass the stub"
    assert_eq "real" "${RUN_OUT}" "selected script runs under the real interpreter"

    run python3 "${TEST_TMPDIR}/other_helper.py"
    assert_rc "0" "${RUN_RC}" "non-selected script still uses the stub"
    assert_eq "stubbed" "${RUN_OUT}" "stub output preserved for non-selected scripts"
}

test_shell_tests_do_not_use_raw_python3_stub_fixture() {
    mock_reset

    local audit
    audit="$(
        python3 - <<'PY'
from pathlib import Path

repo_root = Path.cwd()
test_root = repo_root / "scripts" / "evidence_packs" / "tests"
pattern = 'fixture_write "' + 'python3.stub' + '"'
violations = []

for path in sorted(test_root.rglob("test_*.sh")):
    if any(part.startswith(".") for part in path.parts):
        continue
    text = path.read_text(encoding="utf-8")
    if pattern in text:
        violations.append(str(path))

if violations:
    print("\n".join(violations))
PY
    )"

    if [[ -n "${audit}" ]]; then
        t_fail "raw python3.stub fixture usage found: ${audit}"
    fi
}
