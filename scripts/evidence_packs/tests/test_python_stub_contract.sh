#!/usr/bin/env bash

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
pattern = 'fixture_write "' + 'python3.stub' + '"'
violations = []

for path in sorted(repo_root.rglob("test_*.sh")):
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
