#!/usr/bin/env bash

test_create_edited_model_unknown_type_exits_nonzero() {
    mock_reset
    # shellcheck source=../model_creation.sh
    source "${TEST_ROOT}/scripts/lib/model_creation.sh"

    local rc=0
    if ( create_edited_model "/b" "${TEST_TMPDIR}/out/model" "nope" "8" "128" "ffn" "0" ); then
        rc=0
    else
        rc=$?
    fi
    assert_rc "1" "${rc}" "unknown edit type exits"
}

test_create_pruned_model_invokes_python_wrapper() {
    mock_reset
    # shellcheck source=../model_creation.sh
    source "${TEST_ROOT}/scripts/lib/model_creation.sh"

    local calls="${TEST_TMPDIR}/python.calls"
    _cmd_python() { echo "python $*" >> "${calls}"; cat >/dev/null || true; return 0; }

    create_pruned_model "${TEST_TMPDIR}/baseline" "${TEST_TMPDIR}/out/pruned" "0.1" "ffn" "0"
    assert_file_exists "${calls}" "python called"
    assert_match "python - ${TEST_TMPDIR}/baseline ${TEST_TMPDIR}/out/pruned 0.1 ffn" "$(cat "${calls}")" "args passed via argv"
}

test_create_pruned_model_returns_nonzero_when_parent_dir_is_file() {
    mock_reset
    # shellcheck source=../model_creation.sh
    source "${TEST_ROOT}/scripts/lib/model_creation.sh"

    local calls="${TEST_TMPDIR}/python.calls"
    _cmd_python() { echo "python $*" >> "${calls}"; cat >/dev/null || true; return 0; }

    local out_file="${TEST_TMPDIR}/out_file"
    echo "not a directory" > "${out_file}"

    run create_pruned_model "${TEST_TMPDIR}/baseline" "${out_file}/pruned" "0.1" "ffn" "0"
    assert_ne "0" "${RUN_RC}" "mkdir failure returns non-zero"
    [[ ! -f "${calls}" ]] || t_fail "python should not be called when mkdir fails"
}

test_create_edited_model_quant_rtn_invokes_python_wrapper() {
    mock_reset
    # shellcheck source=../model_creation.sh
    source "${TEST_ROOT}/scripts/lib/model_creation.sh"

    local calls="${TEST_TMPDIR}/python.calls"
    _cmd_python() { echo "python $*" >> "${calls}"; cat >/dev/null || true; return 0; }

    create_edited_model "${TEST_TMPDIR}/baseline" "${TEST_TMPDIR}/out/edited" "quant_rtn" "8" "128" "ffn" "0"
    assert_file_exists "${calls}" "python called for quant_rtn"
    assert_match "python - ${TEST_TMPDIR}/baseline ${TEST_TMPDIR}/out/edited 8 128 ffn" "$(cat "${calls}")" "args passed via argv"
}

test_create_lowrank_model_invokes_python_wrapper() {
    mock_reset
    # shellcheck source=../model_creation.sh
    source "${TEST_ROOT}/scripts/lib/model_creation.sh"

    local calls="${TEST_TMPDIR}/python.calls"
    _cmd_python() { echo "python $*" >> "${calls}"; cat >/dev/null || true; return 0; }

    create_lowrank_model "${TEST_TMPDIR}/baseline" "${TEST_TMPDIR}/out/lowrank" "256" "ffn" "0"
    assert_file_exists "${calls}" "python called for lowrank"
    assert_match "python - ${TEST_TMPDIR}/baseline ${TEST_TMPDIR}/out/lowrank 256 ffn" "$(cat "${calls}")" "args passed via argv"
}

test_create_fp4_model_invokes_python_wrapper() {
    mock_reset
    # shellcheck source=../model_creation.sh
    source "${TEST_ROOT}/scripts/lib/model_creation.sh"

    local calls="${TEST_TMPDIR}/python.calls"
    _cmd_python() { echo "python $*" >> "${calls}"; cat >/dev/null || true; return 0; }

    create_fp4_model "${TEST_TMPDIR}/baseline" "${TEST_TMPDIR}/out/fp4" "e2m1" "ffn" "0"
    assert_file_exists "${calls}" "python called for fp4"
    assert_match "python - ${TEST_TMPDIR}/baseline ${TEST_TMPDIR}/out/fp4 e2m1 ffn" "$(cat "${calls}")" "args passed via argv"
}

test_create_error_model_invokes_python_wrapper() {
    mock_reset
    # shellcheck source=../model_creation.sh
    source "${TEST_ROOT}/scripts/lib/model_creation.sh"

    local calls="${TEST_TMPDIR}/python.calls"
    _cmd_python() { echo "python $*" >> "${calls}"; cat >/dev/null || true; return 0; }

    create_error_model "${TEST_TMPDIR}/baseline" "${TEST_TMPDIR}/out/error" "nan_injection" "0"
    assert_file_exists "${calls}" "python called for error model"
    assert_match "python - ${TEST_TMPDIR}/baseline ${TEST_TMPDIR}/out/error nan_injection" "$(cat "${calls}")" "args passed via argv"
}
