#!/usr/bin/env bash

test_run_suite_help_prints_header() {
    mock_reset

    local out rc
    set +e
    out="$(bash -x ./scripts/proof_packs/run_suite.sh --help)"
    rc=$?
    set -e

    assert_rc "0" "${rc}" "help exits 0"
    assert_match "InvarLock Proof Pack Suite" "${out}" "help header"
}

test_run_suite_entrypoint_parses_calibrate_only_and_run_only_flags() {
    mock_reset

    source ./scripts/proof_packs/run_suite.sh

    pack_apply_suite() { return 0; }
    pack_run_suite() { echo "${PACK_SUITE_MODE}:${RESUME_FLAG}" > "${TEST_TMPDIR}/entrypoint.flags"; }

    pack_entrypoint --calibrate-only --out "${TEST_TMPDIR}/out1"
    assert_eq "calibrate-only:false" "$(cat "${TEST_TMPDIR}/entrypoint.flags")" "calibrate-only sets mode without resume"

    pack_entrypoint --run-only --out "${TEST_TMPDIR}/out2"
    assert_eq "run-only:true" "$(cat "${TEST_TMPDIR}/entrypoint.flags")" "run-only sets mode and implies resume"
}

test_run_suite_entrypoint_sets_default_output_dir() {
    mock_reset

    source ./scripts/proof_packs/run_suite.sh

    pack_apply_suite() { return 0; }
    pack_run_suite() { echo "${OUTPUT_DIR}" > "${TEST_TMPDIR}/entrypoint.output_dir"; }
    date() { echo "20250101_000000"; }

    OUTPUT_DIR=""
    pack_entrypoint --resume

    assert_file_exists "${TEST_TMPDIR}/entrypoint.output_dir" "entrypoint ran"
    assert_eq "./proof_pack_runs/subset_20250101_000000" "$(cat "${TEST_TMPDIR}/entrypoint.output_dir")" "default output dir uses deterministic date"
}

test_run_suite_entrypoint_parses_net_flag() {
    mock_reset

    source ./scripts/proof_packs/run_suite.sh

    pack_apply_suite() { return 0; }
    pack_run_suite() { echo "${PACK_NET}" > "${TEST_TMPDIR}/entrypoint.net"; }

    pack_entrypoint --net 1 --out "${TEST_TMPDIR}/out"

    assert_eq "1" "$(cat "${TEST_TMPDIR}/entrypoint.net")" "net flag propagates"
}
