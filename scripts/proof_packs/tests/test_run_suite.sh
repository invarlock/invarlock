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

    pack_entrypoint --errors-only --out "${TEST_TMPDIR}/out_err"
    assert_eq "errors-only:false" "$(cat "${TEST_TMPDIR}/entrypoint.flags")" "errors-only sets mode without resume"

    pack_entrypoint --run-only --out "${TEST_TMPDIR}/out2"
    assert_eq "run-only:true" "$(cat "${TEST_TMPDIR}/entrypoint.flags")" "run-only sets mode and implies resume"
}

test_run_suite_entrypoint_parses_scenario_ids_flag() {
    mock_reset

    source ./scripts/proof_packs/run_suite.sh

    pack_apply_suite() { return 0; }
    pack_run_suite() { echo "${PACK_SCENARIO_IDS:-}" > "${TEST_TMPDIR}/entrypoint.scenario_ids"; }

    pack_entrypoint --scenario-ids "a,b,c" --out "${TEST_TMPDIR}/out"
    assert_eq "a,b,c" "$(cat "${TEST_TMPDIR}/entrypoint.scenario_ids")" "scenario ids propagated"
}

test_run_suite_entrypoint_parses_models_flag() {
    mock_reset

    source ./scripts/proof_packs/run_suite.sh

    pack_apply_suite() {
        MODEL_1="orig/model1"
        MODEL_2="orig/model2"
        export MODEL_1 MODEL_2
        return 0
    }
    pack_run_suite() {
        pack_model_list | paste -sd "," - > "${TEST_TMPDIR}/entrypoint.models"
    }

    pack_entrypoint --models "org/modelA,org/modelB" --out "${TEST_TMPDIR}/out"
    assert_eq "org/modelA,org/modelB" "$(cat "${TEST_TMPDIR}/entrypoint.models")" "models override suite defaults"
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

test_run_suite_entrypoint_errors_on_missing_values() {
    mock_reset

    source ./scripts/proof_packs/run_suite.sh

    pack_apply_suite() { return 0; }
    pack_run_suite() { return 0; }

    run pack_entrypoint --suite
    assert_rc "2" "${RUN_RC}" "missing suite value"

    run pack_entrypoint --net
    assert_rc "2" "${RUN_RC}" "missing net value"

    run pack_entrypoint --out
    assert_rc "2" "${RUN_RC}" "missing out value"

    run pack_entrypoint --models
    assert_rc "2" "${RUN_RC}" "missing models value"

    run pack_entrypoint --scenario-ids
    assert_rc "2" "${RUN_RC}" "missing scenario-ids value"

    run pack_entrypoint --determinism
    assert_rc "2" "${RUN_RC}" "missing determinism value"

    run pack_entrypoint --repeats nope
    assert_rc "2" "${RUN_RC}" "invalid repeats value"
}

test_run_suite_entrypoint_parses_determinism_and_repeats_values() {
    mock_reset

    source ./scripts/proof_packs/run_suite.sh

    pack_apply_suite() { return 0; }
    pack_run_suite() { echo "${PACK_DETERMINISM}:${PACK_REPEATS}" > "${TEST_TMPDIR}/entrypoint.det"; }

    pack_entrypoint --determinism strict --repeats 2 --out "${TEST_TMPDIR}/out"

    assert_eq "strict:2" "$(cat "${TEST_TMPDIR}/entrypoint.det")" "determinism and repeats set"
}

test_run_suite_entrypoint_validates_net_and_unknown_args() {
    mock_reset

    source ./scripts/proof_packs/run_suite.sh

    pack_apply_suite() { return 0; }
    pack_run_suite() { return 0; }

    run pack_entrypoint --net 2 --out "${TEST_TMPDIR}/out"
    assert_rc "2" "${RUN_RC}" "invalid net value"

    run pack_entrypoint --nope
    assert_rc "2" "${RUN_RC}" "unknown arg returns 2"
}

test_run_suite_entrypoint_handles_double_dash() {
    mock_reset

    source ./scripts/proof_packs/run_suite.sh

    pack_apply_suite() { return 0; }
    pack_run_suite() { echo "${PACK_SUITE}" > "${TEST_TMPDIR}/entrypoint.suite"; }

    pack_entrypoint -- --suite full

    assert_eq "subset" "$(cat "${TEST_TMPDIR}/entrypoint.suite")" "double-dash stops parsing"
}

test_run_suite_entrypoint_determinism_branches() {
    mock_reset

    source ./scripts/proof_packs/run_suite.sh

    PACK_DETERMINISM="strict"
    pack_apply_entrypoint_determinism
    assert_eq "0" "${CUDNN_BENCHMARK}" "strict disables benchmark"
    assert_eq ":4096:8" "${CUBLAS_WORKSPACE_CONFIG}" "strict sets cublas config"

    PACK_DETERMINISM="bogus"
    pack_apply_entrypoint_determinism
    assert_eq "throughput" "${PACK_DETERMINISM}" "invalid determinism defaults"
    assert_eq "1" "${CUDNN_BENCHMARK}" "throughput enables benchmark"
}

test_run_suite_entrypoint_errors_on_invalid_suite() {
    mock_reset

    source ./scripts/proof_packs/run_suite.sh

    pack_apply_suite() { return 2; }
    pack_run_suite() { return 0; }

    run pack_entrypoint --suite nope --out "${TEST_TMPDIR}/out"
    assert_rc "2" "${RUN_RC}" "invalid suite returns 2"
}

test_run_suite_entrypoint_rejects_empty_and_oversized_model_lists() {
    mock_reset

    source ./scripts/proof_packs/run_suite.sh

    pack_apply_suite() { return 0; }
    pack_run_suite() { return 0; }

    run pack_entrypoint --models " , , " --out "${TEST_TMPDIR}/out_empty"
    assert_rc "2" "${RUN_RC}" "blank model list is rejected"
    assert_match "no valid model ids" "${RUN_ERR}" "blank model error explains failure"

    run pack_entrypoint --models "m1,m2,m3,m4,m5,m6,m7,m8,m9" --out "${TEST_TMPDIR}/out_many"
    assert_rc "2" "${RUN_RC}" "more than eight models is rejected"
    assert_match "up to 8 models" "${RUN_ERR}" "oversized list error explains limit"
}
