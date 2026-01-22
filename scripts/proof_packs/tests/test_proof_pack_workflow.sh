#!/usr/bin/env bash

test_pack_build_pack_and_verify_pack_end_to_end_v2_layout() {
    mock_reset

    PACK_GPG_SIGN=0
    PACK_SKIP_HTML=1
    PACK_PACK_LAYOUT=v2
    PACK_SUITE=subset
    PACK_NET=0
    PACK_DETERMINISM=throughput
    PACK_REPEATS=0

    local run_dir="${TEST_TMPDIR}/run"
    local pack_dir="${TEST_TMPDIR}/pack"

    mkdir -p "${run_dir}/reports" "${run_dir}/analysis" "${run_dir}/state"
    echo "PASS" > "${run_dir}/reports/final_verdict.txt"
    echo '{"ok":true}' > "${run_dir}/reports/final_verdict.json"
    echo 'model,edit' > "${run_dir}/analysis/eval_results.csv"
    echo 'm,quant_rtn' >> "${run_dir}/analysis/eval_results.csv"
    echo '{"model_list":["m"],"models":{"m":{"revision":"rev"}}}' > "${run_dir}/state/model_revisions.json"

    mkdir -p "${run_dir}/m/certificates/clean/quant_rtn"
    echo '{}' > "${run_dir}/m/certificates/clean/quant_rtn/evaluation.cert.json"

    source "${TEST_ROOT}/scripts/proof_packs/run_pack.sh"

    run pack_build_pack "${run_dir}" "${pack_dir}"
    assert_rc "0" "${RUN_RC}" "pack_build_pack succeeds"

    assert_file_exists "${pack_dir}/manifest.json" "manifest written"
    assert_file_exists "${pack_dir}/checksums.sha256" "checksums written"
    assert_file_exists "${pack_dir}/README.md" "readme written"
    assert_file_exists "${pack_dir}/results/verification_summary.json" "verification summary written"
    assert_file_exists "${pack_dir}/certs/m/clean/quant_rtn/evaluation.cert.json" "cert copied"

    assert_file_exists "${pack_dir}/metadata/manifest.json" "manifest copied to metadata"
    assert_file_exists "${pack_dir}/metadata/checksums.sha256" "checksums copied to metadata"

    local verify_json="${TEST_TMPDIR}/verify.json"
    run bash "${TEST_ROOT}/scripts/proof_packs/verify_pack.sh" --pack "${pack_dir}" --json-out "${verify_json}"
    assert_rc "0" "${RUN_RC}" "verify_pack succeeds"
    assert_file_exists "${verify_json}" "verify json written"
}

