#!/usr/bin/env bash

test_result_compiler_generate_verdict_passes_manifest_when_present() {
    mock_reset

    # shellcheck source=../lib/result_compiler.sh
    source "${TEST_ROOT}/scripts/proof_packs/lib/result_compiler.sh"

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    mkdir -p "${OUTPUT_DIR}/state" "${OUTPUT_DIR}/reports"
    printf '%s\n' '{"schema":"proof_pack_scenarios_v1","schema_version":1,"scenarios":[]}' > "${OUTPUT_DIR}/state/scenarios.json"

    log_section() { :; }
    log() { printf '%s\n' "$*" >> "${TEST_TMPDIR}/result_compiler.log"; }
    _pack_result_compiler_root() { echo "${TEST_ROOT}/scripts/proof_packs"; }
    python3() {
        printf '%s\n' "$*" > "${TEST_TMPDIR}/python.args"
        mkdir -p "${OUTPUT_DIR}/reports"
        printf '%s\n' 'PASS' > "${OUTPUT_DIR}/reports/final_verdict.txt"
        printf '%s\n' '{"verdict":"PASS"}' > "${OUTPUT_DIR}/reports/final_verdict.json"
        return 0
    }

    generate_verdict

    assert_match "--manifest[[:space:]]+${OUTPUT_DIR}/state/scenarios.json" "$(cat "${TEST_TMPDIR}/python.args")" "manifest forwarded to verdict generator"
    assert_match "final_verdict.json" "$(cat "${TEST_TMPDIR}/result_compiler.log")" "result compiler logs final verdict artifact"
}
