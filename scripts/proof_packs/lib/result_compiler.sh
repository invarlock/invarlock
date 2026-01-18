#!/usr/bin/env bash
# result_compiler.sh - Analysis and verdict compilation for proof packs.
#
# This harness compiles an assurance-focused verdict from InvarLock certificates
# produced during the run.

_pack_result_compiler_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

compile_results() {
    mkdir -p "${OUTPUT_DIR}/analysis" "${OUTPUT_DIR}/reports"
}

run_analysis() {
    # Optional, non-gating analysis artifacts can be written under ${OUTPUT_DIR}/analysis.
    mkdir -p "${OUTPUT_DIR}/analysis"
}

generate_verdict() {
    log_section "FINAL VERDICT"

    local root
    root="$(_pack_result_compiler_root)"

    python3 "${root}/python/verdict_generator.py" --output-dir "${OUTPUT_DIR}"
    log "Wrote: ${OUTPUT_DIR}/reports/final_verdict.txt"
    log "Wrote: ${OUTPUT_DIR}/reports/final_verdict.json"
}
