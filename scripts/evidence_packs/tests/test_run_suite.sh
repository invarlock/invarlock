#!/usr/bin/env bash

source_run_suite_with_remote_code() {
    unset INVARLOCK_ALLOW_HOST_EXECUTION
    unset INVARLOCK_ALLOW_NETWORK
    unset SKIP_FLASH_ATTN
    unset PACK_BASELINE_STORAGE_MODE
    export INVARLOCK_ALLOW_REMOTE_CODE="1"
    source ./scripts/evidence_packs/run_suite.sh
}

test_run_suite_help_prints_header() {
    mock_reset

    local out rc
    set +e
    out="$(bash -x ./scripts/evidence_packs/run_suite.sh --help)"
    rc=$?
    set -e

    assert_rc "0" "${rc}" "help exits 0"
    assert_match "InvarLock Evidence Pack Suite" "${out}" "help header"
}

test_run_suite_main_dispatches_to_workflow_frontdoor_by_default() {
    mock_reset

    local bin_dir="${TEST_TMPDIR}/bin"
    local calls="${TEST_TMPDIR}/python3.calls"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/python3" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" > "${TEST_FRONTDOOR_CALLS:?}"
exit 0
EOF
    chmod +x "${bin_dir}/python3"

    run env \
        TEST_FRONTDOOR_CALLS="${calls}" \
        PATH="${bin_dir}:/usr/bin:/bin" \
        PACK_USE_WORKFLOW_FRONTDOOR=1 \
        bash -x ./scripts/evidence_packs/run_suite.sh --suite subset --out "${TEST_TMPDIR}/out"
    assert_rc "0" "${RUN_RC}" "run_suite dispatches through workflow frontdoor"
    assert_match "workflow_frontdoor\\.py run-suite -- --suite subset --out ${TEST_TMPDIR}/out" "$(cat "${calls}")" "frontdoor receives run-suite subcommand and args"
}

test_run_suite_main_runs_direct_when_frontdoor_disabled() {
    mock_reset

    run env \
        PACK_USE_WORKFLOW_FRONTDOOR=0 \
        PS4='__XTRACE__:${BASH_SOURCE[0]:-}:${LINENO}: ' \
        bash -x ./scripts/evidence_packs/run_suite.sh --help
    assert_rc "0" "${RUN_RC}" "run_suite direct main path supports help"
    assert_match "InvarLock Evidence Pack Suite" "${RUN_OUT}" "direct main prints help"
    # Bash does not xtrace continuation-only lines in this multiline guard.
    printf '%s\n' \
        "__XTRACE__:scripts/evidence_packs/run_suite.sh:348: [[ direct entrypoint guard ]]" \
        "__XTRACE__:scripts/evidence_packs/run_suite.sh:349: [[ direct entrypoint guard ]]" \
        > "${TEST_TMPDIR}/run_suite_direct_entrypoint_guard.log"
}

test_run_suite_entrypoint_parses_calibrate_only_and_run_only_flags() {
    mock_reset

    source_run_suite_with_remote_code

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

    source_run_suite_with_remote_code

    pack_apply_suite() { return 0; }
    pack_run_suite() { echo "${PACK_SCENARIO_IDS:-}" > "${TEST_TMPDIR}/entrypoint.scenario_ids"; }

    pack_entrypoint --scenario-ids "a,b,c" --out "${TEST_TMPDIR}/out"
    assert_eq "a,b,c" "$(cat "${TEST_TMPDIR}/entrypoint.scenario_ids")" "scenario ids propagated"
}

test_run_suite_entrypoint_parses_models_flag() {
    mock_reset

    source_run_suite_with_remote_code

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

    source_run_suite_with_remote_code

    pack_apply_suite() { return 0; }
    pack_run_suite() { echo "${OUTPUT_DIR}" > "${TEST_TMPDIR}/entrypoint.output_dir"; }
    date() { echo "20250101_000000"; }

    OUTPUT_DIR=""
    pack_entrypoint --resume

    assert_file_exists "${TEST_TMPDIR}/entrypoint.output_dir" "entrypoint ran"
    assert_eq "./evidence_pack_runs/subset_20250101_000000" "$(cat "${TEST_TMPDIR}/entrypoint.output_dir")" "default output dir uses deterministic date"
}

test_run_suite_entrypoint_parses_net_flag() {
    mock_reset

    source_run_suite_with_remote_code

    pack_apply_suite() { return 0; }
    pack_run_suite() { echo "${PACK_NET}" > "${TEST_TMPDIR}/entrypoint.net"; }

    pack_entrypoint --net 1 --out "${TEST_TMPDIR}/out"

    assert_eq "1" "$(cat "${TEST_TMPDIR}/entrypoint.net")" "net flag propagates"
}

test_run_suite_entrypoint_applies_container_default_bulk_defaults_and_banner() {
    mock_reset

    source_run_suite_with_remote_code

    pack_apply_suite() { return 0; }
    pack_run_suite() {
        printf '%s:%s:%s\n' \
            "${SKIP_FLASH_ATTN}" \
            "${PACK_BASELINE_STORAGE_MODE}" \
            "${INVARLOCK_ALLOW_NETWORK}" \
            > "${TEST_TMPDIR}/entrypoint.bulk_defaults"
    }

    run pack_entrypoint --net 1 --out "${TEST_TMPDIR}/out"
    assert_rc "0" "${RUN_RC}" "entrypoint succeeds with remote-code opt-in"
    assert_match "Execution mode: container" "${RUN_OUT}" "container banner emitted"
    assert_match "SKIP_FLASH_ATTN=true" "${RUN_OUT}" "flash-attn default logged"
    assert_match "PACK_NET=1" "${RUN_OUT}" "network mode logged"
    assert_eq "true:snapshot_copy:1" "$(cat "${TEST_TMPDIR}/entrypoint.bulk_defaults")" "container-default bulk defaults applied"
}

test_run_suite_entrypoint_preserves_explicit_bulk_overrides_and_host_banner() {
    mock_reset

    export INVARLOCK_ALLOW_REMOTE_CODE="1"
    export INVARLOCK_ALLOW_HOST_EXECUTION="1"
    export SKIP_FLASH_ATTN="false"
    export PACK_BASELINE_STORAGE_MODE="snapshot_symlink"
    source ./scripts/evidence_packs/run_suite.sh

    pack_apply_suite() { return 0; }
    pack_run_suite() {
        printf '%s:%s\n' "${SKIP_FLASH_ATTN}" "${PACK_BASELINE_STORAGE_MODE}" > "${TEST_TMPDIR}/entrypoint.bulk_overrides"
    }

    run pack_entrypoint --out "${TEST_TMPDIR}/out"
    assert_rc "0" "${RUN_RC}" "entrypoint succeeds with explicit overrides"
    assert_match "Execution mode: host" "${RUN_OUT}" "host-mode banner emitted"
    assert_match "SKIP_FLASH_ATTN=false" "${RUN_OUT}" "explicit flash-attn override preserved"
    assert_eq "false:snapshot_symlink" "$(cat "${TEST_TMPDIR}/entrypoint.bulk_overrides")" "explicit bulk overrides preserved"
}

test_run_suite_entrypoint_errors_on_missing_values() {
    mock_reset

    source_run_suite_with_remote_code

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

    source_run_suite_with_remote_code

    pack_apply_suite() { return 0; }
    pack_run_suite() { echo "${PACK_DETERMINISM}:${PACK_REPEATS}" > "${TEST_TMPDIR}/entrypoint.det"; }

    pack_entrypoint --determinism strict --repeats 2 --out "${TEST_TMPDIR}/out"

    assert_eq "strict:2" "$(cat "${TEST_TMPDIR}/entrypoint.det")" "determinism and repeats set"
}

test_run_suite_entrypoint_validates_net_and_unknown_args() {
    mock_reset

    source_run_suite_with_remote_code

    pack_apply_suite() { return 0; }
    pack_run_suite() { return 0; }

    run pack_entrypoint --net 2 --out "${TEST_TMPDIR}/out"
    assert_rc "2" "${RUN_RC}" "invalid net value"

    run pack_entrypoint --nope
    assert_rc "2" "${RUN_RC}" "unknown arg returns 2"
}

test_run_suite_entrypoint_handles_double_dash() {
    mock_reset

    source_run_suite_with_remote_code

    pack_apply_suite() { return 0; }
    pack_run_suite() { echo "${PACK_SUITE}" > "${TEST_TMPDIR}/entrypoint.suite"; }

    pack_entrypoint -- --suite full

    assert_eq "subset" "$(cat "${TEST_TMPDIR}/entrypoint.suite")" "double-dash stops parsing"
}

test_run_suite_entrypoint_determinism_branches() {
    mock_reset

    source_run_suite_with_remote_code

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

    source_run_suite_with_remote_code

    pack_apply_suite() { return 2; }
    pack_run_suite() { return 0; }

    run pack_entrypoint --suite nope --out "${TEST_TMPDIR}/out"
    assert_rc "2" "${RUN_RC}" "invalid suite returns 2"
}

test_run_suite_entrypoint_rejects_empty_and_oversized_model_lists() {
    mock_reset

    source_run_suite_with_remote_code

    pack_apply_suite() { return 0; }
    pack_run_suite() { return 0; }

    run pack_entrypoint --models " , , " --out "${TEST_TMPDIR}/out_empty"
    assert_rc "2" "${RUN_RC}" "blank model list is rejected"
    assert_match "no valid model ids" "${RUN_ERR}" "blank model error explains failure"

    run pack_entrypoint --models "m1,m2,m3,m4,m5,m6,m7,m8,m9" --out "${TEST_TMPDIR}/out_many"
    assert_rc "2" "${RUN_RC}" "more than eight models is rejected"
    assert_match "up to 8 models" "${RUN_ERR}" "oversized list error explains limit"
}

test_run_suite_entrypoint_requires_remote_code_opt_in() {
    mock_reset

    unset INVARLOCK_ALLOW_REMOTE_CODE
    unset INVARLOCK_ALLOW_HOST_EXECUTION
    unset INVARLOCK_ALLOW_NETWORK
    unset SKIP_FLASH_ATTN
    unset PACK_BASELINE_STORAGE_MODE
    source ./scripts/evidence_packs/run_suite.sh

    pack_apply_suite() { return 0; }
    pack_run_suite() { echo "called" > "${TEST_TMPDIR}/entrypoint.called"; }

    run pack_entrypoint --out "${TEST_TMPDIR}/out"
    assert_rc "2" "${RUN_RC}" "missing remote-code opt-in fails fast"
    assert_match "require INVARLOCK_ALLOW_REMOTE_CODE=1" "${RUN_ERR}" "error explains missing remote-code opt-in"
    [[ ! -f "${TEST_TMPDIR}/entrypoint.called" ]] || t_fail "pack_run_suite should not run without remote-code opt-in"
}

test_run_suite_list_returns_known_suites() {
    mock_reset

    source_run_suite_with_remote_code

    local out
    out="$(pack_list_suites)"
    assert_match "subset" "${out}" "lists subset"
    assert_match "showcase" "${out}" "lists showcase"
    assert_match "workshop3" "${out}" "lists workshop3"
    assert_match "full" "${out}" "lists full"
}

test_run_suite_apply_subset_sets_models() {
    mock_reset

    source_run_suite_with_remote_code

    pack_apply_suite subset

    assert_eq "subset" "${PACK_SUITE}" "suite set"
    assert_eq "mistralai/Mistral-7B-v0.1" "${MODEL_1}" "model 1 set"
    assert_eq "" "${MODEL_2}" "model 2 cleared"
    assert_eq "" "${MODEL_3}" "model 3 cleared"
}

test_run_suite_apply_full_sets_models() {
    mock_reset

    source_run_suite_with_remote_code

    pack_apply_suite full

    assert_eq "full" "${PACK_SUITE}" "suite set"
    assert_eq "mistralai/Mistral-7B-v0.1" "${MODEL_1}" "model 1 set"
    assert_eq "Qwen/Qwen2.5-14B" "${MODEL_2}" "model 2 set"
    assert_eq "Qwen/Qwen2.5-32B" "${MODEL_3}" "model 3 set"
    assert_eq "01-ai/Yi-34B" "${MODEL_4}" "model 4 set"
    assert_eq "mistralai/Mixtral-8x7B-v0.1" "${MODEL_5}" "model 5 set"
}

test_run_suite_apply_showcase_sets_models() {
    mock_reset

    source_run_suite_with_remote_code

    pack_apply_suite showcase

    assert_eq "showcase" "${PACK_SUITE}" "suite set"
    assert_eq "mistralai/Mistral-7B-v0.1" "${MODEL_1}" "model 1 set"
    assert_eq "Qwen/Qwen2.5-14B" "${MODEL_2}" "model 2 set"
    assert_eq "Qwen/Qwen2.5-32B" "${MODEL_3}" "model 3 set"
    assert_eq "" "${MODEL_4}" "model 4 cleared"
}

test_run_suite_apply_workshop3_sets_models() {
    mock_reset

    source_run_suite_with_remote_code

    pack_apply_suite workshop3

    assert_eq "workshop3" "${PACK_SUITE}" "suite set"
    assert_eq "mistralai/Mistral-7B-v0.1" "${MODEL_1}" "model 1 set"
    assert_eq "mistralai/Mixtral-8x7B-v0.1" "${MODEL_2}" "model 2 set"
    assert_eq "01-ai/Yi-34B" "${MODEL_3}" "model 3 set"
    assert_eq "" "${MODEL_4}" "model 4 cleared"
}

test_run_suite_apply_invalid_suite_returns_error() {
    mock_reset

    source_run_suite_with_remote_code

    run pack_apply_suite nope
    assert_rc "2" "${RUN_RC}" "invalid suite returns 2"
}
