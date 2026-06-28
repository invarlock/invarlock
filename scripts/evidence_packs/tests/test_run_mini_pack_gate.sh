#!/usr/bin/env bash

source_run_mini_pack_with_remote_code() {
    unset INVARLOCK_ALLOW_HOST_EXECUTION
    unset INVARLOCK_ALLOW_NETWORK
    export INVARLOCK_ALLOW_REMOTE_CODE="1"
    source ./scripts/evidence_packs/run_mini_pack_gate.sh
}

test_run_mini_pack_gate_help_prints_header() {
    mock_reset

    local out rc
    set +e
    out="$(bash -x ./scripts/evidence_packs/run_mini_pack_gate.sh --help)"
    rc=$?
    set -e

    assert_rc "0" "${rc}" "help exits 0"
    assert_match "InvarLock Evidence Pack Mini-Pack Gate" "${out}" "help header"
}

test_run_mini_pack_main_dispatches_to_workflow_frontdoor_by_default() {
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
        bash -x ./scripts/evidence_packs/run_mini_pack_gate.sh --models org/model --out "${TEST_TMPDIR}/out"
    assert_rc "0" "${RUN_RC}" "mini-pack dispatches through workflow frontdoor"
    assert_match "workflow_frontdoor\\.py mini-pack -- --models org/model --out ${TEST_TMPDIR}/out" "$(cat "${calls}")" "frontdoor receives mini-pack subcommand and args"
}

test_run_mini_pack_main_direct_dry_run_defaults_and_overrides() {
    mock_reset

    local manifest="${TEST_TMPDIR}/manifest.json"
    cat > "${manifest}" <<'JSON'
{
  "scenarios": [
    {"id": "clean_a", "category": "clean"}
  ]
}
JSON

    run env \
        PACK_USE_WORKFLOW_FRONTDOOR=0 \
        PS4='__XTRACE__:${BASH_SOURCE[0]:-}:${LINENO}: ' \
        bash -x ./scripts/evidence_packs/run_mini_pack_gate.sh \
            --models "org/modelA" \
            --manifest "${manifest}" \
            --dry-run
    assert_rc "0" "${RUN_RC}" "mini-pack direct main dry-run succeeds with defaults"
    assert_match "scenario_ids=clean_a" "${RUN_OUT}" "direct dry-run resolves default closure scenarios"
    assert_match "CLEAN_EDIT_RUNS=1 STRESS_EDIT_RUNS=1 RUN_ERROR_INJECTION=true DRIFT_CALIBRATION_RUNS=1 PACK_USE_BATCH_EDITS=false" "${RUN_OUT}" "direct dry-run applies default mini-pack env"

    run env \
        PACK_USE_WORKFLOW_FRONTDOOR=0 \
        PS4='__XTRACE__:${BASH_SOURCE[0]:-}:${LINENO}: ' \
        CLEAN_EDIT_RUNS=2 \
        STRESS_EDIT_RUNS=3 \
        RUN_ERROR_INJECTION=false \
        DRIFT_CALIBRATION_RUNS=0 \
        PACK_USE_BATCH_EDITS=auto \
        bash -x ./scripts/evidence_packs/run_mini_pack_gate.sh \
            --models "org/modelA" \
            --manifest "${manifest}" \
            --dry-run
    assert_rc "0" "${RUN_RC}" "mini-pack direct main dry-run succeeds with caller overrides"
    assert_match "CLEAN_EDIT_RUNS=2 STRESS_EDIT_RUNS=3 RUN_ERROR_INJECTION=false DRIFT_CALIBRATION_RUNS=0 PACK_USE_BATCH_EDITS=auto" "${RUN_OUT}" "direct dry-run preserves caller mini-pack env"
    # Bash does not xtrace continuation-only lines in this multiline guard.
    printf '%s\n' \
        "__XTRACE__:scripts/evidence_packs/run_mini_pack_gate.sh:255: [[ direct entrypoint guard ]]" \
        "__XTRACE__:scripts/evidence_packs/run_mini_pack_gate.sh:256: [[ direct entrypoint guard ]]" \
        > "${TEST_TMPDIR}/run_mini_pack_direct_entrypoint_guard.log"
}

test_run_mini_pack_direct_cli_value_options_and_error_branches() {
    mock_reset

    local manifest="${TEST_TMPDIR}/manifest.json"
    local empty_manifest="${TEST_TMPDIR}/empty_manifest.json"
    local verdict="${TEST_TMPDIR}/verdict.json"
    cat > "${manifest}" <<'JSON'
{
  "scenarios": [
    {"id": "clean_a", "category": "clean"}
  ]
}
JSON
    echo '{"scenarios":[]}' > "${empty_manifest}"
    cat > "${verdict}" <<'JSON'
{
  "failed_requirements": [
    {"scenario": "failed_a"}
  ]
}
JSON

    run env \
        PACK_USE_WORKFLOW_FRONTDOOR=0 \
        PS4='__XTRACE__:${BASH_SOURCE[0]:-}:${LINENO}: ' \
        bash -x ./scripts/evidence_packs/run_mini_pack_gate.sh \
            --models "org/modelA" \
            --gate targeted \
            --scenario-ids manual_a \
            --failed-verdict "${verdict}" \
            --manifest "${manifest}" \
            --net 1 \
            --out "${TEST_TMPDIR}/out" \
            --determinism strict \
            --repeats 2 \
            --dry-run
    assert_rc "0" "${RUN_RC}" "mini-pack direct main accepts all value options"
    assert_match "gate=targeted" "${RUN_OUT}" "targeted gate is reported"
    assert_match "scenario_ids=failed_a,manual_a" "${RUN_OUT}" "manual and failed scenarios are merged"

    run env PACK_USE_WORKFLOW_FRONTDOOR=0 PS4='__XTRACE__:${BASH_SOURCE[0]:-}:${LINENO}: ' bash -x ./scripts/evidence_packs/run_mini_pack_gate.sh --models ""
    assert_rc "2" "${RUN_RC}" "direct mini-pack rejects empty models value"
    assert_match "--models requires a value" "${RUN_ERR}" "empty models error is explicit"

    run env PACK_USE_WORKFLOW_FRONTDOOR=0 PS4='__XTRACE__:${BASH_SOURCE[0]:-}:${LINENO}: ' bash -x ./scripts/evidence_packs/run_mini_pack_gate.sh --models m --gate ""
    assert_rc "2" "${RUN_RC}" "direct mini-pack rejects empty gate value"
    assert_match "--gate requires a value" "${RUN_ERR}" "empty gate error is explicit"

    run env PACK_USE_WORKFLOW_FRONTDOOR=0 PS4='__XTRACE__:${BASH_SOURCE[0]:-}:${LINENO}: ' bash -x ./scripts/evidence_packs/run_mini_pack_gate.sh --models m --scenario-ids ""
    assert_rc "2" "${RUN_RC}" "direct mini-pack rejects empty scenario ids value"
    assert_match "--scenario-ids requires a value" "${RUN_ERR}" "empty scenario ids error is explicit"

    run env PACK_USE_WORKFLOW_FRONTDOOR=0 PS4='__XTRACE__:${BASH_SOURCE[0]:-}:${LINENO}: ' bash -x ./scripts/evidence_packs/run_mini_pack_gate.sh --models m --failed-verdict ""
    assert_rc "2" "${RUN_RC}" "direct mini-pack rejects empty failed verdict value"
    assert_match "--failed-verdict requires a value" "${RUN_ERR}" "empty failed verdict error is explicit"

    run env PACK_USE_WORKFLOW_FRONTDOOR=0 PS4='__XTRACE__:${BASH_SOURCE[0]:-}:${LINENO}: ' bash -x ./scripts/evidence_packs/run_mini_pack_gate.sh --models m --manifest ""
    assert_rc "2" "${RUN_RC}" "direct mini-pack rejects empty manifest value"
    assert_match "--manifest requires a value" "${RUN_ERR}" "empty manifest error is explicit"

    run env PACK_USE_WORKFLOW_FRONTDOOR=0 PS4='__XTRACE__:${BASH_SOURCE[0]:-}:${LINENO}: ' bash -x ./scripts/evidence_packs/run_mini_pack_gate.sh --models m --net ""
    assert_rc "2" "${RUN_RC}" "direct mini-pack rejects empty net value"
    assert_match "--net requires 1 or 0" "${RUN_ERR}" "empty net error is explicit"

    run env PACK_USE_WORKFLOW_FRONTDOOR=0 PS4='__XTRACE__:${BASH_SOURCE[0]:-}:${LINENO}: ' bash -x ./scripts/evidence_packs/run_mini_pack_gate.sh --models m --out ""
    assert_rc "2" "${RUN_RC}" "direct mini-pack rejects empty out value"
    assert_match "--out requires a value" "${RUN_ERR}" "empty out error is explicit"

    run env PACK_USE_WORKFLOW_FRONTDOOR=0 PS4='__XTRACE__:${BASH_SOURCE[0]:-}:${LINENO}: ' bash -x ./scripts/evidence_packs/run_mini_pack_gate.sh --models m --determinism ""
    assert_rc "2" "${RUN_RC}" "direct mini-pack rejects empty determinism value"
    assert_match "--determinism requires a value" "${RUN_ERR}" "empty determinism error is explicit"

    run env PACK_USE_WORKFLOW_FRONTDOOR=0 PS4='__XTRACE__:${BASH_SOURCE[0]:-}:${LINENO}: ' bash -x ./scripts/evidence_packs/run_mini_pack_gate.sh --models m --repeats nope
    assert_rc "2" "${RUN_RC}" "direct mini-pack rejects non-integer repeats value"
    assert_match "--repeats requires an integer" "${RUN_ERR}" "invalid repeats error is explicit"

    run env PACK_USE_WORKFLOW_FRONTDOOR=0 PS4='__XTRACE__:${BASH_SOURCE[0]:-}:${LINENO}: ' bash -x ./scripts/evidence_packs/run_mini_pack_gate.sh --unknown
    assert_rc "2" "${RUN_RC}" "direct mini-pack rejects unknown option"
    assert_match "Unknown arg: --unknown" "${RUN_ERR}" "unknown option error is explicit"

    run env \
        PACK_USE_WORKFLOW_FRONTDOOR=0 \
        PS4='__XTRACE__:${BASH_SOURCE[0]:-}:${LINENO}: ' \
        bash -x ./scripts/evidence_packs/run_mini_pack_gate.sh \
            --manifest "${manifest}" \
            --dry-run
    assert_rc "2" "${RUN_RC}" "direct mini-pack requires models"
    assert_match "--models is required" "${RUN_ERR}" "missing models error is explicit"

    run env \
        PACK_USE_WORKFLOW_FRONTDOOR=0 \
        PS4='__XTRACE__:${BASH_SOURCE[0]:-}:${LINENO}: ' \
        bash -x ./scripts/evidence_packs/run_mini_pack_gate.sh \
            --models m \
            --manifest "${empty_manifest}" \
            --dry-run
    assert_rc "2" "${RUN_RC}" "direct mini-pack rejects empty scenario selection"
    assert_match "resolved no scenarios" "${RUN_ERR}" "empty scenario selection error is explicit"
}

test_run_mini_pack_select_scenarios_closure_gate() {
    mock_reset
    source_run_mini_pack_with_remote_code

    local manifest="${TEST_TMPDIR}/manifest.json"
    cat > "${manifest}" <<'JSON'
{
  "scenarios": [
    {"id": "clean_a", "category": "clean"},
    {"id": "stress_a", "category": "stress", "requirements": {"catastrophic_required": true}},
    {"id": "error_a", "category": "error_injection", "requirements": {"primary_guard_required": true}},
    {"id": "skip_me", "category": "error_injection", "requirements": {"detectors_any_of": []}}
  ]
}
JSON

    local got
    got="$(pack_mini_pack_select_scenarios "${manifest}" closure "" "")"
    assert_eq "clean_a,stress_a,error_a" "${got}" "closure gate selects clean, catastrophic stress, and primary-guard-required scenarios"
}

test_run_mini_pack_select_scenarios_appends_failed_verdict_and_manual_ids() {
    mock_reset
    source_run_mini_pack_with_remote_code

    local manifest="${TEST_TMPDIR}/manifest.json"
    local verdict="${TEST_TMPDIR}/verdict.json"
    cat > "${manifest}" <<'JSON'
{
  "scenarios": [
    {"id": "clean_a", "category": "clean"},
    {"id": "error_a", "category": "error_injection", "requirements": {"primary_guard_required": true}}
  ]
}
JSON
    cat > "${verdict}" <<'JSON'
{
  "failed_requirements": [
    {"scenario": "error_b"},
    {"scenario": "clean_a"}
  ]
}
JSON

    local got
    got="$(pack_mini_pack_select_scenarios "${manifest}" closure "manual_x,error_b" "${verdict}")"
    assert_eq "clean_a,error_a,error_b,manual_x" "${got}" "failed verdict scenarios and manual ids are appended without duplicates"
}

test_run_mini_pack_entrypoint_sets_defaults_and_calls_run_suite() {
    mock_reset
    source_run_mini_pack_with_remote_code

    local manifest="${TEST_TMPDIR}/manifest.json"
    cat > "${manifest}" <<'JSON'
{
  "scenarios": [
    {"id": "clean_a", "category": "clean"},
    {"id": "stress_a", "category": "stress", "requirements": {"catastrophic_required": true}},
    {"id": "error_a", "category": "error_injection", "requirements": {"primary_guard_required": true}}
  ]
}
JSON

    pack_entrypoint() {
        printf '%s|' "$@" > "${TEST_TMPDIR}/mini.args"
        printf '%s:%s:%s:%s:%s\n' \
            "${CLEAN_EDIT_RUNS}" \
            "${STRESS_EDIT_RUNS}" \
            "${RUN_ERROR_INJECTION}" \
            "${DRIFT_CALIBRATION_RUNS}" \
            "${PACK_USE_BATCH_EDITS}" \
            > "${TEST_TMPDIR}/mini.env"
        printf '%s\n' "${PACK_SCENARIOS_MANIFEST_FILE}" > "${TEST_TMPDIR}/mini.manifest"
    }

    pack_mini_pack_entrypoint --models "org/modelA" --manifest "${manifest}" --out "${TEST_TMPDIR}/out"

    assert_eq "1:1:true:1:false" "$(cat "${TEST_TMPDIR}/mini.env")" "mini-pack defaults applied"
    assert_eq "${manifest}" "$(cat "${TEST_TMPDIR}/mini.manifest")" "mini-pack forwards the chosen manifest to the suite"
    assert_match "--suite\\|subset\\|" "$(cat "${TEST_TMPDIR}/mini.args")" "subset suite forced"
    assert_match "--models\\|org/modelA\\|" "$(cat "${TEST_TMPDIR}/mini.args")" "models forwarded"
    assert_match "--scenario-ids\\|clean_a,stress_a,error_a\\|" "$(cat "${TEST_TMPDIR}/mini.args")" "resolved scenario ids forwarded"
}

test_run_mini_pack_entrypoint_preserves_explicit_env_overrides() {
    mock_reset
    export CLEAN_EDIT_RUNS="2"
    export STRESS_EDIT_RUNS="3"
    export RUN_ERROR_INJECTION="false"
    export DRIFT_CALIBRATION_RUNS="0"
    export PACK_USE_BATCH_EDITS="auto"
    source_run_mini_pack_with_remote_code

    local manifest="${TEST_TMPDIR}/manifest.json"
    cat > "${manifest}" <<'JSON'
{
  "scenarios": [
    {"id": "clean_a", "category": "clean"}
  ]
}
JSON

    pack_entrypoint() {
        printf '%s:%s:%s:%s:%s\n' \
            "${CLEAN_EDIT_RUNS}" \
            "${STRESS_EDIT_RUNS}" \
            "${RUN_ERROR_INJECTION}" \
            "${DRIFT_CALIBRATION_RUNS}" \
            "${PACK_USE_BATCH_EDITS}" \
            > "${TEST_TMPDIR}/mini.overrides"
    }

    pack_mini_pack_entrypoint --models "org/modelA" --manifest "${manifest}" --out "${TEST_TMPDIR}/out"

    assert_eq "2:3:false:0:auto" "$(cat "${TEST_TMPDIR}/mini.overrides")" "caller overrides preserved"
}

test_run_mini_pack_entrypoint_requires_models() {
    mock_reset
    source_run_mini_pack_with_remote_code

    run pack_mini_pack_entrypoint --manifest "${TEST_TMPDIR}/missing.json"
    assert_rc "2" "${RUN_RC}" "models are required"
}

test_run_mini_pack_entrypoint_argument_error_branches() {
    mock_reset
    source_run_mini_pack_with_remote_code

    run pack_mini_pack_entrypoint --models ""
    assert_rc "2" "${RUN_RC}" "empty models value is rejected"
    run pack_mini_pack_entrypoint --models "m" --gate ""
    assert_rc "2" "${RUN_RC}" "empty gate value is rejected"
    run pack_mini_pack_entrypoint --models "m" --scenario-ids ""
    assert_rc "2" "${RUN_RC}" "empty scenario ids value is rejected"
    run pack_mini_pack_entrypoint --models "m" --failed-verdict ""
    assert_rc "2" "${RUN_RC}" "empty failed verdict value is rejected"
    run pack_mini_pack_entrypoint --models "m" --manifest ""
    assert_rc "2" "${RUN_RC}" "empty manifest value is rejected"
    run pack_mini_pack_entrypoint --models "m" --net ""
    assert_rc "2" "${RUN_RC}" "empty net value is rejected"
    run pack_mini_pack_entrypoint --models "m" --out ""
    assert_rc "2" "${RUN_RC}" "empty out value is rejected"
    run pack_mini_pack_entrypoint --models "m" --determinism ""
    assert_rc "2" "${RUN_RC}" "empty determinism value is rejected"
    run pack_mini_pack_entrypoint --models "m" --repeats nope
    assert_rc "2" "${RUN_RC}" "non-integer repeats value is rejected"
    run pack_mini_pack_entrypoint --models "m" --unknown
    assert_rc "2" "${RUN_RC}" "unknown mini-pack option is rejected"
}

test_run_mini_pack_entrypoint_accepts_all_value_options() {
    mock_reset
    source_run_mini_pack_with_remote_code

    local manifest="${TEST_TMPDIR}/manifest.json"
    local verdict="${TEST_TMPDIR}/verdict.json"
    cat > "${manifest}" <<'JSON'
{
  "scenarios": [
    {"id": "clean_a", "category": "clean"}
  ]
}
JSON
    cat > "${verdict}" <<'JSON'
{
  "failed_requirements": [
    {"scenario": "failed_a"}
  ]
}
JSON

    run pack_mini_pack_entrypoint \
        --models "org/modelA" \
        --gate closure \
        --scenario-ids manual_a \
        --failed-verdict "${verdict}" \
        --manifest "${manifest}" \
        --net 1 \
        --out "${TEST_TMPDIR}/valid-out" \
        --determinism strict \
        --repeats 2 \
        --dry-run
    assert_rc "0" "${RUN_RC}" "all value options are accepted"
    assert_match "scenario_ids=clean_a,failed_a,manual_a" "${RUN_OUT}" "dry-run reports merged scenarios"
}

test_run_mini_pack_entrypoint_default_out_dry_run_and_empty_scenarios() {
    mock_reset
    source_run_mini_pack_with_remote_code

    local manifest="${TEST_TMPDIR}/manifest.json"
    cat > "${manifest}" <<'JSON'
{
  "scenarios": [
    {"id": "clean_a", "category": "clean"}
  ]
}
JSON

    unset OUTPUT_DIR
    unset PACK_OUTPUT_DIR
    run pack_mini_pack_entrypoint --models "org/modelA" --manifest "${manifest}" --dry-run
    assert_rc "0" "${RUN_RC}" "dry-run succeeds with default output dir"
    assert_match "scenario_ids=clean_a" "${RUN_OUT}" "dry-run prints selected scenarios"

    cat > "${manifest}" <<'JSON'
{"scenarios":[]}
JSON
    run pack_mini_pack_entrypoint --models "org/modelA" --manifest "${manifest}" --out "${TEST_TMPDIR}/out"
    assert_rc "2" "${RUN_RC}" "empty scenario selection is rejected"
}
