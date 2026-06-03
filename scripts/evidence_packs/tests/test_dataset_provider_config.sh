#!/usr/bin/env bash

test_dataset_provider_config_default_string_provider() {
    mock_reset

    # shellcheck source=../lib/config/dataset_provider_config.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/dataset_provider_config.sh"

    unset INVARLOCK_DATASET INVARLOCK_DATASET_PROVIDER_YAML INVARLOCK_DATASET_PROVIDER_JSON

    local out
    out="$(pack_render_dataset_provider_yaml "")"
    assert_match 'provider: "wikitext2"' "${out}" "defaults to wikitext2 string provider"
}

test_dataset_provider_config_yaml_override_supports_blank_lines() {
    mock_reset

    # shellcheck source=../lib/config/dataset_provider_config.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/dataset_provider_config.sh"

    INVARLOCK_DATASET="wikitext2"
    INVARLOCK_DATASET_PROVIDER_YAML=$'kind: hf_text\n\nfoo: bar'
    export INVARLOCK_DATASET INVARLOCK_DATASET_PROVIDER_YAML

    local out
    out="$(pack_render_dataset_provider_yaml "${INVARLOCK_DATASET}")"
    assert_match '  provider:' "${out}" "emits provider mapping header"
    assert_match '    kind: hf_text' "${out}" "override mapping included"
    assert_match '    #' "${out}" "blank line preserved as comment"
    assert_match '    foo: bar' "${out}" "override key preserved"
}

test_dataset_provider_config_indent_helper_handles_single_line_input() {
    mock_reset

    # shellcheck source=../lib/config/dataset_provider_config.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/dataset_provider_config.sh"

    local out
    out="$(_pack_indent_lines "    " "alpha: beta")"
    assert_eq $'    alpha: beta' "${out}" "indent helper prefixes non-empty single lines"
}

test_dataset_provider_config_hf_text_defaults_c4_config_and_uses_cache_dir() {
    mock_reset

    # shellcheck source=../lib/config/dataset_provider_config.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/dataset_provider_config.sh"

    INVARLOCK_DATASET="hf_text"
    # Legacy "c4" should be migrated to "allenai/c4"
    INVARLOCK_HF_DATASET_NAME="c4"
    unset INVARLOCK_HF_CONFIG_NAME INVARLOCK_HF_DATASET_CONFIG_NAME
    INVARLOCK_HF_TEXT_FIELD="text"
    INVARLOCK_HF_MAX_SAMPLES="123"
    HF_DATASETS_CACHE="${TEST_TMPDIR}/hf_cache"
    export INVARLOCK_DATASET INVARLOCK_HF_DATASET_NAME INVARLOCK_HF_TEXT_FIELD INVARLOCK_HF_MAX_SAMPLES HF_DATASETS_CACHE

    local out
    out="$(pack_render_dataset_provider_yaml "${INVARLOCK_DATASET}")"
    assert_match 'kind: hf_text' "${out}" "hf_text mapping emitted"
    assert_match 'dataset_name: "allenai/c4"' "${out}" "c4 migrated to allenai/c4"
    assert_match 'config_name: "en"' "${out}" "allenai/c4 defaults config_name=en"
    assert_match 'max_samples: 123' "${out}" "numeric max_samples propagated"
    # trust_remote_code no longer auto-set for allenai/c4 (Parquet-based)
    assert_match "cache_dir: \"${HF_DATASETS_CACHE}\"" "${out}" "cache_dir propagated"
}

test_dataset_provider_config_hf_text_trust_remote_code_can_be_forced_false() {
    mock_reset

    # shellcheck source=../lib/config/dataset_provider_config.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/dataset_provider_config.sh"

    INVARLOCK_DATASET="hf_text"
    INVARLOCK_HF_DATASET_NAME="allenai/c4"
    INVARLOCK_HF_TRUST_REMOTE_CODE="false"
    export INVARLOCK_DATASET INVARLOCK_HF_DATASET_NAME INVARLOCK_HF_TRUST_REMOTE_CODE

    local out
    out="$(pack_render_dataset_provider_yaml "${INVARLOCK_DATASET}")"
    assert_match 'dataset_name: "allenai/c4"' "${out}" "dataset_name propagated"
    assert_match 'trust_remote_code: false' "${out}" "explicit false emitted"
}

test_dataset_provider_config_hf_text_trust_remote_code_truthy_values_emit_true() {
    mock_reset

    # shellcheck source=../lib/config/dataset_provider_config.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/dataset_provider_config.sh"

    INVARLOCK_DATASET="hf_text"
    INVARLOCK_HF_DATASET_NAME="demo/dataset"
    INVARLOCK_HF_TRUST_REMOTE_CODE=" yes "
    INVARLOCK_ALLOW_REMOTE_CODE="1"
    export INVARLOCK_DATASET INVARLOCK_HF_DATASET_NAME INVARLOCK_HF_TRUST_REMOTE_CODE INVARLOCK_ALLOW_REMOTE_CODE

    local out
    out="$(pack_render_dataset_provider_yaml "${INVARLOCK_DATASET}")"
    assert_match 'dataset_name: "demo/dataset"' "${out}" "dataset_name propagated"
    assert_match 'trust_remote_code: true' "${out}" "truthy override emitted"
}

test_dataset_provider_config_hf_text_trust_remote_code_requires_explicit_allow() {
    mock_reset

    # shellcheck source=../lib/config/dataset_provider_config.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/dataset_provider_config.sh"

    INVARLOCK_DATASET="hf_text"
    INVARLOCK_HF_DATASET_NAME="demo/dataset"
    INVARLOCK_HF_TRUST_REMOTE_CODE="true"
    unset INVARLOCK_ALLOW_REMOTE_CODE
    export INVARLOCK_DATASET INVARLOCK_HF_DATASET_NAME INVARLOCK_HF_TRUST_REMOTE_CODE

    local out=""
    local rc=0
    out="$(pack_render_dataset_provider_yaml "${INVARLOCK_DATASET}" 2>&1)" || rc=$?
    assert_eq "2" "${rc}" "missing explicit allow aborts remote-code dataset config"
    assert_match 'requires INVARLOCK_ALLOW_REMOTE_CODE=1' "${out}" "error explains remote-code opt-in"
}

test_dataset_provider_config_hf_text_omits_config_and_cache_and_sanitizes_max_samples() {
    mock_reset

    # shellcheck source=../lib/config/dataset_provider_config.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/dataset_provider_config.sh"

    INVARLOCK_DATASET="hf_text"
    INVARLOCK_HF_DATASET_NAME="wikitext"
    unset INVARLOCK_HF_CONFIG_NAME INVARLOCK_HF_DATASET_CONFIG_NAME
    INVARLOCK_HF_TEXT_FIELD="text"
    INVARLOCK_HF_MAX_SAMPLES="bogus"
    unset HF_DATASETS_CACHE INVARLOCK_HF_CACHE_DIR
    export INVARLOCK_DATASET INVARLOCK_HF_DATASET_NAME INVARLOCK_HF_TEXT_FIELD INVARLOCK_HF_MAX_SAMPLES

    local out
    out="$(pack_render_dataset_provider_yaml "${INVARLOCK_DATASET}")"
    assert_match 'kind: hf_text' "${out}" "hf_text mapping emitted"
    assert_match 'dataset_name: "wikitext"' "${out}" "dataset_name propagated"
    assert_match 'text_field: "text"' "${out}" "text_field propagated"
    assert_match 'max_samples: 2000' "${out}" "non-numeric max_samples falls back"
    if [[ "${out}" =~ config_name: ]]; then
        t_fail "config_name should be omitted when unset and dataset != c4"
    fi
    if [[ "${out}" =~ cache_dir: ]]; then
        t_fail "cache_dir should be omitted when unset"
    fi
}

test_dataset_provider_config_local_jsonl_file_branch() {
    mock_reset

    # shellcheck source=../lib/config/dataset_provider_config.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/dataset_provider_config.sh"

    INVARLOCK_DATASET="local_jsonl"
    INVARLOCK_LOCAL_JSONL_FILE="/data/file.jsonl"
    unset INVARLOCK_LOCAL_JSONL_PATH INVARLOCK_LOCAL_JSONL_DATA_FILES
    INVARLOCK_LOCAL_JSONL_TEXT_FIELD="text"
    INVARLOCK_LOCAL_JSONL_MAX_SAMPLES="12"
    export INVARLOCK_DATASET INVARLOCK_LOCAL_JSONL_FILE INVARLOCK_LOCAL_JSONL_TEXT_FIELD INVARLOCK_LOCAL_JSONL_MAX_SAMPLES

    local out
    out="$(pack_render_dataset_provider_yaml "${INVARLOCK_DATASET}")"
    assert_match 'kind: local_jsonl' "${out}" "local_jsonl mapping emitted"
    assert_match 'file: "/data/file.jsonl"' "${out}" "file branch emitted"
    assert_match 'max_samples: 12' "${out}" "numeric max_samples propagated"
}

test_dataset_provider_config_local_jsonl_path_branch_and_sanitizes_max_samples() {
    mock_reset

    # shellcheck source=../lib/config/dataset_provider_config.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/dataset_provider_config.sh"

    INVARLOCK_DATASET="local_jsonl"
    unset INVARLOCK_LOCAL_JSONL_FILE INVARLOCK_LOCAL_JSONL_DATA_FILES
    INVARLOCK_LOCAL_JSONL_PATH="/data"
    INVARLOCK_LOCAL_JSONL_MAX_SAMPLES="nope"
    export INVARLOCK_DATASET INVARLOCK_LOCAL_JSONL_PATH INVARLOCK_LOCAL_JSONL_MAX_SAMPLES

    local out
    out="$(pack_render_dataset_provider_yaml "${INVARLOCK_DATASET}")"
    assert_match 'kind: local_jsonl' "${out}" "local_jsonl mapping emitted"
    assert_match 'path: "/data"' "${out}" "path branch emitted"
    assert_match 'max_samples: 2000' "${out}" "non-numeric max_samples falls back"
}

test_dataset_provider_config_local_jsonl_data_files_branch() {
    mock_reset

    # shellcheck source=../lib/config/dataset_provider_config.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/dataset_provider_config.sh"

    INVARLOCK_DATASET="local_jsonl"
    unset INVARLOCK_LOCAL_JSONL_FILE INVARLOCK_LOCAL_JSONL_PATH
    INVARLOCK_LOCAL_JSONL_DATA_FILES="/data/*.jsonl"
    INVARLOCK_LOCAL_JSONL_MAX_SAMPLES="34"
    export INVARLOCK_DATASET INVARLOCK_LOCAL_JSONL_DATA_FILES INVARLOCK_LOCAL_JSONL_MAX_SAMPLES

    local out
    out="$(pack_render_dataset_provider_yaml "${INVARLOCK_DATASET}")"
    assert_match 'kind: local_jsonl' "${out}" "local_jsonl mapping emitted"
    assert_match 'data_files: "/data/\*\.jsonl"' "${out}" "data_files branch emitted"
    assert_match 'max_samples: 34' "${out}" "numeric max_samples propagated"
}

test_dataset_provider_config_local_jsonl_allows_missing_paths() {
    mock_reset

    # shellcheck source=../lib/config/dataset_provider_config.sh
    source "${TEST_ROOT}/scripts/evidence_packs/lib/config/dataset_provider_config.sh"

    INVARLOCK_DATASET="local_jsonl"
    unset INVARLOCK_LOCAL_JSONL_FILE INVARLOCK_LOCAL_JSONL_PATH INVARLOCK_LOCAL_JSONL_DATA_FILES
    export INVARLOCK_DATASET

    local out
    out="$(pack_render_dataset_provider_yaml "${INVARLOCK_DATASET}")"
    assert_match 'kind: local_jsonl' "${out}" "local_jsonl mapping emitted"
    if [[ "${out}" =~ (file:|path:|data_files:) ]]; then
        t_fail "expected local_jsonl without file/path/data_files keys"
    fi
}
