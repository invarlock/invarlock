#!/usr/bin/env bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/validation_suite_test_helpers.sh"

test_pack_validation_list_run_gpu_ids_prefers_gpu_id_list_and_falls_back_to_num_gpus() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    GPU_ID_LIST="2,3,,4"
    local out
    out="$(list_run_gpu_ids)"
    assert_match "^2" "${out}" "GPU_ID_LIST parsing uses comma split"

    GPU_ID_LIST=""
    NUM_GPUS="not-a-number"
    out="$(list_run_gpu_ids)"
    assert_match "^0" "${out}" "fallback generates numeric ids"
}

test_pack_validation_configure_gpu_pool_parses_sources_and_validates_ids() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    # CUDA_VISIBLE_DEVICES branch
    CUDA_VISIBLE_DEVICES="0,1"
    NUM_GPUS=""
    configure_gpu_pool
    assert_eq "0,1" "${GPU_ID_LIST}" "GPU_ID_LIST from CUDA_VISIBLE_DEVICES"
    assert_eq "2" "${NUM_GPUS}" "NUM_GPUS inferred"

    # GPU_ID_LIST branch
    CUDA_VISIBLE_DEVICES=""
    GPU_ID_LIST="0"
    NUM_GPUS="bogus"
    configure_gpu_pool
    assert_eq "0" "${GPU_ID_LIST}" "GPU_ID_LIST preserved"
    assert_eq "1" "${NUM_GPUS}" "NUM_GPUS sanitized to available count"

    # nvidia-smi discovery branch
    CUDA_VISIBLE_DEVICES=""
    GPU_ID_LIST=""
    fixture_write "nvidia-smi/indices" $'0\n1\n2\n'
    NUM_GPUS="5"
    configure_gpu_pool
    assert_eq "0,1,2" "${GPU_ID_LIST}" "clamps to available GPUs"
    assert_eq "3" "${NUM_GPUS}" "clamped NUM_GPUS"

    # NUM_GPUS <1 clamp branch
    CUDA_VISIBLE_DEVICES="0,1"
    GPU_ID_LIST=""
    NUM_GPUS="0"
    configure_gpu_pool
    assert_eq "0" "${GPU_ID_LIST}" "requested <1 clamps to first GPU"
    assert_eq "1" "${NUM_GPUS}" "requested <1 clamps to 1"
}

test_pack_validation_configure_gpu_pool_errors_on_non_numeric_invalid_or_empty() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    # Non-numeric branch
    local rc=0
    CUDA_VISIBLE_DEVICES="0,a"
    ( configure_gpu_pool ) || rc=$?
    assert_ne "0" "${rc}" "non-numeric id triggers error_exit"

    # Invalid id branch
    fixture_write "nvidia-smi/invalid_ids" "$(printf '99\n')"
    CUDA_VISIBLE_DEVICES="99"
    rc=0
    ( configure_gpu_pool ) || rc=$?
    assert_ne "0" "${rc}" "invalid id triggers error_exit"

    # No usable ids branch
    CUDA_VISIBLE_DEVICES=","
    rc=0
    ( configure_gpu_pool ) || rc=$?
    assert_ne "0" "${rc}" "empty gpu list triggers error_exit"
}

test_pack_validation_format_gb_as_tb_returns_empty_for_invalid_input() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local out
    out="$(format_gb_as_tb "nope")"
    assert_eq "" "${out}" "invalid gb returns empty string"
}

test_pack_validation_get_free_disk_gb_parses_df_output() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local path="${TEST_TMPDIR}/disk"
    mkdir -p "${path}"

    mock_df_set_output "$(cat <<'EOF'
Filesystem  1G-blocks  Used Available Use% Mounted on
/dev/mock      1000    10       987G   1% /
EOF
)"

    assert_eq "987" "$(get_free_disk_gb "${path}")" "extracts available GB from df -BG output"
}

test_pack_validation_estimate_model_weights_covers_known_patterns_and_local_path() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local local_model="${TEST_TMPDIR}/local_model"
    mkdir -p "${local_model}"

    local out rc
    set +e
    out="$(estimate_model_weights_gb "${local_model}")"
    rc=$?
    set -e
    assert_ne "0" "${rc}" "local model path returns unknown"

    assert_eq "90" "$(estimate_model_weights_gb "mistralai/Mixtral-8x7B-v0.1")" "MoE special-case"
    assert_eq "144" "$(estimate_model_weights_gb "org/Thing-72B")" "72B"
    assert_eq "140" "$(estimate_model_weights_gb "org/Thing-70B")" "70B"
    assert_eq "68" "$(estimate_model_weights_gb "01-ai/Yi-34B")" "34B"
    assert_eq "64" "$(estimate_model_weights_gb "Qwen/Qwen2.5-32B")" "32B"
    assert_eq "28" "$(estimate_model_weights_gb "Qwen/Qwen2.5-14B")" "14B"
    assert_eq "26" "$(estimate_model_weights_gb "Qwen/Qwen2.5-13B")" "13B"
    assert_eq "14" "$(estimate_model_weights_gb "Qwen/Qwen2.5-7B")" "7B alt"
    assert_eq "14" "$(estimate_model_weights_gb "mistralai/Mistral-7B-v0.1")" "7B"
    assert_eq "18" "$(estimate_model_weights_gb "Qwen/Qwen3.5-9B")" "9B"
    assert_eq "16" "$(estimate_model_weights_gb "Qwen/Qwen3-8B")" "8B"
    assert_eq "4" "$(estimate_model_weights_gb "google/gemma-4-e2b")" "embedded 2B"
    assert_eq "3" "$(estimate_model_weights_gb "TinyLlama/TinyLlama-1.1B-Chat-v1.0")" "decimal 1.1B"
}

test_pack_validation_estimate_model_weights_default_case_returns_nonzero() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local rc=0
    local out
    set +e
    out="$(estimate_model_weights_gb "unknown/NoMatch")"
    rc=$?
    set -e
    assert_ne "0" "${rc}" "unknown model id returns non-zero"
    assert_eq "" "${out}" "unknown model id prints no estimate"
}

test_pack_validation_edit_creators_run_offline_with_stubbed_python() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    log() { :; }
    log_section() { :; }
    _cmd_python() { return 0; }

    create_pruned_model "${TEST_TMPDIR}/baseline" "${TEST_TMPDIR}/edits/prune/model" "0.1" "ffn" "0"
    create_lowrank_model "${TEST_TMPDIR}/baseline" "${TEST_TMPDIR}/edits/svd/model" "256" "ffn" "0"
    create_fp8_model "${TEST_TMPDIR}/baseline" "${TEST_TMPDIR}/edits/fp8/model" "e4m3fn" "ffn" "0"
    create_error_model "${TEST_TMPDIR}/baseline" "${TEST_TMPDIR}/errors/nan/model" "nan_injection" "0"
}

test_pack_validation_estimate_planned_storage_accounts_for_modes_and_unknown_models() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    # Force the hub cache to appear on a different device than OUTPUT_DIR.
    export HF_HUB_CACHE="${TEST_TMPDIR}/hub"
    mkdir -p "${HF_HUB_CACHE}" "${OUTPUT_DIR}"
    mock_df_set_output ""  # clear global output
    fixture_write "df.P.out" "$(printf 'Filesystem 512-blocks Used Available Capacity Mounted on\n/dev/outdev 1 1 1 1%% %s\n' "${OUTPUT_DIR}")"
    fixture_write "df.P.hub" "$(printf 'Filesystem 512-blocks Used Available Capacity Mounted on\n/dev/hubdev 1 1 1 1%% %s\n' "${HF_HUB_CACHE}")"

    RUN_ERROR_INJECTION="true"
    PACK_BASELINE_STORAGE_MODE="snapshot_symlink"
    MODEL_1="mistralai/Mistral-7B-v0.1"
    MODEL_2="unknown/NoProfile"
    MODEL_3=""
    MODEL_4=""
    MODEL_5=""
    MODEL_6=""
    MODEL_7=""
    MODEL_8=""

    local rc=0
    local out
    set +e
    out="$(estimate_planned_model_storage_gb)"
    rc=$?
    set -e
    assert_ne "0" "${rc}" "unknown models return non-zero"
    assert_eq "" "${out}" "unknown model returns empty planned gb"
}

test_pack_validation_estimate_planned_storage_succeeds_when_all_models_are_known() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    HF_HUB_CACHE=""
    MODEL_1="mistralai/Mistral-7B-v0.1"
    MODEL_2="mistralai/Mistral-7B-v0.1"
    MODEL_3="mistralai/Mistral-7B-v0.1"
    MODEL_4="mistralai/Mistral-7B-v0.1"
    MODEL_5="mistralai/Mistral-7B-v0.1"
    MODEL_6="mistralai/Mistral-7B-v0.1"
    MODEL_7="mistralai/Mistral-7B-v0.1"
    MODEL_8="mistralai/Mistral-7B-v0.1"
    RUN_ERROR_INJECTION="false"

    local out
    out="$(estimate_planned_model_storage_gb)"
    assert_match "^[0-9]+$" "${out}" "planned gb computed"
}

test_pack_validation_disk_preflight_allows_resume_but_aborts_without_resume() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    get_free_disk_gb() { echo "10"; }
    estimate_planned_model_storage_gb() { echo "1000"; }

    MIN_FREE_DISK_GB="bogus"

    RESUME_FLAG="true"
    disk_preflight

    RESUME_FLAG="false"
    error_exit() { exit 99; }
    local rc=0
    ( disk_preflight ) || rc=$?
    assert_eq "99" "${rc}" "non-resume path aborts via error_exit"
}

test_pack_validation_disk_preflight_fails_closed_for_release_review() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    error_exit() { exit 97; }
    PACK_RELEASE_REVIEW=1

    get_free_disk_gb() { echo ""; }
    estimate_planned_model_storage_gb() { echo "10"; }
    local rc=0
    ( disk_preflight ) || rc=$?
    assert_eq "97" "${rc}" "release-review aborts when free disk is unknown"

    get_free_disk_gb() { echo "5000"; }
    estimate_planned_model_storage_gb() { echo ""; }
    rc=0
    ( disk_preflight ) || rc=$?
    assert_eq "97" "${rc}" "release-review aborts when storage estimate is unknown"

    RESUME_FLAG="true"
    get_free_disk_gb() { echo "10"; }
    estimate_planned_model_storage_gb() { echo "1000"; }
    rc=0
    ( disk_preflight ) || rc=$?
    assert_eq "97" "${rc}" "release-review resume does not bypass insufficient disk"
}

test_pack_validation_disk_preflight_non_release_allows_unknown_estimates() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    PACK_RELEASE_REVIEW=0
    get_free_disk_gb() { echo ""; }
    estimate_planned_model_storage_gb() { t_fail "planned storage should not be requested without free disk"; }

    disk_preflight

    get_free_disk_gb() { echo "5000"; }
    estimate_planned_model_storage_gb() { echo ""; }

    disk_preflight
}

test_pack_validation_disk_preflight_returns_ok_when_disk_is_sufficient() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    get_free_disk_gb() { echo "5000"; }
    estimate_planned_model_storage_gb() { echo "10"; }
    MIN_FREE_DISK_GB="200"

    disk_preflight
}

test_pack_validation_estimate_planned_storage_counts_snapshot_copy_baseline_materialization() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    EDIT_TYPES_CLEAN=()
    EDIT_TYPES_STRESS=()
    RUN_ERROR_INJECTION="false"
    PACK_BASELINE_STORAGE_MODE="snapshot_copy"

    pack_model_list() { printf '%s\n' "org/model"; }
    estimate_model_weights_gb() { echo "10"; }

    local total
    total="$(estimate_planned_model_storage_gb)"
    assert_eq "30" "${total}" "snapshot_copy counts cache download, baseline materialization, and one edit peak copy"
}

test_pack_validation_estimate_planned_storage_uses_scenario_error_count_fallback() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local fake_validation_dir="${TEST_TMPDIR}/fake_pack/lib/validation"
    mkdir -p "${fake_validation_dir}" "${TEST_TMPDIR}/fake_pack/lib"
    printf '{"scenarios":[]}\n' > "${TEST_TMPDIR}/fake_pack/lib/scenarios.json"
    _PACK_VALIDATION_LIB_DIR="${fake_validation_dir}"

    HF_HUB_CACHE=""
    RUN_ERROR_INJECTION="true"
    PACK_CLEANUP_MODELS="1"
    PACK_BASELINE_STORAGE_MODE="snapshot_symlink"
    PACK_USE_BATCH_EDITS="false"
    CLEAN_EDIT_RUNS="1"
    STRESS_EDIT_RUNS="0"

    pack_model_list() { printf '%s\n' "org/model"; }
    pack_count_edit_scenarios() { printf '%s\n' "1|0|fixture"; }
    estimate_model_weights_gb() { printf '%s\n' "10"; }
    _pack_validation_state() {
        if [[ "${1:-}" == "count-generation-kind" ]]; then
            printf '%s\n' "not-a-number"
            return 0
        fi
        return 1
    }

    local total
    total="$(estimate_planned_model_storage_gb)"
    assert_eq "30" "${total}" "invalid scenario error count falls back and contributes one cleanup-mode error peak"
}

test_pack_validation_disk_preflight_describes_cache_backed_symlink_mode() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    get_free_disk_gb() { echo "10"; }
    estimate_planned_model_storage_gb() { echo "1000"; }

    PACK_BASELINE_STORAGE_MODE="snapshot_symlink"
    MIN_FREE_DISK_GB="200"

    local rc=0
    ( disk_preflight ) || rc=$?
    assert_eq "1" "${rc}" "disk preflight aborts through error_exit"
    assert_match "cache-backed symlink tree" "$(cat "${OUTPUT_DIR}/logs/main.log")" "message explains snapshot_symlink semantics"
}

test_pack_validation_handle_disk_pressure_shutdown_and_reclaim_branches() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    QUEUE_DIR="${OUTPUT_DIR}/queue"
    mkdir -p "${QUEUE_DIR}"
    reclaim_orphaned_tasks() { echo "reclaimed:$1" >> "${TEST_TMPDIR}/reclaim.calls"; }
    list_run_gpu_ids() { printf '0\n1\n'; }

    error_exit() { exit 7; }

    # signal_shutdown exists branch
    signal_shutdown() { echo "shutdown:$1" >> "${TEST_TMPDIR}/shutdown.calls"; }
    local rc=0
    ( handle_disk_pressure "1" "200" ) || rc=$?
    assert_eq "7" "${rc}" "handle_disk_pressure aborts"
    assert_file_exists "${TEST_TMPDIR}/shutdown.calls" "signal_shutdown called"
    assert_file_exists "${TEST_TMPDIR}/reclaim.calls" "reclaim called"

    # signal_shutdown missing branch
    rm -f "${TEST_TMPDIR}/shutdown.calls"
    unset -f signal_shutdown 2>/dev/null || true
    rc=0
    ( handle_disk_pressure "1" "200" ) || rc=$?
    assert_eq "7" "${rc}" "handle_disk_pressure aborts when touching SHUTDOWN"
    assert_file_exists "${OUTPUT_DIR}/workers/SHUTDOWN" "shutdown marker touched"
}

test_pack_validation_setup_pack_environment_sets_fp8_flag_and_propagates_failure() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    python3() { printf '%s\n' "ok" "[FP8_NATIVE_SUPPORT=true]"; }
    setup_pack_environment
    assert_eq "true" "${FP8_NATIVE_SUPPORT}" "FP8_NATIVE_SUPPORT true"

    python3() { printf '%s\n' "ok" "[PACK_GPU_MEM_GB=48]" "[FP8_NATIVE_SUPPORT=true]"; }
    GPU_MEMORY_GB=""
    setup_pack_environment
    assert_eq "48" "${GPU_MEMORY_GB}" "GPU_MEMORY_GB set from PACK_GPU_MEM_GB"

    python3() { printf '%s\n' "ok" "[FP8_NATIVE_SUPPORT=false]"; }
    setup_pack_environment
    assert_eq "false" "${FP8_NATIVE_SUPPORT}" "FP8_NATIVE_SUPPORT false"

    python3() { printf '%s\n' "boom"; return 3; }
    local rc=0
    ( setup_pack_environment ) || rc=$?
    assert_eq "3" "${rc}" "propagates python3 rc"
}

test_pack_validation_check_dependencies_flash_attn_branches_and_package_installs() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    PACK_NET="1"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    fixture_write "timeout.stub" ""
    local req_dir="${TEST_TMPDIR}/requirements/evidence-packs"
    mkdir -p "${req_dir}"
    : > "${req_dir}/flash-attn.txt"
    : > "${req_dir}/protobuf.txt"
    : > "${req_dir}/sentencepiece.txt"
    pack_evidence_pack_requirement_path() {
        printf '%s/%s.txt\n' "${req_dir}" "$1"
    }

    python3() {
        if [[ "${1:-}" == "-c" ]]; then
            local code="${2:-}"
            case "${code}" in
                *"import torch; assert torch.cuda.is_available"*) return 0 ;;
                *"import transformers"*) return 0 ;;
                *"import invarlock"*) return 0 ;;
                *"import yaml"*) return 0 ;;
                *"import lm_eval"*) return 0 ;;
                *"import sysconfig; exit(0 if sysconfig.get_config_var('INCLUDEPY')"*) return "${SYS_INCLUDECHECK_RC:-0}" ;;
                *"print(sysconfig.get_config_var('INCLUDEPY'))"*) echo "${TEST_TMPDIR}/include"; return 0 ;;
                *"import flash_attn; print('Flash Attention OK')"*) return "${FLASH_ATTN_CHECK_RC:-0}" ;;
                *"import flash_attn"*) return "${FLASH_ATTN_VERIFY_RC:-0}" ;;
                *"import google.protobuf"*) return "${PROTOBUF_IMPORT_RC:-0}" ;;
                *"import sentencepiece"*) return "${SENTENCEPIECE_IMPORT_RC:-0}" ;;
                *) return 0 ;;
            esac
        fi
        if [[ "${1:-}" == "-m" && "${2:-}" == "pip" ]]; then
            return "${PIP_RC:-0}"
        fi
        return 0
    }

    # flash_attn available branch
    FLASH_ATTN_CHECK_RC=0
    SKIP_FLASH_ATTN="false"
    check_dependencies
    assert_eq "true" "${FLASH_ATTENTION_AVAILABLE}" "flash-attn available"

    # flash_attn skipped branch
    FLASH_ATTN_CHECK_RC=1
    SKIP_FLASH_ATTN="true"
    check_dependencies
    assert_eq "false" "${FLASH_ATTENTION_AVAILABLE}" "flash-attn skipped"

    # flash_attn missing and no python headers branch
    FLASH_ATTN_CHECK_RC=1
    SKIP_FLASH_ATTN="false"
    SYS_INCLUDECHECK_RC=1
    check_dependencies
    assert_eq "false" "${FLASH_ATTENTION_AVAILABLE}" "flash-attn missing and no headers"

    # flash_attn install branches (timeout ok/import ok, timeout ok/import fail, timeout fail)
    SYS_INCLUDECHECK_RC=0
    mkdir -p "${TEST_TMPDIR}/include"
    : > "${TEST_TMPDIR}/include/Python.h"

    FLASH_ATTN_CHECK_RC=1
    FLASH_ATTN_VERIFY_RC=0
    fixture_write "timeout.rc" "0"
    check_dependencies
    assert_eq "true" "${FLASH_ATTENTION_AVAILABLE}" "flash-attn installed and import succeeded"

    FLASH_ATTN_CHECK_RC=1
    FLASH_ATTN_VERIFY_RC=1
    fixture_write "timeout.rc" "0"
    check_dependencies
    assert_eq "false" "${FLASH_ATTENTION_AVAILABLE}" "flash-attn installed but import failed"

    FLASH_ATTN_CHECK_RC=1
    FLASH_ATTN_VERIFY_RC=1
    fixture_write "timeout.rc" "1"
    check_dependencies
    assert_eq "false" "${FLASH_ATTENTION_AVAILABLE}" "flash-attn install failed"
    assert_match "flash-attn install failed" "$(cat "${LOG_FILE}")" "flash-attn failed install fallback logged"

    FLASH_ATTN_CHECK_RC=1
    FLASH_ATTN_VERIFY_RC=1
    fixture_write "timeout.rc" "137"
    check_dependencies
    assert_eq "false" "${FLASH_ATTENTION_AVAILABLE}" "flash-attn killed build falls back"
    assert_match "flash-attn install failed" "$(cat "${LOG_FILE}")" "flash-attn killed build fallback logged"
    assert_match "--only-binary=:all:" "$(cat "${TEST_TMPDIR}/fixtures/timeout.calls")" "flash-attn install avoids source builds by default"

    : > "${TEST_TMPDIR}/fixtures/timeout.calls"
    PACK_FLASH_ATTN_ALLOW_SOURCE_BUILD=1
    FLASH_ATTN_CHECK_RC=1
    FLASH_ATTN_VERIFY_RC=1
    fixture_write "timeout.rc" "1"
    check_dependencies
    assert_eq "false" "${FLASH_ATTENTION_AVAILABLE}" "explicit flash-attn source build failure falls back"
    assert_match "--no-build-isolation" "$(cat "${TEST_TMPDIR}/fixtures/timeout.calls")" "flash-attn source build opt-in preserves source build args"
    [[ "$(cat "${TEST_TMPDIR}/fixtures/timeout.calls")" != *"--only-binary=:all:"* ]] || t_fail "source-build opt-in must not use only-binary"
    unset PACK_FLASH_ATTN_ALLOW_SOURCE_BUILD

    # protobuf + sentencepiece install branches
    PROTOBUF_IMPORT_RC=1
    SENTENCEPIECE_IMPORT_RC=1
    PIP_RC=0
    check_dependencies
}

test_pack_validation_prepare_flash_attn_build_toolchain_uses_pinned_nvcc() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    PACK_NET="1"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    local req_dir="${TEST_TMPDIR}/requirements/evidence-packs"
    mkdir -p "${req_dir}"
    : > "${req_dir}/cuda-nvcc.txt"
    pack_evidence_pack_requirement_path() {
        printf '%s/%s.txt\n' "${req_dir}" "$1"
    }

    python3() {
        if [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "install" ]]; then
            printf '%s\n' "$*" > "${TEST_TMPDIR}/pip.args"
            return 0
        fi
        if [[ "${1:-}" == "-" ]]; then
            cat >/dev/null
            printf '%s\n' "${TEST_TMPDIR}/site/nvidia/cu12"
            return 0
        fi
        return 0
    }

    PATH="/usr/bin:/bin"
    pack_prepare_flash_attn_build_toolchain "true"

    assert_match "requirements/evidence-packs/cuda-nvcc.txt" "$(cat "${TEST_TMPDIR}/pip.args")" "cuda-nvcc lock installed"
    assert_match "--no-deps" "$(cat "${TEST_TMPDIR}/pip.args")" "cuda-nvcc install stays no-deps"
    assert_eq "${TEST_TMPDIR}/site/nvidia/cu12" "${CUDA_HOME}" "CUDA_HOME points at pinned cuda-nvcc"
    assert_eq "${TEST_TMPDIR}/site/nvidia/cu12" "${CUDA_PATH}" "CUDA_PATH points at pinned cuda-nvcc"
    assert_match "^${TEST_TMPDIR}/site/nvidia/cu12/bin:" "${PATH}" "pinned nvcc is prepended to PATH"
}

test_pack_validation_flash_attn_toolchain_fallbacks_and_try_install_restore_errexit() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    PACK_NET="0"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    log() { printf '%s\n' "$*" >> "${TEST_TMPDIR}/flash.log"; }
    pack_prepare_flash_attn_build_toolchain "true"

    PACK_NET="1"
    pack_prepare_flash_attn_build_toolchain "false"

    local req_dir="${TEST_TMPDIR}/requirements"
    mkdir -p "${req_dir}"
    : > "${req_dir}/cuda-nvcc.txt"
    pack_evidence_pack_requirement_path() {
        printf '%s/%s.txt\n' "${req_dir}" "$1"
    }
    pack_install_pinned_requirement() {
        return 9
    }

    pack_prepare_flash_attn_build_toolchain "true"
    assert_match "cuda-nvcc install failed" "$(cat "${TEST_TMPDIR}/flash.log")" "failed cuda-nvcc install falls back to existing toolkit"

    local flash_req="${TEST_TMPDIR}/flash-attn.txt"
    : > "${flash_req}"
    timeout() { return 37; }

    local rc=0
    local errexit_after="set"
    set +e
    pack_try_install_flash_attn "${flash_req}"
    rc=$?
    case "$-" in
        *e*) errexit_after="set" ;;
        *) errexit_after="unset" ;;
    esac
    set -e

    assert_eq "37" "${rc}" "flash-attn install propagates timeout rc"
    assert_eq "unset" "${errexit_after}" "pack_try_install_flash_attn restores disabled errexit state"
}

test_pack_validation_check_dependencies_installs_and_records_missing_paths() {
    mock_reset

    local rc=0

    (
        OUTPUT_DIR="${TEST_TMPDIR}/deps_bootstrap"
        PACK_NET="1"
        source ./scripts/evidence_packs/lib/validation/validation_suite.sh
        pack_setup_output_dirs
        command() {
            if [[ "${1:-}" == "-v" ]]; then
                return 0
            fi
            builtin command "$@"
        }
        local pip_checks=0
        python3() {
            if [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "--version" ]]; then
                pip_checks=$((pip_checks + 1))
                [[ ${pip_checks} -ge 2 ]]
                return $?
            fi
            if [[ "${1:-}" == "-m" && "${2:-}" == "ensurepip" ]]; then
                return 0
            fi
            if [[ "${1:-}" == "-c" ]]; then
                return 0
            fi
            return 0
        }
        check_dependencies
    )

    rc=0
    (
        OUTPUT_DIR="${TEST_TMPDIR}/deps_offline_missing"
        PACK_NET="0"
        source ./scripts/evidence_packs/lib/validation/validation_suite.sh
        pack_setup_output_dirs
        error_exit() { exit 31; }
        command() {
            if [[ "${1:-}" == "-v" ]]; then
                return 0
            fi
            builtin command "$@"
        }
        python3() {
            if [[ "${1:-}" == "-c" ]]; then
                local code="${2:-}"
                case "${code}" in
                    *"import torch; assert torch.cuda.is_available"*|*"import transformers"*|*"import invarlock"*) return 0 ;;
                    *"import sysconfig; exit(0 if sysconfig.get_config_var('INCLUDEPY')"*) return 0 ;;
                    *"print(sysconfig.get_config_var('INCLUDEPY'))"*) printf '%s\n' "${TEST_TMPDIR}/include"; return 0 ;;
                    *) return 1 ;;
                esac
            fi
            return 0
        }
        mkdir -p "${TEST_TMPDIR}/include"
        : > "${TEST_TMPDIR}/include/Python.h"
        check_dependencies
    ) || rc=$?
    assert_eq "31" "${rc}" "offline missing optional dependencies abort through error_exit"

    rc=0
    (
        OUTPUT_DIR="${TEST_TMPDIR}/deps_net_no_pip"
        PACK_NET="1"
        source ./scripts/evidence_packs/lib/validation/validation_suite.sh
        pack_setup_output_dirs
        error_exit() { exit 32; }
        command() {
            if [[ "${1:-}" == "-v" ]]; then
                return 0
            fi
            builtin command "$@"
        }
        python3() {
            if [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "--version" ]]; then
                return 1
            fi
            if [[ "${1:-}" == "-m" && "${2:-}" == "ensurepip" ]]; then
                return 1
            fi
            if [[ "${1:-}" == "-c" ]]; then
                local code="${2:-}"
                case "${code}" in
                    *"import torch; assert torch.cuda.is_available"*|*"import transformers"*|*"import invarlock"*) return 0 ;;
                    *) return 1 ;;
                esac
            fi
            return 0
        }
        check_dependencies
    ) || rc=$?
    assert_eq "32" "${rc}" "net mode with unavailable pip records missing installable deps"

    rc=0
    (
        OUTPUT_DIR="${TEST_TMPDIR}/deps_net_install_fails"
        PACK_NET="1"
        source ./scripts/evidence_packs/lib/validation/validation_suite.sh
        pack_setup_output_dirs
        error_exit() { exit 33; }
        command() {
            if [[ "${1:-}" == "-v" ]]; then
                return 0
            fi
            builtin command "$@"
        }
        pack_install_pinned_requirement() { return 1; }
        pack_prepare_flash_attn_build_toolchain() { :; }
        pack_try_install_flash_attn() { return 1; }
        pack_evidence_pack_requirement_path() { printf '%s/%s.txt\n' "${TEST_TMPDIR}" "$1"; }
        python3() {
            if [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "--version" ]]; then
                return 0
            fi
            if [[ "${1:-}" == "-c" ]]; then
                local code="${2:-}"
                case "${code}" in
                    *"import torch; assert torch.cuda.is_available"*|*"import transformers"*|*"import invarlock"*) return 0 ;;
                    *"import sysconfig; exit(0 if sysconfig.get_config_var('INCLUDEPY')"*) return 0 ;;
                    *"print(sysconfig.get_config_var('INCLUDEPY'))"*) printf '%s\n' "${TEST_TMPDIR}/include"; return 0 ;;
                    *) return 1 ;;
                esac
            fi
            return 0
        }
        mkdir -p "${TEST_TMPDIR}/include"
        : > "${TEST_TMPDIR}/include/Python.h"
        check_dependencies
    ) || rc=$?
    assert_eq "33" "${rc}" "failed pinned installs are reported as missing dependencies"
}

test_pack_validation_check_dependencies_errors_when_missing() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    log_section() { :; }
    log() { :; }
    error_exit() { exit 11; }

    python3() { return 1; }
    local rc=0
    PATH=""
    ( check_dependencies ) || rc=$?
    assert_eq "11" "${rc}" "missing dependencies trigger error_exit"
}

test_pack_validation_preflight_datasets_success_and_offline_failure_paths() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    LOG_FILE="${TEST_TMPDIR}/preflight.log"
    LOG_LOCK="${TEST_TMPDIR}/preflight.lock"
    PACK_NET="0"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    : > "${LOG_FILE}"
    error_exit() { return 1; }

    local repo_root
    repo_root="$(pwd)"
    python3() {
        if [[ "${1:-}" == "${repo_root}/scripts/evidence_packs/python/runtime_tools.py" && "${2:-}" == "dataset-preflight" ]]; then
            if [[ "${DATASET_PREFLIGHT_MODE:-ok}" == "ok" ]]; then
                printf '%s\n' "dataset ok"
                return 0
            fi
            printf '%s\n' "dataset missing"
            return 1
        fi
        command python3 "$@"
    }

    pack_preflight_datasets
    assert_match "Dataset preflight: OK" "$(cat "${LOG_FILE}")" "successful preflight is logged"

    : > "${LOG_FILE}"
    DATASET_PREFLIGHT_MODE="fail"
    run pack_preflight_datasets
    assert_rc "1" "${RUN_RC}" "failing preflight returns non-zero"
}

test_pack_validation_source_libs_errors_when_required_libs_are_missing() {
    mock_reset

    source ./scripts/evidence_packs/lib/validation/validation_suite.sh

    local file
    for file in scheduler.sh task_functions.sh gpu_worker.sh; do
        local sandbox
        sandbox="$(_make_validation_suite_sandbox)"
        case "${file}" in
            scheduler.sh|gpu_worker.sh) rm -f "${sandbox}/lib/queue/${file}" ;;
            task_functions.sh) rm -f "${sandbox}/lib/tasks/${file}" ;;
        esac

        local rc=0
        (
            _pack_script_dir() { echo "${sandbox}"; }
            pack_source_libs
        ) || rc=$?
        assert_ne "0" "${rc}" "missing ${file} fails pack_source_libs"
    done
}

test_pack_validation_check_dependencies_covers_pip_bootstrap_and_missing_install_paths() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    PACK_NET="1"
    source ./scripts/evidence_packs/lib/validation/validation_suite.sh
    pack_setup_output_dirs

    log_section() { :; }
    log() { printf '%s\n' "$*" >> "${TEST_TMPDIR}/dep.log"; }
    error_exit() { exit 17; }
    timeout() { shift; "$@"; }
    command() {
        if [[ "${1:-}" == "-v" && "${2:-}" == "invarlock" ]]; then
            return 0
        fi
        builtin command "$@"
    }

    local pip_version_calls=0
    local installed_modules=""
    local mode="bootstrap"
    python3() {
        if [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "--version" ]]; then
            pip_version_calls=$((pip_version_calls + 1))
            case "${mode}" in
                bootstrap)
                    [[ ${pip_version_calls} -ge 2 ]]
                    return $?
                    ;;
                nopip)
                    return 1
                    ;;
                *)
                    return 0
                    ;;
            esac
        fi
        if [[ "${1:-}" == "-m" && "${2:-}" == "ensurepip" ]]; then
            [[ "${mode}" == "bootstrap" ]]
            return $?
        fi
        if [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "install" ]]; then
            local pip_args="$*"
            if [[ "${pip_args}" == *"requirements/evidence-packs/accelerate.txt"* && "${pip_args}" != *"--no-deps"* ]]; then
                return 1
            fi
            case "${pip_args}" in
                *"requirements/evidence-packs/huggingface_hub.txt"*|\
                *"requirements/evidence-packs/accelerate.txt"*|\
                *"requirements/evidence-packs/pyyaml.txt"*|\
                *"requirements/evidence-packs/protobuf.txt"*|\
                *"requirements/evidence-packs/sentencepiece.txt"*)
                    if [[ "${mode}" == "failinstalls" ]]; then
                        return 1
                    fi
                    case "${pip_args}" in
                        *"requirements/evidence-packs/huggingface_hub.txt"*) installed_modules="${installed_modules} huggingface_hub" ;;
                        *"requirements/evidence-packs/accelerate.txt"*) installed_modules="${installed_modules} accelerate" ;;
                        *"requirements/evidence-packs/pyyaml.txt"*) installed_modules="${installed_modules} yaml" ;;
                        *"requirements/evidence-packs/protobuf.txt"*) installed_modules="${installed_modules} google.protobuf" ;;
                        *"requirements/evidence-packs/sentencepiece.txt"*) installed_modules="${installed_modules} sentencepiece" ;;
                    esac
                    return 0
                    ;;
                *"requirements/evidence-packs/flash-attn.txt"*)
                    if [[ "${mode}" == "flashfail" ]]; then
                        return 1
                    fi
                    return 0
                    ;;
            esac
            return 0
        fi
        if [[ "${1:-}" == "-c" ]]; then
            local code="${2:-}"
            case "${code}" in
                *"import torch; assert torch.cuda.is_available"*) return 0 ;;
                *"import transformers"*) return 0 ;;
                *"import invarlock"*) return 0 ;;
                *"import flash_attn; print('Flash Attention OK')"*) return 1 ;;
                *"import sysconfig; exit(0 if sysconfig.get_config_var('INCLUDEPY')"*) return 0 ;;
                *"print(sysconfig.get_config_var('INCLUDEPY'))"*) echo "${TEST_TMPDIR}/include"; return 0 ;;
                *"import flash_attn"*)
                    if [[ "${mode}" == "bootstrap" ]]; then
                        return 0
                    fi
                    return 1
                    ;;
                *"import huggingface_hub"*)
                    [[ " ${installed_modules} " == *" huggingface_hub "* ]] && return 0
                    return 1
                    ;;
                *"import accelerate"*)
                    [[ " ${installed_modules} " == *" accelerate "* ]] && return 0
                    return 1
                    ;;
                *"import yaml"*)
                    [[ " ${installed_modules} " == *" yaml "* ]] && return 0
                    return 1
                    ;;
                *"import google.protobuf"*)
                    [[ " ${installed_modules} " == *" google.protobuf "* ]] && return 0
                    return 1
                    ;;
                *"import sentencepiece"*)
                    [[ " ${installed_modules} " == *" sentencepiece "* ]] && return 0
                    return 1
                    ;;
            esac
        fi
        return 0
    }

    mkdir -p "${TEST_TMPDIR}/include"
    : > "${TEST_TMPDIR}/include/Python.h"
    check_dependencies

    mode="failinstalls"
    pip_version_calls=0
    installed_modules=""
    : > "${TEST_TMPDIR}/dep.log"
    local rc=0
    ( check_dependencies ) || rc=$?
    assert_eq "17" "${rc}" "failed dependency installs abort via error_exit"

    mode="nopip"
    pip_version_calls=0
    installed_modules=""
    : > "${TEST_TMPDIR}/dep.log"
    rc=0
    ( check_dependencies ) || rc=$?
    assert_eq "17" "${rc}" "missing pip/install dependencies abort via error_exit"

    PACK_NET="0"
    : > "${TEST_TMPDIR}/dep.log"
    rc=0
    ( check_dependencies ) || rc=$?
    assert_eq "17" "${rc}" "offline missing optional deps still aborts"
    assert_match "Flash Attention 2: Not found \\(offline\\), using eager attention" "$(cat "${TEST_TMPDIR}/dep.log")" "offline flash-attn fallback logged"
}
