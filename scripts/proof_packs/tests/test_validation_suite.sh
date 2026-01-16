#!/usr/bin/env bash

test_pack_validation_cleanup_kills_spawned_pids_and_exits_with_previous_rc() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh

    local rc=0
    (
        set +e
        pids=(111 222)
        LOG_LOCK="${TEST_TMPDIR}/log.lock"

        kill() {
            local sig="${1:-}"
            local pid="${2:-}"
            if [[ "${sig}" == "-0" ]]; then
                [[ "${pid}" == "111" ]]
                return $?
            fi
            return 0
        }

        false
        cleanup
    ) || rc=$?

    assert_rc "1" "${rc}" "cleanup exits with previous rc"
}

test_pack_validation_determinism_strict_sets_compile_off() {
    mock_reset

    PACK_DETERMINISM="strict"
    source ./scripts/proof_packs/lib/validation_suite.sh

    assert_eq "strict" "${PACK_DETERMINISM}" "strict preserved"
    assert_eq "0" "${LMEVAL_TORCH_COMPILE}" "strict disables torch compile"
}

test_pack_validation_determinism_invalid_defaults_to_throughput() {
    mock_reset

    PACK_DETERMINISM="not-a-preset"
    source ./scripts/proof_packs/lib/validation_suite.sh

    assert_eq "throughput" "${PACK_DETERMINISM}" "invalid preset coerces to throughput"
    assert_eq "1" "${LMEVAL_TORCH_COMPILE}" "throughput enables torch compile"
}

test_pack_validation_bash4_guard_reports_error_on_bash3() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh
    local rc=0
    if pack_require_bash4; then
        rc=0
    else
        rc=$?
    fi
    assert_ne "0" "${rc}" "bash4 guard should fail under bash 3"
}

test_pack_validation_bash4_guard_succeeds_when_bash4_is_reported() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh

    pack_is_bash4() { return 0; }
    pack_require_bash4
}

test_pack_validation_setup_hf_cache_dirs_errors_when_home_is_file() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh

    local hf_home="${TEST_TMPDIR}/hf_file"
    : > "${hf_home}"
    export HF_HOME="${hf_home}"

    local rc=0
    if pack_setup_hf_cache_dirs; then
        rc=0
    else
        rc=$?
    fi
    assert_ne "0" "${rc}" "expected mkdir failure when HF_HOME is a file"
}

test_pack_validation_setup_hf_cache_dirs_creates_directories_and_returns_zero() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh

    export HF_HOME="${TEST_TMPDIR}/hf"
    unset HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE

    pack_setup_hf_cache_dirs
    assert_dir_exists "${HF_HOME}" "HF_HOME created"
    assert_dir_exists "${HF_HOME}/hub" "HF_HUB_CACHE created"
}

test_pack_validation_run_determinism_repeats_writes_summary() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    PACK_SUITE="subset"
    PACK_DETERMINISM="strict"
    PACK_REPEATS="2"
    source ./scripts/proof_packs/lib/validation_suite.sh

    pack_setup_output_dirs

    local model_id="org/model"
    local model_name
    model_name="$(sanitize_model_name "${model_id}")"
    mkdir -p "${OUTPUT_DIR}/${model_name}"
    mkdir -p "${TEST_TMPDIR}/baseline"
    echo "${TEST_TMPDIR}/baseline" > "${OUTPUT_DIR}/${model_name}/.baseline_path"

    PACK_MODEL_LIST=("${model_id}")

    process_edit() {
        mkdir -p "${TEST_TMPDIR}/edit"
        echo "${TEST_TMPDIR}/edit"
    }

    run_invarlock_certify() {
        local output_dir="$3"
        local run_name="$4"
        local run_dir="${output_dir}/${run_name}"
        mkdir -p "${run_dir}"
        local count_file="${TEST_TMPDIR}/repeat.count"
        local count=0
        if [[ -f "${count_file}" ]]; then
            count="$(cat "${count_file}")"
        fi
        count=$((count + 1))
        echo "${count}" > "${count_file}"
        cat > "${run_dir}/evaluation.cert.json" << EOF
{"verdict": {"primary_metric_ratio": ${count}.01}}
EOF
    }

    pack_run_determinism_repeats

    local path="${OUTPUT_DIR}/analysis/determinism_repeats.json"
    assert_file_exists "${path}" "determinism repeats file written"
    assert_match "\"completed\": 2" "$(cat "${path}")" "repeat count recorded"
}

_make_validation_suite_sandbox() {
    local sandbox
    sandbox="$(mktemp -d "${TEST_TMPDIR}/pack_validation_suite.XXXXXX")"
    mkdir -p "${sandbox}/lib"
    cp "${TEST_ROOT}/scripts/proof_packs/lib/"*.sh "${sandbox}/lib/"
    echo "${sandbox}"
}

test_pack_validation_source_libs_nested_layout_succeeds_and_exports_loaded_flags() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh

    pack_source_libs

    assert_match "/scripts/proof_packs/lib$" "${LIB_DIR}" "LIB_DIR points at scripts/proof_packs/lib"
    assert_eq "1" "${TASK_SERIALIZATION_LOADED:-}" "task_serialization loaded"
    assert_eq "1" "${QUEUE_MANAGER_LOADED:-}" "queue_manager loaded"
    assert_eq "1" "${SCHEDULER_LOADED:-}" "scheduler loaded"
    assert_eq "1" "${TASK_FUNCTIONS_LOADED:-}" "task_functions loaded"
    assert_eq "1" "${GPU_WORKER_LOADED:-}" "gpu_worker loaded"
    assert_eq "1" "${FAULT_TOLERANCE_LOADED:-}" "fault_tolerance loaded"
}

test_pack_validation_source_libs_falls_back_to_lib_dir_when_missing() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh

    local sandbox
    sandbox="$(_make_validation_suite_sandbox)"

    rm -rf "${sandbox}/lib"

    local rc=0
    (
        _pack_script_dir() { echo "${sandbox}"; }
        pack_source_libs
    ) || rc=$?
    assert_ne "0" "${rc}" "expected pack_source_libs failure when scripts/proof_packs/lib is missing"
}

test_pack_validation_source_libs_flat_layout_errors_when_queue_manager_missing() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh

    local sandbox
    sandbox="$(_make_validation_suite_sandbox)"

    mv "${sandbox}/lib" "${sandbox}/lib.__bak__"
    ln -s "lib.__bak__/runtime.sh" "${sandbox}/runtime.sh"
    ln -s "lib.__bak__/task_serialization.sh" "${sandbox}/task_serialization.sh"

    local rc=0
    (
        _pack_script_dir() { echo "${sandbox}"; }
        pack_source_libs
    ) || rc=$?
    assert_ne "0" "${rc}" "expected failure when queue_manager is missing in flat layout"
}

test_pack_validation_source_libs_flat_layout_errors_when_scheduler_missing() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh

    local sandbox
    sandbox="$(_make_validation_suite_sandbox)"

    mv "${sandbox}/lib" "${sandbox}/lib.__bak__"
    ln -s "lib.__bak__/runtime.sh" "${sandbox}/runtime.sh"
    ln -s "lib.__bak__/task_serialization.sh" "${sandbox}/task_serialization.sh"
    ln -s "lib.__bak__/queue_manager.sh" "${sandbox}/queue_manager.sh"

    local rc=0
    (
        _pack_script_dir() { echo "${sandbox}"; }
        pack_source_libs
    ) || rc=$?
    assert_ne "0" "${rc}" "expected failure when scheduler is missing in flat layout"
}

test_pack_validation_source_libs_flat_layout_errors_when_task_functions_missing() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh

    local sandbox
    sandbox="$(_make_validation_suite_sandbox)"

    mv "${sandbox}/lib" "${sandbox}/lib.__bak__"
    ln -s "lib.__bak__/runtime.sh" "${sandbox}/runtime.sh"
    ln -s "lib.__bak__/task_serialization.sh" "${sandbox}/task_serialization.sh"
    ln -s "lib.__bak__/queue_manager.sh" "${sandbox}/queue_manager.sh"
    ln -s "lib.__bak__/scheduler.sh" "${sandbox}/scheduler.sh"

    local rc=0
    (
        _pack_script_dir() { echo "${sandbox}"; }
        pack_source_libs
    ) || rc=$?
    assert_ne "0" "${rc}" "expected failure when task_functions is missing in flat layout"
}

test_pack_validation_source_libs_flat_layout_errors_when_gpu_worker_missing() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh

    local sandbox
    sandbox="$(_make_validation_suite_sandbox)"

    mv "${sandbox}/lib" "${sandbox}/lib.__bak__"
    ln -s "lib.__bak__/runtime.sh" "${sandbox}/runtime.sh"
    ln -s "lib.__bak__/task_serialization.sh" "${sandbox}/task_serialization.sh"
    ln -s "lib.__bak__/queue_manager.sh" "${sandbox}/queue_manager.sh"
    ln -s "lib.__bak__/scheduler.sh" "${sandbox}/scheduler.sh"
    ln -s "lib.__bak__/task_functions.sh" "${sandbox}/task_functions.sh"

    local rc=0
    (
        _pack_script_dir() { echo "${sandbox}"; }
        pack_source_libs
    ) || rc=$?
    assert_ne "0" "${rc}" "expected failure when gpu_worker is missing in flat layout"
}

test_pack_validation_list_run_gpu_ids_prefers_gpu_id_list_and_falls_back_to_num_gpus() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh

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
    source ./scripts/proof_packs/lib/validation_suite.sh
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
    source ./scripts/proof_packs/lib/validation_suite.sh
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

    source ./scripts/proof_packs/lib/validation_suite.sh

    local out
    out="$(format_gb_as_tb "nope")"
    assert_eq "" "${out}" "invalid gb returns empty string"
}

test_pack_validation_get_free_disk_gb_parses_df_output() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh

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

    source ./scripts/proof_packs/lib/validation_suite.sh

    local local_model="${TEST_TMPDIR}/local_model"
    mkdir -p "${local_model}"

    local out rc
    set +e
    out="$(estimate_model_weights_gb "${local_model}")"
    rc=$?
    set -e
    assert_ne "0" "${rc}" "local model path returns unknown"

    assert_eq "90" "$(estimate_model_weights_gb "mistralai/Mixtral-8x7B-v0.1")" "MoE special-case"
    assert_eq "144" "$(estimate_model_weights_gb "Qwen/Qwen1.5-72B")" "72B"
    assert_eq "140" "$(estimate_model_weights_gb "NousResearch/Llama-2-70b-hf")" "70B"
    assert_eq "68" "$(estimate_model_weights_gb "01-ai/Yi-34B")" "34B"
    assert_eq "64" "$(estimate_model_weights_gb "Qwen/Qwen2.5-32B")" "32B"
    assert_eq "28" "$(estimate_model_weights_gb "Qwen/Qwen2.5-14B")" "14B"
    assert_eq "26" "$(estimate_model_weights_gb "meta-llama/Llama-2-13b-hf")" "13B"
    assert_eq "14" "$(estimate_model_weights_gb "mistralai/Mistral-7B-v0.1")" "7B"
}

test_pack_validation_estimate_model_weights_default_case_returns_nonzero() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh

    local rc=0
    local out
    set +e
    out="$(estimate_model_weights_gb "unknown/NoMatch")"
    rc=$?
    set -e
    assert_ne "0" "${rc}" "unknown model id returns non-zero"
    assert_eq "" "${out}" "unknown model id prints no estimate"
}

test_pack_validation_process_edit_parses_specs_dispatches_creators_and_covers_resume_and_failure_paths() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    local baseline="${TEST_TMPDIR}/baseline"
    mkdir -p "${baseline}"
    local model_out="${TEST_TMPDIR}/model_out"
    mkdir -p "${model_out}"

    create_edited_model() { mkdir -p "${2}"; : > "${2}/config.json"; return "${CREATE_RC:-0}"; }
    create_fp8_model() { mkdir -p "${2}"; : > "${2}/config.json"; return 0; }
    create_pruned_model() { mkdir -p "${2}"; : > "${2}/config.json"; return 0; }
    create_lowrank_model() { mkdir -p "${2}"; : > "${2}/config.json"; return 0; }

    RESUME_MODE="false"

    # 3-part spec parsing + fp8_quant dir naming + case dispatch + success echo
    local fp8_path
    fp8_path="$(process_edit "${baseline}" "fp8_quant:e4m3fn:ffn" "clean" "m" "0" "${model_out}")"
    assert_dir_exists "${fp8_path}" "fp8 edit created"

    # quant_rtn dir naming + case dispatch
    local quant_path
    quant_path="$(process_edit "${baseline}" "quant_rtn:8:128:ffn" "stress" "m" "0" "${model_out}")"
    assert_dir_exists "${quant_path}" "quant edit created"

    # magnitude_prune invalid sparsity triggers validation error
    local rc=0
    ( process_edit "${baseline}" "magnitude_prune:notnum:ffn" "clean" "m" "0" "${model_out}" ) || rc=$?
    assert_ne "0" "${rc}" "invalid prune sparsity returns non-zero"

    # magnitude_prune valid path hits prune dir naming + case dispatch
    local prune_path
    prune_path="$(process_edit "${baseline}" "magnitude_prune:0.1:ffn" "clean" "m" "0" "${model_out}")"
    assert_dir_exists "${prune_path}" "prune edit created"

    # lowrank_svd dir naming + case dispatch
    local svd_path
    svd_path="$(process_edit "${baseline}" "lowrank_svd:256:ffn" "clean" "m" "0" "${model_out}")"
    assert_dir_exists "${svd_path}" "svd edit created"

    # Resume mode skips creation when directory + config exist
    RESUME_MODE="true"
    local resume_dir="${model_out}/models/fp8_e4m3fn_clean"
    mkdir -p "${resume_dir}"
    : > "${resume_dir}/config.json"
    assert_eq "${resume_dir}" "$(process_edit "${baseline}" "fp8_quant:e4m3fn:ffn" "clean" "m" "0" "${model_out}" | tail -n 1)" "resume echoes existing path"

    # Unknown edit type case arm
    rc=0
    ( process_edit "${baseline}" "unknown_type:1:2:3" "clean" "m" "0" "${model_out}" ) || rc=$?
    assert_ne "0" "${rc}" "unknown edit type returns non-zero"

    # Failure branch when creator returns non-zero
    RESUME_MODE="false"
    CREATE_RC=5
    rc=0
    ( process_edit "${baseline}" "quant_rtn:8:128:ffn" "clean" "m" "0" "${model_out}" ) || rc=$?
    assert_eq "1" "${rc}" "failed creator propagates as 1"
}

test_pack_validation_edit_creators_run_offline_with_stubbed_python() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    log() { :; }
    log_section() { :; }
    _cmd_python() { cat >/dev/null || true; return 0; }

    create_pruned_model "${TEST_TMPDIR}/baseline" "${TEST_TMPDIR}/edits/prune/model" "0.1" "ffn" "0"
    create_lowrank_model "${TEST_TMPDIR}/baseline" "${TEST_TMPDIR}/edits/svd/model" "256" "ffn" "0"
    create_fp8_model "${TEST_TMPDIR}/baseline" "${TEST_TMPDIR}/edits/fp8/model" "e4m3fn" "ffn" "0"
    create_error_model "${TEST_TMPDIR}/baseline" "${TEST_TMPDIR}/errors/nan/model" "nan_injection" "0"
}

test_pack_validation_estimate_planned_storage_accounts_for_modes_and_unknown_models() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh

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
    source ./scripts/proof_packs/lib/validation_suite.sh

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
    source ./scripts/proof_packs/lib/validation_suite.sh
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

test_pack_validation_disk_preflight_returns_ok_when_disk_is_sufficient() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh

    get_free_disk_gb() { echo "5000"; }
    estimate_planned_model_storage_gb() { echo "10"; }
    MIN_FREE_DISK_GB="200"

    disk_preflight
}

test_pack_validation_handle_disk_pressure_shutdown_and_reclaim_branches() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
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
    source ./scripts/proof_packs/lib/validation_suite.sh
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
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    fixture_write "timeout.stub" ""

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

    # protobuf + sentencepiece install branches
    PROTOBUF_IMPORT_RC=1
    SENTENCEPIECE_IMPORT_RC=1
    PIP_RC=0
    check_dependencies
}

test_pack_validation_check_dependencies_errors_when_missing() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
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

test_pack_validation_setup_model_early_returns_for_local_or_cached_paths_and_errors_on_failed_download() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs
    PACK_NET=1
    pack_model_revision() { echo "rev"; }

    local local_model="${TEST_TMPDIR}/local_model"
    mkdir -p "${local_model}"
    assert_eq "${local_model}" "$(setup_model "${local_model}" 0)" "local path returns unchanged"

    local model_id="Test/Model"
    local model_name
    model_name="$(sanitize_model_name "${model_id}")"
    mkdir -p "${OUTPUT_DIR}/models/${model_name}/baseline"
    assert_eq "${OUTPUT_DIR}/models/${model_name}/baseline" "$(setup_model "${model_id}" 0)" "cached sanitized baseline preferred"

    rm -rf "${OUTPUT_DIR}/models/${model_name}"
    mkdir -p "${OUTPUT_DIR}/models/model/baseline"
    assert_eq "${OUTPUT_DIR}/models/model/baseline" "$(setup_model "${model_id}" 0)" "cached basename baseline honored"

    rm -rf "${OUTPUT_DIR}/models"
    mkdir -p "${OUTPUT_DIR}/models"

    fixture_write "python3.stub" ""
    local rc=0
    local out
    set +e
    out="$(setup_model "Remote/NoCache" 0)"
    rc=$?
    set -e
    assert_ne "0" "${rc}" "stubbed download without marker fails"
    assert_eq "" "${out}" "failed download returns empty baseline path"
}

test_pack_validation_setup_model_cleans_incomplete_baseline_dir_on_download_failure() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs
    PACK_NET=1
    pack_model_revision() { echo "rev"; }

    local model_id="Remote/Incomplete"
    local model_name
    model_name="$(sanitize_model_name "${model_id}")"
    local baseline_path="${OUTPUT_DIR}/models/${model_name}/baseline"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/python3" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

count_file="${TEST_TMPDIR:-}/python3.count"
count=0
if [[ -f "${count_file}" ]]; then
    count="$(cat "${count_file}" 2>/dev/null || echo "0")"
fi
count=$((count + 1))
printf '%s\n' "${count}" > "${count_file}"

stdin_file="${TEST_TMPDIR:-}/python3.stdin"
mkdir -p "${TEST_TMPDIR:-}" 2>/dev/null || true
cat > "${stdin_file}"

output_dir="$(awk -F'"' '$0 ~ /^output_dir = Path/ { print $2; exit }' "${stdin_file}")"
[[ -n "${output_dir}" ]] && mkdir -p "${output_dir}"

# Simulate a download failure by not creating the success marker.
exit 1
EOF
    chmod +x "${bin_dir}/python3"
    export PATH="${bin_dir}:$PATH"
    hash -r 2>/dev/null || true

    local rc=0
    local out
    set +e
    out="$(setup_model "${model_id}" 0)"
    rc=$?
    set -e
    assert_ne "0" "${rc}" "download failure returns non-zero"
    assert_eq "" "${out}" "download failure returns empty baseline path"
    if [[ -d "${baseline_path}" ]]; then
        t_fail "baseline dir should be removed after download failure baseline_path='${baseline_path}'"
    fi
    assert_eq "1" "$(cat "${TEST_TMPDIR}/python3.count")" "python3 invoked for first attempt"

    set +e
    out="$(setup_model "${model_id}" 0)"
    rc=$?
    set -e
    assert_ne "0" "${rc}" "second attempt should not treat incomplete baseline as cached success"
    assert_eq "" "${out}" "second attempt still returns empty baseline path"
    assert_eq "2" "$(cat "${TEST_TMPDIR}/python3.count")" "python3 invoked again (no stale cached baseline dir)"
}

test_pack_validation_setup_model_succeeds_when_python_stub_creates_success_marker() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs
    PACK_NET=1
    pack_model_revision() { echo "rev"; }

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/python3" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

stdin_file="${TEST_TMPDIR:-}/python3.stdin"
mkdir -p "${TEST_TMPDIR:-}" 2>/dev/null || true
cat > "${stdin_file}"

output_dir="$(awk -F'"' '$0 ~ /^output_dir = Path/ { print $2; exit }' "${stdin_file}")"
marker="$(awk -F'"' '$0 ~ /^success_marker = Path/ { print $2; exit }' "${stdin_file}")"
printf 'output_dir=%s\nmarker=%s\n' "${output_dir}" "${marker}" > "${TEST_TMPDIR:-}/python3.parsed" 2>/dev/null || true

[[ -n "${output_dir}" ]] && mkdir -p "${output_dir}"
[[ -n "${marker}" ]] && : > "${marker}"
exit 0
EOF
    chmod +x "${bin_dir}/python3"
    export PATH="${bin_dir}:$PATH"
    hash -r 2>/dev/null || true

    assert_eq "${bin_dir}/python3" "$(command -v python3)" "python3 resolves to the test stub"

    local rc=0
    local out
    set +e
    out="$(setup_model "Remote/WithMarker" 0)"
    rc=$?
    set -e

    assert_rc "0" "${rc}" "setup_model returns success when marker is present"
    assert_match "/baseline$" "${out}" "returns baseline path on success"
    assert_dir_exists "${out}" "baseline directory created"
}

test_pack_validation_estimate_model_params_defaults_to_7_without_config() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh

    local model_dir="${TEST_TMPDIR}/model"
    mkdir -p "${model_dir}"
    assert_eq "7" "$(estimate_model_params "${model_dir}")" "missing config defaults to 7B"
}

test_pack_validation_estimate_model_params_classifies_when_config_is_present() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh

    local model_dir="${TEST_TMPDIR}/model"
    mkdir -p "${model_dir}"
    cat > "${model_dir}/config.json" <<'EOF'
{"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32000}
EOF

    assert_eq "7" "$(estimate_model_params "${model_dir}")" "config-based estimation classifies small bucket"
}

test_pack_validation_get_model_invarlock_config_covers_all_case_arms() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh

    assert_eq "2048:1024:64:64:96" "$(get_model_invarlock_config 7)" "7B config"
    assert_eq "1536:768:48:48:64" "$(get_model_invarlock_config 13)" "13B config"
    assert_eq "1024:512:40:40:48" "$(get_model_invarlock_config 30)" "30B config"
    assert_eq "1024:512:36:36:32" "$(get_model_invarlock_config 40)" "40B config"
    assert_eq "1024:512:40:40:24" "$(get_model_invarlock_config moe)" "moe config"
    assert_eq "128:64:8:8:2" "$(get_model_invarlock_config 70)" "70B config"
    assert_eq "1024:512:40:40:32" "$(get_model_invarlock_config unknown)" "default config"
}

test_pack_validation_create_edited_model_quant_rtn_and_unknown_edit_type_branches() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    fixture_write "python3.stub" ""

    create_edited_model "${TEST_TMPDIR}/baseline" "${TEST_TMPDIR}/edited" "quant_rtn" "8" "128" "ffn" "0"

    error_exit() { exit 4; }
    local rc=0
    ( create_edited_model "${TEST_TMPDIR}/baseline" "${TEST_TMPDIR}/edited" "nope" "8" "128" "ffn" "0" ) || rc=$?
    assert_eq "4" "${rc}" "unknown edit type aborts via error_exit"
}

test_pack_validation_run_lmeval_batch_size_selection_flash_attention_and_results_handling() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    fixture_write "python3.stub" ""

    estimate_model_params() { echo "${MODEL_SIZE_RETURN}"; }

    CUDA_VISIBLE_DEVICES="0,1"
    LM_EVAL_PARALLELIZE="true"
    FLASH_ATTENTION_AVAILABLE="true"

    local sizes=("70" "40" "30" "moe" "7")
    local size
    for size in "${sizes[@]}"; do
        MODEL_SIZE_RETURN="${size}"
        local dir="${TEST_TMPDIR}/eval_${size}"
        mkdir -p "${dir}"
        printf '{"ok":true}\n' > "${dir}/results.json"
        run_lmeval "${TEST_TMPDIR}/model_${size}" "${dir}/out.json" "mmlu" "auto" "0" "0"
        assert_file_exists "${dir}/out.json" "results moved into output file"
    done

    # Flash Attention disabled pattern branch
    MODEL_SIZE_RETURN="7"
    local falcon_dir="${TEST_TMPDIR}/eval_falcon"
    mkdir -p "${falcon_dir}"
    printf '{"ok":true}\n' > "${falcon_dir}/results.json"
    run_lmeval "falcon-model" "${falcon_dir}/out.json" "mmlu" "auto" "0" "0"
}

test_pack_validation_run_lmeval_logs_warning_when_results_missing() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    fixture_write "python3.stub" ""
    estimate_model_params() { echo "7"; }

    local dir="${TEST_TMPDIR}/eval_empty"
    mkdir -p "${dir}"
    run run_lmeval "${TEST_TMPDIR}/model" "${dir}/out.json" "mmlu" "auto" "0" "0"
    assert_ne "0" "${RUN_RC}" "missing results returns non-zero"
}

test_pack_validation_run_lmeval_returns_nonzero_when_results_move_fails() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    fixture_write "python3.stub" ""
    estimate_model_params() { echo "7"; }

    local dir="${TEST_TMPDIR}/eval_move_fail"
    mkdir -p "${dir}"
    printf '{"ok":true}\n' > "${dir}/results.json"

    mv() { return 1; }
    run run_lmeval "${TEST_TMPDIR}/model" "${dir}/out.json" "mmlu" "auto" "0" "0"
    assert_ne "0" "${RUN_RC}" "results move failure returns non-zero"
}

test_pack_validation_generate_invarlock_config_attn_and_strict_accelerator_flags() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh

    FLASH_ATTENTION_AVAILABLE="true"
    PACK_DETERMINISM="strict"
    local cfg="${TEST_TMPDIR}/cfg.yaml"
    generate_invarlock_config "model" "${cfg}" "edit"
    assert_file_exists "${cfg}" "config generated"
    assert_match "flash_attention_2" "$(cat "${cfg}")" "attn implementation emitted"

    FLASH_ATTENTION_AVAILABLE="false"
    PACK_DETERMINISM="throughput"
    generate_invarlock_config "model" "${cfg}" "edit"
    assert_match "flash_attention_2 not available" "$(cat "${cfg}")" "comment emitted when FA2 unavailable"
}

test_pack_validation_run_single_calibration_large_model_and_report_copy_branch() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    fixture_write "python3.stub" ""
    fixture_write "invarlock.create_report" ""
    estimate_model_params() { echo "${MODEL_SIZE_RETURN}"; }

    local run_dir="${TEST_TMPDIR}/cal/run1"
    local log_file="${TEST_TMPDIR}/cal/run.log"
    MODEL_SIZE_RETURN="70"
    run_single_calibration "model" "${run_dir}" "42" "1" "1" "1" "${log_file}" "0"
    assert_file_exists "${run_dir}/baseline_report.json" "report copied when present"

    MODEL_SIZE_RETURN="7"
    run_single_calibration "model" "${TEST_TMPDIR}/cal/run2" "42" "1" "1" "1" "${log_file}" "0"
}

test_pack_validation_run_invarlock_calibration_failure_paths_and_labels() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    fixture_write "python3.stub" ""

    estimate_model_params() { echo "${MODEL_SIZE_RETURN}"; }

    MODEL_SIZE_RETURN="moe"
    run_single_calibration() { return 0; }
    run_invarlock_calibration "model" "mixtral" "${TEST_TMPDIR}/cal/moe" "1" "${TEST_TMPDIR}/presets" "0"

    MODEL_SIZE_RETURN="7"
    run_single_calibration() { return 1; }
    local rc=0
    ( run_invarlock_calibration "model" "small" "${TEST_TMPDIR}/cal/fail" "2" "${TEST_TMPDIR}/presets" "0" ) || rc=$?
    assert_ne "0" "${rc}" "all calibration runs failing returns non-zero"
}

test_pack_validation_run_invarlock_certify_preset_optional_and_cert_copy_paths() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    fixture_write "python3.stub" ""

    local preset_dir="${TEST_TMPDIR}/presets"
    mkdir -p "${preset_dir}"
    : > "${preset_dir}/calibrated_preset_model.yaml"

    estimate_model_params() { echo "${MODEL_SIZE_RETURN}"; }

    fixture_write "invarlock.create_cert" ""
    MODEL_SIZE_RETURN="70"
    run_invarlock_certify "subject" "baseline" "${TEST_TMPDIR}/certs" "run_ok" "${preset_dir}" "model" "0"

    # alt-cert path when canonical cert missing
    rm -f "${TEST_TMPDIR}/fixtures/invarlock.create_cert"
    local cert_dir="${TEST_TMPDIR}/certs/run_alt/cert/nested"
    mkdir -p "${cert_dir}"
    printf '{"ok":true}\n' > "${cert_dir}/evaluation.cert.json"
    MODEL_SIZE_RETURN="7"
    run_invarlock_certify "subject" "baseline" "${TEST_TMPDIR}/certs" "run_alt" "${preset_dir}" "model" "0"
}

test_pack_validation_process_model_branches_across_baseline_edits_and_error_injection() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    setup_model() { return 1; }
    local rc=0
    ( process_model "bad/model" "0" ) || rc=$?
    assert_ne "0" "${rc}" "baseline setup failure returns non-zero"

    # Happy path: stub heavy steps but still exercise orchestration branches.
    setup_model() { mkdir -p "${TEST_TMPDIR}/baseline"; echo "${TEST_TMPDIR}/baseline"; }
    run_lmeval() { :; }
    run_invarlock_calibration() { :; }
    process_edit() { mkdir -p "${TEST_TMPDIR}/editdir"; echo "${TEST_TMPDIR}/editdir"; }
    run_invarlock_certify() { :; }
    create_error_model() { mkdir -p "${2}"; : > "${2}/config.json"; }

    local model_name
    model_name="$(sanitize_model_name "ok/model")"
    local model_output_dir="${OUTPUT_DIR}/${model_name}"
    mkdir -p "${model_output_dir}/certificates/errors/nan_injection"
    printf '{"ok":true}\n' > "${model_output_dir}/certificates/errors/nan_injection/evaluation.cert.json"

    process_model "ok/model" "0"
}

test_pack_validation_main_dynamic_resume_and_monitoring_branches_offline() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    # Stub early heavyweight phases.
    check_dependencies() { :; }
    configure_gpu_pool() { NUM_GPUS=2; GPU_ID_LIST="0,1"; export NUM_GPUS GPU_ID_LIST; }
    disk_preflight() { :; }
    setup_pack_environment() { :; }
    handle_disk_pressure() { echo "disk_pressure:$1:$2" >> "${TEST_TMPDIR}/disk_pressure.calls"; return 0; }

    # Existing queue with tasks for --resume branch coverage.
    RESUME_FLAG="true"
    mkdir -p "${OUTPUT_DIR}/queue"/{pending,ready,running,completed,failed}
    printf '{"id":"t1","status":"running"}\n' > "${OUTPUT_DIR}/queue/running/t1.task"
    printf '{"id":"t2","status":"failed"}\n' > "${OUTPUT_DIR}/queue/failed/t2.task"

    init_queue() {
        QUEUE_DIR="${OUTPUT_DIR}/queue"
        GPU_RESERVATION_DIR="${OUTPUT_DIR}/gpu_reservations"
        mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed} "${OUTPUT_DIR}/workers" "${GPU_RESERVATION_DIR}"
        export QUEUE_DIR GPU_RESERVATION_DIR
    }
    init_gpu_reservations() { GPU_RESERVATION_DIR="${OUTPUT_DIR}/gpu_reservations"; export GPU_RESERVATION_DIR; }
    refresh_task_memory_from_profiles() { echo "refresh:$1" >> "${TEST_TMPDIR}/mem.calls"; }
    export_memory_plan() { echo "plan:$1" >> "${TEST_TMPDIR}/mem.calls"; }
    resolve_dependencies() { echo 1; }
    cancel_tasks_with_failed_dependencies() { echo 2; }
    get_queue_stats() { echo ""; }
    apply_work_stealing_boost() { :; }
    reclaim_orphaned_tasks() { echo "reclaim:$1" >> "${TEST_TMPDIR}/reclaim.calls"; }
    count_tasks() {
        if [[ "${1:-}" == "failed" ]]; then
            echo 1
        else
            echo 0
        fi
    }
    print_queue_stats() { :; }
    compile_results() { :; }
    run_analysis() { :; }
    generate_verdict() { :; }

    list_run_gpu_ids() { printf '0\n1\n'; }

    local empty_checks=0
    is_queue_empty() {
        empty_checks=$((empty_checks + 1))
        [[ ${empty_checks} -ge 2 ]]
    }

    # Stub worker scripts so start_worker doesn't run real gpu_worker.
    local stub_lib="${TEST_TMPDIR}/stub_lib"
    mkdir -p "${stub_lib}"
    for f in task_serialization.sh queue_manager.sh scheduler.sh task_functions.sh fault_tolerance.sh; do
        printf '%s\n' "#!/usr/bin/env bash" > "${stub_lib}/${f}"
    done
    cat > "${stub_lib}/gpu_worker.sh" <<'EOF'
#!/usr/bin/env bash
gpu_worker() {
    local gpu_id="$1"
    local output_dir="$2"
    mkdir -p "${output_dir}/workers"
    echo "searching" > "${output_dir}/workers/gpu_${gpu_id}.status"
    : > "${output_dir}/workers/gpu_${gpu_id}.heartbeat"
    if [[ "${gpu_id}" == "1" ]]; then
        return 1
    fi
    return 0
}
EOF

    LIB_DIR="${stub_lib}"
    export LIB_DIR

    WORKER_TIMEOUT=1
    MIN_FREE_DISK_GB="bogus"
    get_free_disk_gb() { echo "1"; }

    # Ensure heartbeat appears stale for GPU 0 so "stuck" branch triggers.
    fixture_append "stat/mtime" "$(printf '%s %s\n' "${OUTPUT_DIR}/workers/gpu_0.heartbeat" "1699990000")"

    kill() {
        local sig="${1:-}"
        local pid="${2:-}"
        if [[ "${sig}" == "-0" ]]; then
            local pid0
            pid0="$(cat "${OUTPUT_DIR}/workers/gpu_0.pid" 2>/dev/null || echo "")"
            [[ -n "${pid0}" && "${pid}" == "${pid0}" ]] && return 0
            return 1
        fi
        return 0
    }

    signal_shutdown() { echo "shutdown:$1" >> "${TEST_TMPDIR}/shutdown.calls"; }

    main_dynamic
    assert_file_exists "${TEST_TMPDIR}/shutdown.calls" "signal_shutdown called on empty queue"
}

test_pack_validation_main_dynamic_fresh_task_generation_and_touch_shutdown_branch() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    check_dependencies() { :; }
    configure_gpu_pool() { NUM_GPUS=1; GPU_ID_LIST="0"; export NUM_GPUS GPU_ID_LIST; }
    disk_preflight() { :; }
    setup_pack_environment() { :; }
    handle_disk_pressure() { return 0; }

    RESUME_FLAG="false"

    init_queue() {
        QUEUE_DIR="${OUTPUT_DIR}/queue"
        mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed} "${OUTPUT_DIR}/workers"
        export QUEUE_DIR
    }
    generate_all_tasks() { echo "generated" >> "${TEST_TMPDIR}/tasks.calls"; }
    resolve_dependencies() { echo 0; }
    count_tasks() { echo 0; }
    print_queue_stats() { :; }
    compile_results() { :; }
    run_analysis() { :; }
    generate_verdict() { :; }

    list_run_gpu_ids() { printf '0\n'; }
    is_queue_empty() { return 0; }

    local stub_lib="${TEST_TMPDIR}/stub_lib"
    mkdir -p "${stub_lib}"
    for f in task_serialization.sh queue_manager.sh scheduler.sh task_functions.sh fault_tolerance.sh; do
        printf '%s\n' "#!/usr/bin/env bash" > "${stub_lib}/${f}"
    done
    cat > "${stub_lib}/gpu_worker.sh" <<'EOF'
#!/usr/bin/env bash
gpu_worker() { return 0; }
EOF
    LIB_DIR="${stub_lib}"
    export LIB_DIR

    MIN_FREE_DISK_GB="bogus"
    get_free_disk_gb() { echo "999"; }

    main_dynamic
    assert_file_exists "${TEST_TMPDIR}/tasks.calls" "fresh run generates tasks"
    assert_file_exists "${OUTPUT_DIR}/workers/SHUTDOWN" "touch shutdown when signal_shutdown missing"
}

test_pack_validation_main_dynamic_calibrate_only_stops_after_presets_even_with_pending_tasks() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    export OUTPUT_DIR
    export PACK_SUITE_MODE="calibrate-only"

    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    check_dependencies() { :; }
    configure_gpu_pool() { NUM_GPUS=1; GPU_ID_LIST="0"; export NUM_GPUS GPU_ID_LIST; }
    disk_preflight() { :; }
    setup_pack_environment() { :; }
    handle_disk_pressure() { return 0; }

    RESUME_FLAG="true"

    init_queue() {
        QUEUE_DIR="${OUTPUT_DIR}/queue"
        mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed} "${OUTPUT_DIR}/workers"
        export QUEUE_DIR
    }
    resolve_dependencies() { echo 0; }
    reclaim_orphaned_tasks() { :; }
    print_queue_stats() { :; }
    count_tasks() { echo 0; }

    # Ensure queue is NOT empty (pending contains non-calibration task), but
    # calibration-only early exit triggers because all preset tasks are completed.
    init_queue
    printf '{"status":"pending"}\n' > "${QUEUE_DIR}/pending/model_EVAL_BASELINE_001_dead.task"
    printf '{"status":"completed","task_type":"GENERATE_PRESET"}\n' > "${QUEUE_DIR}/completed/model_GENERATE_PRESET_001_beef.task"

    list_run_gpu_ids() { printf '0\n'; }

    # Stub worker scripts so start_worker doesn't run real gpu_worker.
    local stub_lib="${TEST_TMPDIR}/stub_lib"
    mkdir -p "${stub_lib}"
    for f in task_serialization.sh queue_manager.sh scheduler.sh task_functions.sh fault_tolerance.sh; do
        printf '%s\n' "#!/usr/bin/env bash" > "${stub_lib}/${f}"
    done
    cat > "${stub_lib}/gpu_worker.sh" <<'EOF'
#!/usr/bin/env bash
gpu_worker() { return 0; }
EOF
    LIB_DIR="${stub_lib}"
    export LIB_DIR

    compile_results() { echo "compile" >> "${TEST_TMPDIR}/analysis.calls"; }
    run_analysis() { echo "analysis" >> "${TEST_TMPDIR}/analysis.calls"; }
    generate_verdict() { echo "verdict" >> "${TEST_TMPDIR}/analysis.calls"; }

    signal_shutdown() { echo "shutdown:$1" >> "${TEST_TMPDIR}/shutdown.calls"; }
    get_free_disk_gb() { echo "999"; }

    main_dynamic

    assert_file_exists "${TEST_TMPDIR}/shutdown.calls" "calibration-only run signals shutdown early"
    [[ ! -f "${TEST_TMPDIR}/analysis.calls" ]] || t_fail "analysis should not run for calibration-only mode"
}

test_pack_validation_main_wrapper_parses_progress_and_reports_failed_tasks_offline() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    log() { :; }
    log_section() { :; }

    check_dependencies() { :; }
    configure_gpu_pool() { NUM_GPUS=1; GPU_ID_LIST="0"; export NUM_GPUS GPU_ID_LIST; }
    disk_preflight() { :; }
    setup_pack_environment() { :; }

    RESUME_FLAG="false"
    init_queue() {
        QUEUE_DIR="${OUTPUT_DIR}/queue"
        GPU_RESERVATION_DIR="${OUTPUT_DIR}/gpu_reservations"
        mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed} "${OUTPUT_DIR}/workers" "${GPU_RESERVATION_DIR}"
        export QUEUE_DIR GPU_RESERVATION_DIR
    }
    init_gpu_reservations() { GPU_RESERVATION_DIR="${OUTPUT_DIR}/gpu_reservations"; export GPU_RESERVATION_DIR; }
    generate_all_tasks() { :; }

    resolve_dependencies() { echo 0; }
    get_queue_stats() { echo "1:2:3:4:5:6"; }
    apply_work_stealing_boost() { echo "boost" >> "${TEST_TMPDIR}/boost.calls"; }

    mkdir -p "${OUTPUT_DIR}/queue/failed"
    : > "${OUTPUT_DIR}/queue/failed/t2.task"
    get_task_id() { echo "${1}" >> "${TEST_TMPDIR}/task_id.calls"; echo "t2"; }
    get_task_field() { echo "${1}:${2}" >> "${TEST_TMPDIR}/task_field.calls"; echo "boom"; }

    count_tasks() { [[ "${1:-}" == "failed" ]] && echo 1 || echo 0; }
    print_queue_stats() { :; }
    list_run_gpu_ids() { printf '0\n'; }

    local empty_checks=0
    is_queue_empty() {
        empty_checks=$((empty_checks + 1))
        [[ ${empty_checks} -ge 2 ]]
    }
    get_free_disk_gb() { echo "999"; }

    python3() {
        echo "python3 $*" >> "${TEST_TMPDIR}/python3.calls"
        case "${1:-}" in
            -c|-m) return 0 ;;
        esac
        cat >/dev/null || true
        return 0
    }

    local stub_lib="${TEST_TMPDIR}/stub_lib"
    mkdir -p "${stub_lib}"
    for f in task_serialization.sh queue_manager.sh scheduler.sh task_functions.sh fault_tolerance.sh; do
        printf '%s\n' "#!/usr/bin/env bash" > "${stub_lib}/${f}"
    done
    cat > "${stub_lib}/gpu_worker.sh" <<'EOF'
#!/usr/bin/env bash
gpu_worker() { return 0; }
EOF
    LIB_DIR="${stub_lib}"
    export LIB_DIR

    kill() { return 0; }

    run main
    assert_rc "0" "${RUN_RC}" "main completes offline"
    assert_file_exists "${TEST_TMPDIR}/boost.calls" "progress path applies work-stealing boost"
    assert_file_exists "${TEST_TMPDIR}/task_id.calls" "failed task reporting reads task ids"
    assert_file_exists "${TEST_TMPDIR}/python3.calls" "analysis steps invoke python3"
}

test_pack_validation_setup_output_dirs_returns_nonzero_when_output_dir_is_file() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh

    OUTPUT_DIR="${TEST_TMPDIR}/out_file"
    echo "not a dir" > "${OUTPUT_DIR}"

    run pack_setup_output_dirs
    assert_rc "1" "${RUN_RC}" "mkdir failure propagates as non-zero"
}


test_pack_validation_pack_output_dir_defaults_to_pack_output_dir() {
    mock_reset

    unset OUTPUT_DIR
    PACK_OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh

    assert_eq "${PACK_OUTPUT_DIR}" "${OUTPUT_DIR}" "PACK_OUTPUT_DIR seeds OUTPUT_DIR"
    unset PACK_OUTPUT_DIR OUTPUT_DIR
}


test_pack_validation_pack_model_list_and_revisions_branches() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    mkdir -p "${OUTPUT_DIR}/state"
    source ./scripts/proof_packs/lib/validation_suite.sh

    MODEL_1="org/model1"
    MODEL_2=""
    pack_model_list_array
    assert_eq "org/model1" "${PACK_MODEL_LIST[0]}" "fallback populates model list"

    mapfile() {
        local flag="$1"
        local target="$2"
        local -a values=()
        while IFS= read -r line; do
            values+=("${line}")
        done
        if [[ "${flag}" == "-t" ]]; then
            eval "${target}=()"
            local value
            for value in "${values[@]}"; do
                eval "${target}+=(\"${value}\")"
            done
        fi
    }
    PACK_MODEL_LIST=()
    pack_model_list_array
    assert_eq "org/model1" "${PACK_MODEL_LIST[0]}" "mapfile populates model list"
    unset -f mapfile

    rm -f "${OUTPUT_DIR}/state/model_revisions.json"
    run pack_load_model_revisions
    assert_rc "1" "${RUN_RC}" "missing revisions file returns non-zero"

    run pack_model_revision "org/model1"
    assert_rc "1" "${RUN_RC}" "missing revisions file returns non-zero"

    echo '{"models":{"org/model1":{"revision":"abc"}}}' > "${OUTPUT_DIR}/state/model_revisions.json"
    run pack_load_model_revisions
    assert_rc "0" "${RUN_RC}" "load revisions succeeds"
    assert_eq "${OUTPUT_DIR}/state/model_revisions.json" "${PACK_MODEL_REVISIONS_FILE}" "revisions file set"

    run pack_model_revision "org/model1"
    assert_rc "0" "${RUN_RC}" "revision lookup succeeds"
    assert_eq "abc" "${RUN_OUT}" "revision returned"

    echo '{"models":{"org/model1":{"gated":true}}}' > "${OUTPUT_DIR}/state/model_revisions.json"
    run pack_load_model_revisions
    assert_rc "1" "${RUN_RC}" "gated model revisions fail"
}


test_pack_validation_preflight_models_error_branches() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh

    fixture_write "python3.stub" ""
    fixture_write "python3.rc" "0"

    local error_log="${TEST_TMPDIR}/error.msg"
    : > "${error_log}"
    error_exit() { echo "$1" >> "${error_log}"; return 0; }

    PACK_NET="0"
    run pack_preflight_models "${OUTPUT_DIR}" "org/model"
    assert_match "Preflight requires" "$(cat "${error_log}")" "preflight requires net"

    PACK_NET="1"
    set +u
    run pack_preflight_models "${OUTPUT_DIR}"
    set -u
    assert_match "No models provided" "$(cat "${error_log}")" "preflight requires models"
}


test_pack_validation_setup_hf_cache_dirs_requires_output_dir() {
    mock_reset

    OUTPUT_DIR=""
    PACK_OUTPUT_DIR=""
    source ./scripts/proof_packs/lib/validation_suite.sh

    run pack_setup_hf_cache_dirs
    assert_rc "1" "${RUN_RC}" "missing OUTPUT_DIR fails"
}


test_pack_validation_estimate_planned_model_storage_mapfile() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh

    EDIT_TYPES_CLEAN=("quant_rtn:clean:ffn")
    EDIT_TYPES_STRESS=()
    RUN_ERROR_INJECTION="false"

    pack_model_list() { printf '%s\n' "org/model"; }
    estimate_model_weights_gb() { echo "10"; }

    mapfile() {
        local flag="$1"
        local target="$2"
        local -a values=()
        while IFS= read -r line; do
            values+=("${line}")
        done
        if [[ "${flag}" == "-t" ]]; then
            eval "${target}=()"
            local value
            for value in "${values[@]}"; do
                eval "${target}+=(\"${value}\")"
            done
        fi
    }

    local total
    total="$(estimate_planned_model_storage_gb)"
    unset -f mapfile
    assert_eq "20" "${total}" "planned storage sums weights and edits"
}


test_pack_validation_setup_model_revision_branches() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    LOG_FILE="${TEST_TMPDIR}/log.txt"
    : > "${LOG_FILE}"
    source ./scripts/proof_packs/lib/validation_suite.sh

    error_exit() { echo "$1" > "${TEST_TMPDIR}/error.msg"; return 1; }

    pack_model_revision() { echo ""; }
    PACK_NET="1"
    run setup_model "org/model" "0"
    assert_rc "1" "${RUN_RC}" "missing revision fails with net"

    PACK_NET="0"
    run setup_model "org/model" "0"
    assert_rc "1" "${RUN_RC}" "missing revision fails offline"

    pack_model_revision() { echo "rev1"; }
    PACK_NET="0"
    run setup_model "org/model" "0"
    assert_rc "1" "${RUN_RC}" "offline missing cache fails"
}


test_pack_validation_process_edit_skipped_status() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh

    resolve_edit_params() { jq -n '{status:"skipped"}'; }
    log() { :; }

    run process_edit "baseline" "quant_rtn:clean:ffn" "clean" "model" "0" "${TEST_TMPDIR}/out"
    assert_rc "0" "${RUN_RC}" "skipped edit returns success"
    assert_eq "" "${RUN_OUT}" "skipped edit returns empty output"
}


test_pack_validation_generate_invarlock_config_guard_order() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh

    local cfg="${TEST_TMPDIR}/cfg.yaml"
    PACK_GUARDS_ORDER="variance,invariants"
    generate_invarlock_config "model" "${cfg}" "edit"
    assert_match "variance" "$(cat "${cfg}")" "guard order uses csv"

    PACK_GUARDS_ORDER=" , "
    generate_invarlock_config "model" "${cfg}" "edit"
    assert_match "invariants" "$(cat "${cfg}")" "guard order defaults when empty"
}


test_pack_validation_run_determinism_repeats_branch_coverage() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    OUTPUT_DIR=""
    PACK_REPEATS="1"
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "requires OUTPUT_DIR"

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    pack_setup_output_dirs

    PACK_REPEATS="0"
    run pack_run_determinism_repeats
    assert_rc "0" "${RUN_RC}" "zero repeats returns success"

    PACK_REPEATS="bad"
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "invalid repeats fails"

    PACK_REPEATS="1"
    PACK_MODEL_LIST=()
    pack_model_list() { :; }
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "missing models fails"

    pack_model_list() { printf '%s\n' "org/model"; }
    PACK_MODEL_LIST=()
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "missing baseline path fails"

    local model_id="org/model"
    local model_name
    model_name="$(sanitize_model_name "${model_id}")"
    local model_output_dir="${OUTPUT_DIR}/${model_name}"
    mkdir -p "${model_output_dir}"
    local baseline_dir="${TEST_TMPDIR}/baseline"
    mkdir -p "${baseline_dir}"
    echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"

    PACK_MODEL_LIST=("${model_id}")
    PACK_REPEATS="1"

    declare -a EDIT_TYPES_CLEAN=()
    declare -a EDIT_TYPES_STRESS=()
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "missing edit specs fails"

    declare -a EDIT_TYPES_CLEAN=()
    EDIT_TYPES_STRESS=("quant_rtn:4:32:ffn")
    process_edit() { return 1; }
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "process_edit failure returns non-zero"

    EDIT_TYPES_CLEAN=("quant_rtn:clean:ffn")
    EDIT_TYPES_STRESS=()
    process_edit() { echo ""; return 0; }
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "empty edit path fails"

    process_edit() { return 1; }
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "process_edit failure returns non-zero"

    process_edit() { mkdir -p "${TEST_TMPDIR}/edit2"; echo "${TEST_TMPDIR}/edit2"; }
    run_invarlock_certify() { return 1; }
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "certify failure returns non-zero"

    process_edit() { mkdir -p "${TEST_TMPDIR}/edit3"; echo "${TEST_TMPDIR}/edit3"; }
    run_invarlock_certify() { return 0; }
    mkdir() {
        for arg in "$@"; do
            if [[ "${arg}" == *"/determinism/"* ]]; then
                return 1
            fi
        done
        command mkdir "$@"
    }
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "determinism mkdir failure returns non-zero"
    unset -f mkdir

    process_edit() { mkdir -p "${TEST_TMPDIR}/edit4"; echo "${TEST_TMPDIR}/edit4"; }
    run_invarlock_certify() { return 0; }
    mkdir() {
        for arg in "$@"; do
            if [[ "${arg}" == *"/analysis" ]]; then
                return 1
            fi
        done
        command mkdir "$@"
    }
    run pack_run_determinism_repeats
    assert_rc "1" "${RUN_RC}" "analysis mkdir failure returns non-zero"
    unset -f mkdir
}


test_pack_validation_source_libs_prefers_lib_subdir() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh

    local root="${TEST_TMPDIR}/pkg"
    mkdir -p "${root}/lib"
    for f in task_serialization.sh queue_manager.sh scheduler.sh task_functions.sh gpu_worker.sh; do
        printf '%s\n' "#!/usr/bin/env bash" > "${root}/lib/${f}"
    done

    _pack_script_dir() { echo "${root}"; }
    pack_source_libs
    assert_eq "${root}/lib" "${LIB_DIR}" "lib subdir selected"
}


test_pack_validation_source_libs_uses_parent_lib_dir() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh

    local root="${TEST_TMPDIR}/pkg"
    mkdir -p "${root}/lib" "${root}/child"
    for f in task_serialization.sh queue_manager.sh scheduler.sh task_functions.sh gpu_worker.sh; do
        printf '%s\n' "#!/usr/bin/env bash" > "${root}/lib/${f}"
    done

    _pack_script_dir() { echo "${root}/child"; }
    pack_source_libs
    assert_eq "${root}/lib" "${LIB_DIR}" "parent lib dir selected"
}


test_pack_validation_process_model_clean_and_stress_skip_branches() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    setup_model() { mkdir -p "${TEST_TMPDIR}/baseline"; echo "${TEST_TMPDIR}/baseline"; }
    run_lmeval() { :; }
    run_invarlock_calibration() { :; }
    process_edit() { :; }
    run_invarlock_certify() { :; }
    create_error_model() { :; }

    EDIT_TYPES_CLEAN=("quant_rtn:clean:ffn")
    EDIT_TYPES_STRESS=("quant_rtn:4:32:ffn")
    RUN_ERROR_INJECTION="false"

    CALIBRATE_CLEAN_EDITS="true"
    CLEAN_EDIT_RUNS=1
    STRESS_EDIT_RUNS=1
    task_calibrate_clean_edits() { echo "calibrate" > "${TEST_TMPDIR}/cal.calls"; }
    process_model "model/a" "0"
    assert_file_exists "${TEST_TMPDIR}/cal.calls" "clean calibration invoked"

    CLEAN_EDIT_RUNS=0
    STRESS_EDIT_RUNS=1
    process_model "model/b" "0"

    CLEAN_EDIT_RUNS=1
    STRESS_EDIT_RUNS=0
    process_model "model/c" "0"
}


test_pack_validation_main_dynamic_demote_ready_tasks_for_calibration_only() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    check_dependencies() { :; }
    configure_gpu_pool() { NUM_GPUS=1; GPU_ID_LIST="0"; export NUM_GPUS GPU_ID_LIST; }
    disk_preflight() { :; }
    setup_pack_environment() { :; }
    handle_disk_pressure() { return 0; }
    RESUME_FLAG="false"

    init_queue() {
        QUEUE_DIR="${OUTPUT_DIR}/queue"
        mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed} "${OUTPUT_DIR}/workers"
        export QUEUE_DIR
    }
    generate_all_tasks() { :; }
    resolve_dependencies() { echo 0; }
    demote_ready_tasks_for_calibration_only() { echo "demote" > "${TEST_TMPDIR}/demote.calls"; }
    count_tasks() { echo 0; }
    print_queue_stats() { :; }
    compile_results() { :; }
    run_analysis() { :; }
    generate_verdict() { :; }
    list_run_gpu_ids() { printf '0\n'; }

    local empty_checks=0
    is_queue_empty() {
        empty_checks=$((empty_checks + 1))
        [[ ${empty_checks} -ge 1 ]]
    }
    get_free_disk_gb() { echo "999"; }

    local stub_lib="${TEST_TMPDIR}/stub_lib"
    mkdir -p "${stub_lib}"
    for f in task_serialization.sh queue_manager.sh scheduler.sh task_functions.sh fault_tolerance.sh; do
        printf '%s\n' "#!/usr/bin/env bash" > "${stub_lib}/${f}"
    done
    cat > "${stub_lib}/gpu_worker.sh" <<'EOF'
#!/usr/bin/env bash
gpu_worker() { return 0; }
EOF
    LIB_DIR="${stub_lib}"
    export LIB_DIR

    main_dynamic
    assert_file_exists "${TEST_TMPDIR}/demote.calls" "demote_ready_tasks_for_calibration_only invoked"
}


test_pack_validation_main_dynamic_calibrate_only_without_signal_shutdown() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    export OUTPUT_DIR
    export PACK_SUITE_MODE="calibrate-only"

    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    check_dependencies() { :; }
    configure_gpu_pool() { NUM_GPUS=1; GPU_ID_LIST="0"; export NUM_GPUS GPU_ID_LIST; }
    disk_preflight() { :; }
    setup_pack_environment() { :; }
    handle_disk_pressure() { return 0; }

    RESUME_FLAG="true"

    init_queue() {
        QUEUE_DIR="${OUTPUT_DIR}/queue"
        mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed} "${OUTPUT_DIR}/workers"
        export QUEUE_DIR
    }
    resolve_dependencies() { echo 0; }
    reclaim_orphaned_tasks() { :; }
    print_queue_stats() { :; }
    count_tasks() { echo 0; }

    init_queue
    printf '{"status":"pending"}\n' > "${QUEUE_DIR}/pending/model_EVAL_BASELINE_001_dead.task"
    printf '{"status":"completed","task_type":"GENERATE_PRESET"}\n' > "${QUEUE_DIR}/completed/model_GENERATE_PRESET_001_beef.task"

    list_run_gpu_ids() { printf '0\n'; }

    local stub_lib="${TEST_TMPDIR}/stub_lib"
    mkdir -p "${stub_lib}"
    for f in task_serialization.sh queue_manager.sh scheduler.sh task_functions.sh fault_tolerance.sh; do
        printf '%s\n' "#!/usr/bin/env bash" > "${stub_lib}/${f}"
    done
    cat > "${stub_lib}/gpu_worker.sh" <<'EOF'
#!/usr/bin/env bash
gpu_worker() { return 0; }
EOF
    LIB_DIR="${stub_lib}"
    export LIB_DIR

    compile_results() { :; }
    run_analysis() { :; }
    generate_verdict() { :; }
    get_free_disk_gb() { echo "999"; }

    main_dynamic
    assert_file_exists "${OUTPUT_DIR}/workers/SHUTDOWN" "touch shutdown when signal_shutdown missing"
}


test_pack_validation_main_dynamic_warns_on_determinism_repeats_failure() {
    mock_reset

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    source ./scripts/proof_packs/lib/validation_suite.sh
    pack_setup_output_dirs

    check_dependencies() { :; }
    configure_gpu_pool() { NUM_GPUS=1; GPU_ID_LIST="0"; export NUM_GPUS GPU_ID_LIST; }
    disk_preflight() { :; }
    setup_pack_environment() { :; }
    handle_disk_pressure() { return 0; }
    RESUME_FLAG="false"

    init_queue() {
        QUEUE_DIR="${OUTPUT_DIR}/queue"
        mkdir -p "${QUEUE_DIR}"/{pending,ready,running,completed,failed} "${OUTPUT_DIR}/workers"
        export QUEUE_DIR
    }
    generate_all_tasks() { :; }
    resolve_dependencies() { echo 0; }
    count_tasks() { echo 0; }
    print_queue_stats() { :; }
    compile_results() { :; }
    run_analysis() { :; }
    generate_verdict() { :; }
    list_run_gpu_ids() { printf '0\n'; }

    local empty_checks=0
    is_queue_empty() {
        empty_checks=$((empty_checks + 1))
        [[ ${empty_checks} -ge 1 ]]
    }
    get_free_disk_gb() { echo "999"; }

    PACK_REPEATS="1"
    pack_run_determinism_repeats() { echo "repeats" > "${TEST_TMPDIR}/repeats.calls"; return 1; }
    log() { echo "$*" >> "${TEST_TMPDIR}/log.msg"; }
    log_section() { :; }

    local stub_lib="${TEST_TMPDIR}/stub_lib"
    mkdir -p "${stub_lib}"
    for f in task_serialization.sh queue_manager.sh scheduler.sh task_functions.sh fault_tolerance.sh; do
        printf '%s\n' "#!/usr/bin/env bash" > "${stub_lib}/${f}"
    done
    cat > "${stub_lib}/gpu_worker.sh" <<'EOF'
#!/usr/bin/env bash
gpu_worker() { return 0; }
EOF
    LIB_DIR="${stub_lib}"
    export LIB_DIR

    main_dynamic
    assert_file_exists "${TEST_TMPDIR}/repeats.calls" "determinism repeats invoked"
}


test_pack_validation_pack_run_suite_branches() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh

    cleanup() { return 0; }
    pack_apply_network_mode() { :; }

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    PACK_NET="0"

    pack_require_bash4() { return 1; }
    run pack_run_suite
    assert_rc "1" "${RUN_RC}" "bash4 requirement enforced"
    trap - EXIT INT TERM HUP QUIT

    pack_require_bash4() { return 0; }
    OUTPUT_DIR=""
    run pack_run_suite
    assert_rc "1" "${RUN_RC}" "missing output dir fails"
    trap - EXIT INT TERM HUP QUIT

    OUTPUT_DIR="${TEST_TMPDIR}/out"
    pack_source_libs() { return 1; }
    run pack_run_suite
    assert_rc "1" "${RUN_RC}" "pack_source_libs failure returns non-zero"
    trap - EXIT INT TERM HUP QUIT

    pack_source_libs() { return 0; }
    pack_setup_output_dirs() { return 1; }
    run pack_run_suite
    assert_rc "1" "${RUN_RC}" "pack_setup_output_dirs failure returns non-zero"
    trap - EXIT INT TERM HUP QUIT

    pack_setup_output_dirs() { return 0; }
    pack_setup_hf_cache_dirs() { return 1; }
    run pack_run_suite
    assert_rc "1" "${RUN_RC}" "pack_setup_hf_cache_dirs failure returns non-zero"
    trap - EXIT INT TERM HUP QUIT

    pack_setup_hf_cache_dirs() { return 0; }
    pack_model_list_array() { PACK_MODEL_LIST=("model"); }
    pack_load_model_revisions() { return 0; }
    main_dynamic() { :; }
    PACK_OUTPUT_DIR_ABSOLUTE="true"
    OUTPUT_DIR="rel_out"
    PACK_NET="0"
    run pack_run_suite
    assert_rc "0" "${RUN_RC}" "absolute output dir path succeeds"
    assert_match '^/' "${OUTPUT_DIR}" "output dir normalized to absolute"
    trap - EXIT INT TERM HUP QUIT
    PACK_OUTPUT_DIR_ABSOLUTE="false"

    pack_model_list_array() { PACK_MODEL_LIST=(); }
    local error_log="${TEST_TMPDIR}/error.calls"
    : > "${error_log}"
    error_exit() { echo "$1" >> "${error_log}"; return 0; }
    OUTPUT_DIR="${TEST_TMPDIR}/out2"
    run pack_run_suite
    assert_rc "0" "${RUN_RC}" "missing model list triggers error_exit"
    assert_match "No models configured" "$(cat "${error_log}")" "error_exit called for empty models"
    trap - EXIT INT TERM HUP QUIT

    pack_model_list_array() { PACK_MODEL_LIST=("model"); }
    pack_preflight_models() { echo "preflight" > "${TEST_TMPDIR}/preflight.calls"; }
    OUTPUT_DIR="${TEST_TMPDIR}/out3"
    PACK_NET="1"
    run pack_run_suite
    assert_rc "0" "${RUN_RC}" "preflight path succeeds"
    assert_file_exists "${TEST_TMPDIR}/preflight.calls" "preflight invoked"
    trap - EXIT INT TERM HUP QUIT

    PACK_NET="0"
    pack_load_model_revisions() { return 1; }
    local offline_log="${TEST_TMPDIR}/offline.calls"
    : > "${offline_log}"
    error_exit() { echo "$1" >> "${offline_log}"; return 0; }
    OUTPUT_DIR="${TEST_TMPDIR}/out4"
    run pack_run_suite
    assert_rc "0" "${RUN_RC}" "offline revisions failure triggers error_exit"
    assert_match "Offline mode requires" "$(cat "${offline_log}")" "offline error recorded"
    trap - EXIT INT TERM HUP QUIT
}


test_pack_apply_network_mode_sets_env_flags() {
    mock_reset

    source ./scripts/proof_packs/lib/validation_suite.sh

    PACK_NET="0"
    pack_apply_network_mode "1"
    assert_eq "1" "${PACK_NET}" "net mode sets PACK_NET"
    assert_eq "1" "${INVARLOCK_ALLOW_NETWORK}" "network allowed"
    assert_eq "0" "${HF_DATASETS_OFFLINE}" "datasets online"
    assert_eq "0" "${TRANSFORMERS_OFFLINE}" "transformers online"
    assert_eq "0" "${HF_HUB_OFFLINE}" "hub online"

    pack_apply_network_mode "0"
    assert_eq "0" "${PACK_NET}" "offline mode sets PACK_NET"
    assert_eq "0" "${INVARLOCK_ALLOW_NETWORK}" "network disabled"
    assert_eq "1" "${HF_DATASETS_OFFLINE}" "datasets offline"
    assert_eq "1" "${TRANSFORMERS_OFFLINE}" "transformers offline"
    assert_eq "1" "${HF_HUB_OFFLINE}" "hub offline"
}
