#!/usr/bin/env bash
# task_baseline.sh - Baseline, calibration, and preset task implementations
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Usage: sourced by task_functions.sh or tests for task execution helpers

TASK_MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=task_common.sh
[[ -z "${TASK_COMMON_LOADED:-}" ]] && source "${TASK_MODULE_DIR}/task_common.sh"

# ============ TASK: SETUP_EVALUATE_BASELINE_REPORT ============

# Materialize the shared noop baseline report before edit/error evaluations claim
# GPU workers. This keeps parallel workers from blocking inside evaluate tasks
# while one worker generates the reusable baseline report.
task_setup_evaluate_baseline_report() {
    local model_name="$1"
    local gpu_id="$2"
    local output_dir="$3"
    local log_file="$4"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_path
    baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)
    local model_id
    model_id=$(cat "${model_output_dir}/.model_id" 2>/dev/null || true)

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    local abs_baseline_path
    abs_baseline_path="$(cd "$(dirname "${baseline_path}")" && pwd)/$(basename "${baseline_path}")"

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] preparing shared evaluate baseline report for ${model_name}" >> "${log_file}"

    local model_size
    model_size=$(_estimate_model_size "${baseline_path}")
    if [[ -z "${model_size}" || "${model_size}" == "7" ]] && [[ -n "${model_id}" ]]; then
        model_size=$(_get_model_size_from_name "${model_id}")
    fi

    local config seq_len stride preview_n final_n eval_batch
    config=$(_get_invarlock_config "${model_size}")
    IFS=':' read -r seq_len stride preview_n final_n eval_batch <<< "${config}"

    if [[ "${stride}" -ne "${seq_len}" ]]; then
        stride="${seq_len}"
        echo "  Pairing override: seq=${seq_len}, stride=${stride}" >> "${log_file}"
    fi

    local profile_flag="ci"
    local min_windows
    min_windows="$(_default_ci_min_windows "${seq_len}")"
    if [[ "${profile_flag}" == "ci" && "${min_windows}" =~ ^[0-9]+$ && "${min_windows}" -gt 0 ]]; then
        if [[ "${preview_n}" -lt "${min_windows}" || "${final_n}" -lt "${min_windows}" ]]; then
            preview_n="${min_windows}"
            final_n="${min_windows}"
            echo "  CI window override: preview=${preview_n}, final=${final_n}" >> "${log_file}"
        fi
    fi

    local tier="${INVARLOCK_TIER:-balanced}"
    local dataset_kind=""
    dataset_kind="$(pack_dataset_provider_kind "${INVARLOCK_DATASET:-}")"
    local effective_plan_json=""
    effective_plan_json="$(
        _plan_effective_ci_schedule \
            "${abs_baseline_path}" \
            "${model_size}" \
            "${tier}" \
            "${dataset_kind}" \
            "validation" \
            "42"
    )" || {
        echo "ERROR: Effective CI planning failed unexpectedly" >> "${log_file}"
        return 1
    }
    local selected_schedule=""
    selected_schedule="$(_apply_effective_ci_schedule "${effective_plan_json}" "${log_file}")" || return 1
    if [[ -n "${selected_schedule}" ]]; then
        IFS=':' read -r seq_len stride preview_n final_n <<< "${selected_schedule}"
    fi

    local bootstrap_replicates
    bootstrap_replicates="$(_resolve_bootstrap_replicates "${model_size}" "${tier}")"
    local baseline_report_root="${model_output_dir}/baseline_reports/${profile_flag}_${tier}_seq${seq_len}_pv${preview_n}_fn${final_n}"
    local baseline_report_file=""
    baseline_report_file=$(
        _ensure_evaluate_baseline_report \
            "${baseline_report_root}" \
            "${abs_baseline_path}" \
            "${profile_flag}" \
            "${tier}" \
            "${seq_len}" \
            "${stride}" \
            "${preview_n}" \
            "${final_n}" \
            "${eval_batch}" \
            "${bootstrap_replicates}" \
            "${model_size}" \
            "${log_file}"
    ) || {
        echo "ERROR: Failed to prepare reusable baseline report for ${model_name}" >> "${log_file}"
        return 1
    }

    echo "  Prepared reusable baseline report: ${baseline_report_file}" >> "${log_file}"
}

# ============ TASK: SETUP_BASELINE ============

# Download and setup baseline model
# Usage: task_setup_baseline <model_id> <model_name> <gpu_id> <output_dir> <log_file>
task_setup_baseline() {
    local model_id="$1"
    local model_name="$2"
    local gpu_id="$3"
    local output_dir="$4"
    local log_file="$5"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_dir="${model_output_dir}/models/baseline"

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Setting up baseline: ${model_id}" >> "${log_file}"

    # Check if already exists (resume mode)
    if [[ -d "${baseline_dir}" && -f "${baseline_dir}/config.json" ]]; then
        echo "  Baseline already exists, skipping download" >> "${log_file}"
        # Store baseline path for other tasks
        echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
        # Also store original model_id for model size detection in other tasks
        echo "${model_id}" > "${model_output_dir}/.model_id"
        _write_model_profile "${baseline_dir}" "${model_id}"
        if type update_model_task_memory &>/dev/null; then
            update_model_task_memory "${model_name}" "${output_dir}" "${model_id}"
        fi
        return 0
    fi

    mkdir -p "${model_output_dir}"/{models,evals,reports}

    # Use the main script's setup_model function if available
    if type setup_model &>/dev/null; then
        local baseline_path
        local exit_code=0
        baseline_path=$(setup_model "${model_id}" "${gpu_id}") || exit_code=$?

        if [[ ${exit_code} -eq 0 && -n "${baseline_path}" && -d "${baseline_path}" ]]; then
            echo "  Baseline ready at: ${baseline_path}" >> "${log_file}"
            echo "${baseline_path}" > "${model_output_dir}/.baseline_path"
            # Store original model_id for model size detection
            echo "${model_id}" > "${model_output_dir}/.model_id"
            _write_model_profile "${baseline_path}" "${model_id}"
            if type update_model_task_memory &>/dev/null; then
                update_model_task_memory "${model_name}" "${output_dir}" "${model_id}"
            fi
            return 0
        else
            echo "  ERROR: Failed to setup baseline" >> "${log_file}"
            return 1
        fi
    else
        # Inline implementation
        echo "  Downloading model ${model_id}..." >> "${log_file}"

        local revision=""
        revision=$(_task_get_model_revision "${model_id}" || true)
        if [[ -z "${revision}" ]]; then
            if [[ "${PACK_NET}" == "1" ]]; then
                echo "  ERROR: Missing pinned revision for ${model_id}; run preflight (--net 1)." >> "${log_file}"
            else
                echo "  ERROR: Offline mode requires model revisions. Run with --net 1 to preflight." >> "${log_file}"
            fi
            return 1
        fi

        if [[ "${PACK_NET}" != "1" ]]; then
            echo "  ERROR: Offline mode requested and baseline not cached for ${model_id}." >> "${log_file}"
            return 1
        fi

        # CUDA_VISIBLE_DEVICES is inherited from execute_task() for multi-GPU support
        local exit_code=0
        PACK_MODEL_REVISION="${revision}" \
            _cmd_python "${SCRIPT_DIR}/../../python/task_tools.py" download-baseline \
                --model-id "${model_id}" \
                --output-dir "${baseline_dir}" >> "${log_file}" 2>&1 || exit_code=$?

        if [[ ${exit_code} -eq 0 && -f "${baseline_dir}/config.json" ]]; then
            echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
            # Store original model_id for model size detection
            echo "${model_id}" > "${model_output_dir}/.model_id"
            _write_model_profile "${baseline_dir}" "${model_id}"
            if type update_model_task_memory &>/dev/null; then
                update_model_task_memory "${model_name}" "${output_dir}" "${model_id}"
            fi
            return 0
        fi
        return 1
    fi
}

# ============ TASK: CALIBRATION_RUN ==========

# Run single InvarLock calibration
# Usage: task_calibration_run <model_name> <gpu_id> <run_num> <seed> <output_dir> <log_file>
task_calibration_run() {
    local model_name="$1"
    local gpu_id="$2"
    local run_num="$3"
    local seed="$4"
    local output_dir="$5"
    local log_file="$6"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)
    local model_id=$(cat "${model_output_dir}/.model_id" 2>/dev/null || true)
    local run_dir="${model_output_dir}/reports/calibration/run_${run_num}"

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    # Check if already done
    if [[ -f "${run_dir}/baseline_report.json" || -f "${run_dir}/evaluation.report.json" ]]; then
        echo "  Calibration run ${run_num} already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Running calibration run ${run_num} (seed=${seed})" >> "${log_file}"

    mkdir -p "${run_dir}"

    # Get model-aware config using wrapper functions (try main script, then fallback)
    # First try to get model size from baseline path, then from stored model_id
    local model_size
    model_size=$(_estimate_model_size "${baseline_path}")
    if [[ -z "${model_size}" || "${model_size}" == "7" ]] && [[ -n "${model_id}" ]]; then
        # Fallback: detect from model_id string
        model_size=$(_get_model_size_from_name "${model_id}")
    fi

    # Get configuration for this model size
    local config
    config=$(_get_invarlock_config "${model_size}")

    IFS=':' read -r seq_len stride preview_n final_n eval_batch <<< "${config}"
    local params_json="${TASK_PARAMS:-}"
    local applied_override=0
    if [[ -n "${params_json}" && "${params_json}" != "null" ]]; then
        local override_seq_len override_stride override_batch
        override_seq_len=$(echo "${params_json}" | jq -r '.seq_len // empty' 2>/dev/null)
        override_stride=$(echo "${params_json}" | jq -r '.stride // empty' 2>/dev/null)
        override_batch=$(echo "${params_json}" | jq -r '.batch_size // empty' 2>/dev/null)
        if [[ "${override_seq_len}" =~ ^[0-9]+$ ]]; then
            seq_len="${override_seq_len}"
            applied_override=1
        fi
        if [[ "${override_stride}" =~ ^[0-9]+$ ]]; then
            stride="${override_stride}"
            applied_override=1
        fi
        if [[ "${override_batch}" =~ ^[0-9]+$ ]]; then
            eval_batch="${override_batch}"
            applied_override=1
        fi
        if [[ "${stride}" -gt "${seq_len}" ]]; then
            stride=$((seq_len / 2))
            [[ ${stride} -lt 1 ]] && stride=1
            applied_override=1
        fi
    fi

    # Force non-overlapping windows during calibration to avoid pairing mismatches
    stride="${seq_len}"

    echo "  Model size: ${model_size}, Config: seq=${seq_len}, stride=${stride}, windows=${preview_n}+${final_n}, batch=${eval_batch}" >> "${log_file}"
    if [[ ${applied_override} -eq 1 ]]; then
        echo "  OOM override applied: seq=${seq_len}, stride=${stride}, batch=${eval_batch}" >> "${log_file}"
    fi

    # For large models, use INVARLOCK_SKIP_OVERHEAD_CHECK to avoid loading
    # both baseline and edited models simultaneously (which would exceed 180GB).
    local profile_flag="ci"
    local min_windows
    min_windows="$(_default_ci_min_windows "${seq_len}")"
    if [[ "${profile_flag}" == "ci" && "${min_windows}" =~ ^[0-9]+$ && "${min_windows}" -gt 0 ]]; then
        if [[ "${preview_n}" -lt "${min_windows}" || "${final_n}" -lt "${min_windows}" ]]; then
            preview_n="${min_windows}"
            final_n="${min_windows}"
            applied_override=1
            echo "  CI window override: preview=${preview_n}, final=${final_n}" >> "${log_file}"
        fi
    fi
    local tier="${INVARLOCK_TIER:-balanced}"
    local dataset_kind=""
    dataset_kind="$(pack_dataset_provider_kind "${INVARLOCK_DATASET:-}")"
    local effective_plan_json=""
    effective_plan_json="$(
        _plan_effective_ci_schedule \
            "${baseline_path}" \
            "${model_size}" \
            "${tier}" \
            "${dataset_kind}" \
            "validation" \
            "${seed}"
    )" || {
        echo "ERROR: Effective CI planning failed unexpectedly" >> "${log_file}"
        return 1
    }
    local selected_schedule=""
    selected_schedule="$(_apply_effective_ci_schedule "${effective_plan_json}" "${log_file}")" || return 1
    if [[ -n "${selected_schedule}" ]]; then
        IFS=':' read -r seq_len stride preview_n final_n <<< "${selected_schedule}"
    fi
    local bootstrap_replicates=2000


    if _is_large_model "${model_size}"; then
        bootstrap_replicates=1000
    fi
    if [[ -n "${INVARLOCK_BOOTSTRAP_N:-}" ]]; then
        bootstrap_replicates="${INVARLOCK_BOOTSTRAP_N}"
    fi

    echo "  Calibration: enforcing window_overlap_fraction=0.0" >> "${log_file}"
    echo "  Calibration: context.run.skip_overhead_check=true" >> "${log_file}"
    local skip_overhead_config_yaml=$'context:\n  run:\n    skip_overhead_check: true'
    local -a extra_env=(INVARLOCK_WINDOW_OVERLAP_FRACTION=0.0 INVARLOCK_SKIP_OVERHEAD_CHECK=1)
    if _is_large_model "${model_size}"; then
        echo "  Large model (${model_size}): SKIP_OVERHEAD_CHECK=1" >> "${log_file}"
    fi
    if pack_remote_code_allowed; then
        extra_env+=(INVARLOCK_ALLOW_REMOTE_CODE=1)
    fi

    local config_root_base
    config_root_base="$(cd "${run_dir}" && pwd)"
    local config_root="${config_root_base}/config_root"
    mkdir -p "${config_root}/runtime/profiles"
    cat > "${config_root}/runtime/profiles/ci.yaml" << YAML
model:
  device_map: "auto"
  dtype: "bfloat16"
$(pack_model_trust_remote_code_yaml "  ")
  low_cpu_mem_usage: true
dataset:
  preview_n: ${preview_n}
  final_n: ${final_n}
eval:
  bootstrap:
    replicates: ${bootstrap_replicates}
    alpha: 0.05

context:
  run:
    skip_overhead_check: true
YAML

    extra_env+=("PYTHONPATH=${PACK_REPO_PYTHONPATH}")
    extra_env+=("INVARLOCK_CONFIG_ROOT=${config_root}")

    # Generate config YAML
    local config_yaml="${run_dir}/calibration_config.yaml"
    local guards_order_csv="${PACK_GUARDS_ORDER:-}"
    local -a raw_guards_order=()
    if [[ -n "${guards_order_csv}" ]]; then
        IFS=',' read -ra raw_guards_order <<< "${guards_order_csv}"
    fi
    local -a guards_order=()
    local g
    for g in "${raw_guards_order[@]}"; do
        g="$(echo "${g}" | xargs)"
        [[ -z "${g}" ]] && continue
        guards_order+=("${g}")
    done
    if [[ ${#guards_order[@]} -eq 0 ]]; then
        guards_order=("invariants" "spectral" "rmt" "variance" "invariants")
    fi
    local guards_order_yaml=""
    for g in "${guards_order[@]}"; do
        guards_order_yaml+=$'    - '"${g}"$'\n'
    done

    local dataset_provider_yaml
    dataset_provider_yaml="$(pack_render_dataset_provider_yaml "${INVARLOCK_DATASET:-wikitext2}")"

    cat > "${config_yaml}" << YAML_EOF
model:
  id: "${baseline_path}"
  adapter: "hf_auto"
  device: "auto"
  device_map: "auto"
  dtype: "bfloat16"
$(pack_model_trust_remote_code_yaml "  ")
  low_cpu_mem_usage: true

dataset:
${dataset_provider_yaml}
  preview_n: ${preview_n}
  final_n: ${final_n}
  seq_len: ${seq_len}
  stride: ${stride}
  seed: ${seed}

edit:
  name: "noop"

guards:
  order:
${guards_order_yaml}

eval:
  bootstrap:
    replicates: ${bootstrap_replicates}
    parallel: true
  batch_size: ${eval_batch}
  window_overlap_fraction: 0.0

${skip_overhead_config_yaml}

auto:
  enabled: true
  tier: "${tier}"
  probes: 0
YAML_EOF

    local profile_flag="ci"

    # CUDA_VISIBLE_DEVICES is inherited from execute_task() for multi-GPU support
    local exit_code=0
    (
        export "${extra_env[@]}"
        _pack_run_from_config \
            --config "${config_yaml}" \
            --profile "${profile_flag}" \
            --out "${run_dir}" \
            >> "${log_file}" 2>&1
    ) || exit_code=$?

    # Copy report to standard location
    local report_file=$(find "${run_dir}" -name "report*.json" -type f 2>/dev/null | head -1)
    if [[ -n "${report_file}" ]]; then
        cp "${report_file}" "${run_dir}/baseline_report.json" 2>/dev/null || true
        local conversion_rc=0
        _cmd_python "${SCRIPT_DIR}/../../python/task_tools.py" evaluation-report \
            --report "${report_file}" \
            --out "${run_dir}/evaluation.report.json" >> "${log_file}" 2>&1 || conversion_rc=$?
        if [[ ${conversion_rc} -ne 0 ]]; then
            echo "WARNING: calibration report kept without evaluation.report.json from ${report_file} (exit=${conversion_rc})" >> "${log_file}"
        fi
    fi

    return ${exit_code}
}

# ============ TASK: GENERATE_PRESET ============

# Generate calibrated preset from calibration runs
# Usage: task_generate_preset <model_name> <output_dir> <log_file>
task_generate_preset() {
    local model_name="$1"
    local output_dir="$2"
    local log_file="$3"

    local model_output_dir="${output_dir}/${model_name}"
    local cal_dir="${model_output_dir}/reports/calibration"
    local preset_dir="${output_dir}/presets"
    local preset_base="${preset_dir}/calibrated_preset_${model_name}"

    if [[ -f "${preset_base}.yaml" || -f "${preset_base}.json" ]]; then
        echo "  Preset already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Generating calibrated preset" >> "${log_file}"

    mkdir -p "${preset_dir}"

    # Get baseline path and model_id to estimate model size
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)
    local model_id=$(cat "${model_output_dir}/.model_id" 2>/dev/null || true)

    # Get model-aware config for seq_len/stride using wrapper functions
    # (these handle fallback when main script functions aren't available)
    local model_size
    model_size=$(_estimate_model_size "${baseline_path}")
    if [[ -z "${model_size}" || "${model_size}" == "7" ]] && [[ -n "${model_id}" ]]; then
        # Fallback: detect from model_id string
        model_size=$(_get_model_size_from_name "${model_id}")
    fi

    # Get config using wrapper (tries main script, then fallback)
    local config
    config=$(_get_invarlock_config "${model_size}")

    IFS=':' read -r seq_len stride preview_n final_n eval_batch <<< "${config}"
    local tier="${INVARLOCK_TIER:-balanced}"
    local dataset_kind=""
    dataset_kind="$(pack_dataset_provider_kind "${INVARLOCK_DATASET:-}")"
    local effective_plan_json=""
    effective_plan_json="$(
        _plan_effective_ci_schedule \
            "${baseline_path}" \
            "${model_size}" \
            "${tier}" \
            "${dataset_kind}" \
            "validation" \
            "42"
    )" || {
        echo "ERROR: Effective CI planning failed unexpectedly" >> "${log_file}"
        return 1
    }
    local selected_schedule=""
    selected_schedule="$(_apply_effective_ci_schedule "${effective_plan_json}" "${log_file}")" || return 1
    if [[ -n "${selected_schedule}" ]]; then
        IFS=':' read -r seq_len stride preview_n final_n <<< "${selected_schedule}"
    fi

    # Export for use in Python script
    export PRESET_SEQ_LEN="${seq_len}"
    export PRESET_STRIDE="${stride}"
    export PRESET_PREVIEW_N="${preview_n}"
    export PRESET_FINAL_N="${final_n}"

    local exit_code=0
    local evidence_packs_dir
    evidence_packs_dir="$(cd "${SCRIPT_DIR}/../.." && pwd)"
    local generator="${evidence_packs_dir}/python/preset_generator.py"
    _cmd_python "${generator}" \
        --cal-dir "${cal_dir}" \
        --preset-file "${preset_base}.yaml" \
        --model-name "${model_name}" \
        --model-path "${baseline_path}" \
        --tier "${tier}" \
        --dataset-provider "${INVARLOCK_DATASET:-wikitext2}" \
        --seq-len "${seq_len}" \
        --stride "${stride}" \
        --preview-n "${preview_n}" \
        --final-n "${final_n}" \
        --edit-types "quant_rtn,fp8_quant,magnitude_prune,lowrank_svd" \
        >> "${log_file}" 2>&1 || exit_code=$?

    return ${exit_code}
}
