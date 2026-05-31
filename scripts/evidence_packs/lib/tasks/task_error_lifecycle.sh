#!/usr/bin/env bash
# task_error_lifecycle.sh - Error model creation, evaluation, and cleanup tasks
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Usage: sourced by task_functions.sh or tests for task execution helpers

TASK_MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=task_common.sh
[[ -z "${TASK_COMMON_LOADED:-}" ]] && source "${TASK_MODULE_DIR}/task_common.sh"

# ============ TASK: CREATE_ERROR ============

# Create error-injected model
# Usage: task_create_error <model_name> <gpu_id> <error_type> <error_env_json> <output_dir> <log_file>
task_create_error() {
    local model_name="$1"
    local gpu_id="$2"
    local error_type="$3"
    local error_env_json="${4:-{}}"
    local output_dir="$5"
    local log_file="$6"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)
    local error_path="${model_output_dir}/models/error_${error_type}"

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    # Only treat error models as cached when the injector completed fully.
    # error_metadata.json is written by task_tools.py create-error-model; partial
    # directories (e.g. OOM-killed saves) may still contain config.json.
    if [[ -d "${error_path}" && -f "${error_path}/config.json" && -f "${error_path}/error_metadata.json" ]]; then
        echo "  Error model ${error_type} already exists, skipping" >> "${log_file}"
        return 0
    fi
    if [[ -d "${error_path}" && -f "${error_path}/config.json" && ! -f "${error_path}/error_metadata.json" ]]; then
        echo "  WARNING: Found incomplete error model (missing error_metadata.json); recreating: ${error_path}" >> "${log_file}"
        rm -rf "${error_path}" 2>/dev/null || true
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Creating error model: ${error_type}" >> "${log_file}"

    local -a injector_env=()
    if [[ -n "${error_env_json}" && "${error_env_json}" != "null" ]]; then
        mapfile -t injector_env < <(
            printf '%s\n' "${error_env_json}" | jq -r '
                objects
                | to_entries[]
                | select((.key | type) == "string")
                | select(.key | startswith("INVARLOCK_"))
                | select(.key | test("^[A-Z][A-Z0-9_]*$"))
                | select((.value | type) as $t | ($t == "string" or $t == "number" or $t == "boolean"))
                | "\(.key)=\(.value | tostring)"
            ' 2>/dev/null
        )
    fi
    if [[ ${#injector_env[@]} -gt 0 ]]; then
        echo "  Injector env overrides: ${injector_env[*]}" >> "${log_file}"
    fi

    local create_rc=0
    if type create_error_model &>/dev/null; then
        if [[ ${#injector_env[@]} -gt 0 ]]; then
            local -a injector_keys=()
            local -a injector_prev_values=()
            local -a injector_had_prev=()
            local idx=0
            for entry in "${injector_env[@]}"; do
                local env_key="${entry%%=*}"
                injector_keys+=("${env_key}")
                if [[ ${!env_key+x} ]]; then
                    injector_had_prev+=("1")
                    injector_prev_values+=("${!env_key}")
                else
                    injector_had_prev+=("0")
                    injector_prev_values+=("")
                fi
                export "${entry}"
            done
            create_error_model "${baseline_path}" "${error_path}" "${error_type}" "${gpu_id}" >> "${log_file}" 2>&1 || create_rc=$?
            for idx in "${!injector_keys[@]}"; do
                if [[ "${injector_had_prev[$idx]}" == "1" ]]; then
                    export "${injector_keys[$idx]}=${injector_prev_values[$idx]}"
                else
                    unset "${injector_keys[$idx]}"
                fi
            done
        else
            create_error_model "${baseline_path}" "${error_path}" "${error_type}" "${gpu_id}" >> "${log_file}" 2>&1 || create_rc=$?
        fi
    else
        echo "ERROR: create_error_model not available" >> "${log_file}"
        return 1
    fi
    if [[ ${create_rc} -ne 0 ]]; then
        echo "  ERROR: create_error_model failed (exit=${create_rc})" >> "${log_file}"
        return 1
    fi

    if [[ -d "${error_path}" && -f "${error_path}/config.json" && -f "${error_path}/error_metadata.json" ]]; then
        echo "  Created: ${error_path}" >> "${log_file}"
        return 0
    else
        echo "  ERROR: Failed to create error model" >> "${log_file}"
        return 1
    fi
}

# ============ TASK: evaluate_ERROR ============

error_requires_inline_baseline_eval() {
    local error_type="$1"
    case "${error_type}" in
        nan_injection|inf_injection|shape_mismatch|missing_tensors|weight_tying_break)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

error_supports_structural_failure_report() {
    local error_type="$1"
    case "${error_type}" in
        nan_injection|inf_injection|shape_mismatch|missing_tensors|weight_tying_break)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# evaluate error-injected model
# Usage: task_evaluate_error <model_name> <gpu_id> <error_type> <output_dir> <log_file>
task_evaluate_error() {
    local model_name="$1"
    local gpu_id="$2"
    local error_type="$3"
    local output_dir="$4"
    local log_file="$5"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)
    local model_id=$(cat "${model_output_dir}/.model_id" 2>/dev/null || true)
    local error_path="${model_output_dir}/models/error_${error_type}"
    local cert_dir="${model_output_dir}/reports/errors/${error_type}"
    local cert_file="${cert_dir}/evaluation.report.json"
    local preset_dir="${output_dir}/presets"

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    if [[ ! -d "${error_path}" ]]; then
        echo "ERROR: Error model not found: ${error_path}" >> "${log_file}"
        return 1
    fi

    local abs_baseline_path
    abs_baseline_path="$(cd "$(dirname "${baseline_path}")" && pwd)/$(basename "${baseline_path}")"
    local abs_error_path
    abs_error_path="$(cd "$(dirname "${error_path}")" && pwd)/$(basename "${error_path}")"

    if [[ -f "${cert_file}" ]]; then
        echo "  Evaluation report for error ${error_type} already exists, skipping" >> "${log_file}"
        return 0
    fi

    # Repair known config inconsistencies for missing_tensors error models created by older pack versions.
    if [[ "${error_type}" == "missing_tensors" ]]; then
        local repair_script="${SCRIPT_DIR}/../../python/task_tools.py"
        if [[ -f "${repair_script}" && -f "${abs_baseline_path}/config.json" && -f "${abs_error_path}/config.json" ]]; then
            _cmd_python "${repair_script}" repair-missing-tensors-config "${abs_baseline_path}/config.json" "${abs_error_path}/config.json" >> "${log_file}" 2>&1 || true
        fi
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] evaluating error model: ${error_type}" >> "${log_file}"

    mkdir -p "${cert_dir}"

    local abs_cert_dir
    abs_cert_dir="$(cd "${cert_dir}" && pwd)"
    local abs_log_file
    abs_log_file="$(cd "$(dirname "${log_file}")" && pwd)/$(basename "${log_file}")"
    cert_dir="${abs_cert_dir}"
    cert_file="${cert_dir}/evaluation.report.json"
    log_file="${abs_log_file}"

    # Get model size for config and profile decision
    local model_size
    model_size=$(_estimate_model_size "${baseline_path}")
    if [[ -z "${model_size}" || "${model_size}" == "7" ]] && [[ -n "${model_id}" ]]; then
        model_size=$(_get_model_size_from_name "${model_id}")
    fi

    # Get model-aware config for window counts (needed for CI override)
    local config seq_len stride preview_n final_n eval_batch
    config=$(_get_invarlock_config "${model_size}")
    IFS=':' read -r seq_len stride preview_n final_n eval_batch <<< "${config}"
    local params_json="${TASK_PARAMS:-}"
    local applied_override=0
    if [[ -n "${params_json}" && "${params_json}" != "null" ]]; then
        local override_seq_len override_stride
        override_seq_len=$(echo "${params_json}" | jq -r '.seq_len // empty' 2>/dev/null)
        override_stride=$(echo "${params_json}" | jq -r '.stride // empty' 2>/dev/null)
        if [[ "${override_seq_len}" =~ ^[0-9]+$ ]]; then
            seq_len="${override_seq_len}"
            applied_override=1
        fi
        if [[ "${override_stride}" =~ ^[0-9]+$ ]]; then
            stride="${override_stride}"
            applied_override=1
        fi
        if [[ "${stride}" -gt "${seq_len}" ]]; then
            stride=$((seq_len / 2))
            [[ ${stride} -lt 1 ]] && stride=1
            applied_override=1
        fi
        if [[ ${applied_override} -eq 1 ]]; then
            echo "  OOM override: seq=${seq_len}, stride=${stride}" >> "${log_file}"
        fi
    fi
    if [[ "${stride}" -ne "${seq_len}" ]]; then
        stride="${seq_len}"
        applied_override=1
        echo "  Pairing override: seq=${seq_len}, stride=${stride}" >> "${log_file}"
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
    if error_requires_inline_baseline_eval "${error_type}"; then
        echo "  Baseline report reuse disabled for structural error: ${error_type}" >> "${log_file}"
    else
        :
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
                "${log_file}" \
                || true
        )
    fi
    local -a baseline_report_args=()
    if [[ -n "${baseline_report_file}" && -f "${baseline_report_file}" ]]; then
        local abs_baseline_report_file
        abs_baseline_report_file="$(_stage_baseline_report_for_eval "${baseline_report_file}" "${cert_dir}" "${log_file}")" || {
            echo "ERROR: Failed to stage baseline report for evaluate runtime: ${baseline_report_file}" >> "${log_file}"
            return 1
        }
        baseline_report_args=(--baseline-report "${abs_baseline_report_file}")
        echo "  Reusing baseline report: ${baseline_report_file}" >> "${log_file}"
    else
        echo "  WARNING: Baseline report unavailable; will run per-cert baseline evaluation" >> "${log_file}"
    fi

    local -a extra_env=()
    extra_env+=("PYTHONPATH=${PACK_REPO_PYTHONPATH}")
    local skip_overhead_config_yaml=""
    local skip_overhead_in_preset="0"
    if _is_large_model "${model_size}"; then
        skip_overhead_config_yaml=$'context:\n  run:\n    skip_overhead_check: true'
        skip_overhead_in_preset="1"
        echo "  Large model (${model_size}): context.run.skip_overhead_check=true" >> "${log_file}"
    fi
    extra_env+=(INVARLOCK_STORE_EVAL_WINDOWS=1)
    if pack_remote_code_allowed; then
        extra_env+=(INVARLOCK_ALLOW_REMOTE_CODE=1)
    fi

    local config_root_base
    config_root_base="$(cd "${cert_dir}" && pwd)"
    local config_root="${config_root_base}/config_root"
    mkdir -p "${config_root}/runtime/profiles"
    cat > "${config_root}/runtime/profiles/ci.yaml" << YAML
model:
  device_map: "auto"
  dtype: "bfloat16"
$(pack_model_trust_remote_code_yaml "  ")
  low_cpu_mem_usage: true
dataset:
  seq_len: ${seq_len}
  stride: ${stride}
  preview_n: ${preview_n}
  final_n: ${final_n}
eval:
  bootstrap:
    replicates: ${bootstrap_replicates}
    alpha: 0.05
${skip_overhead_config_yaml}
YAML

    extra_env+=("INVARLOCK_CONFIG_ROOT=${config_root}")

    # Find calibrated preset (must have seq_len/stride embedded)
    local preset_file=""
    for ext in yaml json; do
        local f="${preset_dir}/calibrated_preset_${model_name}.${ext}"
        if [[ -f "${f}" ]]; then
            preset_file="${f}"
            break
        fi
    done

    # If no preset found, we need to create one with model-specific params
    if [[ -z "${preset_file}" || ! -f "${preset_file}" ]]; then
        echo "  WARNING: No preset found for ${model_name}, creating minimal preset" >> "${log_file}"

        # Config already parsed above (seq_len, stride, preview_n, final_n, eval_batch)
        # Create minimal preset with seq_len/stride
        mkdir -p "${preset_dir}"
        preset_file="${preset_dir}/calibrated_preset_${model_name}.yaml"
        local dataset_provider_yaml
        dataset_provider_yaml="$(pack_render_dataset_provider_yaml "${INVARLOCK_DATASET:-wikitext2}")"
        cat > "${preset_file}" << PRESET_YAML
dataset:
${dataset_provider_yaml}
  split: validation
  seq_len: ${seq_len}
  stride: ${stride}
  preview_n: ${preview_n}
  final_n: ${final_n}
  seed: 42
PRESET_YAML
        echo "  Created preset: ${preset_file}" >> "${log_file}"
    fi

    # Run evaluate in isolated working directory to avoid temp file race conditions
    # (invarlock creates tmp/.evaluate/ in the current directory which conflicts in parallel runs)
    local work_dir="${cert_dir}/.workdir"
    mkdir -p "${work_dir}"
    local abs_preset_file
    abs_preset_file="$(_stage_preset_for_eval "${preset_file}" "${cert_dir}" "${log_file}")" || {
        echo "ERROR: Failed to stage preset for evaluate runtime: ${preset_file}" >> "${log_file}"
        return 1
    }
    _normalize_staged_preset_for_eval "${abs_preset_file}" "${seq_len}" "${stride}" "${preview_n}" "${final_n}" "${skip_overhead_in_preset}" "${log_file}" || {
        echo "ERROR: Failed to normalize staged preset for evaluate runtime: ${abs_preset_file}" >> "${log_file}"
        return 1
    }

    # CUDA_VISIBLE_DEVICES is inherited from execute_task() for multi-GPU support
    local exit_code=0
    local -a evaluate_extra_args=()
    evaluate_extra_args+=(--timing-json "${cert_dir}/evaluate_timing.json")
    if _pack_defer_report_rendering_enabled; then
        evaluate_extra_args+=(--defer-report-rendering)
    fi
    (
        cd "${work_dir}" || exit 1
        env "${extra_env[@]}" invarlock evaluate \
            --baseline "${abs_baseline_path}" \
            "${baseline_report_args[@]}" \
            --subject "${abs_error_path}" \
            --profile "${profile_flag}" \
            --tier "${tier}" \
            --out "${cert_dir}" \
            --report-out "${cert_dir}" \
            --preset "${abs_preset_file}" \
            "${evaluate_extra_args[@]}" >> "${log_file}" 2>&1
    ) || exit_code=$?

    # Find and copy report (only the canonical cert)
    if [[ ! -f "${cert_file}" ]]; then
        local found_cert
        found_cert=$(find "${cert_dir}" -name "evaluation.report.json" -type f 2>/dev/null | sort | tail -1)
        if [[ -n "${found_cert}" && -f "${found_cert}" && "${found_cert}" != "${cert_file}" ]]; then
            cp "${found_cert}" "${cert_file}" 2>/dev/null || true
        fi
    fi

    if [[ ! -f "${cert_file}" ]] && ! error_supports_structural_failure_report "${error_type}"; then
        local report_file=""
        report_file=$(find "${cert_dir}" -name "report*.json" -type f 2>/dev/null | sort | tail -1)
        if [[ -n "${report_file}" && -f "${report_file}" ]]; then
            _cmd_python "${SCRIPT_DIR}/../../python/task_tools.py" evaluation-report \
                --report "${report_file}" \
                --out "${cert_file}" >> "${log_file}" 2>&1 || true
        fi
    fi

    if [[ ! -f "${cert_file}" && ${exit_code} -ne 0 ]] && error_supports_structural_failure_report "${error_type}"; then
        local source_report_file=""
        local edited_report_file=""
        local edited_events_file=""
        local source_runtime_manifest=""
        source_report_file=$({ find "${cert_dir}/source" -name "report.json" -type f 2>/dev/null || true; } | sort | tail -1)
        edited_report_file=$({ find "${cert_dir}/edited" -name "report.json" -type f 2>/dev/null || true; } | sort | tail -1)
        edited_events_file=$({ find "${cert_dir}/edited" -name "events.jsonl" -type f 2>/dev/null || true; } | sort | tail -1)
        source_runtime_manifest=$({ find "${cert_dir}/source" -name "runtime.manifest.json" -type f 2>/dev/null || true; } | sort | tail -1)
        local structural_report_args=(
            "${SCRIPT_DIR}/../../python/task_tools.py"
            structural-failure-report
            --error-type "${error_type}"
            --out "${cert_file}"
            --message "invarlock evaluate exited ${exit_code} without evaluation.report.json"
        )
        if [[ -n "${source_report_file}" && -f "${source_report_file}" ]]; then
            structural_report_args+=(--source-report "${source_report_file}")
        fi
        if [[ -n "${edited_report_file}" && -f "${edited_report_file}" ]]; then
            structural_report_args+=(--edited-report "${edited_report_file}")
        fi
        if [[ -n "${edited_events_file}" && -f "${edited_events_file}" ]]; then
            structural_report_args+=(--edited-events "${edited_events_file}")
        fi
        if [[ -n "${source_runtime_manifest}" && -f "${source_runtime_manifest}" ]]; then
            structural_report_args+=(--source-runtime-manifest "${source_runtime_manifest}")
        fi
        _cmd_python "${structural_report_args[@]}" >> "${log_file}" 2>&1 || true
        if [[ -f "${cert_file}" ]]; then
            echo "  WARNING: synthesized structural-failure evaluation.report.json for structural error: ${error_type}" >> "${log_file}"
        fi
    fi

    # Compare-mode evaluate cannot directly expose delta-style RMT signals because
    # guards are prepared/finalized on the same loaded model. For the RMT probe
    # scenario, emit an explicit cross-model artifact on shared windows.
    if [[ "${error_type}" == rmt_norm_noise* && "${PACK_ENABLE_RMT_CROSS_PROBE:-1}" != "0" ]]; then
        local probe_script="${SCRIPT_DIR}/../../python/rmt_cross_model_probe.py"
        local probe_out="${cert_dir}/rmt_probe.json"
        if [[ -f "${probe_script}" && -n "${baseline_report_file}" && -f "${baseline_report_file}" ]]; then
            local probe_rc=0
            local probe_args=()
            if pack_remote_code_allowed; then
                probe_args+=(--trust-remote-code)
            fi
            _cmd_python "${probe_script}" \
                --baseline-model "${abs_baseline_path}" \
                --subject-model "${abs_error_path}" \
                --baseline-report "${baseline_report_file}" \
                --out "${probe_out}" \
                --tier "${tier}" \
                --profile "${profile_flag}" \
                --activation-windows "${PACK_RMT_PROBE_WINDOWS:-64}" \
                "${probe_args[@]}" \
                >> "${log_file}" 2>&1 || probe_rc=$?
            if [[ ${probe_rc} -ne 0 ]]; then
                echo "  WARNING: RMT cross-model probe failed (exit=${probe_rc})" >> "${log_file}"
            fi
        else
            echo "  WARNING: Skipping RMT cross-model probe (missing script or baseline report)" >> "${log_file}"
        fi
    fi

    # VE/variance is a remediation guard and is muted under compare-mode evaluation
    # because the subject run uses a no-op edit. Emit an explicit probe artifact on
    # shared windows for the VE demo scenario.
    if [[ "${error_type}" == ve_mlp_scale_skew* && "${PACK_ENABLE_VE_CROSS_PROBE:-1}" != "0" ]]; then
        local ve_probe_script="${SCRIPT_DIR}/../../python/ve_cross_model_probe.py"
        local ve_probe_out="${cert_dir}/ve_probe.json"
        if [[ -f "${ve_probe_script}" && -n "${baseline_report_file}" && -f "${baseline_report_file}" ]]; then
            local ve_probe_rc=0
            local ve_probe_args=()
            if pack_remote_code_allowed; then
                ve_probe_args+=(--trust-remote-code)
            fi
            _cmd_python "${ve_probe_script}" \
                --baseline-model "${abs_baseline_path}" \
                --subject-model "${abs_error_path}" \
                --baseline-report "${baseline_report_file}" \
                --out "${ve_probe_out}" \
                --tier "${tier}" \
                --profile "${profile_flag}" \
                --calibration-windows "${PACK_VE_PROBE_WINDOWS:-12}" \
                --min-coverage "${PACK_VE_PROBE_MIN_COVERAGE:-10}" \
                "${ve_probe_args[@]}" \
                >> "${log_file}" 2>&1 || ve_probe_rc=$?
            if [[ ${ve_probe_rc} -ne 0 ]]; then
                echo "  WARNING: VE cross-model probe failed (exit=${ve_probe_rc})" >> "${log_file}"
            fi
        else
            echo "  WARNING: Skipping VE cross-model probe (missing script or baseline report)" >> "${log_file}"
        fi
    fi

    # Same as evaluate_EDIT: keep the task successful when the report exists even
    # if the CLI exited non-zero (common for injected failures).
    if [[ ${exit_code} -ne 0 && -f "${cert_file}" ]]; then
        echo "  WARNING: invarlock evaluate exited ${exit_code} but wrote evaluation.report.json; treating as success" >> "${log_file}"
        exit_code=0
    fi

    return ${exit_code}
}

# ============ TASK: CLEANUP_ERROR ============
task_cleanup_error() {
    local model_name="$1"
    local error_type="$2"
    local output_dir="$3"
    local log_file="$4"

    if [[ "${PACK_CLEANUP_MODELS:-1}" == "0" ]]; then
        echo "  Cleanup disabled (PACK_CLEANUP_MODELS=0); skipping error cleanup" >> "${log_file}"
        return 0
    fi

    if [[ -z "${error_type}" ]]; then
        echo "ERROR: CLEANUP_ERROR missing error_type" >> "${log_file}"
        return 1
    fi

    local model_output_dir="${output_dir}/${model_name}"
    local models_root="${model_output_dir}/models"
    local models_root_abs
    models_root_abs="$(cd "${models_root}" 2>/dev/null && pwd -P)" || {
        echo "ERROR: Models root missing for cleanup: ${models_root}" >> "${log_file}"
        return 1
    }
    local error_parent_rel
    error_parent_rel="$(dirname "${error_type}")"
    local error_basename
    error_basename="error_$(basename "${error_type}")"
    local error_parent_abs
    error_parent_abs="$(cd "${models_root_abs}/${error_parent_rel}" 2>/dev/null && pwd -P)" || {
        echo "ERROR: Refusing to delete path outside models root: ${models_root}/error_${error_type}" >> "${log_file}"
        return 1
    }
    local error_path="${error_parent_abs}/${error_basename}"

    if [[ "${error_path}" != "${models_root_abs}/"* ]]; then
        echo "ERROR: Refusing to delete path outside models root: ${error_path}" >> "${log_file}"
        return 1
    fi
    if [[ ! -e "${error_path}" ]]; then
        echo "  Error path already absent: ${error_path}" >> "${log_file}"
        return 0
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Cleaning up error model: ${error_type}" >> "${log_file}"
    rm -rf "${error_path}" >> "${log_file}" 2>&1 || {
        echo "ERROR: Failed to remove error path: ${error_path}" >> "${log_file}"
        return 1
    }

    echo "  Removed: ${error_path}" >> "${log_file}"
    return 0
}
