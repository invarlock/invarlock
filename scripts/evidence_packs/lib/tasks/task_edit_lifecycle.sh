#!/usr/bin/env bash
# task_edit_lifecycle.sh - Edit model creation, evaluation, and cleanup tasks
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Usage: sourced by task_functions.sh or tests for task execution helpers

TASK_MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=task_common.sh
[[ -z "${TASK_COMMON_LOADED:-}" ]] && source "${TASK_MODULE_DIR}/task_common.sh"

# ============ TASK: CREATE_EDIT ============

# Create edited model
# Usage: task_create_edit <model_name> <gpu_id> <edit_spec> <version> <output_dir> <log_file>
task_create_edit() {
    local model_name="$1"
    local gpu_id="$2"
    local edit_spec="$3"
    local version="$4"
    local output_dir="$5"
    local log_file="$6"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    local resolved
    resolved=$(resolve_edit_params "${model_output_dir}" "${edit_spec}" "${version}")
    local status
    status=$(echo "${resolved}" | jq -r '.status')
    if [[ "${status}" == "skipped" ]]; then
        echo "  Clean edit skipped by tuned preset: ${edit_spec}" >> "${log_file}"
        return 0
    fi
    if [[ "${status}" != "selected" ]]; then
        echo "ERROR: Unable to resolve edit spec (${edit_spec}): ${status}" >> "${log_file}"
        return 1
    fi

    local edit_type param1 param2 scope edit_dir_name
    edit_type=$(echo "${resolved}" | jq -r '.edit_type')
    param1=$(echo "${resolved}" | jq -r '.param1')
    param2=$(echo "${resolved}" | jq -r '.param2')
    scope=$(echo "${resolved}" | jq -r '.scope')
    edit_dir_name=$(echo "${resolved}" | jq -r '.edit_dir_name')
    if [[ -z "${edit_dir_name}" || "${edit_dir_name}" == "null" ]]; then
        echo "ERROR: Empty edit_dir_name for ${edit_spec}" >> "${log_file}"
        return 1
    fi

    local edit_path="${model_output_dir}/models/${edit_dir_name}"

    # Check if already exists
    if _edit_artifact_complete "${edit_path}"; then
        echo "  Edit ${edit_dir_name} already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Creating edit: ${edit_dir_name}" >> "${log_file}"

    local create_rc=0
    _task_create_model_variant "${baseline_path}" "${edit_path}" "${edit_type}" "${param1}" "${param2}" "${scope}" "${gpu_id}" >> "${log_file}" 2>&1 || create_rc=$?
    if [[ ${create_rc} -ne 0 ]]; then
        return 1
    fi

    # Verify creation
    if _edit_artifact_complete "${edit_path}"; then
        echo "  Created: ${edit_path}" >> "${log_file}"
        return 0
    else
        echo "  ERROR: Failed to create edit" >> "${log_file}"
        return 1
    fi
}

_edit_artifact_has_weights() {
    local edit_path="$1"
    compgen -G "${edit_path}/*.safetensors" >/dev/null \
        || [[ -f "${edit_path}/model.safetensors" ]] \
        || [[ -f "${edit_path}/model.safetensors.index.json" ]] \
        || [[ -f "${edit_path}/pytorch_model.bin" ]] \
        || [[ -f "${edit_path}/pytorch_model.bin.index.json" ]]
}

_edit_artifact_has_tokenizer() {
    local edit_path="$1"
    [[ -f "${edit_path}/tokenizer.json" ]] \
        || [[ -f "${edit_path}/tokenizer_config.json" ]] \
        || [[ -f "${edit_path}/tokenizer.model" ]] \
        || [[ -f "${edit_path}/special_tokens_map.json" ]]
}

_edit_artifact_complete() {
    local edit_path="$1"
    _cmd_python "${SCRIPT_DIR}/../../python/editing/validate_artifact.py" "${edit_path}" --require-metadata >/dev/null 2>&1
}

# ============ TASK: CREATE_EDITS_BATCH ============

# Create all edited models from one parsed batch.
# By default the Python helper reloads the baseline per edit to avoid GPU
# memory spikes from deep-copying large loaded models. Set
# PACK_BATCH_EDIT_STRATEGY=deepcopy for small models where single-load
# throughput is preferred.
# Usage: task_create_edits_batch <model_name> <gpu_id> <edit_specs_json> <output_dir> <log_file>
task_create_edits_batch() {
    local model_name="$1"
    local gpu_id="$2"
    local edit_specs_json="$3"
    local output_dir="$4"
    local log_file="$5"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Creating batch edits" >> "${log_file}"
    echo "  Baseline: ${baseline_path}" >> "${log_file}"

    # Process each edit spec using Python for efficient batch creation
    # CUDA_VISIBLE_DEVICES is inherited from execute_task() for multi-GPU support
    local exit_code=0
    _cmd_python "${SCRIPT_DIR}/../../python/create_edits_batch.py" \
        --baseline "${baseline_path}" \
        --model-output-dir "${model_output_dir}" \
        --edit-specs-json "${edit_specs_json}" >> "${log_file}" 2>&1 || exit_code=$?

    if [[ ${exit_code} -eq 0 ]]; then
        echo "  Batch edit creation complete" >> "${log_file}"
    else
        echo "  ERROR: Batch edit creation failed" >> "${log_file}"
    fi

    return ${exit_code}
}

# ============ TASK: evaluate_EDIT ============

# Run InvarLock evaluate on edited model
# Usage: task_evaluate_edit <model_name> <gpu_id> <edit_spec> <version> <run_num> <output_dir> <log_file>
task_evaluate_edit() {
    local model_name="$1"
    local gpu_id="$2"
    local edit_spec="$3"
    local version="$4"
    local run_num="$5"
    local output_dir="$6"
    local log_file="$7"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)
    local model_id=$(cat "${model_output_dir}/.model_id" 2>/dev/null || true)
    local preset_dir="${output_dir}/presets"

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    local resolved
    resolved=$(resolve_edit_params "${model_output_dir}" "${edit_spec}" "${version}")
    local status
    status=$(echo "${resolved}" | jq -r '.status')
    if [[ "${status}" == "skipped" ]]; then
        echo "  Clean edit skipped by tuned preset: ${edit_spec}" >> "${log_file}"
        return 0
    fi
    if [[ "${status}" != "selected" ]]; then
        echo "ERROR: Unable to resolve edit spec (${edit_spec}): ${status}" >> "${log_file}"
        return 1
    fi
    local edit_type edit_dir_name
    edit_type=$(echo "${resolved}" | jq -r '.edit_type')
    edit_dir_name=$(echo "${resolved}" | jq -r '.edit_dir_name')

    local edit_path="${model_output_dir}/models/${edit_dir_name}"
    local cert_dir="${model_output_dir}/reports/${edit_dir_name}/run_${run_num}"
    local cert_file="${cert_dir}/evaluation.report.json"

    if [[ ! -d "${edit_path}" ]]; then
        echo "ERROR: Edit model not found: ${edit_path}" >> "${log_file}"
        return 1
    fi
    if ! _cmd_python "${SCRIPT_DIR}/../../python/editing/validate_artifact.py" \
        "${edit_path}" \
        --require-metadata \
        --expected-edit-type "${edit_type}" \
        --expected-artifact-class "validation_subject_checkpoint" >/dev/null 2>&1; then
        echo "ERROR: Edit model metadata validation failed: ${edit_path}" >> "${log_file}"
        return 1
    fi

    local abs_baseline_path
    abs_baseline_path="$(cd "$(dirname "${baseline_path}")" && pwd)/$(basename "${baseline_path}")"
    local abs_edit_path
    abs_edit_path="$(cd "$(dirname "${edit_path}")" && pwd)/$(basename "${edit_path}")"

    if [[ -f "${cert_file}" ]]; then
        echo "  Evaluation report for ${edit_dir_name} run ${run_num} already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] evaluating: ${edit_dir_name} run ${run_num}" >> "${log_file}"

    mkdir -p "${cert_dir}"
    if [[ -f "${edit_path}/edit_metadata.json" ]]; then
        cp "${edit_path}/edit_metadata.json" "${cert_dir}/edit_metadata.json"
    fi

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
    # both baseline and edited models simultaneously.
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

    local edit_type=""
    IFS=':' read -r edit_type _ _ _ <<< "${edit_spec}"

    # Find calibrated preset (must have seq_len/stride embedded)
    local preset_file=""
    if [[ -n "${edit_type}" ]]; then
        for ext in yaml json; do
            local f="${preset_dir}/calibrated_preset_${model_name}__${edit_type}.${ext}"
            if [[ -f "${f}" ]]; then
                preset_file="${f}"
                echo "  Using edit-type preset: ${preset_file}" >> "${log_file}"
                break
            fi
        done
    fi
    if [[ -z "${preset_file}" ]]; then
    for ext in yaml json; do
        local f="${preset_dir}/calibrated_preset_${model_name}.${ext}"
        if [[ -f "${f}" ]]; then
            preset_file="${f}"
            break
        fi
    done
    fi

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
    _normalize_staged_preset_for_eval "${abs_preset_file}" "${seq_len}" "${stride}" "${preview_n}" "${final_n}" "${skip_overhead_in_preset}" "${log_file}" "${abs_baseline_report_file:-}" || {
        echo "ERROR: Failed to normalize staged preset for evaluate runtime: ${abs_preset_file}" >> "${log_file}"
        return 1
    }
    local edit_label
    edit_label="$(basename "${abs_edit_path}")"
    edit_label="${edit_label%_stress}"
    edit_label="${edit_label%_clean}"
    if [[ -z "${edit_label}" ]]; then
        edit_label="custom"
    fi

    # CUDA_VISIBLE_DEVICES is inherited from execute_task() for multi-GPU support
    local exit_code=0
    local -a evaluate_extra_args=()
    evaluate_extra_args+=(--timing-json "${cert_dir}/evaluate_timing.json")
    if _pack_defer_report_rendering_enabled; then
        evaluate_extra_args+=(--defer-report-rendering)
    fi
    local evaluate_assurance
    evaluate_assurance="$(_pack_evaluate_assurance_mode)" || {
        echo "ERROR: Invalid evaluate assurance mode" >> "${log_file}"
        return 1
    }
    (
        cd "${work_dir}" || exit 1
        env "${extra_env[@]}" invarlock evaluate \
            --baseline "${abs_baseline_path}" \
            "${baseline_report_args[@]}" \
            --subject "${abs_edit_path}" \
            --edit-label "${edit_label}" \
            --profile "${profile_flag}" \
            --tier "${tier}" \
            --out "${cert_dir}" \
            --report-out "${cert_dir}" \
            --preset "${abs_preset_file}" \
            --assurance "${evaluate_assurance}" \
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

    # Some failure modes (e.g., overhead gate, abort-on-unsafe) still write a
    # structured `report.json` but skip the derived `evaluation.report.json`.
    # Convert when possible so the evidence-pack verdict can still be computed.
    if [[ ! -f "${cert_file}" ]]; then
        local report_file=""
        report_file=$(find "${cert_dir}" -name "report*.json" -type f 2>/dev/null | sort | tail -1)
        if [[ -n "${report_file}" && -f "${report_file}" ]]; then
            local conversion_rc=0
            _cmd_python "${SCRIPT_DIR}/../../python/task_tools.py" evaluation-report \
                --report "${report_file}" \
                --out "${cert_file}" >> "${log_file}" 2>&1 || conversion_rc=$?
            if [[ ${conversion_rc} -ne 0 ]]; then
                echo "  ERROR: failed to generate evaluation.report.json from ${report_file} (exit=${conversion_rc})" >> "${log_file}"
                if [[ ${exit_code} -eq 0 ]]; then
                    exit_code=${conversion_rc}
                fi
            fi
        fi
    fi

    # InvarLock may exit non-zero (e.g., abort-on-unsafe in CI/release profiles)
    # while still writing the canonical report. The evidence-pack harness only needs
    # the report artifact; treat this as success to avoid wasteful retries.
    if [[ ${exit_code} -ne 0 && -f "${cert_file}" ]]; then
        echo "  WARNING: invarlock evaluate exited ${exit_code} but wrote evaluation.report.json; treating as success" >> "${log_file}"
        exit_code=0
    fi

    return ${exit_code}
}

# ============ TASK: CLEANUP_EDIT ============
task_cleanup_edit() {
    local model_name="$1"
    local edit_spec="$2"
    local version="$3"
    local output_dir="$4"
    local log_file="$5"

    if [[ "${PACK_CLEANUP_MODELS:-1}" == "0" ]]; then
        echo "  Cleanup disabled (PACK_CLEANUP_MODELS=0); skipping edit cleanup" >> "${log_file}"
        return 0
    fi

    local model_output_dir="${output_dir}/${model_name}"
    local resolved
    resolved=$(resolve_edit_params "${model_output_dir}" "${edit_spec}" "${version}")
    local status
    status=$(echo "${resolved}" | jq -r '.status')
    if [[ "${status}" == "skipped" ]]; then
        echo "  Clean edit skipped by tuned preset: ${edit_spec}" >> "${log_file}"
        return 0
    fi
    if [[ "${status}" != "selected" ]]; then
        echo "ERROR: Unable to resolve edit spec (${edit_spec}): ${status}" >> "${log_file}"
        return 1
    fi
    local edit_dir_name
    edit_dir_name=$(echo "${resolved}" | jq -r '.edit_dir_name')
    if [[ -z "${edit_dir_name}" || "${edit_dir_name}" == "null" ]]; then
        echo "ERROR: Empty edit_dir_name for ${edit_spec}" >> "${log_file}"
        return 1
    fi

    local models_root="${model_output_dir}/models"
    local models_root_abs
    models_root_abs="$(cd "${models_root}" 2>/dev/null && pwd -P)" || {
        echo "ERROR: Models root missing for cleanup: ${models_root}" >> "${log_file}"
        return 1
    }
    local edit_parent_rel
    edit_parent_rel="$(dirname "${edit_dir_name}")"
    local edit_basename
    edit_basename="$(basename "${edit_dir_name}")"
    local edit_parent_abs
    edit_parent_abs="$(cd "${models_root_abs}/${edit_parent_rel}" 2>/dev/null && pwd -P)" || {
        echo "ERROR: Refusing to delete path outside models root: ${models_root}/${edit_dir_name}" >> "${log_file}"
        return 1
    }
    local edit_path="${edit_parent_abs}/${edit_basename}"
    local baseline_path="${models_root_abs}/baseline"

    if [[ "${edit_path}" == "${baseline_path}" ]]; then
        echo "ERROR: Refusing to delete baseline path: ${edit_path}" >> "${log_file}"
        return 1
    fi
    if [[ "${edit_path}" != "${models_root_abs}/"* ]]; then
        echo "ERROR: Refusing to delete path outside models root: ${edit_path}" >> "${log_file}"
        return 1
    fi
    if [[ ! -e "${edit_path}" ]]; then
        echo "  Edit path already absent: ${edit_path}" >> "${log_file}"
        return 0
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Cleaning up edit model: ${edit_dir_name}" >> "${log_file}"
    rm -rf "${edit_path}" >> "${log_file}" 2>&1 || {
        echo "ERROR: Failed to remove edit path: ${edit_path}" >> "${log_file}"
        return 1
    }

    echo "  Removed: ${edit_path}" >> "${log_file}"
    return 0
}
