#!/usr/bin/env bash
# config_generator.sh - InvarLock config generation + certify helpers for proof packs.

_PACK_CONFIG_GENERATOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PACK_CONFIG_GENERATOR_PY_DIR="${_PACK_CONFIG_GENERATOR_DIR}/../python"

# ============ INVARLOCK CONFIG FOR PROOF PACKS ============
generate_invarlock_config() {
    local model_path="$1"
    local output_yaml="$2"
    local edit_name="${3:-noop}"
    local seed="${4:-42}"
    local preview_n="${5:-${INVARLOCK_PREVIEW_WINDOWS}}"
    local final_n="${6:-${INVARLOCK_FINAL_WINDOWS}}"
    local bootstrap_n="${7:-${INVARLOCK_BOOTSTRAP_N:-2000}}"
    local seq_len="${8:-${INVARLOCK_SEQ_LEN}}"
    local stride="${9:-${INVARLOCK_STRIDE}}"
    local eval_batch="${10:-${INVARLOCK_EVAL_BATCH}}"

    # Use auto adapter for generic causal LM support (Mistral, Mixtral, Qwen, MPT, Falcon, etc.)
    local adapter="hf_auto"
    local dataset_provider="${INVARLOCK_DATASET}"

    local attn_impl_yaml=""
    if [[ "${FLASH_ATTENTION_AVAILABLE}" == "true" ]]; then
        attn_impl_yaml='attn_implementation: "flash_attention_2"'
    else
        attn_impl_yaml='# flash_attention_2 not available'
    fi
    local accel_compile="true"
    local accel_tf32="true"
    local accel_benchmark="true"
    if [[ "${PACK_DETERMINISM}" == "strict" ]]; then
        accel_compile="false"
        accel_tf32="false"
        accel_benchmark="false"
    fi

    # Window overlap control (calibration and eval safety)
    local eval_overlap="${INVARLOCK_WINDOW_OVERLAP_FRACTION:-0.0}"

    # Optional: override guard order for the suite (comma-separated list).
    # Default is a lightweight chain to keep calibration tractable on 70B+.
    local guards_order_csv="${PACK_GUARDS_ORDER:-}"
    local -a guards_order=()
    if [[ -n "${guards_order_csv}" ]]; then
        IFS=',' read -ra guards_order <<< "${guards_order_csv}"
    else
        guards_order=("invariants" "spectral" "rmt" "variance" "invariants")
    fi
    local guards_order_yaml=""
    local g
    for g in "${guards_order[@]}"; do
        g="$(echo "${g}" | xargs)"
        [[ -z "${g}" ]] && continue
        guards_order_yaml+=$'    - '"${g}"$'\n'
    done
    if [[ -z "${guards_order_yaml}" ]]; then
        guards_order_yaml=$'    - invariants\n    - spectral\n    - rmt\n    - variance\n    - invariants\n'
    fi

    cat > "${output_yaml}" << YAML_EOF
# Auto-generated InvarLock config for proof packs
# Platform: proof pack runner


model:
  id: "${model_path}"
  adapter: "${adapter}"
  device: "auto"
  device_map: "auto"
  torch_dtype: "bfloat16"
  trust_remote_code: true
  low_cpu_mem_usage: true
  ${attn_impl_yaml}

dataset:
  provider: "${dataset_provider}"
  preview_n: ${preview_n}
  final_n: ${final_n}
  seq_len: ${seq_len}
  stride: ${stride}
  seed: ${seed}
  num_workers: 8
  prefetch_factor: 4
  pin_memory: true

edit:
  name: "${edit_name}"

guards:
  order:
${guards_order_yaml}

eval:
  window_overlap_fraction: ${eval_overlap}
  bootstrap:
    replicates: ${bootstrap_n}
    parallel: true
  max_pm_ratio: 2.0
  batch_size: ${eval_batch}


auto:
  enabled: true
  tier: "${INVARLOCK_TIER}"
  probes: 0

output:
  dir: "."

accelerator:
  compile: ${accel_compile}
  tf32: ${accel_tf32}
  benchmark: ${accel_benchmark}
  memory_efficient_attention: false
  gradient_checkpointing: false

memory:
  target_fraction: 0.92
  preallocate: true
  cache_enabled: true
YAML_EOF
}
export -f generate_invarlock_config

# ============ CALIBRATION RUN ============
run_single_calibration() {
    local model_path="$1"
    local run_dir="$2"
    local seed="$3"
    local preview_n="$4"
    local final_n="$5"
    local bootstrap_n="$6"
    local log_file="$7"
    local gpu_id="${8:-0}"
    local seq_len="${9:-${INVARLOCK_SEQ_LEN}}"
    local stride="${10:-${INVARLOCK_STRIDE}}"
    local eval_batch="${11:-${INVARLOCK_EVAL_BATCH}}"

    mkdir -p "${run_dir}"
    local config_yaml="${run_dir}/calibration_config.yaml"

    INVARLOCK_WINDOW_OVERLAP_FRACTION=0.0 generate_invarlock_config \
        "${model_path}" \
        "${config_yaml}" \
        "noop" \
        "${seed}" \
        "${preview_n}" \
        "${final_n}" \
        "${bootstrap_n}" \
        "${seq_len}" \
        "${stride}" \
        "${eval_batch}"

    # For large models, skip overhead check to avoid OOM (task-local via env)
    local model_size
    model_size=$(estimate_model_params "${model_path}")

    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"

    local exit_code=0
    # Enforce no-overlap windows and skip overhead checks to avoid E001/pairing issues

    if [[ "${model_size}" == "70" || "${model_size}" == "72" || "${model_size}" == "moe" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Large model (${model_size}): using INVARLOCK_SKIP_OVERHEAD_CHECK=1 for calibration" >> "${log_file}"
    fi

    INVARLOCK_WINDOW_OVERLAP_FRACTION=0.0 \
    INVARLOCK_SKIP_OVERHEAD_CHECK=1 \
    CUDA_VISIBLE_DEVICES="${cuda_devices}" invarlock run \
        --config "${config_yaml}" \
        --profile ci \
        --out "${run_dir}" \
        >> "${log_file}" 2>&1 || exit_code=$?

    # Generate certificate from report
    local report_file
    report_file=$(find "${run_dir}" -name "report*.json" -type f 2>/dev/null | head -1)
    if [[ -n "${report_file}" ]]; then
        cp "${report_file}" "${run_dir}/baseline_report.json" 2>/dev/null || true
        python3 "${_PACK_CONFIG_GENERATOR_PY_DIR}/certificate_from_report.py" \
            --report "${report_file}" \
            --out "${run_dir}/evaluation.cert.json" >> "${log_file}" 2>&1 || true
    fi

    return ${exit_code}
}
export -f run_single_calibration

# ============ CALIBRATION ORCHESTRATION ============
run_invarlock_calibration() {
    local model_path="$1"
    local model_name="$2"
    local output_dir="$3"
    local num_runs="$4"
    local preset_output_dir="$5"
    local gpu_id="${6:-0}"

    local model_size
    model_size=$(estimate_model_params "${model_path}")
    local bootstrap_n="${INVARLOCK_BOOTSTRAP_N:-2000}"

    # Get model-size-aware configuration
    local config
    config=$(get_model_invarlock_config "${model_size}")
    IFS=':' read -r effective_seq_len effective_stride effective_preview_n effective_final_n effective_eval_batch <<< "${config}"
    # Force non-overlapping windows for calibration to avoid pairing mismatches
    effective_stride="${effective_seq_len}"
    export INVARLOCK_WINDOW_OVERLAP_FRACTION=0.0

    # Log calibration start with proper model size label
    if [[ "${model_size}" == "moe" ]]; then
        log "  Calibration: ${num_runs} runs on GPU ${gpu_id} (MoE architecture)"
    else
        log "  Calibration: ${num_runs} runs on GPU ${gpu_id} (${model_size}B params)"
    fi
    log "    Config: seq_len=${effective_seq_len}, stride=${effective_stride}, windows=${effective_preview_n}+${effective_final_n}"

    mkdir -p "${output_dir}" "${preset_output_dir}"

    local calibration_failures=0
    for run in $(seq 1 "${num_runs}"); do
        local seed=$((41 + run))
        local run_dir="${output_dir}/run_${run}"
        local run_log="${OUTPUT_DIR}/logs/calibration_${model_name}_run${run}.log"

        if ! run_single_calibration \
            "${model_path}" \
            "${run_dir}" \
            "${seed}" \
            "${effective_preview_n}" \
            "${effective_final_n}" \
            "${bootstrap_n}" \
            "${run_log}" \
            "${gpu_id}" \
            "${effective_seq_len}" \
            "${effective_stride}" \
            "${effective_eval_batch}"; then
            log "  WARNING: Calibration run ${run} failed for ${model_name}"
            calibration_failures=$((calibration_failures + 1))
        fi
    done

    if [[ ${calibration_failures} -eq ${num_runs} ]]; then
        log "  ERROR: All calibration runs failed for ${model_name}"
        log "         Skipping preset generation (no valid calibration data)"
        return 1
    fi

    # Generate calibrated preset
    local safe_model_name="${model_name//\//_}"
    local preset_file="${preset_output_dir}/calibrated_preset_${safe_model_name}.yaml"
    python3 "${_PACK_CONFIG_GENERATOR_PY_DIR}/preset_generator.py" \
        --cal-dir "${output_dir}" \
        --preset-file "${preset_file}" \
        --model-name "${model_name}" \
        --model-path "${model_path}" \
        --tier "${INVARLOCK_TIER}" \
        --dataset-provider "${INVARLOCK_DATASET}" \
        --seq-len "${effective_seq_len}" \
        --stride "${effective_stride}" \
        --preview-n "${effective_preview_n}" \
        --final-n "${effective_final_n}"
}

# ============ CERTIFY WITH PROOF PACK SETTINGS ============
run_invarlock_certify() {
    local subject_path="$1"
    local baseline_path="$2"
    local output_dir="$3"
    local run_name="$4"
    local preset_dir="$5"
    local model_name="$6"
    local gpu_id="${7:-0}"

    local run_dir="${output_dir}/${run_name}"
    local cert_dir="${run_dir}/cert"
    mkdir -p "${run_dir}" "${cert_dir}"

    local calibrated_preset=""
    for ext in yaml json; do
        local preset_path="${preset_dir}/calibrated_preset_${model_name}.${ext}"
        if [[ -f "${preset_path}" ]]; then
            calibrated_preset="${preset_path}"
            break
        fi
    done

    local cmd_args=(
        "invarlock" "certify"
        "--source" "${baseline_path}"
        "--edited" "${subject_path}"
        "--profile" "ci"
        "--tier" "${INVARLOCK_TIER}"
        "--out" "${run_dir}"
        "--cert-out" "${cert_dir}"
    )

    if [[ -n "${calibrated_preset}" && -f "${calibrated_preset}" ]]; then
        cmd_args+=("--preset" "${calibrated_preset}")
    fi

    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"

    local exit_code=0
    # For large models, skip overhead check to avoid OOM (task-local via env)
    local model_size
    model_size=$(estimate_model_params "${baseline_path}")
    if [[ "${model_size}" == "70" || "${model_size}" == "72" || "${model_size}" == "moe" ]]; then
        INVARLOCK_SKIP_OVERHEAD_CHECK=1 \
        CUDA_VISIBLE_DEVICES="${cuda_devices}" "${cmd_args[@]}" \
            >> "${OUTPUT_DIR}/logs/gpu_${gpu_id}.log" 2>&1 || exit_code=$?
    else
        CUDA_VISIBLE_DEVICES="${cuda_devices}" "${cmd_args[@]}" \
            >> "${OUTPUT_DIR}/logs/gpu_${gpu_id}.log" 2>&1 || exit_code=$?
    fi

    # Copy certificate to standard location (only the canonical cert)
    local cert_file="${cert_dir}/evaluation.cert.json"
    if [[ -f "${cert_file}" ]]; then
        cp "${cert_file}" "${run_dir}/evaluation.cert.json" 2>/dev/null || true
    else
        local alt_cert
        alt_cert=$(find "${cert_dir}" -name "evaluation.cert.json" -type f 2>/dev/null | head -1)
        if [[ -n "${alt_cert}" && -f "${alt_cert}" ]]; then
            cp "${alt_cert}" "${run_dir}/evaluation.cert.json" 2>/dev/null || true
        fi
    fi

    return ${exit_code}
}
export -f run_invarlock_certify
