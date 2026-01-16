#!/usr/bin/env bash
# lmeval_runner.sh - lm-eval orchestration for proof packs.

# ============ LM-EVAL OPTIMIZATION ============
run_lmeval() {
    local model_path="$1"
    local output_file="$2"
    local tasks="$3"
    local batch_size="$4"
    local num_fewshot="$5"
    local gpu_id="${6:-0}"

    local start_time=$(date +%s)

    # Determine effective batch size based on model size
    local effective_batch_size="${batch_size}"
    if [[ "${batch_size}" == "auto" ]]; then
        local model_size=$(estimate_model_params "${model_path}")
        case "${model_size}" in
            "70"|"72")
                effective_batch_size="${EVAL_BATCH_SIZE_LARGE}"
                ;;
            "40")
                effective_batch_size="${EVAL_BATCH_SIZE_MEDIUM}"
                ;;
            "30")
                effective_batch_size="${EVAL_BATCH_SIZE_MEDIUM}"  # MPT-30B uses medium
                ;;
            "moe")
                effective_batch_size="${EVAL_BATCH_SIZE_MOE}"  # Mixtral/MoE models
                ;;
            *)
                effective_batch_size="${EVAL_BATCH_SIZE_SMALL}"
                ;;
        esac
        # Log with proper label for model size (avoid "moeB params")
        if [[ "${model_size}" == "moe" ]]; then
            log "  📦 MoE model detected, batch size: ${effective_batch_size}"
        else
            log "  📦 Model ~${model_size}B params, batch size: ${effective_batch_size}"
        fi
    fi

    mkdir -p "$(dirname "${output_file}")"

    local model_args="pretrained=${model_path},trust_remote_code=True,dtype=bfloat16"
    local parallelize_flag="${LM_EVAL_PARALLELIZE:-true}"
    parallelize_flag=$(echo "${parallelize_flag}" | tr '[:upper:]' '[:lower:]')
    if [[ "${CUDA_VISIBLE_DEVICES:-}" == *","* && "${parallelize_flag}" != "false" && "${parallelize_flag}" != "0" ]]; then
        model_args="${model_args},parallelize=True"
    fi
    if [[ "${FLASH_ATTENTION_AVAILABLE}" == "true" ]]; then
        # Only enable Flash Attention 2 for architectures known to support it.
        # Reuse the same pattern filter as setup_model() to avoid crashes.
        local model_lower
        model_lower=$(echo "${model_path}" | tr '[:upper:]' '[:lower:]')
        local use_fa2="true"
        local pattern
        for pattern in falcon mpt- gpt2 bloom opt- gpt-j gpt-neo codegen santacoder stablelm; do
            if [[ "${model_lower}" == *"${pattern}"* ]]; then
                use_fa2="false"
                break
            fi
        done
        if [[ "${use_fa2}" == "true" ]]; then
            model_args="${model_args},attn_implementation=flash_attention_2"
        else
            log "  Flash Attention 2 disabled for eval on model path: ${model_path}"
        fi
    fi

    log "  🚀 Starting lm-eval on GPU ${gpu_id}..."

    local exit_code=0
    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    local torch_compile="${LMEVAL_TORCH_COMPILE:-0}"
    CUDA_VISIBLE_DEVICES="${cuda_devices}" \
    TORCH_COMPILE="${torch_compile}" \
    python3 -m lm_eval \
        --model hf \
        --model_args "${model_args}" \
        --tasks "${tasks}" \
        --batch_size "${effective_batch_size}" \
        --num_fewshot "${num_fewshot}" \
        --output_path "$(dirname "${output_file}")" \
        --log_samples \
        2>&1 | tee -a "${OUTPUT_DIR}/logs/gpu_${gpu_id}.log" || exit_code=$?

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    local results_file
    results_file=$(find "$(dirname "${output_file}")" -name "results*.json" -type f 2>/dev/null | head -1)
    if [[ -n "${results_file}" ]]; then
        if mv "${results_file}" "${output_file}"; then
            log "  ✅ Results saved: ${output_file} (${duration}s)"
        else
            log "  ⚠️  Failed to move results to: ${output_file}"
            exit_code=1
        fi
    else
        log "  ⚠️  No results file found"
        [[ ${exit_code} -eq 0 ]] && exit_code=1
    fi

    return ${exit_code}
}
export -f run_lmeval

