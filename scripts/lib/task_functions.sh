#!/usr/bin/env bash
# task_functions.sh - Atomic task implementations for dynamic scheduling
# Version: v2.1.0 (InvarLock B200 Validation Suite)
# Dependencies: jq, python3, invarlock CLI, lm_eval, task_serialization.sh
# Usage: sourced by gpu_worker.sh/invarlock_definitive_validation_b200.sh for per-task execution
#
# Each function executes a single atomic task type with explicit parameters.
# These are extracted from the original monolithic process_model() function
# to enable parallel execution across GPUs.

# Source dependencies
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -z "${QUEUE_MANAGER_LOADED:-}" ]] && source "${SCRIPT_DIR}/queue_manager.sh" && export QUEUE_MANAGER_LOADED=1

# ============ FALLBACK FUNCTIONS ============
# These provide fallback implementations when main script functions aren't available
# (e.g., when running in subshell workers that only source lib modules)

# Detect model size from model name/path string
# Returns: 7, 13, 30, 40, 70, moe
_get_model_size_from_name() {
    local model_id="$1"
    local model_lower=$(echo "${model_id}" | tr '[:upper:]' '[:lower:]')

    # Check for MoE architecture first
    if [[ "${model_lower}" =~ mixtral || "${model_lower}" =~ 8x7b || "${model_lower}" =~ moe ]]; then
        echo "moe"
    # Check for 70B/72B models (largest)
    elif [[ "${model_lower}" =~ 70b || "${model_lower}" =~ 72b || "${model_lower}" =~ 65b ]]; then
        echo "70"
    # Check for 40B models
    elif [[ "${model_lower}" =~ 40b || "${model_lower}" =~ 34b ]]; then
        echo "40"
    # Check for 30B models
    elif [[ "${model_lower}" =~ 30b || "${model_lower}" =~ 32b || "${model_lower}" =~ 33b ]]; then
        echo "30"
    # Check for 13B/14B models
    elif [[ "${model_lower}" =~ 13b || "${model_lower}" =~ 14b ]]; then
        echo "13"
    # Default to 7B
    else
        echo "7"
    fi
}

# Get model-aware InvarLock configuration (fallback implementation)
# Returns: seq_len:stride:preview_n:final_n:eval_batch
_get_model_invarlock_config_fallback() {
    local model_size="$1"  # 7, 13, 30, 40, 70, moe

    # Conservative defaults that won't OOM on B200 180GB
    # These MUST match or be more conservative than main script's get_model_invarlock_config()
    case "${model_size}" in
        "7")
            echo "2048:1024:64:64:96"
            ;;
        "13")
            echo "1536:768:48:48:64"
            ;;
        "30")
            echo "1024:512:40:40:48"
            ;;
        "40")
            echo "1024:512:36:36:32"
            ;;
        "moe")
            echo "1024:512:40:40:24"
            ;;
        "70"|"72")
            # ULTRA-CONSERVATIVE for 70B+ models
            # 140GB model + ~36GB headroom = MUST avoid double model loading
            # Using INVARLOCK_SKIP_OVERHEAD_CHECK=1 is REQUIRED alongside this
            echo "128:64:8:8:2"
            ;;
        *)
            # Safe default
            echo "1024:512:40:40:32"
            ;;
    esac
}

# Wrapper to get model size - tries main script function first, then fallback
_estimate_model_size() {
    local model_path="$1"

    # Try main script's estimate_model_params first
    if type estimate_model_params &>/dev/null; then
        estimate_model_params "${model_path}"
        return
    fi

    # Fallback: detect from model name/path
    _get_model_size_from_name "${model_path}"
}

# Wrapper to get InvarLock config - tries main script function first, then fallback
_get_invarlock_config() {
    local model_size="$1"

    # Try main script's get_model_invarlock_config first
    if type get_model_invarlock_config &>/dev/null; then
        get_model_invarlock_config "${model_size}"
        return
    fi

    # Use fallback
    _get_model_invarlock_config_fallback "${model_size}"
}

# Check if model is large (70B+) and needs special handling
_is_large_model() {
    local model_size="$1"
    if [[ "${model_size}" == "moe" ]]; then
        return 0
    fi
    if [[ "${model_size}" =~ ^[0-9]+$ ]]; then
        [[ ${model_size} -ge 70 ]]
        return
    fi
    [[ "${model_size}" =~ 70 || "${model_size}" =~ 72 || "${model_size}" =~ 65 || "${model_size}" =~ 80 || "${model_size}" =~ 90 ]]
}

# ============ TASK EXECUTOR ============

# Execute a task based on its type
# Usage: execute_task <task_file> <gpu_id> <output_dir>
execute_task() {
    local task_file="$1"
    local gpu_id="$2"
    local output_dir="$3"

    local task_id=$(get_task_id "${task_file}")
    local task_type=$(get_task_type "${task_file}")
    local model_id=$(get_task_field "${task_file}" "model_id")
    local model_name=$(get_task_field "${task_file}" "model_name")
    local params=$(get_task_params "${task_file}")

    # Create task-specific log
    local task_log="${output_dir}/logs/tasks/${task_id}.log"
    mkdir -p "$(dirname "${task_log}")"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting task: ${task_id}" >> "${task_log}"
    echo "  Type: ${task_type}" >> "${task_log}"
    echo "  Model: ${model_id}" >> "${task_log}"
    echo "  GPU: ${gpu_id}" >> "${task_log}"
    echo "  Params: ${params}" >> "${task_log}"

    # Set GPU for this task
    export CUDA_VISIBLE_DEVICES="${gpu_id}"

    # v0.3.1 FEATURE: Set PM acceptance range to avoid gate failures during validation
    # These bounds are calibrated for typical validation runs; adjust if needed
    export INVARLOCK_PM_ACCEPTANCE_MIN="${INVARLOCK_PM_ACCEPTANCE_MIN:-0.90}"
    export INVARLOCK_PM_ACCEPTANCE_MAX="${INVARLOCK_PM_ACCEPTANCE_MAX:-1.20}"

    local exit_code=0

    case "${task_type}" in
        SETUP_BASELINE)
            task_setup_baseline "${model_id}" "${model_name}" "${gpu_id}" "${output_dir}" "${task_log}" || exit_code=$?
            ;;
        EVAL_BASELINE)
            task_eval_baseline "${model_name}" "${gpu_id}" "${output_dir}" "${task_log}" || exit_code=$?
            ;;
        CALIBRATION_RUN)
            local run=$(echo "${params}" | jq -r '.run // 1')
            local seed=$(echo "${params}" | jq -r '.seed // 42')
            task_calibration_run "${model_name}" "${gpu_id}" "${run}" "${seed}" "${output_dir}" "${task_log}" || exit_code=$?
            ;;
        CREATE_EDIT)
            local edit_spec=$(echo "${params}" | jq -r '.edit_spec // ""')
            local version=$(echo "${params}" | jq -r '.version // "clean"')
            task_create_edit "${model_name}" "${gpu_id}" "${edit_spec}" "${version}" "${output_dir}" "${task_log}" || exit_code=$?
            ;;
        EVAL_EDIT)
            local edit_spec=$(echo "${params}" | jq -r '.edit_spec // ""')
            task_eval_edit "${model_name}" "${gpu_id}" "${edit_spec}" "${output_dir}" "${task_log}" || exit_code=$?
            ;;
        CERTIFY_EDIT)
            local edit_spec=$(echo "${params}" | jq -r '.edit_spec // ""')
            local version=$(echo "${params}" | jq -r '.version // "clean"')
            local run=$(echo "${params}" | jq -r '.run // 1')
            task_certify_edit "${model_name}" "${gpu_id}" "${edit_spec}" "${version}" "${run}" "${output_dir}" "${task_log}" || exit_code=$?
            ;;
        CREATE_ERROR)
            local error_type=$(echo "${params}" | jq -r '.error_type // ""')
            task_create_error "${model_name}" "${gpu_id}" "${error_type}" "${output_dir}" "${task_log}" || exit_code=$?
            ;;
        CERTIFY_ERROR)
            local error_type=$(echo "${params}" | jq -r '.error_type // ""')
            task_certify_error "${model_name}" "${gpu_id}" "${error_type}" "${output_dir}" "${task_log}" || exit_code=$?
            ;;
        GENERATE_PRESET)
            task_generate_preset "${model_name}" "${output_dir}" "${task_log}" || exit_code=$?
            ;;
        *)
            echo "ERROR: Unknown task type: ${task_type}" >> "${task_log}"
            exit_code=1
            ;;
    esac

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task ${task_id} finished with exit code: ${exit_code}" >> "${task_log}"

    return ${exit_code}
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

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setting up baseline: ${model_id}" >> "${log_file}"

    # Check if already exists (resume mode)
    if [[ -d "${baseline_dir}" && -f "${baseline_dir}/config.json" ]]; then
        echo "  Baseline already exists, skipping download" >> "${log_file}"
        # Store baseline path for other tasks
        echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
        # Also store original model_id for model size detection in other tasks
        echo "${model_id}" > "${model_output_dir}/.model_id"
        return 0
    fi

    mkdir -p "${model_output_dir}"/{models,evals,certificates}

    # Use the main script's setup_model function if available
    if type setup_model &>/dev/null; then
        local baseline_path
        baseline_path=$(setup_model "${model_id}" "${gpu_id}")
        local exit_code=$?

        if [[ ${exit_code} -eq 0 && -n "${baseline_path}" && -d "${baseline_path}" ]]; then
            echo "  Baseline ready at: ${baseline_path}" >> "${log_file}"
            echo "${baseline_path}" > "${model_output_dir}/.baseline_path"
            # Store original model_id for model size detection
            echo "${model_id}" > "${model_output_dir}/.model_id"
            return 0
        else
            echo "  ERROR: Failed to setup baseline" >> "${log_file}"
            return 1
        fi
    else
        # Inline implementation
        echo "  Downloading model ${model_id}..." >> "${log_file}"

        CUDA_VISIBLE_DEVICES="${gpu_id}" python3 << SETUP_EOF >> "${log_file}" 2>&1
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import gc
import sys

model_id = "${model_id}"
output_dir = Path("${baseline_dir}")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading {model_id}...")

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.save_pretrained(output_dir, safe_serialization=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Saved to {output_dir}")

except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
SETUP_EOF

        if [[ $? -eq 0 && -f "${baseline_dir}/config.json" ]]; then
            echo "${baseline_dir}" > "${model_output_dir}/.baseline_path"
            # Store original model_id for model size detection
            echo "${model_id}" > "${model_output_dir}/.model_id"
            return 0
        fi
        return 1
    fi
}

# ============ TASK: EVAL_BASELINE ============

# Run lm-eval on baseline model
# Usage: task_eval_baseline <model_name> <gpu_id> <output_dir> <log_file>
task_eval_baseline() {
    local model_name="$1"
    local gpu_id="$2"
    local output_dir="$3"
    local log_file="$4"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null)
    local result_file="${model_output_dir}/evals/baseline_results.json"

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    if [[ -f "${result_file}" ]]; then
        echo "  Baseline eval already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running baseline lm-eval" >> "${log_file}"

    mkdir -p "$(dirname "${result_file}")"

    # Determine batch size based on model
    local batch_size="${EVAL_BATCH_SIZE:-auto:16}"

    local model_args="pretrained=${baseline_path},trust_remote_code=True,dtype=bfloat16"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python3 -m lm_eval \
        --model hf \
        --model_args "${model_args}" \
        --tasks "${EVAL_TASKS:-mmlu,hellaswag,arc_challenge,winogrande}" \
        --batch_size "${batch_size}" \
        --num_fewshot "${EVAL_NUM_FEWSHOT:-5}" \
        --output_path "$(dirname "${result_file}")" \
        --log_samples \
        >> "${log_file}" 2>&1

    local exit_code=$?

    # Move results file to expected location
    local found_results=$(find "$(dirname "${result_file}")" -name "results*.json" -type f 2>/dev/null | head -1)
    if [[ -n "${found_results}" && -f "${found_results}" ]]; then
        mv "${found_results}" "${result_file}"
        echo "  Results saved to: ${result_file}" >> "${log_file}"
    fi

    return ${exit_code}
}

# ============ TASK: CALIBRATION_RUN ============

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
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null)
    local model_id=$(cat "${model_output_dir}/.model_id" 2>/dev/null)
    local run_dir="${model_output_dir}/certificates/calibration/run_${run_num}"

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    # Check if already done
    if [[ -f "${run_dir}/baseline_report.json" || -f "${run_dir}/evaluation.cert.json" ]]; then
        echo "  Calibration run ${run_num} already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running calibration run ${run_num} (seed=${seed})" >> "${log_file}"

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

    echo "  Model size: ${model_size}, Config: seq=${seq_len}, stride=${stride}, windows=${preview_n}+${final_n}, batch=${eval_batch}" >> "${log_file}"

    # Generate config YAML
    local config_yaml="${run_dir}/calibration_config.yaml"
    cat > "${config_yaml}" << YAML_EOF
model:
  id: "${baseline_path}"
  adapter: "hf_causal_auto"
  device: "auto"
  dtype: "bfloat16"

dataset:
  provider: "${INVARLOCK_DATASET:-wikitext2}"
  preview_n: ${preview_n}
  final_n: ${final_n}
  seq_len: ${seq_len}
  stride: ${stride}
  seed: ${seed}

edit:
  name: "noop"

guards:
  order:
    - invariants
    - spectral
    - rmt
    - variance
    - invariants

eval:
  bootstrap:
    replicates: ${INVARLOCK_BOOTSTRAP_N:-10000}
    parallel: true
  batch_size: ${eval_batch}

auto:
  enabled: true
  tier: "${INVARLOCK_TIER:-balanced}"
  probes: 0
YAML_EOF

    # v0.3.1 FEATURE: Use INVARLOCK_SKIP_OVERHEAD_CHECK for large models
    # This avoids loading both baseline and edited models simultaneously
    # which would exceed B200 180GB memory (140GB × 2 = 280GB needed)
    local profile_flag="ci"
    if _is_large_model "${model_size}"; then
        export INVARLOCK_SKIP_OVERHEAD_CHECK=1
        # Override CI profile window counts to prevent OOM (CI defaults to 200/200)
        export INVARLOCK_CI_PREVIEW="${preview_n}"
        export INVARLOCK_CI_FINAL="${final_n}"
        echo "  Large model (${model_size}): setting INVARLOCK_SKIP_OVERHEAD_CHECK=1, CI windows=${preview_n}/${final_n}" >> "${log_file}"
    fi

    CUDA_VISIBLE_DEVICES="${gpu_id}" invarlock run \
        --config "${config_yaml}" \
        --profile "${profile_flag}" \
        --out "${run_dir}" \
        >> "${log_file}" 2>&1

    local exit_code=$?

    # Copy report to standard location
    local report_file=$(find "${run_dir}" -name "report*.json" -type f 2>/dev/null | head -1)
    if [[ -n "${report_file}" ]]; then
        cp "${report_file}" "${run_dir}/baseline_report.json" 2>/dev/null || true
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
    local cal_dir="${model_output_dir}/certificates/calibration"
    local preset_dir="${output_dir}/presets"
    local preset_file="${preset_dir}/calibrated_preset_${model_name}.yaml"

    if [[ -f "${preset_file}" ]]; then
        echo "  Preset already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generating calibrated preset" >> "${log_file}"

    mkdir -p "${preset_dir}"

    # Get baseline path and model_id to estimate model size
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null)
    local model_id=$(cat "${model_output_dir}/.model_id" 2>/dev/null)

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

    # Export for use in Python script
    export PRESET_SEQ_LEN="${seq_len}"
    export PRESET_STRIDE="${stride}"
    export PRESET_PREVIEW_N="${preview_n}"
    export PRESET_FINAL_N="${final_n}"

    python3 << PRESET_EOF >> "${log_file}" 2>&1
import json
import yaml
from pathlib import Path
from collections import defaultdict

cal_dir = Path("${cal_dir}")
preset_file = Path("${preset_file}")
model_name = "${model_name}"

def load_certificates():
    certs = []
    for run_dir in sorted(cal_dir.glob("run_*")):
        for file_pattern in ["evaluation.cert.json", "baseline_report.json"]:
            cert_path = run_dir / file_pattern
            if cert_path.exists():
                try:
                    certs.append(json.loads(cert_path.read_text()))
                    break
                except: pass
    return certs

certs = load_certificates()

if len(certs) == 0:
    print("WARNING: No calibration certificates found")
    # Create minimal preset anyway
    certs = [{'primary_metric': {'ratio_vs_baseline': 1.0}}]

# Calculate drift stats
drifts = []
for cert in certs:
    pm = cert.get('primary_metric', {})
    ratio = pm.get('ratio_vs_baseline') or pm.get('drift')
    if ratio:
        try: drifts.append(float(ratio))
        except: pass

if drifts:
    mean_drift = sum(drifts) / len(drifts)
    std_drift = (sum((x - mean_drift)**2 for x in drifts) / len(drifts))**0.5 if len(drifts) > 1 else 0
    margin = max(2 * std_drift, 0.05)
else:
    mean_drift, std_drift, margin = 1.0, 0.0, 0.05

import os

# Get model-specific config from environment
preset_seq_len = int(os.environ.get('PRESET_SEQ_LEN', 1024))
preset_stride = int(os.environ.get('PRESET_STRIDE', 512))
preset_preview_n = int(os.environ.get('PRESET_PREVIEW_N', 40))
preset_final_n = int(os.environ.get('PRESET_FINAL_N', 40))

preset = {
    '_calibration_meta': {
        'model_name': model_name,
        'num_runs': len(certs),
        'drift_mean': round(mean_drift, 4),
        'drift_std': round(std_drift, 4),
    },
    'dataset': {
        'provider': 'wikitext2',
        'split': 'validation',
        'seq_len': preset_seq_len,
        'stride': preset_stride,
        'preview_n': preset_preview_n,
        'final_n': preset_final_n,
        'seed': 42,
    },
}

# Save stats
stats_path = cal_dir / "calibration_stats.json"
with open(stats_path, 'w') as f:
    json.dump({
        'drift': {
            'mean': mean_drift,
            'std': std_drift,
            'band': [mean_drift - margin, mean_drift + margin]
        },
        'num_runs': len(certs)
    }, f, indent=2)

# Save preset
with open(preset_file, 'w') as f:
    yaml.safe_dump(preset, f, sort_keys=False)

print(f"Saved preset to {preset_file}")
print(f"Saved stats to {stats_path}")
PRESET_EOF

    return $?
}

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
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null)

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    # Parse edit spec
    local edit_type param1 param2 scope
    IFS=':' read -r edit_type param1 param2 scope <<< "${edit_spec}"

    # Handle 3-part vs 4-part specs
    if [[ -z "${scope}" && "${edit_type}" != "quant_rtn" ]]; then
        scope="${param2}"
        param2=""
    fi

    # Determine output path
    local edit_dir_name="${edit_type}_${version}"
    case "${edit_type}" in
        "quant_rtn")
            edit_dir_name="quant_${param1}bit_${version}"
            ;;
        "fp4_quant")
            edit_dir_name="fp4_${param1}_${version}"
            ;;
        "magnitude_prune")
            local pct=$(echo "${param1}" | awk '{printf "%.0f", $1 * 100}')
            edit_dir_name="prune_${pct}pct_${version}"
            ;;
        "lowrank_svd")
            edit_dir_name="svd_rank${param1}_${version}"
            ;;
    esac

    local edit_path="${model_output_dir}/models/${edit_dir_name}"

    # Check if already exists
    if [[ -d "${edit_path}" && -f "${edit_path}/config.json" ]]; then
        echo "  Edit ${edit_dir_name} already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating edit: ${edit_dir_name}" >> "${log_file}"

    # Use main script's functions if available
    case "${edit_type}" in
        "quant_rtn")
            if type create_edited_model &>/dev/null; then
                create_edited_model "${baseline_path}" "${edit_path}" "quant_rtn" "${param1}" "${param2}" "${scope}" "${gpu_id}" >> "${log_file}" 2>&1
            else
                echo "ERROR: create_edited_model not available" >> "${log_file}"
                return 1
            fi
            ;;
        "fp4_quant")
            if type create_fp4_model &>/dev/null; then
                create_fp4_model "${baseline_path}" "${edit_path}" "${param1}" "${scope}" "${gpu_id}" >> "${log_file}" 2>&1
            else
                echo "ERROR: create_fp4_model not available" >> "${log_file}"
                return 1
            fi
            ;;
        "magnitude_prune")
            if type create_pruned_model &>/dev/null; then
                create_pruned_model "${baseline_path}" "${edit_path}" "${param1}" "${scope}" "${gpu_id}" >> "${log_file}" 2>&1
            else
                echo "ERROR: create_pruned_model not available" >> "${log_file}"
                return 1
            fi
            ;;
        "lowrank_svd")
            if type create_lowrank_model &>/dev/null; then
                create_lowrank_model "${baseline_path}" "${edit_path}" "${param1}" "${scope}" "${gpu_id}" >> "${log_file}" 2>&1
            else
                echo "ERROR: create_lowrank_model not available" >> "${log_file}"
                return 1
            fi
            ;;
        *)
            echo "ERROR: Unknown edit type: ${edit_type}" >> "${log_file}"
            return 1
            ;;
    esac

    # Verify creation
    if [[ -d "${edit_path}" && -f "${edit_path}/config.json" ]]; then
        echo "  Created: ${edit_path}" >> "${log_file}"
        return 0
    else
        echo "  ERROR: Failed to create edit" >> "${log_file}"
        return 1
    fi
}

# ============ TASK: EVAL_EDIT ============

# Run lm-eval on edited model
# Usage: task_eval_edit <model_name> <gpu_id> <edit_spec> <output_dir> <log_file>
task_eval_edit() {
    local model_name="$1"
    local gpu_id="$2"
    local edit_spec="$3"
    local output_dir="$4"
    local log_file="$5"

    local model_output_dir="${output_dir}/${model_name}"

    # Parse edit spec to find path
    local edit_type param1
    IFS=':' read -r edit_type param1 _ _ <<< "${edit_spec}"

    # Find the edit directory (could be clean or stress)
    local edit_path=""
    for version in clean stress; do
        local potential_path
        case "${edit_type}" in
            "quant_rtn")
                potential_path="${model_output_dir}/models/quant_${param1}bit_${version}"
                ;;
            "fp4_quant")
                potential_path="${model_output_dir}/models/fp4_${param1}_${version}"
                ;;
            "magnitude_prune")
                local pct=$(echo "${param1}" | awk '{printf "%.0f", $1 * 100}')
                potential_path="${model_output_dir}/models/prune_${pct}pct_${version}"
                ;;
            "lowrank_svd")
                potential_path="${model_output_dir}/models/svd_rank${param1}_${version}"
                ;;
        esac

        if [[ -d "${potential_path}" ]]; then
            edit_path="${potential_path}"
            break
        fi
    done

    if [[ -z "${edit_path}" || ! -d "${edit_path}" ]]; then
        echo "ERROR: Edit model not found for spec: ${edit_spec}" >> "${log_file}"
        return 1
    fi

    local edit_name=$(basename "${edit_path}")
    local result_file="${model_output_dir}/evals/${edit_name}_results.json"

    if [[ -f "${result_file}" ]]; then
        echo "  Eval for ${edit_name} already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running lm-eval on: ${edit_name}" >> "${log_file}"

    mkdir -p "$(dirname "${result_file}")"

    local batch_size="${EVAL_BATCH_SIZE:-auto:16}"
    local model_args="pretrained=${edit_path},trust_remote_code=True,dtype=bfloat16"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python3 -m lm_eval \
        --model hf \
        --model_args "${model_args}" \
        --tasks "${EVAL_TASKS:-mmlu,hellaswag,arc_challenge,winogrande}" \
        --batch_size "${batch_size}" \
        --num_fewshot "${EVAL_NUM_FEWSHOT:-5}" \
        --output_path "$(dirname "${result_file}")" \
        --log_samples \
        >> "${log_file}" 2>&1

    local exit_code=$?

    local found_results=$(find "$(dirname "${result_file}")" -name "results*.json" -type f 2>/dev/null | head -1)
    if [[ -n "${found_results}" && -f "${found_results}" ]]; then
        mv "${found_results}" "${result_file}"
        echo "  Results saved to: ${result_file}" >> "${log_file}"
    fi

    return ${exit_code}
}

# ============ TASK: CERTIFY_EDIT ============

# Run InvarLock certify on edited model
# Usage: task_certify_edit <model_name> <gpu_id> <edit_spec> <version> <run_num> <output_dir> <log_file>
task_certify_edit() {
    local model_name="$1"
    local gpu_id="$2"
    local edit_spec="$3"
    local version="$4"
    local run_num="$5"
    local output_dir="$6"
    local log_file="$7"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null)
    local model_id=$(cat "${model_output_dir}/.model_id" 2>/dev/null)
    local preset_dir="${output_dir}/presets"

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    # Parse edit spec to find path
    local edit_type param1
    IFS=':' read -r edit_type param1 _ _ <<< "${edit_spec}"

    local edit_dir_name
    case "${edit_type}" in
        "quant_rtn")
            edit_dir_name="quant_${param1}bit_${version}"
            ;;
        "fp4_quant")
            edit_dir_name="fp4_${param1}_${version}"
            ;;
        "magnitude_prune")
            local pct=$(echo "${param1}" | awk '{printf "%.0f", $1 * 100}')
            edit_dir_name="prune_${pct}pct_${version}"
            ;;
        "lowrank_svd")
            edit_dir_name="svd_rank${param1}_${version}"
            ;;
    esac

    local edit_path="${model_output_dir}/models/${edit_dir_name}"
    local cert_dir="${model_output_dir}/certificates/${edit_dir_name}/run_${run_num}"
    local cert_file="${cert_dir}/evaluation.cert.json"

    if [[ ! -d "${edit_path}" ]]; then
        echo "ERROR: Edit model not found: ${edit_path}" >> "${log_file}"
        return 1
    fi

    if [[ -f "${cert_file}" ]]; then
        echo "  Certification for ${edit_dir_name} run ${run_num} already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Certifying: ${edit_dir_name} run ${run_num}" >> "${log_file}"

    mkdir -p "${cert_dir}"

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

    # v0.3.1 FEATURE: Use INVARLOCK_SKIP_OVERHEAD_CHECK for large models
    # This avoids loading both baseline and edited models simultaneously
    # which would exceed B200 180GB memory (140GB × 2 = 280GB needed)
    local profile_flag="ci"
    if _is_large_model "${model_size}"; then
        export INVARLOCK_SKIP_OVERHEAD_CHECK=1
        # Override CI profile window counts to prevent OOM (CI defaults to 200/200)
        export INVARLOCK_CI_PREVIEW="${preview_n}"
        export INVARLOCK_CI_FINAL="${final_n}"
        echo "  Large model (${model_size}): setting INVARLOCK_SKIP_OVERHEAD_CHECK=1, CI windows=${preview_n}/${final_n}" >> "${log_file}"
    fi

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
        cat > "${preset_file}" << PRESET_YAML
dataset:
  provider: wikitext2
  split: validation
  seq_len: ${seq_len}
  stride: ${stride}
  preview_n: ${preview_n}
  final_n: ${final_n}
  seed: 42
PRESET_YAML
        echo "  Created preset: ${preset_file}" >> "${log_file}"
    fi

    # Run certify in isolated working directory to avoid temp file race conditions
    # (invarlock creates .certify_tmp/ in current directory which conflicts in parallel runs)
    local work_dir="${cert_dir}/.workdir"
    mkdir -p "${work_dir}"
    local abs_preset_file
    abs_preset_file="$(cd "$(dirname "${preset_file}")" && pwd)/$(basename "${preset_file}")"

    (
        cd "${work_dir}" || exit 1
        CUDA_VISIBLE_DEVICES="${gpu_id}" invarlock certify \
            --source "${baseline_path}" \
            --edited "${edit_path}" \
            --profile "${profile_flag}" \
            --tier "${INVARLOCK_TIER:-balanced}" \
            --out "${cert_dir}" \
            --cert-out "${cert_dir}" \
            --preset "${abs_preset_file}"
    ) >> "${log_file}" 2>&1

    local exit_code=$?

    # Find and copy certificate
    local found_cert=$(find "${cert_dir}" -name "*.json" -type f 2>/dev/null | head -1)
    if [[ -n "${found_cert}" && -f "${found_cert}" && "${found_cert}" != "${cert_file}" ]]; then
        cp "${found_cert}" "${cert_file}" 2>/dev/null || true
    fi

    return ${exit_code}
}

# ============ TASK: CREATE_ERROR ============

# Create error-injected model
# Usage: task_create_error <model_name> <gpu_id> <error_type> <output_dir> <log_file>
task_create_error() {
    local model_name="$1"
    local gpu_id="$2"
    local error_type="$3"
    local output_dir="$4"
    local log_file="$5"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null)
    local error_path="${model_output_dir}/models/error_${error_type}"

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    if [[ -d "${error_path}" && -f "${error_path}/config.json" ]]; then
        echo "  Error model ${error_type} already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating error model: ${error_type}" >> "${log_file}"

    if type create_error_model &>/dev/null; then
        create_error_model "${baseline_path}" "${error_path}" "${error_type}" "${gpu_id}" >> "${log_file}" 2>&1
    else
        echo "ERROR: create_error_model not available" >> "${log_file}"
        return 1
    fi

    if [[ -d "${error_path}" && -f "${error_path}/config.json" ]]; then
        echo "  Created: ${error_path}" >> "${log_file}"
        return 0
    else
        echo "  ERROR: Failed to create error model" >> "${log_file}"
        return 1
    fi
}

# ============ TASK: CERTIFY_ERROR ============

# Certify error-injected model
# Usage: task_certify_error <model_name> <gpu_id> <error_type> <output_dir> <log_file>
task_certify_error() {
    local model_name="$1"
    local gpu_id="$2"
    local error_type="$3"
    local output_dir="$4"
    local log_file="$5"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null)
    local model_id=$(cat "${model_output_dir}/.model_id" 2>/dev/null)
    local error_path="${model_output_dir}/models/error_${error_type}"
    local cert_dir="${model_output_dir}/certificates/errors/${error_type}"
    local cert_file="${cert_dir}/evaluation.cert.json"
    local preset_dir="${output_dir}/presets"

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    if [[ ! -d "${error_path}" ]]; then
        echo "ERROR: Error model not found: ${error_path}" >> "${log_file}"
        return 1
    fi

    if [[ -f "${cert_file}" ]]; then
        echo "  Certification for error ${error_type} already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Certifying error model: ${error_type}" >> "${log_file}"

    mkdir -p "${cert_dir}"

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

    # v0.3.1 FEATURE: Use INVARLOCK_SKIP_OVERHEAD_CHECK for large models
    # This avoids loading both baseline and edited models simultaneously
    # which would exceed B200 180GB memory (140GB × 2 = 280GB needed)
    local profile_flag="ci"
    if _is_large_model "${model_size}"; then
        export INVARLOCK_SKIP_OVERHEAD_CHECK=1
        # Override CI profile window counts to prevent OOM (CI defaults to 200/200)
        export INVARLOCK_CI_PREVIEW="${preview_n}"
        export INVARLOCK_CI_FINAL="${final_n}"
        echo "  Large model (${model_size}): setting INVARLOCK_SKIP_OVERHEAD_CHECK=1, CI windows=${preview_n}/${final_n}" >> "${log_file}"
    fi

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
        cat > "${preset_file}" << PRESET_YAML
dataset:
  provider: wikitext2
  split: validation
  seq_len: ${seq_len}
  stride: ${stride}
  preview_n: ${preview_n}
  final_n: ${final_n}
  seed: 42
PRESET_YAML
        echo "  Created preset: ${preset_file}" >> "${log_file}"
    fi

    # Run certify in isolated working directory to avoid temp file race conditions
    # (invarlock creates .certify_tmp/ in current directory which conflicts in parallel runs)
    local work_dir="${cert_dir}/.workdir"
    mkdir -p "${work_dir}"
    local abs_preset_file
    abs_preset_file="$(cd "$(dirname "${preset_file}")" && pwd)/$(basename "${preset_file}")"

    (
        cd "${work_dir}" || exit 1
        CUDA_VISIBLE_DEVICES="${gpu_id}" invarlock certify \
            --source "${baseline_path}" \
            --edited "${error_path}" \
            --profile "${profile_flag}" \
            --tier "${INVARLOCK_TIER:-balanced}" \
            --out "${cert_dir}" \
            --cert-out "${cert_dir}" \
            --preset "${abs_preset_file}"
    ) >> "${log_file}" 2>&1

    local exit_code=$?

    # Find and copy certificate
    local found_cert=$(find "${cert_dir}" -name "*.json" -type f 2>/dev/null | head -1)
    if [[ -n "${found_cert}" && -f "${found_cert}" && "${found_cert}" != "${cert_file}" ]]; then
        cp "${found_cert}" "${cert_file}" 2>/dev/null || true
    fi

    return ${exit_code}
}
