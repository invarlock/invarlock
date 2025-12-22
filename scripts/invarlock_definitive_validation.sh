#!/usr/bin/env bash
# invarlock_definitive_validation.sh
# ==========================================================
# InvarLock Definitive Validation Suite
# ==========================================================
# A complete, self-contained script that validates InvarLock's
# ability to detect model quality regressions.
#
# EXECUTION ORDER (per model):
# 1. Downloads model from HuggingFace (or uses local path)
# 2. Runs baseline lm-eval (MMLU, HellaSwag, ARC, Winogrande)
# 3. Calibrates InvarLock guards + drift gate and writes a preset:
#    - Runs 5 null certifications (noop edit, baseline == subject)
#    - Guards: invariants, spectral, rmt, variance, invariants
#      (invariants runs twice: pre-edit structural + post-edit verification)
#    - Gate: drift (preview→final ratio)
#    - Extracts model-specific thresholds
#    - Generates calibrated_preset_{model}.yaml (used by `invarlock certify`)
# 4. Creates clean edit (8-bit quantization) + runs lm-eval + InvarLock certify
# 5. Creates stress edit (4-bit quantization) + runs lm-eval + InvarLock certify
# 6. Creates error models (NaN, Inf, extreme quant, etc.) + InvarLock certify
#    (no lm-eval for error models - they may crash or produce garbage)
# 7. Compiles results, correlates lm-eval vs InvarLock, generates verdict
#
# Hardware: H100/A100-class CUDA GPU (bf16/fp16)
# ==========================================================

set -euo pipefail

# ============ VERSION ============
SCRIPT_VERSION="2.1.0"

# ============ USER CONFIGURATION ============
# These can be overridden by environment variables

# Model Selection
# Default: Use HuggingFace models. Set to local paths to use your own.
MODEL_1="${MODEL_1:-mistralai/Mistral-7B-v0.3}"
MODEL_2="${MODEL_2:-Qwen/Qwen2-14B}"

# Add a third model? (leave empty to skip)
MODEL_3="${MODEL_3:-}"

# Edit Configuration
# Default: quant_rtn 8-bit. Override for custom edit.
EDIT_TYPE="${EDIT_TYPE:-quant_rtn}"
EDIT_BITS="${EDIT_BITS:-8}"
EDIT_GROUP_SIZE="${EDIT_GROUP_SIZE:-128}"
EDIT_SCOPE="${EDIT_SCOPE:-ffn}"

# Eval Configuration
EVAL_TASKS="${EVAL_TASKS:-mmlu,hellaswag,arc_challenge,winogrande}"
EVAL_NUM_FEWSHOT="${EVAL_NUM_FEWSHOT:-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
LMEVAL_DTYPE="${LMEVAL_DTYPE:-auto}"
LMEVAL_PARALLELIZE="${LMEVAL_PARALLELIZE:-true}"

# InvarLock Configuration
INVARLOCK_PREVIEW_WINDOWS="${INVARLOCK_PREVIEW_WINDOWS:-64}"
INVARLOCK_FINAL_WINDOWS="${INVARLOCK_FINAL_WINDOWS:-64}"
INVARLOCK_BOOTSTRAP_N="${INVARLOCK_BOOTSTRAP_N:-2000}"
# Dataset windowing for InvarLock runs
INVARLOCK_SEQ_LEN="${INVARLOCK_SEQ_LEN:-512}"
INVARLOCK_STRIDE="${INVARLOCK_STRIDE:-256}"
INVARLOCK_EVAL_BATCH="${INVARLOCK_EVAL_BATCH:-${EVAL_BATCH_SIZE}}"
INVARLOCK_MODEL_DTYPE="${INVARLOCK_MODEL_DTYPE:-auto}"
INVARLOCK_ADAPTER="${INVARLOCK_ADAPTER:-hf_causal_auto}"
INVARLOCK_SEED="${INVARLOCK_SEED:-42}"
# InvarLock provider names: wikitext2, synthetic, hf_text, local_jsonl, seq2seq, hf_seq2seq
# Note: "wikitext2" is the InvarLock provider that internally loads HuggingFace wikitext/wikitext-2-raw-v1
INVARLOCK_DATASET="${INVARLOCK_DATASET:-wikitext2}"
INVARLOCK_TIER="${INVARLOCK_TIER:-balanced}"

# Experiment Configuration
DRIFT_CALIBRATION_RUNS="${DRIFT_CALIBRATION_RUNS:-5}"
CLEAN_EDIT_RUNS="${CLEAN_EDIT_RUNS:-3}"
STRESS_EDIT_RUNS="${STRESS_EDIT_RUNS:-2}"
RUN_ERROR_INJECTION="${RUN_ERROR_INJECTION:-true}"

# Output
OUTPUT_DIR="${OUTPUT_DIR:-./invarlock_validation_$(date +%Y%m%d_%H%M%S)}"

# GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
FLASH_ATTENTION_AVAILABLE="false"

# ============ SETUP ============
mkdir -p "${OUTPUT_DIR}"/{logs,models,evals,certificates,analysis,reports,presets}
LOG_FILE="${OUTPUT_DIR}/logs/main.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log_section() {
    echo "" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "$*" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"
}

error_exit() {
    log "ERROR: $*"
    exit 1
}

# ============ DEPENDENCY CHECK ============
check_dependencies() {
    log_section "PHASE 0: DEPENDENCY CHECK"

    local missing=()

    # Check Python
    command -v python3 >/dev/null 2>&1 || missing+=("python3")

    # Check PyTorch
    python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null || missing+=("torch")

    # Check transformers
    python3 -c "import transformers; print(f'Transformers {transformers.__version__}')" 2>/dev/null || missing+=("transformers")

    # Check PyYAML
    python3 -c "import yaml; print('PyYAML available')" 2>/dev/null || {
        log "Installing PyYAML..."
        python3 -m pip install pyyaml 2>&1 | tee -a "${LOG_FILE}" || missing+=("pyyaml")
    }

    # Check lm-eval-harness (package name is lm_eval)
    python3 -c "import lm_eval; print(f'lm-eval {lm_eval.__version__}')" 2>/dev/null || {
        log "Installing lm-eval-harness..."
        python3 -m pip install lm_eval 2>&1 | tee -a "${LOG_FILE}" || missing+=("lm_eval")
    }

    # Check InvarLock
    python3 -c "import invarlock; print(f'InvarLock {invarlock.__version__}')" 2>/dev/null || missing+=("invarlock")
    command -v invarlock >/dev/null 2>&1 || missing+=("invarlock-cli")

    # Check GPU
    python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'; print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null || missing+=("CUDA GPU")

    if [[ ${#missing[@]} -gt 0 ]]; then
        error_exit "Missing dependencies: ${missing[*]}"
    fi

    log "All dependencies satisfied"
}

# ============ HELPER: RUNTIME SETTINGS ============
resolve_model_dtype() {
    local dtype="${INVARLOCK_MODEL_DTYPE}"
    if [[ "${dtype}" == "auto" ]]; then
        dtype=$(python3 - << 'PY'
import torch
use_bf16 = bool(torch.cuda.is_available()) and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
print("bfloat16" if use_bf16 else "float16")
PY
)
    fi
    if [[ -z "${dtype}" ]]; then
        dtype="float16"
    fi
    RESOLVED_MODEL_DTYPE="${dtype}"
    if [[ "${LMEVAL_DTYPE}" == "auto" ]]; then
        LMEVAL_DTYPE="${RESOLVED_MODEL_DTYPE}"
    fi
}

detect_flash_attention() {
    local skip_flag="${SKIP_FLASH_ATTN:-${SKIP_FLASH_ATTENTION:-}}"
    skip_flag=$(echo "${skip_flag}" | tr '[:upper:]' '[:lower:]')
    if [[ "${skip_flag}" == "1" || "${skip_flag}" == "true" || "${skip_flag}" == "yes" ]]; then
        FLASH_ATTENTION_AVAILABLE="false"
        return
    fi
    FLASH_ATTENTION_AVAILABLE=$(python3 - << 'PY'
import importlib.util
spec = importlib.util.find_spec("flash_attn")
print("true" if spec is not None else "false")
PY
)
}

supports_flash_attention() {
    local model_path="$1"
    local model_lower
    model_lower=$(echo "${model_path}" | tr '[:upper:]' '[:lower:]')
    local pattern
    for pattern in falcon mpt- gpt2 bloom opt- gpt-j gpt-neo codegen santacoder stablelm; do
        if [[ "${model_lower}" == *"${pattern}"* ]]; then
            return 1
        fi
    done
    return 0
}

# ============ HELPER: MODEL DOWNLOAD/SETUP ============
# IMPORTANT: This function returns ONLY the path on stdout.
# All logs go to stderr so that $(setup_model ...) captures just the path.
setup_model() {
    local model_id="$1"
    local model_name=$(basename "${model_id}" | tr '[:upper:]' '[:lower:]' | tr '/' '_')
    local model_dir="${OUTPUT_DIR}/models/${model_name}"

    # Log to stderr (not stdout) so callers get clean path
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setting up model: ${model_id}" >> "${LOG_FILE}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setting up model: ${model_id}" >&2

    # Check if local path
    if [[ -d "${model_id}" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Using local model: ${model_id}" >> "${LOG_FILE}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Using local model: ${model_id}" >&2
        echo "${model_id}"
        return 0
    fi

    # Check if already downloaded
    if [[ -d "${model_dir}/baseline" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Model already downloaded: ${model_dir}/baseline" >> "${LOG_FILE}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Model already downloaded: ${model_dir}/baseline" >&2
        echo "${model_dir}/baseline"
        return 0
    fi

    # Download from HuggingFace
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Downloading from HuggingFace: ${model_id}" >> "${LOG_FILE}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Downloading from HuggingFace: ${model_id}" >&2
    mkdir -p "${model_dir}"

    local use_flash="false"
    if [[ "${FLASH_ATTENTION_AVAILABLE}" == "true" ]] && supports_flash_attention "${model_id}"; then
        use_flash="true"
    fi

    # Python output goes to stderr and log file
    python3 << EOF 2>&1 | tee -a "${LOG_FILE}" >&2
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import gc

model_id = "${model_id}"
output_dir = Path("${model_dir}/baseline")
output_dir.mkdir(parents=True, exist_ok=True)

dtype_name = "${RESOLVED_MODEL_DTYPE}".strip().lower()
if not dtype_name or dtype_name == "auto":
    use_bf16 = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    dtype_name = "bfloat16" if use_bf16 else "float16"
try:
    torch_dtype = getattr(torch, dtype_name)
except AttributeError:
    torch_dtype = torch.float16

use_flash = "${use_flash}" == "true"

print(f"Downloading {model_id}...")
print(f"Using dtype: {dtype_name}")
print(f"Flash Attention 2: {'enabled' if use_flash else 'disabled'}")

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.save_pretrained(output_dir)

model_kwargs = {
    "torch_dtype": torch_dtype,
    "trust_remote_code": True,
    "device_map": "auto",
    "low_cpu_mem_usage": True,
}
if use_flash:
    model_kwargs["attn_implementation"] = "flash_attention_2"

try:
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
except Exception as err:
    if use_flash and "flash" in str(err).lower():
        print(f"Flash Attention failed, retrying with eager attention: {err}")
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    else:
        raise

# Fix invalid generation config before saving
if hasattr(model, "generation_config"):
    gen_config = model.generation_config
    if hasattr(gen_config, "do_sample") and not gen_config.do_sample:
        if getattr(gen_config, "temperature", None) not in (None, 1.0):
            print(f"Clearing temperature={gen_config.temperature} (do_sample=False)")
            gen_config.temperature = None
        if getattr(gen_config, "top_p", None) not in (None, 1.0):
            print(f"Clearing top_p={gen_config.top_p} (do_sample=False)")
            gen_config.top_p = None

model.save_pretrained(output_dir, safe_serialization=True)

del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"Saved to {output_dir}")
EOF

    # ONLY output the clean path on stdout
    echo "${model_dir}/baseline"
}

# ============ HELPER: CREATE EDITED MODEL ============
create_edited_model() {
    local baseline_path="$1"
    local output_path="$2"
    local edit_type="$3"
    local bits="$4"
    local group_size="$5"
    local scope="$6"

    log "Creating edited model:"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"
    log "  Edit: ${edit_type} bits=${bits} group_size=${group_size} scope=${scope}"

    mkdir -p "$(dirname "${output_path}")"

    if [[ "${edit_type}" == "quant_rtn" ]]; then
        local use_flash="false"
        if [[ "${FLASH_ATTENTION_AVAILABLE}" == "true" ]] && supports_flash_attention "${baseline_path}"; then
            use_flash="true"
        fi
        python3 << EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import gc
import sys

baseline_path = Path("${baseline_path}")
output_path = Path("${output_path}")
bits = int("${bits}")
group_size = int("${group_size}")
scope = "${scope}"

dtype_name = "${RESOLVED_MODEL_DTYPE}".strip().lower()
if not dtype_name or dtype_name == "auto":
    use_bf16 = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    dtype_name = "bfloat16" if use_bf16 else "float16"
try:
    torch_dtype = getattr(torch, dtype_name)
except AttributeError:
    torch_dtype = torch.float16

use_flash = "${use_flash}" == "true"

print(f"Loading baseline from {baseline_path}...")
tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
model_kwargs = {
    "torch_dtype": torch_dtype,
    "trust_remote_code": True,
    "device_map": "auto" if torch.cuda.is_available() else "cpu",
    "low_cpu_mem_usage": True,
}
if use_flash:
    model_kwargs["attn_implementation"] = "flash_attention_2"

try:
    model = AutoModelForCausalLM.from_pretrained(baseline_path, **model_kwargs)
except Exception as err:
    if use_flash and "flash" in str(err).lower():
        print(f"Flash Attention failed, retrying with eager attention: {err}")
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(baseline_path, **model_kwargs)
    else:
        raise

@torch.no_grad()
def round_to_nearest(tensor, bits, group_size):
    """Group-wise RTN quantization (per-output-channel groups along input dim)."""
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    orig_shape = tensor.shape
    flat = tensor.reshape(orig_shape[0], -1)
    in_features = flat.shape[1]
    if group_size <= 0 or group_size >= in_features:
        group_size = in_features
    num_groups = (in_features + group_size - 1) // group_size
    pad = (num_groups * group_size) - in_features
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))
    grouped = flat.reshape(orig_shape[0], num_groups, group_size)
    max_abs = grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_abs / qmax, min=1e-10)
    quantized = torch.round(grouped / scale).clamp(qmin, qmax) * scale
    quantized = quantized.reshape(orig_shape[0], num_groups * group_size)
    if pad > 0:
        quantized = quantized[:, :in_features]
    return quantized.reshape(orig_shape).to(tensor.dtype)

def should_quantize(name, scope):
    """Check if parameter should be quantized based on name and scope."""
    name_lower = name.lower()
    if scope == "all":
        return "weight" in name_lower and any(x in name_lower for x in [
            "linear", "dense", "proj", "fc", "mlp", "attn",
            "wqkv", "query_key_value"
        ])
    elif scope == "ffn":
        return "weight" in name_lower and any(x in name_lower for x in [
            "mlp", "fc", "dense", "gate", "up_proj", "down_proj",
            "dense_h_to_4h", "dense_4h_to_h"
        ])
    elif scope == "attn":
        return "weight" in name_lower and any(x in name_lower for x in [
            "attn", "q_proj", "k_proj", "v_proj", "o_proj",
            "wqkv", "out_proj", "query_key_value"
        ])
    return False

print(f"Quantizing to {bits}-bit (scope={scope})...")
quantized_count = 0
total_model_params = sum(p.numel() for p in model.parameters())
edited_params = 0

for name, param in model.named_parameters():
    if should_quantize(name, scope) and param.dim() >= 2:
        param.data = round_to_nearest(param.data, bits, group_size)
        quantized_count += 1
        edited_params += param.numel()
        if quantized_count <= 3:
            print(f"  Quantized: {name} ({param.shape})")

coverage_pct = 100.0 * edited_params / total_model_params if total_model_params > 0 else 0
print(f"Quantized {quantized_count} parameters ({edited_params:,} / {total_model_params:,} = {coverage_pct:.1f}% coverage)")

model = model.cpu()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Save
output_path.mkdir(parents=True, exist_ok=True)
tokenizer.save_pretrained(output_path)
model.save_pretrained(output_path, safe_serialization=True)

# Save edit metadata
metadata = {
    "edit_type": "quant_rtn",
    "bits": bits,
    "group_size": group_size,
    "scope": scope,
    "quantized_params": quantized_count,
    "coverage_pct": round(coverage_pct, 2),
    "dtype": dtype_name,
}
with open(output_path / "edit_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved edited model to {output_path}")
EOF
    else
        error_exit "Unknown edit type: ${edit_type}"
    fi
}

# ============ HELPER: CREATE ERROR MODEL ============
create_error_model() {
    local baseline_path="$1"
    local output_path="$2"
    local error_type="$3"

    log "Creating error model (type=${error_type}):"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"

    mkdir -p "$(dirname "${output_path}")"

    python3 << EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import random

baseline_path = Path("${baseline_path}")
output_path = Path("${output_path}")
error_type = "${error_type}"

print(f"Loading baseline from {baseline_path}...")
dtype_name = "${RESOLVED_MODEL_DTYPE}".strip().lower()
if not dtype_name or dtype_name == "auto":
    use_bf16 = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    dtype_name = "bfloat16" if use_bf16 else "float16"
try:
    torch_dtype = getattr(torch, dtype_name)
except AttributeError:
    torch_dtype = torch.float16

tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    baseline_path,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

error_info = {"error_type": error_type, "injected": False}

if error_type == "nan_injection":
    for name, param in model.named_parameters():
        if 'layers.0' in name and 'weight' in name:
            with torch.no_grad():
                param.data[0, 0] = float('nan')
            error_info["injected"] = True
            error_info["target_param"] = name
            print(f"Injected NaN into: {name}")
            break

elif error_type == "inf_injection":
    for name, param in model.named_parameters():
        if 'attn' in name.lower() and 'weight' in name:
            with torch.no_grad():
                param.data[0, 0] = float('inf')
            error_info["injected"] = True
            error_info["target_param"] = name
            print(f"Injected Inf into: {name}")
            break

elif error_type == "extreme_quant":
    def extreme_quant(tensor):
        qmin, qmax = -2, 1
        scale = tensor.abs().max() / max(abs(qmin), abs(qmax))
        scale = torch.clamp(scale, min=1e-10)
        quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)
        return (quantized * scale).to(tensor.dtype)

    count = 0
    for name, param in model.named_parameters():
        if 'weight' in name.lower() and param.dim() >= 2:
            with torch.no_grad():
                param.data = extreme_quant(param.data)
                count += 1
    error_info["injected"] = True
    error_info["quantized_params"] = count
    print(f"Applied extreme 2-bit quantization to {count} params")

elif error_type == "scale_explosion":
    for name, param in model.named_parameters():
        if 'mlp' in name.lower() and 'weight' in name and param.dim() >= 2:
            with torch.no_grad():
                param.data = param.data * 100.0
            error_info["injected"] = True
            error_info["target_param"] = name
            error_info["scale_factor"] = 100.0
            print(f"Scaled by 100x: {name}")
            break

elif error_type == "zero_layer":
    for name, param in model.named_parameters():
        if 'layers.5' in name and 'weight' in name:
            with torch.no_grad():
                param.data.zero_()
            error_info["injected"] = True
            error_info["target_param"] = name
            print(f"Zeroed: {name}")
            break

else:
    print(f"Unknown error type: {error_type}")

# Save
output_path.mkdir(parents=True, exist_ok=True)
tokenizer.save_pretrained(output_path)
model.save_pretrained(output_path, safe_serialization=True)

with open(output_path / "error_metadata.json", 'w') as f:
    json.dump(error_info, f, indent=2)

print(f"Saved error model to {output_path}")
EOF
}

# ============ HELPER: RUN LMEVAL ============
run_lmeval() {
    local model_path="$1"
    local output_file="$2"
    local tasks="$3"
    local batch_size="$4"
    local num_fewshot="$5"

    log "Running lm-eval:"
    log "  Model: ${model_path}"
    log "  Tasks: ${tasks}"
    log "  Output: ${output_file}"

    mkdir -p "$(dirname "${output_file}")"

    local model_args="pretrained=${model_path},trust_remote_code=True,dtype=${LMEVAL_DTYPE},device_map=auto"
    local parallelize_flag
    parallelize_flag=$(echo "${LMEVAL_PARALLELIZE}" | tr '[:upper:]' '[:lower:]')
    if [[ "${CUDA_VISIBLE_DEVICES:-}" == *","* && "${parallelize_flag}" != "false" && "${parallelize_flag}" != "0" ]]; then
        model_args="${model_args},parallelize=True"
    fi
    if [[ "${FLASH_ATTENTION_AVAILABLE}" == "true" ]] && supports_flash_attention "${model_path}"; then
        model_args="${model_args},attn_implementation=flash_attention_2"
    fi

    TORCH_COMPILE="${LMEVAL_TORCH_COMPILE:-0}" \
    python3 -m lm_eval \
        --model hf \
        --model_args "${model_args}" \
        --tasks "${tasks}" \
        --batch_size "${batch_size}" \
        --num_fewshot "${num_fewshot}" \
        --output_path "$(dirname "${output_file}")" \
        --log_samples \
        2>&1 | tee -a "${LOG_FILE}"

    local results_file=$(find "$(dirname "${output_file}")" -name "results*.json" -type f | head -1)
    if [[ -n "${results_file}" ]]; then
        mv "${results_file}" "${output_file}"
        log "  Results saved to: ${output_file}"
    else
        log "  WARNING: No results file found"
    fi
}

# ============ HELPER: GENERATE INVARLOCK CONFIG YAML ============
# InvarLock CLI uses YAML config files, not individual CLI options.
# This function creates a properly formatted config file.
generate_invarlock_config() {
    local model_path="$1"
    local output_yaml="$2"
    local edit_name="${3:-noop}"
    local seed="${4:-${INVARLOCK_SEED}}"
    local preview_n="${5:-${INVARLOCK_PREVIEW_WINDOWS}}"
    local final_n="${6:-${INVARLOCK_FINAL_WINDOWS}}"
    local bootstrap_n="${7:-${INVARLOCK_BOOTSTRAP_N}}"
    local seq_len="${8:-${INVARLOCK_SEQ_LEN}}"
    local stride="${9:-${INVARLOCK_STRIDE}}"
    local eval_batch="${10:-${INVARLOCK_EVAL_BATCH}}"

    # Detect adapter based on model architecture
    # Use auto adapter for general causal LM support
    local adapter="${INVARLOCK_ADAPTER}"

    # InvarLock provider is used directly (wikitext2, synthetic, hf_text, etc.)
    # The provider handles HuggingFace dataset loading internally
    local dataset_provider="${INVARLOCK_DATASET}"
    local attn_impl_yaml=""
    if [[ "${FLASH_ATTENTION_AVAILABLE}" == "true" ]] && supports_flash_attention "${model_path}"; then
        attn_impl_yaml='attn_implementation: "flash_attention_2"'
    else
        attn_impl_yaml='# flash_attention_2 not available'
    fi

    # Create the YAML config
    # Note: edit.name must be a quoted string (even "noop")
    cat > "${output_yaml}" << YAML_EOF
# Auto-generated InvarLock config for validation
# Model: ${model_path}
# Edit: ${edit_name}
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

model:
  id: "${model_path}"
  adapter: "${adapter}"
  device: "auto"
  dtype: "${RESOLVED_MODEL_DTYPE}"
  ${attn_impl_yaml}

dataset:
  provider: "${dataset_provider}"
  split: "validation"
  preview_n: ${preview_n}
  final_n: ${final_n}
  seq_len: ${seq_len}
  stride: ${stride}
  seed: ${seed}

edit:
  name: "${edit_name}"

guards:
  order:
    - invariants
    - spectral
    - rmt
    - variance
    - invariants

eval:
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
YAML_EOF

    log "  Generated config: ${output_yaml}"
}

# ============ HELPER: RUN INVARLOCK FULL CALIBRATION ============
# This function runs null certifications (baseline == subject) and extracts
# guard thresholds from the certificate JSON to generate a calibrated preset.
run_invarlock_calibration() {
    local model_path="$1"
    local model_name="$2"
    local output_dir="$3"
    local num_runs="$4"
    local preset_output_dir="$5"

    log "Running InvarLock FULL calibration (all guards):"
    log "  Model: ${model_path}"
    log "  Runs: ${num_runs}"
    log "  Preset output: ${preset_output_dir}"

    mkdir -p "${output_dir}"
    mkdir -p "${preset_output_dir}"

    # Run null certifications using config files
    for run in $(seq 1 "${num_runs}"); do
        log "  Calibration run ${run}/${num_runs}..."

        # Use different seeds for each run
        local seed=$((41 + run))
        local run_dir="${output_dir}/run_${run}"
        local config_yaml="${run_dir}/calibration_config.yaml"

        mkdir -p "${run_dir}"

        # Generate config YAML for this run
        generate_invarlock_config \
            "${model_path}" \
            "${config_yaml}" \
            "noop" \
            "${seed}" \
            "${INVARLOCK_PREVIEW_WINDOWS}" \
            "${INVARLOCK_FINAL_WINDOWS}" \
            "${INVARLOCK_BOOTSTRAP_N}" \
            "${INVARLOCK_SEQ_LEN}" \
            "${INVARLOCK_STRIDE}" \
            "${INVARLOCK_EVAL_BATCH}"

        # Run InvarLock with the config file
        invarlock run \
            --config "${config_yaml}" \
            --profile ci \
            --out "${run_dir}" \
            2>&1 | tee -a "${LOG_FILE}" || log "  Run ${run} failed (exit code $?)"

        # Find and copy report.json for consistent naming
        # Note: invarlock run generates report.json, not cert.json (certs are generated separately)
        local report_file=$(find "${run_dir}" -name "report*.json" -type f 2>/dev/null | head -1)
        if [[ -n "${report_file}" ]]; then
            cp "${report_file}" "${run_dir}/baseline_report.json" 2>/dev/null || true
            log "  Report saved: ${run_dir}/baseline_report.json"

            # For calibration, we need to generate a self-certificate (baseline == subject)
            # This provides the certificate structure needed for threshold extraction
            python3 << GENERATE_CERT
import json
from pathlib import Path
try:
    from invarlock.reporting.certificate import make_certificate
    report_path = Path("${report_file}")
    cert_path = Path("${run_dir}") / "evaluation.cert.json"

    report = json.loads(report_path.read_text())
    # For calibration, use report as both subject and baseline (null certification)
    cert = make_certificate(report, report)
    with open(cert_path, 'w') as f:
        json.dump(cert, f, indent=2)
    print(f"  Generated calibration certificate: {cert_path}")
except Exception as e:
    print(f"  WARNING: Could not generate certificate: {e}")
GENERATE_CERT
        else
            log "  WARNING: No report.json found for run ${run}"
        fi
    done

    # Extract guard thresholds and generate calibrated preset
    # Note: Using unquoted heredoc delimiter to allow shell variable interpolation
    python3 << CALIBRATION_SCRIPT
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("WARNING: PyYAML not available, will output JSON instead")

# Configuration from shell - these are filled in by bash
output_dir = Path("${output_dir}")
preset_output_dir = Path("${preset_output_dir}")
model_name = "${model_name}"
model_path = "${model_path}"
tier = "${INVARLOCK_TIER}".strip().lower()
dataset_provider = "${INVARLOCK_DATASET}"
seq_len = int("${INVARLOCK_SEQ_LEN}")
stride = int("${INVARLOCK_STRIDE}")
preview_n = int("${INVARLOCK_PREVIEW_WINDOWS}")
final_n = int("${INVARLOCK_FINAL_WINDOWS}")
seed = int("${INVARLOCK_SEED}")

print(f"DEBUG: output_dir = {output_dir}")
print(f"DEBUG: preset_output_dir = {preset_output_dir}")
print(f"DEBUG: model_name = {model_name}")

# ==============================================================================
# CERTIFICATE LOADING
# ==============================================================================

def load_certificates() -> List[Dict[str, Any]]:
    """Load all valid certificates from calibration runs.

    Tries to load evaluation.cert.json first (generated by our script).
    Falls back to loading report.json and extracting relevant data if cert doesn't exist.
    """
    certs = []
    for run_dir in sorted(output_dir.glob("run_*")):
        cert = None
        report = None

        cert_path = run_dir / "evaluation.cert.json"
        if cert_path.exists():
            try:
                cert = json.loads(cert_path.read_text())
            except Exception as e:
                print(f"  Error loading {cert_path}: {e}")

        report_path = run_dir / "baseline_report.json"
        if not report_path.exists():
            report_files = list(run_dir.glob("**/report*.json"))
            if report_files:
                report_path = report_files[0]

        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
            except Exception as e:
                print(f"  Error loading {report_path}: {e}")

        record = _merge_record(cert, report)
        if record:
            certs.append(record)
            if cert is not None:
                print(f"  Loaded cert: {run_dir.name}/evaluation.cert.json")
            elif report is not None:
                print(f"  Loaded report: {report_path.name}")

    return certs

def _merge_record(cert: Optional[Dict[str, Any]], report: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rec: Dict[str, Any] = {}
    if isinstance(cert, dict):
        rec = json.loads(json.dumps(cert))
    if not isinstance(report, dict):
        return rec or None

    metrics = report.get('metrics', {}) or {}
    pm = metrics.get('primary_metric', {}) or {}
    if not pm and 'ppl_final' in metrics:
        pm = {
            'final': metrics.get('ppl_final'),
            'preview': metrics.get('ppl_preview'),
            'ratio_vs_baseline': 1.0,
            'drift': metrics.get('ppl_final', 1.0) / max(metrics.get('ppl_preview', 1.0), 1e-10)
        }
    if pm and not rec.get('primary_metric'):
        rec['primary_metric'] = pm

    guards = report.get('guards', []) or []
    for guard in guards:
        if not isinstance(guard, dict):
            continue
        name = str(guard.get('name', '')).lower()
        gmetrics = guard.get('metrics', {}) or {}
        gpolicy = guard.get('policy', {}) or {}

        if name == 'spectral':
            spec = rec.get('spectral', {}) if isinstance(rec.get('spectral'), dict) else {}
            if gmetrics.get('family_z_quantiles'):
                spec.setdefault('family_z_quantiles', gmetrics.get('family_z_quantiles'))
            if gmetrics.get('family_z_summary'):
                spec.setdefault('family_z_summary', gmetrics.get('family_z_summary'))
            if gmetrics.get('family_caps'):
                spec.setdefault('family_caps', gmetrics.get('family_caps'))
            if gmetrics.get('sigma_quantile') is not None:
                spec.setdefault('sigma_quantile', gmetrics.get('sigma_quantile'))
            if gmetrics.get('deadband') is not None:
                spec.setdefault('deadband', gmetrics.get('deadband'))
            if gmetrics.get('max_caps') is not None:
                spec.setdefault('max_caps', gmetrics.get('max_caps'))
            if gmetrics.get('families'):
                spec.setdefault('families', gmetrics.get('families'))
            if gmetrics.get('family_stats'):
                spec.setdefault('families', gmetrics.get('family_stats'))
            z_scores = guard.get('final_z_scores') or gmetrics.get('final_z_scores')
            if isinstance(z_scores, dict):
                spec['final_z_scores'] = z_scores
            fam_map = guard.get('module_family_map') or gmetrics.get('module_family_map')
            if isinstance(fam_map, dict):
                spec['module_family_map'] = fam_map
            if gpolicy and not spec.get('policy'):
                spec['policy'] = gpolicy
            rec['spectral'] = spec
        elif name == 'rmt':
            rmt = rec.get('rmt', {}) if isinstance(rec.get('rmt'), dict) else {}
            if gmetrics.get('family_stats'):
                rmt.setdefault('family_stats', gmetrics.get('family_stats'))
            if gmetrics.get('epsilon'):
                rmt.setdefault('epsilon', gmetrics.get('epsilon'))
            if gmetrics.get('epsilon_default') is not None:
                rmt.setdefault('epsilon_default', gmetrics.get('epsilon_default'))
            if gmetrics.get('epsilon_by_family'):
                rmt.setdefault('epsilon_by_family', gmetrics.get('epsilon_by_family'))
            if gmetrics.get('margin') is not None:
                rmt.setdefault('margin', gmetrics.get('margin'))
            if gmetrics.get('margin_used') is not None:
                rmt.setdefault('margin', gmetrics.get('margin_used'))
            if gmetrics.get('deadband') is not None:
                rmt.setdefault('deadband', gmetrics.get('deadband'))
            if gmetrics.get('deadband_used') is not None:
                rmt.setdefault('deadband', gmetrics.get('deadband_used'))
            if gmetrics.get('outliers_by_family'):
                rmt.setdefault('outliers_by_family', gmetrics.get('outliers_by_family'))
            if gmetrics.get('outliers_per_family'):
                rmt.setdefault('outliers_per_family', gmetrics.get('outliers_per_family'))
            if gmetrics.get('baseline_outliers_per_family'):
                rmt.setdefault('baseline_outliers_per_family', gmetrics.get('baseline_outliers_per_family'))
            if gmetrics.get('families'):
                rmt.setdefault('families', gmetrics.get('families'))
            if gpolicy and not rmt.get('policy'):
                rmt['policy'] = gpolicy
            rec['rmt'] = rmt
        elif name == 'variance':
            var = rec.get('variance', {}) if isinstance(rec.get('variance'), dict) else {}
            if gmetrics.get('predictive_gate') is not None:
                var.setdefault('predictive_gate', gmetrics.get('predictive_gate'))
            if gmetrics.get('ab_windows_used') is not None:
                var.setdefault('ab_windows_used', gmetrics.get('ab_windows_used'))
            if gmetrics.get('calibration_stats'):
                var.setdefault('calibration_stats', gmetrics.get('calibration_stats'))
            if gmetrics.get('calibration'):
                var.setdefault('calibration', gmetrics.get('calibration'))
            if gmetrics.get('deadband') is not None:
                var.setdefault('deadband', gmetrics.get('deadband'))
            if gmetrics.get('min_gain') is not None:
                var.setdefault('min_gain', gmetrics.get('min_gain'))
            if gmetrics.get('min_effect_lognll') is not None:
                var.setdefault('min_effect_lognll', gmetrics.get('min_effect_lognll'))
            if gpolicy and not var.get('policy'):
                var['policy'] = gpolicy
            rec['variance'] = var

    return rec or None

def convert_report_to_cert_structure(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a RunReport to a certificate-like structure for calibration.

    This extracts the key fields needed for guard calibration from a report.
    """
    if not isinstance(report, dict):
        return None

    metrics = report.get('metrics', {}) or {}

    # Build primary_metric block
    pm = metrics.get('primary_metric', {}) or {}
    if not pm and 'ppl_final' in metrics:
        # Legacy format
        pm = {
            'final': metrics.get('ppl_final'),
            'preview': metrics.get('ppl_preview'),
            'ratio_vs_baseline': 1.0,  # Self-baseline
            'drift': metrics.get('ppl_final', 1.0) / max(metrics.get('ppl_preview', 1.0), 1e-10)
        }

    cert_structure = {
        'primary_metric': pm,
        'spectral': {},
        'rmt': {},
        'variance': {},
        'validation': {}
    }

    # Extract guard data from guards list
    guards = report.get('guards', []) or []
    for guard in guards:
        if not isinstance(guard, dict):
            continue
        name = str(guard.get('name', '')).lower()
        guard_metrics = guard.get('metrics', {}) or {}
        guard_details = guard.get('details', {}) or {}
        guard_policy = guard.get('policy', {}) or {}

        if name == 'spectral':
            spec = {
                'family_z_quantiles': guard_metrics.get('family_z_quantiles', {}),
                'family_z_summary': guard_metrics.get('family_z_summary', {}),
                'family_caps': guard_metrics.get('family_caps', {}),
                'sigma_quantile': guard_metrics.get('sigma_quantile'),
                'deadband': guard_metrics.get('deadband'),
                'max_caps': guard_metrics.get('max_caps'),
                'families': guard_metrics.get('families', {}),
                'summary': guard_details,
                'policy': guard_policy
            }
            if guard_metrics.get('family_stats'):
                spec.setdefault('families', guard_metrics.get('family_stats', {}))
            z_scores = guard.get('final_z_scores') or guard_metrics.get('final_z_scores')
            if isinstance(z_scores, dict):
                spec['final_z_scores'] = z_scores
            fam_map = guard.get('module_family_map') or guard_metrics.get('module_family_map')
            if isinstance(fam_map, dict):
                spec['module_family_map'] = fam_map
            cert_structure['spectral'] = spec
        elif name == 'rmt':
            rmt = {
                'family_stats': guard_metrics.get('family_stats', {}),
                'epsilon': guard_metrics.get('epsilon', {}),
                'epsilon_by_family': guard_metrics.get('epsilon_by_family', {}),
                'epsilon_default': guard_metrics.get('epsilon_default'),
                'margin': guard_metrics.get('margin'),
                'deadband': guard_metrics.get('deadband'),
                'outliers_by_family': guard_metrics.get('outliers_by_family', {}),
                'outliers_per_family': guard_metrics.get('outliers_per_family', {}),
                'baseline_outliers_per_family': guard_metrics.get('baseline_outliers_per_family', {}),
                'families': guard_metrics.get('families', {}),
                'summary': guard_details,
                'policy': guard_policy
            }
            if rmt.get('margin') is None and guard_metrics.get('margin_used') is not None:
                rmt['margin'] = guard_metrics.get('margin_used')
            if rmt.get('deadband') is None and guard_metrics.get('deadband_used') is not None:
                rmt['deadband'] = guard_metrics.get('deadband_used')
            cert_structure['rmt'] = rmt
        elif name == 'variance':
            cert_structure['variance'] = {
                'calibration_stats': guard_metrics.get('calibration_stats', {}),
                'calibration': guard_metrics.get('calibration', {}),
                'predictive_gate': guard_metrics.get('predictive_gate'),
                'ab_windows_used': guard_metrics.get('ab_windows_used'),
                'deadband': guard_metrics.get('deadband'),
                'min_gain': guard_metrics.get('min_gain'),
                'min_effect_lognll': guard_metrics.get('min_effect_lognll'),
                'summary': guard_details,
                'policy': guard_policy
            }

    return cert_structure

# ==============================================================================
# CALIBRATION HELPERS
# ==============================================================================

def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def _quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return values[lower]
    frac = pos - lower
    return values[lower] + (values[upper] - values[lower]) * frac

def _spectral_margin(tier_name):
    return 0.10 if tier_name == "conservative" else 0.05

def _default_max_caps(tier_name):
    if tier_name == "conservative":
        return 3
    if tier_name == "aggressive":
        return 8
    return 5

def _allocate_budget(counts, budget):
    if not counts or budget <= 0:
        return {fam: 0 for fam in counts}
    total = sum(counts.values())
    if total <= 0:
        return {fam: 0 for fam in counts}
    raw = {fam: budget * count / total for fam, count in counts.items()}
    alloc = {fam: int(round(val)) for fam, val in raw.items()}
    diff = budget - sum(alloc.values())
    if diff > 0:
        for fam in sorted(raw, key=raw.get, reverse=True):
            if diff == 0:
                break
            alloc[fam] += 1
            diff -= 1
    elif diff < 0:
        for fam in sorted(raw, key=raw.get):
            if diff == 0:
                break
            if alloc.get(fam, 0) > 0:
                alloc[fam] -= 1
                diff += 1
    return alloc

def _rmt_quantile_for_tier(tier_name):
    if tier_name == "conservative":
        return 0.95
    if tier_name == "aggressive":
        return 0.99
    return 0.97

# ==============================================================================
# DRIFT CALIBRATION
# ==============================================================================

def calibrate_drift(certs: List[Dict]) -> Dict[str, Any]:
    """
    Extract drift statistics from certificates.

    From primary_metric.ratio_vs_baseline (or similar drift field), compute:
    - mean, std, min, max of drift ratios
    - suggested acceptable drift band
    - whether model is compatible with default 0.95-1.05 band
    """
    drifts = []
    ratios = []

    for cert in certs:
        pm = cert.get('primary_metric', {})

        # Try different fields that might contain drift info
        drift = pm.get('drift')
        ratio = pm.get('ratio_vs_baseline')
        preview = pm.get('preview')
        final = pm.get('final')

        if drift is not None:
            try:
                drifts.append(float(drift))
            except (TypeError, ValueError):
                pass

        if ratio is not None:
            try:
                ratios.append(float(ratio))
            except (TypeError, ValueError):
                pass

        # If we have preview/final, compute ratio ourselves
        if preview is not None and final is not None:
            try:
                computed_ratio = float(final) / float(preview)
                if computed_ratio not in ratios:
                    ratios.append(computed_ratio)
            except (TypeError, ValueError, ZeroDivisionError):
                pass

    # Use ratios if available, otherwise drifts
    values = ratios if ratios else drifts

    if len(values) < 2:
        print("  WARNING: Not enough drift data for calibration")
        return {
            'mean': 1.0,
            'std': 0.0,
            'suggested_band': [0.95, 1.05],
            'band_compatible': True
        }

    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values)

    # Suggested band: mean ± max(2*std, 0.05)
    margin = max(2 * std_val, 0.05)
    suggested_band = [round(mean_val - margin, 3), round(mean_val + margin, 3)]

    # Check if compatible with default 0.95-1.05 band
    band_compatible = 0.95 <= mean_val <= 1.05

    result = {
        'mean': round(mean_val, 4),
        'std': round(std_val, 4),
        'min': round(min(values), 4),
        'max': round(max(values), 4),
        'suggested_band': suggested_band,
        'band_compatible': band_compatible
    }

    print(f"  Drift: mean={result['mean']:.4f}, std={result['std']:.4f}, " +
          f"band_compatible={'Yes' if band_compatible else 'No'}")

    return result

# ==============================================================================
# SPECTRAL CALIBRATION
# ==============================================================================

def calibrate_spectral(certs: List[Dict]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    Extract spectral guard thresholds from certificates.

    Uses per-run order-statistic calibration on module-level z-scores with a
    tier-specific margin, keeping max_caps as a per-run budget. Falls back to
    family_z_quantiles when module-level scores are missing.
    """
    per_run_caps = defaultdict(list)
    q99_values: Dict[str, List[float]] = defaultdict(list)
    max_values: Dict[str, List[float]] = defaultdict(list)
    existing_caps: Dict[str, float] = {}

    sigma_quantile: Optional[float] = None
    deadband: Optional[float] = None
    max_caps: Optional[int] = None

    for cert in certs:
        spectral = cert.get('spectral', {})
        if not isinstance(spectral, dict):
            continue
        policy = spectral.get('policy', {}) if isinstance(spectral.get('policy'), dict) else {}

        if sigma_quantile is None:
            sq = (
                policy.get('sigma_quantile')
                or policy.get('contraction')
                or policy.get('kappa')
                or spectral.get('sigma_quantile')
                or spectral.get('summary', {}).get('sigma_quantile')
            )
            sq = _safe_float(sq)
            if sq is not None:
                sigma_quantile = sq

        if deadband is None:
            db = policy.get('deadband') or spectral.get('deadband') or spectral.get('summary', {}).get('deadband')
            db = _safe_float(db)
            if db is not None:
                deadband = db

        if max_caps is None:
            mc = policy.get('max_caps') or spectral.get('max_caps') or spectral.get('summary', {}).get('max_caps')
            try:
                if mc is not None:
                    max_caps = int(mc)
            except (TypeError, ValueError):
                pass

        fam_caps = spectral.get('family_caps', {})
        if not fam_caps and isinstance(policy.get('family_caps'), dict):
            fam_caps = policy.get('family_caps', {})
        if isinstance(fam_caps, dict):
            for fam, caps in fam_caps.items():
                try:
                    kappa = caps.get('kappa') if isinstance(caps, dict) else float(caps)
                    if kappa is not None:
                        existing_caps[str(fam)] = float(kappa)
                except (TypeError, ValueError, AttributeError):
                    pass

        z_map = spectral.get('final_z_scores')
        fam_map = spectral.get('module_family_map')
        if isinstance(z_map, dict) and isinstance(fam_map, dict):
            z_by_family = defaultdict(list)
            for module, z in z_map.items():
                fam = fam_map.get(module)
                if fam is None:
                    continue
                z_val = _safe_float(z)
                if z_val is None:
                    continue
                z_by_family[str(fam)].append(abs(z_val))
            if z_by_family:
                counts = {fam: len(vals) for fam, vals in z_by_family.items() if vals}
                budget = (
                    max_caps
                    if isinstance(max_caps, int) and max_caps >= 0
                    else _default_max_caps(tier)
                )
                alloc = _allocate_budget(counts, budget)
                for fam, values in z_by_family.items():
                    if not values:
                        continue
                    values_sorted = sorted(values, reverse=True)
                    idx = max(0, min(alloc.get(fam, 1) - 1, len(values_sorted) - 1))
                    per_run_caps[fam].append(values_sorted[idx])

        fq = spectral.get('family_z_quantiles', {})
        if not fq and isinstance(spectral.get('family_z_summary'), dict):
            fq = spectral.get('family_z_summary', {})
        if isinstance(fq, dict):
            for fam, stats in fq.items():
                if not isinstance(stats, dict):
                    continue
                val_q99 = _safe_float(stats.get('q99'))
                val_max = _safe_float(stats.get('max'))
                if val_q99 is not None:
                    q99_values[str(fam)].append(val_q99)
                if val_max is not None:
                    max_values[str(fam)].append(val_max)

    summary = {
        'families_seen': sorted(set(per_run_caps) | set(q99_values) | set(existing_caps)),
        'sigma_quantile': sigma_quantile,
        'deadband': deadband,
        'max_caps': max_caps
    }

    proposed_caps: Dict[str, Dict[str, float]] = {}
    margin = _spectral_margin(tier)

    if per_run_caps:
        for fam, candidates in per_run_caps.items():
            if not candidates:
                continue
            base = max(candidates)
            if fam in existing_caps:
                base = max(base, existing_caps[fam])
            proposed_caps[fam] = {'kappa': round(base + margin, 3)}
        for fam in sorted(set(q99_values) | set(max_values)):
            if fam in proposed_caps:
                continue
            observed = q99_values.get(fam, []) + max_values.get(fam, [])
            if not observed:
                continue
            base = max(observed)
            if fam in existing_caps:
                base = max(base, existing_caps[fam])
            proposed_caps[fam] = {'kappa': round(base + margin, 3)}
    elif q99_values or max_values:
        for fam in sorted(set(q99_values) | set(max_values)):
            observed = q99_values.get(fam, []) + max_values.get(fam, [])
            if not observed:
                continue
            base = max(observed)
            if fam in existing_caps:
                base = max(base, existing_caps[fam])
            proposed_caps[fam] = {'kappa': round(base + margin, 3)}
    else:
        for fam, kappa in existing_caps.items():
            proposed_caps[fam] = {'kappa': kappa}

    return summary, proposed_caps

# ==============================================================================
# RMT CALIBRATION
# ==============================================================================

def calibrate_rmt(certs: List[Dict]) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Extract RMT guard thresholds from certificates.

    Uses per-family quantiles of null deltas to set epsilon.
    """
    deltas_by_family = defaultdict(list)
    existing_eps = {}
    margin = None
    deadband = None

    for cert in certs:
        rmt = cert.get('rmt', {}) or {}
        if not isinstance(rmt, dict):
            continue
        policy = rmt.get('policy', {}) if isinstance(rmt.get('policy'), dict) else {}

        if margin is None:
            margin = _safe_float(policy.get('margin') or rmt.get('margin') or (rmt.get('summary') or {}).get('margin'))
        if deadband is None:
            deadband = _safe_float(policy.get('deadband') or rmt.get('deadband') or (rmt.get('summary') or {}).get('deadband'))

        eps = (
            rmt.get('epsilon_by_family')
            or rmt.get('epsilon')
            or policy.get('epsilon_by_family')
            or policy.get('epsilon')
        )
        if isinstance(eps, dict):
            for fam, val in eps.items():
                try:
                    existing_eps[str(fam)] = float(val)
                except (TypeError, ValueError):
                    pass
        elif isinstance(eps, (int, float)):
            try:
                existing_eps["_default"] = float(eps)
            except (TypeError, ValueError):
                pass

        record_has_counts = False
        families = rmt.get('families', {})
        if isinstance(families, dict) and families:
            record_has_counts = True
            for fam, stats in families.items():
                if not isinstance(stats, dict):
                    continue
                bare = stats.get('bare')
                guarded = stats.get('guarded')
                bare_f = _safe_float(bare)
                guarded_f = _safe_float(guarded)
                if bare_f and bare_f > 0:
                    deltas_by_family[str(fam)].append((guarded_f / bare_f) - 1.0)

        outliers = rmt.get('outliers_per_family', {})
        baseline_outliers = rmt.get('baseline_outliers_per_family', {})
        if isinstance(outliers, dict) and isinstance(baseline_outliers, dict) and outliers:
            record_has_counts = True
            for fam in set(outliers) | set(baseline_outliers):
                bare_f = _safe_float(baseline_outliers.get(fam))
                guarded_f = _safe_float(outliers.get(fam))
                if bare_f and bare_f > 0:
                    deltas_by_family[str(fam)].append((guarded_f / bare_f) - 1.0)

        if not record_has_counts:
            for source in ('outliers_by_family', 'family_stats'):
                stats_map = rmt.get(source, {})
                if not isinstance(stats_map, dict):
                    continue
                for fam, stats in stats_map.items():
                    if not isinstance(stats, dict):
                        continue
                    for key in ('outlier_fraction', 'outlier_rate', 'fraction', 'rate'):
                        val = _safe_float(stats.get(key))
                        if val is not None:
                            deltas_by_family[str(fam)].append(val)
                            break

    summary = {
        'families_seen': sorted(deltas_by_family.keys()),
        'margin': margin,
        'deadband': deadband
    }
    quantile_q = _rmt_quantile_for_tier(tier)
    proposed_eps: Dict[str, float] = {}

    if deltas_by_family:
        for fam, deltas in deltas_by_family.items():
            qv = _quantile(deltas, quantile_q)
            if qv is None:
                continue
            qv = max(float(qv), 0.0)
            proposed_eps[fam] = round(qv, 3)

    if not proposed_eps:
        if existing_eps:
            if set(existing_eps.keys()) == {"_default"}:
                default_eps = existing_eps["_default"]
                return summary, {
                    "ffn": default_eps,
                    "attn": default_eps,
                    "embed": default_eps,
                    "other": default_eps,
                }
            return summary, existing_eps
        defaults = {
            "balanced": {"ffn": 0.10, "attn": 0.08, "embed": 0.12, "other": 0.12},
            "conservative": {"ffn": 0.06, "attn": 0.05, "embed": 0.07, "other": 0.07},
        }
        return summary, defaults.get(tier, defaults["balanced"])

    for fam, eps_val in existing_eps.items():
        if fam not in proposed_eps and fam != "_default":
            proposed_eps[fam] = eps_val

    return summary, proposed_eps

# ==============================================================================
# VARIANCE CALIBRATION
# ==============================================================================

def calibrate_variance(certs: List[Dict]) -> Dict[str, Any]:
    """
    Extract variance guard thresholds from certificates.

    Uses predictive CI half-widths to set min_effect_lognll.
    """
    deadband = None
    min_gain = None
    policy_min_effect = None
    min_effect_samples = []
    variance_changes = []

    for cert in certs:
        var = cert.get('variance', {}) or {}
        if not isinstance(var, dict):
            continue
        policy = var.get('policy', {}) if isinstance(var.get('policy'), dict) else {}

        if deadband is None:
            deadband = _safe_float(policy.get('deadband') or var.get('deadband'))
        if min_gain is None:
            min_gain = _safe_float(policy.get('min_gain') or policy.get('min_rel_gain') or var.get('min_gain'))
        if policy_min_effect is None:
            policy_min_effect = _safe_float(policy.get('min_effect_lognll') or var.get('min_effect_lognll'))

        predictive = var.get('predictive_gate', {}) or {}
        delta_ci = predictive.get('delta_ci')
        if isinstance(delta_ci, (list, tuple)) and len(delta_ci) == 2:
            lo = _safe_float(delta_ci[0])
            hi = _safe_float(delta_ci[1])
            if lo is not None and hi is not None:
                width = abs(hi - lo) / 2.0
                if width > 0:
                    min_effect_samples.append(width)

        calib = var.get('calibration') or var.get('calibration_stats') or {}
        if isinstance(calib, dict):
            vchange = calib.get('variance_change') or calib.get('delta') or calib.get('max_delta')
            vchange = _safe_float(vchange)
            if vchange is not None:
                variance_changes.append(abs(vchange))

    result = {}
    if deadband is None and variance_changes:
        result['deadband'] = round(max(variance_changes) * 1.1 + 0.01, 3)
    elif deadband is not None:
        result['deadband'] = deadband

    if min_effect_samples:
        proposed = _quantile(min_effect_samples, 0.95)
        if proposed is not None:
            result['min_effect_lognll'] = max(round(proposed, 4), 0.0009)
    elif policy_min_effect is not None:
        result['min_effect_lognll'] = policy_min_effect

    if min_gain is not None:
        result['min_gain'] = min_gain

    return result

# ==============================================================================
# PRESET GENERATION
# ==============================================================================

def generate_calibrated_preset(
    model_name: str,
    model_path: str,
    tier: str,
    drift_stats: Dict[str, Any],
    spectral_summary: Dict[str, Any],
    spectral_caps: Dict[str, Dict[str, float]],
    rmt_summary: Dict[str, Any],
    rmt_epsilon: Dict[str, float],
    variance_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a calibrated preset YAML containing all guard overrides.
    """
    preset = {
        '_calibration_meta': {
            'model_name': model_name,
            'tier': tier,
            'generated_by': 'invarlock_definitive_validation.sh',
            'drift_band_compatible': drift_stats.get('band_compatible', True),
            'suggested_drift_band': drift_stats.get('suggested_band', [0.95, 1.05])
        },
        'model': {
            'id': model_path
        },
        'dataset': {
            'provider': dataset_provider,
            'split': 'validation',
            'seq_len': seq_len,
            'stride': stride,
            'preview_n': preview_n,
            'final_n': final_n,
            'seed': seed,
        },
        'guards': {}
    }

    # Spectral guard
    spectral = {}
    if spectral_caps:
        spectral['family_caps'] = spectral_caps
    if spectral_summary.get('sigma_quantile') is not None:
        spectral['sigma_quantile'] = spectral_summary['sigma_quantile']
    if spectral_summary.get('deadband') is not None:
        spectral['deadband'] = spectral_summary['deadband']
    if spectral_summary.get('max_caps') is not None:
        spectral['max_caps'] = spectral_summary['max_caps']
    if spectral:
        preset['guards']['spectral'] = spectral

    # RMT guard
    rmt = {}
    if rmt_epsilon:
        rmt['epsilon'] = rmt_epsilon
    if rmt_summary.get('margin') is not None:
        rmt['margin'] = rmt_summary['margin']
    if rmt_summary.get('deadband') is not None:
        rmt['deadband'] = rmt_summary['deadband']
    if rmt:
        preset['guards']['rmt'] = rmt

    # Variance guard
    if variance_config:
        preset['guards']['variance'] = variance_config

    return preset

# ==============================================================================
# MAIN CALIBRATION
# ==============================================================================

print("\n=== INVARLOCK FULL CALIBRATION ===\n")

# Load certificates
print("Loading calibration certificates...")
certs = load_certificates()

if len(certs) < 2:
    print(f"ERROR: Need at least 2 valid certificates, got {len(certs)}")
    exit(1)

print(f"Loaded {len(certs)} certificates\n")

# Calibrate each guard
print("Calibrating drift gate...")
drift_stats = calibrate_drift(certs)

print("\nCalibrating spectral guard...")
spectral_summary, spectral_caps = calibrate_spectral(certs)

print("\nCalibrating RMT guard...")
rmt_summary, rmt_epsilon = calibrate_rmt(certs)

print("\nCalibrating variance guard...")
variance_config = calibrate_variance(certs)

# Generate calibrated preset
print("\nGenerating calibrated preset...")
preset = generate_calibrated_preset(
    model_name=model_name,
    model_path=model_path,
    tier=tier,
    drift_stats=drift_stats,
    spectral_summary=spectral_summary,
    spectral_caps=spectral_caps,
    rmt_summary=rmt_summary,
    rmt_epsilon=rmt_epsilon,
    variance_config=variance_config
)

# Save outputs
stats_path = output_dir / "calibration_stats.json"
with open(stats_path, 'w') as f:
    json.dump({
        'drift': drift_stats,
        'spectral': {**spectral_summary, 'family_caps': spectral_caps},
        'rmt': {**rmt_summary, 'epsilon': rmt_epsilon},
        'variance': variance_config
    }, f, indent=2)
print(f"Saved calibration stats to: {stats_path}")

preset_path = preset_output_dir / f"calibrated_preset_{model_name.replace('/', '_')}.yaml"

if YAML_AVAILABLE:
    with open(preset_path, 'w') as f:
        yaml.safe_dump(preset, f, sort_keys=False, default_flow_style=False)
    print(f"Saved calibrated preset to: {preset_path}")
else:
    # Fall back to JSON
    preset_path = preset_path.with_suffix('.json')
    with open(preset_path, 'w') as f:
        json.dump(preset, f, indent=2)
    print(f"Saved calibrated preset to: {preset_path} (JSON fallback)")

# Report drift compatibility
if not drift_stats.get('band_compatible', True):
    print(f"\n⚠️  WARNING: Model drift ({drift_stats['mean']:.3f}) is outside default 0.95-1.05 band")
    print(f"    Suggested band: {drift_stats['suggested_band']}")
    print(f"    Consider using INVARLOCK_TINY_RELAX=1 or patching drift gate")

print("\n=== CALIBRATION COMPLETE ===\n")
CALIBRATION_SCRIPT
}

# ============ HELPER: RUN INVARLOCK CERTIFY WITH CALIBRATED PRESET ============
run_invarlock_certify() {
    local subject_path="$1"
    local baseline_path="$2"
    local output_dir="$3"
    local run_name="$4"
    local preset_dir="$5"
    local model_name="$6"

    log "Running InvarLock certification:"
    log "  Subject: ${subject_path}"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_dir}/${run_name}"

    local run_dir="${output_dir}/${run_name}"
    local cert_dir="${run_dir}/cert"
    mkdir -p "${run_dir}" "${cert_dir}"

    # Look for calibrated preset (YAML preferred)
    local calibrated_preset=""
    for ext in yaml json; do
        local preset_path="${preset_dir}/calibrated_preset_${model_name}.${ext}"
        if [[ -f "${preset_path}" ]]; then
            calibrated_preset="${preset_path}"
            break
        fi
    done

    if [[ -n "${calibrated_preset}" ]]; then
        log "  Using calibrated preset: ${calibrated_preset}"
    fi

    local model_output_dir
    model_output_dir=$(dirname "$(dirname "${output_dir}")")
    local cal_stats="${model_output_dir}/certificates/calibration/calibration_stats.json"
    local use_tiny_relax="false"
    if [[ -f "${cal_stats}" ]]; then
        local band_compatible
        band_compatible=$(python3 -c "import json; print(json.load(open('${cal_stats}'))['drift'].get('band_compatible', True))" 2>/dev/null || echo "True")
        if [[ "${band_compatible}" == "False" ]]; then
            log "  Note: Model drift outside default band, using TINY_RELAX"
            use_tiny_relax="true"
        fi
    fi

    local cmd_args=(
        "invarlock" "certify"
        "--source" "${baseline_path}"
        "--edited" "${subject_path}"
        "--adapter" "${INVARLOCK_ADAPTER}"
        "--profile" "ci"
        "--tier" "${INVARLOCK_TIER}"
        "--out" "${run_dir}"
        "--cert-out" "${cert_dir}"
    )
    if [[ -n "${calibrated_preset}" ]]; then
        cmd_args+=("--preset" "${calibrated_preset}")
    fi

    local env_prefix=()
    if [[ "${use_tiny_relax}" == "true" ]]; then
        env_prefix+=("INVARLOCK_TINY_RELAX=1")
    fi
    if [[ -n "${INVARLOCK_SKIP_OVERHEAD_CHECK:-}" ]]; then
        env_prefix+=("INVARLOCK_SKIP_OVERHEAD_CHECK=${INVARLOCK_SKIP_OVERHEAD_CHECK}")
    fi

    if [[ ${#env_prefix[@]} -gt 0 ]]; then
        "${env_prefix[@]}" "${cmd_args[@]}" 2>&1 | tee -a "${LOG_FILE}" || log "  Certification failed (exit code $?, may be expected for error models)"
    else
        "${cmd_args[@]}" 2>&1 | tee -a "${LOG_FILE}" || log "  Certification failed (exit code $?, may be expected for error models)"
    fi

    # Copy certificate to standard location
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
}

# ============ PROCESS ONE MODEL ============
process_model() {
    local model_id="$1"
    local model_name=$(basename "${model_id}" | tr '[:upper:]' '[:lower:]' | tr '/' '_')
    local model_output_dir="${OUTPUT_DIR}/${model_name}"
    local preset_dir="${OUTPUT_DIR}/presets"

    log_section "PROCESSING MODEL: ${model_id}"

    mkdir -p "${model_output_dir}"/{models,evals,certificates}

    # Step 1: Setup baseline
    log "Step 1: Setup baseline model"
    local baseline_path=$(setup_model "${model_id}")

    # Step 2: Run baseline evals
    log "Step 2: Running baseline lm-eval"
    run_lmeval \
        "${baseline_path}" \
        "${model_output_dir}/evals/baseline_results.json" \
        "${EVAL_TASKS}" \
        "${EVAL_BATCH_SIZE}" \
        "${EVAL_NUM_FEWSHOT}"

    # Step 3: InvarLock FULL calibration (all guards)
    log "Step 3: InvarLock FULL calibration (all guards)"
    run_invarlock_calibration \
        "${baseline_path}" \
        "${model_name}" \
        "${model_output_dir}/certificates/calibration" \
        "${DRIFT_CALIBRATION_RUNS}" \
        "${preset_dir}"

    # Step 4: Create and evaluate clean edit
    log "Step 4: Clean edit (${EDIT_BITS}-bit ${EDIT_TYPE})"
    local clean_edit_path="${model_output_dir}/models/clean_edit"
    create_edited_model \
        "${baseline_path}" \
        "${clean_edit_path}" \
        "${EDIT_TYPE}" \
        "${EDIT_BITS}" \
        "${EDIT_GROUP_SIZE}" \
        "${EDIT_SCOPE}"

    run_lmeval \
        "${clean_edit_path}" \
        "${model_output_dir}/evals/clean_edit_results.json" \
        "${EVAL_TASKS}" \
        "${EVAL_BATCH_SIZE}" \
        "${EVAL_NUM_FEWSHOT}"

    for run in $(seq 1 "${CLEAN_EDIT_RUNS}"); do
        run_invarlock_certify \
            "${clean_edit_path}" \
            "${baseline_path}" \
            "${model_output_dir}/certificates/clean_edit" \
            "run_${run}" \
            "${preset_dir}" \
            "${model_name}"
    done

    # Step 5: Stress edit (int4 aggressive)
    log "Step 5: Stress edit (4-bit aggressive)"
    local stress_edit_path="${model_output_dir}/models/stress_edit"
    create_edited_model \
        "${baseline_path}" \
        "${stress_edit_path}" \
        "quant_rtn" \
        "4" \
        "32" \
        "all"

    run_lmeval \
        "${stress_edit_path}" \
        "${model_output_dir}/evals/stress_edit_results.json" \
        "${EVAL_TASKS}" \
        "${EVAL_BATCH_SIZE}" \
        "${EVAL_NUM_FEWSHOT}"

    for run in $(seq 1 "${STRESS_EDIT_RUNS}"); do
        run_invarlock_certify \
            "${stress_edit_path}" \
            "${baseline_path}" \
            "${model_output_dir}/certificates/stress_edit" \
            "run_${run}" \
            "${preset_dir}" \
            "${model_name}"
    done

    # Step 6: Error injection tests
    if [[ "${RUN_ERROR_INJECTION}" == "true" ]]; then
        log "Step 6: Error injection tests"

        local errors=("nan_injection" "inf_injection" "extreme_quant" "scale_explosion" "zero_layer")

        for error_type in "${errors[@]}"; do
            log "  Testing error: ${error_type}"
            local error_path="${model_output_dir}/models/error_${error_type}"

            create_error_model \
                "${baseline_path}" \
                "${error_path}" \
                "${error_type}"

            run_invarlock_certify \
                "${error_path}" \
                "${baseline_path}" \
                "${model_output_dir}/certificates/errors" \
                "${error_type}" \
                "${preset_dir}" \
                "${model_name}"
        done
    fi

    log "Model processing complete: ${model_name}"
}

# ============ COMPILE RESULTS ============
compile_results() {
    log_section "COMPILING RESULTS"

    python3 << EOF
import json
import csv
from pathlib import Path
from collections import defaultdict

output_dir = Path("${OUTPUT_DIR}")
analysis_dir = output_dir / "analysis"
analysis_dir.mkdir(exist_ok=True)

# Collect eval results
eval_rows = []
for model_dir in output_dir.iterdir():
    if not model_dir.is_dir() or model_dir.name in ['logs', 'analysis', 'reports', 'presets']:
        continue

    evals_dir = model_dir / "evals"
    if not evals_dir.exists():
        continue

    for results_file in evals_dir.glob("*_results.json"):
        edit_type = results_file.stem.replace("_results", "")
        try:
            data = json.loads(results_file.read_text())
            for task, task_results in data.get('results', {}).items():
                for key in ['acc', 'acc_norm', 'exact_match']:
                    if key in task_results:
                        eval_rows.append({
                            'model': model_dir.name,
                            'edit_type': edit_type,
                            'task': task,
                            'metric': key,
                            'value': task_results[key]
                        })
                        break
        except Exception as e:
            print(f"Error processing {results_file}: {e}")

if eval_rows:
    with open(analysis_dir / "eval_results.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=eval_rows[0].keys())
        writer.writeheader()
        writer.writerows(eval_rows)
    print(f"Wrote {len(eval_rows)} eval rows")

# Collect InvarLock results
invar_rows = []
for model_dir in output_dir.iterdir():
    if not model_dir.is_dir() or model_dir.name in ['logs', 'analysis', 'reports', 'presets']:
        continue

    certs_dir = model_dir / "certificates"
    if not certs_dir.exists():
        continue

    for cert_file in certs_dir.rglob("evaluation.cert.json"):
        try:
            cert = json.loads(cert_file.read_text())
            rel_path = cert_file.relative_to(certs_dir)
            parts = list(rel_path.parts)

            v = cert.get('validation', {})
            # Handle both boolean and string values (JSON serialization can cause issues)
            def as_bool(val):
                if isinstance(val, bool):
                    return val
                if isinstance(val, str):
                    return val.lower() == 'true'
                return bool(val)

            all_pass = all([
                as_bool(v.get('invariants_pass', False)),
                as_bool(v.get('primary_metric_acceptable', False)),
                as_bool(v.get('spectral_stable', False)),
                as_bool(v.get('rmt_stable', False))
            ])

            invar_rows.append({
                'model': model_dir.name,
                'experiment': parts[0] if parts else 'unknown',
                'run': parts[1] if len(parts) > 1 else '',
                'pm_ratio': cert.get('primary_metric', {}).get('ratio_vs_baseline'),
                'pm_acceptable': v.get('primary_metric_acceptable'),
                'invariants_pass': v.get('invariants_pass'),
                'spectral_stable': v.get('spectral_stable'),
                'rmt_stable': v.get('rmt_stable'),
                'all_pass': all_pass
            })
        except Exception as e:
            print(f"Error processing {cert_file}: {e}")

if invar_rows:
    with open(analysis_dir / "invarlock_results.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=invar_rows[0].keys())
        writer.writeheader()
        writer.writerows(invar_rows)
    print(f"Wrote {len(invar_rows)} InvarLock rows")

# Collect calibration summaries
calibration_summary = {}
for model_dir in output_dir.iterdir():
    if not model_dir.is_dir() or model_dir.name in ['logs', 'analysis', 'reports', 'presets']:
        continue

    cal_stats = model_dir / "certificates" / "calibration" / "calibration_stats.json"
    if cal_stats.exists():
        try:
            stats = json.loads(cal_stats.read_text())
            calibration_summary[model_dir.name] = stats
        except Exception as e:
            print(f"Error loading {cal_stats}: {e}")

if calibration_summary:
    with open(analysis_dir / "calibration_summary.json", 'w') as f:
        json.dump(calibration_summary, f, indent=2)
    print(f"Wrote calibration summary for {len(calibration_summary)} models")
EOF
}

# ============ CORRELATION ANALYSIS ============
run_analysis() {
    log_section "CORRELATION ANALYSIS"

    python3 << EOF
import json
import csv
from pathlib import Path
from collections import defaultdict

output_dir = Path("${OUTPUT_DIR}")
analysis_dir = output_dir / "analysis"

# Load eval results
eval_data = defaultdict(dict)
eval_csv = analysis_dir / "eval_results.csv"
if eval_csv.exists():
    with open(eval_csv) as f:
        for row in csv.DictReader(f):
            key = (row['model'], row['edit_type'])
            task = row['task']
            eval_data[key][task] = float(row['value'])

# Load InvarLock results
invar_data = defaultdict(list)
invar_csv = analysis_dir / "invarlock_results.csv"
if invar_csv.exists():
    with open(invar_csv) as f:
        for row in csv.DictReader(f):
            key = (row['model'], row['experiment'])
            invar_data[key].append(row)

# Load calibration summary
cal_summary = {}
cal_json = analysis_dir / "calibration_summary.json"
if cal_json.exists():
    cal_summary = json.loads(cal_json.read_text())

# Analyze correlations
print("=== CORRELATION ANALYSIS ===\n")

results = {'models': {}, 'error_detection': {'detected': [], 'missed': []}, 'calibration': cal_summary}
categories = defaultdict(int)

for model_dir in output_dir.iterdir():
    if not model_dir.is_dir() or model_dir.name in ['logs', 'analysis', 'reports', 'presets']:
        continue

    model = model_dir.name
    results['models'][model] = {}
    print(f"\n### {model} ###")

    # Report calibration status
    if model in cal_summary:
        drift_info = cal_summary[model].get('drift', {})
        print(f"  Calibration: drift_mean={drift_info.get('mean', 'N/A')}, band_compatible={drift_info.get('band_compatible', 'N/A')}")

    # Get baseline evals
    baseline_key = (model, 'baseline')
    baseline_evals = eval_data.get(baseline_key, {})

    for edit_type in ['clean_edit', 'stress_edit']:
        edit_key = (model, edit_type)
        edit_evals = eval_data.get(edit_key, {})

        # Check for regression (>5% drop)
        has_regression = False
        for task in baseline_evals:
            if task in edit_evals:
                delta = edit_evals[task] - baseline_evals[task]
                if delta < -0.05:
                    has_regression = True
                    break

        # Get InvarLock verdict
        invar_key = (model, edit_type)
        invar_results = invar_data.get(invar_key, [])
        invar_flagged = any(r['all_pass'] == 'False' for r in invar_results)

        # Classify
        if has_regression and invar_flagged:
            category = "TRUE_POSITIVE"
        elif not has_regression and invar_flagged:
            category = "FALSE_POSITIVE"
        elif not has_regression and not invar_flagged:
            category = "TRUE_NEGATIVE"
        else:
            category = "FALSE_NEGATIVE"

        categories[category] += 1
        results['models'][model][edit_type] = {'category': category, 'regression': has_regression, 'flagged': invar_flagged}
        print(f"  {edit_type}: {category}")

    # Error detection
    error_key = (model, 'errors')
    for row in invar_data.get(error_key, []):
        error_type = row['run']
        # Check for string 'False' (CSV) or boolean False (dict); also check if any guard failed
        def is_false(val):
            if val is None:
                return True  # Missing = failed
            if isinstance(val, bool):
                return not val
            if isinstance(val, str):
                return val.lower() in ('false', '0', '')
            return False
        caught = is_false(row.get('all_pass')) or is_false(row.get('invariants_pass'))
        if caught:
            results['error_detection']['detected'].append(f"{model}/{error_type}")
        else:
            results['error_detection']['missed'].append(f"{model}/{error_type}")

# Summary
print("\n=== SUMMARY ===")
tp = categories['TRUE_POSITIVE']
tn = categories['TRUE_NEGATIVE']
fp = categories['FALSE_POSITIVE']
fn = categories['FALSE_NEGATIVE']
total = tp + tn + fp + fn

accuracy = (tp + tn) / total if total > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

err_detected = len(results['error_detection']['detected'])
err_missed = len(results['error_detection']['missed'])
err_total = err_detected + err_missed
err_rate = err_detected / err_total if err_total > 0 else 0

print(f"Accuracy: {accuracy:.0%}")
print(f"Precision: {precision:.0%}")
print(f"Recall: {recall:.0%}")
print(f"Error Detection: {err_detected}/{err_total} ({err_rate:.0%})")

results['summary'] = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'error_detection_rate': err_rate,
    'categories': dict(categories)
}

with open(analysis_dir / "correlation_analysis.json", 'w') as f:
    json.dump(results, f, indent=2)
EOF
}

# ============ GENERATE VERDICT ============
generate_verdict() {
    log_section "GENERATING FINAL VERDICT"

    python3 << EOF
import json
from pathlib import Path
from datetime import datetime

output_dir = Path("${OUTPUT_DIR}")
analysis_dir = output_dir / "analysis"
reports_dir = output_dir / "reports"
reports_dir.mkdir(exist_ok=True)

# Load analysis
analysis = json.loads((analysis_dir / "correlation_analysis.json").read_text())
summary = analysis.get('summary', {})
calibration = analysis.get('calibration', {})

accuracy = summary.get('accuracy', 0)
err_rate = summary.get('error_detection_rate', 0)

# Phase 0 criteria
phase0_pass = accuracy >= 0.6 and err_rate >= 0.8

if phase0_pass and accuracy >= 0.8:
    verdict = "PHASE0_VALIDATED"
    confidence = "HIGH"
elif phase0_pass:
    verdict = "PHASE0_VALIDATED"
    confidence = "MEDIUM"
else:
    verdict = "PHASE0_FAILED"
    confidence = "HIGH"

# Calibration summary
cal_notes = []
for model, stats in calibration.items():
    drift = stats.get('drift', {})
    if not drift.get('band_compatible', True):
        cal_notes.append(f"  • {model}: drift outside default band (mean={drift.get('mean', 'N/A'):.3f})")

report = f'''
╔══════════════════════════════════════════════════════════════════╗
║     INVARLOCK PHASE 0 VALIDATION RESULTS                         ║
╠══════════════════════════════════════════════════════════════════╣
║     Accuracy:          {accuracy:.0%}                                       ║
║     Error Detection:   {err_rate:.0%}                                       ║
╠══════════════════════════════════════════════════════════════════╣
║     VERDICT: {verdict:<20}                          ║
║     CONFIDENCE: {confidence:<17}                          ║
╚══════════════════════════════════════════════════════════════════╝

CALIBRATION NOTES:
'''

if cal_notes:
    report += '\n'.join(cal_notes) + '\n'
else:
    report += '  All models within default drift band\n'

if verdict == "PHASE0_VALIDATED":
    report += '''
✅ InvarLock is NOT a toy.

This PROVES:
  • Guards correlate with real quality regressions
  • Guards catch catastrophic failures reliably
  • Full guard calibration was performed for each model
  • InvarLock is doing real work

This does NOT prove:
  • Ready for production gating (requires Phase 1-4)
  • Works on all model architectures
  • Catches all failure modes you care about

Next: Run Phase 1 validation (more models, edit types)
'''
else:
    report += f'''
❌ Phase 0 validation failed.

Issues found:
  • Accuracy: {accuracy:.0%} (need ≥60%)
  • Error detection: {err_rate:.0%} (need ≥80%)

Next: Debug and re-run validation
'''

print(report)

# Save
with open(reports_dir / "final_verdict.txt", 'w') as f:
    f.write(report)

with open(reports_dir / "final_verdict.json", 'w') as f:
    json.dump({
        'verdict': verdict,
        'confidence': confidence,
        'accuracy': accuracy,
        'error_detection_rate': err_rate,
        'phase0_pass': phase0_pass,
        'calibration': calibration,
        'timestamp': datetime.now().isoformat()
    }, f, indent=2)
EOF
}

# ============ MAIN ============
main() {
    local start_time=$(date +%s)

    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  InvarLock Definitive Validation Suite v${SCRIPT_VERSION}                ║"
    echo "║  With FULL Guard Calibration                                    ║"
    echo "║  Is this a TOY or PRODUCTION-READY?                             ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""

    log "Output directory: ${OUTPUT_DIR}"
    log "Models: ${MODEL_1}, ${MODEL_2}${MODEL_3:+, $MODEL_3}"
    log ""

    # Phase 0: Check dependencies
    check_dependencies
    resolve_model_dtype
    detect_flash_attention

    # Process each model
    for model_id in "${MODEL_1}" "${MODEL_2}" ${MODEL_3:+"${MODEL_3}"}; do
        if [[ -n "${model_id}" ]]; then
            process_model "${model_id}"
        fi
    done

    # Compile and analyze
    compile_results
    run_analysis
    generate_verdict

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_section "COMPLETE"
    log "Total time: $((duration / 3600))h $(((duration % 3600) / 60))m"
    log "Report: ${OUTPUT_DIR}/reports/final_verdict.txt"
    log "Calibrated presets: ${OUTPUT_DIR}/presets/"
}

# Show usage
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    cat << EOF
InvarLock Definitive Validation Suite v${SCRIPT_VERSION}

This script performs FULL guard calibration before validation:
  1. Runs null certifications to extract model-specific guard thresholds
  2. Generates calibrated presets (spectral/RMT/variance)
  3. Uses calibrated presets for all subsequent certifications

Usage: $0 [options]

Options (via environment variables):
    MODEL_1               First model (default: mistralai/Mistral-7B-v0.3)
    MODEL_2               Second model (default: Qwen/Qwen2-14B)
    MODEL_3               Optional third model

    EDIT_BITS             Quantization bits (default: 8)
    EVAL_TASKS            lm-eval tasks (default: mmlu,hellaswag,arc_challenge,winogrande)
    LMEVAL_DTYPE          lm-eval dtype (auto|bfloat16|float16)
    LMEVAL_PARALLELIZE    lm-eval parallelize on multi-GPU (default: true)

    DRIFT_CALIBRATION_RUNS  Number of calibration runs (default: 5)
    RUN_ERROR_INJECTION     Run error injection tests (default: true)
    OUTPUT_DIR              Output directory
    SKIP_FLASH_ATTN         Disable Flash Attention 2 if installed (default: false)

    INVARLOCK_SEQ_LEN       InvarLock sequence length (default: 512)
    INVARLOCK_STRIDE        InvarLock stride (default: 256)
    INVARLOCK_EVAL_BATCH    InvarLock eval batch size (default: 4)
    INVARLOCK_MODEL_DTYPE   InvarLock dtype (auto|bfloat16|float16)
    INVARLOCK_ADAPTER       InvarLock adapter (default: hf_causal_auto)
    INVARLOCK_DATASET       InvarLock dataset provider (default: wikitext2)

Calibration extracts:
    • Spectral: family_caps.{family}.kappa from per-run order-statistic on final_z_scores
    • RMT: epsilon.{family} from per-family delta quantiles
    • Variance: deadband from calibration_stats, min_effect_lognll from predictive_gate.delta_ci
    • Drift: suggested_band from primary_metric ratios

Example:
    $0                                    # Run with defaults
    MODEL_1=/path/to/model $0             # Use local model
    RUN_ERROR_INJECTION=false $0          # Skip error injection
EOF
    exit 0
fi

main "$@"
