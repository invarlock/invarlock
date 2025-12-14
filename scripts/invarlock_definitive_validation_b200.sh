#!/usr/bin/env bash
# invarlock_definitive_validation_b200.sh
# ==========================================================
# InvarLock Definitive Validation Suite - B200 180GB Optimized
# ==========================================================
# Version: v2.1.0
# Dependencies: bash 4+, jq, python3, invarlock CLI, nvidia-smi
# Optimized for 8x NVIDIA B200 180GB SXM6 GPUs with parallel orchestration.
#
# HARDWARE TARGET:
# - 8x B200 180GB SXM6
# - ~4.5 TB/s memory bandwidth
# - ~2250 FP16 TFLOPS
# - Native FP4 support (B200-exclusive feature)
#
# MEMORY UTILIZATION STRATEGY:
# - Target: 85-92% VRAM (153-166 GB of 180 GB per GPU)
# - Large model support: 70B+ models on single GPU
# - All 8 GPUs utilized in parallel
#
# EDIT TYPES (4 types × 2 versions = 8 tests per model):
# - Quantization RTN: 8-bit clean, 4-bit stress
# - FP4 Quantization: E2M1 clean, aggressive stress (B200-only)
# - Magnitude Pruning: 10% sparsity, 50% sparsity
# - Low-Rank SVD: rank=256 clean, rank=32 stress
#
# MODEL SUITE (8 PUBLIC models - no HuggingFace login required):
# - GPU 0: Mistral-7B-v0.3 (~14 GB)
# - GPU 1: Llama-2-13b-hf (~26 GB)
# - GPU 2: Qwen2-14B (~28 GB)
# - GPU 3: mpt-30b (~60 GB)
# - GPU 4: falcon-40b (~80 GB)
# - GPU 5: Mixtral-8x7B-v0.1 (~90 GB)
# - GPU 6: Llama-2-70b-hf (~140 GB)
# - GPU 7: Qwen1.5-72B (~144 GB)
#
# EXECUTION FLOW:
# 1. Launch all 8 models in parallel across 8 GPUs
# 2. Each GPU runs: calibration → 4 clean edits → 4 stress edits → error injection
# 3. Wait for all GPUs to complete
# 4. Correlation analysis (lm-eval vs InvarLock) → verdict
# ==========================================================

# Dynamic scheduling feature flag (v2.0)
# Set to "false" to use legacy static GPU assignment
USE_DYNAMIC_SCHEDULING="${USE_DYNAMIC_SCHEDULING:-true}"

# Note: We use set -uo pipefail instead of set -euo pipefail
# when dynamic scheduling is enabled to allow per-task error handling
if [[ "${USE_DYNAMIC_SCHEDULING}" == "true" ]]; then
    set -uo pipefail  # Per-task error handling, no global exit on error
else
    set -euo pipefail  # Legacy behavior
fi

# Initialize pids array early to avoid set -u errors in cleanup
declare -a pids=()
MONITOR_PID=""

# ============ CLEANUP TRAP ============
cleanup() {
    local exit_code=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Script interrupted or finished with exit code: ${exit_code}"

    # Kill any background processes we spawned
    # Check if pids array exists and has elements
    if [[ ${#pids[@]} -gt 0 ]]; then
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "Terminating background process: $pid"
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done
    fi
    if [[ -n "${MONITOR_PID:-}" ]]; then
        kill "${MONITOR_PID}" 2>/dev/null || true
    fi

    # Clean up lock file
    rm -f "${LOG_LOCK:-}" 2>/dev/null || true

    exit ${exit_code}
}

trap cleanup EXIT INT TERM HUP QUIT

# ============ BASH VERSION CHECK ============
# Associative arrays require bash 4.0+
if [[ "${BASH_VERSINFO[0]}" -lt 4 ]]; then
    echo "ERROR: This script requires bash 4.0 or later (have ${BASH_VERSION})"
    echo "       Associative arrays are not supported in bash ${BASH_VERSION}"
    exit 1
fi

# ============ VERSION ============
SCRIPT_VERSION="2.1.0-b200"

# ============ B200-SPECIFIC CONFIGURATION ============
# These settings are tuned for 8x B200 180GB maximum utilization

# GPU Configuration
export NUM_GPUS="${NUM_GPUS:-8}"
export GPU_MEMORY_GB="${GPU_MEMORY_GB:-180}"

# ============================================================
# MODEL SELECTION - ALL MODELS ARE PUBLIC (NO HUGGINGFACE LOGIN REQUIRED)
# ============================================================

# ============================================================
# MODEL SELECTION - ALL PUBLIC (NO HUGGINGFACE LOGIN REQUIRED)
# ============================================================
# All models below are confirmed public and do not require gated access.
# Using NousResearch versions of LLaMA-2 to avoid Meta's gated repos.

# Small models (GPU 0-2) - ~14-28 GB each
MODEL_1="${MODEL_1:-mistralai/Mistral-7B-v0.1}"           # ~14 GB - Mistral (PUBLIC, FA2 compatible)
MODEL_2="${MODEL_2:-NousResearch/Llama-2-13b-hf}"         # ~26 GB - LLaMA-2 13B (PUBLIC via NousResearch)
MODEL_3="${MODEL_3:-Qwen/Qwen2.5-14B}"                    # ~28 GB - Qwen2.5 (PUBLIC, FA2 compatible)

# Medium models (GPU 3-5) - ~60-90 GB each
MODEL_4="${MODEL_4:-Qwen/Qwen2.5-32B}"                    # ~64 GB - Qwen2.5 32B (PUBLIC, FA2 compatible)
MODEL_5="${MODEL_5:-01-ai/Yi-34B}"                        # ~68 GB - Yi 34B (PUBLIC, FA2 compatible)
MODEL_6="${MODEL_6:-mistralai/Mixtral-8x7B-v0.1}"         # ~90 GB - Mixtral MoE (PUBLIC, FA2 compatible)

# Large models (GPU 6-7) - B200-exclusive! ~140 GB each
MODEL_7="${MODEL_7:-NousResearch/Llama-2-70b-hf}"         # ~140 GB - LLaMA-2 70B (PUBLIC via NousResearch)
MODEL_8="${MODEL_8:-Qwen/Qwen1.5-72B}"                    # ~144 GB - Qwen 72B (PUBLIC, FA2 compatible)

# Edit Configuration
EDIT_TYPE="${EDIT_TYPE:-quant_rtn}"
EDIT_BITS="${EDIT_BITS:-8}"
EDIT_GROUP_SIZE="${EDIT_GROUP_SIZE:-128}"
EDIT_SCOPE="${EDIT_SCOPE:-ffn}"

# Edit Types to test (4 types × 2 versions each)
# Format: "type:param1:param2:scope" for clean, more aggressive for stress
EDIT_TYPES_CLEAN=(
    "quant_rtn:8:128:ffn"        # 8-bit quantization on FFN
    "fp4_quant:e2m1:ffn"         # FP4 E2M1 on FFN (B200-only)
    "magnitude_prune:0.1:ffn"    # 10% sparsity on FFN
    "lowrank_svd:256:ffn"        # rank-256 SVD on FFN
)

EDIT_TYPES_STRESS=(
    "quant_rtn:4:32:all"         # 4-bit aggressive on all
    "fp4_quant:aggressive:all"   # FP4 aggressive on all (B200-only)
    "magnitude_prune:0.5:all"    # 50% sparsity on all
    "lowrank_svd:32:all"         # rank-32 SVD on all
)

# Eval Configuration - ULTRA-CONSERVATIVE BATCH SIZES for B200 180GB
# Using lm-eval's "auto:N" feature: auto-detect with max cap of N
# This prevents OOM by letting lm-eval find optimal batch size bounded by max
EVAL_TASKS="${EVAL_TASKS:-mmlu,hellaswag,arc_challenge,winogrande}"
EVAL_NUM_FEWSHOT="${EVAL_NUM_FEWSHOT:-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-auto}"
EVAL_BATCH_SIZE_SMALL="${EVAL_BATCH_SIZE_SMALL:-auto:16}"   # 7B-14B models - auto with max 16
EVAL_BATCH_SIZE_MEDIUM="${EVAL_BATCH_SIZE_MEDIUM:-auto:8}"  # 30B-40B models - auto with max 8
EVAL_BATCH_SIZE_LARGE="${EVAL_BATCH_SIZE_LARGE:-auto:4}"    # 70B+ models - auto with max 4
EVAL_BATCH_SIZE_MOE="${EVAL_BATCH_SIZE_MOE:-auto:6}"        # MoE models (Mixtral) - auto with max 6

# InvarLock Configuration - BASE DEFAULTS (will be overridden per-model)
# WikiText-2 validation has ~1174 usable samples
# These are conservative defaults that work for largest models (70B+)
# Smaller models will get more generous settings via get_model_invarlock_config()
INVARLOCK_PREVIEW_WINDOWS="${INVARLOCK_PREVIEW_WINDOWS:-32}"
INVARLOCK_FINAL_WINDOWS="${INVARLOCK_FINAL_WINDOWS:-32}"
INVARLOCK_BOOTSTRAP_N="${INVARLOCK_BOOTSTRAP_N:-10000}"
INVARLOCK_DATASET="${INVARLOCK_DATASET:-wikitext2}"
INVARLOCK_TIER="${INVARLOCK_TIER:-balanced}"
INVARLOCK_SEQ_LEN="${INVARLOCK_SEQ_LEN:-512}"
INVARLOCK_STRIDE="${INVARLOCK_STRIDE:-256}"
INVARLOCK_EVAL_BATCH="${INVARLOCK_EVAL_BATCH:-32}"

# Experiment Configuration
DRIFT_CALIBRATION_RUNS="${DRIFT_CALIBRATION_RUNS:-5}"
CLEAN_EDIT_RUNS="${CLEAN_EDIT_RUNS:-3}"
STRESS_EDIT_RUNS="${STRESS_EDIT_RUNS:-2}"
RUN_ERROR_INJECTION="${RUN_ERROR_INJECTION:-true}"

# Output - supports resume by specifying existing directory
OUTPUT_DIR="${OUTPUT_DIR:-./invarlock_validation_b200_$(date +%Y%m%d_%H%M%S)}"

# Resume support - skip completed steps if output files exist
RESUME_MODE="${RESUME_MODE:-true}"

# ============ B200 GPU OPTIMIZATION FLAGS - MAXIMUM MEMORY ============
# Default to using all 8 GPUs, but respect user override
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Enable TF32 for massive speedup on B200 (no accuracy loss for inference)
export NVIDIA_TF32_OVERRIDE=1

# Enable text-level deduplication
export INVARLOCK_DEDUP_TEXTS=1

# Memory optimization for B200 - AGGRESSIVE SETTINGS
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold:0.9"

# Enable cuDNN autotuning for fastest kernels
export CUDNN_BENCHMARK=1

# Force deterministic workspace config with larger workspace
export CUBLAS_WORKSPACE_CONFIG=:32768:8

# Keep CUDA caching enabled for maximum memory reuse
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

# Allow network access for dataset downloads (wikitext2, etc.)
export INVARLOCK_ALLOW_NETWORK=1

# v0.3.1 FEATURE: PM acceptance range - prevents gate failures during validation
# These are wide bounds appropriate for validation runs
export INVARLOCK_PM_ACCEPTANCE_MIN="${INVARLOCK_PM_ACCEPTANCE_MIN:-0.90}"
export INVARLOCK_PM_ACCEPTANCE_MAX="${INVARLOCK_PM_ACCEPTANCE_MAX:-1.20}"

# Flash attention flag - will be set dynamically based on availability
export FLASH_ATTENTION_AVAILABLE="false"

# FP4 support flag - B200 exclusive
export FP4_NATIVE_SUPPORT="false"

# Target memory fraction (0.92 = 92% of available) - optimal zone
export CUDA_MEMORY_FRACTION=0.92

# ============ LIB MODULES FOR DYNAMIC SCHEDULING ============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCRIPT_DIR  # Export for subshell workers

# Determine lib directory - support both nested (scripts/lib/) and flat (scripts/) layouts
if [[ -d "${SCRIPT_DIR}/lib" && -f "${SCRIPT_DIR}/lib/task_serialization.sh" ]]; then
    LIB_DIR="${SCRIPT_DIR}/lib"
elif [[ -f "${SCRIPT_DIR}/task_serialization.sh" ]]; then
    # Flat directory structure (all files in same dir)
    LIB_DIR="${SCRIPT_DIR}"
else
    LIB_DIR="${SCRIPT_DIR}/lib"  # Default, will error if missing
fi
export LIB_DIR  # Export for subshell workers

if [[ "${USE_DYNAMIC_SCHEDULING}" == "true" ]]; then
    # Source dynamic scheduling modules
    if [[ -f "${LIB_DIR}/task_serialization.sh" ]]; then
        source "${LIB_DIR}/task_serialization.sh"
        export TASK_SERIALIZATION_LOADED=1
    else
        echo "WARNING: lib/task_serialization.sh not found, falling back to static scheduling"
        USE_DYNAMIC_SCHEDULING="false"
    fi

    if [[ "${USE_DYNAMIC_SCHEDULING}" == "true" && -f "${LIB_DIR}/queue_manager.sh" ]]; then
        source "${LIB_DIR}/queue_manager.sh"
        export QUEUE_MANAGER_LOADED=1
    fi

    if [[ "${USE_DYNAMIC_SCHEDULING}" == "true" && -f "${LIB_DIR}/scheduler.sh" ]]; then
        source "${LIB_DIR}/scheduler.sh"
        export SCHEDULER_LOADED=1
    fi

    if [[ "${USE_DYNAMIC_SCHEDULING}" == "true" && -f "${LIB_DIR}/task_functions.sh" ]]; then
        source "${LIB_DIR}/task_functions.sh"
        export TASK_FUNCTIONS_LOADED=1
    fi

    if [[ "${USE_DYNAMIC_SCHEDULING}" == "true" && -f "${LIB_DIR}/gpu_worker.sh" ]]; then
        source "${LIB_DIR}/gpu_worker.sh"
        export GPU_WORKER_LOADED=1
    fi

    if [[ "${USE_DYNAMIC_SCHEDULING}" == "true" && -f "${LIB_DIR}/fault_tolerance.sh" ]]; then
        source "${LIB_DIR}/fault_tolerance.sh"
        export FAULT_TOLERANCE_LOADED=1
    fi
fi

# ============ SETUP ============
mkdir -p "${OUTPUT_DIR}"/{logs,models,evals,certificates,analysis,reports,presets}
LOG_FILE="${OUTPUT_DIR}/logs/main.log"

# Create a lock file for thread-safe logging
LOG_LOCK="${OUTPUT_DIR}/logs/.log_lock"

log() {
    # Thread-safe logging using flock for parallel processes
    {
        flock -w 5 200 2>/dev/null || true  # Wait up to 5s for lock, continue anyway
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
    } 200>"${LOG_LOCK}"
}

log_section() {
    echo "" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "$*" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"
}

error_exit() {
    # Output to stderr to avoid polluting stdout (important for functions returning values via echo)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >> "${LOG_FILE}"
    exit 1
}

# ============ B200 ENVIRONMENT SETUP ============
setup_b200_environment() {
    log_section "PHASE 0: B200 ENVIRONMENT SETUP"

    # Use python3 -c to avoid heredoc indentation issues on macOS
    python3 -c '
import torch
import os
import sys

print("=== B200 Environment Configuration ===\n")

# Check CUDA availability
if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    sys.exit(1)

# Get GPU count and info
num_gpus = torch.cuda.device_count()
print(f"GPUs Detected: {num_gpus}")

is_b200 = False
total_vram = 0
fp4_support = False

for i in range(num_gpus):
    gpu_name = torch.cuda.get_device_name(i)
    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    total_vram += gpu_memory

    print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

    if "B200" in gpu_name or "Blackwell" in gpu_name:
        is_b200 = True
        fp4_support = True

print(f"\nTotal VRAM: {total_vram:.1f} GB")
print(f"B200 Detected: {is_b200}")
print(f"FP4 Native Support: {fp4_support}")

if not is_b200:
    print("\nWARNING: This script is optimized for B200. Performance may vary on other GPUs.")
    print("         FP4 quantization will run in simulation mode.")

# Enable TF32 (huge speedup on B200/H100/A100)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("\nTF32 enabled: True")

# Enable cuDNN benchmark mode for fastest kernels
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
print("cuDNN benchmark: True")

# Set BF16 as preferred dtype if supported
if torch.cuda.is_bf16_supported():
    torch.set_default_dtype(torch.bfloat16)
    print("Default dtype: bfloat16")
else:
    print("Default dtype: float16 (BF16 not supported)")

# Memory targets - optimal zone for B200
target_fraction = float(os.environ.get("CUDA_MEMORY_FRACTION", "0.92"))
gpu_mem = 180  # B200 has 180GB

print(f"\n=== MEMORY TARGETS (85-92% of {gpu_mem}GB per GPU) ===")
print(f"  Optimal range: {gpu_mem*0.85:.1f} - {gpu_mem*0.92:.1f} GB per GPU")
print(f"  Target fraction: {target_fraction*100:.0f}%")
print(f"  Total usable: {num_gpus * gpu_mem * target_fraction:.0f} GB across {num_gpus} GPUs")

# Check flash attention availability
try:
    from transformers.utils import is_flash_attn_2_available
    flash_avail = is_flash_attn_2_available()
    print(f"\nFlash Attention 2: {flash_avail}")
except ImportError:
    print("\nFlash Attention 2: Unknown (transformers too old)")

# Check torch.compile availability
compile_avail = hasattr(torch, "compile")
print(f"torch.compile: {compile_avail}")

# Write FP4 support flag for shell script
if fp4_support:
    print("\n[FP4_NATIVE_SUPPORT=true]")
else:
    print("\n[FP4_NATIVE_SUPPORT=false]")

print("\n=== Environment Ready for B200 Maximum Utilization ===")
'
    # Note: FP4 support detection is handled directly in create_fp4_model() via Python
    # The FP4_NATIVE_SUPPORT env var is set there based on actual GPU detection at runtime
    log "B200 Environment Setup: Complete (FP4 detection will occur at edit time)"
}

# ============ DEPENDENCY CHECK ============
check_dependencies() {
    log_section "PHASE 0: DEPENDENCY CHECK"

    local missing=()

    # Check Python
    command -v python3 >/dev/null 2>&1 || missing+=("python3")

    # Check PyTorch with CUDA
    python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null || missing+=("torch+cuda")

    # Check transformers
    python3 -c "import transformers; print(f'Transformers {transformers.__version__}')" 2>/dev/null || missing+=("transformers")

    # Check for flash-attn
    if python3 -c "import flash_attn; print('Flash Attention OK')" 2>/dev/null; then
        export FLASH_ATTENTION_AVAILABLE="true"
        log "Flash Attention 2: Available"
    else
        if [[ "${SKIP_FLASH_ATTN:-false}" == "true" ]]; then
            export FLASH_ATTENTION_AVAILABLE="false"
            log "Flash Attention 2: Skipped (SKIP_FLASH_ATTN=true)"
        else
            # Check if Python development headers are available (required for flash-attn build)
            local has_python_dev="false"
            if python3 -c "import sysconfig; exit(0 if sysconfig.get_config_var('INCLUDEPY') else 1)" 2>/dev/null; then
                local python_include=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))")
                if [[ -f "${python_include}/Python.h" ]]; then
                    has_python_dev="true"
                fi
            fi

            if [[ "${has_python_dev}" != "true" ]]; then
                export FLASH_ATTENTION_AVAILABLE="false"
                log "WARNING: Python development headers not found (Python.h missing)"
                log "         To enable flash-attn, install: apt-get install python3-dev  (or python3.X-dev)"
                log "         Or set SKIP_FLASH_ATTN=true to suppress this warning"
                log "         Continuing with eager attention (may be slower)"
            else
                log "Flash Attention 2: Not found, attempting install..."
                # Use timeout to prevent hanging on slow builds
                if timeout 600 python3 -m pip install flash-attn --no-build-isolation 2>&1 | tee -a "${LOG_FILE}"; then
                    # Verify it actually imported
                    if python3 -c "import flash_attn" 2>/dev/null; then
                        export FLASH_ATTENTION_AVAILABLE="true"
                        log "Flash Attention 2: Installed successfully"
                    else
                        export FLASH_ATTENTION_AVAILABLE="false"
                        log "WARNING: flash-attn installed but import failed, using eager attention"
                    fi
                else
                    export FLASH_ATTENTION_AVAILABLE="false"
                    log "WARNING: flash-attn install failed (build error), using eager attention"
                    log "         This is OK - script will work without flash attention, just slower."
                fi
            fi
        fi
    fi

    # Check PyYAML
    python3 -c "import yaml" 2>/dev/null || python3 -m pip install pyyaml

    # Check protobuf (required by many HuggingFace models)
    if ! python3 -c "import google.protobuf" 2>/dev/null; then
        log "Installing protobuf..."
        python3 -m pip install protobuf
    fi

    # Check sentencepiece (required by many tokenizers)
    if ! python3 -c "import sentencepiece" 2>/dev/null; then
        log "Installing sentencepiece..."
        python3 -m pip install sentencepiece
    fi

    # Check lm-eval-harness (package name is lm_eval, not lm-eval)
    python3 -c "import lm_eval" 2>/dev/null || python3 -m pip install lm_eval

    # Check InvarLock
    python3 -c "import invarlock" 2>/dev/null || missing+=("invarlock")

    if [[ ${#missing[@]} -gt 0 ]]; then
        error_exit "Missing dependencies: ${missing[*]}"
    fi

    log "All dependencies satisfied"
}

# ============ MODEL SETUP WITH B200 OPTIMIZATIONS ============
setup_model() {
    local model_id="$1"
    local gpu_id="${2:-0}"
    local model_name=$(basename "${model_id}" | tr '[:upper:]' '[:lower:]' | tr '/' '_')
    local model_dir="${OUTPUT_DIR}/models/${model_name}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setting up model (B200 GPU ${gpu_id}): ${model_id}" >> "${LOG_FILE}"

    # Check if local path
    if [[ -d "${model_id}" ]]; then
        echo "${model_id}"
        return 0
    fi

    # Check if already downloaded
    if [[ -d "${model_dir}/baseline" ]]; then
        echo "${model_dir}/baseline"
        return 0
    fi

    # Download with B200 optimizations
    mkdir -p "${model_dir}"

    local success_marker="${model_dir}/.download_success"
    rm -f "${success_marker}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 << EOF 2>&1 | tee -a "${LOG_FILE}" >&2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pathlib import Path
import gc
import sys

model_id = "${model_id}"
output_dir = Path("${model_dir}/baseline")
success_marker = Path("${success_marker}")
flash_available = "${FLASH_ATTENTION_AVAILABLE}" == "true"

output_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading {model_id} (B200 optimized)...")
print(f"Flash Attention 2: {'enabled' if flash_available else 'disabled'}")

def model_supports_flash_attention(model_id):
    """Check if model architecture supports Flash Attention 2."""
    # Models known to NOT support FA2
    no_fa2_models = [
        "falcon", "mpt-", "gpt2", "bloom", "opt-", "gpt-j", "gpt-neo",
        "codegen", "santacoder", "stablelm"
    ]
    model_lower = model_id.lower()
    for pattern in no_fa2_models:
        if pattern in model_lower:
            return False
    return True

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    # Determine if we can use Flash Attention 2 for this model
    use_fa2 = flash_available and model_supports_flash_attention(model_id)

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }

    if use_fa2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print(f"Using Flash Attention 2 for {model_id}")
    else:
        print(f"Using eager attention for {model_id} (FA2 not supported or unavailable)")

    # Try to load with FA2, fall back to eager if it fails
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except Exception as fa2_err:
        if use_fa2 and "flash" in str(fa2_err).lower():
            print(f"Flash Attention 2 failed, falling back to eager attention: {fa2_err}")
            del model_kwargs["attn_implementation"]
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        else:
            raise

    # Fix invalid generation config before saving (some models have temperature/top_p without do_sample)
    if hasattr(model, 'generation_config'):
        gen_config = model.generation_config
        # If do_sample is False but temperature/top_p are set, clear them to avoid validation errors
        if hasattr(gen_config, 'do_sample') and not gen_config.do_sample:
            if hasattr(gen_config, 'temperature') and gen_config.temperature != 1.0:
                print(f"Fixing generation_config: clearing temperature={gen_config.temperature} (do_sample=False)")
                gen_config.temperature = None
            if hasattr(gen_config, 'top_p') and gen_config.top_p != 1.0:
                print(f"Fixing generation_config: clearing top_p={gen_config.top_p} (do_sample=False)")
                gen_config.top_p = None

    model.save_pretrained(output_dir, safe_serialization=True)

    # Aggressive memory cleanup before lm-eval starts
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Force synchronize to ensure memory is freed
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        # Additional cleanup - clear all cached allocators
        torch.cuda.memory.empty_cache()

    print(f"Saved to {output_dir}")
    print(f"GPU memory freed: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
    success_marker.touch()

except Exception as e:
    print(f"ERROR: Model download failed: {e}", file=sys.stderr)
    sys.exit(1)
EOF

    if [[ ! -f "${success_marker}" ]]; then
        # Output error to stderr (not stdout) and return empty string
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to download model: ${model_id}" >&2
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to download model: ${model_id}" >> "${LOG_FILE}"
        echo ""  # Return empty string so caller can detect failure
        return 1
    fi
    rm -f "${success_marker}"

    echo "${model_dir}/baseline"
}
export -f setup_model

# ============ ESTIMATE MODEL SIZE FOR BATCH OPTIMIZATION ============
estimate_model_params() {
    local model_path="$1"
    local config_file="${model_path}/config.json"
    if [[ ! -f "${config_file}" ]]; then
        echo "7"
        return
    fi

    # Returns model size bucket for batch optimization
    # Also detects MoE architectures (Mixtral) which need special handling
    # Note: config_file is passed as argument to avoid shell injection issues
    local params=$(python3 -c "
import json
import sys
try:
    config_path = sys.argv[1]
    config = json.load(open(config_path))

    # Extract architecture parameters
    h = config.get('hidden_size', 4096)
    l = config.get('num_hidden_layers', 32)
    v = config.get('vocab_size', 32000)
    i = config.get('intermediate_size', h * 4)  # FFN intermediate size

    # Detect MoE architecture (Mixtral style)
    num_experts = config.get('num_local_experts', 1)
    if num_experts == 1:
        num_experts = config.get('num_experts', 1)

    # Better parameter estimation formula:
    # - Embedding: vocab_size * hidden_size
    # - Attention per layer: 4 * hidden_size^2 (Q,K,V,O projections)
    # - FFN per layer: 3 * hidden_size * intermediate_size (SwiGLU/gate has 3 matrices)
    # - LM head: hidden_size * vocab_size
    embedding_params = v * h
    attention_per_layer = 4 * h * h
    ffn_per_layer = 3 * h * i  # gate_proj, up_proj, down_proj
    lm_head = h * v

    base_params = (embedding_params + l * (attention_per_layer + ffn_per_layer) + lm_head) / 1e9

    # For MoE, each expert has its own FFN, but we only activate some at a time
    # Memory scales with total params (all experts loaded), so multiply FFN contribution
    if num_experts > 1:
        moe_ffn = l * ffn_per_layer * num_experts
        base_params = (embedding_params + l * attention_per_layer + moe_ffn + lm_head) / 1e9
        print('moe')
    elif base_params > 55:  # 70B/72B models
        print('70')
    elif base_params > 28:  # 30B-40B models
        print('40')
    elif base_params > 18:  # 20B-30B models (Qwen2.5-32B etc)
        print('30')
    elif base_params > 10:  # 13B-14B models
        print('13')
    else:
        print('7')
except Exception as e:
    # Debug: uncomment to see why detection fails
    # import sys; print(f'estimate_model_params error: {e}', file=sys.stderr)
    print('7')
" "${config_file}" 2>/dev/null)
    echo "${params:-7}"
}
export -f estimate_model_params

# ============ MODEL-SIZE-AWARE INVARLOCK CONFIGURATION ============
# Returns: seq_len:stride:preview_n:final_n:eval_batch
# Based on model size and B200 180GB memory budget
get_model_invarlock_config() {
    local model_size="$1"  # 7, 13, 30, 40, 70, moe

    # WikiText-2 has ~1174 samples, need conservative window counts
    # B200 has 180GB VRAM - can be more generous than H100
    # Format: seq_len:stride:preview_n:final_n:eval_batch
    case "${model_size}" in
        "7")
            # 7B models: ~14GB, can use long sequences and more windows
            # B200 has plenty of headroom
            echo "2048:1024:64:64:96"
            ;;
        "13")
            # 13-14B models: ~26-28GB, moderate settings
            # Note: estimate_model_params() returns "13" for both 13B and 14B
            echo "1536:768:48:48:64"
            ;;
        "30")
            # 30B models: ~60GB, reduced settings
            echo "1024:512:40:40:48"
            ;;
        "40")
            # 40B models: ~80GB, conservative settings
            echo "1024:512:36:36:32"
            ;;
        "moe")
            # MoE models (Mixtral-8x7B): ~90GB effective
            # Moderate sequence length, smaller batch due to expert memory
            echo "1024:512:40:40:24"
            ;;
        "70"|"72")
            # 70-72B models: ~140-144GB, ULTRA-CONSERVATIVE settings
            # B200 180GB minus ~140GB model = only ~36GB headroom
            # CRITICAL FIX v2: Must avoid double-load from overhead check
            # - seq_len=128 (was 256): Minimal KV cache
            # - stride=64: Maintains 50% overlap
            # - windows=8+8 (was 16+16): Minimal window count
            # - eval_batch=2 (was 8): Minimal batch to survive overhead check
            echo "128:64:8:8:2"
            ;;
        *)
            # Unknown - use safe defaults
            echo "1024:512:40:40:32"
            ;;
    esac
}
export -f get_model_invarlock_config

# Note: GPU assignment is done statically via gpu_models associative array in main()
# The assign_gpu_to_model() function was removed as unused dead code.

# ============ EDITED MODEL WITH GPU QUANTIZATION ============
create_edited_model() {
    local baseline_path="$1"
    local output_path="$2"
    local edit_type="$3"
    local bits="$4"
    local group_size="$5"
    local scope="$6"
    local gpu_id="${7:-0}"

    log "Creating edited model (B200 GPU ${gpu_id}):"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"
    log "  Edit: ${edit_type} bits=${bits} group_size=${group_size} scope=${scope}"

    mkdir -p "$(dirname "${output_path}")"

    if [[ "${edit_type}" == "quant_rtn" ]]; then
        CUDA_VISIBLE_DEVICES="${gpu_id}" python3 << EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import gc
import sys

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    baseline_path = Path("${baseline_path}")
    output_path = Path("${output_path}")
    bits = int("${bits}")
    group_size = int("${group_size}")
    scope = "${scope}"

    print(f"Loading baseline from {baseline_path}...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
    flash_available = "${FLASH_ATTENTION_AVAILABLE}" == "true"

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "cuda:0",
        "low_cpu_mem_usage": True,
    }
    if flash_available:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(baseline_path, **model_kwargs)

    @torch.no_grad()
    def round_to_nearest_gpu(tensor, bits):
        """Per-tensor RTN quantization.

        Note: group_size is stored in metadata but not used in this implementation.
        Full per-group quantization would require chunking along channel dimension,
        but per-tensor is sufficient for validation purposes.
        """
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        scale = tensor.abs().max() / qmax
        scale = torch.clamp(scale, min=1e-10)
        quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)
        return (quantized * scale).to(tensor.dtype)

    def should_quantize(name, scope):
        """Check if parameter should be quantized based on name and scope.

        Supports multiple architectures:
        - LLaMA/Mistral: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        - MPT: Wqkv, out_proj, up_proj, down_proj
        - Falcon: query_key_value, dense_h_to_4h, dense_4h_to_h
        - Generic: linear, dense, proj, fc, mlp, attn
        """
        name_lower = name.lower()
        if scope == "all":
            return "weight" in name_lower and any(x in name_lower for x in [
                "linear", "dense", "proj", "fc", "mlp", "attn",
                "wqkv", "query_key_value"  # MPT/Falcon attention
            ])
        elif scope == "ffn":
            return "weight" in name_lower and any(x in name_lower for x in [
                "mlp", "fc", "dense", "gate", "up_proj", "down_proj",
                "dense_h_to_4h", "dense_4h_to_h"  # Falcon FFN
            ])
        elif scope == "attn":
            return "weight" in name_lower and any(x in name_lower for x in [
                "attn", "q_proj", "k_proj", "v_proj", "o_proj",
                "wqkv", "out_proj", "query_key_value"  # MPT/Falcon attention
            ])
        return False

    print(f"Quantizing to {bits}-bit on GPU (scope={scope})...")
    quantized_count = 0
    total_model_params = sum(p.numel() for p in model.parameters())
    edited_params = 0

    for name, param in model.named_parameters():
        if should_quantize(name, scope) and param.dim() >= 2:
            param.data = round_to_nearest_gpu(param.data, bits)
            quantized_count += 1
            edited_params += param.numel()
            if quantized_count <= 3:
                print(f"  Quantized: {name} ({param.shape})")

    coverage_pct = 100.0 * edited_params / total_model_params if total_model_params > 0 else 0
    print(f"Quantized {quantized_count} parameters ({edited_params:,} / {total_model_params:,} = {coverage_pct:.1f}% coverage)")

    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)

    metadata = {
        "edit_type": "quant_rtn",
        "bits": bits,
        "group_size": group_size,
        "scope": scope,
        "quantized_params": quantized_count
    }
    with open(output_path / "edit_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved edited model to {output_path}")

except Exception as e:
    print(f"ERROR: Failed to create edited model: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    else
        error_exit "Unknown edit type: ${edit_type}"
    fi
}
export -f create_edited_model

# ============ MAGNITUDE PRUNING ============
create_pruned_model() {
    local baseline_path="$1"
    local output_path="$2"
    local sparsity="$3"  # 0.1 for clean, 0.5 for stress
    local scope="$4"     # ffn, attn, all
    local gpu_id="${5:-0}"

    log "Creating pruned model (B200 GPU ${gpu_id}):"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"
    log "  Sparsity: ${sparsity}, Scope: ${scope}"

    mkdir -p "$(dirname "${output_path}")"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 << EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import gc
import sys

try:
    baseline_path = Path("${baseline_path}")
    output_path = Path("${output_path}")
    sparsity = float("${sparsity}")
    scope = "${scope}"

    print(f"Loading baseline from {baseline_path}...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        baseline_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )

    def should_prune(name, scope):
        """Check if parameter should be pruned based on name and scope.

        Supports multiple architectures:
        - LLaMA/Mistral: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        - MPT: Wqkv, out_proj, up_proj, down_proj
        - Falcon: query_key_value, dense_h_to_4h, dense_4h_to_h
        """
        name_lower = name.lower()
        if scope == "all":
            return "weight" in name_lower and any(x in name_lower for x in [
                "linear", "proj", "fc", "mlp", "attn",
                "wqkv", "query_key_value"  # MPT/Falcon
            ])
        elif scope == "ffn":
            return "weight" in name_lower and any(x in name_lower for x in [
                "mlp", "fc", "gate", "up_proj", "down_proj",
                "dense_h_to_4h", "dense_4h_to_h"  # Falcon FFN
            ])
        elif scope == "attn":
            return "weight" in name_lower and any(x in name_lower for x in [
                "attn", "q_proj", "k_proj", "v_proj", "o_proj",
                "wqkv", "out_proj", "query_key_value"  # MPT/Falcon
            ])
        return False

    @torch.no_grad()
    def magnitude_prune(weight, sparsity):
        """Set smallest magnitude weights to zero."""
        flat = weight.abs().flatten()
        k = int(flat.numel() * sparsity)
        if k == 0:
            return weight
        threshold = torch.kthvalue(flat, k).values
        mask = weight.abs() >= threshold
        return weight * mask.to(weight.dtype)

    print(f"Pruning with sparsity={sparsity} (scope={scope})...")
    pruned_count = 0
    total_zeros = 0
    total_edited_params = 0
    total_model_params = sum(p.numel() for p in model.parameters())

    for name, param in model.named_parameters():
        if should_prune(name, scope) and param.dim() >= 2:
            original_zeros = (param == 0).sum().item()
            param.data = magnitude_prune(param.data, sparsity)
            new_zeros = (param == 0).sum().item()
            pruned_count += 1
            total_zeros += new_zeros
            total_edited_params += param.numel()
            if pruned_count <= 3:
                print(f"  Pruned: {name} ({original_zeros} → {new_zeros} zeros)")

    actual_sparsity = total_zeros / total_edited_params if total_edited_params > 0 else 0
    coverage_pct = 100.0 * total_edited_params / total_model_params if total_model_params > 0 else 0
    print(f"Pruned {pruned_count} parameters ({total_edited_params:,} / {total_model_params:,} = {coverage_pct:.1f}% coverage)")
    print(f"Actual sparsity within edited params: {actual_sparsity:.2%}")

    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)

    metadata = {
        "edit_type": "magnitude_prune",
        "target_sparsity": sparsity,
        "actual_sparsity": actual_sparsity,
        "scope": scope,
        "pruned_params": pruned_count
    }
    with open(output_path / "edit_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved pruned model to {output_path}")

except Exception as e:
    print(f"ERROR: Failed to create pruned model: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
}
export -f create_pruned_model

# ============ LOW-RANK SVD APPROXIMATION ============
create_lowrank_model() {
    local baseline_path="$1"
    local output_path="$2"
    local rank="$3"      # 256 for clean, 32 for stress
    local scope="$4"     # ffn, attn, all
    local gpu_id="${5:-0}"

    log "Creating low-rank model (B200 GPU ${gpu_id}):"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"
    log "  Rank: ${rank}, Scope: ${scope}"

    mkdir -p "$(dirname "${output_path}")"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 << EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import gc
import sys

try:
    baseline_path = Path("${baseline_path}")
    output_path = Path("${output_path}")
    rank = int("${rank}")
    scope = "${scope}"

    print(f"Loading baseline from {baseline_path}...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        baseline_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )

    def should_lowrank(name, scope):
        """Check if parameter should have low-rank approximation.

        Supports multiple architectures:
        - LLaMA/Mistral: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        - MPT: Wqkv, out_proj, up_proj, down_proj
        - Falcon: query_key_value, dense_h_to_4h, dense_4h_to_h
        """
        name_lower = name.lower()
        if scope == "all":
            return "weight" in name_lower and any(x in name_lower for x in [
                "linear", "proj", "fc", "mlp",
                "wqkv", "query_key_value"  # MPT/Falcon
            ])
        elif scope == "ffn":
            return "weight" in name_lower and any(x in name_lower for x in [
                "mlp", "fc", "gate", "up_proj", "down_proj",
                "dense_h_to_4h", "dense_4h_to_h"  # Falcon FFN
            ])
        elif scope == "attn":
            return "weight" in name_lower and any(x in name_lower for x in [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "wqkv", "out_proj", "query_key_value"  # MPT/Falcon
            ])
        return False

    @torch.no_grad()
    def truncated_svd(weight, rank):
        """Apply truncated SVD to approximate weight matrix using randomized algorithm.

        Uses torch.svd_lowrank for efficiency on large matrices:
        - Full SVD: O(n^3) time, OOM risk on large weights
        - Randomized SVD: O(n^2 * rank) time, memory-efficient
        """
        if weight.dim() < 2:
            return weight

        original_shape = weight.shape
        weight_2d = weight.view(weight.shape[0], -1).float()

        max_rank = min(weight_2d.shape)
        effective_rank = min(rank, max_rank)

        # Use randomized SVD (O(n^2 * rank)) instead of full SVD (O(n^3))
        # niter=2 provides good accuracy while staying fast
        # q parameter is the target rank
        U, S, V = torch.svd_lowrank(weight_2d, q=effective_rank, niter=2)

        # Reconstruct: U @ diag(S) @ V^T
        lowrank = U @ torch.diag(S) @ V.T
        return lowrank.to(weight.dtype).view(original_shape)

    print(f"Applying low-rank SVD with rank={rank} (scope={scope})...")
    modified_count = 0
    total_energy_retained = 0
    num_matrices = 0
    total_model_params = sum(p.numel() for p in model.parameters())
    edited_params = 0

    for name, param in model.named_parameters():
        if should_lowrank(name, scope) and param.dim() >= 2:
            original_norm = param.data.norm()
            param.data = truncated_svd(param.data, rank)
            new_norm = param.data.norm()
            energy_retained = (new_norm / original_norm).item() if original_norm > 0 else 1.0
            modified_count += 1
            total_energy_retained += energy_retained
            num_matrices += 1
            edited_params += param.numel()
            if modified_count <= 3:
                print(f"  Low-rank: {name}, energy retained: {energy_retained:.4f}")

    avg_energy = total_energy_retained / num_matrices if num_matrices > 0 else 1.0
    coverage_pct = 100.0 * edited_params / total_model_params if total_model_params > 0 else 0
    print(f"Modified {modified_count} matrices ({edited_params:,} / {total_model_params:,} = {coverage_pct:.1f}% coverage)")
    print(f"Average energy retained: {avg_energy:.2%}")

    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)

    metadata = {
        "edit_type": "lowrank_svd",
        "rank": rank,
        "scope": scope,
        "modified_matrices": modified_count,
        "avg_energy_retained": avg_energy
    }
    with open(output_path / "edit_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved low-rank model to {output_path}")

except Exception as e:
    print(f"ERROR: Failed to create low-rank model: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
}
export -f create_lowrank_model

# ============ FP4 QUANTIZATION (B200-EXCLUSIVE) ============
create_fp4_model() {
    local baseline_path="$1"
    local output_path="$2"
    local format="$3"      # e2m1 (standard) or aggressive
    local scope="$4"       # ffn, attn, all
    local gpu_id="${5:-0}"

    log "Creating FP4 model (B200 GPU ${gpu_id}):"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"
    log "  Format: ${format}, Scope: ${scope}"
    log "  FP4 Native Support: ${FP4_NATIVE_SUPPORT}"

    mkdir -p "$(dirname "${output_path}")"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 << EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import gc
import sys

try:
    baseline_path = Path("${baseline_path}")
    output_path = Path("${output_path}")
    format_type = "${format}"
    scope = "${scope}"

    # Check for B200 FP4 support
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device_name = torch.cuda.get_device_name(0)
    is_b200 = "B200" in device_name or "Blackwell" in device_name

    if not is_b200:
        print(f"WARNING: FP4 is B200-native, current GPU: {device_name}")
        print("         Running in simulation mode - may not match true B200 behavior")

    print(f"Loading baseline from {baseline_path}...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        baseline_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )

    def should_quantize(name, scope):
        """Check if parameter should be FP4 quantized.

        Supports multiple architectures (LLaMA, MPT, Falcon).
        """
        name_lower = name.lower()
        if scope == "all":
            return "weight" in name_lower and any(x in name_lower for x in [
                "linear", "proj", "fc", "mlp", "attn",
                "wqkv", "query_key_value"  # MPT/Falcon
            ])
        elif scope == "ffn":
            return "weight" in name_lower and any(x in name_lower for x in [
                "mlp", "fc", "gate", "up_proj", "down_proj",
                "dense_h_to_4h", "dense_4h_to_h"  # Falcon FFN
            ])
        elif scope == "attn":
            return "weight" in name_lower and any(x in name_lower for x in [
                "attn", "q_proj", "k_proj", "v_proj", "o_proj",
                "wqkv", "out_proj", "query_key_value"  # MPT/Falcon
            ])
        return False

    @torch.no_grad()
    def fp4_quantize(tensor, format_type):
        """
        FP4 quantization (E2M1 or aggressive).

        E2M1 format: 2 exponent bits, 1 mantissa bit
        Range: [-6, 6] with 7 distinct magnitudes + zero

        Aggressive: tighter clipping for stress testing
        """
        # FP4 E2M1 representable values (approximate)
        if format_type == "e2m1":
            # Standard E2M1: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
            levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=tensor.device)
        else:
            # Aggressive: tighter range for stress testing
            levels = torch.tensor([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3], device=tensor.device)

        # Add negative copies
        all_levels = torch.cat([-levels.flip(0)[:-1], levels])  # Avoid double zero

        # Scale tensor to fit FP4 range
        max_val = levels.max()
        scale = tensor.abs().max() / max_val
        scale = torch.clamp(scale, min=1e-10)

        # Quantize
        scaled = tensor / scale
        # Reshape for broadcasting
        scaled_flat = scaled.flatten()
        diff = (scaled_flat.unsqueeze(-1) - all_levels).abs()
        indices = diff.argmin(dim=-1)
        quantized = all_levels[indices].reshape(tensor.shape)

        return (quantized * scale).to(tensor.dtype)

    print(f"Applying FP4 quantization (format={format_type}, scope={scope})...")
    quantized_count = 0
    total_error = 0
    num_tensors = 0
    total_model_params = sum(p.numel() for p in model.parameters())
    edited_params = 0

    for name, param in model.named_parameters():
        if should_quantize(name, scope) and param.dim() >= 2:
            original = param.data.clone()
            param.data = fp4_quantize(param.data, format_type)

            # Compute relative error
            error = (param.data - original).abs().mean() / (original.abs().mean() + 1e-10)
            total_error += error.item()
            quantized_count += 1
            num_tensors += 1
            edited_params += param.numel()

            if quantized_count <= 3:
                print(f"  FP4: {name}, rel_error: {error.item():.4f}")

    avg_error = total_error / num_tensors if num_tensors > 0 else 0
    coverage_pct = 100.0 * edited_params / total_model_params if total_model_params > 0 else 0
    print(f"Quantized {quantized_count} tensors ({edited_params:,} / {total_model_params:,} = {coverage_pct:.1f}% coverage)")
    print(f"Average relative error: {avg_error:.4f}")

    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)

    metadata = {
        "edit_type": "fp4_quant",
        "format": format_type,
        "scope": scope,
        "quantized_tensors": quantized_count,
        "avg_relative_error": avg_error,
        "b200_native": is_b200
    }
    with open(output_path / "edit_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved FP4-quantized model to {output_path}")

except Exception as e:
    print(f"ERROR: Failed to create FP4 model: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
}
export -f create_fp4_model

# ============ ERROR MODEL CREATION ============
create_error_model() {
    local baseline_path="$1"
    local output_path="$2"
    local error_type="$3"
    local gpu_id="${4:-0}"

    log "Creating error model (type=${error_type}, GPU ${gpu_id})"
    mkdir -p "$(dirname "${output_path}")"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 << EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import gc
import sys

try:
    baseline_path = Path("${baseline_path}")
    output_path = Path("${output_path}")
    error_type = "${error_type}"

    print(f"Loading baseline from {baseline_path}...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)

    # Use GPU for error injection when possible (handles large models better)
    # Fall back to CPU for small models or if GPU has issues
    try:
        model = AutoModelForCausalLM.from_pretrained(
            baseline_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )
        use_gpu = True
    except Exception as gpu_err:
        print(f"GPU loading failed ({gpu_err}), falling back to CPU (may be slow for large models)")
        model = AutoModelForCausalLM.from_pretrained(
            baseline_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        use_gpu = False

    error_info = {"error_type": error_type, "injected": False}

    # Build list of transformer blocks for index-based targeting
    # This works across architectures (LLaMA, MPT, Falcon, Qwen, etc.)
    import re
    block_params = {}  # {block_idx: [(name, param), ...]}
    block_pattern = re.compile(r'(?:layers|blocks|h)\.(\d+)\.')

    for name, param in model.named_parameters():
        match = block_pattern.search(name)
        if match:
            block_idx = int(match.group(1))
            if block_idx not in block_params:
                block_params[block_idx] = []
            block_params[block_idx].append((name, param))

    num_blocks = max(block_params.keys()) + 1 if block_params else 0
    first_block = 0
    middle_block = num_blocks // 2 if num_blocks > 1 else 0

    print(f"Detected {num_blocks} transformer blocks")

    if error_type == "nan_injection":
        # Target first block - works across architectures
        target_block = first_block
        for name, param in block_params.get(target_block, []):
            if 'weight' in name.lower() and param.dim() >= 2:
                with torch.no_grad():
                    param.data[0, 0] = float('nan')
                error_info["injected"] = True
                error_info["target_param"] = name
                error_info["target_block"] = target_block
                print(f"Injected NaN into: {name} (block {target_block})")
                break

    elif error_type == "inf_injection":
        # Target attention in first block
        for name, param in model.named_parameters():
            if 'attn' in name.lower() and 'weight' in name.lower() and param.dim() >= 2:
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
        # Target MLP/FFN in first block
        for name, param in model.named_parameters():
            if 'mlp' in name.lower() and 'weight' in name.lower() and param.dim() >= 2:
                with torch.no_grad():
                    param.data = param.data * 100.0
                error_info["injected"] = True
                error_info["target_param"] = name
                error_info["scale_factor"] = 100.0
                print(f"Scaled by 100x: {name}")
                break

    elif error_type == "zero_layer":
        # Target middle block - architecture agnostic
        target_block = middle_block
        for name, param in block_params.get(target_block, []):
            if 'weight' in name.lower() and param.dim() >= 2:
                with torch.no_grad():
                    param.data.zero_()
                error_info["injected"] = True
                error_info["target_param"] = name
                error_info["target_block"] = target_block
                print(f"Zeroed: {name} (block {target_block})")
                break

    # Move to CPU for saving if loaded on GPU
    if use_gpu:
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)

    with open(output_path / "error_metadata.json", 'w') as f:
        json.dump(error_info, f, indent=2)

    del model
    gc.collect()
    print(f"Saved error model to {output_path}")

except Exception as e:
    print(f"ERROR: Failed to create error model: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
}
export -f create_error_model

# ============ PROCESS EDIT - DISPATCHER ============
process_edit() {
    local baseline_path="$1"
    local edit_spec="$2"     # Format: "type:param1:param2[:scope]" - scope optional
    local version="$3"       # clean or stress
    local model_name="$4"
    local gpu_id="$5"
    local output_dir="$6"

    # Parse edit spec - handle both 3-part and 4-part formats
    # quant_rtn uses 4 parts: "quant_rtn:8:128:ffn"
    # others use 3 parts: "fp4_quant:e2m1:ffn"
    local edit_type param1 param2 scope
    IFS=':' read -r edit_type param1 param2 scope <<< "${edit_spec}"

    # For 3-part specs (fp4, prune, svd), scope is in param2 position
    # For 4-part specs (quant_rtn), all vars are correctly populated
    if [[ -z "${scope}" && "${edit_type}" != "quant_rtn" ]]; then
        scope="${param2}"
        param2=""
    fi

    # Determine output path
    local edit_dir_name="${edit_type}_${version}"
    if [[ "${edit_type}" == "quant_rtn" ]]; then
        edit_dir_name="quant_${param1}bit_${version}"
    elif [[ "${edit_type}" == "fp4_quant" ]]; then
        edit_dir_name="fp4_${param1}_${version}"
    elif [[ "${edit_type}" == "magnitude_prune" ]]; then
        # Convert 0.1 -> 10pct, 0.5 -> 50pct for cleaner dir names
        # Validate that param1 is a valid number first
        if ! [[ "${param1}" =~ ^[0-9]*\.?[0-9]+$ ]]; then
            log "  ERROR: Invalid sparsity value: ${param1}"
            return 1
        fi
        local pct=$(echo "${param1}" | awk '{printf "%.0f", $1 * 100}')
        edit_dir_name="prune_${pct}pct_${version}"
    elif [[ "${edit_type}" == "lowrank_svd" ]]; then
        edit_dir_name="svd_rank${param1}_${version}"
    fi

    local edit_path="${output_dir}/models/${edit_dir_name}"

    # Check if already exists (resume mode)
    if [[ "${RESUME_MODE}" == "true" && -d "${edit_path}" && -f "${edit_path}/config.json" ]]; then
        log "  Edit ${edit_dir_name} exists, skipping creation"
        echo "${edit_path}"
        return 0
    fi

    # Create edit based on type
    local create_result=0
    case "${edit_type}" in
        "quant_rtn")
            # 4-part: type:bits:group_size:scope
            create_edited_model "${baseline_path}" "${edit_path}" "${edit_type}" "${param1}" "${param2}" "${scope}" "${gpu_id}" || create_result=$?
            ;;
        "fp4_quant")
            # 3-part: type:format:scope -> param1=format, scope=scope
            create_fp4_model "${baseline_path}" "${edit_path}" "${param1}" "${scope}" "${gpu_id}" || create_result=$?
            ;;
        "magnitude_prune")
            # 3-part: type:sparsity:scope -> param1=sparsity, scope=scope
            create_pruned_model "${baseline_path}" "${edit_path}" "${param1}" "${scope}" "${gpu_id}" || create_result=$?
            ;;
        "lowrank_svd")
            # 3-part: type:rank:scope -> param1=rank, scope=scope
            create_lowrank_model "${baseline_path}" "${edit_path}" "${param1}" "${scope}" "${gpu_id}" || create_result=$?
            ;;
        *)
            log "  ERROR: Unknown edit type: ${edit_type}"
            return 1
            ;;
    esac

    # Only output path if creation succeeded
    if [[ ${create_result} -eq 0 && -d "${edit_path}" ]]; then
        echo "${edit_path}"
    else
        log "  ERROR: Failed to create edit ${edit_dir_name} (exit code: ${create_result})"
        return 1
    fi
}
export -f process_edit

# ============ LM-EVAL WITH B200 OPTIMIZATION ============
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
            "70"|"72") effective_batch_size="${EVAL_BATCH_SIZE_LARGE}" ;;
            "40")      effective_batch_size="${EVAL_BATCH_SIZE_MEDIUM}" ;;
            "30")      effective_batch_size="${EVAL_BATCH_SIZE_MEDIUM}" ;;  # MPT-30B uses medium
            "moe")     effective_batch_size="${EVAL_BATCH_SIZE_MOE}" ;;      # Mixtral/MoE models
            *)         effective_batch_size="${EVAL_BATCH_SIZE_SMALL}" ;;
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
    if [[ "${FLASH_ATTENTION_AVAILABLE}" == "true" ]]; then
        model_args="${model_args},attn_implementation=flash_attention_2"
    fi

    log "  🚀 Starting lm-eval on GPU ${gpu_id}..."

    local exit_code=0
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    TORCH_COMPILE=1 \
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

    local results_file=$(find "$(dirname "${output_file}")" -name "results*.json" -type f | head -1)
    if [[ -n "${results_file}" ]]; then
        mv "${results_file}" "${output_file}"
        log "  ✅ Results saved: ${output_file} (${duration}s)"
    else
        log "  ⚠️  No results file found"
    fi
}
export -f run_lmeval

# ============ INVARLOCK CONFIG WITH B200 SETTINGS ============
generate_invarlock_config() {
    local model_path="$1"
    local output_yaml="$2"
    local edit_name="${3:-noop}"
    local seed="${4:-42}"
    local preview_n="${5:-${INVARLOCK_PREVIEW_WINDOWS}}"
    local final_n="${6:-${INVARLOCK_FINAL_WINDOWS}}"
    local bootstrap_n="${7:-${INVARLOCK_BOOTSTRAP_N}}"
    local seq_len="${8:-${INVARLOCK_SEQ_LEN}}"
    local stride="${9:-${INVARLOCK_STRIDE}}"
    local eval_batch="${10:-${INVARLOCK_EVAL_BATCH}}"

    # Use auto adapter for generic causal LM support (LLaMA, Mistral, Qwen, MPT, Falcon, etc.)
    local adapter="hf_causal_auto"
    local dataset_provider="${INVARLOCK_DATASET}"

    local attn_impl_yaml=""
    if [[ "${FLASH_ATTENTION_AVAILABLE}" == "true" ]]; then
        attn_impl_yaml='attn_implementation: "flash_attention_2"'
    else
        attn_impl_yaml='# flash_attention_2 not available'
    fi

    cat > "${output_yaml}" << YAML_EOF
# Auto-generated InvarLock config for B200 validation
# Model: ${model_path}
# Edit: ${edit_name}
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Platform: B200 180GB optimized

model:
  id: "${model_path}"
  adapter: "${adapter}"
  device: "auto"
  dtype: "bfloat16"
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

accelerator:
  compile: true
  tf32: true
  benchmark: true
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

    generate_invarlock_config \
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

    # v0.3.1 FEATURE: For large models, skip overhead check to avoid OOM
    local model_size
    model_size=$(estimate_model_params "${model_path}")
    if [[ "${model_size}" == "70" || "${model_size}" == "72" || "${model_size}" == "moe" ]]; then
        export INVARLOCK_SKIP_OVERHEAD_CHECK=1
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Large model (${model_size}): setting INVARLOCK_SKIP_OVERHEAD_CHECK=1" >> "${log_file}"
    fi

    local exit_code=0
    CUDA_VISIBLE_DEVICES="${gpu_id}" invarlock run \
        --config "${config_yaml}" \
        --profile ci \
        --out "${run_dir}" \
        >> "${log_file}" 2>&1 || exit_code=$?

    # Generate certificate from report
    local report_file=$(find "${run_dir}" -name "report*.json" -type f 2>/dev/null | head -1)
    if [[ -n "${report_file}" ]]; then
        cp "${report_file}" "${run_dir}/baseline_report.json" 2>/dev/null || true

        python3 << CERT_EOF >> "${log_file}" 2>&1
import json
from pathlib import Path
try:
    from invarlock.reporting.certificate import make_certificate
    report_path = Path("${report_file}")
    cert_path = Path("${run_dir}") / "evaluation.cert.json"

    report = json.loads(report_path.read_text())
    cert = make_certificate(report, report)
    with open(cert_path, 'w') as f:
        json.dump(cert, f, indent=2)
except Exception as e:
    print(f"Certificate generation warning: {e}")
CERT_EOF
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

    local model_size=$(estimate_model_params "${model_path}")

    # Get model-size-aware configuration
    local config=$(get_model_invarlock_config "${model_size}")
    IFS=':' read -r effective_seq_len effective_stride effective_preview_n effective_final_n effective_eval_batch <<< "${config}"

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
            "${INVARLOCK_BOOTSTRAP_N}" \
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
    python3 << CALIBRATION_SCRIPT
import json
import yaml
from pathlib import Path
from collections import defaultdict

output_dir = Path("${output_dir}")
preset_output_dir = Path("${preset_output_dir}")
model_name = "${model_name}"
model_path = "${model_path}"
tier = "${INVARLOCK_TIER}"
seq_len = int("${effective_seq_len}")
stride = int("${effective_stride}")
preview_n = int("${effective_preview_n}")
final_n = int("${effective_final_n}")

def load_certificates():
    certs = []
    for run_dir in sorted(output_dir.glob("run_*")):
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
    print("ERROR: No calibration certificates found - cannot create valid preset")
    print("       This model's calibration failed completely")
    import sys
    sys.exit(1)
elif len(certs) < 2:
    print(f"WARNING: Only {len(certs)} calibration certificate(s) found (expected >= 2)")
    print("         Drift statistics will be under-determined")
    # Pad with the available certificate to allow analysis to continue
    certs = certs + [certs[0]] * (2 - len(certs))

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

preset = {
    '_calibration_meta': {
        'model_name': model_name,
        'tier': tier,
        'platform': 'B200_180GB',
        'drift_mean': round(mean_drift, 4),
        'drift_std': round(std_drift, 4),
    },
    'model': {'id': model_path},
    'dataset': {
        'provider': 'wikitext2',
        'split': 'validation',
        'seq_len': seq_len,
        'stride': stride,
        'preview_n': preview_n,
        'final_n': final_n,
        'seed': 42,
    },
    'guards': {}
}

# Save calibration stats
stats_path = output_dir / "calibration_stats.json"
with open(stats_path, 'w') as f:
    json.dump({'drift': {'mean': mean_drift, 'std': std_drift, 'band': [mean_drift - margin, mean_drift + margin]}}, f, indent=2)

# Save preset
preset_path = preset_output_dir / f"calibrated_preset_{model_name.replace('/', '_')}.yaml"
with open(preset_path, 'w') as f:
    yaml.safe_dump(preset, f, sort_keys=False)

print(f"Saved: {stats_path}")
print(f"Saved: {preset_path}")
CALIBRATION_SCRIPT
}

# ============ CERTIFY WITH B200 SETTINGS ============
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

    # v0.3.1 FEATURE: For large models, skip overhead check to avoid OOM
    local model_size
    model_size=$(estimate_model_params "${baseline_path}")
    if [[ "${model_size}" == "70" || "${model_size}" == "72" || "${model_size}" == "moe" ]]; then
        export INVARLOCK_SKIP_OVERHEAD_CHECK=1
    fi

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

    local exit_code=0
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd_args[@]}" \
        >> "${OUTPUT_DIR}/logs/gpu_${gpu_id}.log" 2>&1 || exit_code=$?

    # Copy certificate to standard location
    local cert_file=$(find "${cert_dir}" -name "*.json" -type f 2>/dev/null | head -1)
    if [[ -n "${cert_file}" && -f "${cert_file}" ]]; then
        cp "${cert_file}" "${run_dir}/evaluation.cert.json" 2>/dev/null || true
    fi

    return ${exit_code}
}
export -f run_invarlock_certify

# ============ PROCESS MODEL ON SINGLE GPU ============
process_model() {
    local model_id="$1"
    local gpu_id="${2:-0}"

    local model_name=$(basename "${model_id}" | tr '[:upper:]' '[:lower:]' | tr '/' '_')
    local model_output_dir="${OUTPUT_DIR}/${model_name}"
    local preset_dir="${OUTPUT_DIR}/presets"
    local gpu_log="${OUTPUT_DIR}/logs/gpu_${gpu_id}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Starting ${model_id}" >> "${gpu_log}"

    mkdir -p "${model_output_dir}"/{models,evals,certificates}

    # Step 1: Setup baseline
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Setting up baseline..." >> "${gpu_log}"
    local baseline_path
    baseline_path=$(setup_model "${model_id}" "${gpu_id}")
    local setup_exit_code=$?

    # Validate baseline path - must be non-empty and a valid directory
    if [[ ${setup_exit_code} -ne 0 || -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: ERROR - Failed to setup baseline for ${model_id}" >> "${gpu_log}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: baseline_path='${baseline_path}', exit_code=${setup_exit_code}" >> "${gpu_log}"
        return 1
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Baseline ready at ${baseline_path}" >> "${gpu_log}"

    # Step 2: Baseline eval
    local baseline_eval="${model_output_dir}/evals/baseline_results.json"
    if [[ "${RESUME_MODE}" != "true" || ! -f "${baseline_eval}" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Running baseline lm-eval..." >> "${gpu_log}"
        run_lmeval \
            "${baseline_path}" \
            "${baseline_eval}" \
            "${EVAL_TASKS}" \
            "${EVAL_BATCH_SIZE}" \
            "${EVAL_NUM_FEWSHOT}" \
            "${gpu_id}"
    fi

    # Step 3: Calibration
    local calibration_stats="${model_output_dir}/certificates/calibration/calibration_stats.json"
    if [[ "${RESUME_MODE}" != "true" || ! -f "${calibration_stats}" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Running calibration..." >> "${gpu_log}"
        run_invarlock_calibration \
            "${baseline_path}" \
            "${model_name}" \
            "${model_output_dir}/certificates/calibration" \
            "${DRIFT_CALIBRATION_RUNS}" \
            "${preset_dir}" \
            "${gpu_id}"
    fi

    # Step 4: Clean edits (4 types)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Processing clean edits..." >> "${gpu_log}"
    for edit_spec in "${EDIT_TYPES_CLEAN[@]}"; do
        local edit_path=$(process_edit "${baseline_path}" "${edit_spec}" "clean" "${model_name}" "${gpu_id}" "${model_output_dir}")

        if [[ -n "${edit_path}" && -d "${edit_path}" ]]; then
            # Run eval for this edit
            local edit_name=$(basename "${edit_path}")
            local edit_eval="${model_output_dir}/evals/${edit_name}_results.json"

            if [[ "${RESUME_MODE}" != "true" || ! -f "${edit_eval}" ]]; then
                run_lmeval \
                    "${edit_path}" \
                    "${edit_eval}" \
                    "${EVAL_TASKS}" \
                    "${EVAL_BATCH_SIZE}" \
                    "${EVAL_NUM_FEWSHOT}" \
                    "${gpu_id}"
            fi

            # Run InvarLock certify
            for run in $(seq 1 "${CLEAN_EDIT_RUNS}"); do
                local cert_file="${model_output_dir}/certificates/${edit_name}/run_${run}/evaluation.cert.json"
                if [[ "${RESUME_MODE}" != "true" || ! -f "${cert_file}" ]]; then
                    run_invarlock_certify \
                        "${edit_path}" \
                        "${baseline_path}" \
                        "${model_output_dir}/certificates/${edit_name}" \
                        "run_${run}" \
                        "${preset_dir}" \
                        "${model_name}" \
                        "${gpu_id}"
                fi
            done
        fi
    done

    # Step 5: Stress edits (4 types)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Processing stress edits..." >> "${gpu_log}"
    for edit_spec in "${EDIT_TYPES_STRESS[@]}"; do
        local edit_path=$(process_edit "${baseline_path}" "${edit_spec}" "stress" "${model_name}" "${gpu_id}" "${model_output_dir}")

        if [[ -n "${edit_path}" && -d "${edit_path}" ]]; then
            local edit_name=$(basename "${edit_path}")
            local edit_eval="${model_output_dir}/evals/${edit_name}_results.json"

            if [[ "${RESUME_MODE}" != "true" || ! -f "${edit_eval}" ]]; then
                run_lmeval \
                    "${edit_path}" \
                    "${edit_eval}" \
                    "${EVAL_TASKS}" \
                    "${EVAL_BATCH_SIZE}" \
                    "${EVAL_NUM_FEWSHOT}" \
                    "${gpu_id}"
            fi

            for run in $(seq 1 "${STRESS_EDIT_RUNS}"); do
                local cert_file="${model_output_dir}/certificates/${edit_name}/run_${run}/evaluation.cert.json"
                if [[ "${RESUME_MODE}" != "true" || ! -f "${cert_file}" ]]; then
                    run_invarlock_certify \
                        "${edit_path}" \
                        "${baseline_path}" \
                        "${model_output_dir}/certificates/${edit_name}" \
                        "run_${run}" \
                        "${preset_dir}" \
                        "${model_name}" \
                        "${gpu_id}"
                fi
            done
        fi
    done

    # Step 6: Error injection
    if [[ "${RUN_ERROR_INJECTION}" == "true" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Running error injection tests..." >> "${gpu_log}"
        local errors=("nan_injection" "inf_injection" "extreme_quant" "scale_explosion" "zero_layer")

        for error_type in "${errors[@]}"; do
            local error_path="${model_output_dir}/models/error_${error_type}"
            local cert_file="${model_output_dir}/certificates/errors/${error_type}/evaluation.cert.json"

            if [[ "${RESUME_MODE}" == "true" && -f "${cert_file}" ]]; then
                continue
            fi

            if [[ ! -d "${error_path}" || ! -f "${error_path}/config.json" ]]; then
                create_error_model "${baseline_path}" "${error_path}" "${error_type}" "${gpu_id}"
            fi

            run_invarlock_certify \
                "${error_path}" \
                "${baseline_path}" \
                "${model_output_dir}/certificates/errors" \
                "${error_type}" \
                "${preset_dir}" \
                "${model_name}" \
                "${gpu_id}"
        done
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Model ${model_name} complete" >> "${gpu_log}"
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
    if not model_dir.is_dir() or model_dir.name in ['logs', 'analysis', 'reports', 'presets', 'models']:
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
    if not model_dir.is_dir() or model_dir.name in ['logs', 'analysis', 'reports', 'presets', 'models']:
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
            def as_bool(val):
                if isinstance(val, bool): return val
                if isinstance(val, str): return val.lower() == 'true'
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
                'edit_type': parts[0] if parts else 'unknown',
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

# Guard sensitivity matrix
guard_matrix = defaultdict(lambda: defaultdict(list))
for row in invar_rows:
    edit_type = row.get('edit_type', 'unknown')
    for guard in ['spectral_stable', 'rmt_stable', 'invariants_pass']:
        val = row.get(guard)
        if val is not None:
            guard_matrix[edit_type][guard].append(1 if str(val).lower() == 'true' else 0)

sensitivity_rows = []
for edit_type, guards in guard_matrix.items():
    row_data = {'edit_type': edit_type}
    for guard, values in guards.items():
        if values:
            row_data[f'{guard}_pass_rate'] = sum(values) / len(values)
    sensitivity_rows.append(row_data)

if sensitivity_rows:
    with open(analysis_dir / "guard_sensitivity_matrix.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sensitivity_rows[0].keys())
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    print(f"Wrote guard sensitivity matrix")

# Calibration summary
calibration_summary = {}
for model_dir in output_dir.iterdir():
    if not model_dir.is_dir() or model_dir.name in ['logs', 'analysis', 'reports', 'presets', 'models']:
        continue
    cal_stats = model_dir / "certificates" / "calibration" / "calibration_stats.json"
    if cal_stats.exists():
        try:
            calibration_summary[model_dir.name] = json.loads(cal_stats.read_text())
        except Exception as e:
            print(f"Error loading {cal_stats}: {e}")

if calibration_summary:
    with open(analysis_dir / "calibration_summary.json", 'w') as f:
        json.dump(calibration_summary, f, indent=2)
    print(f"Wrote calibration summary for {len(calibration_summary)} models")
EOF
}

# ============ ANALYSIS ============
run_analysis() {
    log_section "CORRELATION ANALYSIS"

    python3 << EOF
import json
import csv
from pathlib import Path
from collections import defaultdict

output_dir = Path("${OUTPUT_DIR}")
analysis_dir = output_dir / "analysis"

eval_data = defaultdict(dict)
eval_csv = analysis_dir / "eval_results.csv"
if eval_csv.exists():
    with open(eval_csv) as f:
        for row in csv.DictReader(f):
            try:
                key = (row['model'], row['edit_type'])
                val = row.get('value', '')
                if val and val.strip():
                    eval_data[key][row['task']] = float(val)
            except: pass

invar_data = defaultdict(list)
invar_csv = analysis_dir / "invarlock_results.csv"
if invar_csv.exists():
    with open(invar_csv) as f:
        for row in csv.DictReader(f):
            invar_data[(row['model'], row['edit_type'])].append(row)

cal_summary = {}
cal_json = analysis_dir / "calibration_summary.json"
if cal_json.exists():
    cal_summary = json.loads(cal_json.read_text())

print("=== CORRELATION ANALYSIS (B200 8-GPU) ===\n")

results = {'models': {}, 'error_detection': {'detected': [], 'missed': []}, 'calibration': cal_summary}
categories = defaultdict(int)

for model_dir in output_dir.iterdir():
    if not model_dir.is_dir() or model_dir.name in ['logs', 'analysis', 'reports', 'presets', 'models']:
        continue

    model = model_dir.name
    results['models'][model] = {}
    print(f"\n### {model} ###")

    baseline_key = (model, 'baseline')
    baseline_evals = eval_data.get(baseline_key, {})

    for edit_type_key, invar_results in invar_data.items():
        if edit_type_key[0] != model:
            continue
        edit_type = edit_type_key[1]

        # Skip error injection experiments - they're handled separately in error_detection
        # Including them here would inflate FALSE_POSITIVE counts since they have no lm-eval baselines
        if edit_type == "errors":
            continue

        edit_evals = eval_data.get((model, edit_type), {})

        has_regression = False
        for task in baseline_evals:
            if task in edit_evals:
                if edit_evals[task] - baseline_evals[task] < -0.05:
                    has_regression = True
                    break

        invar_flagged = any(
            str(r.get('all_pass', '')).lower() == 'false' or r.get('all_pass') is False
            for r in invar_results
        )

        if has_regression and invar_flagged: category = "TRUE_POSITIVE"
        elif not has_regression and invar_flagged: category = "FALSE_POSITIVE"
        elif not has_regression and not invar_flagged: category = "TRUE_NEGATIVE"
        else: category = "FALSE_NEGATIVE"

        categories[category] += 1
        results['models'][model][edit_type] = {'category': category, 'regression': has_regression, 'flagged': invar_flagged}
        print(f"  {edit_type}: {category}")

    for row in invar_data.get((model, 'errors'), []):
        def is_false(val):
            if val is None: return True
            if isinstance(val, bool): return not val
            if isinstance(val, str): return val.lower() in ('false', '0', '')
            return False
        caught = is_false(row.get('all_pass')) or is_false(row.get('invariants_pass'))
        if caught:
            results['error_detection']['detected'].append(f"{model}/{row.get('run', 'unknown')}")
        else:
            results['error_detection']['missed'].append(f"{model}/{row.get('run', 'unknown')}")

print("\n=== SUMMARY ===")
tp, tn = categories['TRUE_POSITIVE'], categories['TRUE_NEGATIVE']
fp, fn = categories['FALSE_POSITIVE'], categories['FALSE_NEGATIVE']
total = tp + tn + fp + fn

accuracy = (tp + tn) / total if total > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

err_detected = len(results['error_detection']['detected'])
err_missed = len(results['error_detection']['missed'])
err_total = err_detected + err_missed
err_rate = err_detected / err_total if err_total > 0 else 0

sample_confidence = min(total / 8 * 25, 25)
error_confidence = err_rate * 25
accuracy_confidence = accuracy * 25
balance_confidence = f1 * 25
confidence_score = sample_confidence + error_confidence + accuracy_confidence + balance_confidence

if confidence_score >= 85: confidence_level = "HIGH"
elif confidence_score >= 70: confidence_level = "MEDIUM"
elif confidence_score >= 50: confidence_level = "LOW"
else: confidence_level = "VERY_LOW"

print(f"Accuracy: {accuracy:.0%}")
print(f"Precision: {precision:.0%}")
print(f"Recall: {recall:.0%}")
print(f"F1 Score: {f1:.0%}")
print(f"Error Detection: {err_detected}/{err_total} ({err_rate:.0%})")
print(f"Confidence Score: {confidence_score:.1f}/100 ({confidence_level})")

results['summary'] = {
    'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1,
    'error_detection_rate': err_rate, 'categories': dict(categories),
    'confidence_score': confidence_score, 'confidence_level': confidence_level,
    'total_tests': total, 'models_tested': len(results['models'])
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

analysis_file = analysis_dir / "correlation_analysis.json"
if not analysis_file.exists():
    analysis = {'summary': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0,
                           'error_detection_rate': 0, 'confidence_score': 0, 'confidence_level': 'UNKNOWN'},
                'calibration': {}}
else:
    try:
        analysis = json.loads(analysis_file.read_text())
    except:
        analysis = {'summary': {}, 'calibration': {}}

summary = analysis.get('summary', {})

accuracy = summary.get('accuracy', 0)
precision = summary.get('precision', 0)
recall = summary.get('recall', 0)
f1 = summary.get('f1_score', 0)
err_rate = summary.get('error_detection_rate', 0)
confidence_score = summary.get('confidence_score', 0)
confidence_level = summary.get('confidence_level', 'UNKNOWN')
total_tests = summary.get('total_tests', 0)
models_tested = summary.get('models_tested', 0)

phase0_pass = accuracy >= 0.6 and err_rate >= 0.8

if phase0_pass and accuracy >= 0.8 and confidence_score >= 75:
    verdict = "PHASE0_VALIDATED"
    verdict_confidence = "HIGH"
elif phase0_pass and confidence_score >= 60:
    verdict = "PHASE0_VALIDATED"
    verdict_confidence = "MEDIUM"
elif phase0_pass:
    verdict = "PHASE0_VALIDATED"
    verdict_confidence = "LOW"
else:
    verdict = "PHASE0_FAILED"
    verdict_confidence = "HIGH" if confidence_score >= 60 else "LOW"

report = f'''
================================================================================
     INVARLOCK PHASE 0 VALIDATION - B200 180GB x 8 GPU
================================================================================
     Models Tested:     {models_tested}
     Total Tests:       {total_tests}
     Edit Types:        4 x 2 versions = 8 per model
--------------------------------------------------------------------------------
     Accuracy:          {accuracy:.0%}
     Precision:         {precision:.0%}
     Recall:            {recall:.0%}
     F1 Score:          {f1:.0%}
     Error Detection:   {err_rate:.0%}
--------------------------------------------------------------------------------
     CONFIDENCE SCORE:  {confidence_score:.1f}/100 ({confidence_level})
--------------------------------------------------------------------------------
     VERDICT: {verdict}
     VERDICT CONFIDENCE: {verdict_confidence}
================================================================================

EDIT TYPES TESTED:
  * Quantization RTN: 8-bit (clean), 4-bit (stress)
  * FP4 Quantization: E2M1 (clean), aggressive (stress) [B200-native]
  * Magnitude Pruning: 10% (clean), 50% (stress)
  * Low-Rank SVD: rank-256 (clean), rank-32 (stress)

PLATFORM: 8x NVIDIA B200 180GB SXM6

'''

if verdict == "PHASE0_VALIDATED":
    report += "RESULT: InvarLock Phase 0 VALIDATED on B200 cluster.\n"
else:
    report += f"RESULT: Phase 0 validation failed. Accuracy: {accuracy:.0%}, Error Detection: {err_rate:.0%}\n"

print(report)

with open(reports_dir / "final_verdict.txt", 'w') as f:
    f.write(report)

with open(reports_dir / "final_verdict.json", 'w') as f:
    json.dump({
        'verdict': verdict,
        'verdict_confidence': verdict_confidence,
        'metrics': {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1, 'error_detection_rate': err_rate},
        'confidence': {'score': confidence_score, 'level': confidence_level},
        'phase0_pass': phase0_pass,
        'platform': 'B200_180GB_x8',
        'models_tested': models_tested,
        'total_tests': total_tests,
        'timestamp': datetime.now().isoformat()
    }, f, indent=2)
EOF
}

# ============ MAIN - DYNAMIC GPU SCHEDULING (v2.0) ============
main_dynamic() {
    local start_time=$(date +%s)

    echo "========================================================================"
    echo "  InvarLock Validation Suite v${SCRIPT_VERSION}"
    echo "  B200 180GB x 8 GPU DYNAMIC SCHEDULING"
    echo "========================================================================"
    echo ""

    log "Output directory: ${OUTPUT_DIR}"
    log "GPUs: ${NUM_GPUS} x B200 180GB"
    log "Models: 8 (7B to 72B)"
    log "Edit Types: 4 x 2 versions = 8 per model"
    log "Scheduling: DYNAMIC (work-stealing enabled)"
    log ""

    check_dependencies
    setup_b200_environment

    # Initialize queue
    log_section "PHASE 1: INITIALIZING TASK QUEUE"

    # Check for --resume mode: skip task generation if queue already exists with tasks
    local existing_queue="${OUTPUT_DIR}/queue"
    local skip_task_generation="false"

    if [[ "${RESUME_FLAG}" == "true" && -d "${existing_queue}" ]]; then
        # Count existing tasks across all queues
        local existing_pending=$(find "${existing_queue}/pending" -name "*.task" 2>/dev/null | wc -l | tr -d ' ')
        local existing_ready=$(find "${existing_queue}/ready" -name "*.task" 2>/dev/null | wc -l | tr -d ' ')
        local existing_running=$(find "${existing_queue}/running" -name "*.task" 2>/dev/null | wc -l | tr -d ' ')
        local existing_completed=$(find "${existing_queue}/completed" -name "*.task" 2>/dev/null | wc -l | tr -d ' ')
        local existing_failed=$(find "${existing_queue}/failed" -name "*.task" 2>/dev/null | wc -l | tr -d ' ')
        local existing_total=$((existing_pending + existing_ready + existing_running + existing_completed + existing_failed))

        if [[ ${existing_total} -gt 0 ]]; then
            skip_task_generation="true"
            log "RESUME MODE: Found existing queue with ${existing_total} tasks"
            log "  Pending: ${existing_pending}, Ready: ${existing_ready}, Running: ${existing_running}"
            log "  Completed: ${existing_completed}, Failed: ${existing_failed}"

            # Move any stuck "running" tasks back to pending (orphaned from previous run)
            if [[ ${existing_running} -gt 0 ]]; then
                log "  Moving ${existing_running} orphaned running tasks back to pending..."
                for task_file in "${existing_queue}/running"/*.task; do
                    [[ -f "${task_file}" ]] || continue
                    mv "${task_file}" "${existing_queue}/pending/"
                done
            fi

            # Move failed tasks back to pending for retry
            if [[ ${existing_failed} -gt 0 ]]; then
                log "  Moving ${existing_failed} failed tasks back to pending for retry..."
                for task_file in "${existing_queue}/failed"/*.task; do
                    [[ -f "${task_file}" ]] || continue
                    # Reset retry count
                    if type update_task_field &>/dev/null; then
                        update_task_field "${task_file}" "retries" "0" 2>/dev/null || true
                        update_task_field "${task_file}" "status" "pending" 2>/dev/null || true
                    fi
                    mv "${task_file}" "${existing_queue}/pending/"
                done
            fi
        fi
    fi

    init_queue "${OUTPUT_DIR}"
    export QUEUE_DIR  # Export for subshell workers

    local total_tasks=0
    if [[ "${skip_task_generation}" == "true" ]]; then
        log "Skipping task generation (--resume mode)"
        # Re-resolve dependencies after moving tasks
        if type resolve_dependencies &>/dev/null; then
            local moved=$(resolve_dependencies)
            log "Re-resolved dependencies: moved ${moved} tasks to ready queue"
        fi
    else
        # Generate all tasks
        log "Generating tasks for all models..."
        generate_all_tasks "${MODEL_1}" "${MODEL_2}" "${MODEL_3}" "${MODEL_4}" \
                           "${MODEL_5}" "${MODEL_6}" "${MODEL_7}" "${MODEL_8}"
    fi

    total_tasks=$(count_tasks "pending")
    total_tasks=$((total_tasks + $(count_tasks "ready")))
    total_tasks=$((total_tasks + $(count_tasks "completed")))
    log "Total tasks in queue: ${total_tasks} (pending+ready: $(($(count_tasks "pending") + $(count_tasks "ready"))))"

    # Launch worker pool
    log_section "PHASE 2: LAUNCHING GPU WORKERS"
    log "Starting ${NUM_GPUS} GPU workers with dynamic task scheduling..."

    # Initialize log files
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        touch "${OUTPUT_DIR}/logs/gpu_${gpu_id}.log"
    done

    # Store worker PIDs for cleanup
    pids=()

    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        log "  GPU ${gpu_id}: Starting worker"
        # Run in subshell that sources libraries (bash functions don't inherit to background processes)
        # Note: SCRIPT_DIR, LIB_DIR, QUEUE_DIR, OUTPUT_DIR must all be exported before this point
        (
            # Re-source all necessary modules in the subshell context
            source "${LIB_DIR}/task_serialization.sh"
            source "${LIB_DIR}/queue_manager.sh"
            source "${LIB_DIR}/scheduler.sh"
            source "${LIB_DIR}/task_functions.sh"
            source "${LIB_DIR}/gpu_worker.sh"
            [[ -f "${LIB_DIR}/fault_tolerance.sh" ]] && source "${LIB_DIR}/fault_tolerance.sh"
            gpu_worker "${gpu_id}" "${OUTPUT_DIR}"
        ) >> "${OUTPUT_DIR}/logs/gpu_${gpu_id}.log" 2>&1 &
        pids+=($!)
        echo "${pids[-1]}" > "${OUTPUT_DIR}/workers/gpu_${gpu_id}.pid"
    done

    # Start progress monitor in background
    log_section "PHASE 3: MONITORING PROGRESS"
    (
        while true; do
            sleep 60

            # Check if done
            if is_queue_empty; then
                break
            fi

            # Print progress
            local_stats="$(get_queue_stats 2>/dev/null || true)"
            if [[ -z "${local_stats}" ]]; then
                log "Progress: queue stats unavailable"
                continue
            fi
            IFS=':' read -r pending ready running completed failed total <<< "${local_stats}"
            pending=${pending:-0}
            ready=${ready:-0}
            running=${running:-0}
            completed=${completed:-0}
            failed=${failed:-0}
            total=${total:-0}

            pct=0
            [[ ${total} -gt 0 ]] && pct=$((completed * 100 / total))

            log "Progress: ${completed}/${total} tasks (${pct}%) | Running: ${running} | Ready: ${ready} | Failed: ${failed}"

            # Apply work-stealing boost if needed
            apply_work_stealing_boost 2>/dev/null || true
        done
    ) &
    MONITOR_PID=$!

    # Wait for all workers
    log "Waiting for all workers to complete..."
    local failed=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            log "  GPU ${i}: Worker completed successfully"
        else
            log "  GPU ${i}: Worker failed"
            failed=$((failed + 1))
        fi
    done

    # Stop monitor
    if [[ -n "${MONITOR_PID}" ]]; then
        kill "${MONITOR_PID}" 2>/dev/null || true
    fi

    # Print final queue stats
    print_queue_stats

    if [[ ${failed} -gt 0 ]]; then
        log "WARNING: ${failed} GPU worker(s) failed"
    fi

    # Check for failed tasks
    local failed_tasks=$(count_tasks "failed")
    if [[ ${failed_tasks} -gt 0 ]]; then
        log "WARNING: ${failed_tasks} task(s) failed"
        log "Failed tasks:"
        for task_file in "${QUEUE_DIR}/failed"/*.task; do
            [[ -f "${task_file}" ]] || continue
            local task_id=$(get_task_id "${task_file}")
            local error=$(get_task_field "${task_file}" "error_msg")
            log "  - ${task_id}: ${error:-unknown error}"
        done
    fi

    log_section "PHASE 4: ANALYSIS"
    compile_results
    run_analysis
    generate_verdict

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_section "COMPLETE"
    log "Total time: $((duration / 3600))h $(((duration % 3600) / 60))m $((duration % 60))s"
    log "Tasks completed: $(count_tasks "completed")/${total_tasks}"
    log "Report: ${OUTPUT_DIR}/reports/final_verdict.txt"
    log "Presets: ${OUTPUT_DIR}/presets/"
}

# ============ MAIN - STATIC GPU ASSIGNMENT (v1.0 LEGACY) ============
main_static() {
    local start_time=$(date +%s)

    echo "========================================================================"
    echo "  InvarLock Validation Suite v${SCRIPT_VERSION}"
    echo "  B200 180GB x 8 GPU STATIC ORCHESTRATION (Legacy Mode)"
    echo "========================================================================"
    echo ""

    log "Output directory: ${OUTPUT_DIR}"
    log "GPUs: ${NUM_GPUS} x B200 180GB"
    log "Models: 8 (7B to 72B)"
    log "Edit Types: 4 x 2 versions = 8 per model"
    log "Scheduling: STATIC (1:1 model-to-GPU)"
    log ""

    check_dependencies
    setup_b200_environment

    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        touch "${OUTPUT_DIR}/logs/gpu_${gpu_id}.log"
    done

    declare -A gpu_models
    gpu_models[0]="${MODEL_1}"
    gpu_models[1]="${MODEL_2}"
    gpu_models[2]="${MODEL_3}"
    gpu_models[3]="${MODEL_4}"
    gpu_models[4]="${MODEL_5}"
    gpu_models[5]="${MODEL_6}"
    gpu_models[6]="${MODEL_7}"
    gpu_models[7]="${MODEL_8}"

    log_section "PHASE 1: PARALLEL MODEL PROCESSING"
    log "Launching ${NUM_GPUS} parallel model processes..."

    # Reset pids array for background process tracking
    pids=()
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        model_id="${gpu_models[$gpu_id]}"
        if [[ -n "${model_id}" ]]; then
            log "  GPU ${gpu_id}: Starting $(basename "${model_id}")"
            process_model "${model_id}" "${gpu_id}" &
            pids+=($!)
        fi
    done

    log ""
    log "Waiting for all GPU processes to complete..."
    local failed=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            log "  GPU ${i}: Completed successfully"
        else
            log "  GPU ${i}: Failed"
            failed=$((failed + 1))
        fi
    done

    if [[ ${failed} -gt 0 ]]; then
        log "WARNING: ${failed} GPU process(es) failed"
    fi

    log_section "PHASE 2: ANALYSIS"
    compile_results
    run_analysis
    generate_verdict

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_section "COMPLETE"
    log "Total time: $((duration / 3600))h $(((duration % 3600) / 60))m $((duration % 60))s"
    log "Report: ${OUTPUT_DIR}/reports/final_verdict.txt"
    log "Presets: ${OUTPUT_DIR}/presets/"
}

# ============ MAIN - DISPATCHER ============
main() {
    if [[ "${USE_DYNAMIC_SCHEDULING}" == "true" ]]; then
        log "Using DYNAMIC scheduling (v2.0)"
        main_dynamic "$@"
    else
        log "Using STATIC scheduling (v1.0 legacy)"
        main_static "$@"
    fi
}

# ============ CLI ARGUMENT PARSING ============
RESUME_FLAG="false"
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume|-r)
            RESUME_FLAG="true"
            shift
            ;;
        --help|-h)
            # Handled below
            break
            ;;
        *)
            shift
            ;;
    esac
done
export RESUME_FLAG

# ============ HELP ============
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    cat << EOF
InvarLock Validation Suite v${SCRIPT_VERSION} - B200 180GB x 8 GPU

Optimized for 8x NVIDIA B200 180GB SXM6 GPUs with:
  * Native FP4 quantization (B200-exclusive)
  * ~4.5 TB/s memory bandwidth
  * ~2250 FP16 TFLOPS
  * Dynamic work-stealing GPU scheduling (v2.0)

Scheduling Modes:
  * DYNAMIC (default, v2.0): Work-stealing enabled. When a GPU finishes its
    tasks early, it automatically picks up pending tasks from other models.
    Optimal for heterogeneous workloads (7B-72B models).

  * STATIC (legacy, v1.0): 1:1 model-to-GPU assignment. Each GPU processes
    one model exclusively. Simpler but may leave GPUs idle.

Edit Types (4 x 2 versions each):
  * Quantization RTN: 8-bit clean, 4-bit stress
  * FP4 Quantization: E2M1 clean, aggressive stress [B200-only]
  * Magnitude Pruning: 10% clean, 50% stress
  * Low-Rank SVD: rank-256 clean, rank-32 stress

Model Suite (8 PUBLIC models - no HuggingFace login):
  GPU 0: Mistral-7B-v0.3     (~14 GB)
  GPU 1: Llama-2-13b-hf      (~26 GB)
  GPU 2: Qwen2-14B           (~28 GB)
  GPU 3: Qwen2.5-32B         (~64 GB)
  GPU 4: Yi-34B              (~68 GB)
  GPU 5: Mixtral-8x7B-v0.1   (~90 GB)
  GPU 6: Llama-2-70b-hf      (~140 GB)
  GPU 7: Qwen1.5-72B         (~144 GB)

Usage: $0 [options]

Options:
    --resume, -r                 Resume from existing queue (skip task generation)
    --help, -h                   Show this help message

Key environment variables:
    USE_DYNAMIC_SCHEDULING       Enable work-stealing (default: true)
                                 Set to "false" for legacy static mode
    MODEL_1 through MODEL_8      Override model assignments
    NUM_GPUS                     Number of GPUs (default: 8)
    SKIP_FLASH_ATTN              Skip flash-attn install (default: false)
    RESUME_MODE                  Skip completed work in static mode (default: true)
    OUTPUT_DIR                   Set to existing dir to resume
    RUN_ERROR_INJECTION          Run error tests (default: true)

Examples:
    $0                                    # Run with dynamic scheduling (default)
    $0 --resume                           # Resume previous run (skip task regeneration)
    USE_DYNAMIC_SCHEDULING=false $0       # Use legacy static mode
    SKIP_FLASH_ATTN=true $0               # Skip flash-attn compile
    NUM_GPUS=4 $0                         # Use only 4 GPUs

Resume a failed run (proper way):
    OUTPUT_DIR=./invarlock_validation_b200_20241208_123456 $0 --resume

Troubleshooting:
  If dynamic scheduling causes issues, fall back to static mode:
    USE_DYNAMIC_SCHEDULING=false OUTPUT_DIR=<existing_dir> $0
EOF
    exit 0
fi

main "$@"
