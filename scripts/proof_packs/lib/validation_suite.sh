#!/usr/bin/env bash
# validation_suite.sh
# ==========================================================
# InvarLock Proof Pack Validation Suite
# ==========================================================
# Version: proof-packs-v1
# Dependencies: bash 4+, jq, python3, invarlock CLI, nvidia-smi
# Hardware-agnostic: runs on NVIDIA GPUs where models fit VRAM.
# Designed for multi-GPU scheduling with dynamic work-stealing.
#
# EDIT TYPES (4 types × 2 versions = 8 tests per model):
# - Quantization RTN (group-wise): clean calibrated per model, 4-bit stress
# - FP8 Quantization: clean calibrated per model, E5M2 stress
# - Magnitude Pruning: clean calibrated per model, 50% stress
# - Low-Rank SVD: clean calibrated per model, rank-32 stress
#
# MODEL SUITES:
# - Defined in scripts/proof_packs/suites.sh (ungated-only models).
# - Subset targets single-GPU runs; full targets multi-GPU hardware.
#
# EXECUTION FLOW:
# 1. Optional preflight to pin model revisions
# 2. Launch models across available GPUs
# 3. Each GPU runs: calibration → edits → error injection
# 4. Correlation analysis (lm-eval vs InvarLock) → verdict
# ==========================================================

# Dynamic scheduling is always enabled.
# Static scheduling has been removed.
# Uses a "small_first" priority strategy. Multi-GPU is used only when the
# per-task profile exceeds per-GPU memory; adaptive under-allocation is disabled
# by default to avoid OOM.

# Initialize pids array early (used by cleanup trap when executed)
declare -a pids=()

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
    # Clean up lock file
    rm -f "${LOG_LOCK:-}" 2>/dev/null || true

    exit ${exit_code}
}

# Trap + strict mode are enabled only when executed as a script (not when sourced).

pack_require_bash4() {
    # Associative arrays require bash 4.0+
    if ! pack_is_bash4; then
        echo "ERROR: This script requires bash 4.0 or later (have ${BASH_VERSION})" >&2
        echo "       Associative arrays are not supported in bash ${BASH_VERSION}" >&2
        return 1
    fi
    return 0
}

# Overrideable hook for tests (simulate bash4 on bash3 without env vars).
pack_is_bash4() {
    [[ "${BASH_VERSINFO[0]}" -ge 4 ]]
}

# ============ VERSION ============
SCRIPT_VERSION="proof-packs-v1"

# ============ PACK CONFIGURATION ============
# Settings tuned for multi-GPU proof packs; defaults stay conservative.

# GPU Configuration (auto-detected at runtime unless explicitly set)
NUM_GPUS="${NUM_GPUS:-}"
export GPU_MEMORY_GB="${GPU_MEMORY_GB:-}"

# Determinism/throughput toggle for this harness (independent of InvarLock CLI presets).
# - throughput (default): enable TF32 + cuDNN benchmark for maximum speed.
# - strict: prefer deterministic-friendly flags and avoid overriding CLI presets.
PACK_DETERMINISM="${PACK_DETERMINISM:-throughput}"
case "${PACK_DETERMINISM}" in
    strict|throughput) : ;;
    *)
        PACK_DETERMINISM="throughput"
        ;;
esac
export PACK_DETERMINISM
if [[ "${PACK_DETERMINISM}" == "strict" ]]; then
    export LMEVAL_TORCH_COMPILE=0
else
    export LMEVAL_TORCH_COMPILE=1
fi

PACK_SUITE="${PACK_SUITE:-subset}"
PACK_NET="${PACK_NET:-0}"
PACK_REPEATS="${PACK_REPEATS:-0}"
PACK_OUTPUT_DIR="${PACK_OUTPUT_DIR:-}"
PACK_MODEL_REVISIONS_FILE="${PACK_MODEL_REVISIONS_FILE:-}"
if [[ -n "${PACK_OUTPUT_DIR}" && -z "${OUTPUT_DIR:-}" ]]; then
    OUTPUT_DIR="${PACK_OUTPUT_DIR}"
fi

# ============================================================
# MODEL SELECTION (DEFAULT FULL SUITE)
# ============================================================
# Defaults are ungated/public. run_suite.sh overrides these via suites.sh.
# Approx VRAM below is weights-only; exact per-task memory is computed from
# `model_profile.json` after download.

# Small models (fit on a single high-memory GPU under typical settings)
# Note: leave a MODEL_N unset to use the default below; set it to an empty string to disable.
if [[ ! ${MODEL_1+x} ]]; then MODEL_1="mistralai/Mistral-7B-v0.1"; fi           # ~14 GB
if [[ ! ${MODEL_2+x} ]]; then MODEL_2="NousResearch/Llama-2-13b-hf"; fi         # ~26 GB
if [[ ! ${MODEL_3+x} ]]; then MODEL_3="Qwen/Qwen2.5-14B"; fi                    # ~28 GB

# Medium/MoE models
if [[ ! ${MODEL_4+x} ]]; then MODEL_4="Qwen/Qwen2.5-32B"; fi                    # ~64 GB
if [[ ! ${MODEL_5+x} ]]; then MODEL_5="01-ai/Yi-34B"; fi                        # ~68 GB
if [[ ! ${MODEL_6+x} ]]; then MODEL_6="mistralai/Mixtral-8x7B-v0.1"; fi         # ~90 GB

# Large models
if [[ ! ${MODEL_7+x} ]]; then MODEL_7="NousResearch/Llama-2-70b-hf"; fi         # ~140 GB
if [[ ! ${MODEL_8+x} ]]; then MODEL_8="Qwen/Qwen1.5-72B"; fi                    # ~144 GB

pack_model_list() {
    local -a models=(
        "${MODEL_1:-}" "${MODEL_2:-}" "${MODEL_3:-}" "${MODEL_4:-}"
        "${MODEL_5:-}" "${MODEL_6:-}" "${MODEL_7:-}" "${MODEL_8:-}"
    )
    local model
    for model in "${models[@]}"; do
        [[ -n "${model}" ]] && printf '%s\n' "${model}"
    done
}

pack_model_list_array() {
    PACK_MODEL_LIST=()
    if command -v mapfile >/dev/null 2>&1; then
        mapfile -t PACK_MODEL_LIST < <(pack_model_list)
        return 0
    fi
    while IFS= read -r model; do
        [[ -n "${model}" ]] || continue
        PACK_MODEL_LIST+=("${model}")
    done < <(pack_model_list)
}

pack_model_revisions_path() {
    local path="${PACK_MODEL_REVISIONS_FILE:-${OUTPUT_DIR}/state/model_revisions.json}"
    echo "${path}"
}

pack_load_model_revisions() {
    local path
    path="$(pack_model_revisions_path)"
    if [[ -f "${path}" ]]; then
        PACK_MODEL_REVISIONS_FILE="${path}"
        export PACK_MODEL_REVISIONS_FILE
        local gated
        gated=$(python3 - "${path}" <<'PY' 2>/dev/null || echo "1"
import json
import sys

path = sys.argv[1]
try:
    data = json.loads(open(path).read())
except Exception:
    raise SystemExit(1)

models = data.get("models", {})
for model_id, entry in models.items():
    if entry.get("gated") or entry.get("private"):
        print(model_id)
        raise SystemExit(0)
print("")
PY
)
        if [[ -n "${gated}" ]]; then
            echo "ERROR: model_revisions.json includes gated/private models; proof packs require ungated models." >&2
            return 1
        fi
        return 0
    fi
    return 1
}

pack_model_revision() {
    local model_id="$1"
    local path
    path="$(pack_model_revisions_path)"
    [[ -f "${path}" ]] || return 1
    python3 - "${path}" "${model_id}" <<'PY' 2>/dev/null
import json
import sys

path = sys.argv[1]
model_id = sys.argv[2]
try:
    data = json.loads(open(path).read())
except Exception:
    raise SystemExit(0)

revision = data.get("models", {}).get(model_id, {}).get("revision") or ""
print(revision)
PY
}

pack_preflight_models() {
    local output_dir="$1"
    shift
    local -a models=("$@")
    if [[ "${PACK_NET}" != "1" ]]; then
        error_exit "Preflight requires --net 1 (PACK_NET=1)."
    fi
    if [[ ${#models[@]} -eq 0 ]]; then
        error_exit "No models provided for preflight."
    fi

    mkdir -p "${output_dir}/state"
    local out_file="${output_dir}/state/model_revisions.json"

    python3 - "${out_file}" "${models[@]}" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError
except Exception as exc:
    print("ERROR: huggingface_hub is required for preflight; install it before running with --net 1.", file=sys.stderr)
    sys.exit(2)

out_file = Path(sys.argv[1])
model_ids = sys.argv[2:]
api = HfApi(token=False)

payload = {
    "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "suite": os.environ.get("PACK_SUITE", ""),
    "model_list": model_ids,
    "models": {},
}

errors = []
for model_id in model_ids:
    try:
        info = api.model_info(model_id, token=False)
    except Exception as err:
        status = getattr(getattr(err, "response", None), "status_code", None)
        if status in (401, 403):
            msg = "requires authentication (gated/private)"
        else:
            msg = str(err)
        print(f"ERROR: {model_id} is not publicly accessible ({msg})", file=sys.stderr)
        errors.append(model_id)
        continue

    gated = bool(getattr(info, "gated", False))
    private = bool(getattr(info, "private", False))
    if gated or private:
        print(f"ERROR: {model_id} is gated/private; proof packs require ungated models.", file=sys.stderr)
        errors.append(model_id)
        continue

    payload["models"][model_id] = {
        "revision": info.sha,
        "resolved_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gated": gated,
        "private": private,
    }

if errors:
    sys.exit(2)

out_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(f"Wrote model revisions to {out_file}")
PY
    PACK_MODEL_REVISIONS_FILE="${out_file}"
    export PACK_MODEL_REVISIONS_FILE
}

# Edit Configuration
EDIT_TYPE="${EDIT_TYPE:-quant_rtn}"
EDIT_BITS="${EDIT_BITS:-8}"
EDIT_GROUP_SIZE="${EDIT_GROUP_SIZE:-128}"
EDIT_SCOPE="${EDIT_SCOPE:-ffn}"

# Edit Types to test (4 types × 2 versions each)
# Clean specs are calibrated per model using lm-eval; use "clean" sentinel.
EDIT_TYPES_CLEAN=(
    "quant_rtn:clean:ffn"        # Clean RTN (calibrated bits/group_size on FFN)
    "fp8_quant:clean:ffn"        # Clean FP8 (calibrated format on FFN)
    "magnitude_prune:clean:ffn"  # Clean pruning (calibrated sparsity on FFN)
    "lowrank_svd:clean:ffn"      # Clean low-rank (calibrated rank on FFN)
)

EDIT_TYPES_STRESS=(
    "quant_rtn:4:32:all"         # 4-bit group-wise RTN on all
    "fp8_quant:e5m2:all"         # FP8 E5M2 on all (stress)
    "magnitude_prune:0.5:all"    # 50% sparsity on all
    "lowrank_svd:32:all"         # rank-32 SVD on all
)

# Eval Configuration - conservative batch sizes for large GPUs
# Using lm-eval's "auto:N" feature: auto-detect with max cap of N
# This prevents OOM by letting lm-eval find optimal batch size bounded by max
EVAL_TASKS="${EVAL_TASKS:-mmlu,hellaswag,arc_challenge,winogrande}"
EVAL_NUM_FEWSHOT="${EVAL_NUM_FEWSHOT:-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-auto}"
EVAL_BATCH_SIZE_SMALL="${EVAL_BATCH_SIZE_SMALL:-auto:16}"   # 7B-14B models - auto with max 16
EVAL_BATCH_SIZE_MEDIUM="${EVAL_BATCH_SIZE_MEDIUM:-auto:8}"  # 30B-40B models - auto with max 8
EVAL_BATCH_SIZE_LARGE="${EVAL_BATCH_SIZE_LARGE:-auto:4}"    # 70B+ models - auto with max 4
EVAL_BATCH_SIZE_MOE="${EVAL_BATCH_SIZE_MOE:-auto:6}"        # MoE models (Mixtral) - auto with max 6
EVAL_CONTEXT_LEN="${EVAL_CONTEXT_LEN:-2048}"

# Clean edit calibration (lm-eval only; no InvarLock signal)
CALIBRATE_CLEAN_EDITS="${CALIBRATE_CLEAN_EDITS:-true}"
CLEAN_EVAL_TASKS="${CLEAN_EVAL_TASKS:-${EVAL_TASKS}}"
CLEAN_EVAL_LIMIT="${CLEAN_EVAL_LIMIT:-200}"            # 0 disables limit
CLEAN_EVAL_NUM_FEWSHOT="${CLEAN_EVAL_NUM_FEWSHOT:-${EVAL_NUM_FEWSHOT}}"
CLEAN_QUANT_BITS="${CLEAN_QUANT_BITS:-8}"
CLEAN_QUANT_GROUP_SIZES="${CLEAN_QUANT_GROUP_SIZES:-128,64,32}"
CLEAN_PRUNE_LEVELS="${CLEAN_PRUNE_LEVELS:-0.1,0.05,0.02}"
CLEAN_SVD_RANK_RATIOS="${CLEAN_SVD_RANK_RATIOS:-0.25,0.35,0.5}"
CLEAN_FP8_FORMATS="${CLEAN_FP8_FORMATS:-e4m3fn}"

# InvarLock Configuration - BASE DEFAULTS (will be overridden per-model)
# WikiText-2 validation has ~1174 usable samples
# These are conservative defaults that work for largest models (70B+)
# Smaller models will get more generous settings via get_model_invarlock_config()
INVARLOCK_PREVIEW_WINDOWS="${INVARLOCK_PREVIEW_WINDOWS:-32}"
INVARLOCK_FINAL_WINDOWS="${INVARLOCK_FINAL_WINDOWS:-32}"
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

# Memory planning overheads (GB) for task budgeting.
MODEL_LOAD_OVERHEAD_GB="${MODEL_LOAD_OVERHEAD_GB:-4}"
EDIT_OVERHEAD_GB="${EDIT_OVERHEAD_GB:-8}"
BATCH_EDIT_OVERHEAD_GB="${BATCH_EDIT_OVERHEAD_GB:-8}"
EVAL_OVERHEAD_GB="${EVAL_OVERHEAD_GB:-6}"
INVARLOCK_OVERHEAD_GB="${INVARLOCK_OVERHEAD_GB:-6}"

# Task timeout (seconds). Set to 0 or empty to disable.
export TASK_TIMEOUT_DEFAULT="${TASK_TIMEOUT_DEFAULT:-21600}"

# Output - supports resume by specifying existing directory.
# When executed, the entrypoint populates a date-stamped default.
OUTPUT_DIR="${OUTPUT_DIR:-}"

# ============================================================
# HUGGINGFACE CACHE LOCATION (CRITICAL ON GPU NODES)
# ============================================================
# HuggingFace defaults to writing caches under ~/.cache (e.g., /root/.cache when
# running as root). On many GPU nodes, / or /root is small, causing silent ENOSPC
# failures during dataset/model downloads while GPUs sit idle.
#
# Default behavior for this suite: co-locate caches under OUTPUT_DIR so they land
# on the same (usually large) filesystem as the run artifacts.
#
# Override by exporting HF_HOME / HF_HUB_CACHE / HF_DATASETS_CACHE /
# TRANSFORMERS_CACHE before running this script.
pack_setup_hf_cache_dirs() {
    if [[ -z "${OUTPUT_DIR:-}" ]]; then
        echo "ERROR: OUTPUT_DIR is not set; use --out or PACK_OUTPUT_DIR." >&2
        return 1
    fi
    export HF_HOME="${HF_HOME:-${OUTPUT_DIR}/.hf}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
    if ! mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"; then
        echo "ERROR: Failed to create HuggingFace cache directories under: ${HF_HOME}" >&2
        return 1
    fi
    return 0
}

pack_run_determinism_repeats() {
    local repeats="${PACK_REPEATS:-0}"
    if [[ -z "${OUTPUT_DIR:-}" ]]; then
        echo "ERROR: OUTPUT_DIR is not set; use --out or PACK_OUTPUT_DIR." >&2
        return 1
    fi
    if [[ -z "${repeats}" || "${repeats}" == "0" ]]; then
        return 0
    fi
    if ! [[ "${repeats}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: PACK_REPEATS must be an integer" >&2
        return 1
    fi

    if [[ -z "${PACK_MODEL_LIST[*]:-}" ]]; then
        pack_model_list_array
    fi

    local model_id="${PACK_MODEL_LIST[0]:-}"
    if [[ -z "${model_id}" ]]; then
        echo "ERROR: PACK_REPEATS requested but no models configured." >&2
        return 1
    fi

    local model_name
    model_name=$(sanitize_model_name "${model_id}")
    local model_output_dir="${OUTPUT_DIR}/${model_name}"
    local baseline_path=""
    if [[ -f "${model_output_dir}/.baseline_path" ]]; then
        baseline_path="$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)"
    fi
    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: PACK_REPEATS requires a baseline path for ${model_name}." >&2
        return 1
    fi

    local edit_spec=""
    local repeat_mode="clean"
    if [[ ${#EDIT_TYPES_CLEAN[@]} -gt 0 ]]; then
        edit_spec="${EDIT_TYPES_CLEAN[0]}"
        repeat_mode="clean"
    elif [[ ${#EDIT_TYPES_STRESS[@]} -gt 0 ]]; then
        edit_spec="${EDIT_TYPES_STRESS[0]}"
        repeat_mode="stress"
    else
        echo "ERROR: PACK_REPEATS requested but no edit specs configured." >&2
        return 1
    fi

    local edit_path
    edit_path=$(process_edit "${baseline_path}" "${edit_spec}" "${repeat_mode}" "${model_name}" "0" "${model_output_dir}") || return 1
    if [[ -z "${edit_path}" || ! -d "${edit_path}" ]]; then
        echo "ERROR: Failed to prepare edit for determinism repeats." >&2
        return 1
    fi
    local edit_name
    edit_name=$(basename "${edit_path}")

    local preset_dir="${OUTPUT_DIR}/presets"
    local det_dir="${OUTPUT_DIR}/determinism/${model_name}/${edit_name}"
    mkdir -p "${det_dir}" || return 1

    local -a certs=()
    local run
    for run in $(seq 1 "${repeats}"); do
        run_invarlock_certify "${edit_path}" "${baseline_path}" "${det_dir}" "repeat_${run}" "${preset_dir}" "${model_name}" "0" || return 1
        local cert_path="${det_dir}/repeat_${run}/evaluation.cert.json"
        if [[ -f "${cert_path}" ]]; then
            certs+=("${cert_path}")
        fi
    done

    mkdir -p "${OUTPUT_DIR}/analysis" || return 1
    python3 - "${OUTPUT_DIR}/analysis/determinism_repeats.json" "${model_id}" "${edit_name}" "${repeats}" "${PACK_DETERMINISM}" "${PACK_SUITE}" "${certs[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

out_path = Path(sys.argv[1])
model_id = sys.argv[2]
edit_name = sys.argv[3]
try:
    requested = int(sys.argv[4])
except Exception:
    requested = 0
mode = sys.argv[5]
suite = sys.argv[6]
cert_paths = [Path(p) for p in sys.argv[7:]]

hashes = []
ratios = []
errors = []

for path in cert_paths:
    try:
        raw = path.read_bytes()
        hashes.append(hashlib.sha256(raw).hexdigest())
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        errors.append(f"{path}: {exc}")
        continue

    ratio = None
    verdict = data.get("verdict") or {}
    metrics = data.get("metrics") or {}
    for candidate in (
        verdict.get("primary_metric_ratio"),
        verdict.get("primary_metric_ratio_raw"),
        verdict.get("primary_metric_ratio_mean"),
        metrics.get("primary_metric_ratio"),
        metrics.get("primary_metric_ratio_mean"),
    ):
        if isinstance(candidate, (int, float)):
            ratio = float(candidate)
            break
    if ratio is not None:
        ratios.append(ratio)

hashes_match = bool(hashes) and len(set(hashes)) == 1
ratio_summary = None
if ratios:
    ratio_summary = {
        "min": min(ratios),
        "max": max(ratios),
        "delta": max(ratios) - min(ratios),
    }

payload = {
    "requested": requested,
    "completed": len(cert_paths),
    "mode": mode,
    "suite": suite,
    "model_id": model_id,
    "edit_name": edit_name,
    "cert_hashes_match": hashes_match,
    "cert_hashes": hashes,
    "primary_metric_ratio": ratio_summary,
    "errors": errors,
    "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
}

out_path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

# Resume support - skip completed steps if output files exist
RESUME_MODE="${RESUME_MODE:-true}"

# ============ GPU OPTIMIZATION FLAGS ============
# GPU selection is configured at runtime:
# - If `CUDA_VISIBLE_DEVICES` is explicitly set (e.g., by Slurm or the user), it is respected.
# - Otherwise, the harness detects available GPUs and uses all of them.
# The selected pool is exported as `GPU_ID_LIST` (physical GPU indices) for scheduler/workers.

# TF32 / cuDNN benchmark behavior depends on PACK_DETERMINISM:
# - throughput: enable TF32 + benchmark for speed (script-level runs).
# - strict: avoid overriding determinism-friendly flags; rely on InvarLock presets.
if [[ "${PACK_DETERMINISM}" == "strict" ]]; then
    export NVIDIA_TF32_OVERRIDE=0
    export CUDNN_BENCHMARK=0
else
    export NVIDIA_TF32_OVERRIDE=1
    export CUDNN_BENCHMARK=1
fi

# Enable text-level deduplication
export INVARLOCK_DEDUP_TEXTS=1

# Memory optimization for large-model runs
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold:0.9"
unset PYTORCH_ALLOC_CONF 2>/dev/null || true

# Force deterministic workspace config with larger workspace (strict only)
if [[ "${PACK_DETERMINISM}" == "strict" ]]; then
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
else
    unset CUBLAS_WORKSPACE_CONFIG 2>/dev/null || true
fi

# Keep CUDA caching enabled for maximum memory reuse
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

pack_apply_network_mode() {
    local mode="${1:-${PACK_NET}}"
    mode=$(echo "${mode}" | tr '[:upper:]' '[:lower:]')
    case "${mode}" in
        1|true|yes|on)
            PACK_NET=1
            export INVARLOCK_ALLOW_NETWORK=1
            export HF_DATASETS_OFFLINE=0
            export TRANSFORMERS_OFFLINE=0
            export HF_HUB_OFFLINE=0
            export HF_HUB_DISABLE_TELEMETRY=1
            ;;
        *)
            PACK_NET=0
            export INVARLOCK_ALLOW_NETWORK=0
            export HF_DATASETS_OFFLINE=1
            export TRANSFORMERS_OFFLINE=1
            export HF_HUB_OFFLINE=1
            export HF_HUB_DISABLE_TELEMETRY=1
            ;;
    esac
}

pack_apply_network_mode "${PACK_NET}"

# PM acceptance range used during validation
# These bounds help avoid unnecessary gate failures during validation runs
export INVARLOCK_PM_ACCEPTANCE_MIN="${INVARLOCK_PM_ACCEPTANCE_MIN:-0.90}"
export INVARLOCK_PM_ACCEPTANCE_MAX="${INVARLOCK_PM_ACCEPTANCE_MAX:-1.20}"

# Flash attention flag - will be set dynamically based on availability
export FLASH_ATTENTION_AVAILABLE="false"

# FP8 support flag - detected in setup
export FP8_NATIVE_SUPPORT="false"
# FP4 support flag retained for optional FP4 edits
export FP4_NATIVE_SUPPORT="${FP4_NATIVE_SUPPORT:-false}"

# Target memory fraction (0.92 = 92% of available) - optimal zone
export CUDA_MEMORY_FRACTION=0.92

if ! declare -F _pack_script_dir >/dev/null 2>&1; then
    _pack_script_dir() {
        cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
    }
fi

pack_source_libs() {
    # ============ LIB MODULES FOR DYNAMIC SCHEDULING ============
    SCRIPT_DIR="$(_pack_script_dir)"
    export SCRIPT_DIR  # Export for subshell workers

    # Determine lib directory - support lib/ and flat layouts.
    if [[ -f "${SCRIPT_DIR}/task_serialization.sh" ]]; then
        LIB_DIR="${SCRIPT_DIR}"
    elif [[ -d "${SCRIPT_DIR}/lib" && -f "${SCRIPT_DIR}/lib/task_serialization.sh" ]]; then
        LIB_DIR="${SCRIPT_DIR}/lib"
    elif [[ -d "${SCRIPT_DIR}/../lib" && -f "${SCRIPT_DIR}/../lib/task_serialization.sh" ]]; then
        LIB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/lib"
    else
        LIB_DIR="${SCRIPT_DIR}"
    fi
    export LIB_DIR  # Export for subshell workers

    # Source dynamic scheduling modules (required - optimal configuration)
    if [[ -f "${LIB_DIR}/task_serialization.sh" ]]; then
        source "${LIB_DIR}/task_serialization.sh"
        export TASK_SERIALIZATION_LOADED=1
    else
        echo "ERROR: lib/task_serialization.sh not found (dynamic scheduling is required)" >&2
        return 1
    fi

    if [[ -f "${LIB_DIR}/queue_manager.sh" ]]; then
        source "${LIB_DIR}/queue_manager.sh"
        export QUEUE_MANAGER_LOADED=1
    else
        echo "ERROR: lib/queue_manager.sh not found" >&2
        return 1
    fi

    if [[ -f "${LIB_DIR}/scheduler.sh" ]]; then
        source "${LIB_DIR}/scheduler.sh"
        export SCHEDULER_LOADED=1
    else
        echo "ERROR: lib/scheduler.sh not found" >&2
        return 1
    fi

    if [[ -f "${LIB_DIR}/task_functions.sh" ]]; then
        source "${LIB_DIR}/task_functions.sh"
        export TASK_FUNCTIONS_LOADED=1
    else
        echo "ERROR: lib/task_functions.sh not found" >&2
        return 1
    fi

    if [[ -f "${LIB_DIR}/gpu_worker.sh" ]]; then
        source "${LIB_DIR}/gpu_worker.sh"
        export GPU_WORKER_LOADED=1
    else
        echo "ERROR: lib/gpu_worker.sh not found" >&2
        return 1
    fi

    if [[ -f "${LIB_DIR}/fault_tolerance.sh" ]]; then
        source "${LIB_DIR}/fault_tolerance.sh"
        export FAULT_TOLERANCE_LOADED=1
    fi

    return 0
}

# Fallback resolver for clean edit specs when task_functions isn't sourced.
if ! declare -F resolve_edit_params >/dev/null 2>&1; then
resolve_edit_params() {
    local model_output_dir="$1"
    local edit_spec="$2"
    local version_hint="${3:-}"

    python3 - "${model_output_dir}" "${edit_spec}" "${version_hint}" <<'PY'
import json
import sys
from pathlib import Path

model_output_dir = Path(sys.argv[1])
edit_spec = sys.argv[2] if len(sys.argv) > 2 else ""
version_hint = sys.argv[3] if len(sys.argv) > 3 else ""

parts = edit_spec.split(":") if edit_spec else []
edit_type = parts[0] if parts else ""
param1 = parts[1] if len(parts) > 1 else ""
param2 = parts[2] if len(parts) > 2 else ""
scope = parts[3] if len(parts) > 3 else ""

if edit_type != "quant_rtn" and not scope:
    scope = param2
    param2 = ""

if edit_type == "quant_rtn" and not scope:
    if param1 and param2:
        scope = param2
        param2 = ""

clean_spec = param1 == "clean"
status = "selected"
reason = ""
edit_dir_name = ""

if clean_spec:
    clean_file = model_output_dir / "state" / "clean_edit_params.json"
    if not clean_file.exists():
        status = "missing"
    else:
        try:
            data = json.loads(clean_file.read_text())
        except Exception:
            data = {}
        entry = data.get(edit_type) or {}
        status = str(entry.get("status") or "missing")
        reason = str(entry.get("reason") or "")
        if status == "selected":
            if edit_type == "quant_rtn":
                param1 = str(entry.get("bits", ""))
                param2 = str(entry.get("group_size", ""))
                scope = str(entry.get("scope") or scope or "")
            elif edit_type in ("fp8_quant", "fp4_quant"):
                param1 = str(entry.get("format", ""))
                scope = str(entry.get("scope") or scope or "")
            elif edit_type == "magnitude_prune":
                param1 = str(entry.get("sparsity", ""))
                scope = str(entry.get("scope") or scope or "")
            elif edit_type == "lowrank_svd":
                param1 = str(entry.get("rank", ""))
                scope = str(entry.get("scope") or scope or "")
            edit_dir_name = str(entry.get("edit_dir_name") or "")
else:
    def _is_int(val):
        try:
            int(val)
            return True
        except Exception:
            return False

    def _is_float(val):
        try:
            float(val)
            return True
        except Exception:
            return False

    if edit_type == "quant_rtn":
        if not (_is_int(param1) and _is_int(param2)):
            status = "invalid"
            reason = "invalid_quant_params"
    elif edit_type == "magnitude_prune":
        if not _is_float(param1):
            status = "invalid"
            reason = "invalid_prune_sparsity"
    elif edit_type == "lowrank_svd":
        if not _is_int(param1):
            status = "invalid"
            reason = "invalid_lowrank_rank"
    elif edit_type in ("fp8_quant", "fp4_quant"):
        if not param1:
            status = "invalid"
            reason = "invalid_fp_format"

version = version_hint or ("clean" if clean_spec else "")

if status == "selected" and not edit_dir_name:
    if edit_type == "quant_rtn":
        edit_dir_name = f"quant_{param1}bit_{version}" if version else ""
    elif edit_type == "fp8_quant":
        edit_dir_name = f"fp8_{param1}_{version}" if version else ""
    elif edit_type == "fp4_quant":
        edit_dir_name = f"fp4_{param1}_{version}" if version else ""
    elif edit_type == "magnitude_prune":
        try:
            pct = int(float(param1) * 100)
        except Exception:
            pct = 0
        edit_dir_name = f"prune_{pct}pct_{version}" if version else ""
    elif edit_type == "lowrank_svd":
        edit_dir_name = f"svd_rank{param1}_{version}" if version else ""
    else:
        edit_dir_name = f"{edit_type}_{version}" if version else ""

payload = {
    "status": status,
    "reason": reason,
    "edit_type": edit_type,
    "param1": param1,
    "param2": param2,
    "scope": scope,
    "version": version,
    "edit_dir_name": edit_dir_name,
}
print(json.dumps(payload))
PY
}
fi

pack_setup_output_dirs() {
    # ============ SETUP ============
    mkdir -p "${OUTPUT_DIR}"/{logs,models,evals,certificates,analysis,reports,presets,workers,state} || return 1
    LOG_FILE="${OUTPUT_DIR}/logs/main.log"

    # Create a lock file for thread-safe logging
    LOG_LOCK="${OUTPUT_DIR}/logs/.log_lock"
    return 0
}

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

sanitize_model_name() {
    local model_id="$1"
    echo "${model_id}" \
        | tr '[:upper:]' '[:lower:]' \
        | sed 's#/#__#g' \
        | tr ' ' '_' \
        | tr -cd '[:alnum:]_-'
}

# ============ GPU SELECTION / TOPOLOGY ============
# Stable pool of physical GPU indices used by this run.
# - If CUDA_VISIBLE_DEVICES is set, it is treated as an explicit physical GPU list.
# - Otherwise, we detect all GPUs via nvidia-smi and use them.
#
# NOTE: Workers/tasks will override CUDA_VISIBLE_DEVICES per-task (single- or multi-GPU),
# so we keep the pool in GPU_ID_LIST for scheduler enumeration.
GPU_ID_LIST="${GPU_ID_LIST:-}"

# Print newline-separated GPU IDs for this run.
# Defaults to 0..NUM_GPUS-1 when GPU_ID_LIST isn't set yet.
list_run_gpu_ids() {
    if [[ -n "${GPU_ID_LIST:-}" ]]; then
        echo "${GPU_ID_LIST}" | tr -d ' ' | tr ',' '\n' | sed '/^$/d'
    else
        local total="${NUM_GPUS:-8}"
        if ! [[ "${total}" =~ ^[0-9]+$ ]]; then
            total=8
        fi
        [[ ${total} -lt 1 ]] && total=1
        seq 0 $((total - 1))
    fi
}

configure_gpu_pool() {
    # Identify candidate GPU IDs
    local source="nvidia-smi"
    local raw_list=""
    local -a candidates=()

    # Prefer CUDA_VISIBLE_DEVICES if set (Slurm/Ray commonly set this).
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        source="CUDA_VISIBLE_DEVICES"
        raw_list="${CUDA_VISIBLE_DEVICES}"
    elif [[ -n "${GPU_ID_LIST:-}" ]]; then
        # Fallback: allow callers to set GPU_ID_LIST directly.
        source="GPU_ID_LIST"
        raw_list="${GPU_ID_LIST}"
    fi

    if [[ -n "${raw_list}" ]]; then
        IFS=',' read -ra candidates <<< "${raw_list}"
    else
        while IFS= read -r id; do
            [[ -n "${id}" ]] && candidates+=("${id}")
        done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    fi

    # Sanitize and validate IDs
    local -a cleaned=()
    local id
    for id in "${candidates[@]}"; do
        id=$(echo "${id}" | tr -d ' ')
        [[ -z "${id}" ]] && continue
        if ! [[ "${id}" =~ ^[0-9]+$ ]]; then
            error_exit "Non-numeric GPU id in ${source}: '${id}'. Set CUDA_VISIBLE_DEVICES to numeric indices."
        fi
        if ! nvidia-smi -i "${id}" &>/dev/null; then
            error_exit "GPU id '${id}' from ${source} is not valid on this host."
        fi
        cleaned+=("${id}")
    done

    if [[ ${#cleaned[@]} -eq 0 ]]; then
        error_exit "No usable GPU ids found (${source})."
    fi

    # Determine how many GPUs to use.
    local requested="${NUM_GPUS:-}"
    if [[ -z "${requested}" ]]; then
        requested="${#cleaned[@]}"
    fi
    if ! [[ "${requested}" =~ ^[0-9]+$ ]]; then
        requested="${#cleaned[@]}"
    fi
    if [[ ${requested} -lt 1 ]]; then
        requested=1
    fi
    if [[ ${requested} -gt ${#cleaned[@]} ]]; then
        log "WARNING: NUM_GPUS=${requested} > available ${#cleaned[@]} from ${source}; clamping"
        requested=${#cleaned[@]}
    fi

    local -a selected=("${cleaned[@]:0:${requested}}")
    GPU_ID_LIST=$(IFS=','; echo "${selected[*]}")
    export GPU_ID_LIST
    export NUM_GPUS="${#selected[@]}"

    # Normalize CUDA_VISIBLE_DEVICES so torch + subprocesses see the same pool.
    export CUDA_VISIBLE_DEVICES="${GPU_ID_LIST}"

    log "GPU pool configured from ${source}: NUM_GPUS=${NUM_GPUS}, GPU_ID_LIST=${GPU_ID_LIST}"
}

# ============ DISK PRESSURE HARDENING ============
# Abort early under low disk to avoid half-written artifacts and cascading failures.
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-200}"

format_gb_as_tb() {
    local gb="$1"
    if [[ -z "${gb}" || ! "${gb}" =~ ^[0-9]+$ ]]; then
        echo ""
        return 0
    fi
    awk -v gb="${gb}" 'BEGIN { printf "%.1f", gb / 1024.0 }'
}

get_free_disk_gb() {
    local path="$1"
    local free_disk
    free_disk=$(df -BG "${path}" 2>/dev/null | awk 'NR==2 {gsub(/G/,""); print $4}')
    [[ -z "${free_disk}" || ! "${free_disk}" =~ ^[0-9]+$ ]] && return 1
    echo "${free_disk}"
}

estimate_model_weights_gb() {
    local model_id="$1"
    [[ -z "${model_id}" ]] && return 1
    if [[ -d "${model_id}" ]]; then
        return 1  # Unknown for local paths without a profile.
    fi
    local lower
    lower="$(echo "${model_id}" | tr '[:upper:]' '[:lower:]')"

    # Special-case MoE naming.
    if [[ "${lower}" == *"mixtral"* || "${lower}" == *"8x7b"* ]]; then
        echo 90
        return 0
    fi

    case "${lower}" in
        *"72b"*) echo 144 ;;
        *"70b"*) echo 140 ;;
        *"34b"*) echo 68 ;;
        *"32b"*) echo 64 ;;
        *"14b"*) echo 28 ;;
        *"13b"*) echo 26 ;;
        *"7b"*) echo 14 ;;
        *) return 1 ;;
    esac
}

estimate_planned_model_storage_gb() {
    local -a models=()
    if command -v mapfile >/dev/null 2>&1; then
        mapfile -t models < <(pack_model_list)
    else
        while IFS= read -r model; do
            [[ -n "${model}" ]] || continue
            models+=("${model}")
        done < <(pack_model_list)
    fi

    local edits_total=$(( ${#EDIT_TYPES_CLEAN[@]} + ${#EDIT_TYPES_STRESS[@]} ))
    local errors_total=0
    if [[ "${RUN_ERROR_INJECTION}" == "true" ]]; then
        errors_total=5  # nan_injection, inf_injection, extreme_quant, scale_explosion, zero_layer
    fi

    local baseline_mode="${PACK_BASELINE_STORAGE_MODE:-snapshot_symlink}"
    local baseline_copy=1
    if [[ "${baseline_mode}" == "snapshot_symlink" ]]; then
        baseline_copy=0  # baseline files are symlinks to HF hub cache blobs
    fi

    local hub_cache_on_output_fs=1
    if [[ -n "${HF_HUB_CACHE:-}" ]]; then
        local out_dev=""
        local hub_dev=""
        out_dev=$(df -P "${OUTPUT_DIR}" 2>/dev/null | awk 'NR==2 {print $1}' || true)
        hub_dev=$(df -P "${HF_HUB_CACHE}" 2>/dev/null | awk 'NR==2 {print $1}' || true)
        if [[ -n "${out_dev}" && -n "${hub_dev}" && "${out_dev}" != "${hub_dev}" ]]; then
            hub_cache_on_output_fs=0
        fi
    fi

    local total_gb=0
    local unknown=0
    local model_id
    for model_id in "${models[@]}"; do
        [[ -n "${model_id}" ]] || continue

        local w_gb=""
        w_gb="$(estimate_model_weights_gb "${model_id}" 2>/dev/null || true)"
        if [[ -z "${w_gb}" || ! "${w_gb}" =~ ^[0-9]+$ ]]; then
            unknown=$((unknown + 1))
            continue
        fi

        # Storage copies:
        # - 1× HF hub cache download (when model_id is remote)
        # - 1× baseline saved under OUTPUT_DIR (unless snapshot_symlink mode)
        # - N× edits (currently saved as full bf16 copies)
        # - M× error models (also full copies) when enabled
        local hub_copy=1
        [[ -d "${model_id}" ]] && hub_copy=0
        [[ ${hub_cache_on_output_fs} -eq 0 ]] && hub_copy=0
        local effective_baseline_copy=${baseline_copy}
        [[ -d "${model_id}" ]] && effective_baseline_copy=0
        local copies=$((hub_copy + effective_baseline_copy + edits_total + errors_total))

        total_gb=$((total_gb + (w_gb * copies)))
    done

    [[ ${unknown} -gt 0 ]] && return 1
    echo "${total_gb}"
}

disk_preflight() {
    [[ "${PACK_SKIP_DISK_PREFLIGHT:-0}" == "1" ]] && return 0

    local free_gb=""
    free_gb=$(get_free_disk_gb "${OUTPUT_DIR}" 2>/dev/null || echo "")
    [[ -z "${free_gb}" ]] && return 0

    local planned_gb=""
    planned_gb=$(estimate_planned_model_storage_gb 2>/dev/null || echo "")
    [[ -z "${planned_gb}" ]] && return 0

    local min_free="${MIN_FREE_DISK_GB:-200}"
    if ! [[ "${min_free}" =~ ^[0-9]+$ ]]; then
        min_free=200
    fi

    local required_gb=$((planned_gb + min_free))
    if [[ ${free_gb} -ge ${required_gb} ]]; then
        return 0
    fi

    log_section "ABORTING: DISK PREFLIGHT"
    log "ERROR: Estimated storage for this configuration: ~${planned_gb}GB (~$(format_gb_as_tb "${planned_gb}")TB) for model weights alone."
    log "       Free disk on output filesystem: ${free_gb}GB (~$(format_gb_as_tb "${free_gb}")TB)."
    log "       This suite saves full bf16 copies of edits (+ error models if enabled)."
    log "       Baseline storage mode: ${PACK_BASELINE_STORAGE_MODE:-snapshot_symlink} (snapshot_symlink avoids a full extra baseline copy)."
    log "       Fix: mount a larger volume and set OUTPUT_DIR, or run the subset suite, or set RUN_ERROR_INJECTION=false."
    log "       Override (not recommended): PACK_SKIP_DISK_PREFLIGHT=1"

    # Resume mode may already have artifacts; allow user to proceed if explicitly resuming.
    if [[ "${RESUME_FLAG:-false}" == "true" ]]; then
        log "WARNING: --resume mode enabled; continuing despite preflight estimate."
        return 0
    fi

    error_exit "Insufficient disk for planned run (need >= ${required_gb}GB incl MIN_FREE_DISK_GB=${min_free})."
}

write_disk_pressure_state() {
    local free_gb="$1"
    local min_gb="$2"
    mkdir -p "${OUTPUT_DIR}/state" 2>/dev/null || true
    cat > "${OUTPUT_DIR}/state/disk_pressure.json" << EOF
{
  "detected_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "free_gb": ${free_gb},
  "min_free_gb": ${min_gb},
  "output_dir": "${OUTPUT_DIR}"
}
EOF
}

handle_disk_pressure() {
    local free_gb="$1"
    local min_gb="$2"

    log_section "ABORTING: DISK PRESSURE"
    log "ERROR: Low disk space in output filesystem: ${free_gb}GB free (< ${min_gb}GB)."
    log "       Free disk space and resume with: OUTPUT_DIR=${OUTPUT_DIR} $0 --resume"
    log "       (Override threshold: MIN_FREE_DISK_GB=0 to disable, or set a smaller value)"

    write_disk_pressure_state "${free_gb}" "${min_gb}"

    # Stop workers and aggressively stop running tasks so they don't keep writing.
    if type signal_shutdown &>/dev/null; then
        signal_shutdown "${OUTPUT_DIR}"
    else
        touch "${OUTPUT_DIR}/workers/SHUTDOWN"
    fi

    # Kill task process groups and move running tasks back to pending for resume.
    # Guard for early failures before the queue is initialized.
    if [[ -n "${QUEUE_DIR:-}" && -d "${QUEUE_DIR}" ]]; then
        local gpu_id
        for gpu_id in $(list_run_gpu_ids); do
            reclaim_orphaned_tasks "${gpu_id}" >> "${LOG_FILE}" 2>&1 || true
        done
    fi

    error_exit "Aborted due to disk pressure (free ${free_gb}GB < ${min_gb}GB)."
}

# ============ GPU ENVIRONMENT SETUP ============
setup_pack_environment() {
    log_section "PHASE 0: GPU ENVIRONMENT SETUP"

    local env_report
    env_report=$(python3 - <<'PY'
import os
import sys
import torch

print("=== Proof Pack Environment Configuration ===\n")

if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    sys.exit(1)

num_gpus = torch.cuda.device_count()
print(f"GPUs Detected: {num_gpus}")

mode = str(os.environ.get("PACK_DETERMINISM", "throughput")).strip().lower()
if mode not in {"throughput", "strict"}:
    mode = "throughput"

fp8_support = hasattr(torch, "float8_e4m3fn")

gpu_names = []
gpu_mem_gb = []
total_vram = 0.0

for i in range(num_gpus):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    gpu_names.append(name)
    gpu_mem_gb.append(mem)
    total_vram += mem
    print(f"  GPU {i}: {name} ({mem:.1f} GB)")

min_vram = min(gpu_mem_gb) if gpu_mem_gb else 0.0
primary_name = gpu_names[0] if gpu_names else ""
print(f"\nTotal VRAM: {total_vram:.1f} GB")
print(f"Min GPU VRAM: {min_vram:.1f} GB")
print(f"FP8 Support: {fp8_support}")

if mode == "strict":
    print("\nDeterminism mode: strict (PACK_DETERMINISM=strict)")
    try:
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception:
        print("WARNING: deterministic algorithms could not be fully enabled")
    try:
        cudnn_mod = getattr(torch.backends, "cudnn", None)
        if cudnn_mod is not None:
            cudnn_mod.benchmark = False
            cudnn_mod.enabled = True
            if hasattr(cudnn_mod, "deterministic"):
                cudnn_mod.deterministic = True
            if hasattr(cudnn_mod, "allow_tf32"):
                cudnn_mod.allow_tf32 = False
    except Exception:
        pass
    try:
        matmul = getattr(getattr(torch.backends, "cuda", object()), "matmul", None)
        if matmul is not None and hasattr(matmul, "allow_tf32"):
            matmul.allow_tf32 = False
    except Exception:
        pass
    print("\nTF32 enabled: False")
    print("cuDNN benchmark: False")
else:
    print("\nDeterminism mode: throughput (PACK_DETERMINISM=throughput)")
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        cudnn_mod = getattr(torch.backends, "cudnn", None)
        if cudnn_mod is not None:
            cudnn_mod.allow_tf32 = True
            cudnn_mod.benchmark = True
            cudnn_mod.enabled = True
    except Exception:
        pass
    print("\nTF32 enabled: True")
    print("cuDNN benchmark: True")

if torch.cuda.is_bf16_supported():
    torch.set_default_dtype(torch.bfloat16)
    print("Default dtype: bfloat16")
else:
    print("Default dtype: float16 (BF16 not supported)")

try:
    from transformers.utils import is_flash_attn_2_available
    flash_avail = is_flash_attn_2_available()
    print(f"\nFlash Attention 2: {flash_avail}")
except Exception:
    print("\nFlash Attention 2: Unknown (transformers too old)")

compile_avail = hasattr(torch, "compile")
print(f"torch.compile: {compile_avail}")

print(f"\n[PACK_GPU_NAME={primary_name}]")
print(f"[PACK_GPU_MEM_GB={int(round(min_vram))}]")
print(f"[PACK_GPU_COUNT={num_gpus}]")
if fp8_support:
    print("[FP8_NATIVE_SUPPORT=true]")
else:
    print("[FP8_NATIVE_SUPPORT=false]")

print("\n=== Environment Ready for Proof Pack Runs ===")
PY
)
    local setup_rc=$?
    if [[ ${setup_rc} -ne 0 ]]; then
        printf '%s\n' "${env_report}"
        return ${setup_rc}
    fi
    printf '%s\n' "${env_report}"

    PACK_GPU_NAME=$(printf '%s\n' "${env_report}" | sed -n 's/^\[PACK_GPU_NAME=//p' | sed 's/\]$//' | tail -1)
    PACK_GPU_MEM_GB=$(printf '%s\n' "${env_report}" | sed -n 's/^\[PACK_GPU_MEM_GB=//p' | sed 's/\]$//' | tail -1)
    PACK_GPU_COUNT=$(printf '%s\n' "${env_report}" | sed -n 's/^\[PACK_GPU_COUNT=//p' | sed 's/\]$//' | tail -1)
    if [[ "${env_report}" == *"[FP8_NATIVE_SUPPORT=true]"* ]]; then
        export FP8_NATIVE_SUPPORT="true"
    else
        export FP8_NATIVE_SUPPORT="false"
    fi
    if [[ -n "${PACK_GPU_MEM_GB}" && -z "${GPU_MEMORY_GB}" ]]; then
        GPU_MEMORY_GB="${PACK_GPU_MEM_GB}"
    fi
    export PACK_GPU_NAME PACK_GPU_MEM_GB PACK_GPU_COUNT GPU_MEMORY_GB
    log "GPU Environment Setup: Complete (FP8_NATIVE_SUPPORT=${FP8_NATIVE_SUPPORT})"
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

    # Check InvarLock (Python module and CLI)
    python3 -c "import invarlock" 2>/dev/null || missing+=("invarlock")
    command -v invarlock >/dev/null 2>&1 || missing+=("invarlock-cli")

    # Check shell utilities used by the suite
    command -v jq >/dev/null 2>&1 || missing+=("jq")
    command -v nvidia-smi >/dev/null 2>&1 || missing+=("nvidia-smi")
    command -v flock >/dev/null 2>&1 || missing+=("flock")
    command -v timeout >/dev/null 2>&1 || missing+=("timeout")

    if [[ ${#missing[@]} -gt 0 ]]; then
        error_exit "Missing dependencies: ${missing[*]}"
    fi

    log "All dependencies satisfied"
}

# ============ MODEL SETUP WITH PROOF PACK OPTIMIZATIONS ============
setup_model() {
    local model_id="$1"
    local gpu_id="${2:-0}"
    local model_name
    model_name=$(sanitize_model_name "${model_id}")
    local basename_name
    basename_name=$(basename "${model_id}" | tr '[:upper:]' '[:lower:]' | tr '/' '_')
    local model_dir="${OUTPUT_DIR}/models/${model_name}"
    local basename_dir="${OUTPUT_DIR}/models/${basename_name}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setting up model (GPU ${gpu_id}): ${model_id}" >> "${LOG_FILE}"

    # Check if local path
    if [[ -d "${model_id}" ]]; then
        echo "${model_id}"
        return 0
    fi

    # Check if already downloaded (prefer sanitized path, but honor basename fallback)
    if [[ -d "${model_dir}/baseline" ]]; then
        echo "${model_dir}/baseline"
        return 0
    fi
    if [[ -d "${basename_dir}/baseline" ]]; then
        echo "${basename_dir}/baseline"
        return 0
    fi

    local revision=""
    revision=$(pack_model_revision "${model_id}" || true)
    if [[ -z "${revision}" ]]; then
        if [[ "${PACK_NET}" == "1" ]]; then
            error_exit "Missing pinned revision for ${model_id}; run preflight (--net 1)."
        else
            error_exit "Offline mode requires model revisions. Run with --net 1 to preflight."
        fi
    fi

    if [[ "${PACK_NET}" != "1" ]]; then
        echo "ERROR: Offline mode requested and baseline not cached for ${model_id}." >&2
        echo "       Run with --net 1 to populate the cache." >&2
        return 1
    fi

    # Download with proof pack optimizations
    mkdir -p "${model_dir}"

    local success_marker="${model_dir}/.download_success"
    rm -f "${success_marker}"

    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    {
        PACK_MODEL_REVISION="${revision}" CUDA_VISIBLE_DEVICES="${cuda_devices}" python3 << EOF
import json
import os
import sys
from pathlib import Path

model_id = "${model_id}"
output_dir = Path("${model_dir}/baseline")
success_marker = Path("${success_marker}")
flash_available = "${FLASH_ATTENTION_AVAILABLE}" == "true"
revision = os.environ.get("PACK_MODEL_REVISION") or None

# Storage strategy for the baseline directory:
# - snapshot_symlink (default): create baseline/ as symlinks to HF hub cache blobs (saves ~1× weights on disk)
# - snapshot_copy: copy snapshot files into baseline/ (duplicates hub cache)
# - save_pretrained: load with transformers and write a full baseline copy (also duplicates hub cache)
baseline_mode = os.environ.get("PACK_BASELINE_STORAGE_MODE", "snapshot_symlink").strip().lower()

output_dir.mkdir(parents=True, exist_ok=True)

rev_label = f"@{revision}" if revision else ""
print(f"Downloading {model_id}{rev_label} (proof pack optimized)...")
print(f"Baseline storage mode: {baseline_mode}")
if os.environ.get("HF_HUB_CACHE"):
    print(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}")
print(f"Flash Attention 2: {'enabled' if flash_available else 'disabled'}")


def model_supports_flash_attention(model_id: str) -> bool:
    no_fa2_models = [
        "falcon",
        "mpt-",
        "gpt2",
        "bloom",
        "opt-",
        "gpt-j",
        "gpt-neo",
        "codegen",
        "santacoder",
        "stablelm",
    ]
    model_lower = model_id.lower()
    return not any(pattern in model_lower for pattern in no_fa2_models)


def sanitize_generation_config(model_dir: Path) -> None:
    gen_path = model_dir / "generation_config.json"
    if not gen_path.is_file():
        return
    try:
        gen = json.loads(gen_path.read_text())
    except Exception:
        return

    if gen.get("do_sample") is False:
        temp = gen.get("temperature", None)
        if temp not in (None, 1.0):
            print(f"Fixing generation_config.json: clearing temperature={temp} (do_sample=False)")
            gen["temperature"] = None
        top_p = gen.get("top_p", None)
        if top_p not in (None, 1.0):
            print(f"Fixing generation_config.json: clearing top_p={top_p} (do_sample=False)")
            gen["top_p"] = None
        try:
            gen_path.write_text(json.dumps(gen, indent=2) + "\n")
        except Exception:
            pass


def write_model_profile(model_dir: Path, model_id: str) -> None:
    weights_bytes = 0
    for pat in ("*.safetensors", "*.bin"):
        for fp in model_dir.glob(pat):
            try:
                weights_bytes += fp.stat().st_size
            except OSError:
                pass

    cfg_path = model_dir / "config.json"
    config = {}
    if cfg_path.is_file():
        try:
            config = json.loads(cfg_path.read_text())
        except Exception:
            config = {}

    profile = {
        "model_id": model_id,
        "revision": revision,
        "weights_bytes": weights_bytes,
        "weights_gb": round(weights_bytes / (1024**3), 3),
        "hidden_size": config.get("hidden_size"),
        "num_layers": config.get("num_hidden_layers"),
        "num_heads": config.get("num_attention_heads"),
        "num_kv_heads": config.get("num_key_value_heads") or config.get("num_attention_heads"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "dtype_bytes": 2,
    }
    (model_dir / "model_profile.json").write_text(json.dumps(profile, indent=2) + "\n")


def download_snapshot(repo_id: str, model_dir: Path, mode: str) -> None:
    from huggingface_hub import snapshot_download

    local_dir_use_symlinks = mode == "snapshot_symlink"
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=local_dir_use_symlinks,
        cache_dir=os.environ.get("HF_HUB_CACHE"),
        resume_download=True,
        revision=revision,
    )


try:
    if baseline_mode in ("snapshot_symlink", "snapshot_copy"):
        try:
            download_snapshot(model_id, output_dir, baseline_mode)
            sanitize_generation_config(output_dir)
            write_model_profile(output_dir, model_id)
            success_marker.touch()
            print(f"Saved to {output_dir} (snapshot)")
            sys.exit(0)
        except Exception as snap_err:
            print(f"WARNING: snapshot_download failed, falling back to save_pretrained: {snap_err}", file=sys.stderr)
            baseline_mode = "save_pretrained"

    # Fallback: create a full baseline copy via transformers (duplicates HF hub cache).
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mode = os.environ.get("PACK_DETERMINISM", "throughput").strip().lower()
    if mode == "strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cache_dir = os.environ.get("HF_HUB_CACHE")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=cache_dir, revision=revision
    )
    tokenizer.save_pretrained(output_dir)

    use_fa2 = flash_available and model_supports_flash_attention(model_id)
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "cache_dir": cache_dir,
        "revision": revision,
    }
    if use_fa2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print(f"Using Flash Attention 2 for {model_id}")
    else:
        print(f"Using eager attention for {model_id} (FA2 not supported or unavailable)")

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
    if hasattr(model, "generation_config"):
        gen_config = model.generation_config
        if getattr(gen_config, "do_sample", True) is False:
            if getattr(gen_config, "temperature", 1.0) not in (None, 1.0):
                print(f"Fixing generation_config: clearing temperature={gen_config.temperature} (do_sample=False)")
                gen_config.temperature = None
            if getattr(gen_config, "top_p", 1.0) not in (None, 1.0):
                print(f"Fixing generation_config: clearing top_p={gen_config.top_p} (do_sample=False)")
                gen_config.top_p = None

    model.save_pretrained(output_dir, safe_serialization=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory.empty_cache()

    sanitize_generation_config(output_dir)
    write_model_profile(output_dir, model_id)
    success_marker.touch()
    print(f"Saved to {output_dir} (save_pretrained)")

except Exception as e:
    print(f"ERROR: Model download failed: {e}", file=sys.stderr)
    sys.exit(1)
EOF
    } 2>&1 | tee -a "${LOG_FILE}" >&2

	    if [[ ! -f "${success_marker}" ]]; then
	        # Output error to stderr (not stdout) and return empty string
	        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to download model: ${model_id}" >&2
	        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to download model: ${model_id}" >> "${LOG_FILE}"
	        # The Python downloader creates the baseline directory before downloading.
	        # If the download fails, remove the incomplete baseline dir so future runs
	        # don't treat it as a cached success.
	        rm -rf "${model_dir}/baseline" 2>/dev/null || true
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
# Based on model size and available GPU memory budget
get_model_invarlock_config() {
    local model_size="$1"  # 7, 13, 30, 40, 70, moe

    # WikiText-2 has ~1174 samples; defaults assume high-memory GPUs.
    # Format: seq_len:stride:preview_n:final_n:eval_batch
    case "${model_size}" in
        "7")
            # 7B models: ~14GB, can use longer sequences and more windows
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
            # 70-72B models: ~140-144GB, ultra-conservative settings
            # Keep headroom for baseline/edited overlap and overhead checks.
            # Settings chosen to avoid double-loading baseline and edited models during overhead checks:
            # - seq_len=128: Minimal KV cache
            # - stride=64: Maintains 50% overlap
            # - windows=8+8: Minimal window count
            # - eval_batch=2: Minimal batch to avoid OOM
            echo "128:64:8:8:2"
            ;;
        *)
            # Unknown - use safe defaults
            echo "1024:512:40:40:32"
            ;;
    esac
}
export -f get_model_invarlock_config

# GPU placement is handled by the dynamic scheduler (required_gpus + reservations).
# There is no fixed GPU→model mapping.

# ============ EDITED MODEL WITH GPU QUANTIZATION ============
create_edited_model() {
    local baseline_path="$1"
    local output_path="$2"
    local edit_type="$3"
    local bits="$4"
    local group_size="$5"
    local scope="$6"
    local gpu_id="${7:-0}"

    log "Creating edited model (GPU ${gpu_id}):"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"
    log "  Edit: ${edit_type} bits=${bits} group_size=${group_size} scope=${scope}"

    mkdir -p "$(dirname "${output_path}")"

    if [[ "${edit_type}" == "quant_rtn" ]]; then
        local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
        CUDA_VISIBLE_DEVICES="${cuda_devices}" python3 << EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import gc
import os
import sys

try:
    mode = os.environ.get("PACK_DETERMINISM", "throughput").strip().lower()
    if mode == "strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
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
    def round_to_nearest_gpu(tensor, bits, group_size):
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
            param.data = round_to_nearest_gpu(param.data, bits, group_size)
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

    log "Creating pruned model (GPU ${gpu_id}):"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"
    log "  Sparsity: ${sparsity}, Scope: ${scope}"

    mkdir -p "$(dirname "${output_path}")"

    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    CUDA_VISIBLE_DEVICES="${cuda_devices}" python3 << EOF
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
        device_map="auto",
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

    log "Creating low-rank model (GPU ${gpu_id}):"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"
    log "  Rank: ${rank}, Scope: ${scope}"

    mkdir -p "$(dirname "${output_path}")"

    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    CUDA_VISIBLE_DEVICES="${cuda_devices}" python3 << EOF
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
        device_map="auto",
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

        # Reconstruct: (U * S) @ V^T (avoid materializing diag(S))
        lowrank = (U * S) @ V.T
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

# ============ FP8 QUANTIZATION (SIMULATED) ============
create_fp8_model() {
    local baseline_path="$1"
    local output_path="$2"
    local format="$3"      # e4m3fn or e5m2
    local scope="$4"       # ffn, attn, all
    local gpu_id="${5:-0}"

    log "Creating FP8 model (GPU ${gpu_id}):"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"
    log "  Format: ${format}, Scope: ${scope}"

    mkdir -p "$(dirname "${output_path}")"

    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    CUDA_VISIBLE_DEVICES="${cuda_devices}" python3 << EOF
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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    print(f"Loading baseline from {baseline_path}...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        baseline_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    def should_quantize(name, scope):
        name_lower = name.lower()
        if scope == "all":
            return "weight" in name_lower and any(x in name_lower for x in [
                "linear", "proj", "fc", "mlp", "attn",
                "wqkv", "query_key_value"
            ])
        if scope == "ffn":
            return "weight" in name_lower and any(x in name_lower for x in [
                "mlp", "fc", "gate", "up_proj", "down_proj",
                "dense_h_to_4h", "dense_4h_to_h"
            ])
        if scope == "attn":
            return "weight" in name_lower and any(x in name_lower for x in [
                "attn", "q_proj", "k_proj", "v_proj", "o_proj",
                "wqkv", "out_proj", "query_key_value"
            ])
        return False

    if format_type in {"e4m3", "e4m3fn", "e4m3fnuz"}:
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    else:
        fp8_dtype = getattr(torch, "float8_e5m2", None)

    if fp8_dtype is None:
        print("WARNING: torch float8 dtype not available; falling back to float16 quantization")

    @torch.no_grad()
    def quantize_fp8(tensor):
        if fp8_dtype is None:
            return tensor.to(torch.float16).to(tensor.dtype)
        return tensor.to(fp8_dtype).to(tensor.dtype)

    print(f"Applying FP8 quantization (format={format_type}, scope={scope})...")
    quantized_count = 0
    num_tensors = 0
    rel_error_total = 0.0
    edited_params = 0
    total_model_params = sum(p.numel() for p in model.parameters())

    for name, param in model.named_parameters():
        if not should_quantize(name, scope) or param.dim() < 2:
            continue
        original = param.data.clone()
        param.data = quantize_fp8(param.data)
        num_tensors += 1
        quantized_count += 1
        edited_params += param.numel()
        denom = original.abs().mean() + 1e-10
        rel_error_total += float((param.data - original).abs().mean() / denom)
        if quantized_count <= 3:
            print(f"  FP8: {name}")

    avg_error = rel_error_total / max(num_tensors, 1)
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
        "edit_type": "fp8_quant",
        "format": format_type,
        "scope": scope,
        "quantized_tensors": quantized_count,
        "avg_relative_error": avg_error,
    }
    with open(output_path / "edit_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved FP8-quantized model to {output_path}")

except Exception as e:
    print(f"ERROR: Failed to create FP8 model: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
}
export -f create_fp8_model

# ============ FP4 QUANTIZATION (SIMULATED) ============
create_fp4_model() {
    local baseline_path="$1"
    local output_path="$2"
    local format="$3"      # e2m1 (standard) or aggressive
    local scope="$4"       # ffn, attn, all
    local gpu_id="${5:-0}"

    log "Creating FP4 model (GPU ${gpu_id}):"
    log "  Baseline: ${baseline_path}"
    log "  Output: ${output_path}"
    log "  Format: ${format}, Scope: ${scope}"
    log "  FP4 Native Support: ${FP4_NATIVE_SUPPORT}"

    mkdir -p "$(dirname "${output_path}")"

    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    CUDA_VISIBLE_DEVICES="${cuda_devices}" python3 << EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import gc
import os
import sys

try:
    baseline_path = Path("${baseline_path}")
    output_path = Path("${output_path}")
    format_type = "${format}"
    scope = "${scope}"

    # Check for native FP4 support (Blackwell-class GPUs)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device_name = torch.cuda.get_device_name(0)
    is_blackwell = "B200" in device_name or "Blackwell" in device_name

    require_native = os.environ.get("INVARLOCK_REQUIRE_FP4_NATIVE", "false").strip().lower() in ("1", "true", "yes")
    te_available = False
    try:
        import transformer_engine.pytorch as te  # noqa: F401
        te_available = True
    except Exception as e:
        if require_native:
            raise RuntimeError("TransformerEngine not available for native FP4 validation") from e

    fp4_native = bool(is_blackwell and te_available)

    if not fp4_native:
        if is_blackwell:
            print("WARNING: Blackwell-class GPU detected but TransformerEngine not available.")
            print("         FP4 quantization is simulated; no FP4 Tensor Core validation.")
        else:
            print(f"WARNING: FP4 native support is Blackwell-class; current GPU: {device_name}")
            print("         FP4 quantization is simulated; results may not match native FP4 behavior")

    print(f"Loading baseline from {baseline_path}...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        baseline_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
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
            levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=tensor.device, dtype=torch.float16)
        else:
            # Aggressive: tighter range for stress testing
            levels = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], device=tensor.device, dtype=torch.float16)

        # Memory-safe nearest-level quantization via bucketize (no NxK diff matrix).
        max_val = float(levels[-1].item())
        scale = tensor.abs().amax().float() / max_val
        scale = torch.clamp(scale, min=1e-10).to(device=tensor.device, dtype=torch.float16)

        thresholds = ((levels[:-1] + levels[1:]) / 2).to(device=tensor.device)

        flat = tensor.view(-1)
        n = flat.numel()
        chunk_elems = 5_000_000  # Bound peak temp memory (idx is int64)
        for start in range(0, n, chunk_elems):
            end = min(start + chunk_elems, n)
            chunk = flat[start:end]
            scaled = chunk.to(torch.float16) / scale
            idx = torch.bucketize(scaled.abs(), thresholds)
            q = levels[idx] * scaled.sign()
            chunk.copy_((q * scale).to(chunk.dtype))

        return tensor

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
        "pack_native": is_b200,
        "fp4_native": fp4_native,
        "transformer_engine": te_available
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

    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    CUDA_VISIBLE_DEVICES="${cuda_devices}" python3 << EOF
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
            device_map="auto",
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

    local resolved
    resolved=$(resolve_edit_params "${output_dir}" "${edit_spec}" "${version}")
    local status
    status=$(echo "${resolved}" | jq -r '.status')
    if [[ "${status}" == "skipped" ]]; then
        log "  Clean edit skipped by calibration: ${edit_spec}"
        return 0
    fi
    if [[ "${status}" != "selected" ]]; then
        log "  ERROR: Unable to resolve edit spec (${edit_spec}): ${status}"
        return 1
    fi

    local edit_type param1 param2 scope edit_dir_name
    edit_type=$(echo "${resolved}" | jq -r '.edit_type')
    param1=$(echo "${resolved}" | jq -r '.param1')
    param2=$(echo "${resolved}" | jq -r '.param2')
    scope=$(echo "${resolved}" | jq -r '.scope')
    edit_dir_name=$(echo "${resolved}" | jq -r '.edit_dir_name')

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
        "fp8_quant")
            # 3-part: type:format:scope -> param1=format, scope=scope
            create_fp8_model "${baseline_path}" "${edit_path}" "${param1}" "${scope}" "${gpu_id}" || create_result=$?
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

    # Use auto adapter for generic causal LM support (LLaMA, Mistral, Qwen, MPT, Falcon, etc.)
    local adapter="hf_causal_auto"
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
        guards_order=("invariants" "variance" "invariants")
    fi
    local guards_order_yaml=""
    local g
    for g in "${guards_order[@]}"; do
        g="$(echo "${g}" | xargs)"
        [[ -z "${g}" ]] && continue
        guards_order_yaml+=$'    - '"${g}"$'\n'
    done
    if [[ -z "${guards_order_yaml}" ]]; then
        guards_order_yaml=$'    - invariants\n    - variance\n    - invariants\n'
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

    # Force no-overlap calibration to avoid pairing mismatches
    python3 - "${config_yaml}" <<'PY'
import sys, yaml, pathlib
path = pathlib.Path(sys.argv[1])
cfg = yaml.safe_load(path.read_text())
cfg.setdefault('eval', {})['window_overlap_fraction'] = 0.0
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

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
    local bootstrap_n="${INVARLOCK_BOOTSTRAP_N:-2000}"

    # Get model-size-aware configuration
    local config=$(get_model_invarlock_config "${model_size}")
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
    python3 << CALIBRATION_SCRIPT
import json
import math
import statistics
from pathlib import Path
from collections import defaultdict

try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False

output_dir = Path("${output_dir}")
preset_output_dir = Path("${preset_output_dir}")
model_name = "${model_name}"
model_path = "${model_path}"
tier = "${INVARLOCK_TIER}".strip().lower()
dataset_provider = "${INVARLOCK_DATASET}"
seq_len = int("${effective_seq_len}")
stride = int("${effective_stride}")
preview_n = int("${effective_preview_n}")
final_n = int("${effective_final_n}")

guards_order = None
assurance_cfg = None
if YAML_AVAILABLE:
    cfg_path = None
    for candidate in sorted(output_dir.glob("run_*/calibration_config.yaml")):
        cfg_path = candidate
        break
    if cfg_path is not None:
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
            if isinstance(cfg, dict):
                guards_block = cfg.get("guards") or {}
                if isinstance(guards_block, dict):
                    order = guards_block.get("order")
                    if isinstance(order, list) and order:
                        guards_order = [str(item) for item in order]
                ab = cfg.get("assurance")
                if isinstance(ab, dict) and ab:
                    assurance_cfg = ab
        except Exception:
            guards_order = None

if guards_order is None:
    guards_order = ["invariants", "variance", "invariants"]

enabled_guards = set(guards_order)

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

def _merge_record(cert, report):
    rec = {}
    if isinstance(cert, dict):
        rec = json.loads(json.dumps(cert))
    if not isinstance(report, dict):
        return rec or None

    # Primary metric from report when cert is missing it.
    metrics = report.get("metrics", {}) or {}
    pm = metrics.get("primary_metric", {}) or {}
    if not pm and "ppl_final" in metrics:
        pm = {
            "final": metrics.get("ppl_final"),
            "preview": metrics.get("ppl_preview"),
        }
        try:
            pm["ratio_vs_baseline"] = float(pm["final"]) / max(float(pm["preview"]), 1e-10)
        except Exception:
            pass
    if pm and not rec.get("primary_metric"):
        rec["primary_metric"] = pm

    guards = report.get("guards", []) or []
    for guard in guards:
        if not isinstance(guard, dict):
            continue
        name = str(guard.get("name", "")).lower()
        gmetrics = guard.get("metrics", {}) or {}
        gpolicy = guard.get("policy", {}) or {}

        if name == "spectral":
            spec = rec.get("spectral", {}) if isinstance(rec.get("spectral"), dict) else {}
            if gmetrics.get("family_z_quantiles"):
                spec.setdefault("family_z_quantiles", gmetrics.get("family_z_quantiles"))
            if gmetrics.get("family_z_summary"):
                spec.setdefault("family_z_summary", gmetrics.get("family_z_summary"))
            if gmetrics.get("family_caps"):
                spec.setdefault("family_caps", gmetrics.get("family_caps"))
            if gmetrics.get("sigma_quantile") is not None:
                spec.setdefault("sigma_quantile", gmetrics.get("sigma_quantile"))
            if gmetrics.get("deadband") is not None:
                spec.setdefault("deadband", gmetrics.get("deadband"))
            if gmetrics.get("max_caps") is not None:
                spec.setdefault("max_caps", gmetrics.get("max_caps"))
            if gmetrics.get("families"):
                spec.setdefault("families", gmetrics.get("families"))
            if gmetrics.get("family_stats"):
                spec.setdefault("families", gmetrics.get("family_stats"))
            z_scores = guard.get("final_z_scores") or gmetrics.get("final_z_scores")
            if isinstance(z_scores, dict):
                spec["final_z_scores"] = z_scores
            fam_map = guard.get("module_family_map") or gmetrics.get("module_family_map")
            if isinstance(fam_map, dict):
                spec["module_family_map"] = fam_map
            if gpolicy and not spec.get("policy"):
                spec["policy"] = gpolicy
            rec["spectral"] = spec

        elif name == "rmt":
            rmt = rec.get("rmt", {}) if isinstance(rec.get("rmt"), dict) else {}
            for key in ("outliers_per_family", "baseline_outliers_per_family", "families"):
                val = gmetrics.get(key)
                if isinstance(val, dict) and val:
                    rmt.setdefault(key, val)
            epsilon_by_family = gmetrics.get("epsilon_by_family")
            if epsilon_by_family:
                rmt.setdefault("epsilon_by_family", epsilon_by_family)
            else:
                epsilon = gmetrics.get("epsilon")
                if epsilon is not None:
                    if isinstance(epsilon, dict):
                        rmt.setdefault("epsilon_by_family", epsilon)
                    else:
                        rmt.setdefault("epsilon_default", epsilon)
            if gmetrics.get("epsilon_default") is not None:
                rmt.setdefault("epsilon_default", gmetrics.get("epsilon_default"))
            if gmetrics.get("margin_used") is not None:
                rmt.setdefault("margin", gmetrics.get("margin_used"))
            if gmetrics.get("deadband_used") is not None:
                rmt.setdefault("deadband", gmetrics.get("deadband_used"))
            if gpolicy and not rmt.get("policy"):
                rmt["policy"] = gpolicy
            rec["rmt"] = rmt

        elif name == "variance":
            var = rec.get("variance", {}) if isinstance(rec.get("variance"), dict) else {}
            if gmetrics.get("predictive_gate") is not None:
                var.setdefault("predictive_gate", gmetrics.get("predictive_gate"))
            if gmetrics.get("ab_windows_used") is not None:
                var.setdefault("ab_windows_used", gmetrics.get("ab_windows_used"))
            if gmetrics.get("deadband") is not None:
                var.setdefault("deadband", gmetrics.get("deadband"))
            if gmetrics.get("min_gain") is not None:
                var.setdefault("min_gain", gmetrics.get("min_gain"))
            if gmetrics.get("min_effect_lognll") is not None:
                var.setdefault("min_effect_lognll", gmetrics.get("min_effect_lognll"))
            if gmetrics.get("calibration") is not None:
                var.setdefault("calibration", gmetrics.get("calibration"))
            if gmetrics.get("calibration_stats") is not None:
                var.setdefault("calibration_stats", gmetrics.get("calibration_stats"))
            if gpolicy and not var.get("policy"):
                var["policy"] = gpolicy
            rec["variance"] = var

    return rec or None

def load_records():
    records = []
    for run_dir in sorted(output_dir.glob("run_*")):
        cert = None
        report = None
        cert_path = run_dir / "evaluation.cert.json"
        if cert_path.exists():
            try:
                cert = json.loads(cert_path.read_text())
            except Exception:
                cert = None
        report_path = run_dir / "baseline_report.json"
        if not report_path.exists():
            report_files = list(run_dir.glob("**/report*.json"))
            if report_files:
                report_path = report_files[0]
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
            except Exception:
                report = None

        record = _merge_record(cert, report)
        if record:
            records.append(record)
    return records

records = load_records()
if len(records) == 0:
    print("ERROR: No calibration records found - cannot create valid preset")
    import sys
    sys.exit(1)
if len(records) < 2:
    print(f"WARNING: Only {len(records)} calibration record(s) found (expected >= 2)")

def calibrate_drift(recs):
    try:
        ratios = []
        for rec in recs:
            pm = rec.get("primary_metric", {}) or {}
            ratio = pm.get("ratio_vs_baseline") or pm.get("drift")
            if ratio is None:
                preview = pm.get("preview")
                final = pm.get("final")
                if preview is not None and final is not None:
                    try:
                        ratio = float(final) / max(float(preview), 1e-10)
                    except Exception:
                        ratio = None
            if ratio is not None:
                try:
                    ratios.append(float(ratio))
                except Exception:
                    pass

        ratios = [r for r in ratios if math.isfinite(r)]
        if len(ratios) < 2:
            base = ratios[0] if ratios else 1.0
            return {
                "mean": float(base),
                "std": 0.0,
                "min": float(base),
                "max": float(base),
                "suggested_band": [0.95, 1.05],
                "band_compatible": True,
            }

        try:
            mean = sum(ratios) / len(ratios)
        except Exception:
            mean = 1.0
        try:
            var = sum((r - mean) ** 2 for r in ratios) / max(len(ratios), 1)
            std = math.sqrt(var) if math.isfinite(var) else 0.0
        except Exception:
            std = 0.0
        margin = max(2 * std, 0.05)
        band = [round(mean - margin, 3), round(mean + margin, 3)]
        return {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(min(ratios), 4),
            "max": round(max(ratios), 4),
            "suggested_band": band,
            "band_compatible": 0.95 <= mean <= 1.05,
        }
    except Exception as e:
        print(f"ERROR: failed to compute drift stats: {e}")
        return {
            "mean": 1.0,
            "std": 0.0,
            "min": 1.0,
            "max": 1.0,
            "suggested_band": [0.95, 1.05],
            "band_compatible": True,
        }

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

def calibrate_spectral(recs):
    per_run_caps = defaultdict(list)
    q99_values = defaultdict(list)
    max_values = defaultdict(list)
    existing_caps = {}
    sigma_quantile = None
    deadband = None
    max_caps = None

    for rec in recs:
        spec = rec.get("spectral", {}) or {}
        if not isinstance(spec, dict):
            continue
        policy = spec.get("policy", {}) if isinstance(spec.get("policy"), dict) else {}

        if sigma_quantile is None:
            sq = (
                policy.get("sigma_quantile")
                or policy.get("contraction")
                or policy.get("kappa")
                or spec.get("sigma_quantile")
                or (spec.get("summary") or {}).get("sigma_quantile")
            )
            sq = _safe_float(sq)
            if sq is not None:
                sigma_quantile = sq

        if deadband is None:
            db = policy.get("deadband") or spec.get("deadband") or (spec.get("summary") or {}).get("deadband")
            db = _safe_float(db)
            if db is not None:
                deadband = db

        if max_caps is None:
            mc = policy.get("max_caps") or spec.get("max_caps") or (spec.get("summary") or {}).get("max_caps")
            try:
                if mc is not None:
                    max_caps = int(mc)
            except Exception:
                pass

        fam_caps = spec.get("family_caps", {})
        if not fam_caps and isinstance(policy.get("family_caps"), dict):
            fam_caps = policy.get("family_caps", {})
        if isinstance(fam_caps, dict):
            for fam, cap in fam_caps.items():
                try:
                    if isinstance(cap, dict):
                        cap = cap.get("kappa")
                    existing_caps[str(fam)] = float(cap)
                except Exception:
                    pass

        z_map = spec.get("final_z_scores")
        fam_map = spec.get("module_family_map")
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

        fq = spec.get("family_z_quantiles", {})
        if not fq and isinstance(spec.get("family_z_summary"), dict):
            fq = spec.get("family_z_summary", {})
        if isinstance(fq, dict):
            for fam, stats in fq.items():
                if not isinstance(stats, dict):
                    continue
                val_q99 = _safe_float(stats.get("q99"))
                val_max = _safe_float(stats.get("max"))
                if val_q99 is not None:
                    q99_values[str(fam)].append(val_q99)
                if val_max is not None:
                    max_values[str(fam)].append(val_max)

    summary = {
        "families_seen": sorted(set(per_run_caps) | set(q99_values) | set(existing_caps)),
        "sigma_quantile": sigma_quantile,
        "deadband": deadband,
        "max_caps": max_caps,
    }

    proposed_caps = {}
    margin = _spectral_margin(tier)
    if per_run_caps:
        for fam, candidates in per_run_caps.items():
            if not candidates:
                continue
            base = max(candidates)
            if fam in existing_caps:
                base = max(base, existing_caps[fam])
            proposed_caps[fam] = {"kappa": round(base + margin, 3)}
        for fam in sorted(set(q99_values) | set(max_values)):
            if fam in proposed_caps:
                continue
            observed = q99_values.get(fam, []) + max_values.get(fam, [])
            if not observed:
                continue
            base = max(observed)
            if fam in existing_caps:
                base = max(base, existing_caps[fam])
            proposed_caps[fam] = {"kappa": round(base + margin, 3)}
    elif q99_values or max_values:
        for fam in sorted(set(q99_values) | set(max_values)):
            observed = q99_values.get(fam, []) + max_values.get(fam, [])
            if not observed:
                continue
            base = max(observed)
            if fam in existing_caps:
                base = max(base, existing_caps[fam])
            proposed_caps[fam] = {"kappa": round(base + margin, 3)}
    else:
        for fam, kappa in existing_caps.items():
            proposed_caps[fam] = {"kappa": kappa}

    return summary, proposed_caps

def _rmt_quantile_for_tier(tier_name):
    if tier_name == "conservative":
        return 0.95
    if tier_name == "aggressive":
        return 0.99
    return 0.97

def calibrate_rmt(recs):
    deltas_by_family = defaultdict(list)
    existing_eps = {}
    margin = None
    deadband = None

    for rec in recs:
        rmt = rec.get("rmt", {}) or {}
        if not isinstance(rmt, dict):
            continue
        policy = rmt.get("policy", {}) if isinstance(rmt.get("policy"), dict) else {}

        if margin is None:
            margin = _safe_float(policy.get("margin") or rmt.get("margin") or (rmt.get("summary") or {}).get("margin"))
        if deadband is None:
            deadband = _safe_float(policy.get("deadband") or rmt.get("deadband") or (rmt.get("summary") or {}).get("deadband"))

        eps = (
            rmt.get("epsilon_by_family")
            or rmt.get("epsilon")
            or policy.get("epsilon_by_family")
            or policy.get("epsilon")
        )
        if isinstance(eps, dict):
            for fam, val in eps.items():
                try:
                    existing_eps[str(fam)] = float(val)
                except Exception:
                    pass
        elif isinstance(eps, (int, float)):
            existing_eps["_default"] = float(eps)

        record_has_counts = False
        families = rmt.get("families", {})
        if isinstance(families, dict) and families:
            record_has_counts = True
            for fam, stats in families.items():
                if not isinstance(stats, dict):
                    continue
                bare = stats.get("bare")
                guarded = stats.get("guarded")
                bare_f = _safe_float(bare)
                guarded_f = _safe_float(guarded)
                if bare_f and bare_f > 0:
                    deltas_by_family[str(fam)].append((guarded_f / bare_f) - 1.0)

        outliers = rmt.get("outliers_per_family", {})
        baseline_outliers = rmt.get("baseline_outliers_per_family", {})
        if isinstance(outliers, dict) and isinstance(baseline_outliers, dict) and outliers:
            record_has_counts = True
            for fam in set(outliers) | set(baseline_outliers):
                bare_f = _safe_float(baseline_outliers.get(fam))
                guarded_f = _safe_float(outliers.get(fam))
                if bare_f and bare_f > 0:
                    deltas_by_family[str(fam)].append((guarded_f / bare_f) - 1.0)

        if not record_has_counts:
            for source in ("outliers_by_family", "family_stats"):
                stats_map = rmt.get(source, {})
                if not isinstance(stats_map, dict):
                    continue
                for fam, stats in stats_map.items():
                    if not isinstance(stats, dict):
                        continue
                    for key in ("outlier_fraction", "outlier_rate", "fraction", "rate"):
                        val = _safe_float(stats.get(key))
                        if val is not None:
                            deltas_by_family[str(fam)].append(val)
                            break

    summary = {"families_seen": sorted(deltas_by_family.keys()), "margin": margin, "deadband": deadband}
    quantile_q = _rmt_quantile_for_tier(tier)
    proposed_eps = {}
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
                return summary, {"ffn": default_eps, "attn": default_eps, "embed": default_eps, "other": default_eps}
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

def calibrate_variance(recs):
    deadband = None
    min_gain = None
    policy_min_effect = None
    min_effect_samples = []
    variance_changes = []

    for rec in recs:
        var = rec.get("variance", {}) or {}
        if not isinstance(var, dict):
            continue
        policy = var.get("policy", {}) if isinstance(var.get("policy"), dict) else {}

        if deadband is None:
            deadband = _safe_float(policy.get("deadband") or var.get("deadband"))
        if min_gain is None:
            min_gain = _safe_float(policy.get("min_gain") or policy.get("min_rel_gain") or var.get("min_gain"))
        if policy_min_effect is None:
            policy_min_effect = _safe_float(policy.get("min_effect_lognll") or var.get("min_effect_lognll"))

        predictive = var.get("predictive_gate", {}) or {}
        delta_ci = predictive.get("delta_ci")
        if isinstance(delta_ci, (list, tuple)) and len(delta_ci) == 2:
            lo = _safe_float(delta_ci[0])
            hi = _safe_float(delta_ci[1])
            if lo is not None and hi is not None:
                width = abs(hi - lo) / 2.0
                if width > 0:
                    min_effect_samples.append(width)

        calib = var.get("calibration") or var.get("calibration_stats") or {}
        if isinstance(calib, dict):
            vchange = calib.get("variance_change") or calib.get("delta") or calib.get("max_delta")
            vchange = _safe_float(vchange)
            if vchange is not None:
                variance_changes.append(abs(vchange))

    result = {}
    if deadband is None and variance_changes:
        result["deadband"] = round(max(variance_changes) * 1.1 + 0.01, 3)
    elif deadband is not None:
        result["deadband"] = deadband

    if min_effect_samples:
        proposed = _quantile(min_effect_samples, 0.95)
        if proposed is not None:
            result["min_effect_lognll"] = max(round(proposed, 4), 0.0009)
    elif policy_min_effect is not None:
        result["min_effect_lognll"] = policy_min_effect

    if min_gain is not None:
        result["min_gain"] = min_gain

    return result

drift_stats = calibrate_drift(records)
spectral_summary, spectral_caps = calibrate_spectral(records)
rmt_summary, rmt_epsilon = calibrate_rmt(records)
variance_config = calibrate_variance(records)

preset = {
    "_calibration_meta": {
        "model_name": model_name,
        "tier": tier,
        "platform": "PACK_180GB",
        "drift_mean": drift_stats.get("mean"),
        "drift_std": drift_stats.get("std"),
        "drift_band_compatible": drift_stats.get("band_compatible"),
        "suggested_drift_band": drift_stats.get("suggested_band"),
    },
    "model": {"id": model_path},
    "dataset": {
        "provider": dataset_provider,
        "split": "validation",
        "seq_len": seq_len,
        "stride": stride,
        "preview_n": preview_n,
        "final_n": final_n,
        "seed": 42,
    },
    "guards": {"order": guards_order},
}

if isinstance(assurance_cfg, dict) and assurance_cfg:
    preset["assurance"] = assurance_cfg

spectral = {}
if spectral_caps:
    spectral["family_caps"] = spectral_caps
if spectral_summary.get("sigma_quantile") is not None:
    spectral["sigma_quantile"] = spectral_summary["sigma_quantile"]
if spectral_summary.get("deadband") is not None:
    spectral["deadband"] = spectral_summary["deadband"]
if spectral_summary.get("max_caps") is not None:
    spectral["max_caps"] = spectral_summary["max_caps"]
if "spectral" in enabled_guards and spectral:
    preset["guards"]["spectral"] = spectral

rmt = {}
if rmt_epsilon:
    rmt["epsilon_by_family"] = rmt_epsilon
if rmt_summary.get("margin") is not None:
    rmt["margin"] = rmt_summary["margin"]
if rmt_summary.get("deadband") is not None:
    rmt["deadband"] = rmt_summary["deadband"]
if "rmt" in enabled_guards and rmt:
    preset["guards"]["rmt"] = rmt

if "variance" in enabled_guards and variance_config:
    preset["guards"]["variance"] = variance_config

stats_path = output_dir / "calibration_stats.json"
with open(stats_path, "w") as f:
    json.dump(
        {
            "guards_order": guards_order,
            "assurance": assurance_cfg,
            "drift": drift_stats,
            "spectral": {**spectral_summary, "family_caps": spectral_caps},
            "rmt": {**rmt_summary, "epsilon_by_family": rmt_epsilon},
            "variance": variance_config,
        },
        f,
        indent=2,
    )

preset_path = preset_output_dir / f"calibrated_preset_{model_name.replace('/', '_')}.yaml"
if YAML_AVAILABLE:
    with open(preset_path, "w") as f:
        yaml.safe_dump(preset, f, sort_keys=False)
else:
    preset_path = preset_path.with_suffix(".json")
    with open(preset_path, "w") as f:
        json.dump(preset, f, indent=2)

print(f"Saved: {stats_path}")
print(f"Saved: {preset_path}")
CALIBRATION_SCRIPT
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

# ============ PROCESS MODEL ON SINGLE GPU ============
process_model() {
    local model_id="$1"
    local gpu_id="${2:-0}"

    local model_name
    model_name=$(sanitize_model_name "${model_id}")
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

    # Step 2.5: Clean edit calibration (lm-eval only)
    if [[ "${CALIBRATE_CLEAN_EDITS:-true}" == "true" && ${CLEAN_EDIT_RUNS:-0} -gt 0 ]]; then
        if type task_calibrate_clean_edits &>/dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Calibrating clean edits..." >> "${gpu_log}"
            task_calibrate_clean_edits "${model_name}" "${gpu_id}" "${OUTPUT_DIR}" "${gpu_log}"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Clean calibration unavailable; skipping" >> "${gpu_log}"
        fi
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

    # Step 4: Clean edits (only when requested)
    if [[ ${CLEAN_EDIT_RUNS:-0} -gt 0 ]]; then
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
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Skipping clean edits (CLEAN_EDIT_RUNS=0)" >> "${gpu_log}"
    fi

    # Step 5: Stress edits (only when requested)
    if [[ ${STRESS_EDIT_RUNS:-0} -gt 0 ]]; then
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
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id}: Skipping stress edits (STRESS_EDIT_RUNS=0)" >> "${gpu_log}"
    fi

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

	    python3 <<- EOF
import json
import csv
import math
from pathlib import Path
from collections import defaultdict

output_dir = Path("${OUTPUT_DIR}")
analysis_dir = output_dir / "analysis"
analysis_dir.mkdir(exist_ok=True)

skip_dirs = {
    "logs",
    "analysis",
    "reports",
    "presets",
    "models",
    "queue",
    "workers",
    "state",
    "evals",
    "certificates",
}

def _is_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name in skip_dirs:
        return False
    if not (path / ".baseline_path").exists():
        return False
    return True


def _pick_metric(task_results: dict):
    for key in (
        "acc_norm,none",
        "acc,none",
        "exact_match,none",
        "acc_norm",
        "acc",
        "exact_match",
    ):
        if key in task_results and isinstance(task_results[key], (int, float)):
            return key, float(task_results[key])
    for key, value in task_results.items():
        if "stderr" in key:
            continue
        if isinstance(value, (int, float)):
            return key, float(value)
    return None, None

# Collect eval results
eval_rows = []
for model_dir in output_dir.iterdir():
    if not _is_model_dir(model_dir):
        continue

    evals_dir = model_dir / "evals"
    if not evals_dir.exists():
        continue

    benchmarks = {"mmlu", "hellaswag", "arc", "winogrande"}
    for results_file in evals_dir.glob("*_results.json"):
        stem = results_file.stem
        base = stem[:-len("_results")] if stem.endswith("_results") else stem
        parts = base.split("_")
        if parts and parts[-1] in benchmarks:
            # Split-eval output: {edit}_{benchmark}_results.json
            edit_type = "_".join(parts[:-1])
        else:
            edit_type = base
        try:
            data = json.loads(results_file.read_text())
            for task, task_results in data.get('results', {}).items():
                if not isinstance(task_results, dict):
                    continue
                metric_key, metric_val = _pick_metric(task_results)
                if metric_key is None:
                    continue
                eval_rows.append({
                    'model': model_dir.name,
                    'edit_type': edit_type,
                    'task': task,
                    'metric': metric_key,
                    'value': metric_val,
                })
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
    if not _is_model_dir(model_dir):
        continue

    certs_dir = model_dir / "certificates"
    if not certs_dir.exists():
        continue

    for cert_file in certs_dir.rglob("evaluation.cert.json"):
        try:
            cert = json.loads(cert_file.read_text())
            rel_path = cert_file.relative_to(certs_dir)
            parts = list(rel_path.parts)

            v = cert.get('validation', {}) or {}
            def as_bool(val):
                if val is None:
                    return False
                if isinstance(val, bool):
                    return val
                if isinstance(val, str):
                    return val.strip().lower() in ('true', '1', 'yes', 'on')
                return bool(val)

            invariants_ok = as_bool(v.get('invariants_pass', False))
            pm_ok = as_bool(v.get('primary_metric_acceptable', False))
            spectral_ok = as_bool(v.get('spectral_stable', False))
            rmt_ok = as_bool(v.get('rmt_stable', False))
            drift_ok = as_bool(v.get('preview_final_drift_acceptable', True))
            hyst_applied = as_bool(v.get('hysteresis_applied', False))

            guard_overhead = cert.get('guard_overhead') or {}
            guard_evaluated = bool(guard_overhead.get('evaluated')) if isinstance(guard_overhead, dict) else False
            overhead_ok = as_bool(v.get('guard_overhead_acceptable', True))

            pm_block = cert.get('primary_metric') or {}
            pm_degraded = as_bool(pm_block.get('degraded')) or as_bool(pm_block.get('invalid'))
            pm_degraded_reason = pm_block.get('degraded_reason')

            # Backwards-compatible "all_pass" used by existing analysis logic.
            all_pass = all([invariants_ok, pm_ok, spectral_ok, rmt_ok]) and not pm_degraded

            # Canonical overall pass aligned with InvarLock console validation block.
            overall_pass = all([invariants_ok, pm_ok, spectral_ok, rmt_ok, drift_ok]) and not pm_degraded
            if guard_evaluated:
                overall_pass = overall_pass and overhead_ok

            conf = cert.get('confidence') or {}
            conf_label = conf.get('label') if isinstance(conf, dict) else None
            conf_label = str(conf_label).strip() if conf_label is not None else ''
            conf_label = conf_label if conf_label else 'Unknown'

            pm = pm_block
            pm_ratio = pm.get('ratio_vs_baseline') if isinstance(pm, dict) else None
            pm_ci_lo = None
            pm_ci_hi = None
            try:
                dci = pm.get('display_ci') if isinstance(pm, dict) else None
                if isinstance(dci, (list, tuple)) and len(dci) == 2:
                    pm_ci_lo = float(dci[0])
                    pm_ci_hi = float(dci[1])
                    if not (math.isfinite(pm_ci_lo) and math.isfinite(pm_ci_hi)):
                        pm_ci_lo = None
                        pm_ci_hi = None
            except Exception:
                pm_ci_lo = None
                pm_ci_hi = None

            # Estimate the effective PM threshold used by the tier policy.
            tier = ''
            try:
                pd_try = cert.get('policy_digest') or {}
                auto_try = cert.get('auto') or {}
                tier = str(pd_try.get('tier_policy_name') or auto_try.get('tier') or '').strip().lower()
            except Exception:
                tier = ''

            pm_threshold = None
            try:
                pol = cert.get('resolved_policy') or {}
                metrics_pol = pol.get('metrics', {}) if isinstance(pol, dict) else {}
                pm_pol = metrics_pol.get('pm_ratio', {}) if isinstance(metrics_pol, dict) else {}
                base = pm_pol.get('ratio_limit_base')
                hyst = pm_pol.get('hysteresis_ratio', 0.0)
                if base is not None:
                    pm_threshold = float(base) + float(hyst or 0.0)
            except Exception:
                pm_threshold = None
            if pm_threshold is None:
                tier_thresholds = {'conservative': 1.05, 'balanced': 1.10, 'aggressive': 1.20}
                base = tier_thresholds.get(tier, 1.10)
                pm_threshold = float(base) + 0.002

            # "Clear" PM failure if CI lower bound is above threshold, or ratio is far above.
            pm_clear_fail = False
            pm_far_fail = False
            pm_far_margin = 0.03  # absolute ratio margin above threshold
            try:
                if pm_ci_lo is not None and pm_threshold is not None:
                    pm_clear_fail = float(pm_ci_lo) > float(pm_threshold)
            except Exception:
                pm_clear_fail = False
            try:
                if isinstance(pm_ratio, (int, float)) and math.isfinite(float(pm_ratio)):
                    pm_far_fail = float(pm_ratio) > (float(pm_threshold) + float(pm_far_margin))
            except Exception:
                pm_far_fail = False

            # Derive degradation if fields are missing but PM is non-finite
            try:
                prev_val = pm.get('preview')
                fin_val = pm.get('final')
                ratio_val = pm_ratio
                def _nonfinite(v):
                    try:
                        return not (isinstance(v, (int, float)) and math.isfinite(float(v)))
                    except Exception:
                        return True
                if not pm_degraded and (_nonfinite(prev_val) or _nonfinite(fin_val) or _nonfinite(ratio_val)):
                    pm_degraded = True
                    pm_degraded_reason = pm_degraded_reason or 'non_finite_pm'
            except Exception:
                pass

            # Triage layer (PASS/REVIEW/FAIL) for shadow-mode style workflows.
            triage_reasons = []
            if pm_degraded:
                triage_reasons.append('primary_metric_degraded')
            if not invariants_ok:
                triage_reasons.append('invariants_fail')
            if not spectral_ok:
                triage_reasons.append('spectral_fail')
            if not rmt_ok:
                triage_reasons.append('rmt_fail')
            if not pm_ok:
                triage_reasons.append('primary_metric_fail')
            if not drift_ok:
                triage_reasons.append('drift_fail')
            if guard_evaluated and not overhead_ok:
                triage_reasons.append('overhead_fail')
            if hyst_applied:
                triage_reasons.append('hysteresis_applied')
            if conf_label != 'High':
                triage_reasons.append(f'confidence_{conf_label.lower()}')

            triage = 'REVIEW'
            if pm_degraded or (not invariants_ok) or (not spectral_ok) or (not rmt_ok):
                triage = 'FAIL'
            elif (not pm_ok) and (pm_clear_fail or pm_far_fail):
                triage = 'FAIL'
                triage_reasons.append('primary_metric_clear' if pm_clear_fail else 'primary_metric_far')
            elif overall_pass and conf_label == 'High' and not hyst_applied:
                triage = 'PASS'
                triage_reasons = []

            triage_reason = 'strict_pass' if triage == 'PASS' else ('|'.join(triage_reasons) if triage_reasons else 'unspecified')

            pd = cert.get('policy_digest') or {}
            meta = cert.get('meta') or {}
            det = meta.get('determinism') or {}

            invar_rows.append({
                'model': model_dir.name,
                'experiment': parts[0] if parts else 'unknown',
                'run': parts[1] if len(parts) > 1 else '',
                'edit_type': parts[0] if parts else 'unknown',
                'pm_ratio': pm_ratio,
                'pm_ci_low': pm_ci_lo,
                'pm_ci_high': pm_ci_hi,
                'pm_threshold': pm_threshold,
                'pm_acceptable': v.get('primary_metric_acceptable'),
                'pm_degraded': pm_degraded,
                'pm_degraded_reason': pm_degraded_reason,
                'preview_final_drift_acceptable': v.get('preview_final_drift_acceptable'),
                'invariants_pass': v.get('invariants_pass'),
                'spectral_stable': v.get('spectral_stable'),
                'rmt_stable': v.get('rmt_stable'),
                'all_pass': all_pass,
                'overall_pass': overall_pass,
                'hysteresis_applied': v.get('hysteresis_applied'),
                'guard_overhead_acceptable': v.get('guard_overhead_acceptable'),
                'confidence_label': conf_label,
                'triage': triage,
                'triage_reason': triage_reason,
                'policy_digest_hash': pd.get('thresholds_hash'),
                'policy_digest_changed': pd.get('changed'),
                'determinism_level': det.get('level'),
                'determinism_profile': det.get('profile'),
                'determinism_requested': det.get('requested'),
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

# Policy digest summary (per model)
policy_summary: dict[str, dict[str, object]] = {}
for row in invar_rows:
    model = row.get('model', 'unknown')
    digest = row.get('policy_digest_hash')
    changed = str(row.get('policy_digest_changed')).lower() == 'true'
    entry = policy_summary.setdefault(
        model,
        {
            'thresholds_hashes': set(),
            'policy_changed_true': 0,
            'total_certs': 0,
        },
    )
    entry['total_certs'] = int(entry.get('total_certs', 0)) + 1
    if digest:
        hashes = entry.setdefault('thresholds_hashes', set())
        if isinstance(hashes, set):
            hashes.add(str(digest))
    if changed:
        entry['policy_changed_true'] = int(entry.get('policy_changed_true', 0)) + 1

if policy_summary:
    serializable: dict[str, dict[str, object]] = {}
    for model, data in policy_summary.items():
        hashes = data.get('thresholds_hashes') or set()
        if isinstance(hashes, set):
            hash_list = sorted(hashes)
        else:
            hash_list = []
        serializable[model] = {
            'unique_thresholds_hashes': hash_list,
            'unique_hash_count': len(hash_list),
            'policy_changed_true': int(data.get('policy_changed_true', 0)),
            'total_certs': int(data.get('total_certs', 0)),
        }
    with open(analysis_dir / "policy_digest_summary.json", 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Wrote policy digest summary for {len(serializable)} models")

# Determinism summary (per model + overall)
det_by_model: dict[str, dict[str, int]] = {}
overall = {'strict': 0, 'tolerance': 0, 'off': 0, 'unknown': 0}
for row in invar_rows:
    model = row.get('model', 'unknown')
    level = str(row.get('determinism_level') or '').strip().lower()
    if not level:
        level = 'unknown'
    if level not in overall:
        level = 'unknown'
    model_counts = det_by_model.setdefault(
        model, {k: 0 for k in overall.keys()}
    )
    model_counts[level] = int(model_counts.get(level, 0)) + 1
    overall[level] = int(overall.get(level, 0)) + 1

if det_by_model:
    det_payload = {
        'by_model': det_by_model,
        'overall': overall,
    }
    with open(analysis_dir / "determinism_summary.json", 'w') as f:
        json.dump(det_payload, f, indent=2)
    print(f"Wrote determinism summary for {len(det_by_model)} models")

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

	    python3 <<- EOF
import json
import csv
import math
from pathlib import Path
from collections import defaultdict

output_dir = Path("${OUTPUT_DIR}")
analysis_dir = output_dir / "analysis"

skip_dirs = {
    "logs",
    "analysis",
    "reports",
    "presets",
    "models",
    "queue",
    "workers",
    "state",
    "evals",
    "certificates",
}

def _is_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name in skip_dirs:
        return False
    if not (path / ".baseline_path").exists():
        return False
    return True

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

print("=== CORRELATION ANALYSIS (Proof Pack) ===\n")

results = {
    'models': {},
    'error_detection': {'detected': [], 'missed': []},
    'calibration': cal_summary,
    'pm_correlation': {},
}
def as_bool(val):
    if val is None: return False
    if isinstance(val, bool): return val
    if isinstance(val, str): return val.strip().lower() in ('true', '1', 'yes', 'on')
    return bool(val)

degraded_edits = 0
degraded_runs = []
categories = defaultdict(int)
# Track (delta_eval, pm_ratio) pairs for correlation analysis
pm_points = []
triage_counts = defaultdict(int)

for model_dir in output_dir.iterdir():
    if not _is_model_dir(model_dir):
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

        # Skip runs that are not part of the edit-vs-eval correlation study.
        # - errors: handled separately in error_detection
        # - calibration: no lm-eval baseline, would inflate TRUE_NEGATIVE counts
        if edit_type in {"errors", "calibration"}:
            continue

        edit_evals = eval_data.get((model, edit_type), {})

        # Determine if this edit has a statistically meaningful regression vs baseline.
        # Use a simple binomial standard error approximation per benchmark.
        N_TABLE = {
            'mmlu': 14042,
            'hellaswag': 10042,
            'arc_challenge': 2590,
            'winogrande': 1767,
        }

        has_regression = False
        deltas = []
        delta_by_task = {}
        regression_tasks = []
        for task, base_val in baseline_evals.items():
            edit_val = edit_evals.get(task)
            if edit_val is None:
                continue
            delta = edit_val - base_val
            deltas.append(delta)
            delta_by_task[task] = delta
            # Map task name back to benchmark key
            task_key = task
            if task_key.startswith('arc'):
                task_key = 'arc_challenge'
            n = N_TABLE.get(task_key, 1000)
            p = max(min(base_val, 0.999), 0.001)
            se = math.sqrt(p * (1.0 - p) / n)
            if delta < -2.0 * se:
                has_regression = True
                regression_tasks.append(task)
                # Keep scanning to accumulate deltas but we already know it's regressed
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        worst_task = None
        worst_delta = None
        if delta_by_task:
            worst_task, worst_delta = min(delta_by_task.items(), key=lambda item: item[1])

        invar_flagged = any(
            str(r.get('all_pass', '')).lower() == 'false' or r.get('all_pass') is False
            for r in invar_results
        )
        degraded_present = any(as_bool(r.get('pm_degraded')) for r in invar_results)
        if degraded_present:
            degraded_edits += 1
            degraded_runs.extend([f"{model}/{r.get('run', 'unknown')}" for r in invar_results if as_bool(r.get('pm_degraded'))])
        if degraded_present:
            invar_flagged = True

        # Aggregate primary-metric ratio for this edit (continuous InvarLock signal)
        pm_vals = []
        for r in invar_results:
            try:
                v = r.get('pm_ratio')
                if v is None or v == '':
                    continue
                pm_vals.append(float(v))
            except Exception:
                continue
	        pm_ratio_mean = sum(pm_vals) / len(pm_vals) if pm_vals else None
	        if pm_ratio_mean is not None and deltas:
	            pm_points.append((mean_delta, math.log(pm_ratio_mean)))

	        # Aggregate triage across replicates: FAIL if any fail, PASS if all pass.
	        triage_votes = []
	        for r in invar_results:
	            t = str(r.get('triage', '') or '').strip().upper()
	            if t:
	                triage_votes.append(t)
	        if any(t == 'FAIL' for t in triage_votes):
	            triage = 'FAIL'
	        elif triage_votes and all(t == 'PASS' for t in triage_votes):
	            triage = 'PASS'
	        else:
	            triage = 'REVIEW'
	        triage_counts[triage] += 1

	        if has_regression and invar_flagged: category = "TRUE_POSITIVE"
	        elif not has_regression and invar_flagged: category = "FALSE_POSITIVE"
	        elif not has_regression and not invar_flagged: category = "TRUE_NEGATIVE"
	        else: category = "FALSE_NEGATIVE"

        categories[category] += 1
	        results['models'][model][edit_type] = {
	            'category': category,
	            'regression': has_regression,
	            'flagged': invar_flagged,
	            'triage': triage,
	            'mean_delta_eval': mean_delta,
	            'delta_by_task': delta_by_task,
	            'regression_tasks': regression_tasks,
	            'worst_delta_task': worst_task,
	            'worst_delta': worst_delta,
	            'mean_pm_ratio': pm_ratio_mean,
	        }
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

# Compute simple Pearson correlation between Δeval and log(pm_ratio)
if len(pm_points) >= 2:
    xs = [p[0] for p in pm_points]
    ys = [p[1] for p in pm_points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    corr = num / (den_x * den_y) if den_x > 0 and den_y > 0 else 0.0
    results['pm_correlation'] = {
        'pearson_r_delta_vs_log_pm': corr,
        'num_points': len(pm_points),
    }
    print(f"\nPM correlation (Δeval vs log pm_ratio): r = {corr:.3f} over {len(pm_points)} edits")
else:
    results['pm_correlation'] = {'pearson_r_delta_vs_log_pm': 0.0, 'num_points': len(pm_points)}

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

# Wilson score intervals for accuracy and error detection rate (95% CI)
def wilson_interval(successes, n, z=1.96):
    if n <= 0:
        return (0.0, 0.0)
    p_hat = successes / n
    denom = 1.0 + z * z / n
    centre = p_hat + z * z / (2 * n)
    margin = z * ((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) ** 0.5
    lower = max(0.0, (centre - margin) / denom)
    upper = min(1.0, (centre + margin) / denom)
    return (lower, upper)

acc_ci = wilson_interval(tp + tn, total) if total > 0 else (0.0, 0.0)
err_ci = wilson_interval(err_detected, err_total) if err_total > 0 else (0.0, 0.0)

# Confidence components:
# - sample_confidence: grows with log(#tests) up to 25
# - error_confidence: grows with number of error injections up to 25
# - accuracy_confidence: higher when CI for accuracy is tight
# - balance_confidence: based on F1
import math as _math
sample_confidence = min((_math.log1p(total) / _math.log1p(64)) * 25, 25) if total > 0 else 0
error_confidence = min((_math.log1p(err_total) / _math.log1p(16)) * 25, 25) if err_total > 0 else 0
acc_ci_width = acc_ci[1] - acc_ci[0]
accuracy_confidence = max(0.0, (1.0 - min(acc_ci_width, 1.0)) * 25)
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
print(f"Degraded edits: {degraded_edits}")
	print(f"Error Detection: {err_detected}/{err_total} ({err_rate:.0%})")
	print(f"Triage (edits): PASS={triage_counts.get('PASS', 0)} REVIEW={triage_counts.get('REVIEW', 0)} FAIL={triage_counts.get('FAIL', 0)}")
	print(f"Confidence Score: {confidence_score:.1f}/100 ({confidence_level})")

	results['summary'] = {
	    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'error_detection_rate': err_rate,
    'categories': dict(categories),
	    'confidence_score': confidence_score,
	    'confidence_level': confidence_level,
	    'triage_counts': dict(triage_counts),
	    'degraded_edits': degraded_edits,
	    'degraded_runs': degraded_runs,
	    'total_tests': total,
	    'models_tested': len(results['models']),
	    'accuracy_ci': acc_ci,
	    'error_rate_ci': err_ci,
	}

with open(analysis_dir / "correlation_analysis.json", 'w') as f:
    json.dump(results, f, indent=2)
EOF
}

# ============ GENERATE VERDICT ============
generate_verdict() {
    log_section "GENERATING FINAL VERDICT"

	    python3 <<- EOF
import json
import os
import re
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
	triage_counts = summary.get('triage_counts', {}) or {}
	triage_pass = triage_counts.get('PASS', 0)
	triage_review = triage_counts.get('REVIEW', 0)
	triage_fail = triage_counts.get('FAIL', 0)
degraded = summary.get('degraded_edits', 0) or 0
degraded_runs = summary.get('degraded_runs', []) or []
models = analysis.get('models', {}) or {}

determinism_path = analysis_dir / "determinism_repeats.json"
determinism_repeats = None
if determinism_path.exists():
    try:
        determinism_repeats = json.loads(determinism_path.read_text())
    except Exception:
        determinism_repeats = None

gpu_count = (os.environ.get("PACK_GPU_COUNT") or os.environ.get("NUM_GPUS") or "").strip() or "unknown"
gpu_mem = (os.environ.get("PACK_GPU_MEM_GB") or os.environ.get("GPU_MEMORY_GB") or "").strip()
gpu_name = (os.environ.get("PACK_GPU_NAME") or "GPU").strip() or "GPU"
gpu_mem_label = f"{gpu_mem}GB" if gpu_mem else "unknown"
tag_name = re.sub(r"[^A-Za-z0-9]+", "_", gpu_name).strip("_") or "GPU"
platform_header = f"{gpu_name} {gpu_mem_label} x {gpu_count} GPU"
platform_line = f"{gpu_count}x {gpu_name} {gpu_mem_label}"
platform_tag = f"{tag_name}_{gpu_mem_label}_x{gpu_count}"

def fmt_delta(val):
    try:
        return f"{float(val):+0.4f}"
    except Exception:
        return "n/a"

def ordered_tasks(delta_by_task):
    order = ["mmlu", "hellaswag", "arc_challenge", "winogrande"]
    tasks = [task for task in order if task in delta_by_task]
    extra = sorted(task for task in delta_by_task if task not in order)
    return tasks + extra

metric_definitions = (
    "METRIC DEFINITIONS:\n"
    "  * Regression: any lm-eval benchmark drop below -2xSE vs baseline.\n"
    "  * Flagged: InvarLock guard failure or primary metric degradation.\n"
    "  * Accuracy/Precision/Recall treat regression as positive class.\n"
)

delta_lines = []
delta_summary = {}
if models:
    delta_lines.append("LM-EVAL DELTAS (edit - baseline):")
    for model in sorted(models):
        edits = models.get(model) or {}
        delta_summary[model] = {}
        if not edits:
            continue
        delta_lines.append(f"  {model}:")
        for edit_name in sorted(edits):
            entry = edits.get(edit_name) or {}
            delta_by_task = entry.get("delta_by_task") or {}
            mean_delta = entry.get("mean_delta_eval")
            worst_task = entry.get("worst_delta_task")
            worst_delta = entry.get("worst_delta")
            regression_tasks = entry.get("regression_tasks") or []
            tasks_ordered = ordered_tasks(delta_by_task)
            if tasks_ordered:
                task_blob = ", ".join(
                    f"{task}:{fmt_delta(delta_by_task.get(task))}" for task in tasks_ordered
                )
            else:
                task_blob = "n/a"
            worst_blob = f"{worst_task}:{fmt_delta(worst_delta)}" if worst_task else "n/a"
            regression_blob = "none" if not regression_tasks else ", ".join(regression_tasks)
            delta_lines.append(
                f"    {edit_name}: mean {fmt_delta(mean_delta)} | worst {worst_blob} | regressions {regression_blob} | [{task_blob}]"
            )
            delta_summary[model][edit_name] = {
                "mean_delta_eval": mean_delta,
                "delta_by_task": delta_by_task,
                "regression_tasks": regression_tasks,
                "worst_delta_task": worst_task,
                "worst_delta": worst_delta,
            }
    delta_section = "\n".join(delta_lines) + "\n\n"
else:
    delta_section = ""

phase0_pass = accuracy >= 0.6 and err_rate >= 0.8
if degraded > 0:
    phase0_pass = False

if degraded > 0:
    verdict = "PHASE0_DEGRADED"
    verdict_confidence = "LOW"
elif phase0_pass and accuracy >= 0.8 and confidence_score >= 75:
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
     INVARLOCK PHASE 0 VALIDATION - {platform_header}
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
	     TRIAGE (edits):    PASS={triage_pass} REVIEW={triage_review} FAIL={triage_fail}
	     DEGRADED CERTS:    {degraded}
	--------------------------------------------------------------------------------
	     VERDICT: {verdict}
	     VERDICT CONFIDENCE: {verdict_confidence}
	================================================================================

{metric_definitions}{delta_section}EDIT TYPES TESTED:
  * Quantization RTN (group-wise): 8-bit (clean), 4-bit (stress)
  * FP8 Quantization (E4M3 clean, E5M2 stress)
  * Magnitude Pruning: 10% (clean), 50% (stress)
  * Low-Rank SVD: rank-256 (clean), rank-32 (stress)

PLATFORM: {platform_line}

'''

if verdict == "PHASE0_VALIDATED":
    report += "RESULT: InvarLock Phase 0 VALIDATED on proof pack hardware.\n"
elif verdict == "PHASE0_DEGRADED":
    report += f"RESULT: Phase 0 degraded. {degraded} certificate(s) reported degraded primary metrics. See runs: {', '.join(degraded_runs) if degraded_runs else 'n/a'}\n"
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
        'triage': {'pass': triage_pass, 'review': triage_review, 'fail': triage_fail},
        'degraded': {'count': degraded, 'runs': degraded_runs},
        'phase0_pass': phase0_pass,
        'platform': platform_tag,
        'platform_name': gpu_name,
        'suite': os.environ.get('PACK_SUITE'),
        'network_mode': 'online' if os.environ.get('PACK_NET') == '1' else 'offline',
        'determinism_mode': os.environ.get('PACK_DETERMINISM'),
        'determinism_repeats': determinism_repeats,
        'models_tested': models_tested,
        'total_tests': total_tests,
        'lm_eval_deltas': delta_summary,
        'timestamp': datetime.now().isoformat()
    }, f, indent=2)


EOF
}

# ============ MAIN - DYNAMIC GPU SCHEDULING (v2.0) ============
main_dynamic() {
    local start_time=$(date +%s)
    local gpu_mem="${PACK_GPU_MEM_GB:-${GPU_MEMORY_GB:-}}"
    local gpu_count_label="${NUM_GPUS:-auto}"
    [[ -z "${gpu_mem}" ]] && gpu_mem="auto"
    [[ -z "${gpu_count_label}" ]] && gpu_count_label="auto"

    echo "========================================================================"
    echo "  InvarLock Proof Pack Suite v${SCRIPT_VERSION}"
    echo "  ${gpu_mem}GB x ${gpu_count_label} GPU DYNAMIC SCHEDULING"
    echo "========================================================================"
    echo ""

    check_dependencies
    configure_gpu_pool
    pack_model_list_array

    # Disk pressure preflight (Slurm/Ray-style node health gate).
    # Abort before starting work to avoid half-written artifacts when storage is nearly full.
    local min_free="${MIN_FREE_DISK_GB:-200}"
    if ! [[ "${min_free}" =~ ^[0-9]+$ ]]; then
        min_free=200
    fi
    local free_gb=""
    free_gb=$(get_free_disk_gb "${OUTPUT_DIR}" 2>/dev/null || echo "")
    if [[ -n "${free_gb}" && ${free_gb} -lt ${min_free} ]]; then
        handle_disk_pressure "${free_gb}" "${min_free}"
    fi

    # Disk capacity preflight based on planned model/edit storage.
    # This prevents expensive GPU time from being spent only to later hit ENOSPC.
    disk_preflight

    setup_pack_environment

    log "Output directory: ${OUTPUT_DIR}"
    log "GPU pool: ${NUM_GPUS} GPU(s) [${GPU_ID_LIST}]"
    local model_count
    model_count=$(pack_model_list | wc -l | tr -d ' ')
    log "Models: ${model_count} (PACK_SUITE=${PACK_SUITE})"
    log "Edit Types: 4 x 2 versions = 8 per model"
    log "Scheduling: DYNAMIC (work-stealing enabled)"
    log ""

    # Initialize queue
    log_section "PHASE 1: INITIALIZING TASK QUEUE"

    # Check for --resume mode: skip task generation if queue already exists with tasks
    local existing_queue="${OUTPUT_DIR}/queue"
    local skip_task_generation="false"
    local resume_total_tasks=0
    local resume_existing_running=0
    local resume_existing_failed=0

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
            resume_total_tasks="${existing_total}"
            resume_existing_running="${existing_running}"
            resume_existing_failed="${existing_failed}"
            log "RESUME MODE: Found existing queue with ${existing_total} tasks"
            log "  Pending: ${existing_pending}, Ready: ${existing_ready}, Running: ${existing_running}"
            log "  Completed: ${existing_completed}, Failed: ${existing_failed}"
        fi
    fi

    init_queue "${OUTPUT_DIR}"
    # Clear any previous shutdown markers so new workers can start cleanly (important for --resume).
    rm -f "${OUTPUT_DIR}/workers/SHUTDOWN" "${OUTPUT_DIR}/workers"/gpu_*.shutdown 2>/dev/null || true
    # Initialize GPU reservation tracking for multi-GPU tasks before workers start.
    if type init_gpu_reservations &>/dev/null; then
        init_gpu_reservations "${OUTPUT_DIR}"
        log "GPU reservations dir: ${GPU_RESERVATION_DIR:-unset}; GPUs: $(list_run_gpu_ids | tr '\n' ',' | sed 's/,$//')"
    fi
    export QUEUE_DIR GPU_RESERVATION_DIR  # Export for subshell workers

    local total_tasks=0
    if [[ "${skip_task_generation}" == "true" ]]; then
        log "Skipping task generation (--resume mode)"
        # Reclaim any stuck running tasks from a previous run (kills stray procs, releases GPU reservations).
        if [[ ${resume_existing_running} -gt 0 ]]; then
            log "Reclaiming ${resume_existing_running} orphaned running task(s) for resume..."
            local gpu_id
            for gpu_id in $(list_run_gpu_ids); do
                reclaim_orphaned_tasks "${gpu_id}" >> "${LOG_FILE}" 2>&1 || true
            done
        fi

        # Move failed tasks back to pending for retry, clearing retry/backoff state for immediate resume.
        if [[ ${resume_existing_failed} -gt 0 ]]; then
            log "Resetting ${resume_existing_failed} failed task(s) back to pending for resume..."
            local task_file
            for task_file in "${QUEUE_DIR}/failed"/*.task; do
                [[ -f "${task_file}" ]] || continue
                local tmp_file="${task_file}.resume.$$"
                jq '(.params // {}) as $p
                    | .status="pending"
                    | .retries=0
                    | .gpu_id=-1
                    | .assigned_gpus=null
                    | .started_at=null
                    | .completed_at=null
                    | .error_msg=null
                    | .params=($p + {retry_after:null,last_error_type:null})' "${task_file}" > "${tmp_file}" 2>/dev/null \
                    && mv "${tmp_file}" "${task_file}" 2>/dev/null \
                    || { rm -f "${tmp_file}" 2>/dev/null || true; }
                mv "${task_file}" "${QUEUE_DIR}/pending/" 2>/dev/null || true
            done
        fi

        # Re-resolve dependencies after reclaim/reset.
        if type resolve_dependencies &>/dev/null; then
            local moved=$(resolve_dependencies)
            log "Re-resolved dependencies: moved ${moved} tasks to ready queue"
        fi
    else
        # Generate all tasks
        log "Generating tasks for all models..."
        log "Config: CLEAN_EDIT_RUNS=${CLEAN_EDIT_RUNS}, STRESS_EDIT_RUNS=${STRESS_EDIT_RUNS}, RUN_ERROR_INJECTION=${RUN_ERROR_INJECTION}, DRIFT_CALIBRATION_RUNS=${DRIFT_CALIBRATION_RUNS}, PACK_USE_BATCH_EDITS=${PACK_USE_BATCH_EDITS:-auto}"
        local model_csv
        model_csv=$(printf '%s\n' "${PACK_MODEL_LIST[@]}" | paste -sd "," -)
        log "Models: ${model_csv:-<none>}"
        generate_all_tasks "${PACK_MODEL_LIST[@]}"
    fi

    if type refresh_task_memory_from_profiles &>/dev/null; then
        refresh_task_memory_from_profiles "${OUTPUT_DIR}"
    fi
    if type export_memory_plan &>/dev/null; then
        export_memory_plan "${OUTPUT_DIR}"
    fi

    # Resolve initial dependencies on fresh runs so workers can start immediately (avoid idle GPUs).
    if [[ "${skip_task_generation}" != "true" ]] && type resolve_dependencies &>/dev/null; then
        local moved_initial=0
        moved_initial=$(resolve_dependencies 2>/dev/null) || moved_initial=0
        log "Resolved initial dependencies: moved ${moved_initial} task(s) to ready queue"
    fi
    if type demote_ready_tasks_for_calibration_only &>/dev/null; then
        demote_ready_tasks_for_calibration_only 2>/dev/null || true
    fi

    total_tasks=$(count_tasks "pending")
    total_tasks=$((total_tasks + $(count_tasks "ready")))
    total_tasks=$((total_tasks + $(count_tasks "completed")))
    log "Total tasks in queue: ${total_tasks} (pending+ready: $(($(count_tasks "pending") + $(count_tasks "ready"))))"

    # Launch worker pool
    log_section "PHASE 2: LAUNCHING GPU WORKERS"
    log "Starting ${NUM_GPUS} GPU workers with dynamic task scheduling..."

    # Initialize log files
    for gpu_id in $(list_run_gpu_ids); do
        touch "${OUTPUT_DIR}/logs/gpu_${gpu_id}.log"
    done

    start_worker() {
        local gpu_id="$1"
        local action="${2:-Starting}"

        # Avoid duplicating a live worker on the same GPU
        local pid_file="${OUTPUT_DIR}/workers/gpu_${gpu_id}.pid"
        if [[ -f "${pid_file}" ]]; then
            local existing_pid
            existing_pid=$(cat "${pid_file}" 2>/dev/null || true)
            if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
                log "  GPU ${gpu_id}: worker already running (PID ${existing_pid}), skipping start"
                return 0
            fi
        fi

        log "  GPU ${gpu_id}: ${action} worker"
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
        pids[${gpu_id}]=$!
        echo "${pids[${gpu_id}]}" > "${OUTPUT_DIR}/workers/gpu_${gpu_id}.pid"
    }

    # Unified monitor loop: progress + dependency resolution + worker health
    log_section "PHASE 3: MONITORING PROGRESS"
    local check_interval=60
    local worker_timeout="${WORKER_TIMEOUT:-2700}"
    local workers_started=0
    while true; do
        if [[ ${workers_started} -eq 0 ]]; then
            for gpu_id in $(list_run_gpu_ids); do
                start_worker "${gpu_id}" "Starting"
            done
            workers_started=1
        fi

        sleep "${check_interval}"

        # Disk pressure check (Slurm/Ray-style node health gate).
        # Abort early to avoid corrupting artifacts when storage is nearly full.
        local min_free="${MIN_FREE_DISK_GB:-200}"
        if ! [[ "${min_free}" =~ ^[0-9]+$ ]]; then
            min_free=200
        fi
        local free_gb=""
        free_gb=$(get_free_disk_gb "${OUTPUT_DIR}" 2>/dev/null || echo "")
        if [[ -n "${free_gb}" && ${free_gb} -lt ${min_free} ]]; then
            handle_disk_pressure "${free_gb}" "${min_free}"
        fi

        # Check if done
        if [[ "${PACK_SUITE_MODE:-full}" == "calibrate-only" ]]; then
            local preset_total=0
            local preset_completed=0
            preset_total=$(find "${QUEUE_DIR}" -type f -name "*_GENERATE_PRESET_*.task" 2>/dev/null | wc -l | tr -d ' ')
            preset_completed=$(find "${QUEUE_DIR}/completed" -type f -name "*_GENERATE_PRESET_*.task" 2>/dev/null | wc -l | tr -d ' ')
            if [[ ${preset_total} -gt 0 && ${preset_completed} -eq ${preset_total} ]]; then
                log "Calibration-only: generated ${preset_completed}/${preset_total} calibrated preset(s); stopping early"
                if type signal_shutdown &>/dev/null; then
                    signal_shutdown "${OUTPUT_DIR}"
                else
                    touch "${OUTPUT_DIR}/workers/SHUTDOWN"
                fi
                break
            fi
        fi
        if is_queue_empty; then
            if type signal_shutdown &>/dev/null; then
                signal_shutdown "${OUTPUT_DIR}"
            else
                touch "${OUTPUT_DIR}/workers/SHUTDOWN"
            fi
            break
        fi

        # Check each worker for liveness and heartbeat
        for gpu_id in $(list_run_gpu_ids); do
            local pid_file="${OUTPUT_DIR}/workers/gpu_${gpu_id}.pid"
            local heartbeat_file="${OUTPUT_DIR}/workers/gpu_${gpu_id}.heartbeat"
            local status_file="${OUTPUT_DIR}/workers/gpu_${gpu_id}.status"

            [[ -f "${pid_file}" ]] || continue
            local pid
            pid=$(cat "${pid_file}" 2>/dev/null || true)
            [[ -z "${pid}" ]] && continue

            if ! kill -0 "${pid}" 2>/dev/null; then
                log "WARNING: Worker GPU ${gpu_id} (PID ${pid}) died"
                wait "${pid}" 2>/dev/null || true
                reclaim_orphaned_tasks "${gpu_id}"
                start_worker "${gpu_id}" "Restarting"
                continue
            fi

            if [[ -f "${heartbeat_file}" ]]; then
                local heartbeat_mtime
                heartbeat_mtime=$(stat -c %Y "${heartbeat_file}" 2>/dev/null || stat -f %m "${heartbeat_file}" 2>/dev/null || echo "")
                if [[ -n "${heartbeat_mtime}" ]]; then
                    local heartbeat_age=$(( $(date +%s) - heartbeat_mtime ))
                    if [[ ${heartbeat_age} -gt ${worker_timeout} ]]; then
                        local status
                        status=$(cat "${status_file}" 2>/dev/null || echo "unknown")
                        log "WARNING: Worker GPU ${gpu_id} stuck (no heartbeat for ${heartbeat_age}s, status: ${status})"
                        kill -9 "${pid}" 2>/dev/null || true
                        wait "${pid}" 2>/dev/null || true
                        reclaim_orphaned_tasks "${gpu_id}"
                        start_worker "${gpu_id}" "Restarting stuck"
                    fi
                fi
            fi
        done

        # Centralized dependency resolution - moved from worker loops to reduce lock contention.
        # Only the monitor (single process) calls this, avoiding 8 workers competing for queue lock.
        local deps_moved=0
        deps_moved=$(resolve_dependencies 2>/dev/null) || deps_moved=0
        if [[ ${deps_moved} -gt 0 ]]; then
            log "Monitor: Promoted ${deps_moved} task(s) from pending to ready queue"
        fi
        local deps_canceled=0
        if type cancel_tasks_with_failed_dependencies &>/dev/null; then
            deps_canceled=$(cancel_tasks_with_failed_dependencies "${CANCEL_BLOCKED_TASKS_GRACE_SECONDS:-90}" 2>/dev/null) || deps_canceled=0
            if [[ ${deps_canceled} -gt 0 ]]; then
                log "Monitor: Marked ${deps_canceled} task(s) failed due to failed dependencies"
            fi
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

    # Wait for all workers
    log "Waiting for all workers to complete..."
    local failed=0
    for gpu_id in $(list_run_gpu_ids); do
        local pid="${pids[${gpu_id}]:-}"
        if [[ -n "${pid}" ]]; then
            if wait "${pid}"; then
                log "  GPU ${gpu_id}: Worker completed successfully"
            else
                log "  GPU ${gpu_id}: Worker failed"
                failed=$((failed + 1))
            fi
        fi
    done

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

    if [[ "${PACK_SUITE_MODE:-full}" == "calibrate-only" ]]; then
        log_section "CALIBRATION CHECKPOINT"
        log "Calibration-only run stopped after preset generation."
        log "Presets: ${OUTPUT_DIR}/presets/"
        log "To continue: OUTPUT_DIR=${OUTPUT_DIR} $0 --run-only"
        return 0
    fi

    if [[ -n "${PACK_REPEATS:-}" && "${PACK_REPEATS}" != "0" ]]; then
        log_section "DETERMINISM REPEATS"
        if ! pack_run_determinism_repeats; then
            log "WARNING: Determinism repeats failed; see logs for details."
        fi
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

# ============ MAIN ============
# Dynamic scheduling with work-stealing is the only supported mode (v2.1.0)
# Static scheduling was removed as it's less efficient.
main() {
    main_dynamic "$@"
}

pack_run_suite() {
    # Enable strict mode only for actual suite execution (tests may source this file).
    set -uo pipefail
    trap cleanup EXIT INT TERM HUP QUIT

    pack_require_bash4 || return 1

    if [[ -z "${OUTPUT_DIR:-}" ]]; then
        echo "ERROR: OUTPUT_DIR is not set; use run_suite.sh --out or PACK_OUTPUT_DIR." >&2
        return 1
    fi
    # Optionally normalize OUTPUT_DIR to an absolute path (set PACK_OUTPUT_DIR_ABSOLUTE=true).
    if [[ -n "${OUTPUT_DIR}" && "${PACK_OUTPUT_DIR_ABSOLUTE:-false}" == "true" ]]; then
        OUTPUT_DIR="$(cd "$(dirname "${OUTPUT_DIR}")" && pwd)/$(basename "${OUTPUT_DIR}")"
    fi
    PACK_OUTPUT_DIR="${OUTPUT_DIR}"
    export PACK_OUTPUT_DIR

    pack_apply_network_mode "${PACK_NET}"
    pack_source_libs || return 1
    pack_setup_output_dirs || return 1
    pack_setup_hf_cache_dirs || return 1

    pack_model_list_array
    if [[ ${#PACK_MODEL_LIST[@]} -eq 0 ]]; then
        error_exit "No models configured for PACK_SUITE=${PACK_SUITE}."
    fi

    if [[ "${PACK_NET}" == "1" ]]; then
        pack_preflight_models "${OUTPUT_DIR}" "${PACK_MODEL_LIST[@]}"
    else
        if ! pack_load_model_revisions; then
            error_exit "Offline mode requires model revisions. Run with --net 1 to preflight."
        fi
    fi

    main_dynamic
}
