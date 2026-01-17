#!/usr/bin/env bash
# task_functions.sh - Atomic task implementations for dynamic scheduling
# Version: proof-packs-v1 (InvarLock Proof Pack Suite)
# Dependencies: jq, python3, invarlock CLI, lm_eval, task_serialization.sh
# Usage: sourced by gpu_worker.sh/validation_suite.sh for per-task execution
#
# Each function executes a single atomic task type with explicit parameters.
# These are extracted from the original monolithic process_model() function
# to enable parallel execution across GPUs.

# Source dependencies
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime.sh
source "${SCRIPT_DIR}/runtime.sh"
[[ -z "${QUEUE_MANAGER_LOADED:-}" ]] && source "${SCRIPT_DIR}/queue_manager.sh" && export QUEUE_MANAGER_LOADED=1

# ============ FALLBACK FUNCTIONS ============
# These provide fallback implementations when main script functions aren't available
# (e.g., when running in subshell workers that only source lib modules)

# Detect model size from model name/path string
# Returns: 7, 13, 30, 40, 70, moe
_get_model_size_from_name() {
    local model_id="$1"
    local model_lower=$(printf '%s' "${model_id}" | tr '[:upper:]' '[:lower:]')

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

    # Conservative defaults that satisfy CI pairing/coverage floors for proof packs.
    # Use zero overlap (stride == seq_len) and ≥180 windows to avoid E001/E005
    # if workers start without the main suite wrapper.
    case "${model_size}" in
        "7")
            echo "2048:2048:192:192:96"
            ;;
        "13")
            echo "1536:1536:192:192:64"
            ;;
        "30")
            echo "1024:1024:192:192:48"
            ;;
        "40")
            echo "1024:1024:192:192:32"
            ;;
        "moe")
            echo "1024:1024:192:192:24"
            ;;
        "70"|"72")
            # Minimal sequence length to cap KV cache, but still meet coverage floors.
            echo "128:128:192:192:2"
            ;;
        *)
            # Safe default
            echo "1024:1024:192:192:32"
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

_task_create_model_variant() {
    local baseline_path="$1"
    local output_path="$2"
    local edit_type="$3"
    local param1="${4:-}"
    local param2="${5:-}"
    local scope="${6:-}"
    local gpu_id="${7:-0}"

    if type create_model_variant &>/dev/null; then
        create_model_variant "${baseline_path}" "${output_path}" "${edit_type}" "${param1}" "${param2}" "${scope}" "${gpu_id}"
        return $?
    fi

    case "${edit_type}" in
        "quant_rtn")
            if ! type create_edited_model &>/dev/null; then
                echo "ERROR: create_edited_model not available" >&2
                return 1
            fi
            create_edited_model "${baseline_path}" "${output_path}" "quant_rtn" "${param1}" "${param2}" "${scope}" "${gpu_id}"
            ;;
        "fp8_quant")
            if ! type create_fp8_model &>/dev/null; then
                echo "ERROR: create_fp8_model not available" >&2
                return 1
            fi
            create_fp8_model "${baseline_path}" "${output_path}" "${param1}" "${scope}" "${gpu_id}"
            ;;
        "magnitude_prune")
            if ! type create_pruned_model &>/dev/null; then
                echo "ERROR: create_pruned_model not available" >&2
                return 1
            fi
            create_pruned_model "${baseline_path}" "${output_path}" "${param1}" "${scope}" "${gpu_id}"
            ;;
        "lowrank_svd")
            if ! type create_lowrank_model &>/dev/null; then
                echo "ERROR: create_lowrank_model not available" >&2
                return 1
            fi
            create_lowrank_model "${baseline_path}" "${output_path}" "${param1}" "${scope}" "${gpu_id}"
            ;;
        *)
            echo "ERROR: Unknown edit type: ${edit_type}" >&2
            return 1
            ;;
    esac
}

_task_get_model_revision() {
    local model_id="$1"
    if type pack_model_revision &>/dev/null; then
        pack_model_revision "${model_id}"
        return
    fi
    local path="${PACK_MODEL_REVISIONS_FILE:-${OUTPUT_DIR:-}/state/model_revisions.json}"
    [[ -f "${path}" ]] || return 0
    python3 - "${path}" "${model_id}" <<'PY' 2>/dev/null
import json
import os
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

# Build lm-eval model_args with optional multi-GPU parallelization.
_get_lmeval_model_args() {
    local model_path="$1"
    local model_args="pretrained=${model_path},trust_remote_code=True,dtype=bfloat16"
    local parallelize_flag="${LM_EVAL_PARALLELIZE:-true}"
    parallelize_flag=$(printf '%s' "${parallelize_flag}" | tr '[:upper:]' '[:lower:]')
    local multi_gpu="false"
    if [[ "${CUDA_VISIBLE_DEVICES:-}" == *","* ]]; then
        multi_gpu="true"
    fi

    if [[ "${multi_gpu}" == "true" ]]; then
        model_args="${model_args},device_map=auto"
        if [[ "${parallelize_flag}" != "false" && "${parallelize_flag}" != "0" ]]; then
            model_args="${model_args},parallelize=True"
        fi
    fi

    echo "${model_args}"
}

# Check if model is large (30B+) and needs special handling.
# Changed threshold from 70 to 30 to fix hang on 30-40B models:
# - Skips overhead check (avoids loading model twice, which can exceed 180GB)
_is_large_model() {
    local model_size="$1"
    if [[ "${model_size}" == "moe" ]]; then
        return 0
    fi
    if [[ "${model_size}" =~ ^[0-9]+$ ]]; then
        [[ ${model_size} -ge 30 ]]
        return
    fi
    [[ "${model_size}" =~ 30 || "${model_size}" =~ 32 || "${model_size}" =~ 34 || "${model_size}" =~ 40 || "${model_size}" =~ 70 || "${model_size}" =~ 72 || "${model_size}" =~ 65 || "${model_size}" =~ 80 || "${model_size}" =~ 90 ]]
}

# Select lm-eval batch size caps based on model size.
_get_eval_batch_size() {
    local model_size="$1"

    case "${model_size}" in
        moe|MoE|MOE)
            echo "${EVAL_BATCH_SIZE_MOE:-auto:6}"
            return
            ;;
    esac

    if [[ "${model_size}" =~ ^[0-9]+$ ]]; then
        if [[ ${model_size} -ge 70 ]]; then
            echo "${EVAL_BATCH_SIZE_LARGE:-auto:4}"
        elif [[ ${model_size} -ge 30 ]]; then
            echo "${EVAL_BATCH_SIZE_MEDIUM:-auto:8}"
        else
            echo "${EVAL_BATCH_SIZE_SMALL:-auto:16}"
        fi
    else
        echo "${EVAL_BATCH_SIZE_SMALL:-auto:16}"
    fi
}

# Resolve an edit spec to concrete parameters and directory name.
# Returns JSON with status, edit_dir_name, and resolved params.
resolve_edit_params() {
    local model_output_dir="$1"
    local edit_spec="$2"
    local version_hint="${3:-}"

    _cmd_python - "${model_output_dir}" "${edit_spec}" "${version_hint}" <<'PY'
import json
import os
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
    tuned_path = (os.environ.get("PACK_TUNED_EDIT_PARAMS_FILE") or "").strip()
    model_id_path = model_output_dir / ".model_id"
    model_id = ""
    if model_id_path.exists():
        try:
            model_id = model_id_path.read_text().strip()
        except Exception:
            model_id = ""
    model_key = model_id or model_output_dir.name

    def _load_tuned_entry():
        if not tuned_path:
            return {}, "missing", "missing_tuned_edit_params_file"
        path = Path(tuned_path)
        if not path.exists():
            return {}, "missing", "missing_tuned_edit_params_file"
        try:
            data = json.loads(path.read_text())
        except Exception:
            return {}, "invalid", "invalid_tuned_edit_params_file"
        if not isinstance(data, dict):
            return {}, "invalid", "invalid_tuned_edit_params_file"

        entry_map = {}
        models = data.get("models")
        if isinstance(models, dict):
            entry_map = (
                models.get(model_key)
                or models.get(model_id)
                or models.get(model_output_dir.name)
                or {}
            )
        if not entry_map and isinstance(data.get(edit_type), dict):
            entry_map = data
        defaults = data.get("defaults")
        entry = entry_map.get(edit_type) or (defaults.get(edit_type) if isinstance(defaults, dict) else {}) or {}
        if not isinstance(entry, dict):
            entry = {}
        status = str(entry.get("status") or "missing")
        reason = str(entry.get("reason") or "")
        return entry, status, reason

    entry, status, reason = _load_tuned_entry()
    if status == "selected":
        if edit_type == "quant_rtn":
            param1 = str(entry.get("bits", ""))
            param2 = str(entry.get("group_size", ""))
            scope = str(entry.get("scope") or scope or "")
        elif edit_type == "fp8_quant":
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
    elif edit_type == "fp8_quant":
        if not param1:
            status = "invalid"
            reason = "invalid_fp_format"

version = version_hint or ("clean" if clean_spec else "")

if status == "selected" and not edit_dir_name:
    if edit_type == "quant_rtn":
        edit_dir_name = f"quant_{param1}bit_{version}" if version else ""
    elif edit_type == "fp8_quant":
        edit_dir_name = f"fp8_{param1}_{version}" if version else ""
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

# Resolve task timeout in seconds (empty/0 disables).
_get_task_timeout() {
    local task_type="$1"
    local default_timeout="${TASK_TIMEOUT_DEFAULT:-}"
    local override_var="TASK_TIMEOUT_${task_type}"
    local override="${!override_var:-}"
    local timeout="${override:-${default_timeout}}"

    if [[ -z "${timeout}" || "${timeout}" == "0" || "${timeout}" == "none" ]]; then
        return
    fi
    if [[ "${timeout}" =~ ^[0-9]+$ ]]; then
        echo "${timeout}"
    fi
}

_kill_task_process_group() {
    local pid="$1"
    local pgid=""
    local self_pgid=""

    pgid=$(_cmd_ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ')
    self_pgid=$(_cmd_ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ')

    if [[ -n "${pgid}" && -n "${self_pgid}" && "${pgid}" != "${self_pgid}" ]]; then
        _cmd_kill -TERM -- "-${pgid}" 2>/dev/null || true
        _sleep 5
        _cmd_kill -KILL -- "-${pgid}" 2>/dev/null || true
    else
        _cmd_kill -TERM "${pid}" 2>/dev/null || true
        _sleep 5
        _cmd_kill -KILL "${pid}" 2>/dev/null || true
    fi
}

_write_model_profile() {
    local baseline_dir="$1"
    local model_id="$2"
    local profile_path="${baseline_dir}/model_profile.json"

    [[ -f "${profile_path}" ]] && return 0
    [[ -d "${baseline_dir}" ]] || return 1

    _cmd_python << PROFILE_EOF >/dev/null 2>&1 || true
import json
from pathlib import Path

baseline_dir = Path("${baseline_dir}")
model_id = "${model_id}"
config_path = baseline_dir / "config.json"

if not config_path.exists():
    raise SystemExit(0)

try:
    cfg = json.loads(config_path.read_text())
except Exception:
    raise SystemExit(0)

def _get(key, *fallbacks):
    val = cfg.get(key)
    if val is not None:
        return val
    for fb in fallbacks:
        val = cfg.get(fb)
        if val is not None:
            return val
    return None

weights_bytes = 0
for pat in ("*.safetensors", "*.bin"):
    for fp in baseline_dir.glob(pat):
        weights_bytes += fp.stat().st_size

profile = {
    "model_id": model_id,
    "weights_bytes": weights_bytes,
    "weights_gb": round(weights_bytes / (1024 ** 3), 3),
    "hidden_size": _get("hidden_size", "n_embd", "d_model"),
    "num_layers": _get("num_hidden_layers", "n_layer"),
    "num_heads": _get("num_attention_heads", "n_head"),
    "num_kv_heads": _get("num_key_value_heads", "num_key_value_groups"),
    "max_position_embeddings": _get("max_position_embeddings", "max_seq_len", "seq_length"),
    "dtype_bytes": 2,
}

(baseline_dir / "model_profile.json").write_text(json.dumps(profile, indent=2))
PROFILE_EOF
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

    # Get assigned GPUs from task file (multi-GPU support)
    # If assigned_gpus is set (e.g., "2,3,4,5" for 4-GPU tasks), use that
    # Otherwise fall back to the single gpu_id parameter
    local assigned_gpus=$(get_task_assigned_gpus "${task_file}")
    assigned_gpus="${assigned_gpus// /}"
    if [[ -z "${assigned_gpus}" || "${assigned_gpus}" == "null" ]]; then
        assigned_gpus="${gpu_id}"
    fi

    # Create task-specific log
    local task_log="${output_dir}/logs/tasks/${task_id}.log"
    mkdir -p "$(dirname "${task_log}")"

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Starting task: ${task_id}" >> "${task_log}"
    echo "  Type: ${task_type}" >> "${task_log}"
    echo "  Model: ${model_id}" >> "${task_log}"
    echo "  GPU(s): ${assigned_gpus}" >> "${task_log}"
    echo "  Params: ${params}" >> "${task_log}"

    # Set GPU(s) for this task - use assigned_gpus from task file (multi-GPU support)
    # This is set once here and inherited by all task function commands
    # This must be unconditional to ensure multi-GPU tasks get all their GPUs
    export CUDA_VISIBLE_DEVICES="${assigned_gpus}"
    export TASK_ID="${task_id}"
    export TASK_PARAMS="${params}"
    export TASK_TYPE="${task_type}"

    # Set PM acceptance range to avoid gate failures during validation
    # These bounds are calibrated for typical validation runs; adjust if needed
    export INVARLOCK_PM_ACCEPTANCE_MIN="${INVARLOCK_PM_ACCEPTANCE_MIN:-0.90}"
    export INVARLOCK_PM_ACCEPTANCE_MAX="${INVARLOCK_PM_ACCEPTANCE_MAX:-1.20}"

    local task_pid_file=""
    if [[ -n "${QUEUE_DIR:-}" && -d "${QUEUE_DIR}/running" ]]; then
        task_pid_file="${QUEUE_DIR}/running/${task_id}.pid"
    fi

    local exit_code=0
    local task_timeout=""
    task_timeout=$(_get_task_timeout "${task_type}")

    local job_control_enabled=0
    case $- in
        *m*)
            job_control_enabled=1
            ;;
    esac
    if [[ ${job_control_enabled} -eq 0 ]]; then
        set -m
    fi

    (
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
            CREATE_EDITS_BATCH)
                # v2.1.0: Batch edit creation - loads model once, creates all 8 edits
                local edit_specs=$(echo "${params}" | jq -r '.edit_specs // "[]"')
                task_create_edits_batch "${model_name}" "${gpu_id}" "${edit_specs}" "${output_dir}" "${task_log}" || exit_code=$?
                ;;
            EVAL_EDIT)
                local edit_spec=$(echo "${params}" | jq -r '.edit_spec // ""')
                task_eval_edit "${model_name}" "${gpu_id}" "${edit_spec}" "${output_dir}" "${task_log}" || exit_code=$?
                ;;
            EVAL_MMLU|EVAL_HELLASWAG|EVAL_ARC|EVAL_WINOGRANDE)
                # v2.1.0: Split eval tasks - individual benchmarks for better parallelism
                local edit_spec=$(echo "${params}" | jq -r '.edit_spec // ""')
                local benchmark=$(echo "${params}" | jq -r '.benchmark // ""')
                task_eval_single_benchmark "${model_name}" "${gpu_id}" "${edit_spec}" "${benchmark}" "${output_dir}" "${task_log}" || exit_code=$?
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

        exit ${exit_code}
    ) &

    local task_pid=$!
    if [[ ${job_control_enabled} -eq 0 ]]; then
        set +m
    fi
    if [[ -n "${task_pid_file}" ]]; then
        echo "${task_pid}" > "${task_pid_file}"
    fi

    local timeout_marker=""
    local timeout_pid=""
    if [[ -n "${task_timeout}" ]]; then
        timeout_marker="${output_dir}/logs/tasks/${task_id}.timeout"
        rm -f "${timeout_marker}"
        (
            _sleep "${task_timeout}"
            if _cmd_kill -0 "${task_pid}" 2>/dev/null; then
                echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Task ${task_id} exceeded timeout (${task_timeout}s), terminating" >> "${task_log}"
                echo "${task_timeout}" > "${timeout_marker}"
                _kill_task_process_group "${task_pid}"
            fi
        ) &
        timeout_pid=$!
    fi

    exit_code=0
    wait "${task_pid}" || exit_code=$?

    if [[ -n "${timeout_pid}" ]]; then
        _cmd_kill "${timeout_pid}" 2>/dev/null || true
        wait "${timeout_pid}" 2>/dev/null || true
    fi

    if [[ -n "${timeout_marker}" && -f "${timeout_marker}" ]]; then
        exit_code=124
        rm -f "${timeout_marker}"
    fi

    if [[ -n "${task_pid_file}" ]]; then
        rm -f "${task_pid_file}"
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Task ${task_id} finished with exit code: ${exit_code}" >> "${task_log}"

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

    mkdir -p "${model_output_dir}"/{models,evals,certificates}

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
        PACK_MODEL_REVISION="${revision}" _cmd_python << SETUP_EOF >> "${log_file}" 2>&1 || exit_code=$?
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import gc
import os
import sys

model_id = "${model_id}"
output_dir = Path("${baseline_dir}")
output_dir.mkdir(parents=True, exist_ok=True)

revision = os.environ.get("PACK_MODEL_REVISION") or None
rev_label = f"@{revision}" if revision else ""
print(f"Downloading {model_id}{rev_label}...")

try:
    mode = os.environ.get("PACK_DETERMINISM", "").strip().lower()
    if mode == "strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif mode == "throughput":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        revision=revision,
    )
    tokenizer.save_pretrained(output_dir)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        revision=revision,
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

# ============ TASK: EVAL_BASELINE ============

# Run lm-eval on baseline model
# Usage: task_eval_baseline <model_name> <gpu_id> <output_dir> <log_file>
task_eval_baseline() {
    local model_name="$1"
    local gpu_id="$2"
    local output_dir="$3"
    local log_file="$4"

    local model_output_dir="${output_dir}/${model_name}"
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)
    local model_id=$(cat "${model_output_dir}/.model_id" 2>/dev/null || true)
    local result_file="${model_output_dir}/evals/baseline_results.json"

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    if [[ -f "${result_file}" ]]; then
        echo "  Baseline eval already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Running baseline lm-eval" >> "${log_file}"

    mkdir -p "$(dirname "${result_file}")"

    # Determine batch size based on model size
    local model_size
    model_size=$(_estimate_model_size "${baseline_path}")
    if [[ -z "${model_size}" || "${model_size}" == "7" ]] && [[ -n "${model_id}" ]]; then
        model_size=$(_get_model_size_from_name "${model_id}")
    fi
    local batch_size
    batch_size=$(_get_eval_batch_size "${model_size}")
    local params_json="${TASK_PARAMS:-}"
    if [[ -n "${params_json}" && "${params_json}" != "null" ]]; then
        local override_batch
        override_batch=$(echo "${params_json}" | jq -r '.batch_size // empty' 2>/dev/null)
        if [[ -n "${override_batch}" && "${override_batch}" != "null" ]]; then
            batch_size="${override_batch}"
            echo "  OOM override: batch_size=${batch_size}" >> "${log_file}"
        fi
    fi

    local model_args
    model_args=$(_get_lmeval_model_args "${baseline_path}")
    local torch_compile="${LMEVAL_TORCH_COMPILE:-0}"

    local tmp_eval_dir="${model_output_dir}/evals/.tmp/${TASK_ID:-baseline_$$}"
    mkdir -p "${tmp_eval_dir}"

    # CUDA_VISIBLE_DEVICES is inherited from execute_task() for multi-GPU support
    local exit_code=0
    TORCH_COMPILE="${torch_compile}" _cmd_python -m lm_eval \
        --model hf \
        --model_args "${model_args}" \
        --tasks "${EVAL_TASKS:-mmlu,hellaswag,arc_challenge,winogrande}" \
        --batch_size "${batch_size}" \
        --num_fewshot "${EVAL_NUM_FEWSHOT:-5}" \
        --output_path "${tmp_eval_dir}" \
        --log_samples \
        >> "${log_file}" 2>&1 || exit_code=$?

    # Move results file to expected location
    local found_results=$(find "${tmp_eval_dir}" -name "results*.json" -type f 2>/dev/null | head -1)
    if [[ -n "${found_results}" && -f "${found_results}" ]]; then
        mv "${found_results}" "${result_file}" 2>/dev/null || {
            echo "  ERROR: Failed to move results to: ${result_file}" >> "${log_file}"
            return 1
        }
        rm -rf "${tmp_eval_dir}" 2>/dev/null || true
        echo "  Results saved to: ${result_file}" >> "${log_file}"
    else
        echo "  ERROR: No results found in ${tmp_eval_dir}" >> "${log_file}"
        [[ ${exit_code} -eq 0 ]] && exit_code=1
    fi

        return ${exit_code}
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
    local min_windows="${INVARLOCK_CERT_MIN_WINDOWS:-192}"
    if [[ "${profile_flag}" == "ci" && "${min_windows}" =~ ^[0-9]+$ && "${min_windows}" -gt 0 ]]; then
        if [[ "${preview_n}" -lt "${min_windows}" || "${final_n}" -lt "${min_windows}" ]]; then
            preview_n="${min_windows}"
            final_n="${min_windows}"
            applied_override=1
            echo "  CI window override: preview=${preview_n}, final=${final_n}" >> "${log_file}"
        fi
    fi
    local bootstrap_replicates=2000


    if _is_large_model "${model_size}"; then
        bootstrap_replicates=1000
    fi
    if [[ -n "${INVARLOCK_BOOTSTRAP_N:-}" ]]; then
        bootstrap_replicates="${INVARLOCK_BOOTSTRAP_N}"
    fi

    echo "  Calibration: enforcing window_overlap_fraction=0.0" >> "${log_file}"
    local -a extra_env=(INVARLOCK_WINDOW_OVERLAP_FRACTION=0.0 INVARLOCK_SKIP_OVERHEAD_CHECK=1)
    if _is_large_model "${model_size}"; then
        echo "  Large model (${model_size}): SKIP_OVERHEAD_CHECK=1" >> "${log_file}"
    fi

    local config_root_base
    config_root_base="$(cd "${run_dir}" && pwd)"
    local config_root="${config_root_base}/config_root"
    mkdir -p "${config_root}/runtime/profiles"
    cat > "${config_root}/runtime/profiles/ci.yaml" << YAML
model:
  device_map: "auto"
dataset:
  preview_n: ${preview_n}
  final_n: ${final_n}
eval:
  bootstrap:
    replicates: ${bootstrap_replicates}
    alpha: 0.05
YAML

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
    if [[ -z "${guards_order_yaml}" ]]; then
        guards_order_yaml=$'    - invariants\n    - spectral\n    - rmt\n    - variance\n    - invariants\n'
    fi

    cat > "${config_yaml}" << YAML_EOF
model:
  id: "${baseline_path}"
  adapter: "hf_causal_auto"
  device: "auto"
  device_map: "auto"
  torch_dtype: "bfloat16"
  trust_remote_code: true
  low_cpu_mem_usage: true

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
${guards_order_yaml}

eval:
  bootstrap:
    replicates: ${bootstrap_replicates}
    parallel: true
  batch_size: ${eval_batch}
  window_overlap_fraction: 0.0

auto:
  enabled: true
  tier: "${INVARLOCK_TIER:-balanced}"
  probes: 0
YAML_EOF

    local profile_flag="ci"

    # CUDA_VISIBLE_DEVICES is inherited from execute_task() for multi-GPU support
    local exit_code=0
    env "${extra_env[@]}" invarlock run \
        --config "${config_yaml}" \
        --profile "${profile_flag}" \
        --out "${run_dir}" \
        >> "${log_file}" 2>&1 || exit_code=$?

    # Copy report to standard location
    local report_file=$(find "${run_dir}" -name "report*.json" -type f 2>/dev/null | head -1)
    if [[ -n "${report_file}" ]]; then
        cp "${report_file}" "${run_dir}/baseline_report.json" 2>/dev/null || true
        _cmd_python << CERT_EOF >> "${log_file}" 2>&1 || true
import json
from pathlib import Path
try:
    from invarlock.reporting.certificate import make_certificate
    report_path = Path("${report_file}")
    cert_path = Path("${run_dir}") / "evaluation.cert.json"
    report = json.loads(report_path.read_text())
    cert = make_certificate(report, report)
    with open(cert_path, "w") as f:
        json.dump(cert, f, indent=2)
except Exception as e:
    print(f"Certificate generation warning: {e}")
CERT_EOF
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

    # Export for use in Python script
    export PRESET_SEQ_LEN="${seq_len}"
    export PRESET_STRIDE="${stride}"
    export PRESET_PREVIEW_N="${preview_n}"
    export PRESET_FINAL_N="${final_n}"

    local exit_code=0
    local proof_packs_dir
    proof_packs_dir="$(cd "${SCRIPT_DIR}/.." && pwd)"
    local generator="${proof_packs_dir}/python/preset_generator.py"
    _cmd_python "${generator}" \
        --cal-dir "${cal_dir}" \
        --preset-file "${preset_base}.yaml" \
        --model-name "${model_name}" \
        --model-path "${baseline_path}" \
        --tier "${INVARLOCK_TIER:-balanced}" \
        --dataset-provider "${INVARLOCK_DATASET:-wikitext2}" \
        --seq-len "${seq_len}" \
        --stride "${stride}" \
        --preview-n "${preview_n}" \
        --final-n "${final_n}" \
        --edit-types "quant_rtn,fp8_quant,magnitude_prune,lowrank_svd" \
        >> "${log_file}" 2>&1 || exit_code=$?

    return ${exit_code}
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
    if [[ -d "${edit_path}" && -f "${edit_path}/config.json" ]]; then
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
    if [[ -d "${edit_path}" && -f "${edit_path}/config.json" ]]; then
        echo "  Created: ${edit_path}" >> "${log_file}"
        return 0
    else
        echo "  ERROR: Failed to create edit" >> "${log_file}"
        return 1
    fi
}

# ============ TASK: CREATE_EDITS_BATCH ============

# Create all edited models with single model load (Batch optimization)
# This loads the baseline model once and creates all 8 edits, avoiding 8× model load overhead
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

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Creating batch edits (8 edits with single model load)" >> "${log_file}"
    echo "  Baseline: ${baseline_path}" >> "${log_file}"

    # Process each edit spec using Python for efficient batch creation
    # CUDA_VISIBLE_DEVICES is inherited from execute_task() for multi-GPU support
    local exit_code=0
    _cmd_python << BATCH_EDIT_EOF >> "${log_file}" 2>&1 || exit_code=$?
import gc
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

baseline_path = Path("${baseline_path}")
model_output_dir = Path("${model_output_dir}")
edit_specs_json = '''${edit_specs_json}'''

try:
    edit_specs = json.loads(edit_specs_json)
except json.JSONDecodeError as e:
    print(f"ERROR: Invalid edit_specs JSON: {e}", file=sys.stderr)
    sys.exit(1)

tuned_path = (os.environ.get("PACK_TUNED_EDIT_PARAMS_FILE") or "").strip()
model_id = ""
model_id_path = model_output_dir / ".model_id"
if model_id_path.exists():
    try:
        model_id = model_id_path.read_text().strip()
    except Exception:
        model_id = ""
model_key = model_id or model_output_dir.name
tuned_params_by_type = {}
tuned_defaults = {}
if tuned_path and Path(tuned_path).exists():
    try:
        data = json.loads(Path(tuned_path).read_text())
    except Exception:
        data = {}
    if isinstance(data, dict):
        model_map = {}
        models = data.get("models")
        if isinstance(models, dict):
            model_map = (
                models.get(model_key)
                or models.get(model_id)
                or models.get(model_output_dir.name)
                or {}
            )
        if not model_map and isinstance(data.get("quant_rtn"), dict):
            model_map = data
        if isinstance(model_map, dict):
            tuned_params_by_type = model_map
        defaults = data.get("defaults")
        if isinstance(defaults, dict):
            tuned_defaults = defaults

print(f"Loading baseline model once for {len(edit_specs)} edits...")

mode = os.environ.get("PACK_DETERMINISM", "").strip().lower()
if mode == "strict":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
elif mode == "throughput":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.set_grad_enabled(False)

tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    baseline_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    low_cpu_mem_usage=True,
)

print(f"Baseline loaded. Creating {len(edit_specs)} edits...")

def parse_edit_spec(spec_str):
    parts = spec_str.split(":")
    edit_type = parts[0] if parts else ""

    def _clean_entry():
        entry = tuned_params_by_type.get(edit_type) or tuned_defaults.get(edit_type) or {}
        if not isinstance(entry, dict):
            entry = {}
        status = str(entry.get("status") or "missing")
        return entry, status

    if edit_type == "quant_rtn":
        if len(parts) > 1 and parts[1] == "clean":
            entry, status = _clean_entry()
            if status == "skipped":
                return {"type": edit_type, "skip": True, "reason": status}
            if status != "selected":
                return {"type": edit_type, "error": status}
            return {
                "type": "quant_rtn",
                "bits": int(entry.get("bits", 8)),
                "group_size": int(entry.get("group_size", 128)),
                "scope": entry.get("scope", parts[2] if len(parts) > 2 else "ffn"),
                "edit_dir_name": entry.get("edit_dir_name"),
            }
        return {"type": "quant_rtn", "bits": int(parts[1]), "group_size": int(parts[2]), "scope": parts[3]}
    if edit_type == "fp8_quant":
        if len(parts) > 1 and parts[1] == "clean":
            entry, status = _clean_entry()
            if status == "skipped":
                return {"type": edit_type, "skip": True, "reason": status}
            if status != "selected":
                return {"type": edit_type, "error": status}
            return {
                "type": "fp8_quant",
                "format": entry.get("format", "e4m3fn"),
                "scope": entry.get("scope", parts[2] if len(parts) > 2 else "ffn"),
                "edit_dir_name": entry.get("edit_dir_name"),
            }
        return {"type": "fp8_quant", "format": parts[1], "scope": parts[2]}
    if edit_type == "magnitude_prune":
        if len(parts) > 1 and parts[1] == "clean":
            entry, status = _clean_entry()
            if status == "skipped":
                return {"type": edit_type, "skip": True, "reason": status}
            if status != "selected":
                return {"type": edit_type, "error": status}
            return {
                "type": "magnitude_prune",
                "ratio": float(entry.get("sparsity", 0.0)),
                "scope": entry.get("scope", parts[2] if len(parts) > 2 else "ffn"),
                "edit_dir_name": entry.get("edit_dir_name"),
            }
        return {"type": "magnitude_prune", "ratio": float(parts[1]), "scope": parts[2]}
    if edit_type == "lowrank_svd":
        if len(parts) > 1 and parts[1] == "clean":
            entry, status = _clean_entry()
            if status == "skipped":
                return {"type": edit_type, "skip": True, "reason": status}
            if status != "selected":
                return {"type": edit_type, "error": status}
            return {
                "type": "lowrank_svd",
                "rank": int(entry.get("rank", 0)),
                "scope": entry.get("scope", parts[2] if len(parts) > 2 else "ffn"),
                "edit_dir_name": entry.get("edit_dir_name"),
            }
        return {"type": "lowrank_svd", "rank": int(parts[1]), "scope": parts[2]}

    return {"type": edit_type, "params": parts[1:]}

def get_edit_dir_name(parsed_spec, version):
    if parsed_spec.get("edit_dir_name"):
        return parsed_spec["edit_dir_name"]
    t = parsed_spec["type"]
    if t == "quant_rtn":
        return f"quant_{parsed_spec['bits']}bit_{version}"
    if t == "fp8_quant":
        return f"fp8_{parsed_spec['format']}_{version}"
    if t == "magnitude_prune":
        pct = int(parsed_spec["ratio"] * 100)
        return f"prune_{pct}pct_{version}"
    if t == "lowrank_svd":
        return f"svd_rank{parsed_spec['rank']}_{version}"
    return f"{t}_{version}"

def _target_modules(scope):
    if scope == "ffn":
        return ["mlp", "feed_forward", "ffn"]
    if scope == "all":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "mlp", "gate", "up", "down"]
    return []

def apply_quantization(model, bits, group_size, scope):
    import copy
    edited = copy.deepcopy(model)
    target_modules = _target_modules(scope)

    qmin = -(2 ** (bits - 1))
    qmax = max((2 ** (bits - 1)) - 1, 1)
    for name, param in edited.named_parameters():
        if not any(t in name.lower() for t in target_modules):
            continue
        if param.dim() < 2:
            continue
        orig_shape = param.shape
        flat = param.reshape(orig_shape[0], -1)
        in_features = flat.shape[1]
        eff_group_size = group_size if group_size > 0 else in_features
        if eff_group_size >= in_features:
            eff_group_size = in_features
        num_groups = (in_features + eff_group_size - 1) // eff_group_size
        pad = (num_groups * eff_group_size) - in_features
        if pad > 0:
            flat = torch.nn.functional.pad(flat, (0, pad))
        grouped = flat.reshape(orig_shape[0], num_groups, eff_group_size)
        max_abs = grouped.abs().amax(dim=-1, keepdim=True)
        scale = torch.clamp(max_abs / qmax, min=1e-10)
        quantized = torch.round(grouped / scale).clamp(qmin, qmax) * scale
        quantized = quantized.reshape(orig_shape[0], num_groups * eff_group_size)
        if pad > 0:
            quantized = quantized[:, :in_features]
        param.data = quantized.reshape(orig_shape).to(param.dtype)
    return edited

def apply_pruning(model, ratio, scope):
    import copy
    edited = copy.deepcopy(model)
    target_modules = _target_modules(scope)

    for name, param in edited.named_parameters():
        if not any(t in name.lower() for t in target_modules):
            continue
        if param.dim() < 2:
            continue
        # quantile only supports float32/float64 reliably, so cast to float before computing threshold
        param_abs = param.detach().float().abs()
        flat = param_abs.view(-1)
        if flat.numel() > 10_000_000:
            sample_size = min(1_000_000, flat.numel())
            idx = torch.randint(0, flat.numel(), (sample_size,), device=flat.device)
            flat_for_quantile = flat[idx]
        else:
            flat_for_quantile = flat
        threshold = torch.quantile(flat_for_quantile, ratio)
        mask = param_abs > threshold
        param.data = (param * mask).to(param.dtype)
    return edited

def apply_lowrank(model, rank, scope):
    import copy
    edited = copy.deepcopy(model)
    target_modules = _target_modules(scope)

    for name, param in edited.named_parameters():
        if not any(t in name.lower() for t in target_modules):
            continue
        if param.dim() != 2:
            continue
        if min(param.shape) <= rank:
            continue
        W = param.data.float()
        k = min(rank, min(W.shape))
        U, S, V = torch.svd_lowrank(W, q=k, niter=2)
        param.data = ((U * S) @ V.T).to(param.dtype)
    return edited

def apply_fp8(model, format_type, scope):
    import copy
    edited = copy.deepcopy(model)
    target_modules = _target_modules(scope)

    dtype = None
    if format_type in {"e4m3", "e4m3fn", "e4m3fnuz"}:
        dtype = getattr(torch, "float8_e4m3fn", None)
    elif format_type in {"e5m2", "e5m2fn", "e5m2fnuz"}:
        dtype = getattr(torch, "float8_e5m2", None)

    def _quantize(tensor):
        if dtype is None:
            return tensor.to(torch.float16).to(tensor.dtype)
        return tensor.to(dtype).to(tensor.dtype)

    for name, param in edited.named_parameters():
        if not any(t in name.lower() for t in target_modules):
            continue
        if param.dim() < 2:
            continue
        param.data = _quantize(param.data)
    return edited

created_count = 0
failed_count = 0

for spec_entry in edit_specs:
    spec_str = spec_entry.get("spec", "")
    version = spec_entry.get("version", "clean")

    parsed = parse_edit_spec(spec_str)
    if parsed.get("skip"):
        print(f"  Skip (tuned edit preset skipped): {spec_str}")
        continue
    if parsed.get("error"):
        raise ValueError(f"Tuned edit preset missing for {spec_str}: {parsed['error']}")
    edit_dir_name = get_edit_dir_name(parsed, version)
    edit_path = model_output_dir / "models" / edit_dir_name

    if (edit_path / "config.json").exists():
        print(f"  Skip (exists): {edit_dir_name}")
        created_count += 1
        continue

    print(f"  Creating: {edit_dir_name}...")

    try:
        edit_path.mkdir(parents=True, exist_ok=True)

        t = parsed["type"]
        if t == "quant_rtn":
            edited_model = apply_quantization(model, parsed["bits"], parsed["group_size"], parsed["scope"])
        elif t == "magnitude_prune":
            edited_model = apply_pruning(model, parsed["ratio"], parsed["scope"])
        elif t == "lowrank_svd":
            edited_model = apply_lowrank(model, parsed["rank"], parsed["scope"])
        elif t == "fp8_quant":
            edited_model = apply_fp8(model, parsed["format"], parsed["scope"])
        else:
            raise ValueError(f"Unknown edit type: {t}")

        edited_model.save_pretrained(edit_path, safe_serialization=True)
        tokenizer.save_pretrained(edit_path)

        del edited_model
        gc.collect()
        torch.cuda.empty_cache()

        print(f"    Saved: {edit_path}")
        created_count += 1

    except Exception as e:
        print(f"    ERROR: {e}", file=sys.stderr)
        failed_count += 1

del model
gc.collect()
torch.cuda.empty_cache()

print(f"Batch complete: {created_count} created, {failed_count} failed")

if failed_count > 0:
    sys.exit(1)
BATCH_EDIT_EOF

    if [[ ${exit_code} -eq 0 ]]; then
        echo "  Batch edit creation complete" >> "${log_file}"
    else
        echo "  ERROR: Batch edit creation failed" >> "${log_file}"
    fi

    return ${exit_code}
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

    local edit_path=""
    if [[ "${param1}" == "clean" ]]; then
        local resolved
        resolved=$(resolve_edit_params "${model_output_dir}" "${edit_spec}" "clean")
        local status
        status=$(echo "${resolved}" | jq -r '.status')
        if [[ "${status}" == "skipped" ]]; then
            echo "  Clean edit skipped by tuned preset: ${edit_spec}" >> "${log_file}"
            return 0
        fi
        if [[ "${status}" != "selected" ]]; then
            echo "ERROR: Unable to resolve tuned clean edit for ${edit_spec} (${status})" >> "${log_file}"
            return 1
        fi
        local edit_dir_name
        edit_dir_name=$(echo "${resolved}" | jq -r '.edit_dir_name')
        edit_path="${model_output_dir}/models/${edit_dir_name}"
    else
        for version in clean stress; do
            local resolved
            resolved=$(resolve_edit_params "${model_output_dir}" "${edit_spec}" "${version}")
            local status
            status=$(echo "${resolved}" | jq -r '.status')
            if [[ "${status}" != "selected" ]]; then
                continue
            fi
            local edit_dir_name
            edit_dir_name=$(echo "${resolved}" | jq -r '.edit_dir_name')
            local potential_path="${model_output_dir}/models/${edit_dir_name}"
            if [[ -d "${potential_path}" ]]; then
                edit_path="${potential_path}"
                break
            fi
        done
    fi

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

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Running lm-eval on: ${edit_name}" >> "${log_file}"

    mkdir -p "$(dirname "${result_file}")"

    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)
    local model_id=$(cat "${model_output_dir}/.model_id" 2>/dev/null || true)
    local model_size
    model_size=$(_estimate_model_size "${baseline_path}")
    if [[ -z "${model_size}" || "${model_size}" == "7" ]] && [[ -n "${model_id}" ]]; then
        model_size=$(_get_model_size_from_name "${model_id}")
    fi
    local batch_size
    batch_size=$(_get_eval_batch_size "${model_size}")
    local params_json="${TASK_PARAMS:-}"
    if [[ -n "${params_json}" && "${params_json}" != "null" ]]; then
        local override_batch
        override_batch=$(echo "${params_json}" | jq -r '.batch_size // empty' 2>/dev/null)
        if [[ -n "${override_batch}" && "${override_batch}" != "null" ]]; then
            batch_size="${override_batch}"
            echo "  OOM override: batch_size=${batch_size}" >> "${log_file}"
        fi
    fi
    local model_args
    model_args=$(_get_lmeval_model_args "${edit_path}")
    local torch_compile="${LMEVAL_TORCH_COMPILE:-0}"
    local tmp_eval_dir="${model_output_dir}/evals/.tmp/${TASK_ID:-${edit_name}_$$}"
    mkdir -p "${tmp_eval_dir}"

    # CUDA_VISIBLE_DEVICES is inherited from execute_task() for multi-GPU support
    local exit_code=0
    TORCH_COMPILE="${torch_compile}" _cmd_python -m lm_eval \
        --model hf \
        --model_args "${model_args}" \
        --tasks "${EVAL_TASKS:-mmlu,hellaswag,arc_challenge,winogrande}" \
        --batch_size "${batch_size}" \
        --num_fewshot "${EVAL_NUM_FEWSHOT:-5}" \
        --output_path "${tmp_eval_dir}" \
        --log_samples \
        >> "${log_file}" 2>&1 || exit_code=$?

    local found_results=$(find "${tmp_eval_dir}" -name "results*.json" -type f 2>/dev/null | head -1)
    if [[ -n "${found_results}" && -f "${found_results}" ]]; then
        mv "${found_results}" "${result_file}" 2>/dev/null || {
            echo "  ERROR: Failed to move results to: ${result_file}" >> "${log_file}"
            return 1
        }
        rm -rf "${tmp_eval_dir}" 2>/dev/null || true
        echo "  Results saved to: ${result_file}" >> "${log_file}"
    else
        echo "  ERROR: No results found in ${tmp_eval_dir}" >> "${log_file}"
        [[ ${exit_code} -eq 0 ]] && exit_code=1
    fi

    return ${exit_code}
}

# ============ TASK: EVAL_SINGLE_BENCHMARK ============

# Run a single lm-eval benchmark on edited model (Split Eval optimization)
# This runs one benchmark (MMLU, HellaSwag, ARC, or WinoGrande) instead of all 4
# Enables 4× parallelism: each benchmark can run on a different GPU
# Usage: task_eval_single_benchmark <model_name> <gpu_id> <edit_spec> <benchmark> <output_dir> <log_file>
task_eval_single_benchmark() {
    local model_name="$1"
    local gpu_id="$2"
    local edit_spec="$3"
    local benchmark="$4"
    local output_dir="$5"
    local log_file="$6"

    local model_output_dir="${output_dir}/${model_name}"

    # Parse edit spec to find path
    local edit_type param1
    IFS=':' read -r edit_type param1 _ _ <<< "${edit_spec}"

    local edit_path=""
    if [[ "${param1}" == "clean" ]]; then
        local resolved
        resolved=$(resolve_edit_params "${model_output_dir}" "${edit_spec}" "clean")
        local status
        status=$(echo "${resolved}" | jq -r '.status')
        if [[ "${status}" == "skipped" ]]; then
            echo "  Clean edit skipped by tuned preset: ${edit_spec}" >> "${log_file}"
            return 0
        fi
        if [[ "${status}" != "selected" ]]; then
            echo "ERROR: Unable to resolve tuned clean edit for ${edit_spec} (${status})" >> "${log_file}"
            return 1
        fi
        local edit_dir_name
        edit_dir_name=$(echo "${resolved}" | jq -r '.edit_dir_name')
        edit_path="${model_output_dir}/models/${edit_dir_name}"
    else
        for version in clean stress; do
            local resolved
            resolved=$(resolve_edit_params "${model_output_dir}" "${edit_spec}" "${version}")
            local status
            status=$(echo "${resolved}" | jq -r '.status')
            if [[ "${status}" != "selected" ]]; then
                continue
            fi
            local edit_dir_name
            edit_dir_name=$(echo "${resolved}" | jq -r '.edit_dir_name')
            local potential_path="${model_output_dir}/models/${edit_dir_name}"
            if [[ -d "${potential_path}" ]]; then
                edit_path="${potential_path}"
                break
            fi
        done
    fi

    if [[ -z "${edit_path}" || ! -d "${edit_path}" ]]; then
        echo "ERROR: Edit model not found for spec: ${edit_spec}" >> "${log_file}"
        return 1
    fi

    local edit_name=$(basename "${edit_path}")
    local result_file="${model_output_dir}/evals/${edit_name}_${benchmark}_results.json"

    if [[ -f "${result_file}" ]]; then
        echo "  Eval ${benchmark} for ${edit_name} already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Running lm-eval ${benchmark} on: ${edit_name}" >> "${log_file}"

    mkdir -p "$(dirname "${result_file}")"

    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)
    local model_id=$(cat "${model_output_dir}/.model_id" 2>/dev/null || true)
    local model_size
    model_size=$(_estimate_model_size "${baseline_path}")
    if [[ -z "${model_size}" || "${model_size}" == "7" ]] && [[ -n "${model_id}" ]]; then
        model_size=$(_get_model_size_from_name "${model_id}")
    fi
    local batch_size
    batch_size=$(_get_eval_batch_size "${model_size}")
    local params_json="${TASK_PARAMS:-}"
    if [[ -n "${params_json}" && "${params_json}" != "null" ]]; then
        local override_batch
        override_batch=$(echo "${params_json}" | jq -r '.batch_size // empty' 2>/dev/null)
        if [[ -n "${override_batch}" && "${override_batch}" != "null" ]]; then
            batch_size="${override_batch}"
            echo "  OOM override: batch_size=${batch_size}" >> "${log_file}"
        fi
    fi
    local model_args
    model_args=$(_get_lmeval_model_args "${edit_path}")
    local torch_compile="${LMEVAL_TORCH_COMPILE:-0}"
    local tmp_eval_dir="${model_output_dir}/evals/.tmp/${TASK_ID:-${edit_name}_${benchmark}_$$}"
    mkdir -p "${tmp_eval_dir}"

    # Map benchmark names to lm-eval task names
    local task_name
    case "${benchmark}" in
        "mmlu")
            task_name="mmlu"
            ;;
        "hellaswag")
            task_name="hellaswag"
            ;;
        "arc")
            task_name="arc_challenge"
            ;;
        "winogrande")
            task_name="winogrande"
            ;;
        *)
            task_name="${benchmark}"
            ;;
    esac

    # CUDA_VISIBLE_DEVICES is inherited from execute_task() for multi-GPU support
    local exit_code=0
    TORCH_COMPILE="${torch_compile}" _cmd_python -m lm_eval \
        --model hf \
        --model_args "${model_args}" \
        --tasks "${task_name}" \
        --batch_size "${batch_size}" \
        --num_fewshot "${EVAL_NUM_FEWSHOT:-5}" \
        --output_path "${tmp_eval_dir}" \
        --log_samples \
        >> "${log_file}" 2>&1 || exit_code=$?

    # Move results file to expected location
    local found_results=$(find "${tmp_eval_dir}" -name "results*.json" -type f 2>/dev/null | head -1)
    if [[ -n "${found_results}" && -f "${found_results}" ]]; then
        mv "${found_results}" "${result_file}" 2>/dev/null || {
            echo "  ERROR: Failed to move results to: ${result_file}" >> "${log_file}"
            return 1
        }
        rm -rf "${tmp_eval_dir}" 2>/dev/null || true
        echo "  Results saved to: ${result_file}" >> "${log_file}"
    else
        echo "  ERROR: No results found in ${tmp_eval_dir}" >> "${log_file}"
        [[ ${exit_code} -eq 0 ]] && exit_code=1
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
    local edit_dir_name
    edit_dir_name=$(echo "${resolved}" | jq -r '.edit_dir_name')

    local edit_path="${model_output_dir}/models/${edit_dir_name}"
    local cert_dir="${model_output_dir}/certificates/${edit_dir_name}/run_${run_num}"
    local cert_file="${cert_dir}/evaluation.cert.json"

    if [[ ! -d "${edit_path}" ]]; then
        echo "ERROR: Edit model not found: ${edit_path}" >> "${log_file}"
        return 1
    fi

    local abs_baseline_path
    abs_baseline_path="$(cd "$(dirname "${baseline_path}")" && pwd)/$(basename "${baseline_path}")"
    local abs_edit_path
    abs_edit_path="$(cd "$(dirname "${edit_path}")" && pwd)/$(basename "${edit_path}")"

    if [[ -f "${cert_file}" ]]; then
        echo "  Certification for ${edit_dir_name} run ${run_num} already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Certifying: ${edit_dir_name} run ${run_num}" >> "${log_file}"

    mkdir -p "${cert_dir}"

    local abs_cert_dir
    abs_cert_dir="$(cd "${cert_dir}" && pwd)"
    local abs_log_file
    abs_log_file="$(cd "$(dirname "${log_file}")" && pwd)/$(basename "${log_file}")"
    cert_dir="${abs_cert_dir}"
    cert_file="${cert_dir}/evaluation.cert.json"
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
    local min_windows="${INVARLOCK_CERT_MIN_WINDOWS:-192}"
    if [[ "${profile_flag}" == "ci" && "${min_windows}" =~ ^[0-9]+$ && "${min_windows}" -gt 0 ]]; then
        if [[ "${preview_n}" -lt "${min_windows}" || "${final_n}" -lt "${min_windows}" ]]; then
            preview_n="${min_windows}"
            final_n="${min_windows}"
            applied_override=1
            echo "  CI window override: preview=${preview_n}, final=${final_n}" >> "${log_file}"
        fi
    fi
    local bootstrap_replicates=2000
    if _is_large_model "${model_size}"; then
        bootstrap_replicates=1000
    fi
    if [[ -n "${INVARLOCK_BOOTSTRAP_N:-}" ]]; then
        bootstrap_replicates="${INVARLOCK_BOOTSTRAP_N}"
    fi

    local -a extra_env=()
    if _is_large_model "${model_size}"; then
        extra_env+=(INVARLOCK_SKIP_OVERHEAD_CHECK=1)
        echo "  Large model (${model_size}): SKIP_OVERHEAD_CHECK=1" >> "${log_file}"
    fi
    extra_env+=(INVARLOCK_STORE_EVAL_WINDOWS=1)

    local config_root_base
    config_root_base="$(cd "${cert_dir}" && pwd)"
    local config_root="${config_root_base}/config_root"
    mkdir -p "${config_root}/runtime/profiles"
    cat > "${config_root}/runtime/profiles/ci.yaml" << YAML
model:
  device_map: "auto"
dataset:
  seq_len: ${seq_len}
  stride: ${stride}
  preview_n: ${preview_n}
  final_n: ${final_n}
eval:
  bootstrap:
    replicates: ${bootstrap_replicates}
    alpha: 0.05
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
    local edit_label
    edit_label="$(basename "${abs_edit_path}")"
    edit_label="${edit_label%_stress}"
    edit_label="${edit_label%_clean}"
    if [[ -z "${edit_label}" ]]; then
        edit_label="custom"
    fi

    # CUDA_VISIBLE_DEVICES is inherited from execute_task() for multi-GPU support
    local exit_code=0
    (
        cd "${work_dir}" || exit 1
        env "${extra_env[@]}" invarlock certify \
            --source "${abs_baseline_path}" \
            --edited "${abs_edit_path}" \
            --edit-label "${edit_label}" \
            --profile "${profile_flag}" \
            --tier "${INVARLOCK_TIER:-balanced}" \
            --out "${cert_dir}" \
            --cert-out "${cert_dir}" \
            --preset "${abs_preset_file}" >> "${log_file}" 2>&1
    ) || exit_code=$?

    # Find and copy certificate (only the canonical cert)
    if [[ ! -f "${cert_file}" ]]; then
        local found_cert
        found_cert=$(find "${cert_dir}" -name "evaluation.cert.json" -type f 2>/dev/null | head -1)
        if [[ -n "${found_cert}" && -f "${found_cert}" && "${found_cert}" != "${cert_file}" ]]; then
            cp "${found_cert}" "${cert_file}" 2>/dev/null || true
        fi
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
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)
    local error_path="${model_output_dir}/models/error_${error_type}"

    if [[ -z "${baseline_path}" || ! -d "${baseline_path}" ]]; then
        echo "ERROR: Baseline path not found for ${model_name}" >> "${log_file}"
        return 1
    fi

    if [[ -d "${error_path}" && -f "${error_path}/config.json" ]]; then
        echo "  Error model ${error_type} already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Creating error model: ${error_type}" >> "${log_file}"

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
    local baseline_path=$(cat "${model_output_dir}/.baseline_path" 2>/dev/null || true)
    local model_id=$(cat "${model_output_dir}/.model_id" 2>/dev/null || true)
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

    local abs_baseline_path
    abs_baseline_path="$(cd "$(dirname "${baseline_path}")" && pwd)/$(basename "${baseline_path}")"
    local abs_error_path
    abs_error_path="$(cd "$(dirname "${error_path}")" && pwd)/$(basename "${error_path}")"

    if [[ -f "${cert_file}" ]]; then
        echo "  Certification for error ${error_type} already exists, skipping" >> "${log_file}"
        return 0
    fi

    echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] Certifying error model: ${error_type}" >> "${log_file}"

    mkdir -p "${cert_dir}"

    local abs_cert_dir
    abs_cert_dir="$(cd "${cert_dir}" && pwd)"
    local abs_log_file
    abs_log_file="$(cd "$(dirname "${log_file}")" && pwd)/$(basename "${log_file}")"
    cert_dir="${abs_cert_dir}"
    cert_file="${cert_dir}/evaluation.cert.json"
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
    local min_windows="${INVARLOCK_CERT_MIN_WINDOWS:-192}"
    if [[ "${profile_flag}" == "ci" && "${min_windows}" =~ ^[0-9]+$ && "${min_windows}" -gt 0 ]]; then
        if [[ "${preview_n}" -lt "${min_windows}" || "${final_n}" -lt "${min_windows}" ]]; then
            preview_n="${min_windows}"
            final_n="${min_windows}"
            applied_override=1
            echo "  CI window override: preview=${preview_n}, final=${final_n}" >> "${log_file}"
        fi
    fi
    local bootstrap_replicates=2000
    if _is_large_model "${model_size}"; then
        bootstrap_replicates=1000
    fi
    if [[ -n "${INVARLOCK_BOOTSTRAP_N:-}" ]]; then
        bootstrap_replicates="${INVARLOCK_BOOTSTRAP_N}"
    fi

    local -a extra_env=()
    if _is_large_model "${model_size}"; then
        extra_env+=(INVARLOCK_SKIP_OVERHEAD_CHECK=1)
        echo "  Large model (${model_size}): SKIP_OVERHEAD_CHECK=1" >> "${log_file}"
    fi
    extra_env+=(INVARLOCK_STORE_EVAL_WINDOWS=1)

    local config_root_base
    config_root_base="$(cd "${cert_dir}" && pwd)"
    local config_root="${config_root_base}/config_root"
    mkdir -p "${config_root}/runtime/profiles"
    cat > "${config_root}/runtime/profiles/ci.yaml" << YAML
model:
  device_map: "auto"
dataset:
  seq_len: ${seq_len}
  stride: ${stride}
  preview_n: ${preview_n}
  final_n: ${final_n}
eval:
  bootstrap:
    replicates: ${bootstrap_replicates}
    alpha: 0.05
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

    # CUDA_VISIBLE_DEVICES is inherited from execute_task() for multi-GPU support
    local exit_code=0
    (
        cd "${work_dir}" || exit 1
        env "${extra_env[@]}" invarlock certify \
            --source "${abs_baseline_path}" \
            --edited "${abs_error_path}" \
            --profile "${profile_flag}" \
            --tier "${INVARLOCK_TIER:-balanced}" \
            --out "${cert_dir}" \
            --cert-out "${cert_dir}" \
            --preset "${abs_preset_file}" >> "${log_file}" 2>&1
    ) || exit_code=$?

    # Find and copy certificate (only the canonical cert)
    if [[ ! -f "${cert_file}" ]]; then
        local found_cert
        found_cert=$(find "${cert_dir}" -name "evaluation.cert.json" -type f 2>/dev/null | head -1)
        if [[ -n "${found_cert}" && -f "${found_cert}" && "${found_cert}" != "${cert_file}" ]]; then
            cp "${found_cert}" "${cert_file}" 2>/dev/null || true
        fi
    fi

    return ${exit_code}
}
