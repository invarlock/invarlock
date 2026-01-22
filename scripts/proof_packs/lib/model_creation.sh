#!/usr/bin/env bash
# model_creation.sh - Shared model creation helpers for workers and main script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime.sh
source "${SCRIPT_DIR}/runtime.sh"

# Directory for Python helpers (keeps model_creation.sh bash-only).
PROOF_PACK_PY_DIR="$(cd "${SCRIPT_DIR}/../python" && pwd)"

# Provide basic logging helpers when not sourced from the main script.
if ! declare -F log >/dev/null 2>&1; then
    :
    log() {
        echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] $*"
    }
fi

if ! declare -F error_exit >/dev/null 2>&1; then
    :
    error_exit() {
        echo "[$(_cmd_date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
        exit 1
    }
fi

_model_creation_run_python() {
    local parent_dir="$1"
    local cuda_devices="$2"
    shift 2

    if ! mkdir -p "${parent_dir}"; then
        return 1
    fi

    CUDA_VISIBLE_DEVICES="${cuda_devices}" _cmd_python "$@"
}

create_model_variant() {
    local baseline_path="$1"
    local output_path="$2"
    local edit_type="$3"
    local param1="${4:-}"
    local param2="${5:-}"
    local scope="${6:-}"
    local gpu_id="${7:-0}"

    [[ "${param1}" == "null" ]] && param1=""
    [[ "${param2}" == "null" ]] && param2=""
    [[ "${scope}" == "null" ]] && scope=""

    case "${edit_type}" in
        "quant_rtn")
            if [[ -z "${param1}" || -z "${param2}" || -z "${scope}" ]]; then
                echo "ERROR: quant_rtn requires bits, group_size, scope" >&2
                return 1
            fi
            create_edited_model "${baseline_path}" "${output_path}" "quant_rtn" "${param1}" "${param2}" "${scope}" "${gpu_id}"
            ;;
        "fp8_quant")
            if [[ -z "${param1}" || -z "${scope}" ]]; then
                echo "ERROR: fp8_quant requires format and scope" >&2
                return 1
            fi
            create_fp8_model "${baseline_path}" "${output_path}" "${param1}" "${scope}" "${gpu_id}"
            ;;
        "magnitude_prune")
            if [[ -z "${param1}" || -z "${scope}" ]]; then
                echo "ERROR: magnitude_prune requires sparsity and scope" >&2
                return 1
            fi
            create_pruned_model "${baseline_path}" "${output_path}" "${param1}" "${scope}" "${gpu_id}"
            ;;
        "lowrank_svd")
            if [[ -z "${param1}" || -z "${scope}" ]]; then
                echo "ERROR: lowrank_svd requires rank and scope" >&2
                return 1
            fi
            create_lowrank_model "${baseline_path}" "${output_path}" "${param1}" "${scope}" "${gpu_id}"
            ;;
        "error_injection")
            if [[ -z "${param1}" ]]; then
                echo "ERROR: error_injection requires error_type" >&2
                return 1
            fi
            create_error_model "${baseline_path}" "${output_path}" "${param1}" "${gpu_id}"
            ;;
        *)
            echo "ERROR: Unknown edit type: ${edit_type}" >&2
            return 1
            ;;
    esac
}
export -f create_model_variant

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

    if [[ "${edit_type}" == "quant_rtn" ]]; then
        local parent_dir
        parent_dir="$(dirname "${output_path}")"
        local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
        _model_creation_run_python \
            "${parent_dir}" \
            "${cuda_devices}" \
            "${PROOF_PACK_PY_DIR}/create_quant_rtn_model.py" \
            "${baseline_path}" \
            "${output_path}" \
            "${bits}" \
            "${group_size}" \
            "${scope}"
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

    local parent_dir
    parent_dir="$(dirname "${output_path}")"
    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    _model_creation_run_python \
        "${parent_dir}" \
        "${cuda_devices}" \
        "${PROOF_PACK_PY_DIR}/create_pruned_model.py" \
        "${baseline_path}" \
        "${output_path}" \
        "${sparsity}" \
        "${scope}"
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

    local parent_dir
    parent_dir="$(dirname "${output_path}")"
    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    _model_creation_run_python \
        "${parent_dir}" \
        "${cuda_devices}" \
        "${PROOF_PACK_PY_DIR}/create_lowrank_model.py" \
        "${baseline_path}" \
        "${output_path}" \
        "${rank}" \
        "${scope}"
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

    local parent_dir
    parent_dir="$(dirname "${output_path}")"
    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    _model_creation_run_python \
        "${parent_dir}" \
        "${cuda_devices}" \
        "${PROOF_PACK_PY_DIR}/create_fp8_model.py" \
        "${baseline_path}" \
        "${output_path}" \
        "${format}" \
        "${scope}"
}
export -f create_fp8_model

# ============ ERROR MODEL CREATION ============
create_error_model() {
    local baseline_path="$1"
    local output_path="$2"
    local error_type="$3"
    local gpu_id="${4:-0}"

    log "Creating error model (type=${error_type}, GPU ${gpu_id})"
    local parent_dir
    parent_dir="$(dirname "${output_path}")"
    local cuda_devices="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"
    _model_creation_run_python \
        "${parent_dir}" \
        "${cuda_devices}" \
        "${PROOF_PACK_PY_DIR}/create_error_model.py" \
        "${baseline_path}" \
        "${output_path}" \
        "${error_type}"
}
export -f create_error_model
