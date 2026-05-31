#!/usr/bin/env bash
# scheduler_gpu_runtime.sh - OOM checks, GPU memory probes, and purge helpers
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Usage: sourced by scheduler.sh

SCHEDULER_MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scheduler_core.sh
[[ -z "${SCHEDULER_CORE_LOADED:-}" ]] && source "${SCHEDULER_MODULE_DIR}/scheduler_core.sh"

# ============ OOM PROTECTION ============

# Pre-check if a task will fit in available GPU memory with safety margin
# Usage: check_oom_safe <task_file> <gpu_ids_csv>
# Returns: 0 if safe, 1 if OOM risk detected
check_oom_safe() {
    local task_file="$1"
    local gpu_ids="$2"
    gpu_ids="${gpu_ids// /}"
    local gpu_id
    local -a gpus=()

    if [[ ! -f "${task_file}" ]]; then
        return 1
    fi

    local model_size=$(get_task_field "${task_file}" "model_size_gb")

    [[ -z "${model_size}" ]] && model_size=20

    local gpu_count=1
    IFS=',' read -ra gpus <<< "${gpu_ids}"
    gpu_count=${#gpus[@]}
    [[ ${gpu_count} -lt 1 ]] && gpu_count=1

    # For multi-GPU tasks, assume sharded load across assigned GPUs.
    local mem_per_gpu=$(( (model_size + gpu_count - 1) / gpu_count ))

    # model_size_gb already includes task multipliers + safety margin.
    # Apply only a small extra headroom for fragmentation.
    local required_mem=$(awk -v base="${mem_per_gpu}" 'BEGIN { printf "%.0f", base * 1.05 }')

    # Check each assigned GPU has enough memory
    IFS=',' read -ra gpus <<< "${gpu_ids}"
    for gpu_id in "${gpus[@]}"; do
        local available=$(get_gpu_available_memory "${gpu_id}")
        if [[ ${available} -lt ${required_mem} ]]; then
            echo "[OOM_CHECK] GPU ${gpu_id} has ${available}GB free but task needs ~${required_mem}GB. RISK DETECTED." >&2
            return 1
        fi
    done

    return 0
}

# Get OOM risk level for a task
# Usage: get_oom_risk_level <task_file> <available_gpus_csv>
# Returns: "low", "medium", "high", or "critical"
get_oom_risk_level() {
    local task_file="$1"
    local gpu_ids="$2"
    gpu_ids="${gpu_ids// /}"
    local gpu_id
    local -a gpus=()

    local model_size=$(get_task_field "${task_file}" "model_size_gb")

    [[ -z "${model_size}" ]] && model_size=20

    local gpu_count=1
    IFS=',' read -ra gpus <<< "${gpu_ids}"
    gpu_count=${#gpus[@]}
    [[ ${gpu_count} -lt 1 ]] && gpu_count=1

    # For multi-GPU tasks, assume sharded load across assigned GPUs.
    local mem_per_gpu=$(( (model_size + gpu_count - 1) / gpu_count ))

    # Get minimum available memory across assigned GPUs
    local min_available=999999
    IFS=',' read -ra gpus <<< "${gpu_ids}"
    for gpu_id in "${gpus[@]}"; do
        local available=$(get_gpu_available_memory "${gpu_id}")
        [[ ${available} -lt ${min_available} ]] && min_available=${available}
    done

    # Calculate headroom percentage
    if [[ ${min_available} -le 0 ]]; then
        echo "critical"
        return
    fi
    local headroom=$(( (min_available - mem_per_gpu) * 100 / min_available ))

    if [[ ${headroom} -lt 5 ]]; then
        echo "critical"  # Less than 5% headroom - very likely OOM
    elif [[ ${headroom} -lt 15 ]]; then
        echo "high"      # 5-15% headroom - significant risk
    elif [[ ${headroom} -lt 30 ]]; then
        echo "medium"    # 15-30% headroom - some risk for memory-intensive ops
    else
        echo "low"       # >30% headroom - should be safe
    fi
}

# Clear GPU memory by running torch.cuda.empty_cache() equivalent
# Usage: purge_gpu_memory <gpu_id>
# Note: This runs a Python snippet to force CUDA memory cleanup
purge_gpu_memory() {
    local gpu_id="$1"

    # Run Python to clear CUDA cache
    CUDA_VISIBLE_DEVICES="${gpu_id}" _cmd_python -c "
	import torch
	if torch.cuda.is_available():
	    torch.cuda.empty_cache()
	    torch.cuda.synchronize()
" 2>/dev/null || true
}

# Purge memory on multiple GPUs
# Usage: purge_multi_gpu_memory <gpu_ids_csv>
purge_multi_gpu_memory() {
    local gpu_ids="$1"
    gpu_ids="${gpu_ids// /}"
    local gpu_id
    local -a gpus=()

    IFS=',' read -ra gpus <<< "${gpu_ids}"
    for gpu_id in "${gpus[@]}"; do
        purge_gpu_memory "${gpu_id}"
    done
}

# Kill all compute processes on the provided GPUs.
# Usage: kill_gpu_processes <gpu_ids_csv>
kill_gpu_processes() {
    local gpu_ids="$1"
    gpu_ids="${gpu_ids// /}"
    [[ -z "${gpu_ids}" ]] && return 0
    local gpu_id
    local -a gpus=()

    IFS=',' read -ra gpus <<< "${gpu_ids}"
    for gpu_id in "${gpus[@]}"; do
        local pids
        pids=$(_cmd_nvidia_smi --query-compute-apps=pid --format=csv,noheader -i "${gpu_id}" 2>/dev/null | tr -d ' ' | awk '/^[0-9]+$/' || true)
        [[ -z "${pids}" ]] && continue

        for pid in ${pids}; do
            _cmd_kill -TERM "${pid}" 2>/dev/null || true
        done

        _sleep 1

        for pid in ${pids}; do
            _cmd_kill -KILL "${pid}" 2>/dev/null || true
        done
    done
}

# ============ GPU MEMORY MANAGEMENT ============

# Get available GPU memory in GB (with caching to reduce nvidia-smi calls)
# Usage: get_gpu_available_memory <gpu_id>
get_gpu_available_memory() {
    local gpu_id="$1"

    # Try to read from cache first
    local cached_val
    cached_val=$(_read_gpu_cache "${gpu_id}" "free_mem")
    if [[ -n "${cached_val}" && "${cached_val}" =~ ^[0-9]+$ ]]; then
        echo "${cached_val}"
        return 0
    fi

    # Cache miss - query nvidia-smi for free memory in MiB.
    local free_output
    if ! free_output=$(_cmd_nvidia_smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null); then
        echo "0"
        return 1
    fi
    local free_mib
    free_mib=$(echo "${free_output}" | head -1 | tr -d ' ')
    if ! [[ "${free_mib}" =~ ^[0-9]+$ ]]; then
        echo "0"
        return 1
    fi

    # Convert MiB to GB (1024 MiB = 1 GiB ≈ 1.07 GB)
    local free_gb=$((free_mib / 1024))

    # Update cache (also refresh idle status since we're querying)
    # Count only actual PID lines (numbers) to avoid empty line issues
    local raw_output
    raw_output=$(_cmd_nvidia_smi --query-compute-apps=pid --format=csv,noheader -i "${gpu_id}" 2>/dev/null || true)
    local processes=0
    if [[ -n "${raw_output}" ]]; then
        processes=$(echo "${raw_output}" | grep -cE '^[0-9]+' 2>/dev/null || echo "0")
    fi
    local is_idle="false"
    [[ "${processes}" -eq 0 ]] && is_idle="true"
    _write_gpu_cache "${gpu_id}" "${free_gb}" "${is_idle}"

    echo "${free_gb}"
}

# Get total GPU memory in GB
# Usage: get_gpu_total_memory <gpu_id>
get_gpu_total_memory() {
    local gpu_id="$1"

    local total_mib
    total_mib=$(_cmd_nvidia_smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | head -1 || true)

    if ! [[ "${total_mib}" =~ ^[0-9]+$ ]]; then
        echo "80"  # Default to 80GB when GPU size unknown
        return 1
    fi

    local total_gb=$((total_mib / 1024))
    echo "${total_gb}"
}

# Get GPU utilization percentage
# Usage: get_gpu_utilization <gpu_id>
get_gpu_utilization() {
    local gpu_id="$1"

    _cmd_nvidia_smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | head -1 || echo "0"
}

# Check if GPU is idle (no processes) - with caching to reduce nvidia-smi calls
# Usage: is_gpu_idle <gpu_id>
is_gpu_idle() {
    local gpu_id="$1"

    # Try to read from cache first
    local cached_idle
    cached_idle=$(_read_gpu_cache "${gpu_id}" "is_idle")
    if [[ -n "${cached_idle}" ]]; then
        [[ "${cached_idle}" == "true" ]]
        return $?
    fi

    # Cache miss - query nvidia-smi
    # Note: When no processes are running, nvidia-smi returns empty output.
    # We use grep -c to count actual PID lines (numbers only) to avoid
    # counting empty lines or error messages.
    local raw_output
    raw_output=$(_cmd_nvidia_smi --query-compute-apps=pid --format=csv,noheader -i "${gpu_id}" 2>/dev/null || true)

    local processes=0
    if [[ -n "${raw_output}" ]]; then
        # Count only lines that contain actual PIDs (numeric values)
        processes=$(echo "${raw_output}" | grep -cE '^[0-9]+' 2>/dev/null || echo "0")
    fi

    local is_idle="false"
    [[ "${processes}" -eq 0 ]] && is_idle="true"

    # Update cache (also refresh free memory since we're querying)
    local free_mib
    free_mib=$(_cmd_nvidia_smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | head -1 || true)
    local free_gb=0
    if [[ "${free_mib}" =~ ^[0-9]+$ ]]; then
        free_gb=$((free_mib / 1024))
    fi
    _write_gpu_cache "${gpu_id}" "${free_gb}" "${is_idle}"

    [[ "${is_idle}" == "true" ]]
}
