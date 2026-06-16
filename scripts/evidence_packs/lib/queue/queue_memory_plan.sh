#!/usr/bin/env bash
# queue_memory_plan.sh - Queue memory-plan refresh and export helpers
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Usage: sourced by queue_manager.sh

QUEUE_MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=queue_core.sh
[[ -z "${QUEUE_CORE_LOADED:-}" ]] && source "${QUEUE_MODULE_DIR}/queue_core.sh"

# Refresh task memory estimates for any models with existing profiles.
# Usage: refresh_task_memory_from_profiles <output_dir>
refresh_task_memory_from_profiles() {
    local output_dir="$1"
    local model_dir

    for model_dir in "${output_dir}"/*; do
        [[ -d "${model_dir}" ]] || continue

        local model_name
        model_name=$(basename "${model_dir}")
        local model_id=""
        if [[ -f "${model_dir}/.model_id" ]]; then
            model_id=$(cat "${model_dir}/.model_id" 2>/dev/null || true)
        fi

        update_model_task_memory "${model_name}" "${output_dir}" "${model_id}"
    done
}

export_memory_plan() {
    local output_dir="$1"
    local plan_dir="${output_dir}/analysis"
    local plan_file="${plan_dir}/memory_plan.csv"
    local tmp_file="${plan_file}.tmp.${BASHPID:-$$}"
    local task_file

    mkdir -p "${plan_dir}"

    local csv_escape
    csv_escape() {
        local val="$1"
        val="${val//\"/\"\"}"
        printf "\"%s\"" "${val}"
    }

    echo "task_id,status,task_type,model_name,model_id,model_size_gb,required_gpus,adaptive_gpus,assigned_gpus,priority" > "${tmp_file}"

    local status
    for status in pending ready running completed failed; do
        [[ -d "${QUEUE_DIR}/${status}" ]] || continue
        for task_file in "${QUEUE_DIR}/${status}"/*.task; do
            [[ -f "${task_file}" ]] || continue

            local task_id
            local task_type
            local model_name
            local model_id
            local model_size_gb
            local required_gpus
            local adaptive_gpus
            local assigned_gpus
            local priority

            task_id=$(get_task_id "${task_file}")
            task_type=$(get_task_type "${task_file}")
            model_name=$(get_task_field "${task_file}" "model_name")
            model_id=$(get_task_field "${task_file}" "model_id")
            model_size_gb=$(get_task_field "${task_file}" "model_size_gb")
            required_gpus=$(get_task_field "${task_file}" "required_gpus")
            adaptive_gpus=$(get_task_field "${task_file}" "adaptive_gpus")
            assigned_gpus=$(get_task_field "${task_file}" "assigned_gpus")
            priority=$(get_task_field "${task_file}" "priority")

            echo "$(csv_escape "${task_id}"),$(csv_escape "${status}"),$(csv_escape "${task_type}"),$(csv_escape "${model_name}"),$(csv_escape "${model_id}"),$(csv_escape "${model_size_gb}"),$(csv_escape "${required_gpus}"),$(csv_escape "${adaptive_gpus}"),$(csv_escape "${assigned_gpus}"),$(csv_escape "${priority}")" >> "${tmp_file}"
        done
    done

    mv "${tmp_file}" "${plan_file}"
}

# Update task memory estimates for a model based on its on-disk profile.
# Usage: update_model_task_memory <model_name> <output_dir> [model_id]
update_model_task_memory() {
    local model_name="$1"
    local output_dir="$2"
    local model_id="${3:-}"
    local profile_path=""
    local baseline_path_file="${output_dir}/${model_name}/.baseline_path"
    local baseline_path=""

    if [[ -f "${baseline_path_file}" ]]; then
        baseline_path=$(cat "${baseline_path_file}" 2>/dev/null || true)
        if [[ -n "${baseline_path}" ]]; then
            profile_path="${baseline_path}/model_profile.json"
        fi
    fi

    if [[ -z "${profile_path}" || ! -f "${profile_path}" ]]; then
        profile_path="${output_dir}/${model_name}/models/baseline/model_profile.json"
    fi

    if [[ -n "${model_id}" && ( -z "${profile_path}" || ! -f "${profile_path}" ) ]]; then
        local model_basename
        model_basename=$(basename "${model_id}" | tr '[:upper:]' '[:lower:]' | tr '/' '_')
        local model_sanitized
        model_sanitized=$(printf '%s' "${model_id}" \
            | tr '[:upper:]' '[:lower:]' \
            | sed 's#/#__#g' \
            | tr ' ' '_' \
            | tr -cd '[:alnum:]_-')
        local candidate
        for candidate in \
            "${output_dir}/models/${model_sanitized}/baseline/model_profile.json" \
            "${output_dir}/models/${model_basename}/baseline/model_profile.json"; do
            if [[ -f "${candidate}" ]]; then
                profile_path="${candidate}"
                break
            fi
        done
    fi

    [[ -f "${profile_path}" ]] || return 0

    local queue_dirs=("${QUEUE_DIR}/pending" "${QUEUE_DIR}/ready")
    for dir in "${queue_dirs[@]}"; do
        for task_file in "${dir}"/*.task; do
            [[ -f "${task_file}" ]] || continue
            local task_model
            task_model=$(get_task_field "${task_file}" "model_name")
            [[ "${task_model}" == "${model_name}" ]] || continue

            local task_type
            task_type=$(get_task_type "${task_file}")

            local result
            result=$(TASK_TYPE="${task_type}" MODEL_ID="${model_id}" PROFILE_PATH="${profile_path}" \
                EVAL_BATCH_SIZE_SMALL="${EVAL_BATCH_SIZE_SMALL:-auto:16}" \
                EVAL_BATCH_SIZE_MEDIUM="${EVAL_BATCH_SIZE_MEDIUM:-auto:8}" \
                EVAL_BATCH_SIZE_LARGE="${EVAL_BATCH_SIZE_LARGE:-auto:4}" \
                EVAL_BATCH_SIZE_MOE="${EVAL_BATCH_SIZE_MOE:-auto:6}" \
                EVAL_CONTEXT_LEN="${EVAL_CONTEXT_LEN:-2048}" \
                MODEL_LOAD_OVERHEAD_GB="${MODEL_LOAD_OVERHEAD_GB:-4}" \
                EDIT_OVERHEAD_GB="${EDIT_OVERHEAD_GB:-8}" \
                BATCH_EDIT_OVERHEAD_GB="${BATCH_EDIT_OVERHEAD_GB:-8}" \
                EVAL_OVERHEAD_GB="${EVAL_OVERHEAD_GB:-6}" \
                INVARLOCK_OVERHEAD_GB="${INVARLOCK_OVERHEAD_GB:-6}" \
                GPU_MEMORY_PER_DEVICE="${GPU_MEMORY_PER_DEVICE:-${GPU_MEMORY_GB:-180}}" \
                NUM_GPUS="${NUM_GPUS:-8}" \
                _runtime_python queue_state.py estimate-task-memory
            )

            local required_mem=""
            local required_gpus=""
            read -r required_mem required_gpus <<< "${result}"

            local current_required_gpus=""
            current_required_gpus=$(get_task_field "${task_file}" "required_gpus" 2>/dev/null || true)
            local preserve_multi_gpu_floor="false"
            if [[ "${current_required_gpus}" =~ ^[0-9]+$ ]] && [[ ${current_required_gpus} -gt 1 ]]; then
                preserve_multi_gpu_floor="true"
            fi

            if [[ -n "${required_mem}" && "${required_mem}" =~ ^[0-9]+$ ]]; then
                local current_mem=""
                current_mem=$(get_task_field "${task_file}" "model_size_gb" 2>/dev/null || true)
                if [[ "${preserve_multi_gpu_floor}" == "true" ]] && [[ "${current_mem}" =~ ^[0-9]+$ ]] && [[ ${current_mem} -gt ${required_mem} ]]; then
                    required_mem="${current_mem}"
                fi
                update_task_field "${task_file}" "model_size_gb" "${required_mem}" "true" 2>/dev/null || true
            fi
            if [[ -n "${required_gpus}" && "${required_gpus}" =~ ^[0-9]+$ ]]; then
                if [[ "${preserve_multi_gpu_floor}" == "true" ]] && [[ "${current_required_gpus}" =~ ^[0-9]+$ ]] && [[ ${current_required_gpus} -gt ${required_gpus} ]]; then
                    required_gpus="${current_required_gpus}"
                fi
                update_task_field "${task_file}" "required_gpus" "${required_gpus}" "true" 2>/dev/null || true
            fi
        done
    done

    if type export_memory_plan &>/dev/null; then
        export_memory_plan "${output_dir}"
    fi
}
