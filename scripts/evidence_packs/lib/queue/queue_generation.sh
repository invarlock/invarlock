#!/usr/bin/env bash
# queue_generation.sh - Progress, task search, and task graph generation
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Usage: sourced by queue_manager.sh

QUEUE_MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=queue_core.sh
if ! declare -F add_task >/dev/null 2>&1; then
    source "${QUEUE_MODULE_DIR}/queue_core.sh"
fi
QUEUE_CORE_LOADED=1
export -n QUEUE_CORE_LOADED 2>/dev/null || true
# shellcheck source=queue_lifecycle.sh
if ! declare -F mark_task_ready >/dev/null 2>&1; then
    source "${QUEUE_MODULE_DIR}/queue_lifecycle.sh"
fi
QUEUE_LIFECYCLE_LOADED=1
export -n QUEUE_LIFECYCLE_LOADED 2>/dev/null || true
# shellcheck source=queue_dependencies.sh
if ! declare -F check_dependencies_met >/dev/null 2>&1; then
    source "${QUEUE_MODULE_DIR}/queue_dependencies.sh"
fi
QUEUE_DEPENDENCIES_LOADED=1
export -n QUEUE_DEPENDENCIES_LOADED 2>/dev/null || true

# ============ PROGRESS TRACKING ============

# Update progress state file
# Usage: update_progress_state
update_progress_state() {
    local state_file="${QUEUE_DIR}/../state/progress.json"
    local tmp_file="${state_file}.tmp.${BASHPID:-$$}"

    IFS=':' read -r pending ready running completed failed total <<< "$(get_queue_stats)"

    local status="running"
    local terminal_state=""
    terminal_state="$(queue_terminal_state 2>/dev/null || true)"
    if [[ -n "${terminal_state}" ]]; then
        status="${terminal_state}"
    fi

    mkdir -p "$(dirname "${state_file}")" 2>/dev/null || true
    _runtime_python queue_state.py progress \
        --output "${tmp_file}" \
        --updated-at "$(_now_iso)" \
        --pending "${pending}" \
        --ready "${ready}" \
        --running "${running}" \
        --completed "${completed}" \
        --failed "${failed}" \
        --total "${total}" \
        --status "${status}" \
        || { rm -f "${tmp_file}" 2>/dev/null || true; return 1; }
    mv -f "${tmp_file}" "${state_file}" 2>/dev/null || {
        rm -f "${tmp_file}" 2>/dev/null || true
        return 1
    }
}

# ============ TASK SEARCH ============

# Find task by ID across all queues
# Usage: find_task <task_id>
# Returns: full path to task file, or empty if not found
find_task() {
    local task_id="$1"
    local status

    for status in pending ready running completed failed; do
        local path="${QUEUE_DIR}/${status}/${task_id}.task"
        if [[ -f "${path}" ]]; then
            echo "${path}"
            return 0
        fi
    done

    return 1
}

# Find tasks by model name
# Usage: find_tasks_by_model <model_name> [status]
find_tasks_by_model() {
    local model_name="$1"
    local status="${2:-}"
    local dir
    local task_file

    local search_dirs=()
    if [[ -n "${status}" ]]; then
        search_dirs=("${QUEUE_DIR}/${status}")
    else
        search_dirs=("${QUEUE_DIR}/pending" "${QUEUE_DIR}/ready" "${QUEUE_DIR}/running" "${QUEUE_DIR}/completed" "${QUEUE_DIR}/failed")
    fi

    for dir in "${search_dirs[@]}"; do
        for task_file in "${dir}"/*.task; do
            [[ -f "${task_file}" ]] || continue
            local task_model=$(get_task_field "${task_file}" "model_name")
            if [[ "${task_model}" == "${model_name}" ]]; then
                echo "${task_file}"
            fi
        done
    done
}

# Find tasks by type
# Usage: find_tasks_by_type <task_type> [status]
find_tasks_by_type() {
    local task_type="$1"
    local status="${2:-}"
    local dir
    local task_file

    local search_dirs=()
    if [[ -n "${status}" ]]; then
        search_dirs=("${QUEUE_DIR}/${status}")
    else
        search_dirs=("${QUEUE_DIR}/pending" "${QUEUE_DIR}/ready" "${QUEUE_DIR}/running" "${QUEUE_DIR}/completed" "${QUEUE_DIR}/failed")
    fi

    for dir in "${search_dirs[@]}"; do
        for task_file in "${dir}"/*.task; do
            [[ -f "${task_file}" ]] || continue
            local type=$(get_task_type "${task_file}")
            if [[ "${type}" == "${task_type}" ]]; then
                echo "${task_file}"
            fi
        done
    done
}

# ============ TASK GRAPH GENERATION ============

# Generate all tasks for a model
# Usage: generate_model_tasks <model_idx> <model_id> <model_name>
generate_model_tasks() {
    local model_idx="$1"
    local model_id="$2"
    local model_name="$3"

    # Calculate model size for memory estimation
    local base_size=$(estimate_model_memory "${model_id}" "evaluate_EDIT")

    # Decide whether to use batch edit creation or per-edit tasks. The batch
    # helper now defaults to reload-per-edit, but very large models still use
    # per-edit CREATE_EDIT tasks to keep scheduling and cleanup granular.
    local use_batch="true"
    local model_lower
    model_lower=$(printf '%s' "${model_id}" | tr '[:upper:]' '[:lower:]')
    if [[ "${model_lower}" =~ 70b || "${model_lower}" =~ 72b || "${model_lower}" =~ 65b || "${model_lower}" =~ mixtral || "${model_lower}" =~ 8x7b || "${model_lower}" =~ moe ]]; then
        use_batch="false"
    elif [[ -n "${base_size}" ]]; then
        # Treat anything >=170GB as "large" and avoid batch edit tasks.
        if [[ "${base_size}" -ge 170 ]]; then
            use_batch="false"
        fi
    fi
    # Allow explicit override via env
    if [[ -n "${PACK_USE_BATCH_EDITS:-}" ]]; then
        if [[ "${PACK_USE_BATCH_EDITS}" == "false" || "${PACK_USE_BATCH_EDITS}" == "0" ]]; then
            use_batch="false"
        elif [[ "${PACK_USE_BATCH_EDITS}" == "true" || "${PACK_USE_BATCH_EDITS}" == "1" ]]; then
            use_batch="true"
        fi
    fi

    # Default to per-edit tasks when model cleanup is enabled to avoid
    # creating all variants up-front (which can exhaust disk on single-node runs).
    local cleanup_models="${PACK_CLEANUP_MODELS:-1}"
    if [[ "${cleanup_models}" != "0" && -z "${PACK_USE_BATCH_EDITS:-}" ]]; then
        use_batch="false"
    fi

    # Track task IDs for dependencies
    local task_ids=()

    # 1. SETUP_BASELINE (no dependencies)
    local setup_id=""
    capture_add_task setup_id "SETUP_BASELINE" "${model_id}" "${model_name}" \
        "$(estimate_model_memory "${model_id}" "SETUP_BASELINE")" \
        "none" '{"model_idx": '"${model_idx}"'}' 90
    task_ids+=("${setup_id}")
    echo "Created: ${setup_id}"

    # 2. CALIBRATION_RUN × N (depend on setup)
    local cal_ids=()
    local calibration_runs="${DRIFT_CALIBRATION_RUNS:-5}"
    if ! [[ "${calibration_runs}" =~ ^[0-9]+$ ]]; then
        calibration_runs=5
    fi
    local preset_ready="${PACK_PRESET_READY:-false}"
    if [[ "${preset_ready}" == "1" ]]; then
        preset_ready="true"
    fi
    if [[ ${calibration_runs} -gt 0 ]]; then
        for run in $(seq 1 "${calibration_runs}"); do
            local cal_id=""
            capture_add_task cal_id "CALIBRATION_RUN" "${model_id}" "${model_name}" \
                "$(estimate_model_memory "${model_id}" "CALIBRATION_RUN")" \
                "${setup_id}" '{"run": '"${run}"', "seed": '"$((41 + run))"'}' 85
            cal_ids+=("${cal_id}")
            task_ids+=("${cal_id}")
            echo "Created: ${cal_id}"
        done
    fi

    # 3. GENERATE_PRESET (depends on all calibration runs)
    local preset_id=""
    if [[ ${calibration_runs} -gt 0 ]]; then
        local cal_deps=$(IFS=','; echo "${cal_ids[*]}")
        capture_add_task preset_id "GENERATE_PRESET" "${model_id}" "${model_name}" \
            5 "${cal_deps}" '{}' 75
        task_ids+=("${preset_id}")
        echo "Created: ${preset_id}"
    else
        echo "Skipping GENERATE_PRESET (calibration_runs=0)"
    fi

    local use_preset="false"
    if [[ ${calibration_runs} -gt 0 || "${preset_ready}" == "true" ]]; then
        use_preset="true"
    fi

    local clean_runs="${CLEAN_EDIT_RUNS:-0}"
    if ! [[ "${clean_runs}" =~ ^-?[0-9]+$ ]]; then
        clean_runs=0
    fi
    if [[ ${clean_runs} -lt 0 ]]; then
        clean_runs=0
    fi

    local stress_runs="${STRESS_EDIT_RUNS:-0}"
    if ! [[ "${stress_runs}" =~ ^-?[0-9]+$ ]]; then
        stress_runs=0
    fi
    if [[ ${stress_runs} -lt 0 ]]; then
        stress_runs=0
    fi

    # 4. Edit creation + evaluate
    # Load edit specs from scenarios.json when available to keep the task graph
    # and verdict contract in sync.
    local pack_root
    pack_root="$(_pack_queue_pack_root)"
    local scenarios_file=""
    local suite_manifest=""
    local using_state_manifest="false"
    # Prefer the run's state manifest (may be filtered by suite tags) when available.
    if [[ -n "${QUEUE_DIR:-}" ]]; then
        local run_root
        run_root="$(cd "${QUEUE_DIR}/.." && pwd)"
        suite_manifest="${run_root}/state/scenarios.json"
        if [[ -f "${suite_manifest}" ]]; then
            scenarios_file="${suite_manifest}"
            using_state_manifest="true"
        fi
    fi
    if [[ -z "${scenarios_file}" ]]; then
        scenarios_file="${pack_root}/scenarios.json"
    fi

    local clean_edits=()
    local stress_edits=()
    local clean_edit_count=0
    local stress_edit_count=0
    local loaded_edit_manifest="false"

    if command -v jq >/dev/null 2>&1 && [[ -f "${scenarios_file}" ]]; then
        local edit_specs_json=""
        edit_specs_json="$(
            jq -c '[.scenarios[]
                | select(.generation.kind=="edit")
                | {spec: .generation.edit_spec, version: .generation.version}]' "${scenarios_file}" 2>/dev/null
        )"
        if [[ -n "${edit_specs_json}" ]]; then
            loaded_edit_manifest="true"
        fi
        local edit_spec=""
        while IFS= read -r edit_spec; do
            [[ -n "${edit_spec}" ]] || continue
            clean_edits+=("${edit_spec}")
            clean_edit_count=$((clean_edit_count + 1))
        done < <(
            jq -r '.scenarios[]
                | select(.generation.kind=="edit" and .generation.version=="clean")
                | .generation.edit_spec' "${scenarios_file}" 2>/dev/null
        )
        while IFS= read -r edit_spec; do
            [[ -n "${edit_spec}" ]] || continue
            stress_edits+=("${edit_spec}")
            stress_edit_count=$((stress_edit_count + 1))
        done < <(
            jq -r '.scenarios[]
                | select(.generation.kind=="edit" and .generation.version=="stress")
                | .generation.edit_spec' "${scenarios_file}" 2>/dev/null
        )
    fi

    if [[ "${loaded_edit_manifest}" != "true" ]]; then
        # Fallback defaults (kept for standalone script use).
        clean_edits=("quant_rtn:clean:ffn" "fp8_quant:clean:ffn" "magnitude_prune:clean:ffn" "lowrank_svd:clean:ffn" "lora_merge:clean:attn" "fine_tune:clean:ffn")
        stress_edits=("quant_rtn:4:32:all" "fp8_quant:e5m2:all" "magnitude_prune:0.5:all" "lowrank_svd:32:all" "lora_merge:8:64:all" "fine_tune:0.0005:3:all")
        clean_edit_count=6
        stress_edit_count=6
    fi

    # Materialize the reusable noop baseline report as a real dependency instead
    # of letting the first evaluate task create it lazily while other GPU workers
    # wait inside claimed tasks.
    local baseline_report_id=""
    local eager_baseline_report="${PACK_EAGER_BASELINE_REPORT:-1}"
    if [[ "${PACK_SUITE_MODE:-full}" != "calibrate-only" && "${use_preset}" == "true" && "${eager_baseline_report}" != "0" && "${eager_baseline_report}" != "false" ]]; then
        if [[ ${clean_runs} -gt 0 || ${stress_runs} -gt 0 || "${RUN_ERROR_INJECTION:-true}" == "true" ]]; then
            capture_add_task baseline_report_id "SETUP_EVALUATE_BASELINE_REPORT" "${model_id}" "${model_name}" \
                "$(estimate_model_memory "${model_id}" "SETUP_EVALUATE_BASELINE_REPORT")" \
                "${setup_id}" '{}' 73
            task_ids+=("${baseline_report_id}")
            echo "Created: ${baseline_report_id}"
        fi
    fi

    # Ensure use_batch is defined (defensive for set -u)
    use_batch=${use_batch:-true}
    # Skip edits entirely if both clean and stress runs are disabled
    if [[ ${clean_runs} -le 0 && ${stress_runs} -le 0 ]]; then
        echo "Skipping edit creation (CLEAN_EDIT_RUNS=0, STRESS_EDIT_RUNS=0)"

    elif [[ "${use_preset}" != "true" ]]; then
        echo "Skipping edit creation (no calibrated preset available)"

    elif [[ "${use_batch}" == "true" ]]; then
        # CREATE_EDITS_BATCH - Create edits from one parsed batch. The Python
        # helper defaults to reload-per-edit to avoid deep-copy memory spikes.
        :

        local -a requested_specs=()
        local edit_spec
        if [[ ${clean_runs} -gt 0 && ${clean_edit_count} -gt 0 ]]; then
            for edit_spec in "${clean_edits[@]}"; do
                requested_specs+=("{\"spec\": \"${edit_spec}\", \"version\": \"clean\"}")
            done
        fi
        if [[ ${stress_runs} -gt 0 && ${stress_edit_count} -gt 0 ]]; then
            for edit_spec in "${stress_edits[@]}"; do
                requested_specs+=("{\"spec\": \"${edit_spec}\", \"version\": \"stress\"}")
            done
        fi
        local requested_json
        requested_json="[$(IFS=','; echo "${requested_specs[*]}")]"

        local edit_deps="${setup_id}"
        local edits_id=""
        capture_add_task edits_id "CREATE_EDITS_BATCH" "${model_id}" "${model_name}" \
            "$(estimate_model_memory "${model_id}" "CREATE_EDITS_BATCH")" \
            "${edit_deps}" '{"edit_specs": '"${requested_json}"', "use_batch": true}' 70
        task_ids+=("${edits_id}")
        echo "Created: ${edits_id}"

        if [[ ${clean_runs} -gt 0 && ${clean_edit_count} -gt 0 ]]; then
            for edit_spec in "${clean_edits[@]}"; do
                local -a eval_ids=()
                local run
                for run in $(seq 1 "${clean_runs}"); do
                    local cert_deps="${edits_id}"
                    if [[ -n "${preset_id}" ]]; then
                        cert_deps="${cert_deps},${preset_id}"
                    fi
                    if [[ -n "${baseline_report_id}" ]]; then
                        cert_deps="${cert_deps},${baseline_report_id}"
                    fi
                    local cert_id=""
                    capture_add_task cert_id "evaluate_EDIT" "${model_id}" "${model_name}" \
                        "$(estimate_model_memory "${model_id}" "evaluate_EDIT")" \
                        "${cert_deps}" '{"edit_spec": "'"${edit_spec}"'", "version": "clean", "run": '"${run}"'}' 74
                    eval_ids+=("${cert_id}")
                    task_ids+=("${cert_id}")
                    echo "Created: ${cert_id}"
                done
                if [[ "${cleanup_models}" != "0" && ${#eval_ids[@]} -gt 0 ]]; then
                    local deps_csv
                    deps_csv="$(IFS=','; echo "${eval_ids[*]}")"
                    local cleanup_id
                    capture_add_task cleanup_id "CLEANUP_EDIT" "${model_id}" "${model_name}" \
                        1 "${deps_csv}" '{"edit_spec": "'"${edit_spec}"'", "version": "clean"}' 80
                    echo "Created: ${cleanup_id}"
                fi
            done
        fi

        if [[ ${stress_runs} -gt 0 && ${stress_edit_count} -gt 0 ]]; then
            for edit_spec in "${stress_edits[@]}"; do
                local -a eval_ids=()
                local run
                for run in $(seq 1 "${stress_runs}"); do
                    local cert_deps="${edits_id}"
                    if [[ -n "${preset_id}" ]]; then
                        cert_deps="${cert_deps},${preset_id}"
                    fi
                    if [[ -n "${baseline_report_id}" ]]; then
                        cert_deps="${cert_deps},${baseline_report_id}"
                    fi
                    local cert_id=""
                    capture_add_task cert_id "evaluate_EDIT" "${model_id}" "${model_name}" \
                        "$(estimate_model_memory "${model_id}" "evaluate_EDIT")" \
                        "${cert_deps}" '{"edit_spec": "'"${edit_spec}"'", "version": "stress", "run": '"${run}"'}' 74
                    eval_ids+=("${cert_id}")
                    task_ids+=("${cert_id}")
                    echo "Created: ${cert_id}"
                done
                if [[ "${cleanup_models}" != "0" && ${#eval_ids[@]} -gt 0 ]]; then
                    local deps_csv
                    deps_csv="$(IFS=','; echo "${eval_ids[*]}")"
                    local cleanup_id
                    capture_add_task cleanup_id "CLEANUP_EDIT" "${model_id}" "${model_name}" \
                        1 "${deps_csv}" '{"edit_spec": "'"${edit_spec}"'", "version": "stress"}' 80
                    echo "Created: ${cleanup_id}"
                fi
            done
        fi
    else
        # CREATE_EDIT - Create single edits (one task per edit) and enqueue eval/evaluate
        if [[ ${clean_runs} -gt 0 && ${clean_edit_count} -gt 0 ]]; then
        for edit_spec in "${clean_edits[@]}"; do
            local edit_deps="${setup_id}"
            local edit_id=""
            capture_add_task edit_id "CREATE_EDIT" "${model_id}" "${model_name}" \
                "$(estimate_model_memory "${model_id}" "CREATE_EDIT")" \
                "${edit_deps}" '{"edit_spec": "'"${edit_spec}"'", "version": "clean", "model_idx": '"${model_idx}"'}' 70
            task_ids+=("${edit_id}")
            echo "Created: ${edit_id}"

            # evaluate_EDIT runs for clean edits (3 by default)
            local -a eval_ids=()
            if [[ "${use_preset}" == "true" ]]; then
                for run in $(seq 1 "${clean_runs}"); do
                    local cert_deps="${edit_id}"
                    if [[ -n "${preset_id}" ]]; then
                        cert_deps="${cert_deps},${preset_id}"
                    fi
                    if [[ -n "${baseline_report_id}" ]]; then
                        cert_deps="${cert_deps},${baseline_report_id}"
                    fi
                    local cert_id=""
                    capture_add_task cert_id "evaluate_EDIT" "${model_id}" "${model_name}" \
                        "$(estimate_model_memory "${model_id}" "evaluate_EDIT")" \
                        "${cert_deps}" '{"edit_spec": "'"${edit_spec}"'", "version": "clean", "run": '"${run}"'}' 74
                    eval_ids+=("${cert_id}")
                    task_ids+=("${cert_id}")
                    echo "Created: ${cert_id}"
                done
            fi
            if [[ "${cleanup_models}" != "0" && ${#eval_ids[@]} -gt 0 ]]; then
                local deps_csv
                deps_csv="$(IFS=','; echo "${eval_ids[*]}")"
                local cleanup_id
                capture_add_task cleanup_id "CLEANUP_EDIT" "${model_id}" "${model_name}" \
                    1 "${deps_csv}" '{"edit_spec": "'"${edit_spec}"'", "version": "clean"}' 80
                echo "Created: ${cleanup_id}"
            fi
        done
        fi

        if [[ ${stress_runs} -gt 0 && ${stress_edit_count} -gt 0 ]]; then
        for edit_spec in "${stress_edits[@]}"; do
            local edit_id=""
            capture_add_task edit_id "CREATE_EDIT" "${model_id}" "${model_name}" \
                "$(estimate_model_memory "${model_id}" "CREATE_EDIT")" \
                "${setup_id}" '{"edit_spec": "'"${edit_spec}"'", "version": "stress", "model_idx": '"${model_idx}"'}' 70
            task_ids+=("${edit_id}")
            echo "Created: ${edit_id}"

            local -a eval_ids=()
            if [[ "${use_preset}" == "true" ]]; then
                for run in $(seq 1 "${stress_runs}"); do
                    local cert_deps="${edit_id}"
                    if [[ -n "${preset_id}" ]]; then
                        cert_deps="${cert_deps},${preset_id}"
                    fi
                    if [[ -n "${baseline_report_id}" ]]; then
                        cert_deps="${cert_deps},${baseline_report_id}"
                    fi
                    local cert_id=""
                    capture_add_task cert_id "evaluate_EDIT" "${model_id}" "${model_name}" \
                        "$(estimate_model_memory "${model_id}" "evaluate_EDIT")" \
                        "${cert_deps}" '{"edit_spec": "'"${edit_spec}"'", "version": "stress", "run": '"${run}"'}' 74
                    eval_ids+=("${cert_id}")
                    task_ids+=("${cert_id}")
                    echo "Created: ${cert_id}"
                done
            fi
            if [[ "${cleanup_models}" != "0" && ${#eval_ids[@]} -gt 0 ]]; then
                local deps_csv
                deps_csv="$(IFS=','; echo "${eval_ids[*]}")"
                local cleanup_id
                capture_add_task cleanup_id "CLEANUP_EDIT" "${model_id}" "${model_name}" \
                    1 "${deps_csv}" '{"edit_spec": "'"${edit_spec}"'", "version": "stress"}' 80
                echo "Created: ${cleanup_id}"
            fi
        done
        fi
    fi

    # 5. Error injection tests
    if [[ "${RUN_ERROR_INJECTION:-true}" == "true" && "${use_preset}" == "true" ]]; then
        local jq_bin=""
        jq_bin="$(type -P jq 2>/dev/null || true)"
        local error_pairs=()
        local error_pair_count=0
        if [[ -n "${jq_bin}" && -f "${scenarios_file}" ]]; then
            local error_pair=""
            while IFS= read -r error_pair; do
                [[ -n "${error_pair}" ]] || continue
                error_pairs+=("${error_pair}")
                error_pair_count=$((error_pair_count + 1))
            done < <(
                "${jq_bin}" -r --arg model_id "${model_id}" --arg model_name "${model_name}" '.scenarios[]
                    | select(.generation.kind=="error")
                    | (.generation.env // {}) as $base_env
                    | (.generation.env_by_model // {}) as $env_by_model
                    | (($env_by_model[$model_id] // $env_by_model[$model_name] // {}) ) as $model_env
                    | [
                        (.generation.error_type // ""),
                        (($base_env + $model_env) | tojson)
                      ]
                    | @tsv' "${scenarios_file}"
            )
        fi
        if [[ ${error_pair_count} -eq 0 && "${using_state_manifest}" != "true" ]]; then
            :
            error_pairs=(
                $'nan_injection\t{}'
                $'inf_injection\t{}'
                $'shape_mismatch\t{}'
                $'missing_tensors\t{}'
                $'extreme_quant\t{}'
                $'scale_explosion\t{}'
                $'rank_collapse\t{}'
                $'norm_collapse\t{}'
                $'weight_tying_break\t{}'
            )
            error_pair_count=9
        fi
        if [[ ${error_pair_count} -gt 0 ]]; then
        for error_pair in "${error_pairs[@]}"; do
            local error_type="${error_pair%%$'\t'*}"
            local error_env_json="{}"
            if [[ "${error_pair}" == *$'\t'* ]]; then
                error_env_json="${error_pair#*$'\t'}"
            fi
            if [[ -z "${error_type}" ]]; then
                continue
            fi

            local params_json
            if [[ -n "${jq_bin}" ]]; then
                local jq_params_builder="${jq_bin}"
                params_json="$(
                    "${jq_params_builder}" -cn \
                        --arg error_type "${error_type}" \
                        --argjson error_env "${error_env_json}" \
                        '{error_type: $error_type, error_env: $error_env}'
                )"
            else
                params_json='{"error_type": "'"${error_type}"'"}'
            fi

            # CREATE_ERROR
            local error_create_id=""
            capture_add_task error_create_id "CREATE_ERROR" "${model_id}" "${model_name}" \
                "$(estimate_model_memory "${model_id}" "CREATE_ERROR")" \
                "${setup_id}" "${params_json}" 60
            echo "Created: ${error_create_id}"

            local cert_deps="${error_create_id}"
            if [[ -n "${preset_id}" ]]; then
                cert_deps="${cert_deps},${preset_id}"
            fi
            if [[ -n "${baseline_report_id}" ]]; then
                cert_deps="${cert_deps},${baseline_report_id}"
            fi
            local error_cert_id=""
            capture_add_task error_cert_id "evaluate_ERROR" "${model_id}" "${model_name}" \
                "$(estimate_model_memory "${model_id}" "evaluate_ERROR")" \
                "${cert_deps}" "${params_json}" 64
            echo "Created: ${error_cert_id}"
            if [[ "${cleanup_models}" != "0" ]]; then
                local cleanup_id
                capture_add_task cleanup_id "CLEANUP_ERROR" "${model_id}" "${model_name}" \
                    1 "${error_cert_id}" "${params_json}" 80
                echo "Created: ${cleanup_id}"
            fi
        done
        fi
    else
        echo "Skipping error injection (RUN_ERROR_INJECTION=false or no preset)"
    fi

    echo "Generated tasks for model ${model_name}"
}


# Generate evaluate tasks for an edit after it exists on disk.
# Usage: generate_evaluate_tasks <model_id> <model_name> <edit_dep_id> <preset_id> <edit_spec> <version> <cert_runs>
generate_evaluate_tasks() {
    local model_id="$1"
    local model_name="$2"
    local edit_dep_id="$3"
    local preset_id="$4"
    local edit_spec="$5"
    local version="$6"
    local cert_runs="$7"
    if ! [[ "${cert_runs}" =~ ^-?[0-9]+$ ]]; then
        cert_runs=1
    fi
    if [[ ${cert_runs} -lt 0 ]]; then
        cert_runs=0
    fi

    # evaluate_EDIT depends on edit creation + preset.
    if [[ ${cert_runs} -gt 0 ]]; then
        for run in $(seq 1 "${cert_runs}"); do
            local cert_deps="${edit_dep_id}"
            if [[ -n "${preset_id}" ]]; then
                cert_deps="${edit_dep_id},${preset_id}"
            fi
            local cert_id=""
            capture_add_task cert_id "evaluate_EDIT" "${model_id}" "${model_name}" \
                "$(estimate_model_memory "${model_id}" "evaluate_EDIT")" \
                "${cert_deps}" '{"edit_spec": "'"${edit_spec}"'", "version": "'"${version}"'", "run": '"${run}"'}' 74
            echo "Created: ${cert_id}"
        done
    fi
}

# Generate all tasks for all models
# Usage: generate_all_tasks <model_id...>
# Pass any number of model IDs as arguments.
generate_all_tasks() {
    local models=("$@")

    export TASK_SEQUENCE=0

    for idx in "${!models[@]}"; do
        local model_id="${models[$idx]}"
        if [[ -n "${model_id}" ]]; then
            # Use full model id (including org) for filesystem-safe name to avoid collisions
            # Example: mistralai/Mistral-7B-v0.1 -> mistralai__mistral-7b-v0.1
            local model_name
            model_name=$(printf '%s' "${model_id}" \
                | tr '[:upper:]' '[:lower:]' \
                | sed 's#/#__#g' \
                | tr ' ' '_' \
                | tr -cd '[:alnum:]_-')
            echo ""
            echo "=== Generating tasks for model $((idx + 1)): ${model_name} ==="
            generate_model_tasks "$((idx + 1))" "${model_id}" "${model_name}"
        fi
    done

    # Initial dependency resolution
    echo ""
    echo "=== Resolving initial dependencies ==="
    local moved=$(resolve_dependencies)
    echo "Moved ${moved} tasks to ready queue"

    # Update state
    update_progress_state

    echo ""
    print_queue_stats
}
