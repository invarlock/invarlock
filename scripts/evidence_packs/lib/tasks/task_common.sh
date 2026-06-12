#!/usr/bin/env bash
# task_common.sh - Shared helpers for dynamic task execution
# Version: evidence-packs-v1 (InvarLock Evidence Pack Suite)
# Dependencies: jq, python3, invarlock CLI, task_serialization.sh
# Usage: sourced by gpu_worker.sh/validation_suite.sh for per-task execution
#
# Each function executes a single atomic task type with explicit parameters.
# These are extracted from the original monolithic process_model() function
# to enable parallel execution across GPUs.

# Source dependencies
export TASK_COMMON_LOADED=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK_REPO_ROOT="${PACK_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
PACK_REPO_PYTHONPATH="${PACK_REPO_ROOT}/src"
# shellcheck source=../core/runtime.sh
source "${SCRIPT_DIR}/../core/runtime.sh"
# shellcheck source=../config/dataset_provider_config.sh
source "${SCRIPT_DIR}/../config/dataset_provider_config.sh"
[[ -z "${QUEUE_MANAGER_LOADED:-}" ]] && source "${SCRIPT_DIR}/../queue/queue_manager.sh" && export QUEUE_MANAGER_LOADED=1

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

    # Conservative defaults that satisfy CI pairing/coverage floors for evidence packs.
    # Use zero overlap (stride == seq_len) and ≥180 windows to avoid E001/E005
    # if workers start without the main suite wrapper.
    case "${model_size}" in
        "7")
            # Shorter sequences are substantially faster on WT-2 (many samples are
            # short, so longer seq_len mostly pads).
            echo "512:512:192:192:96"
            ;;
        "13")
            echo "512:512:192:192:64"
            ;;
        "30")
            echo "1024:1024:192:192:48"
            ;;
        "40")
            echo "1024:1024:192:192:32"
            ;;
        "moe")
            echo "1024:1024:192:192:8"
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

_bootstrap_replicates_floor_for_tier() {
    local tier="${1:-balanced}"
    case "${tier}" in
        conservative)
            echo "1500"
            ;;
        balanced)
            echo "1200"
            ;;
        aggressive)
            echo "800"
            ;;
        *)
            echo "1200"
            ;;
    esac
}

_resolve_bootstrap_replicates() {
    local model_size="$1"
    local tier="${2:-balanced}"

    local bootstrap_replicates=2000
    if _is_large_model "${model_size}"; then
        bootstrap_replicates=1000
    fi
    if [[ -n "${INVARLOCK_BOOTSTRAP_N:-}" ]]; then
        bootstrap_replicates="${INVARLOCK_BOOTSTRAP_N}"
    fi

    local floor=""
    floor="$(_bootstrap_replicates_floor_for_tier "${tier}")"
    if [[ "${bootstrap_replicates}" =~ ^[0-9]+$ && "${floor}" =~ ^[0-9]+$ ]]; then
        if [[ "${bootstrap_replicates}" -lt "${floor}" ]]; then
            bootstrap_replicates="${floor}"
        fi
    fi

    echo "${bootstrap_replicates}"
}

_default_ci_min_windows() {
    local seq_len="${1:-}"
    local tier="${2:-${INVARLOCK_TIER:-balanced}}"
    local dataset_kind="${3:-}"

    if [[ -z "${dataset_kind}" ]]; then
        dataset_kind="$(pack_dataset_provider_kind "${INVARLOCK_DATASET:-}")"
    fi

    if [[ -n "${INVARLOCK_CERT_MIN_WINDOWS:-}" ]]; then
        echo "${INVARLOCK_CERT_MIN_WINDOWS}"
        return
    fi

    local default_windows=256
    # The balanced tier enforces a 50k token minimum. On short-text datasets like
    # WikiText-2, larger seq_len does not imply larger effective token counts
    # because many windows are heavily padded. The Qwen2.5-14B clean controls only
    # reached 45,723 total tokens at 352+352 windows, so keep a higher evidence-pack
    # floor for balanced WikiText-2 runs instead of weakening the public policy.
    if [[ "${tier}" == "balanced" && "${dataset_kind}" == "wikitext2" && "${seq_len}" =~ ^[0-9]+$ && "${seq_len}" -ge 512 ]]; then
        default_windows=400
    elif [[ "${seq_len}" =~ ^[0-9]+$ && "${seq_len}" -le 256 ]]; then
        default_windows=352
    fi

    echo "${default_windows}"
}

_effective_ci_planning_target() {
    local model_size="$1"
    local tier="${2:-balanced}"
    local dataset_kind="${3:-wikitext2}"
    [[ "${model_size}" == "13" && "${tier}" == "balanced" && "${dataset_kind}" == "wikitext2" ]]
}

_plan_effective_ci_schedule() {
    local model_ref="$1"
    local model_size="$2"
    local tier="$3"
    local dataset_kind="$4"
    local split="$5"
    local seed="$6"

    if ! _effective_ci_planning_target "${model_size}" "${tier}" "${dataset_kind}"; then
        return 0
    fi
    if [[ -n "${INVARLOCK_CERT_MIN_WINDOWS:-}" ]]; then
        jq -n --arg reason "manual_window_override" '{status:"skipped", reason:$reason}'
        return 0
    fi
    if [[ -z "${model_ref}" || ! -e "${model_ref}" ]]; then
        jq -n --arg reason "missing_model_ref" '{status:"skipped", reason:$reason}'
        return 0
    fi

    local planner="${SCRIPT_DIR}/../../python/task_tools.py"
    local -a candidate_args=()
    local seq_len=""
    for seq_len in 512 768 1024 1536; do
        local min_windows=""
        min_windows="$(_default_ci_min_windows "${seq_len}" "${tier}" "${dataset_kind}")"
        candidate_args+=(--candidate "${seq_len}:${min_windows}:${min_windows}")
    done

    PYTHONPATH="${PACK_REPO_PYTHONPATH}" \
        INVARLOCK_ALLOW_REMOTE_CODE="${INVARLOCK_ALLOW_REMOTE_CODE:-}" \
        _cmd_python "${planner}" \
            plan-effective-windows \
            --model-path "${model_ref}" \
            --dataset-provider "${dataset_kind}" \
            --split "${split}" \
            --seed "${seed}" \
            --tier "${tier}" \
            --profile "ci" \
            "${candidate_args[@]}"
}

_apply_effective_ci_schedule() {
    local plan_json="$1"
    local log_file="$2"

    [[ -n "${plan_json}" ]] || return 0

    local status=""
    status="$(printf '%s' "${plan_json}" | jq -r '.status // "unknown"')"
    if [[ "${status}" == "skipped" ]]; then
        echo "  Effective CI planning skipped: $(printf '%s' "${plan_json}" | jq -r '.reason // "unknown"')" >> "${log_file}"
        return 0
    fi

    local min_tokens_target=""
    local effective_min_tokens=""
    min_tokens_target="$(printf '%s' "${plan_json}" | jq -r '.min_tokens_target // 0')"
    effective_min_tokens="$(printf '%s' "${plan_json}" | jq -r '.effective_min_tokens // 0')"
    echo "  Effective CI planning target: min_tokens=${min_tokens_target}, with_headroom=${effective_min_tokens}" >> "${log_file}"
    while IFS= read -r candidate; do
        [[ -n "${candidate}" ]] || continue
        echo "  Candidate: ${candidate}" >> "${log_file}"
    done < <(
        printf '%s' "${plan_json}" | jq -r '
            .candidates[]? |
            "seq=\(.seq_len) stride=\(.stride) requested=\(.requested_preview)+\(.requested_final) actual=\(.actual_preview)+\(.actual_final) tokens=\(.total_tokens) floor=\(.effective_min_tokens) floor_met=\(.tokens_floor_met) reason=\(.reason)"
        '
    )

    if [[ "${status}" == "selected" ]]; then
        local selected=""
        selected="$(printf '%s' "${plan_json}" | jq -r '.selected | "\(.seq_len):\(.stride):\(.actual_preview):\(.actual_final)"')"
        echo "  Selected effective CI schedule: ${selected}" >> "${log_file}"
        printf '%s\n' "${selected}"
        return 0
    fi

    echo "ERROR: Effective CI planning found no viable balanced WikiText-2 schedule. Switch dataset provider (for example hf_text/allenai-c4)." >> "${log_file}"
    return 1
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
    _cmd_python "${SCRIPT_DIR}/../../python/task_tools.py" model-revision "${path}" "${model_id}" 2>/dev/null
}

# Check if model is large (30B+) and needs special handling.
# Changed threshold from 70 to 13 to cover 14B-class dense checkpoints:
# - Skips overhead check (avoids loading the edited model twice in compare-mode runs)
_is_large_model() {
    local model_size="$1"
    if [[ "${model_size}" == "moe" ]]; then
        return 0
    fi
    if [[ "${model_size}" =~ ^[0-9]+$ ]]; then
        [[ ${model_size} -ge 13 ]]
        return
    fi
    [[ "${model_size}" =~ 13 || "${model_size}" =~ 14 || "${model_size}" =~ 30 || "${model_size}" =~ 32 || "${model_size}" =~ 34 || "${model_size}" =~ 40 || "${model_size}" =~ 70 || "${model_size}" =~ 72 || "${model_size}" =~ 65 || "${model_size}" =~ 80 || "${model_size}" =~ 90 ]]
}

_baseline_report_wait_secs() {
    local model_size="$1"
    local preview_n="$2"
    local final_n="$3"
    local wait_secs="${PACK_BASELINE_REPORT_WAIT_SECS:-240}"

    if _is_large_model "${model_size}"; then
        wait_secs="${PACK_BASELINE_REPORT_WAIT_SECS_LARGE:-1800}"
    else
        local heavy_window_floor="${PACK_BASELINE_REPORT_WAIT_HEAVY_WINDOW_TOTAL_MIN:-800}"
        if ! [[ "${heavy_window_floor}" =~ ^[0-9]+$ ]] || [[ "${heavy_window_floor}" -lt 1 ]]; then
            heavy_window_floor=800
        fi
        if [[ "${preview_n}" =~ ^[0-9]+$ && "${final_n}" =~ ^[0-9]+$ ]]; then
            local total_windows=$((preview_n + final_n))
            if [[ "${total_windows}" -ge "${heavy_window_floor}" ]]; then
                wait_secs="${PACK_BASELINE_REPORT_WAIT_SECS_HEAVY_WINDOWS:-1800}"
            fi
        fi
    fi

    if ! [[ "${wait_secs}" =~ ^[0-9]+$ ]] || [[ "${wait_secs}" -lt 1 ]]; then
        wait_secs=240
    fi
    echo "${wait_secs}"
}

# Resolve the concrete InvarLock adapter name for a model path/ID.
_resolve_invarlock_adapter() {
    local model_id="$1"
    if [[ -z "${model_id}" ]]; then
        return 1
    fi
    _cmd_python "${SCRIPT_DIR}/../../python/task_tools.py" resolve-adapter "${model_id}"
}

_validate_evaluate_baseline_report() {
    local report_path="$1"
    local expected_adapter="$2"
    local expected_profile="$3"
    local expected_tier="$4"
    local expected_assurance="${5:-off}"

    if [[ -z "${report_path}" || ! -f "${report_path}" ]]; then
        return 1
    fi

    _cmd_python "${SCRIPT_DIR}/../../python/task_tools.py" validate-baseline-report \
        "${report_path}" "${expected_adapter}" "${expected_profile}" "${expected_tier}" "${expected_assurance}"
}

_stage_runtime_input_for_eval() {
    local source_file="$1"
    local cert_dir="$2"
    local log_file="$3"
    local label="$4"

    if [[ -z "${source_file}" || ! -f "${source_file}" ]]; then
        return 1
    fi

    local staged_dir="${cert_dir}/runtime_inputs"
    mkdir -p "${staged_dir}" || return 1

    local staged_file="${staged_dir}/$(basename "${source_file}")"
    cp -f "${source_file}" "${staged_file}" >> "${log_file}" 2>&1 || return 1
    if [[ -n "${label}" ]]; then
        echo "  Staged ${label} for evaluate runtime: ${staged_file}" >> "${log_file}"
    fi
    printf '%s\n' "$(cd "$(dirname "${staged_file}")" && pwd)/$(basename "${staged_file}")"
}

_stage_preset_for_eval() {
    _stage_runtime_input_for_eval "$1" "$2" "$3" "preset"
}

_stage_baseline_report_for_eval() {
    _stage_runtime_input_for_eval "$1" "$2" "$3" "baseline report"
}

_pack_defer_report_rendering_enabled() {
    local value="${PACK_DEFER_REPORT_RENDERING:-${PACK_DEFER_OPTIONAL_REPORT_RENDERING:-0}}"
    case "${value}" in
        1|true|TRUE|yes|YES|on|ON)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

_pack_evaluate_assurance_mode() {
    local value="${PACK_EVALUATE_ASSURANCE:-off}"
    case "${value}" in
        strict|off)
            printf '%s\n' "${value}"
            ;;
        *)
            echo "ERROR: PACK_EVALUATE_ASSURANCE must be strict or off, got: ${value}" >&2
            return 1
            ;;
    esac
}

_normalize_staged_preset_for_eval() {
    local staged_preset="$1"
    local seq_len="$2"
    local stride="$3"
    local preview_n="$4"
    local final_n="$5"
    local skip_overhead="$6"
    local log_file="$7"
    local baseline_report="${8:-}"

    if [[ -z "${staged_preset}" || ! -f "${staged_preset}" ]]; then
        return 1
    fi

    local normalize_args=(
        "task_tools.py"
        "normalize-staged-preset"
        --preset "${staged_preset}"
    )
    if [[ -n "${baseline_report}" && -f "${baseline_report}" ]]; then
        normalize_args+=(--baseline-report "${baseline_report}")
    else
        normalize_args+=(
            --seq-len "${seq_len}"
            --stride "${stride}"
            --preview-n "${preview_n}"
            --final-n "${final_n}"
        )
    fi
    if [[ "${skip_overhead}" == "1" ]]; then
        normalize_args+=(--skip-overhead-check)
    fi

    local previous_python_bin="${PYTHON_BIN:-}"
    local had_python_bin="0"
    if [[ -v PYTHON_BIN ]]; then
        had_python_bin="1"
    fi
    if [[ "${had_python_bin}" != "1" ]]; then
        local active_python=""
        active_python="$(command -v python 2>/dev/null || true)"
        if [[ -n "${active_python}" ]] && "${active_python}" -c "import yaml" >/dev/null 2>&1; then
            export PYTHON_BIN="${active_python}"
        fi
    fi

    _runtime_python "${normalize_args[@]}" >> "${log_file}" 2>&1 || {
        if [[ "${had_python_bin}" == "1" ]]; then
            export PYTHON_BIN="${previous_python_bin}"
        else
            unset PYTHON_BIN
        fi
        return 1
    }
    if [[ "${had_python_bin}" == "1" ]]; then
        export PYTHON_BIN="${previous_python_bin}"
    else
        unset PYTHON_BIN
    fi
    if [[ -n "${baseline_report}" && -f "${baseline_report}" ]]; then
        echo "  Normalized staged preset dataset for evaluate runtime from baseline report: ${baseline_report}" >> "${log_file}"
    else
        echo "  Normalized staged preset dataset for evaluate runtime: seq=${seq_len}, stride=${stride}, preview=${preview_n}, final=${final_n}" >> "${log_file}"
    fi
    if [[ "${skip_overhead}" == "1" ]]; then
        echo "  Injected context.run.skip_overhead_check=true into staged preset" >> "${log_file}"
    fi
}

_ensure_evaluate_baseline_report() {
    local baseline_root="$1"
    local abs_baseline_path="$2"
    local profile_flag="$3"
    local tier="$4"
    local seq_len="$5"
    local stride="$6"
    local preview_n="$7"
    local final_n="$8"
    local eval_batch="$9"
    local bootstrap_replicates="${10}"
    local model_size="${11}"
    local log_file="${12}"

    mkdir -p "${baseline_root}"
    local abs_baseline_root
    abs_baseline_root="$(cd "${baseline_root}" && pwd)"

    local baseline_report_file="${abs_baseline_root}/baseline_report.json"

    local adapter_name
    adapter_name="$(_resolve_invarlock_adapter "${abs_baseline_path}" 2>/dev/null || true)"
    adapter_name="$(printf '%s' "${adapter_name}" | xargs)"
    if [[ -z "${adapter_name}" ]]; then
        # Fallback for odd environments; must match what invarlock evaluate will resolve.
        adapter_name="hf_causal"
    fi
    local evaluate_assurance
    evaluate_assurance="$(_pack_evaluate_assurance_mode)" || return 1

    if [[ -f "${baseline_report_file}" ]]; then
        if _validate_evaluate_baseline_report "${baseline_report_file}" "${adapter_name}" "${profile_flag}" "${tier}" "${evaluate_assurance}" 2>/dev/null; then
            echo "${baseline_report_file}"
            return 0
        fi
        rm -f "${baseline_report_file}"
    fi

    local lock_dir="${abs_baseline_root}/.baseline_lock"
    if mkdir "${lock_dir}" 2>/dev/null; then
        # Re-check after acquiring the lock.
        if [[ -f "${baseline_report_file}" ]]; then
            if _validate_evaluate_baseline_report "${baseline_report_file}" "${adapter_name}" "${profile_flag}" "${tier}" "${evaluate_assurance}" 2>/dev/null; then
                rmdir "${lock_dir}" 2>/dev/null || true
                echo "${baseline_report_file}"
                return 0
            fi
            rm -f "${baseline_report_file}"
        fi

        echo "  Generating reusable baseline report (adapter=${adapter_name}, tier=${tier})" >> "${log_file}"

        local baseline_config_root="${abs_baseline_root}/config_root"
        mkdir -p "${baseline_config_root}/runtime/profiles"
        local skip_overhead_config_yaml=""
        if _is_large_model "${model_size}"; then
            skip_overhead_config_yaml=$'context:\n  run:\n    skip_overhead_check: true'
            echo "  Large model (${model_size}): context.run.skip_overhead_check=true" >> "${log_file}"
        fi
        cat > "${baseline_config_root}/runtime/profiles/ci.yaml" << YAML
model:
  device_map: "auto"
  dtype: "bfloat16"
$(pack_model_trust_remote_code_yaml "  ")
  low_cpu_mem_usage: true
dataset:
  seq_len: ${seq_len}
  stride: ${stride}
  preview_n: ${preview_n}
  final_n: ${final_n}
eval:
  bootstrap:
    replicates: ${bootstrap_replicates}
    alpha: 0.05
${skip_overhead_config_yaml}
YAML

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

        local dataset_provider_yaml
        dataset_provider_yaml="$(pack_render_dataset_provider_yaml "${INVARLOCK_DATASET:-wikitext2}")"

        local baseline_yaml="${abs_baseline_root}/baseline_noop.yaml"
        cat > "${baseline_yaml}" << YAML
model:
  id: "${abs_baseline_path}"
  adapter: "${adapter_name}"
  device: "auto"
  device_map: "auto"
  dtype: "bfloat16"
$(pack_model_trust_remote_code_yaml "  ")
  low_cpu_mem_usage: true

dataset:
${dataset_provider_yaml}
  split: validation
  seq_len: ${seq_len}
  stride: ${stride}
  preview_n: ${preview_n}
  final_n: ${final_n}
  seed: 42

edit:
  name: "noop"
  plan: {}

context:
  assurance:
    mode: "${evaluate_assurance}"

assurance:
  mode: "${evaluate_assurance}"

guards:
  order:
${guards_order_yaml}

auto:
  enabled: true
  tier: "${tier}"

eval:
  bootstrap:
    replicates: ${bootstrap_replicates}
    alpha: 0.05
  batch_size: ${eval_batch}
YAML

        local baseline_out="${abs_baseline_root}/runs"
        mkdir -p "${baseline_out}"

        local -a extra_env=()
        if _is_large_model "${model_size}"; then
            extra_env+=(INVARLOCK_SKIP_OVERHEAD_CHECK=1)
        fi
        extra_env+=("PYTHONPATH=${PACK_REPO_PYTHONPATH}")
        extra_env+=(INVARLOCK_STORE_EVAL_WINDOWS=1)
        if pack_remote_code_allowed; then
            extra_env+=(INVARLOCK_ALLOW_REMOTE_CODE=1)
        fi
        extra_env+=("INVARLOCK_CONFIG_ROOT=${baseline_config_root}")

        local exit_code=0
        (
            export "${extra_env[@]}"
            _pack_run_from_config \
                --config "${baseline_yaml}" \
                --profile "${profile_flag}" \
                --tier "${tier}" \
                --out "${baseline_out}" \
                --edit-label "noop" >> "${log_file}" 2>&1
        ) || exit_code=$?

        if [[ ${exit_code} -eq 0 ]]; then
            local report_file
            report_file=$(find "${baseline_out}" -mindepth 2 -maxdepth 2 -name "report.json" -type f 2>/dev/null | sort | tail -1)
            if [[ -n "${report_file}" && -f "${report_file}" ]]; then
                local tmp_report="${baseline_report_file}.tmp"
                cp "${report_file}" "${tmp_report}" 2>/dev/null || true
                if [[ -f "${tmp_report}" ]]; then
                    _cmd_python "${SCRIPT_DIR}/../../python/task_tools.py" stamp-baseline-report-seed \
                        --report "${tmp_report}" \
                        --seed 42 >> "${log_file}" 2>&1 || {
                        echo "  WARNING: Failed to stamp baseline report seed into ${tmp_report}" >> "${log_file}"
                    }
                    mv "${tmp_report}" "${baseline_report_file}" 2>/dev/null || true
                fi
            fi
        fi

        rmdir "${lock_dir}" 2>/dev/null || true

        if [[ -f "${baseline_report_file}" ]] && _validate_evaluate_baseline_report "${baseline_report_file}" "${adapter_name}" "${profile_flag}" "${tier}" "${evaluate_assurance}" 2>/dev/null; then
            echo "${baseline_report_file}"
            return 0
        fi
        rm -f "${baseline_report_file}"
        return 1
    fi

    local wait_interval="${PACK_BASELINE_REPORT_WAIT_INTERVAL_SECS:-2}"
    local wait_secs=""
    wait_secs="$(_baseline_report_wait_secs "${model_size}" "${preview_n}" "${final_n}")"
    if ! [[ "${wait_interval}" =~ ^[0-9]+$ ]] || [[ "${wait_interval}" -lt 1 ]]; then
        wait_interval=2
    fi
    local wait_iters=$((wait_secs / wait_interval))
    if [[ "${wait_iters}" -lt 1 ]]; then
        wait_iters=1
    fi

    echo "  Waiting for baseline report to be generated by another worker... (timeout=${wait_secs}s)" >> "${log_file}"
    for _ in $(seq 1 "${wait_iters}"); do
        if [[ -f "${baseline_report_file}" ]] && _validate_evaluate_baseline_report "${baseline_report_file}" "${adapter_name}" "${profile_flag}" "${tier}" "${evaluate_assurance}" 2>/dev/null; then
            echo "${baseline_report_file}"
            return 0
        fi
        _sleep "${wait_interval}"
    done

    return 1
}

# Resolve an edit spec to concrete parameters and directory name.
# Returns JSON with status, edit_dir_name, and resolved params.
resolve_edit_params() {
    local model_output_dir="$1"
    local edit_spec="$2"
    local version_hint="${3:-}"

    _cmd_python "${SCRIPT_DIR}/../../python/task_tools.py" resolve-edit-params \
        "${model_output_dir}" "${edit_spec}" "${version_hint}"
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

    _cmd_python "${SCRIPT_DIR}/../../python/task_tools.py" write-model-profile \
        "${baseline_dir}" "${model_id}" >/dev/null 2>&1 || true
}
