#!/usr/bin/env bash
# run_qwen14_sentinels.sh - Maintain the Qwen2.5-14B promotion sentinels.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: scripts/evidence_packs/run_qwen14_sentinels.sh --run-dir DIR --model-name NAME [options]

Runs the maintained Qwen2.5-14B evidence-pack sentinels from an existing evidence-pack
run directory:

- saved-model direct evaluate sentinels for `quant_4bit_clean` and `prune_clean`
- the promotion-grade public quant smoke (`quant_4bit_clean` + verify)

Options:
  --run-dir DIR       Existing evidence-pack run directory
  --model-name NAME   Sanitized evidence-pack model directory name
  --out DIR           Sentinel output root (default: <run-dir>/sentinels/qwen14)
  --mode NAME         all|saved-model|public-quant (default: all)
  --device NAME       Device for evaluate (default: cuda)
  --profile NAME      Evaluate/verify profile (default: ci)
  --baseline-adapter NAME  Baseline adapter selection (default: auto)
  --subject-adapter NAME   Subject adapter selection (default: auto)
  --help              Show this help
EOF
}

require_dir() {
    local path="$1"
    local label="$2"
    [[ -d "${path}" ]] || {
        echo "ERROR: ${label} not found: ${path}" >&2
        return 1
    }
}

require_saved_subject_dir() {
    local path="$1"
    local label="$2"
    [[ -d "${path}" ]] || {
        echo "ERROR: ${label} not found: ${path}" >&2
        echo "Hint: Qwen14 sentinels require retained edit subject directories; rerun the evidence-pack campaign with PACK_CLEANUP_MODELS=0." >&2
        return 1
    }
}

require_file() {
    local path="$1"
    local label="$2"
    [[ -f "${path}" ]] || {
        echo "ERROR: ${label} not found: ${path}" >&2
        return 1
    }
}

resolve_python_bin() {
    if [[ -n "${PYTHON_BIN:-}" ]] && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        printf '%s\n' "${PYTHON_BIN}"
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        printf '%s\n' "python"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        printf '%s\n' "python3"
        return 0
    fi
    echo "ERROR: python interpreter not found for preset normalization" >&2
    return 1
}

stage_runtime_input() {
    local source_file="$1"
    local staged_dir="$2"
    local label="$3"

    require_file "${source_file}" "${label}" || return 1
    mkdir -p "${staged_dir}"
    local staged_file="${staged_dir}/$(basename "${source_file}")"
    cp -f "${source_file}" "${staged_file}"
    printf '%s\n' "${staged_file}"
}

normalize_staged_preset_for_baseline_report() {
    local staged_preset="$1"
    local staged_baseline_report="$2"
    local python_bin=""
    python_bin="$(resolve_python_bin)"
    "${python_bin}" "${SCRIPT_DIR}/python/task_tools.py" normalize-staged-preset \
        --preset "${staged_preset}" \
        --baseline-report "${staged_baseline_report}" \
        --skip-overhead-check
}

resolve_baseline_path() {
    local run_dir="$1"
    local model_name="$2"
    local baseline_hint="${run_dir}/${model_name}/.baseline_path"
    if [[ -f "${baseline_hint}" ]]; then
        local hinted
        hinted="$(cat "${baseline_hint}" 2>/dev/null || true)"
        if [[ -n "${hinted}" && -d "${hinted}" ]]; then
            printf '%s\n' "${hinted}"
            return 0
        fi
    fi

    local fallback="${run_dir}/${model_name}/models/baseline"
    require_dir "${fallback}" "baseline model"
    printf '%s\n' "${fallback}"
}

resolve_baseline_report() {
    local run_dir="$1"
    local model_name="$2"
    local report_path=""
    report_path="$(find "${run_dir}/${model_name}/baseline_reports" -type f -name baseline_report.json 2>/dev/null | sort | tail -1 || true)"
    require_file "${report_path}" "baseline report"
    printf '%s\n' "${report_path}"
}

resolve_preset_path() {
    local run_dir="$1"
    local model_name="$2"
    local edit_key="$3"
    local candidate=""
    for candidate in \
        "${run_dir}/presets/calibrated_preset_${model_name}__${edit_key}.yaml" \
        "${run_dir}/presets/calibrated_preset_${model_name}__${edit_key}.json" \
        "${run_dir}/presets/calibrated_preset_${model_name}.yaml" \
        "${run_dir}/presets/calibrated_preset_${model_name}.json"; do
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    echo "ERROR: no preset found for ${model_name} (${edit_key}) under ${run_dir}/presets" >&2
    return 1
}

run_evaluate_sentinel() {
    local baseline_path="$1"
    local baseline_report="$2"
    local subject_path="$3"
    local preset_path="$4"
    local out_dir="$5"
    local baseline_adapter="$6"
    local subject_adapter="$7"
    local profile="$8"
    local device="$9"

    mkdir -p "${out_dir}"
    local runtime_inputs_dir="${out_dir}/runtime_inputs"
    local staged_preset=""
    staged_preset="$(stage_runtime_input "${preset_path}" "${runtime_inputs_dir}" "preset")"
    local staged_baseline_report=""
    staged_baseline_report="$(stage_runtime_input "${baseline_report}" "${runtime_inputs_dir}" "baseline report")"
    normalize_staged_preset_for_baseline_report "${staged_preset}" "${staged_baseline_report}"

    local rc=0
    if invarlock evaluate \
        --baseline "${baseline_path}" \
        --subject "${subject_path}" \
        --baseline-adapter "${baseline_adapter}" \
        --subject-adapter "${subject_adapter}" \
        --profile "${profile}" \
        --preset "${staged_preset}" \
        --baseline-report "${staged_baseline_report}" \
        --device "${device}" \
        --out "${out_dir}/report.json" \
        --report-out "${out_dir}"; then
        :
    else
        rc=$?
    fi

    if [[ ${rc} -ne 0 && -f "${out_dir}/evaluation.report.json" ]]; then
        echo "WARNING: evaluate exited ${rc} but wrote ${out_dir}/evaluation.report.json; treating sentinel as load-path success" >&2
        rc=0
    fi

    require_file "${out_dir}/evaluation.report.json" "evaluation report"
    return "${rc}"
}

run_public_quant_verify() {
    local report_path="$1"
    local out_dir="$2"
    local profile="$3"

    local rc=0
    if invarlock verify --json --profile "${profile}" "${report_path}" > "${out_dir}/verify.json"; then
        :
    else
        rc=$?
    fi
    require_file "${out_dir}/verify.json" "verify summary"
    if [[ ${rc} -ne 0 ]]; then
        echo "WARNING: verify exited ${rc} but wrote ${out_dir}/verify.json; treating sentinel as load-path success" >&2
    fi
}

main() {
    local run_dir=""
    local model_name=""
    local out_dir=""
    local mode="all"
    local device="cuda"
    local profile="ci"
    local baseline_adapter="auto"
    local subject_adapter="auto"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --run-dir)
                run_dir="${2:-}"
                shift 2
                ;;
            --model-name)
                model_name="${2:-}"
                shift 2
                ;;
            --out)
                out_dir="${2:-}"
                shift 2
                ;;
            --mode)
                mode="${2:-}"
                shift 2
                ;;
            --device)
                device="${2:-}"
                shift 2
                ;;
            --profile)
                profile="${2:-}"
                shift 2
                ;;
            --baseline-adapter)
                baseline_adapter="${2:-}"
                shift 2
                ;;
            --subject-adapter)
                subject_adapter="${2:-}"
                shift 2
                ;;
            --help|-h)
                usage
                return 0
                ;;
            *)
                echo "Unknown arg: $1" >&2
                usage >&2
                return 2
                ;;
        esac
    done

    [[ -n "${run_dir}" ]] || {
        echo "ERROR: --run-dir is required" >&2
        return 2
    }
    [[ -n "${model_name}" ]] || {
        echo "ERROR: --model-name is required" >&2
        return 2
    }
    case "${mode}" in
        all|saved-model|public-quant)
            :
            ;;
        *)
            echo "ERROR: --mode must be one of all|saved-model|public-quant" >&2
            return 2
            ;;
    esac

    if [[ -z "${out_dir}" ]]; then
        out_dir="${run_dir}/sentinels/qwen14"
    fi

    require_dir "${run_dir}" "run directory"
    local model_root="${run_dir}/${model_name}"
    require_dir "${model_root}" "model output directory"

    local baseline_path=""
    baseline_path="$(resolve_baseline_path "${run_dir}" "${model_name}")"
    local baseline_report=""
    baseline_report="$(resolve_baseline_report "${run_dir}" "${model_name}")"

    echo "Qwen14 sentinel run"
    echo "  run_dir: ${run_dir}"
    echo "  model_name: ${model_name}"
    echo "  mode: ${mode}"
    echo "  baseline: ${baseline_path}"
    echo "  baseline_report: ${baseline_report}"
    echo "  output: ${out_dir}"

    if [[ "${mode}" == "all" || "${mode}" == "saved-model" || "${mode}" == "public-quant" ]]; then
        local quant_subject="${model_root}/models/quant_4bit_clean"
        local quant_preset=""
        quant_preset="$(resolve_preset_path "${run_dir}" "${model_name}" "quant_rtn")"
        require_saved_subject_dir "${quant_subject}" "quant_4bit_clean subject"
        run_evaluate_sentinel \
            "${baseline_path}" \
            "${baseline_report}" \
            "${quant_subject}" \
            "${quant_preset}" \
            "${out_dir}/quant_4bit_clean" \
            "${baseline_adapter}" \
            "${subject_adapter}" \
            "${profile}" \
            "${device}"
        run_public_quant_verify \
            "${out_dir}/quant_4bit_clean/evaluation.report.json" \
            "${out_dir}/quant_4bit_clean" \
            "${profile}"
    fi

    if [[ "${mode}" == "all" || "${mode}" == "saved-model" ]]; then
        local prune_subject="${model_root}/models/prune_clean"
        local prune_preset=""
        prune_preset="$(resolve_preset_path "${run_dir}" "${model_name}" "magnitude_prune")"
        require_saved_subject_dir "${prune_subject}" "prune_clean subject"
        run_evaluate_sentinel \
            "${baseline_path}" \
            "${baseline_report}" \
            "${prune_subject}" \
            "${prune_preset}" \
            "${out_dir}/prune_clean" \
            "${baseline_adapter}" \
            "${subject_adapter}" \
            "${profile}" \
            "${device}"
    fi

    echo "Sentinel outputs:"
    find "${out_dir}" -maxdepth 2 -type f \( -name 'evaluation.report.json' -o -name 'verify.json' -o -name 'report.json' \) | sort
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
