#!/usr/bin/env bash
# run_suite.sh - CLI entrypoint for proof pack suites.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/validation_suite.sh
source "${SCRIPT_DIR}/lib/validation_suite.sh"
# shellcheck source=suites.sh
source "${SCRIPT_DIR}/suites.sh"

pack_usage() {
    cat <<'EOF'
InvarLock Proof Pack Suite
Usage: scripts/proof_packs/run_suite.sh [options]

WARNING: This runs the suite WITHOUT creating a proof pack.
         For distributable validation artifacts, use scripts/proof_packs/run_pack.sh instead.

Options:
  --suite NAME         Suite name (subset|showcase|workshop3|full)
  --models CSV         Comma-separated model IDs to run (overrides suite defaults)
  --net 1|0            Enable network access for preflight/downloads (default: 0)
  --out DIR            Output directory (default: ./proof_pack_runs/<suite>_<timestamp>)
  --determinism MODE   Determinism mode (strict|throughput)
  --repeats N          Determinism repeat count metadata (default: 0)
  --scenario-ids IDS   Comma-separated scenario IDs to include (filters scenarios.json before queue generation)
  --calibrate-only     Only run calibration tasks (implies PACK_SUITE_MODE=calibrate-only)
  --errors-only        Only run error injection scenarios (still performs calibration unless presets are provided)
  --run-only           Run edits/certs only (implies resume)
  --resume             Resume an existing run directory
  --help               Show this help message
EOF
}

pack_apply_entrypoint_determinism() {
    case "${PACK_DETERMINISM}" in
        strict|throughput)
            :
            ;;
        *)
            PACK_DETERMINISM="throughput"
            ;;
    esac
    export PACK_DETERMINISM

    if [[ "${PACK_DETERMINISM}" == "strict" ]]; then
        export NVIDIA_TF32_OVERRIDE=0
        export CUDNN_BENCHMARK=0
        export CUBLAS_WORKSPACE_CONFIG=:4096:8
    else
        export NVIDIA_TF32_OVERRIDE=1
        export CUDNN_BENCHMARK=1
        unset CUBLAS_WORKSPACE_CONFIG 2>/dev/null || true
    fi
}

pack_entrypoint() {
    set -euo pipefail

    local suite="${PACK_SUITE:-subset}"
    local models_csv="${PACK_MODELS_CSV:-${PACK_MODELS:-}}"
    local net="${PACK_NET:-0}"
    local out="${PACK_OUTPUT_DIR:-${OUTPUT_DIR:-}}"
    local determinism="${PACK_DETERMINISM:-throughput}"
    local repeats="${PACK_REPEATS:-0}"
    local suite_mode="${PACK_SUITE_MODE:-full}"
    local resume_flag="${RESUME_FLAG:-false}"
    local scenario_ids="${PACK_SCENARIO_IDS:-}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                pack_usage
                return 0
                ;;
            --suite)
                suite="${2:-}"
                if [[ -z "${suite}" ]]; then
                    echo "ERROR: --suite requires a value" >&2
                    return 2
                fi
                shift 2
                ;;
            --models)
                models_csv="${2:-}"
                if [[ -z "${models_csv}" ]]; then
                    echo "ERROR: --models requires a value" >&2
                    return 2
                fi
                shift 2
                ;;
            --net)
                net="${2:-}"
                if [[ -z "${net}" ]]; then
                    echo "ERROR: --net requires 1 or 0" >&2
                    return 2
                fi
                shift 2
                ;;
            --out)
                out="${2:-}"
                if [[ -z "${out}" ]]; then
                    echo "ERROR: --out requires a value" >&2
                    return 2
                fi
                shift 2
                ;;
            --scenario-ids)
                scenario_ids="${2:-}"
                if [[ -z "${scenario_ids}" ]]; then
                    echo "ERROR: --scenario-ids requires a value" >&2
                    return 2
                fi
                shift 2
                ;;
            --determinism)
                determinism="${2:-}"
                if [[ -z "${determinism}" ]]; then
                    echo "ERROR: --determinism requires a value" >&2
                    return 2
                fi
                shift 2
                ;;
            --repeats)
                repeats="${2:-}"
                if [[ -z "${repeats}" || ! "${repeats}" =~ ^[0-9]+$ ]]; then
                    echo "ERROR: --repeats requires an integer" >&2
                    return 2
                fi
                shift 2
                ;;
            --resume)
                resume_flag="true"
                shift
                ;;
            --calibrate-only)
                suite_mode="calibrate-only"
                resume_flag="false"
                shift
                ;;
            --errors-only)
                suite_mode="errors-only"
                resume_flag="false"
                shift
                ;;
            --run-only)
                suite_mode="run-only"
                resume_flag="true"
                shift
                ;;
            --)
                shift
                break
                ;;
            *)
                echo "Unknown arg: $1" >&2
                pack_usage >&2
                return 2
                ;;
        esac
    done

    if [[ -z "${out}" ]]; then
        local stamp
        stamp="$(date -u +%Y%m%d_%H%M%S)"
        out="./proof_pack_runs/${suite}_${stamp}"
    fi

    case "${net}" in
        0|1)
            :
            ;;
        *)
            echo "ERROR: --net requires 1 or 0" >&2
            return 2
            ;;
    esac

    PACK_SUITE="${suite}"
    PACK_NET="${net}"
    PACK_DETERMINISM="${determinism}"
    PACK_REPEATS="${repeats}"
    PACK_SUITE_MODE="${suite_mode}"
    RESUME_FLAG="${resume_flag}"
    OUTPUT_DIR="${out}"
    PACK_SCENARIO_IDS="${scenario_ids}"

    PACK_MODELS_CSV="${models_csv}"

    export PACK_SUITE PACK_NET PACK_DETERMINISM PACK_REPEATS PACK_SUITE_MODE RESUME_FLAG OUTPUT_DIR PACK_SCENARIO_IDS PACK_MODELS_CSV

    pack_apply_entrypoint_determinism
    pack_apply_suite "${PACK_SUITE}" || return 2

    if [[ -n "${PACK_MODELS_CSV}" ]]; then
        local -a models=()
        local raw
        IFS=',' read -r -a models <<< "${PACK_MODELS_CSV}"
        local -a cleaned=()
        for raw in "${models[@]}"; do
            raw="${raw#"${raw%%[![:space:]]*}"}"
            raw="${raw%"${raw##*[![:space:]]}"}"
            [[ -n "${raw}" ]] || continue
            cleaned+=("${raw}")
        done

        if [[ ${#cleaned[@]} -eq 0 ]]; then
            echo "ERROR: --models provided no valid model ids" >&2
            return 2
        fi
        if [[ ${#cleaned[@]} -gt 8 ]]; then
            echo "ERROR: --models supports up to 8 models (got ${#cleaned[@]})" >&2
            return 2
        fi

        # Override suite defaults. Leave unused MODEL_N blank to disable them.
        local i
        for i in {1..8}; do
            local var="MODEL_${i}"
            local idx=$((i - 1))
            if [[ ${idx} -lt ${#cleaned[@]} ]]; then
                printf -v "${var}" '%s' "${cleaned[${idx}]}"
            else
                printf -v "${var}" '%s' ""
            fi
            export "${var}"
        done
    fi

    pack_run_suite
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    pack_entrypoint "$@"
fi
