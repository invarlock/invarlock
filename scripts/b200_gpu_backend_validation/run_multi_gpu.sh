#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat << 'EOF'
run_multi_gpu.sh

Run validate_svd_backend_equivalence.py across multiple GPUs (one process per model).

Usage:
  bash scripts/b200_gpu_backend_validation/run_multi_gpu.sh --out-dir <DIR> --model <MODEL> [--model <MODEL> ...] [options]

Options:
  --out-dir <DIR>            Output directory for per-model JSON+log files (required)
  --model <MODEL>            Model id/path (repeatable) (required)
  --python <BIN>             Python interpreter (default: python3)
  --strict-determinism       Enable deterministic flags in each worker
  --spectral                 Also run spectral sigma_max comparisons on module weights
  --module-regex <REGEX>     Module selection regex (passed through)
  --max-modules <N>          Number of modules to hook/measure (passed through)
  --seq-len <N>              Tokenized sequence length (passed through)
  --rmt-margin <F>           Margin multiplier (passed through)
  --rmt-deadband <F>         Deadband (passed through)
  --repeats <N>              Determinism repeats (passed through)

GPU selection:
  - If CUDA_VISIBLE_DEVICES is set, uses that ordered list of GPU IDs.
  - Otherwise, uses nvidia-smi to infer GPU count and assigns 0..N-1.

Notes:
  - Each worker sets CUDA_VISIBLE_DEVICES to a single GPU ID to avoid sharding
    across GPUs with device_map="auto".
EOF
}

OUT_DIR=""
PYTHON_BIN="python3"
STRICT="false"
SPECTRAL="false"
MODULE_REGEX=""
MAX_MODULES=""
SEQ_LEN=""
RMT_MARGIN=""
RMT_DEADBAND=""
REPEATS=""
MODELS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --model)
      MODELS+=("${2:-}")
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-python3}"
      shift 2
      ;;
    --strict-determinism)
      STRICT="true"
      shift
      ;;
    --spectral)
      SPECTRAL="true"
      shift
      ;;
    --module-regex)
      MODULE_REGEX="${2:-}"
      shift 2
      ;;
    --max-modules)
      MAX_MODULES="${2:-}"
      shift 2
      ;;
    --seq-len)
      SEQ_LEN="${2:-}"
      shift 2
      ;;
    --rmt-margin)
      RMT_MARGIN="${2:-}"
      shift 2
      ;;
    --rmt-deadband)
      RMT_DEADBAND="${2:-}"
      shift 2
      ;;
    --repeats)
      REPEATS="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${OUT_DIR}" ]]; then
  echo "ERROR: --out-dir is required" >&2
  exit 2
fi
if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "ERROR: at least one --model is required" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

GPUS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES}"
else
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is unset and nvidia-smi is not available" >&2
    exit 2
  fi
  gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
  if ! [[ "${gpu_count}" =~ ^[0-9]+$ ]] || [[ "${gpu_count}" -le 0 ]]; then
    echo "ERROR: failed to infer GPU count from nvidia-smi" >&2
    exit 2
  fi
  for ((i=0; i<gpu_count; i++)); do
    GPUS+=("${i}")
  done
fi

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "ERROR: no GPUs detected/selected" >&2
  exit 2
fi

PIDS=()
FAIL=0

for idx in "${!MODELS[@]}"; do
  model="${MODELS[$idx]}"
  gpu="${GPUS[$(( idx % ${#GPUS[@]} ))]}"

  safe="$(echo "${model}" | tr '/: ' '___' | tr -cd '[:alnum:]_.-')"
  [[ -z "${safe}" ]] && safe="model_${idx}"

  out_json="${OUT_DIR}/${safe}.json"
  out_log="${OUT_DIR}/${safe}.log"

  cmd=( "${PYTHON_BIN}" "scripts/b200_gpu_backend_validation/validate_svd_backend_equivalence.py"
        "--model" "${model}"
        "--out" "${out_json}" )

  if [[ "${STRICT}" == "true" ]]; then
    cmd+=( "--strict-determinism" )
  fi
  if [[ "${SPECTRAL}" == "true" ]]; then
    cmd+=( "--spectral" )
  fi
  if [[ -n "${MODULE_REGEX}" ]]; then
    cmd+=( "--module-regex" "${MODULE_REGEX}" )
  fi
  if [[ -n "${MAX_MODULES}" ]]; then
    cmd+=( "--max-modules" "${MAX_MODULES}" )
  fi
  if [[ -n "${SEQ_LEN}" ]]; then
    cmd+=( "--seq-len" "${SEQ_LEN}" )
  fi
  if [[ -n "${RMT_MARGIN}" ]]; then
    cmd+=( "--rmt-margin" "${RMT_MARGIN}" )
  fi
  if [[ -n "${RMT_DEADBAND}" ]]; then
    cmd+=( "--rmt-deadband" "${RMT_DEADBAND}" )
  fi
  if [[ -n "${REPEATS}" ]]; then
    cmd+=( "--repeats" "${REPEATS}" )
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu}: ${model}"
  echo "  -> ${out_json}"

  ( CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${out_log}" 2>&1 ) &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

exit "${FAIL}"

