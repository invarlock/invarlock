#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_tiny_optimum_onnx_probe.sh [options]

Export a tiny Hugging Face model with Optimum ONNX and record compatibility
details for pairing the deployment artifact beside InvarLock evidence.

Options:
  --baseline VALUE             Baseline model ID or local path.
                               Default: sshleifer/tiny-gpt2
  --export-dir DIR             Output directory for the ONNX export.
                               Default: examples/integrations/optimum_onnx_export/models/tiny-gpt2-optimum-onnx
  --report-out DIR             Output directory for compatibility artifacts.
                               Default: examples/integrations/optimum_onnx_export/reports/tiny-optimum-onnx
  --task VALUE                 Optimum export task. Default: text-generation
  --device VALUE               Export device. Default: cpu
  --batch-size VALUE           Export batch size. Default: 1
  --sequence-length VALUE      Export sequence length. Default: 8
  --allow-network              Allow model downloads.
  --force                      Replace existing export/report directories.
  -h, --help                   Show this help.

This is a compatibility probe, not a shared InvarLock compare run.
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

baseline="sshleifer/tiny-gpt2"
export_dir="$SCRIPT_DIR/models/tiny-gpt2-optimum-onnx"
report_out="$SCRIPT_DIR/reports/tiny-optimum-onnx"
task="text-generation"
device="cpu"
batch_size="1"
sequence_length="8"
allow_network=0
force=0
original_args=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)
      baseline="${2:-}"
      shift 2
      ;;
    --export-dir)
      export_dir="${2:-}"
      shift 2
      ;;
    --report-out)
      report_out="${2:-}"
      shift 2
      ;;
    --task)
      task="${2:-}"
      shift 2
      ;;
    --device)
      device="${2:-}"
      shift 2
      ;;
    --batch-size)
      batch_size="${2:-}"
      shift 2
      ;;
    --sequence-length)
      sequence_length="${2:-}"
      shift 2
      ;;
    --allow-network)
      allow_network=1
      shift
      ;;
    --force)
      force=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$baseline" || -z "$export_dir" || -z "$report_out" || -z "$task" || -z "$device" || -z "$batch_size" || -z "$sequence_length" ]]; then
  echo "Missing required model, output, or export setting." >&2
  usage >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if ! "$PYTHON_BIN" -c 'import optimum.commands.optimum_cli; import onnxruntime' >/dev/null 2>&1; then
  cat >&2 <<'MSG'
Missing example dependency: optimum-onnx[onnxruntime]

Install Optimum ONNX Runtime dependencies in the environment used for this example:
  python -m pip install "optimum-onnx[onnxruntime]"

The core InvarLock install intentionally does not require Optimum or ONNX Runtime.
MSG
  exit 2
fi

for path in "$export_dir" "$report_out"; do
  if [[ -e "$path" ]]; then
    if [[ "$force" -ne 1 ]]; then
      echo "Output path already exists: $path" >&2
      echo "Pass --force to replace existing generated outputs." >&2
      exit 2
    fi
    rm -rf "$path"
  fi
done
mkdir -p "$report_out"

if [[ "$allow_network" -eq 0 ]]; then
  export HF_HUB_OFFLINE=1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

append_command_log() {
  local label="$1"
  shift

  {
    printf '%s:\n' "$label"
    printf '  '
    printf '%q ' "$@"
    printf '\n\n'
  } >> "$report_out/run_command.txt"
}

{
  printf 'runner_invocation:\n  '
  printf '%q ' "$0" "${original_args[@]}"
  printf '\n\n'
} > "$report_out/run_command.txt"

export_cmd=(
  "$PYTHON_BIN"
  -m optimum.commands.optimum_cli
  export
  onnx
  --model "$baseline"
  --task "$task"
  --device "$device"
  --batch_size "$batch_size"
  --sequence_length "$sequence_length"
  "$export_dir"
)

append_command_log "optimum_export_onnx" "${export_cmd[@]}"
"${export_cmd[@]}"

inspect_cmd=(
  "$PYTHON_BIN"
  "$SCRIPT_DIR/inspect_optimum_onnx_export.py"
  --export-dir "$export_dir"
  --baseline-model "$baseline"
  --output "$report_out/compatibility_probe.json"
)

append_command_log "inspect_optimum_onnx_export" "${inspect_cmd[@]}"
"${inspect_cmd[@]}"
