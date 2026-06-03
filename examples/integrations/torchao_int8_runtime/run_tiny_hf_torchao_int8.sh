#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_tiny_hf_torchao_int8.sh [options]

Prepare a tiny HF causal checkpoint, then compare the normal HF baseline
against the same checkpoint loaded through InvarLock's hf_torchao adapter.

Options:
  --model-dir DIR              Output directory for the generated HF checkpoint.
                               Default: examples/integrations/torchao_int8_runtime/models/tiny-llama-hf-torchao-baseline
  --fixture-dir DIR            Generated local JSONL/preset directory.
                               Default: examples/integrations/torchao_int8_runtime/artifacts/tiny-hf-torchao-int8
  --report-out DIR             Output directory for InvarLock artifacts.
                               Default: examples/integrations/torchao_int8_runtime/reports/tiny-hf-torchao-int8/<artifact-lane>
  --tokenizer-source VALUE     Tokenizer ID or local path. Default: sshleifer/tiny-gpt2
  --profile NAME               InvarLock profile. Default: release
  --tier NAME                  InvarLock tier. Default: balanced
  --lane MODE                  Standard lane shortcut: host or cuda.
  --execution-mode MODE        container or host. Default: container
  --assurance MODE             strict or off. Default: strict
  --runtime-provenance MODE    container or host for verify. Defaults to
                               execution mode.
  --device VALUE               Optional device override.
  --allow-network              Allow tokenizer and dataset downloads.
  --force                      Replace existing generated checkpoint directory.
  --prepare-only               Stop after writing the checkpoint and fixture.
  --no-html                    Skip HTML rendering in the compare wrapper.
  -h, --help                   Show this help.

The default compare path is strict/container-backed. Use --lane host --device
cpu for cpu-host-off, --lane host --device cuda for cuda-host-off, and
--lane cuda for cuda-container-strict evidence.
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

model_dir="$SCRIPT_DIR/models/tiny-llama-hf-torchao-baseline"
fixture_dir="$SCRIPT_DIR/artifacts/tiny-hf-torchao-int8"
report_out="$SCRIPT_DIR/reports/tiny-hf-torchao-int8"
report_out_was_default=1
tokenizer_source="sshleifer/tiny-gpt2"
profile="release"
tier="balanced"
lane=""
execution_mode="container"
assurance="strict"
runtime_provenance=""
device=""
allow_network=0
force=0
prepare_only=0
render_html=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)
      model_dir="${2:-}"
      shift 2
      ;;
    --fixture-dir)
      fixture_dir="${2:-}"
      shift 2
      ;;
    --report-out)
      report_out="${2:-}"
      report_out_was_default=0
      shift 2
      ;;
    --tokenizer-source)
      tokenizer_source="${2:-}"
      shift 2
      ;;
    --profile)
      profile="${2:-}"
      shift 2
      ;;
    --tier)
      tier="${2:-}"
      shift 2
      ;;
    --lane)
      lane="${2:-}"
      shift 2
      ;;
    --execution-mode)
      execution_mode="${2:-}"
      shift 2
      ;;
    --assurance)
      assurance="${2:-}"
      shift 2
      ;;
    --runtime-provenance)
      runtime_provenance="${2:-}"
      shift 2
      ;;
    --device)
      device="${2:-}"
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
    --prepare-only)
      prepare_only=1
      shift
      ;;
    --no-html)
      render_html=0
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

if [[ -z "$model_dir" || -z "$fixture_dir" || -z "$report_out" ]]; then
  echo "Missing required model, fixture, or report path." >&2
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

# shellcheck source=../_shared/preflight.sh
source "$REPO_ROOT/examples/integrations/_shared/preflight.sh"
effective_execution_mode="$(integration_effective_execution_mode "$lane" "$execution_mode")"
effective_assurance="$(integration_effective_assurance "$lane" "$assurance")"
device="$(integration_default_host_device "$effective_execution_mode" "$device")"
effective_device="$(integration_effective_device "$lane" "$device")"
lane_artifact_label="$(integration_lane_artifact_label "$effective_execution_mode" "$effective_assurance" "$effective_device")"
report_out="$(integration_lane_report_out "$report_out" "$report_out_was_default" "$lane_artifact_label")"

integration_log_header "TorchAO integration example"
integration_log_kv "lane" "$lane_artifact_label"
integration_log_kv "python" "$PYTHON_BIN"
integration_log_kv "device" "$effective_device"
integration_log_kv "report_out" "$report_out"

if ! "$PYTHON_BIN" -c 'import torchao' >/dev/null 2>&1; then
  cat >&2 <<'MSG'
Missing example dependency: torchao

Install torchao in the environment used for this example, for example:
  python -m pip install torchao

The core InvarLock install intentionally does not require torchao.
MSG
  exit 2
fi

integration_preflight_host_cuda_device "$PYTHON_BIN" "$effective_execution_mode" "$effective_device" "TorchAO" || exit $?

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

prepare_cmd=(
  "$PYTHON_BIN"
  "$SCRIPT_DIR/prepare_tiny_hf_torchao_fixture.py"
  --model-dir "$model_dir"
  --fixture-dir "$fixture_dir"
  --tokenizer-source "$tokenizer_source"
)
if [[ "$allow_network" -eq 1 ]]; then
  prepare_cmd+=(--allow-network)
fi
if [[ "$force" -eq 1 ]]; then
  prepare_cmd+=(--force)
fi

integration_log_step "prepare tiny HF checkpoint and TorchAO fixture"
"${prepare_cmd[@]}"

if [[ "$prepare_only" -eq 1 ]]; then
  echo "Wrote model checkpoint: $model_dir"
  echo "Wrote fixture: $fixture_dir"
  exit 0
fi

integration_log_step "collect checkpoint, adapter, and fixture metadata"
mkdir -p "$report_out"
cp "$model_dir/checkpoint_refs.json" "$report_out/checkpoint_refs.json"
cp "$model_dir/adapter_runtime_summary.json" "$report_out/adapter_runtime_summary.json"
cp "$fixture_dir/fixture_summary.json" "$report_out/fixture_summary.json"

compare_cmd=(
  "$REPO_ROOT/examples/integrations/_shared/run_invarlock_compare.sh"
  --baseline "$model_dir"
  --subject "$model_dir"
  --baseline-adapter hf_causal
  --subject-adapter hf_torchao
  --profile "$profile"
  --tier "$tier"
  --preset "$fixture_dir/preset.yaml"
  --report-out "$report_out"
  --execution-mode "$execution_mode"
  --assurance "$assurance"
  --edit-label torchao_int8_runtime_quantization
)

if [[ -n "$lane" ]]; then
  compare_cmd+=(--lane "$lane")
fi
if [[ -n "$runtime_provenance" ]]; then
  compare_cmd+=(--runtime-provenance "$runtime_provenance")
fi
if [[ -n "$device" ]]; then
  compare_cmd+=(--device "$device")
fi
if [[ "$lane_artifact_label" == "cuda-container-strict" ]]; then
  compare_cmd+=(--require-backend-inventory)
fi
if [[ "$allow_network" -eq 1 ]]; then
  compare_cmd+=(--allow-network)
fi
if [[ "$render_html" -eq 0 ]]; then
  compare_cmd+=(--no-html)
fi

integration_log_step "run InvarLock comparison"
"${compare_cmd[@]}"
