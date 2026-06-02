#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_tiny_hf_bnb_8bit.sh [options]

Compare a normal HF baseline against the same checkpoint loaded through
InvarLock's hf_bnb adapter with bitsandbytes 8-bit runtime loading.

Options:
  --baseline VALUE             Baseline model ID or local path.
                               Default: generated tiny local Llama checkpoint.
  --subject VALUE              Subject model ID or local path.
                               Default: same as baseline.
  --model-dir DIR              Generated local model directory when baseline and
                               subject are not provided.
                               Default: examples/integrations/hf_bnb/models/tiny-llama-bnb-baseline
  --tokenizer-source VALUE     Tokenizer ID or local path for generated model.
                               Default: sshleifer/tiny-gpt2
  --fixture-dir DIR            Generated local JSONL/preset directory.
                               Default: examples/integrations/hf_bnb/artifacts/tiny-hf-bnb-8bit
  --report-out DIR             Output directory for InvarLock artifacts.
                               Default: examples/integrations/hf_bnb/reports/tiny-hf-bnb-8bit
  --profile NAME               InvarLock profile. Default: ci
  --tier NAME                  InvarLock tier. Default: balanced
  --lane MODE                  Standard lane shortcut: host or cuda.
  --execution-mode MODE        container or host. Default: host
  --assurance MODE             strict or off. Default: off
  --runtime-provenance MODE    container or host for verify. Defaults to
                               execution mode.
  --device VALUE               Optional device override.
  --allow-network              Allow model downloads.
  --force                      Replace the generated local model directory.
  --no-html                    Skip HTML rendering in the compare wrapper.
  -h, --help                   Show this help.

The default path is host-mode so it can validate a local bitsandbytes runtime.
Use --lane host --device cpu for cpu-host-off, --lane host --device cuda for
cuda-host-off, and --lane cuda for cuda-container-strict evidence.
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

baseline=""
subject=""
model_dir="$SCRIPT_DIR/models/tiny-llama-bnb-baseline"
tokenizer_source="sshleifer/tiny-gpt2"
fixture_dir="$SCRIPT_DIR/artifacts/tiny-hf-bnb-8bit"
report_out="$SCRIPT_DIR/reports/tiny-hf-bnb-8bit"
profile="ci"
tier="balanced"
lane=""
execution_mode="host"
assurance="off"
runtime_provenance=""
device=""
allow_network=0
force=0
render_html=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)
      baseline="${2:-}"
      shift 2
      ;;
    --subject)
      subject="${2:-}"
      shift 2
      ;;
    --model-dir)
      model_dir="${2:-}"
      shift 2
      ;;
    --tokenizer-source)
      tokenizer_source="${2:-}"
      shift 2
      ;;
    --fixture-dir)
      fixture_dir="${2:-}"
      shift 2
      ;;
    --report-out)
      report_out="${2:-}"
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

if [[ -z "$fixture_dir" || -z "$report_out" ]]; then
  echo "Missing required baseline, subject, fixture, or report path." >&2
  usage >&2
  exit 2
fi

if [[ -z "$baseline" && -z "$subject" ]]; then
  baseline="$model_dir"
  subject="$model_dir"
elif [[ -z "$baseline" ]]; then
  baseline="$subject"
elif [[ -z "$subject" ]]; then
  subject="$baseline"
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

integration_log_header "bitsandbytes integration example"
integration_log_kv "lane" "$lane_artifact_label"
integration_log_kv "python" "$PYTHON_BIN"
integration_log_kv "device" "$effective_device"
integration_log_kv "report_out" "$report_out"

if [[ "$effective_execution_mode" == "host" ]]; then
  if ! "$PYTHON_BIN" -c 'import bitsandbytes' >/dev/null 2>&1; then
    cat >&2 <<'MSG'
Missing example dependency: bitsandbytes

Run this example in an environment with the Hugging Face and GPU extras, for example:
  uv run --extra hf --extra gpu examples/integrations/hf_bnb/run_tiny_hf_bnb_8bit.sh --allow-network

The core InvarLock install intentionally does not require bitsandbytes.
MSG
    exit 2
  fi
fi

integration_preflight_host_cuda_device "$PYTHON_BIN" "$effective_execution_mode" "$effective_device" "bitsandbytes" || exit $?

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

fixture_cmd=(
  "$PYTHON_BIN"
  "$SCRIPT_DIR/prepare_tiny_hf_bnb_fixture.py"
  --output-dir "$fixture_dir"
  --model-id "$baseline"
)
if [[ "$baseline" == "$model_dir" && "$subject" == "$model_dir" ]]; then
  fixture_cmd+=(
    --model-dir "$model_dir"
    --tokenizer-source "$tokenizer_source"
  )
fi
if [[ "$allow_network" -eq 1 ]]; then
  fixture_cmd+=(--allow-network)
fi
if [[ "$force" -eq 1 ]]; then
  fixture_cmd+=(--force)
fi

integration_log_step "prepare tiny HF checkpoint and local fixture"
fixture_preset="$("${fixture_cmd[@]}")"

integration_log_step "collect fixture metadata"
mkdir -p "$report_out"
cp "$fixture_dir/fixture_summary.json" "$report_out/fixture_summary.json"

compare_cmd=(
  "$REPO_ROOT/examples/integrations/_shared/run_invarlock_compare.sh"
  --baseline "$baseline"
  --subject "$subject"
  --baseline-adapter hf_causal
  --subject-adapter hf_bnb
  --profile "$profile"
  --tier "$tier"
  --preset "$fixture_preset"
  --report-out "$report_out"
  --execution-mode "$execution_mode"
  --assurance "$assurance"
  --edit-label hf_bnb_8bit_runtime_load
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
if [[ "$allow_network" -eq 1 ]]; then
  compare_cmd+=(--allow-network)
fi
if [[ "$render_html" -eq 0 ]]; then
  compare_cmd+=(--no-html)
fi

integration_log_step "run InvarLock comparison"
"${compare_cmd[@]}"
