#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_tiny_peft_lora.sh [options]

Materialize a tiny PEFT LoRA-merged subject checkpoint, then compare it against
the baseline with InvarLock's shared integration wrapper.

Options:
  --baseline VALUE             Baseline model ID or local path.
                               Default: sshleifer/tiny-gpt2
  --subject-dir DIR            Output directory for the merged subject.
                               Default: examples/integrations/peft_lora/models/tiny-gpt2-peft-lora-merged
  --fixture-dir DIR            Generated local JSONL/preset directory.
                               Default: examples/integrations/peft_lora/artifacts/tiny-peft-lora-fixture
  --report-out DIR             Output directory for InvarLock artifacts.
                               Default: examples/integrations/peft_lora/reports/tiny-peft-lora/<artifact-lane>
  --profile NAME               InvarLock profile. Default: release
  --tier NAME                  InvarLock tier. Default: balanced
  --lane MODE                  Standard lane shortcut: host or cuda.
  --execution-mode MODE        container or host. Default: container
  --assurance MODE             strict or off. Default: strict
  --runtime-provenance MODE    container or host for verify. Defaults to
                               execution mode.
  --device VALUE               Optional device override.
  --allow-network              Allow model/dataset downloads.
  --force                      Replace an existing subject directory.
  --materialize-only           Stop after writing the merged subject checkpoint.
  --no-html                    Skip HTML rendering in the compare wrapper.
  -h, --help                   Show this help.

The default compare path is strict/container-backed. Use --lane host --device
cpu for cpu-host-off, --lane host --device cuda for cuda-host-off, and
--lane cuda for cuda-container-strict evidence.
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

baseline="sshleifer/tiny-gpt2"
subject_dir="$SCRIPT_DIR/models/tiny-gpt2-peft-lora-merged"
fixture_dir="$SCRIPT_DIR/artifacts/tiny-peft-lora-fixture"
report_out="$SCRIPT_DIR/reports/tiny-peft-lora"
report_out_was_default=1
profile="release"
tier="balanced"
lane=""
execution_mode="container"
assurance="strict"
runtime_provenance=""
device=""
allow_network=0
force=0
materialize_only=0
render_html=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)
      baseline="${2:-}"
      shift 2
      ;;
    --subject-dir)
      subject_dir="${2:-}"
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
    --materialize-only)
      materialize_only=1
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

if [[ -z "$baseline" || -z "$subject_dir" || -z "$fixture_dir" || -z "$report_out" ]]; then
  echo "Missing required baseline, subject, fixture, or report path." >&2
  usage >&2
  exit 2
fi

select_python_bin() {
  local required_module="$1"
  local candidate
  for candidate in python "$REPO_ROOT/.venv/bin/python" python3; do
    if [[ "$candidate" == */* ]]; then
      [[ -x "$candidate" ]] || continue
    elif ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    if "$candidate" -c "import ${required_module}" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      return
    fi
  done
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
}

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  select_python_bin peft
fi

# shellcheck source=../_shared/preflight.sh
source "$REPO_ROOT/examples/integrations/_shared/preflight.sh"
effective_execution_mode="$(integration_effective_execution_mode "$lane" "$execution_mode")"
effective_assurance="$(integration_effective_assurance "$lane" "$assurance")"
device="$(integration_default_host_device "$effective_execution_mode" "$device")"
effective_device="$(integration_effective_device "$lane" "$device")"
lane_artifact_label="$(integration_lane_artifact_label "$effective_execution_mode" "$effective_assurance" "$effective_device")"
report_out="$(integration_lane_report_out "$report_out" "$report_out_was_default" "$lane_artifact_label")"

integration_log_header "PEFT LoRA integration example"
integration_log_kv "lane" "$lane_artifact_label"
integration_log_kv "python" "$PYTHON_BIN"
integration_log_kv "device" "$effective_device"
integration_log_kv "report_out" "$report_out"

if ! "$PYTHON_BIN" -c 'import peft' >/dev/null 2>&1; then
  cat >&2 <<'MSG'
Missing example dependency: peft

Install PEFT in the environment used for this example, for example:
  python -m pip install peft
  uv pip install --python .venv/bin/python peft

The core InvarLock install intentionally does not require PEFT.
MSG
  exit 2
fi

integration_preflight_host_cuda_device "$PYTHON_BIN" "$effective_execution_mode" "$effective_device" "PEFT LoRA" || exit $?

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

materialize_cmd=(
  "$PYTHON_BIN"
  "$SCRIPT_DIR/materialize_tiny_peft_lora_subject.py"
  --baseline "$baseline"
  --output-dir "$subject_dir"
  --fixture-dir "$fixture_dir"
)
if [[ "$allow_network" -eq 1 ]]; then
  materialize_cmd+=(--allow-network)
fi
if [[ "$force" -eq 1 ]]; then
  materialize_cmd+=(--force)
fi

integration_log_step "materialize tiny LoRA subject and local fixture"
"${materialize_cmd[@]}"

if [[ "$materialize_only" -eq 1 ]]; then
  echo "Wrote subject checkpoint: $subject_dir"
  exit 0
fi

integration_log_step "collect checkpoint, edit, and fixture metadata"
mkdir -p "$report_out"
cp "$subject_dir/checkpoint_refs.json" "$report_out/checkpoint_refs.json"
cp "$subject_dir/external_edit_summary.json" "$report_out/external_edit_summary.json"
cp "$fixture_dir/fixture_summary.json" "$report_out/fixture_summary.json"

compare_cmd=(
  "$REPO_ROOT/examples/integrations/_shared/run_invarlock_compare.sh"
  --baseline "$baseline"
  --subject "$subject_dir"
  --baseline-adapter hf_causal
  --subject-adapter hf_causal
  --profile "$profile"
  --tier "$tier"
  --preset "$fixture_dir/preset.yaml"
  --report-out "$report_out"
  --execution-mode "$execution_mode"
  --assurance "$assurance"
  --edit-label lora_merge
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
