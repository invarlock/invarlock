#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_tiny_awq.sh [options]

Materialize a tiny GPTQModel AWQ subject checkpoint, then compare it against
the generated baseline with InvarLock's shared integration wrapper.

Options:
  --baseline-dir DIR           Output directory for the generated baseline.
                               Default: examples/integrations/awq/models/tiny-llama-baseline
  --subject-dir DIR            Output directory for the quantized subject.
                               Default: examples/integrations/awq/models/tiny-llama-awq-4bit
  --fixture-dir DIR            Generated local JSONL/preset directory.
                               Default: examples/integrations/awq/artifacts/tiny-awq-fixture
  --report-out DIR             Output directory for InvarLock artifacts.
                               Default: examples/integrations/awq/reports/tiny-awq/<artifact-lane>
  --tokenizer-source VALUE     Tokenizer ID or local path. Default: sshleifer/tiny-gpt2
  --quantize-device VALUE      CUDA device for AWQ materialization. Default: cuda:0
  --awq-backend VALUE          GPTQModel/Transformers AWQ backend. Default: torch_awq
  --profile NAME               InvarLock profile. Default: ci
  --tier NAME                  InvarLock tier. Default: balanced
  --lane MODE                  Standard lane shortcut: host or cuda.
  --execution-mode MODE        container or host. Default: host
  --assurance MODE             strict or off. Default: off
  --runtime-provenance MODE    container or host for verify. Defaults to
                               execution mode.
  --device VALUE               InvarLock device. AWQ supports cuda only here.
                               Default: cuda
  --allow-network              Allow tokenizer downloads.
  --force                      Replace existing generated model and fixture dirs.
  --materialize-only           Stop after writing checkpoints and fixture.
  --no-html                    Skip HTML rendering in the compare wrapper.
  -h, --help                   Show this help.

The default path is host-mode on a CUDA host so it can validate the local
GPTQModel AWQ runtime. Use --lane host --device cuda for cuda-host-off and
--lane cuda for cuda-container-strict evidence. AWQ materialization and
evaluation use CUDA in this example.
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

baseline_dir="$SCRIPT_DIR/models/tiny-llama-baseline"
subject_dir="$SCRIPT_DIR/models/tiny-llama-awq-4bit"
fixture_dir="$SCRIPT_DIR/artifacts/tiny-awq-fixture"
report_out="$SCRIPT_DIR/reports/tiny-awq"
report_out_was_default=1
tokenizer_source="sshleifer/tiny-gpt2"
quantize_device="cuda:0"
awq_backend="torch_awq"
profile="ci"
tier="balanced"
lane=""
execution_mode="host"
assurance="off"
runtime_provenance=""
device="cuda"
allow_network=0
force=0
materialize_only=0
render_html=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline-dir)
      baseline_dir="${2:-}"
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
    --tokenizer-source)
      tokenizer_source="${2:-}"
      shift 2
      ;;
    --quantize-device)
      quantize_device="${2:-}"
      shift 2
      ;;
    --awq-backend)
      awq_backend="${2:-}"
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

if [[ -z "$baseline_dir" || -z "$subject_dir" || -z "$fixture_dir" || -z "$report_out" ]]; then
  echo "Missing required baseline, subject, fixture, or report path." >&2
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
python_bin_dir="$(cd -- "$(dirname -- "$PYTHON_BIN")" && pwd)"
if [[ -x "$python_bin_dir/ninja" ]]; then
  export PATH="$python_bin_dir:$PATH"
fi

# shellcheck source=../_shared/preflight.sh
source "$REPO_ROOT/examples/integrations/_shared/preflight.sh"
effective_execution_mode="$(integration_effective_execution_mode "$lane" "$execution_mode")"
effective_assurance="$(integration_effective_assurance "$lane" "$assurance")"
integration_require_strict_acceptance_inputs \
  "$effective_assurance" \
  "${INVARLOCK_EXPECTED_RUNTIME_IMAGE_DIGEST:-}" \
  "${INVARLOCK_ACCEPTANCE_BASELINE_REPORT:-}" \
  "${INVARLOCK_ACCEPTANCE_POLICY_PACK:-}" || exit $?
device="$(integration_default_host_device "$effective_execution_mode" "$device")"
effective_device="$(integration_effective_device "$lane" "$device")"
lane_artifact_label="$(integration_lane_artifact_label "$effective_execution_mode" "$effective_assurance" "$effective_device")"
report_out="$(integration_lane_report_out "$report_out" "$report_out_was_default" "$lane_artifact_label")"

integration_log_header "AWQ integration example"
integration_log_kv "lane" "$lane_artifact_label"
integration_log_kv "python" "$PYTHON_BIN"
integration_log_kv "quantize_device" "$quantize_device"
integration_log_kv "report_out" "$report_out"

if [[ "$effective_device" != cuda* ]]; then
  cat >&2 <<'MSG'
AWQ lanes in this example are CUDA-only.

Use --lane host --device cuda for cuda-host-off, or use --lane cuda with the
documented runtime image for cuda-container-strict evidence.
MSG
  exit 2
fi

if [[ "$quantize_device" != cuda* ]]; then
  cat >&2 <<'MSG'
AWQ materialization requires a CUDA quantization device.

Use --quantize-device cuda:0 or another CUDA device visible to torch.
MSG
  exit 2
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if ! "$PYTHON_BIN" -c 'from invarlock.gptqmodel_runtime import require_gptqmodel_runtime; require_gptqmodel_runtime()' >/dev/null 2>&1; then
  cat >&2 <<'MSG'
Missing example dependency: GPTQModel

Run this example in an environment with the AWQ optional stack on a CUDA host,
for example:
  uv run --extra awq examples/integrations/awq/run_tiny_awq.sh --allow-network --force

The core InvarLock install intentionally does not require GPTQModel.
MSG
  exit 2
fi

integration_preflight_host_cuda_device "$PYTHON_BIN" "$effective_execution_mode" "$effective_device" "AWQ" || exit $?
integration_preflight_gptqmodel_host_runtime \
  "$PYTHON_BIN" "$effective_execution_mode" "$effective_device" || exit $?

if ! "$PYTHON_BIN" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
  cat >&2 <<'MSG'
AWQ materialization requires CUDA.

Run this example on a CUDA host, or use a pre-materialized AWQ subject with the
shared compare wrapper.
MSG
  exit 2
fi

materialize_cmd=(
  "$PYTHON_BIN"
  "$SCRIPT_DIR/materialize_tiny_awq_subject.py"
  --baseline-dir "$baseline_dir"
  --subject-dir "$subject_dir"
  --fixture-dir "$fixture_dir"
  --tokenizer-source "$tokenizer_source"
  --quantize-device "$quantize_device"
  --awq-backend "$awq_backend"
)
if [[ "$allow_network" -eq 1 ]]; then
  materialize_cmd+=(--allow-network)
fi
if [[ "$force" -eq 1 ]]; then
  materialize_cmd+=(--force)
fi

integration_log_step "materialize tiny AWQ subject and local fixture"
integration_run_source_archive_clean "${materialize_cmd[@]}"

if [[ "$materialize_only" -eq 1 ]]; then
  echo "Wrote baseline checkpoint: $baseline_dir"
  echo "Wrote subject checkpoint: $subject_dir"
  echo "Wrote fixture: $fixture_dir"
  exit 0
fi

integration_log_step "collect checkpoint, edit, and fixture metadata"
mkdir -p "$report_out"
cp "$subject_dir/checkpoint_refs.json" "$report_out/checkpoint_refs.json"
cp "$subject_dir/external_edit_summary.json" "$report_out/external_edit_summary.json"
cp "$fixture_dir/fixture_summary.json" "$report_out/fixture_summary.json"

compare_cmd=(
  "$REPO_ROOT/examples/integrations/_shared/run_invarlock_compare.sh"
  --baseline "$baseline_dir"
  --subject "$subject_dir"
  --baseline-adapter hf_causal
  --subject-adapter hf_awq
  --profile "$profile"
  --tier "$tier"
  --preset "$fixture_dir/preset.yaml"
  --report-out "$report_out"
  --execution-mode "$execution_mode"
  --assurance "$assurance"
  --edit-label gptqmodel_awq_4bit
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
  compare_cmd+=(
    --require-backend-inventory
    --require-runtime-quantization-proof
  )
fi
if [[ "$allow_network" -eq 1 ]]; then
  compare_cmd+=(--allow-network)
fi
if [[ "$render_html" -eq 0 ]]; then
  compare_cmd+=(--no-html)
fi

integration_log_step "run InvarLock comparison"
"${compare_cmd[@]}"
