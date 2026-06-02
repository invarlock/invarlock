#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_tiny_torchao_int8_export.sh [options]

Materialize a tiny torchao int8 weight-only export subject checkpoint, then
compare it against the generated baseline with InvarLock's shared integration
wrapper.

Options:
  --baseline-dir DIR           Output directory for the generated baseline.
                               Default: examples/integrations/torchao_int8_export/models/tiny-llama-baseline
  --subject-dir DIR            Output directory for the exported subject.
                               Default: examples/integrations/torchao_int8_export/models/tiny-llama-torchao-int8-export
  --fixture-dir DIR            Generated local JSONL/preset directory.
                               Default: examples/integrations/torchao_int8_export/artifacts/tiny-torchao-int8-export-fixture
  --report-out DIR             Output directory for InvarLock artifacts.
                               Default: examples/integrations/torchao_int8_export/reports/tiny-torchao-int8-export
  --tokenizer-source VALUE     Tokenizer ID or local path. Default: sshleifer/tiny-gpt2
  --profile NAME               InvarLock profile. Default: release
  --tier NAME                  InvarLock tier. Default: balanced
  --execution-mode MODE        container or host. Default: container
  --assurance MODE             strict or off. Default: strict
  --allow-network              Allow tokenizer and dataset downloads.
  --force                      Replace existing generated model directories.
  --materialize-only           Stop after writing checkpoints.
  --no-html                    Skip HTML rendering in the compare wrapper.
  -h, --help                   Show this help.

The default compare path is strict/container-backed. For local host bring-up,
pass --execution-mode host --assurance off.
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

baseline_dir="$SCRIPT_DIR/models/tiny-llama-baseline"
subject_dir="$SCRIPT_DIR/models/tiny-llama-torchao-int8-export"
fixture_dir="$SCRIPT_DIR/artifacts/tiny-torchao-int8-export-fixture"
report_out="$SCRIPT_DIR/reports/tiny-torchao-int8-export"
tokenizer_source="sshleifer/tiny-gpt2"
profile="release"
tier="balanced"
execution_mode="container"
assurance="strict"
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
    --execution-mode)
      execution_mode="${2:-}"
      shift 2
      ;;
    --assurance)
      assurance="${2:-}"
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

if ! "$PYTHON_BIN" -c 'import torchao' >/dev/null 2>&1; then
  cat >&2 <<'MSG'
Missing example dependency: torchao

Install torchao in the environment used for this example, for example:
  python -m pip install torchao

The core InvarLock install intentionally does not require torchao.
MSG
  exit 2
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

materialize_cmd=(
  "$PYTHON_BIN"
  "$SCRIPT_DIR/materialize_tiny_torchao_int8_subject.py"
  --baseline-dir "$baseline_dir"
  --subject-dir "$subject_dir"
  --fixture-dir "$fixture_dir"
  --tokenizer-source "$tokenizer_source"
)
if [[ "$allow_network" -eq 1 ]]; then
  materialize_cmd+=(--allow-network)
fi
if [[ "$force" -eq 1 ]]; then
  materialize_cmd+=(--force)
fi

"${materialize_cmd[@]}"

if [[ "$materialize_only" -eq 1 ]]; then
  echo "Wrote baseline checkpoint: $baseline_dir"
  echo "Wrote subject checkpoint: $subject_dir"
  exit 0
fi

mkdir -p "$report_out"
cp "$subject_dir/checkpoint_refs.json" "$report_out/checkpoint_refs.json"
cp "$subject_dir/external_edit_summary.json" "$report_out/external_edit_summary.json"
cp "$fixture_dir/fixture_summary.json" "$report_out/fixture_summary.json"

compare_cmd=(
  "$REPO_ROOT/examples/integrations/_shared/run_invarlock_compare.sh"
  --baseline "$baseline_dir"
  --subject "$subject_dir"
  --baseline-adapter hf_causal
  --subject-adapter hf_causal
  --profile "$profile"
  --tier "$tier"
  --preset "$fixture_dir/preset.yaml"
  --report-out "$report_out"
  --execution-mode "$execution_mode"
  --assurance "$assurance"
  --edit-label torchao_int8_weight_only_export
)

if [[ "$allow_network" -eq 1 ]]; then
  compare_cmd+=(--allow-network)
fi
if [[ "$render_html" -eq 0 ]]; then
  compare_cmd+=(--no-html)
fi

"${compare_cmd[@]}"
