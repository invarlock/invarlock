#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_public_e2e_release_review.sh [options]

Build a local review bundle from the checked-in external-edit public evidence.

Options:
  --report PATH                Source evaluation report.
                               Default: public_evidence/real_runs/tiny_gpt2_external_magnitude_prune/evidence_pack/reports/report-001/evaluation.report.json
  --output-dir DIR             Output directory for generated handoff artifacts.
                               Default: examples/integrations/public_e2e/reports/tiny-gpt2-external-magnitude-prune
  --profile NAME               InvarLock verification profile. Default: release
  --assurance MODE             Verification assurance mode. Default: strict
  --runtime-provenance MODE    Runtime provenance expectation. Default: container
  --report-url URL             Optional public URL for the model-card block.
  --evidence-url URL           Optional evidence-pack URL for the model-card block.
  --force                      Accepted for scripted parity; outputs are overwritten.
  -h, --help                   Show this help.

Set PYTHON_BIN to choose the Python interpreter. The script also adds
<repo>/src to PYTHONPATH so it can run from a source checkout with dependencies
installed in the selected environment.
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

report="$REPO_ROOT/public_evidence/real_runs/tiny_gpt2_external_magnitude_prune/evidence_pack/reports/report-001/evaluation.report.json"
output_dir="$SCRIPT_DIR/reports/tiny-gpt2-external-magnitude-prune"
profile="release"
assurance="strict"
runtime_provenance="container"
report_url=""
evidence_url=""
original_args=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report)
      report="${2:-}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:-}"
      shift 2
      ;;
    --profile)
      profile="${2:-}"
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
    --report-url)
      report_url="${2:-}"
      shift 2
      ;;
    --evidence-url)
      evidence_url="${2:-}"
      shift 2
      ;;
    --force)
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

if [[ -z "$report" || -z "$output_dir" || -z "$profile" || -z "$assurance" ]]; then
  echo "Missing required report, output directory, profile, or assurance mode." >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$report" ]]; then
  echo "Evaluation report not found: $report" >&2
  exit 2
fi

select_python_bin() {
  local candidate
  for candidate in "$REPO_ROOT/.venv/bin/python" python3 python; do
    if [[ "$candidate" == */* ]]; then
      [[ -x "$candidate" ]] || continue
    elif ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    PYTHON_BIN="$candidate"
    return
  done
  echo "No Python interpreter found. Set PYTHON_BIN." >&2
  exit 2
}

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  select_python_bin
fi

source_dir="$(cd -- "$(dirname -- "$report")" && pwd)"
mkdir -p "$output_dir"

bundle_report="$output_dir/evaluation.report.json"
verify_out="$output_dir/invarlock-verify.json"
html_out="$output_dir/evaluation.html"
mlflow_out="$output_dir/mlflow-tags.json"
model_card_out="$output_dir/model-card-invarlock.md"
review_out="$output_dir/release-review.md"
ci_summary_out="$output_dir/ci-summary.md"
run_summary_out="$output_dir/run_summary.txt"

cp "$report" "$bundle_report"
for sidecar in runtime.manifest.json checkpoint_refs.json external_edit_summary.json evidence.meta.json; do
  if [[ -f "$source_dir/$sidecar" ]]; then
    cp "$source_dir/$sidecar" "$output_dir/$sidecar"
  fi
done
if [[ -f "$source_dir/run_command.txt" ]]; then
  cp "$source_dir/run_command.txt" "$output_dir/source_run_command.txt"
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

run_command_out="$output_dir/run_command.txt"
{
  printf '%q' "$0"
  for arg in "${original_args[@]}"; do
    printf ' %q' "$arg"
  done
  printf '\n'
} > "$run_command_out"

"$PYTHON_BIN" -m invarlock verify \
  "$bundle_report" \
  --profile "$profile" \
  --assurance "$assurance" \
  --runtime-provenance "$runtime_provenance" \
  --json > "$verify_out"

"$PYTHON_BIN" -m invarlock report html \
  --input "$bundle_report" \
  --output "$html_out" \
  --force

"$PYTHON_BIN" -m invarlock report export \
  --evaluation-report "$bundle_report" \
  --format mlflow-tags \
  --policy-profile "$profile" \
  --verify-result "$verify_out" \
  --output "$mlflow_out" \
  --force

model_card_cmd=(
  "$PYTHON_BIN" -m invarlock report export
  --evaluation-report "$bundle_report"
  --format model-card-md
  --policy-profile "$profile"
  --verify-result "$verify_out"
  --output "$model_card_out"
  --force
)
if [[ -n "$report_url" ]]; then
  model_card_cmd+=(--report-url "$report_url")
fi
if [[ -n "$evidence_url" ]]; then
  model_card_cmd+=(--evidence-url "$evidence_url")
fi
"${model_card_cmd[@]}"

"$PYTHON_BIN" -m invarlock report export \
  --evaluation-report "$bundle_report" \
  --format release-review-md \
  --policy-profile "$profile" \
  --verify-result "$verify_out" \
  --output "$review_out" \
  --force

{
  echo "### InvarLock"
  echo
  cat "$review_out"
} > "$ci_summary_out"

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  cat "$ci_summary_out" >> "$GITHUB_STEP_SUMMARY"
fi

{
  echo "InvarLock public end-to-end example complete"
  echo "  status: success"
  echo "  source report: $report"
  echo "  output directory: $output_dir"
  echo "  bundled report: $bundle_report"
  echo "  verify: $verify_out"
  echo "  html: $html_out"
  echo "  mlflow tags: $mlflow_out"
  echo "  model-card block: $model_card_out"
  echo "  release review: $review_out"
  echo "  CI summary: $ci_summary_out"
  echo "  run command: $run_command_out"
} | tee "$run_summary_out"
