#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_public_e2e_release_review.sh [options]

Build a local review bundle from caller-supplied current evidence.

Options:
  --report PATH                Source evaluation report. Required.
  --baseline PATH              Independent raw baseline report required when
                               --assurance strict is selected.
  --policy-pack PATH           Independent acceptance policy pack required when
                               --assurance strict is selected.
  --output-dir DIR             Output directory for generated handoff artifacts.
                               Default: examples/integrations/public_e2e/reports/release-review
  --profile NAME               InvarLock verification profile. Default: ci.
  --assurance MODE             Verification assurance mode. Default: strict.
  --runtime-provenance MODE    Runtime provenance expectation. Default: container
  --expected-runtime-image-digest DIGEST
                               Independent image pin for strict verification.
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

report=""
output_dir="$SCRIPT_DIR/reports/release-review"
profile="ci"
assurance="strict"
baseline=""
policy_pack=""
runtime_provenance="container"
expected_runtime_image_digest=""
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
    --baseline)
      baseline="${2:-}"
      shift 2
      ;;
    --policy-pack)
      policy_pack="${2:-}"
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
    --expected-runtime-image-digest)
      expected_runtime_image_digest="${2:-}"
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

if [[ "$assurance" == "strict" ]]; then
  if [[ "$profile" != "ci" && "$profile" != "release" ]]; then
    echo "Strict assurance requires --profile ci or --profile release." >&2
    exit 2
  fi
  if [[ -z "$expected_runtime_image_digest" ]]; then
    echo "Strict assurance requires --expected-runtime-image-digest." >&2
    exit 2
  fi
  if [[ -z "$baseline" ]]; then
    echo "Strict assurance requires --baseline with independent raw evidence." >&2
    exit 2
  fi
  if [[ -z "$policy_pack" ]]; then
    echo "Strict assurance requires --policy-pack from an independent source." >&2
    exit 2
  fi
fi

if [[ ! -f "$report" ]]; then
  echo "Evaluation report not found: $report" >&2
  exit 2
fi
if [[ -n "$baseline" && ! -f "$baseline" ]]; then
  echo "Baseline report not found: $baseline" >&2
  exit 2
fi
if [[ -n "$policy_pack" && ! -f "$policy_pack" ]]; then
  echo "Policy pack not found: $policy_pack" >&2
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
sidecar_search_dirs=(
  "$source_dir"
  "$source_dir/../../metadata"
  "$source_dir/../.."
  "$source_dir/../../.."
)

copy_first_sidecar() {
  local sidecar="$1"
  local destination="$2"
  local candidate_dir
  for candidate_dir in "${sidecar_search_dirs[@]}"; do
    if [[ -f "$candidate_dir/$sidecar" ]]; then
      cp "$candidate_dir/$sidecar" "$destination"
      return 0
    fi
  done
  return 1
}

bundle_report="$output_dir/evaluation.report.json"
bundle_baseline="$output_dir/baseline.report.json"
bundle_policy_pack="$output_dir/acceptance-policy-pack.json"
verify_out="$output_dir/invarlock-verify.json"
html_out="$output_dir/evaluation.html"
mlflow_out="$output_dir/mlflow-tags.json"
model_card_out="$output_dir/model-card-invarlock.md"
review_out="$output_dir/release-review.md"
ci_summary_out="$output_dir/ci-summary.md"
run_summary_out="$output_dir/run_summary.txt"

cp "$report" "$bundle_report"
if [[ -n "$baseline" ]]; then
  cp "$baseline" "$bundle_baseline"
fi
if [[ -n "$policy_pack" ]]; then
  cp "$policy_pack" "$bundle_policy_pack"
fi
for sidecar in runtime.manifest.json checkpoint_refs.json external_edit_summary.json evidence.meta.json; do
  copy_first_sidecar "$sidecar" "$output_dir/$sidecar" || true
done
copy_first_sidecar "run_command.txt" "$output_dir/source_run_command.txt" || true

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

run_command_out="$output_dir/run_command.txt"
{
  printf '%q' "$0"
  for arg in "${original_args[@]}"; do
    printf ' %q' "$arg"
  done
  printf '\n'
} > "$run_command_out"

expected_digest_args=()
if [[ -n "$expected_runtime_image_digest" ]]; then
  expected_digest_args=(
    --expected-runtime-image-digest
    "$expected_runtime_image_digest"
  )
fi
baseline_args=()
if [[ -n "$baseline" ]]; then
  baseline_args=(--baseline "$bundle_baseline")
fi
policy_args=()
if [[ -n "$policy_pack" ]]; then
  policy_args=(--policy-pack "$bundle_policy_pack")
fi

"$PYTHON_BIN" -m invarlock verify \
  "$bundle_report" \
  --profile "$profile" \
  --assurance "$assurance" \
  --runtime-provenance "$runtime_provenance" \
  "${baseline_args[@]}" \
  "${policy_args[@]}" \
  "${expected_digest_args[@]}" \
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
