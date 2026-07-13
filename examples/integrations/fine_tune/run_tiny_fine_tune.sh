#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_tiny_fine_tune.sh [options]

Fine-tune a tiny subject from an immutable profile, verify its artifacts
independently, then compare it with the pinned baseline.

Options:
  --training-profile ID       Immutable training profile. Defaults to the
                              canonical CPU or CUDA full-fine-tune profile.
  --subject-dir DIR           Output directory for the fine-tuned subject.
  --fixture-dir DIR           Generated local evaluation JSONL/preset directory.
  --report-out DIR            Output directory for InvarLock artifacts. Default
                              ends in reports/tiny-fine-tune/<artifact-lane>.
  --profile NAME              InvarLock evaluation profile. Default: release
  --tier NAME                 InvarLock tier. Default: balanced
  --lane MODE                 Standard lane shortcut: host or cuda.
  --execution-mode MODE       container or host. Default: container
  --assurance MODE            strict or off. Default: strict
  --runtime-provenance MODE   container or host for verify.
  --device VALUE              Optional evaluation-device override.
  --allow-network             Allow retrieval of the pinned model revision.
  --force                     Replace an existing subject directory.
  --materialize-only          Stop after training, verification, and fixture prep.
  --no-html                   Skip HTML rendering in the compare wrapper.
  -h, --help                  Show this help.

The training receipt binds the immutable profile, data, baseline, subject, and
deltas. It does not independently attest optimizer history; trusted execution
or an independent rerun is still required.
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PROFILES_PATH="$REPO_ROOT/scripts/evidence_packs/training_profiles.json"

subject_dir="$SCRIPT_DIR/models/tiny-gpt2-fine-tuned"
fixture_dir="$SCRIPT_DIR/artifacts/tiny-fine-tune-fixture"
report_out="$SCRIPT_DIR/reports/tiny-fine-tune"
report_out_was_default=1
training_profile=""
baseline_override=""
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
    --training-profile) training_profile="${2:-}"; shift 2 ;;
    --baseline) baseline_override="${2:-}"; shift 2 ;;
    --subject-dir) subject_dir="${2:-}"; shift 2 ;;
    --fixture-dir) fixture_dir="${2:-}"; shift 2 ;;
    --report-out) report_out="${2:-}"; report_out_was_default=0; shift 2 ;;
    --profile) profile="${2:-}"; shift 2 ;;
    --tier) tier="${2:-}"; shift 2 ;;
    --lane) lane="${2:-}"; shift 2 ;;
    --execution-mode) execution_mode="${2:-}"; shift 2 ;;
    --assurance) assurance="${2:-}"; shift 2 ;;
    --runtime-provenance) runtime_provenance="${2:-}"; shift 2 ;;
    --device) device="${2:-}"; shift 2 ;;
    --allow-network) allow_network=1; shift ;;
    --force) force=1; shift ;;
    --materialize-only) materialize_only=1; shift ;;
    --no-html) render_html=0; shift ;;
    --learning-rate)
      echo "--learning-rate is incompatible with immutable training profiles; define and review a new profile instead." >&2
      exit 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$subject_dir" || -z "$fixture_dir" || -z "$report_out" ]]; then
  echo "Missing required subject, fixture, or report path." >&2
  exit 2
fi

select_python_bin() {
  local candidate
  for candidate in python "$REPO_ROOT/.venv/bin/python" python3; do
    if [[ "$candidate" == */* ]]; then
      [[ -x "$candidate" ]] || continue
    elif ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    if "$candidate" -c 'import torch, transformers' >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      return
    fi
  done
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
}

python_bin_is_executable() {
  local candidate="$1"
  if [[ "$candidate" == */* ]]; then
    [[ -x "$candidate" ]]
  else
    command -v "$candidate" >/dev/null 2>&1
  fi
}

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  select_python_bin
fi
if ! python_bin_is_executable "$PYTHON_BIN" || ! "$PYTHON_BIN" -c 'import torch, transformers' >/dev/null 2>&1; then
  echo 'Missing training dependencies. Install the `training` extra (for example, `uv sync --extra training`).' >&2
  exit 2
fi

# shellcheck source=../_shared/preflight.sh
source "$REPO_ROOT/examples/integrations/_shared/preflight.sh"
# shellcheck source=../_shared/training_profiles.sh
source "$REPO_ROOT/examples/integrations/_shared/training_profiles.sh"

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

if [[ -z "$training_profile" ]]; then
  training_profile="$(integration_default_training_profile fine_tune "$effective_device")"
fi
integration_load_training_profile "$PYTHON_BIN" "$PROFILES_PATH" "$training_profile" fine_tune || exit $?
integration_preflight_training_device "$PYTHON_BIN" "$TRAINING_DEVICE" "fine-tune" || exit $?
integration_preflight_host_cuda_device "$PYTHON_BIN" "$effective_execution_mode" "$effective_device" "fine-tune" || exit $?

if [[ -n "$baseline_override" && "$baseline_override" != "$TRAINING_MODEL_ID" ]]; then
  echo "--baseline cannot override immutable profile model_id=$TRAINING_MODEL_ID; define a new reviewed profile." >&2
  exit 2
fi
baseline="$TRAINING_MODEL_ID"

integration_log_header "Fine-tune integration example"
integration_log_kv "lane" "$lane_artifact_label"
integration_log_kv "training_profile" "$training_profile"
integration_log_kv "training_device" "$TRAINING_DEVICE"
integration_log_kv "report_out" "$report_out"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
integration_prepare_training_output \
  "$PYTHON_BIN" "$REPO_ROOT" "$subject_dir" "$force" || exit $?

integration_log_step "execute and independently verify immutable full fine-tune profile"
integration_run_training_profile \
  "$PYTHON_BIN" "$REPO_ROOT" "$PROFILES_PATH" "$training_profile" \
  "$subject_dir" "$allow_network"

integration_log_step "prepare deterministic local evaluation fixture"
integration_run_source_archive_clean \
  "$PYTHON_BIN" "$REPO_ROOT/examples/integrations/_shared/prepare_training_fixture.py" \
  --fixture-dir "$fixture_dir" \
  --model-id "$baseline" \
  --format-version tiny-fine-tune-fixture-v1

if [[ "$materialize_only" -eq 1 ]]; then
  echo "Wrote verified subject checkpoint: $subject_dir"
  exit 0
fi

integration_log_step "collect receipt and evaluation-fixture metadata"
mkdir -p "$report_out"
cp "$subject_dir/training_receipt.json" "$report_out/training_receipt.json"
cp "$fixture_dir/fixture_summary.json" "$report_out/fixture_summary.json"
rm -f "$report_out/training_binding.json"

compare_cmd=(
  "$REPO_ROOT/examples/integrations/_shared/run_invarlock_compare.sh"
  --baseline "$baseline"
  --baseline-revision "$TRAINING_MODEL_REVISION"
  --subject "$subject_dir"
  --baseline-adapter hf_causal
  --subject-adapter hf_causal
  --profile "$profile"
  --tier "$tier"
  --preset "$fixture_dir/preset.yaml"
  --report-out "$report_out"
  --execution-mode "$execution_mode"
  --assurance "$assurance"
  --edit-label fine_tune
)
if [[ -n "$lane" ]]; then compare_cmd+=(--lane "$lane"); fi
if [[ -n "$runtime_provenance" ]]; then compare_cmd+=(--runtime-provenance "$runtime_provenance"); fi
if [[ -n "$device" ]]; then compare_cmd+=(--device "$device"); fi
if [[ "$allow_network" -eq 1 ]]; then compare_cmd+=(--allow-network); fi
if [[ "$render_html" -eq 0 ]]; then compare_cmd+=(--no-html); fi

integration_log_step "run InvarLock comparison"
"${compare_cmd[@]}"

integration_log_step "reverify post-evaluation training artifact binding"
integration_finalize_training_binding \
  "$PYTHON_BIN" "$REPO_ROOT" "$PROFILES_PATH" "$training_profile" \
  "$subject_dir" "$report_out/training_receipt.json" "$allow_network" \
  "$report_out"

integration_log_step "stage receipt-bound training evidence proof"
integration_stage_training_evidence \
  "$PYTHON_BIN" "$REPO_ROOT" "$PROFILES_PATH" "$training_profile" \
  "$subject_dir" "$report_out" "$allow_network" "all"
