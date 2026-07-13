#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib/smoke_common.sh"
GPT2_HELPER="$SCRIPT_DIR/gpt2_journey_helpers.py"

WORK_ROOT="${1:-$(mktemp -d -t invarlock_gpt2_user_journey.XXXXXX)}"
PRESET="${INVARLOCK_GPT2_SMOKE_PRESET:-$REPO_ROOT/configs/presets/causal_lm/gpt2_smoke_128.yaml}"
DEFAULT_QUANT_EDIT_CONFIG="$REPO_ROOT/configs/overlays/edits/quant_rtn/8bit_full.yaml"
MODE="${INVARLOCK_SMOKE_MODE:-all}"
if [[ -n "${INVARLOCK_SMOKE_PROFILE:-}" ]]; then
  PROFILE="$INVARLOCK_SMOKE_PROFILE"
elif [[ "$MODE" == "container" ]]; then
  PROFILE="dev"
else
  PROFILE="dev"
fi
ASSURANCE="${INVARLOCK_SMOKE_ASSURANCE:-}"
EDIT_CONFIG="${INVARLOCK_SMOKE_EDIT_CONFIG:-}"
QUANT_EDIT_CONFIG="${INVARLOCK_SMOKE_QUANT_EDIT_CONFIG:-${EDIT_CONFIG:-$DEFAULT_QUANT_EDIT_CONFIG}}"
CUSTOM_EDIT_CONFIG="${INVARLOCK_SMOKE_CUSTOM_EDIT_CONFIG:-${EDIT_CONFIG:-$DEFAULT_QUANT_EDIT_CONFIG}}"
SMOKE_DEVICE="${INVARLOCK_SMOKE_DEVICE:-auto}"
if [[ "$MODE" == "container" ]]; then
  DEFAULT_JOURNEYS="strict-bundle,noop,quantized,edited,negative"
else
  DEFAULT_JOURNEYS="noop,quantized,negative"
fi

case "${INVARLOCK_SMOKE_QUANTIZED:-0}" in
  1|true|TRUE|yes|YES)
    DEFAULT_JOURNEYS="quantized,negative"
    ;;
esac

JOURNEYS_RAW="${INVARLOCK_SMOKE_JOURNEYS:-$DEFAULT_JOURNEYS}"

if [[ -z "$ASSURANCE" ]]; then
  ASSURANCE="off"
  if [[ "$MODE" == "container" && ( "$PROFILE" == "ci" || "$PROFILE" == "release" ) ]]; then
    ASSURANCE="strict"
  fi
fi

if [[ -n "$EDIT_CONFIG" && "$EDIT_CONFIG" != /* ]]; then
  EDIT_CONFIG="$REPO_ROOT/$EDIT_CONFIG"
fi
if [[ -n "$QUANT_EDIT_CONFIG" && "$QUANT_EDIT_CONFIG" != /* ]]; then
  QUANT_EDIT_CONFIG="$REPO_ROOT/$QUANT_EDIT_CONFIG"
fi
if [[ -n "$CUSTOM_EDIT_CONFIG" && "$CUSTOM_EDIT_CONFIG" != /* ]]; then
  CUSTOM_EDIT_CONFIG="$REPO_ROOT/$CUSTOM_EDIT_CONFIG"
fi

if [[ ! -f "$PRESET" ]]; then
  echo "[error] GPT-2 smoke preset not found: $PRESET" >&2
  exit 2
fi

if [[ -n "$EDIT_CONFIG" && ! -f "$EDIT_CONFIG" ]]; then
  echo "[error] GPT-2 smoke edit config not found: $EDIT_CONFIG" >&2
  exit 2
fi
if [[ -n "$QUANT_EDIT_CONFIG" && ! -f "$QUANT_EDIT_CONFIG" ]]; then
  echo "[error] GPT-2 smoke quant edit config not found: $QUANT_EDIT_CONFIG" >&2
  exit 2
fi
if [[ -n "$CUSTOM_EDIT_CONFIG" && ! -f "$CUSTOM_EDIT_CONFIG" ]]; then
  echo "[error] GPT-2 smoke custom edit config not found: $CUSTOM_EDIT_CONFIG" >&2
  exit 2
fi

PYTHON_BIN="$(smoke_select_python "$REPO_ROOT" "${INVARLOCK_PYTHON:-}")"
smoke_setup_pythonpath "$REPO_ROOT"
CLI=("$PYTHON_BIN" -m invarlock)

export INVARLOCK_ALLOW_NETWORK="${INVARLOCK_ALLOW_NETWORK:-1}"
export INVARLOCK_DEDUP_TEXTS="${INVARLOCK_DEDUP_TEXTS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "${INVARLOCK_SMOKE_PLAN:-0}" == "1" ]]; then
  SMOKE_PLAN_WORK_ROOT="$WORK_ROOT" \
  SMOKE_PLAN_PRESET="$PRESET" \
  SMOKE_PLAN_MODE="$MODE" \
  SMOKE_PLAN_PROFILE="$PROFILE" \
  SMOKE_PLAN_ASSURANCE="$ASSURANCE" \
  SMOKE_PLAN_DEVICE="$SMOKE_DEVICE" \
  SMOKE_PLAN_JOURNEYS="$JOURNEYS_RAW" \
  SMOKE_PLAN_QUANT_EDIT_CONFIG="$QUANT_EDIT_CONFIG" \
  SMOKE_PLAN_CUSTOM_EDIT_CONFIG="$CUSTOM_EDIT_CONFIG" \
  SMOKE_PLAN_HOST_HF_CACHE_ROOT="${INVARLOCK_SMOKE_HOST_HF_CACHE_ROOT:-${HF_HOME:-${HOME}/.cache/huggingface}}" \
  SMOKE_PLAN_COMMANDS="$(smoke_plan_markers command "${BASH_SOURCE[0]}")" \
  SMOKE_PLAN_HELPER_CONTRACTS="$(smoke_plan_markers helper "${BASH_SOURCE[0]}")" \
  "$PYTHON_BIN" - <<'PY'
import json
import os

journeys = [
    journey.strip()
    for journey in os.environ["SMOKE_PLAN_JOURNEYS"].split(",")
    if journey.strip()
]
commands = [
    command.strip()
    for command in os.environ["SMOKE_PLAN_COMMANDS"].splitlines()
    if command.strip()
]
helper_contracts = [
    helper.strip()
    for helper in os.environ["SMOKE_PLAN_HELPER_CONTRACTS"].splitlines()
    if helper.strip()
]
plan = {
    "script": "run_gpt2_user_journey_smoke",
    "work_root": os.environ["SMOKE_PLAN_WORK_ROOT"],
    "preset": os.environ["SMOKE_PLAN_PRESET"],
    "mode": os.environ["SMOKE_PLAN_MODE"],
    "profile": os.environ["SMOKE_PLAN_PROFILE"],
    "assurance": os.environ["SMOKE_PLAN_ASSURANCE"],
    "device": os.environ["SMOKE_PLAN_DEVICE"],
    "journeys": journeys,
    "cache": {
        "host_hf_cache_root": os.environ["SMOKE_PLAN_HOST_HF_CACHE_ROOT"],
        "worktree_hf_cache": ".hf",
        "offline_when_cache_complete": True,
    },
    "edit_configs": {
        "quantized": os.environ["SMOKE_PLAN_QUANT_EDIT_CONFIG"],
        "custom": os.environ["SMOKE_PLAN_CUSTOM_EDIT_CONFIG"],
    },
    "commands": commands,
    "helper_contracts": helper_contracts,
    "child_suites": [
        {"suite": "local", "mode": "local", "assurance": "off"},
        {"suite": "container", "mode": "container", "assurance": "off"},
    ],
}
print(json.dumps(plan, sort_keys=True))
PY
  exit 0
fi

mkdir -p "$WORK_ROOT"

RESULTS_TSV="$WORK_ROOT/journey-results.tsv"
SMOKE_CACHE_ROOT="$WORK_ROOT/.hf"
HOST_HF_CACHE_ROOT="${INVARLOCK_SMOKE_HOST_HF_CACHE_ROOT:-${HF_HOME:-${HOME}/.cache/huggingface}}"
FAILED_JOURNEYS=0
TOTAL_JOURNEYS=0
FIRST_EVAL_REPORT=""
LAST_EVAL_REPORT=""

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "journey" "expectation" "status" "verify" "metric" "artifact" "note" \
  > "$RESULTS_TSV"

record_result() {
  local journey="$1"
  local expectation="$2"
  local status="$3"
  local verify="$4"
  local metric="$5"
  local artifact="$6"
  local note="${7:-}"

  TOTAL_JOURNEYS=$((TOTAL_JOURNEYS + 1))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$journey" "$expectation" "$status" "$verify" "$metric" "$artifact" "$note" \
    >> "$RESULTS_TSV"
  if [[ "$status" != "PASS" && "$status" != "SKIP" ]]; then
    FAILED_JOURNEYS=$((FAILED_JOURNEYS + 1))
  fi
}

has_seeded_wikitext_cache() {
  [[ -d "$SMOKE_CACHE_ROOT/hub/datasets--Salesforce--wikitext" ]] || \
    [[ -d "$SMOKE_CACHE_ROOT/hub/datasets--wikitext" ]] || \
    [[ -d "$SMOKE_CACHE_ROOT/datasets/Salesforce___wikitext/wikitext-2-raw-v1" ]] || \
    [[ -d "$SMOKE_CACHE_ROOT/datasets/wikitext/wikitext-2-raw-v1" ]]
}

seed_hf_cache_from_host() {
  local seeded=0
  smoke_copy_cached_tree_if_present "$HOST_HF_CACHE_ROOT/hub/models--gpt2" "$SMOKE_CACHE_ROOT/hub/models--gpt2"
  smoke_copy_cached_tree_if_present "$HOST_HF_CACHE_ROOT/hub/datasets--Salesforce--wikitext" "$SMOKE_CACHE_ROOT/hub/datasets--Salesforce--wikitext"
  smoke_copy_cached_tree_if_present "$HOST_HF_CACHE_ROOT/hub/datasets--wikitext" "$SMOKE_CACHE_ROOT/hub/datasets--wikitext"
  smoke_copy_cached_tree_if_present \
    "$HOST_HF_CACHE_ROOT/datasets/Salesforce___wikitext/wikitext-2-raw-v1" \
    "$SMOKE_CACHE_ROOT/datasets/Salesforce___wikitext/wikitext-2-raw-v1"
  smoke_copy_cached_tree_if_present \
    "$HOST_HF_CACHE_ROOT/datasets/wikitext/wikitext-2-raw-v1" \
    "$SMOKE_CACHE_ROOT/datasets/wikitext/wikitext-2-raw-v1"
  if [[ -d "$SMOKE_CACHE_ROOT/hub/models--gpt2" ]] || has_seeded_wikitext_cache; then
    seeded=1
  fi
  if [[ "$seeded" != "1" ]]; then
    return 1
  fi
  if [[ -d "$SMOKE_CACHE_ROOT/hub/models--gpt2" ]] && has_seeded_wikitext_cache; then
    export INVARLOCK_SMOKE_CACHE_COMPLETE=1
  fi
  return 0
}

prefetch_hf_assets_on_host() {
  if [[ "${INVARLOCK_ALLOW_NETWORK:-0}" != "1" ]]; then
    return 1
  fi
  mkdir -p "$HOST_HF_CACHE_ROOT" "$HOST_HF_CACHE_ROOT/hub" "$HOST_HF_CACHE_ROOT/datasets"
  echo "[smoke] prefetching GPT-2 + WikiText-2 into host HF cache"
  HF_HOME="$HOST_HF_CACHE_ROOT" \
    HF_HUB_CACHE="$HOST_HF_CACHE_ROOT/hub" \
    HF_DATASETS_CACHE="$HOST_HF_CACHE_ROOT/datasets" \
    "$PYTHON_BIN" - <<'PY'
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "gpt2"

AutoTokenizer.from_pretrained(MODEL_ID)
AutoModelForCausalLM.from_pretrained(MODEL_ID)
load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
PY
}

metric_summary() {
  local report="$1"
  "$PYTHON_BIN" "$GPT2_HELPER" metric-summary "$report"
}

verify_reason() {
  local verify_json="$1"
  "$PYTHON_BIN" "$GPT2_HELPER" verify-reason "$verify_json"
}

print_results_table() {
  "$PYTHON_BIN" "$GPT2_HELPER" print-results-table "$RESULTS_TSV"
}

write_final_verdict() {
  local verdict="PASS"
  if [[ "$FAILED_JOURNEYS" -ne 0 ]]; then
    verdict="FAIL"
  fi
  "$PYTHON_BIN" "$GPT2_HELPER" write-final-verdict "$RESULTS_TSV" "$WORK_ROOT/final_verdict.json" "$verdict"
}


# smoke-plan-helper: run_eval_journey
run_eval_journey() {
  local journey="$1"
  local edit_config="$2"
  local expectation="$3"
  local journey_root="$WORK_ROOT/journeys/$journey"
  local run_dir="$journey_root/runs"
  local report_dir="$journey_root/reports/eval"
  local export_dir="$journey_root/exports"
  local verify_json="$journey_root/verify.json"
  local explain_txt="$journey_root/explain.txt"

  mkdir -p "$journey_root" "$export_dir"

  local edit_args=()
  if [[ -n "$edit_config" ]]; then
    edit_args=(--edit-config "$edit_config")
  fi

  echo ""
  echo "==== BEGIN journey:$journey ===="
  echo "[journey] expectation=$expectation"
  if [[ -n "$edit_config" ]]; then
    echo "[journey] edit_config=$edit_config"
  else
    echo "[journey] edit_config=noop"
  fi

  local rc=0
  set +e
  # smoke-plan-command: evaluate
  "${CLI[@]}" evaluate \
    --baseline gpt2 \
    --subject gpt2 \
    --baseline-adapter auto --subject-adapter auto \
    --profile "$PROFILE" \
    --preset "$PRESET" \
    --device "$SMOKE_DEVICE" \
    "${edit_args[@]}" \
    --execution-mode "$EXECUTION_MODE" \
    --assurance "$ASSURANCE" \
    --out "$run_dir" \
    --report-out "$report_dir" \
    --timing
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    record_result "$journey/evaluate" "$expectation" "FAIL" "evaluate_rc=$rc" "-" "$journey_root" "evaluate failed"
    echo "==== END journey:$journey ===="
    return 0
  fi

  local baseline_report=""
  local edited_report=""
  local eval_report="$report_dir/evaluation.report.json"
  baseline_report="$(find "$run_dir/source" -name report.json -print 2>/dev/null | sort | tail -n1 || true)"
  edited_report="$(find "$run_dir/edited" -name report.json -print 2>/dev/null | sort | tail -n1 || true)"

  if [[ -z "$baseline_report" || -z "$edited_report" || ! -f "$eval_report" ]]; then
    record_result "$journey/evaluate" "$expectation" "FAIL" "missing_artifacts" "-" "$journey_root" "missing reports"
    echo "==== END journey:$journey ===="
    return 0
  fi

  echo "[journey] baseline_report=$baseline_report"
  echo "[journey] edited_report=$edited_report"
  echo "[journey] evaluation_report=$eval_report"

  set +e
  # smoke-plan-command: verify
  "${CLI[@]}" verify "$eval_report" "${VERIFY_ARGS[@]}" --json > "$verify_json"
  rc=$?
  set -e
  cat "$verify_json"
  if [[ "$rc" -ne 0 ]]; then
    record_result "$journey/verify" "verify accepts report" "FAIL" "verify_rc=$rc reason=$(verify_reason "$verify_json")" "$(metric_summary "$eval_report")" "$eval_report" "verify failed"
    echo "==== END journey:$journey ===="
    return 0
  fi

  # smoke-plan-command: report validate
  if ! "${CLI[@]}" report validate "$eval_report"; then
    record_result "$journey/report-validate" "schema validates" "FAIL" "ok" "$(metric_summary "$eval_report")" "$eval_report" "schema validation failed"
    echo "==== END journey:$journey ===="
    return 0
  fi

  # smoke-plan-command: report html
  if ! "${CLI[@]}" report html -i "$eval_report" -o "$export_dir/evaluation.html" --force; then
    record_result "$journey/report-html" "HTML renders" "FAIL" "ok" "$(metric_summary "$eval_report")" "$export_dir/evaluation.html" "HTML render failed"
    echo "==== END journey:$journey ===="
    return 0
  fi

  # smoke-plan-command: report explain
  if ! "${CLI[@]}" report explain --subject-report "$edited_report" --baseline-report "$baseline_report" > "$explain_txt"; then
    record_result "$journey/report-explain" "explain renders" "FAIL" "ok" "$(metric_summary "$eval_report")" "$explain_txt" "explain failed"
    echo "==== END journey:$journey ===="
    return 0
  fi
  cat "$explain_txt"

  record_result "$journey/evaluate-verify-report" "$expectation" "PASS" "ok" "$(metric_summary "$eval_report")" "$export_dir/evaluation.html" "evaluate -> verify -> validate -> html -> explain"

  if [[ -z "$FIRST_EVAL_REPORT" ]]; then
    FIRST_EVAL_REPORT="$eval_report"
  fi
  LAST_EVAL_REPORT="$eval_report"
  echo "==== END journey:$journey ===="
}

# smoke-plan-helper: run_negative_journey
run_negative_journey() {
  local source_report="${FIRST_EVAL_REPORT:-$LAST_EVAL_REPORT}"
  local journey="negative"
  local journey_root="$WORK_ROOT/journeys/$journey"
  local eval_report="$journey_root/evaluation.report.json"
  local verify_json="$journey_root/verify.json"
  local export_dir="$journey_root/exports"

  mkdir -p "$journey_root" "$export_dir"

  echo ""
  echo "==== BEGIN journey:$journey ===="
  echo "[journey] expectation=verify rejects a mutated live GPT-2 report"

  if [[ -z "$source_report" || ! -f "$source_report" ]]; then
    record_result "$journey/verify" "verify rejects mutated report" "FAIL" "missing_source" "-" "$journey_root" "no prior successful report"
    echo "==== END journey:$journey ===="
    return 0
  fi

  "$PYTHON_BIN" "$GPT2_HELPER" mutate-negative-report "$source_report" "$eval_report"

  local rc=0
  set +e
  # smoke-plan-command: verify
  "${CLI[@]}" verify "$eval_report" "${VERIFY_ARGS[@]}" --json > "$verify_json"
  rc=$?
  set -e
  cat "$verify_json"

  local reason=""
  reason="$(verify_reason "$verify_json")"
  if [[ "$rc" -eq 0 || "$reason" != "policy_fail" ]]; then
    record_result "$journey/verify" "verify rejects mutated report" "FAIL" "verify_rc=$rc reason=$reason" "$(metric_summary "$eval_report")" "$eval_report" "expected policy_fail"
    echo "==== END journey:$journey ===="
    return 0
  fi

  # smoke-plan-command: report validate
  if ! "${CLI[@]}" report validate "$eval_report"; then
    record_result "$journey/report-validate" "mutated report remains schema-valid" "FAIL" "reason=$reason" "$(metric_summary "$eval_report")" "$eval_report" "schema validation failed"
    echo "==== END journey:$journey ===="
    return 0
  fi

  # smoke-plan-command: report html
  if ! "${CLI[@]}" report html -i "$eval_report" -o "$export_dir/evaluation.html" --force; then
    record_result "$journey/report-html" "HTML renders rejected report" "FAIL" "reason=$reason" "$(metric_summary "$eval_report")" "$export_dir/evaluation.html" "HTML render failed"
    echo "==== END journey:$journey ===="
    return 0
  fi

  # smoke-plan-helper: verify-rejects
  record_result "$journey/verify-rejects" "verify rejects mutated report" "PASS" "verify_rc=$rc reason=$reason" "$(metric_summary "$eval_report")" "$export_dir/evaluation.html" "schema valid; HTML rendered for inspection"
  echo "==== END journey:$journey ===="
}

# smoke-plan-helper: write_strict_bundle_fixture
write_strict_bundle_fixture() {
  local eval_report="$1"
  "$PYTHON_BIN" "$GPT2_HELPER" write-strict-bundle-fixture "$eval_report"
}

# smoke-plan-helper: run_strict_bundle_journey
run_strict_bundle_journey() {
  local journey="strict-bundle"
  local journey_root="$WORK_ROOT/journeys/$journey"
  local eval_report="$journey_root/evaluation.report.json"
  local verify_json="$journey_root/verify.json"
  local export_dir="$journey_root/exports"

  mkdir -p "$journey_root" "$export_dir"

  echo ""
  echo "==== BEGIN journey:$journey ===="
  echo "[journey] expectation=strict report bundle verifies runtime provenance"
  write_strict_bundle_fixture "$eval_report"

  local rc=0
  set +e
  # smoke-plan-command: verify
  "${CLI[@]}" verify "$eval_report" --assurance strict --profile ci --json > "$verify_json"
  rc=$?
  set -e
  cat "$verify_json"
  if [[ "$rc" -ne 0 ]]; then
    record_result "$journey/verify" "strict report bundle verifies" "FAIL" "verify_rc=$rc reason=$(verify_reason "$verify_json")" "$(metric_summary "$eval_report")" "$eval_report" "strict verification failed"
    echo "==== END journey:$journey ===="
    return 0
  fi

  # smoke-plan-command: report validate
  if ! "${CLI[@]}" report validate "$eval_report"; then
    record_result "$journey/report-validate" "schema validates" "FAIL" "ok" "$(metric_summary "$eval_report")" "$eval_report" "schema validation failed"
    echo "==== END journey:$journey ===="
    return 0
  fi

  # smoke-plan-command: report html
  if ! "${CLI[@]}" report html -i "$eval_report" -o "$export_dir/evaluation.html" --force; then
    record_result "$journey/report-html" "HTML renders" "FAIL" "ok" "$(metric_summary "$eval_report")" "$export_dir/evaluation.html" "HTML render failed"
    echo "==== END journey:$journey ===="
    return 0
  fi

  record_result "$journey/verify-report" "strict report bundle verifies" "PASS" "ok" "$(metric_summary "$eval_report")" "$export_dir/evaluation.html" "verify strict -> validate -> html"
  echo "==== END journey:$journey ===="
}

# smoke-plan-helper: append_child_results
append_child_results() {
  local suite="$1"
  local child_root="$2"
  local final_verdict="$child_root/final_verdict.json"

  if [[ ! -f "$final_verdict" ]]; then
    record_result "$suite" "child smoke emits final verdict" "FAIL" "missing_final_verdict" "-" "$child_root" "no final verdict"
    return 1
  fi

  "$PYTHON_BIN" "$GPT2_HELPER" append-child-results "$RESULTS_TSV" "$final_verdict" "$suite"
}

# smoke-plan-helper: run_child_suite
run_child_suite() {
  local suite="$1"
  local child_mode="$2"
  local child_profile="$3"
  local child_assurance="$4"
  local child_journeys="$5"
  local child_root="$WORK_ROOT/$suite"
  local rc=0

  echo ""
  echo "==== BEGIN suite:$suite ===="
  echo "[suite] mode=$child_mode profile=$child_profile assurance=$child_assurance journeys=$child_journeys"
  set +e
  INVARLOCK_SMOKE_MODE="$child_mode" \
    INVARLOCK_SMOKE_PROFILE="$child_profile" \
    INVARLOCK_SMOKE_ASSURANCE="$child_assurance" \
    INVARLOCK_SMOKE_JOURNEYS="$child_journeys" \
    INVARLOCK_SMOKE_QUANT_EDIT_CONFIG="$QUANT_EDIT_CONFIG" \
    INVARLOCK_SMOKE_CUSTOM_EDIT_CONFIG="$CUSTOM_EDIT_CONFIG" \
    INVARLOCK_SMOKE_DEVICE="$SMOKE_DEVICE" \
    INVARLOCK_GPT2_SMOKE_PRESET="$PRESET" \
    bash "$SCRIPT_DIR/run_gpt2_user_journey_smoke.sh" "$child_root"
  rc=$?
  set -e
  if ! append_child_results "$suite" "$child_root"; then
    rc=1
  fi
  echo "[suite] exit_code=$rc"
  echo "==== END suite:$suite ===="
  return "$rc"
}

# smoke-plan-helper: run_all_mode_journeys
run_all_mode_journeys() {
  local default_local_journeys="noop,quantized,negative"
  local default_container_journeys="strict-bundle,noop,quantized,edited,negative"
  case "${INVARLOCK_SMOKE_QUANTIZED:-0}" in
    1|true|TRUE|yes|YES)
      default_local_journeys="quantized,negative"
      default_container_journeys="quantized,negative"
      ;;
  esac
  local local_journeys="${INVARLOCK_SMOKE_LOCAL_JOURNEYS:-${INVARLOCK_SMOKE_JOURNEYS:-$default_local_journeys}}"
  local container_journeys="${INVARLOCK_SMOKE_CONTAINER_JOURNEYS:-${INVARLOCK_SMOKE_JOURNEYS:-$default_container_journeys}}"
  local failed_suites=0
  local rc=0

  echo "[smoke] work_root=$WORK_ROOT"
  echo "[smoke] preset=$PRESET"
  echo "[smoke] mode=all"
  echo "[smoke] local_journeys=$local_journeys"
  echo "[smoke] container_journeys=$container_journeys"

  set +e
  run_child_suite "local" "local" "${INVARLOCK_SMOKE_LOCAL_PROFILE:-dev}" "${INVARLOCK_SMOKE_LOCAL_ASSURANCE:-off}" "$local_journeys"
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    failed_suites=$((failed_suites + 1))
  fi

  set +e
  run_child_suite "container" "container" "${INVARLOCK_SMOKE_CONTAINER_PROFILE:-dev}" "${INVARLOCK_SMOKE_CONTAINER_ASSURANCE:-off}" "$container_journeys"
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    failed_suites=$((failed_suites + 1))
  fi

  FAILED_JOURNEYS="$failed_suites"
  print_results_table
  write_final_verdict
  echo "[smoke] results=$RESULTS_TSV"
  echo "[smoke] final_verdict=$WORK_ROOT/final_verdict.json"
  echo "[smoke] complete"

  if [[ "$failed_suites" -ne 0 ]]; then
    exit 1
  fi
  exit 0
}

if [[ "$MODE" == "all" ]]; then
  run_all_mode_journeys
fi

if [[ "$MODE" == "container" && -z "${INVARLOCK_RUNTIME_IMAGE:-}" ]]; then
  smoke_seed_local_runtime_image "$SMOKE_DEVICE"
fi

if seed_hf_cache_from_host; then
  export INVARLOCK_SMOKE_CACHE_SEEDED=1
elif prefetch_hf_assets_on_host && seed_hf_cache_from_host; then
  export INVARLOCK_SMOKE_CACHE_SEEDED=1
fi

smoke_ensure_writable_hf_cache "$SMOKE_CACHE_ROOT"
smoke_ensure_current_runtime_image "$MODE" "$SMOKE_DEVICE"

if [[ "${INVARLOCK_SMOKE_CACHE_COMPLETE:-0}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

EXECUTION_MODE="container"
RUNTIME_PROVENANCE="container"
if [[ "$MODE" == "local" ]]; then
  EXECUTION_MODE="host"
  RUNTIME_PROVENANCE="host"
fi
VERIFY_ARGS=(--runtime-provenance "$RUNTIME_PROVENANCE")

echo "[smoke] work_root=$WORK_ROOT"
echo "[smoke] preset=$PRESET"
echo "[smoke] journeys=$JOURNEYS_RAW"
echo "[smoke] mode=$MODE profile=$PROFILE assurance=$ASSURANCE"
echo "[smoke] device=$SMOKE_DEVICE"
echo "[smoke] quant_edit_config=$QUANT_EDIT_CONFIG"
echo "[smoke] custom_edit_config=$CUSTOM_EDIT_CONFIG"
echo "[smoke] hf_home=$HF_HOME"
echo "[smoke] hf_datasets_cache=$HF_DATASETS_CACHE"

IFS=',' read -r -a JOURNEYS <<< "$JOURNEYS_RAW"
for journey in "${JOURNEYS[@]}"; do
  journey="${journey//[[:space:]]/}"
  [[ -n "$journey" ]] || continue
  case "$journey" in
    strict-bundle|strict)
      run_strict_bundle_journey
      ;;
    noop)
      run_eval_journey "noop" "" "baseline gpt2 vs no-op gpt2 should pass"
      ;;
    quantized)
      quant_edit_config="$QUANT_EDIT_CONFIG"
      if [[ ! -f "$quant_edit_config" ]]; then
        record_result "quantized/evaluate" "quantized subject should run" "FAIL" "missing_edit_config" "-" "$quant_edit_config" "edit config missing"
      else
        run_eval_journey "quantized" "$quant_edit_config" "baseline gpt2 vs quantized gpt2 should verify"
      fi
      ;;
    edited|custom-edit)
      custom_edit_config="$CUSTOM_EDIT_CONFIG"
      if [[ ! -f "$custom_edit_config" ]]; then
        record_result "$journey/evaluate" "custom edited subject should run" "FAIL" "missing_edit_config" "-" "$custom_edit_config" "edit config missing"
      else
        run_eval_journey "$journey" "$custom_edit_config" "baseline gpt2 vs configured edited subject should verify"
      fi
      ;;
    negative|failure)
      run_negative_journey
      ;;
    *)
      record_result "$journey" "known journey token" "FAIL" "unknown_journey" "-" "$WORK_ROOT" "supported: strict-bundle,noop,quantized,edited,negative"
      ;;
  esac
done

print_results_table
write_final_verdict
echo "[smoke] results=$RESULTS_TSV"
echo "[smoke] final_verdict=$WORK_ROOT/final_verdict.json"
echo "[smoke] complete"

if [[ "$FAILED_JOURNEYS" -ne 0 ]]; then
  exit 1
fi
