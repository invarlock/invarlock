#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

WORK_ROOT="${1:-$(mktemp -d -t invarlock_gpt2_user_journey.XXXXXX)}"
PRESET="${INVARLOCK_GPT2_SMOKE_PRESET:-$REPO_ROOT/configs/presets/causal_lm/gpt2_smoke_128.yaml}"
DEFAULT_QUANT_EDIT_CONFIG="$REPO_ROOT/configs/overlays/edits/quant_rtn/8bit_full.yaml"
MODE="${INVARLOCK_SMOKE_MODE:-local}"
PROFILE="${INVARLOCK_SMOKE_PROFILE:-dev}"
ASSURANCE="${INVARLOCK_SMOKE_ASSURANCE:-}"
EDIT_CONFIG="${INVARLOCK_SMOKE_EDIT_CONFIG:-}"
DEFAULT_JOURNEYS="noop,quantized,negative"

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

if [[ ! -f "$PRESET" ]]; then
  echo "[error] GPT-2 smoke preset not found: $PRESET" >&2
  exit 2
fi

if [[ -n "$EDIT_CONFIG" && ! -f "$EDIT_CONFIG" ]]; then
  echo "[error] GPT-2 smoke edit config not found: $EDIT_CONFIG" >&2
  exit 2
fi

PYTHON_BIN="${INVARLOCK_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(bash "$REPO_ROOT/scripts/select_workspace_python.sh")"
fi
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CLI=("$PYTHON_BIN" -m invarlock)

export INVARLOCK_ALLOW_NETWORK="${INVARLOCK_ALLOW_NETWORK:-1}"
export INVARLOCK_DEDUP_TEXTS="${INVARLOCK_DEDUP_TEXTS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

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

host_gpu_visible() {
  [[ -e /dev/nvidiactl ]] || command -v nvidia-smi >/dev/null 2>&1
}

seed_local_runtime_image() {
  if host_gpu_visible && docker image inspect invarlock-runtime:cuda-local >/dev/null 2>&1; then
    export INVARLOCK_RUNTIME_IMAGE="invarlock-runtime:cuda-local"
    return 0
  fi
  if docker image inspect invarlock-runtime:local >/dev/null 2>&1; then
    export INVARLOCK_RUNTIME_IMAGE="invarlock-runtime:local"
  fi
}

copy_cached_tree_if_present() {
  local source_dir="$1"
  local target_dir="$2"
  if [[ -d "$source_dir" && ! -e "$target_dir" ]]; then
    mkdir -p "$(dirname -- "$target_dir")"
    cp -a "$source_dir" "$target_dir"
  fi
}

seed_hf_cache_from_host() {
  local seeded=0
  copy_cached_tree_if_present "$HOST_HF_CACHE_ROOT/hub/models--gpt2" "$SMOKE_CACHE_ROOT/hub/models--gpt2"
  copy_cached_tree_if_present "$HOST_HF_CACHE_ROOT/hub/datasets--wikitext" "$SMOKE_CACHE_ROOT/hub/datasets--wikitext"
  copy_cached_tree_if_present \
    "$HOST_HF_CACHE_ROOT/datasets/wikitext/wikitext-2-raw-v1" \
    "$SMOKE_CACHE_ROOT/datasets/wikitext/wikitext-2-raw-v1"
  if [[ -d "$SMOKE_CACHE_ROOT/hub/models--gpt2" || -d "$SMOKE_CACHE_ROOT/hub/datasets--wikitext" || -d "$SMOKE_CACHE_ROOT/datasets/wikitext/wikitext-2-raw-v1" ]]; then
    seeded=1
  fi
  if [[ "$seeded" != "1" ]]; then
    return 1
  fi
  if [[ -d "$SMOKE_CACHE_ROOT/hub/models--gpt2" ]] && \
    [[ -d "$SMOKE_CACHE_ROOT/hub/datasets--wikitext" || -d "$SMOKE_CACHE_ROOT/datasets/wikitext/wikitext-2-raw-v1" ]]; then
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
load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
PY
}

ensure_writable_hf_cache() {
  local candidate_home="${HF_HOME:-$SMOKE_CACHE_ROOT}"
  local candidate_datasets="${HF_DATASETS_CACHE:-${candidate_home}/datasets}"
  local candidate_hub="${HF_HUB_CACHE:-${candidate_home}/hub}"

  if mkdir -p "$candidate_home" "$candidate_datasets" "$candidate_hub" >/dev/null 2>&1 \
    && touch "$candidate_datasets/.ivl_smoke_probe" >/dev/null 2>&1; then
    rm -f "$candidate_datasets/.ivl_smoke_probe" >/dev/null 2>&1 || true
    export HF_HOME="$candidate_home"
    export HF_HUB_CACHE="$candidate_hub"
    export HF_DATASETS_CACHE="$candidate_datasets"
    unset TRANSFORMERS_CACHE
    return 0
  fi

  export HF_HOME="$SMOKE_CACHE_ROOT"
  export HF_HUB_CACHE="$SMOKE_CACHE_ROOT/hub"
  export HF_DATASETS_CACHE="$SMOKE_CACHE_ROOT/datasets"
  unset TRANSFORMERS_CACHE
  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
  echo "[smoke] falling back to writable HF cache under $HF_HOME"
}

ensure_current_runtime_image() {
  if [[ "$MODE" != "container" ]]; then
    return 0
  fi
  if [[ -n "${INVARLOCK_RUNTIME_IMAGE:-}" \
    && "${INVARLOCK_RUNTIME_IMAGE}" != "invarlock-runtime:local" \
    && "${INVARLOCK_RUNTIME_IMAGE}" != "invarlock-runtime:cuda-local" ]]; then
    return 0
  fi
  if ! command -v docker >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1; then
    return 0
  fi
  if host_gpu_visible; then
    echo "[smoke] refreshing local CUDA container runtime image"
    make runtime-image-cuda
    export INVARLOCK_RUNTIME_IMAGE="invarlock-runtime:cuda-local"
    return 0
  fi
  echo "[smoke] refreshing local container runtime image"
  make runtime-image
  export INVARLOCK_RUNTIME_IMAGE="invarlock-runtime:local"
}

metric_summary() {
  local report="$1"
  "$PYTHON_BIN" - "$report" <<'PY'
import json
import sys

report = json.loads(open(sys.argv[1], encoding="utf-8").read())
pm = report.get("primary_metric", {})
ratio = pm.get("ratio_vs_baseline")
ci = pm.get("display_ci") or pm.get("ci")
kind = pm.get("kind", "metric")
if isinstance(ratio, (int, float)) and isinstance(ci, list) and len(ci) == 2:
    print(f"{kind} ratio={ratio:.3f} ci={ci[0]:.3f}-{ci[1]:.3f}")
elif isinstance(ratio, (int, float)):
    print(f"{kind} ratio={ratio:.3f}")
else:
    print(f"{kind} metric=n/a")
PY
}

verify_reason() {
  local verify_json="$1"
  "$PYTHON_BIN" - "$verify_json" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(payload.get("summary", {}).get("reason", "unknown"))
PY
}

print_results_table() {
  "$PYTHON_BIN" - "$RESULTS_TSV" <<'PY'
import csv
import sys
from pathlib import Path

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))
columns = ["journey", "expectation", "status", "verify", "metric", "artifact", "note"]

def clean(value: str) -> str:
    value = str(value or "").replace("|", "\\|")
    return value if value else "-"

print("")
print("GPT-2 User Journey Smoke Results")
print("")
print("| " + " | ".join(columns) + " |")
print("| " + " | ".join("---" for _ in columns) + " |")
for row in rows:
    print("| " + " | ".join(clean(row.get(column, "")) for column in columns) + " |")
print("")
print(f"Summary: {sum(row.get('status') == 'PASS' for row in rows)} passed, "
      f"{sum(row.get('status') == 'SKIP' for row in rows)} skipped, "
      f"{sum(row.get('status') not in {'PASS', 'SKIP'} for row in rows)} failed.")
PY
}

write_final_verdict() {
  local verdict="PASS"
  if [[ "$FAILED_JOURNEYS" -ne 0 ]]; then
    verdict="FAIL"
  fi
  "$PYTHON_BIN" - "$RESULTS_TSV" "$WORK_ROOT/final_verdict.json" "$verdict" <<'PY'
import csv
import json
import sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))
payload = {
    "verdict": sys.argv[3],
    "note": "gpt2 user journey smoke",
    "summary": {
        "total": len(rows),
        "passed": sum(row.get("status") == "PASS" for row in rows),
        "skipped": sum(row.get("status") == "SKIP" for row in rows),
        "failed": sum(row.get("status") not in {"PASS", "SKIP"} for row in rows),
    },
    "journeys": rows,
}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

run_evidence_pack_journey() {
  local journey="$1"
  local eval_report="$2"
  local journey_root="$3"
  local pack_dir="$journey_root/evidence_pack"
  local signing_key="$journey_root/evidence_pack_signing_key.pem"
  local public_key="$journey_root/evidence_pack_signing_key.pub.pem"
  local verdict_json="$journey_root/final_verdict.json"
  local pack_verify_json="$journey_root/evidence-pack-verify.json"

  if [[ "$MODE" == "local" ]]; then
    record_result "$journey/evidence-pack" "container evidence-pack" "SKIP" "-" "-" "$pack_dir" "host mode"
    return 0
  fi

  printf '%s\n' "{\"verdict\":\"PASS\",\"note\":\"$journey gpt2 user journey smoke\"}" > "$verdict_json"

  local rc=0
  set +e
  "${CLI[@]}" advanced evidence-pack keygen "$signing_key" \
    --public-key-out "$public_key" \
    --json
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    record_result "$journey/evidence-pack" "build and verify evidence pack" "FAIL" "keygen_rc=$rc" "-" "$pack_dir" "key generation failed"
    return 0
  fi

  set +e
  "${CLI[@]}" advanced evidence-pack build "$pack_dir" \
    --final-verdict "$verdict_json" \
    --report "$eval_report" \
    --signing-key "$signing_key" \
    --profile "$PROFILE" \
    --json
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    record_result "$journey/evidence-pack" "build and verify evidence pack" "FAIL" "build_rc=$rc" "-" "$pack_dir" "build failed"
    return 0
  fi

  "${CLI[@]}" advanced evidence-pack inspect "$pack_dir" --json
  set +e
  "${CLI[@]}" advanced evidence-pack verify "$pack_dir" --json > "$pack_verify_json"
  rc=$?
  set -e
  cat "$pack_verify_json"
  if [[ "$rc" -ne 0 ]]; then
    record_result "$journey/evidence-pack" "build and verify evidence pack" "FAIL" "verify_rc=$rc" "-" "$pack_dir" "verification failed"
    return 0
  fi

  record_result "$journey/evidence-pack" "build and verify evidence pack" "PASS" "ok" "-" "$pack_dir" "container mode"
}

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
  "${CLI[@]}" evaluate \
    --baseline gpt2 \
    --subject gpt2 \
    --adapter auto \
    --profile "$PROFILE" \
    --preset "$PRESET" \
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
  "${CLI[@]}" verify "$eval_report" "${VERIFY_ARGS[@]}" --json > "$verify_json"
  rc=$?
  set -e
  cat "$verify_json"
  if [[ "$rc" -ne 0 ]]; then
    record_result "$journey/verify" "verify accepts report" "FAIL" "verify_rc=$rc reason=$(verify_reason "$verify_json")" "$(metric_summary "$eval_report")" "$eval_report" "verify failed"
    echo "==== END journey:$journey ===="
    return 0
  fi

  if ! "${CLI[@]}" report validate "$eval_report"; then
    record_result "$journey/report-validate" "schema validates" "FAIL" "ok" "$(metric_summary "$eval_report")" "$eval_report" "schema validation failed"
    echo "==== END journey:$journey ===="
    return 0
  fi

  if ! "${CLI[@]}" report html -i "$eval_report" -o "$export_dir/evaluation.html" --force; then
    record_result "$journey/report-html" "HTML renders" "FAIL" "ok" "$(metric_summary "$eval_report")" "$export_dir/evaluation.html" "HTML render failed"
    echo "==== END journey:$journey ===="
    return 0
  fi

  if ! "${CLI[@]}" report explain --subject-report "$edited_report" --baseline-report "$baseline_report" > "$explain_txt"; then
    record_result "$journey/report-explain" "explain renders" "FAIL" "ok" "$(metric_summary "$eval_report")" "$explain_txt" "explain failed"
    echo "==== END journey:$journey ===="
    return 0
  fi
  cat "$explain_txt"

  record_result "$journey/evaluate-verify-report" "$expectation" "PASS" "ok" "$(metric_summary "$eval_report")" "$export_dir/evaluation.html" "evaluate -> verify -> validate -> html -> explain"
  run_evidence_pack_journey "$journey" "$eval_report" "$journey_root"

  if [[ -z "$FIRST_EVAL_REPORT" ]]; then
    FIRST_EVAL_REPORT="$eval_report"
  fi
  LAST_EVAL_REPORT="$eval_report"
  echo "==== END journey:$journey ===="
}

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

  "$PYTHON_BIN" - "$source_report" "$eval_report" <<'PY'
import json
import sys

source, target = sys.argv[1], sys.argv[2]
report = json.loads(open(source, encoding="utf-8").read())
report.setdefault("primary_metric", {})["display_ci"] = [1.20, 1.30]
report.setdefault("meta", {})["failure_smoke_mutation"] = (
    "display_ci intentionally diverges from exp(ci)"
)
with open(target, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

  local rc=0
  set +e
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

  if ! "${CLI[@]}" report validate "$eval_report"; then
    record_result "$journey/report-validate" "mutated report remains schema-valid" "FAIL" "reason=$reason" "$(metric_summary "$eval_report")" "$eval_report" "schema validation failed"
    echo "==== END journey:$journey ===="
    return 0
  fi

  if ! "${CLI[@]}" report html -i "$eval_report" -o "$export_dir/evaluation.html" --force; then
    record_result "$journey/report-html" "HTML renders rejected report" "FAIL" "reason=$reason" "$(metric_summary "$eval_report")" "$export_dir/evaluation.html" "HTML render failed"
    echo "==== END journey:$journey ===="
    return 0
  fi

  record_result "$journey/verify-rejects" "verify rejects mutated report" "PASS" "verify_rc=$rc reason=$reason" "$(metric_summary "$eval_report")" "$export_dir/evaluation.html" "schema valid; HTML rendered for inspection"
  echo "==== END journey:$journey ===="
}

if [[ "$MODE" == "container" && -z "${INVARLOCK_RUNTIME_IMAGE:-}" ]]; then
  seed_local_runtime_image
fi

if seed_hf_cache_from_host; then
  export INVARLOCK_SMOKE_CACHE_SEEDED=1
elif prefetch_hf_assets_on_host && seed_hf_cache_from_host; then
  export INVARLOCK_SMOKE_CACHE_SEEDED=1
fi

ensure_writable_hf_cache
ensure_current_runtime_image

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
echo "[smoke] hf_home=$HF_HOME"
echo "[smoke] hf_datasets_cache=$HF_DATASETS_CACHE"

IFS=',' read -r -a JOURNEYS <<< "$JOURNEYS_RAW"
for journey in "${JOURNEYS[@]}"; do
  journey="${journey//[[:space:]]/}"
  [[ -n "$journey" ]] || continue
  case "$journey" in
    noop)
      run_eval_journey "noop" "" "baseline gpt2 vs no-op gpt2 should pass"
      ;;
    quantized)
      quant_edit_config="${EDIT_CONFIG:-$DEFAULT_QUANT_EDIT_CONFIG}"
      if [[ ! -f "$quant_edit_config" ]]; then
        record_result "quantized/evaluate" "quantized subject should run" "FAIL" "missing_edit_config" "-" "$quant_edit_config" "edit config missing"
      else
        run_eval_journey "quantized" "$quant_edit_config" "baseline gpt2 vs quantized gpt2 should verify"
      fi
      ;;
    edited|custom-edit)
      if [[ -z "$EDIT_CONFIG" ]]; then
        record_result "$journey/evaluate" "custom edited subject should run" "FAIL" "missing_edit_config" "-" "$WORK_ROOT" "set INVARLOCK_SMOKE_EDIT_CONFIG"
      else
        run_eval_journey "$journey" "$EDIT_CONFIG" "baseline gpt2 vs configured edited subject should verify"
      fi
      ;;
    negative|failure)
      run_negative_journey
      ;;
    *)
      record_result "$journey" "known journey token" "FAIL" "unknown_journey" "-" "$WORK_ROOT" "supported: noop,quantized,edited,negative"
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
