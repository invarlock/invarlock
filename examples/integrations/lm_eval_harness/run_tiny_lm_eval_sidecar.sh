#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_tiny_lm_eval_sidecar.sh [options]

Run a tiny LM Evaluation Harness task and normalize its JSON as an InvarLock
sidecar artifact. An optional subject run can be included for baseline-vs-subject
task-metric deltas.

Options:
  --baseline VALUE             Baseline model ID or local path.
                               Default: sshleifer/tiny-gpt2
  --subject VALUE              Optional subject model ID or local path.
  --tasks VALUE                LM Eval task list. Default: wikitext
  --limit VALUE                LM Eval example limit. Default: 1
  --device VALUE               LM Eval device. Default: cpu
  --batch-size VALUE           LM Eval batch size. Default: 1
  --dtype VALUE                Hugging Face model dtype. Default: float32
  --report-out DIR             Output directory for sidecar artifacts.
                               Default: examples/integrations/lm_eval_harness/reports/tiny-lm-eval-sidecar
  --allow-network              Allow model and dataset downloads.
  --force                      Replace an existing report directory.
  -h, --help                   Show this help.

The default command is a smoke run. Use a higher --limit only when you want
meaningful task metrics and have enough local runtime budget.
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

baseline="sshleifer/tiny-gpt2"
subject=""
tasks="wikitext"
limit="1"
device="cpu"
batch_size="1"
dtype="float32"
report_out="$SCRIPT_DIR/reports/tiny-lm-eval-sidecar"
allow_network=0
force=0
original_args=("$@")

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
    --tasks)
      tasks="${2:-}"
      shift 2
      ;;
    --limit)
      limit="${2:-}"
      shift 2
      ;;
    --device)
      device="${2:-}"
      shift 2
      ;;
    --batch-size)
      batch_size="${2:-}"
      shift 2
      ;;
    --dtype)
      dtype="${2:-}"
      shift 2
      ;;
    --report-out)
      report_out="${2:-}"
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

if [[ -z "$baseline" || -z "$tasks" || -z "$limit" || -z "$device" || -z "$batch_size" || -z "$dtype" || -z "$report_out" ]]; then
  echo "Missing required baseline, task, runtime, or report setting." >&2
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

if ! "$PYTHON_BIN" -c 'import lm_eval' >/dev/null 2>&1; then
  cat >&2 <<'MSG'
Missing example dependency: lm_eval

Install LM Evaluation Harness in the environment used for this example:
  python -m pip install "lm_eval[hf]"

The core InvarLock install intentionally does not require LM Evaluation Harness.
MSG
  exit 2
fi

if [[ -e "$report_out" ]]; then
  if [[ "$force" -ne 1 ]]; then
    echo "Report directory already exists: $report_out" >&2
    echo "Pass --force to replace it." >&2
    exit 2
  fi
  rm -rf "$report_out"
fi
mkdir -p "$report_out"

if [[ "$allow_network" -eq 0 ]]; then
  export HF_DATASETS_OFFLINE=1
  export HF_HUB_OFFLINE=1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

model_args_for() {
  local model_ref="$1"
  local model_args="pretrained=$model_ref,dtype=$dtype"
  if [[ "$allow_network" -eq 0 ]]; then
    model_args="$model_args,local_files_only=True"
  fi
  printf '%s\n' "$model_args"
}

append_command_log() {
  local label="$1"
  shift

  {
    printf '%s:\n' "$label"
    printf '  '
    printf '%q ' "$@"
    printf '\n\n'
  } >> "$report_out/run_command.txt"
}

run_lm_eval() {
  local label="$1"
  local model_ref="$2"
  local output_dir="$3"
  local model_args
  model_args="$(model_args_for "$model_ref")"

  mkdir -p "$output_dir"
  local cmd=(
    "$PYTHON_BIN"
    -m lm_eval
    run
    --model hf
    --model_args "$model_args"
    --tasks "$tasks"
    --limit "$limit"
    --device "$device"
    --batch_size "$batch_size"
    --output_path "$output_dir"
  )

  append_command_log "$label" "${cmd[@]}"
  "${cmd[@]}"
}

find_result_json() {
  local output_dir="$1"
  local count=0
  local selected=""
  local path

  while IFS= read -r path; do
    count=$((count + 1))
    selected="$path"
  done < <(find "$output_dir" -type f -name 'results_*.json' | sort)

  if [[ "$count" -ne 1 ]]; then
    echo "Expected exactly one LM Eval results JSON under $output_dir, found $count." >&2
    exit 1
  fi

  printf '%s\n' "$selected"
}

{
  printf 'runner_invocation:\n  '
  printf '%q ' "$0" "${original_args[@]}"
  printf '\n\n'
} > "$report_out/run_command.txt"

baseline_dir="$report_out/baseline"
run_lm_eval "lm_eval_baseline" "$baseline" "$baseline_dir"
baseline_json="$(find_result_json "$baseline_dir")"

subject_json=""
if [[ -n "$subject" ]]; then
  subject_dir="$report_out/subject"
  run_lm_eval "lm_eval_subject" "$subject" "$subject_dir"
  subject_json="$(find_result_json "$subject_dir")"
fi

normalize_cmd=(
  "$PYTHON_BIN"
  "$SCRIPT_DIR/normalize_lm_eval_results.py"
  --baseline-json "$baseline_json"
  --baseline-model "$baseline"
  --tasks "$tasks"
  --limit "$limit"
  --device "$device"
  --command-log "$report_out/run_command.txt"
  --output "$report_out/lm_eval_sidecar_summary.json"
)

if [[ -n "$subject_json" ]]; then
  normalize_cmd+=(--subject-json "$subject_json" --subject-model "$subject")
fi

append_command_log "normalize_sidecar" "${normalize_cmd[@]}"
"${normalize_cmd[@]}"
