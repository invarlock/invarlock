#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_invarlock_compare.sh --baseline MODEL --subject MODEL_OR_PATH [options]

Required:
  --baseline VALUE             Baseline model ID or local path.
  --subject VALUE              Subject model ID or local path.

Options:
  --report-out DIR             Output directory for evaluation artifacts.
                               Default: reports/integration
  --baseline-adapter NAME      Baseline adapter. Default: auto
  --subject-adapter NAME       Subject adapter. Default: auto
  --profile NAME               InvarLock profile. Default: ci
  --tier NAME                  InvarLock tier. Default: balanced
  --execution-mode MODE        container or host. Default: container
  --assurance MODE             strict or off. Default: strict
  --runtime-provenance MODE    container or host for verify. Defaults to
                               execution mode.
  --device VALUE               Optional device override.
  --preset PATH                Optional InvarLock preset path.
  --edit-label VALUE           Optional edit label for BYOE subjects.
  --allow-network              Allow model/dataset downloads for evaluate.
  --no-html                    Skip HTML rendering.
  -h, --help                   Show this help.

The default path is strict/container-backed. For host-mode exploratory runs,
pass both --execution-mode host and --assurance off.
USAGE
}

baseline=""
subject=""
report_out="reports/integration"
baseline_adapter="auto"
subject_adapter="auto"
profile="ci"
tier="balanced"
execution_mode="container"
assurance="strict"
runtime_provenance=""
device=""
preset=""
edit_label=""
allow_network=0
render_html=1
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
    --report-out)
      report_out="${2:-}"
      shift 2
      ;;
    --baseline-adapter)
      baseline_adapter="${2:-}"
      shift 2
      ;;
    --subject-adapter)
      subject_adapter="${2:-}"
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
    --runtime-provenance)
      runtime_provenance="${2:-}"
      shift 2
      ;;
    --device)
      device="${2:-}"
      shift 2
      ;;
    --preset)
      preset="${2:-}"
      shift 2
      ;;
    --edit-label)
      edit_label="${2:-}"
      shift 2
      ;;
    --allow-network)
      allow_network=1
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

if [[ -z "$baseline" || -z "$subject" ]]; then
  echo "Missing required --baseline or --subject." >&2
  usage >&2
  exit 2
fi

if [[ "$execution_mode" == "host" && "$assurance" == "strict" ]]; then
  echo "Host execution requires --assurance off for this shared wrapper." >&2
  exit 2
fi

if [[ -z "$runtime_provenance" ]]; then
  runtime_provenance="$execution_mode"
fi

mkdir -p "$report_out"

evaluate_cmd=(
  invarlock evaluate
  --baseline "$baseline"
  --subject "$subject"
  --baseline-adapter "$baseline_adapter"
  --subject-adapter "$subject_adapter"
  --profile "$profile"
  --tier "$tier"
  --report-out "$report_out"
  --execution-mode "$execution_mode"
  --assurance "$assurance"
)

if [[ "$allow_network" -eq 1 ]]; then
  evaluate_cmd+=(--allow-network)
fi
if [[ -n "$device" ]]; then
  evaluate_cmd+=(--device "$device")
fi
if [[ -n "$preset" ]]; then
  evaluate_cmd+=(--preset "$preset")
fi
if [[ -n "$edit_label" ]]; then
  evaluate_cmd+=(--edit-label "$edit_label")
fi

{
  printf 'wrapper: '
  printf '%q ' "$0" "${original_args[@]}"
  printf '\n'
  printf 'evaluate: '
  printf '%q ' "${evaluate_cmd[@]}"
  printf '\n'
} > "$report_out/run_command.txt"

"${evaluate_cmd[@]}"

report_json="$report_out/evaluation.report.json"
verify_json="$report_out/verify.json"
html_out="$report_out/evaluation.html"

if [[ ! -s "$report_json" ]]; then
  cat >&2 <<MSG
Evaluate completed but did not write the expected report:
  $report_json

Check --report-out path mapping and the evaluate command recorded in:
  $report_out/run_command.txt
MSG
  exit 1
fi

verify_cmd=(
  invarlock verify
  --json \
  --profile "$profile" \
  --assurance "$assurance" \
  --runtime-provenance "$runtime_provenance" \
  "$report_json"
)

printf 'verify: ' >> "$report_out/run_command.txt"
printf '%q ' "${verify_cmd[@]}" >> "$report_out/run_command.txt"
printf '> %q\n' "$verify_json" >> "$report_out/run_command.txt"

"${verify_cmd[@]}" > "$verify_json"

if [[ "$render_html" -eq 1 ]]; then
  html_cmd=(
    invarlock report html
    -i "$report_json"
    -o "$html_out"
    --force
  )
  printf 'html: ' >> "$report_out/run_command.txt"
  printf '%q ' "${html_cmd[@]}" >> "$report_out/run_command.txt"
  printf '\n' >> "$report_out/run_command.txt"
  "${html_cmd[@]}"
fi

echo "Wrote: $report_json"
echo "Wrote: $verify_json"
if [[ "$render_html" -eq 1 ]]; then
  echo "Wrote: $html_out"
fi
