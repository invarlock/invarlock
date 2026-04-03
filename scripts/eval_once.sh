#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <run_dir> <baseline_dir>" >&2
  exit 2
fi

RUN_DIR="$1"
BASELINE_DIR="$2"

invarlock report generate \
  --run "$RUN_DIR" \
  --baseline-run-report "$BASELINE_DIR/report.json" \
  --format report
