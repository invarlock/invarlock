#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <run_dir> <baseline_dir>" >&2
  exit 2
fi

RUN_DIR="$1"
BASELINE_DIR="$2"

invarlock report --run "$RUN_DIR" --baseline "$BASELINE_DIR/report.json" --format cert
