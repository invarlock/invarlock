from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path


def load_script_module(script_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "model_evidence" / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[script_name] = module
    spec.loader.exec_module(module)
    return module


def write_fake_python(path: Path) -> None:
    script = """#!/bin/bash
set -euo pipefail

if [[ -n "${FAKE_PYTHON_LOG:-}" ]]; then
  printf '%s\\n' "$*" >> "$FAKE_PYTHON_LOG"
fi

if [[ "${1:-}" == "-c" ]]; then
  exit 0
fi

if [[ "${1:-}" == "scripts/model_evidence/materialize_vision_text_dataset.py" ]]; then
  output_dir=""
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--output-dir" ]]; then
      output_dir="$2"
      shift 2
      continue
    fi
    shift
  done
  mkdir -p "$output_dir/images"
  printf 'fake-image\\n' > "$output_dir/images/000000-fake.png"
  printf '{"id":"fake","image_path":"images/000000-fake.png","prompt":"what?","answer":"cat","answers":["cat"]}\\n' > "$output_dir/manifest.jsonl"
  printf '{"record_count":1}\\n' > "$output_dir/materialization_summary.json"
  exit 0
fi

if [[ "${1:-}" == "-m" && "${2:-}" == "invarlock" && "${3:-}" == "evaluate" ]]; then
  report_dir=""
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--report-out" ]]; then
      report_dir="$2"
      shift 2
      continue
    fi
    shift
  done
  mkdir -p "$report_dir"
  printf '{"ok": true}\\n' > "$report_dir/evaluation.report.json"
  exit 0
fi

if [[ "${1:-}" == "-m" && "${2:-}" == "invarlock" && "${3:-}" == "verify" ]]; then
  printf '{"status":"fail"}\\n'
  exit 1
fi

printf 'unexpected invocation: %s\\n' "$*" >&2
exit 99
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def write_flaky_fake_python(path: Path) -> None:
    script = """#!/bin/bash
set -euo pipefail

if [[ -n "${FAKE_PYTHON_LOG:-}" ]]; then
  printf '%s\\n' "$*" >> "$FAKE_PYTHON_LOG"
fi

if [[ "${1:-}" == "-c" ]]; then
  exit 0
fi

if [[ "${1:-}" == "scripts/model_evidence/materialize_vision_text_dataset.py" ]]; then
  output_dir=""
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--output-dir" ]]; then
      output_dir="$2"
      shift 2
      continue
    fi
    shift
  done
  mkdir -p "$output_dir/images"
  printf 'fake-image\\n' > "$output_dir/images/000000-fake.png"
  printf '{"id":"fake","image_path":"images/000000-fake.png","prompt":"what?","answer":"cat","answers":["cat"]}\\n' > "$output_dir/manifest.jsonl"
  printf '{"record_count":1}\\n' > "$output_dir/materialization_summary.json"
  exit 0
fi

if [[ "${1:-}" == "-m" && "${2:-}" == "invarlock" && "${3:-}" == "evaluate" ]]; then
  if [[ ! -f "${FAKE_PYTHON_STATE:-}" ]]; then
    : > "${FAKE_PYTHON_STATE:-}"
    kill -TERM $$
  fi
  report_dir=""
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--report-out" ]]; then
      report_dir="$2"
      shift 2
      continue
    fi
    shift
  done
  mkdir -p "$report_dir"
  printf '{"ok": true}\\n' > "$report_dir/evaluation.report.json"
  exit 0
fi

if [[ "${1:-}" == "-m" && "${2:-}" == "invarlock" && "${3:-}" == "verify" ]]; then
  printf '{"status":"ok"}\\n'
  exit 0
fi

printf 'unexpected invocation: %s\\n' "$*" >&2
exit 99
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
