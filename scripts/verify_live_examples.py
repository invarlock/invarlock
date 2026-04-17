#!/usr/bin/env python3
"""Live-verify runnable documentation examples and notebooks.

This orchestrates the two runnable documentation surfaces that materially drift
in practice:

- Markdown CLI snippets, executed through `verify_markdown_bash_blocks.py`
- Jupyter notebooks, executed through `verify_notebooks_smoke.py`

Outputs are written under `tmp/live_examples/` by default so CI and local runs
can archive the exact stdout/stderr that backed a documentation check.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "tmp" / "live_examples"
DEFAULT_NOTEBOOK_PATHS = (
    "notebooks/invarlock_compare_evaluate.ipynb",
    "notebooks/invarlock_custom_datasets.ipynb",
    "notebooks/invarlock_evaluation_report_deep_dive.ipynb",
    "notebooks/invarlock_policy_tiers.ipynb",
    "notebooks/invarlock_python_api.ipynb",
    "notebooks/invarlock_quickstart_cpu.ipynb",
)


def _default_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(ROOT / "src")
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    env.setdefault("INVARLOCK_ALLOW_NETWORK", "1")
    env.setdefault("INVARLOCK_DEDUP_TEXTS", "1")
    env.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _collect_paths(paths: list[str] | None, *, suffixes: set[str]) -> list[Path]:
    if not paths:
        return []

    resolved: set[Path] = set()
    for item in paths:
        path = (Path(item) if Path(item).is_absolute() else (ROOT / item)).resolve()
        if path.is_dir():
            for suffix in suffixes:
                resolved.update(
                    p.resolve() for p in path.rglob(f"*{suffix}") if p.is_file()
                )
        elif path.is_file() and path.suffix.lower() in suffixes:
            resolved.add(path)
    return sorted(resolved)


def _resolve_markdown_paths(paths: list[str] | None) -> list[str]:
    markdown = _collect_paths(paths, suffixes={".md"})
    if not markdown:
        return []
    return [str(path.relative_to(ROOT)) for path in markdown]


def _resolve_notebook_paths(paths: list[str] | None) -> list[str]:
    notebooks = _collect_paths(paths, suffixes={".ipynb"})
    if notebooks:
        return [str(path.relative_to(ROOT)) for path in notebooks]
    if paths:
        return []
    return list(DEFAULT_NOTEBOOK_PATHS)


def _run_subprocess(
    cmd: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
) -> dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[live] Running: {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            print(line, end="")
        returncode = process.wait()
    print(f"[live] Finished rc={returncode}: {' '.join(cmd)}", flush=True)
    return {
        "command": cmd,
        "returncode": int(returncode),
        "log_path": str(log_path.relative_to(ROOT)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Optional docs/notebook files or directories to verify.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory for verification artifacts (default: tmp/live_examples).",
    )
    parser.add_argument(
        "--skip-markdown",
        action="store_true",
        help="Skip markdown CLI snippet execution.",
    )
    parser.add_argument(
        "--markdown-execution-mode",
        default="container",
        choices=("container", "host"),
        help=(
            "Execution mode forwarded to verify_markdown_bash_blocks.py for "
            "markdown command replay."
        ),
    )
    parser.add_argument(
        "--skip-markdown-model-loading",
        action="store_true",
        help=(
            "Skip markdown model-loading commands (`evaluate`, `run`, `calibrate`) "
            "and rely on seeded demo evidence for later verify/report replay."
        ),
    )
    parser.add_argument(
        "--skip-notebook-model-loading",
        action="store_true",
        help=(
            "Skip heavyweight notebook model-loading cells and rely on seeded demo "
            "evidence for later verify/report replay."
        ),
    )
    parser.add_argument(
        "--skip-notebooks",
        action="store_true",
        help="Skip notebook execution.",
    )
    parser.add_argument(
        "--notebook-timeout-s",
        type=int,
        default=3600,
        help="Per-notebook timeout in seconds.",
    )
    parser.add_argument(
        "--run-notebook-pip",
        action="store_true",
        help="Allow notebooks to execute embedded `pip install ...` commands.",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Write the summary even when checks fail, but do not exit non-zero.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    env = _default_env()

    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "output_root": str(output_root),
        "markdown": None,
        "notebooks": None,
    }
    failures: list[str] = []

    if not args.skip_markdown:
        markdown_paths = _resolve_markdown_paths(args.paths)
        markdown_cmd = [
            sys.executable,
            "scripts/verify_markdown_bash_blocks.py",
            "--output-root",
            str(output_root / "markdown"),
            "--execution-mode",
            args.markdown_execution_mode,
        ]
        if markdown_paths:
            markdown_cmd.extend(["--paths", *markdown_paths])
        if args.skip_markdown_model_loading:
            markdown_cmd.append("--skip-model-loading")
        markdown_result = _run_subprocess(
            markdown_cmd,
            env=env,
            log_path=output_root / "markdown" / "run.log",
        )
        summary["markdown"] = markdown_result
        if markdown_result["returncode"] != 0:
            failures.append("markdown")

    if not args.skip_notebooks:
        notebook_paths = _resolve_notebook_paths(args.paths)
        notebook_cmd = [
            sys.executable,
            "scripts/verify_notebooks_smoke.py",
            "--out-root",
            str(output_root / "notebooks"),
            "--timeout-s",
            str(args.notebook_timeout_s),
        ]
        if args.run_notebook_pip:
            notebook_cmd.append("--run-pip")
        if args.skip_notebook_model_loading:
            notebook_cmd.append("--skip-model-loading")
        notebook_cmd.extend(notebook_paths)
        notebook_result = _run_subprocess(
            notebook_cmd,
            env=env,
            log_path=output_root / "notebooks" / "run.log",
        )
        summary["notebooks"] = notebook_result
        if notebook_result["returncode"] != 0:
            failures.append("notebooks")

    summary["ok"] = not failures
    summary["failures"] = failures
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote live-example summary → {summary_path}")
    if failures:
        print(f"Live verification failures: {', '.join(failures)}", file=sys.stderr)
        if not args.allow_errors:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
