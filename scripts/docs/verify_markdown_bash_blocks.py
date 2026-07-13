#!/usr/bin/env python3
"""Live-run concrete bash code blocks from Markdown docs.

Blocks are executed in file-scoped temporary workspaces staged from the current
checkout so workflows like:

- build image
- evaluate
- verify
- render report

can run in order without mutating the developer's working tree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

try:
    from scripts.docs import verify_markdown_demo_inputs as _demo_inputs
    from scripts.docs import verify_markdown_rewrite as _rewrite
except ImportError:  # pragma: no cover - direct script execution path
    import verify_markdown_demo_inputs as _demo_inputs
    import verify_markdown_rewrite as _rewrite

ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / "tmp"
EXECUTION_MODES = ("container", "host")
HOST_EXECUTION_ENV = _rewrite.HOST_EXECUTION_ENV
MODEL_LOADING_COMMANDS = _rewrite.MODEL_LOADING_COMMANDS
DEFAULT_EVALUATE_SMOKE_PRESET = _rewrite.DEFAULT_EVALUATE_SMOKE_PRESET
SMOKE_MODEL_ID_MAP = _rewrite.SMOKE_MODEL_ID_MAP
SMOKE_PATH_MAP = _rewrite.SMOKE_PATH_MAP
SMOKE_SCRIPT_REWRITES = _rewrite.SMOKE_SCRIPT_REWRITES
DEMO_EVALUATION_REPORT_FIXTURE = _demo_inputs.DEMO_EVALUATION_REPORT_FIXTURE
DEMO_RUNTIME_MANIFEST_FIXTURE = _demo_inputs.DEMO_RUNTIME_MANIFEST_FIXTURE

WORKSPACE_STAGE_DIRS = {
    ".github",
    "configs",
    "contracts",
    "docs",
    "public_evidence",
    "requirements",
    "runtime",
    "scripts",
    "src",
    "tests",
}
WORKSPACE_STAGE_FILES = {
    ".dockerignore",
    ".editorconfig",
    ".gitignore",
    ".markdownlint.json",
    ".markdownlintignore",
    ".python-version",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "MANIFEST.in",
    "Makefile",
    "README.md",
    "SECURITY.md",
    "SUPPORT.md",
    "mkdocs.yml",
    "package-lock.json",
    "package.json",
    "pyproject.toml",
    "uv.lock",
}
EXCLUDE_TOP_LEVEL_DIRS = {
    "build",
    ".evaluate_tmp",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-release",
    "node_modules",
    "reports",
    "runs",
    "site",
    "tmp",
    "venv",
}

ANGLE_PLACEHOLDER_PATTERN = re.compile(r"<[^>]+>")
RUN_ID_PLACEHOLDER_PATTERN = re.compile(r"\bruns/\d{8}_\d{6}\b")
INVARLOCK_COMMAND_PATTERN = re.compile(
    r"^(?:[A-Z_][A-Z0-9_]*=[^\s]+\s+)*(?:invarlock\s+|python\s+-m\s+invarlock(?:\.[^\s]+)?\s+).*$"
)

SKIP_TOKENS = (
    "$CONFIG_FILE",
    "...",
    "…",
    "config.yaml",
    "custom_format",
    "make dev-install",
    "my_plugin",
    "my_config.yaml",
    "runs/latest",
    "/path/to/",
    "/absolute/path/to/",
    "<BASELINE_MODEL>",
    "<SUBJECT_MODEL>",
    "<model_or_id>",
    "<edited_model_or_dir>",
    "<source>",
    "<edited>",
    "<ts>",
    "<hf_dir_or_id>",
    "<report.json>",
    "<out.html>",
    "<edited_report.json>",
    "<baseline_report.json>",
)


@dataclass(frozen=True)
class BashBlock:
    file: str
    line: int
    block_index: int
    text: str


def _strip_prompt(s: str) -> str:
    s = s.lstrip()
    return s[2:] if s.startswith("$ ") else s


def _should_skip_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if ANGLE_PLACEHOLDER_PATTERN.search(stripped):
        return True
    if RUN_ID_PLACEHOLDER_PATTERN.search(stripped):
        return True
    return any(token in stripped for token in SKIP_TOKENS)


def _expects_failure(text: str) -> bool:
    return any(
        line.strip().lower().startswith("# docs-live: expect-failure")
        for line in text.splitlines()
    )


def _expected_failure_output(text: str) -> str | None:
    prefix = "# docs-live: expect-failure:"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(prefix):
            return stripped[len(prefix) :].strip() or None
    return None


def _contains_invarlock_command(text: str) -> bool:
    for raw in text.splitlines():
        line = _strip_prompt(raw.strip())
        if not line or line.startswith("#"):
            continue
        if INVARLOCK_COMMAND_PATTERN.match(line):
            return True
    return False


def extract_bash_blocks(paths: list[Path]) -> list[BashBlock]:
    blocks: list[BashBlock] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        in_fence = False
        block_lines: list[str] = []
        start_line = 0
        language = ""
        block_index = 0
        for line_no, line in enumerate(lines, start=1):
            if line.startswith("```"):
                if in_fence:
                    if language == "bash":
                        block_text = "\n".join(block_lines).strip()
                        if block_text and _contains_invarlock_command(block_text):
                            block_index += 1
                            blocks.append(
                                BashBlock(
                                    file=str(path),
                                    line=start_line,
                                    block_index=block_index,
                                    text=block_text,
                                )
                            )
                    in_fence = False
                    block_lines = []
                    language = ""
                else:
                    in_fence = True
                    start_line = line_no + 1
                    info = line[3:].strip().split()
                    language = info[0] if info else ""
                continue
            if in_fence:
                block_lines.append(line)
    return blocks


def iter_markdown_files(root: Path, *, paths: list[str] | None = None) -> list[Path]:
    if paths:
        candidates: set[Path] = set()
        for item in paths:
            path = (Path(item) if Path(item).is_absolute() else (root / item)).resolve()
            if path.is_dir():
                candidates.update(
                    p.resolve() for p in path.rglob("*.md") if p.is_file()
                )
            elif path.is_file() and path.suffix.lower() == ".md":
                candidates.add(path)
        return sorted(candidates)

    md_files: list[Path] = []
    for path in root.glob("**/*.md"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if rel_parts and (
            rel_parts[0].startswith(".") or rel_parts[0] in EXCLUDE_TOP_LEVEL_DIRS
        ):
            continue
        md_files.append(path)
    return sorted(md_files, key=lambda p: str(p))


def _should_stage_workspace_entry(path: Path) -> bool:
    name = path.name
    if name in EXCLUDE_TOP_LEVEL_DIRS:
        return False
    if path.is_dir():
        return name in WORKSPACE_STAGE_DIRS
    return name in WORKSPACE_STAGE_FILES


def _stage_workspace_entry(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, symlinks=True)
        return
    shutil.copy2(source, target)


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink():
        path.unlink()
        return

    last_error: OSError | None = None
    for _ in range(3):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)

    if last_error is not None:
        raise last_error


def _prepare_workspace(workspace: Path) -> None:
    if workspace.exists():
        _remove_tree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    for source in sorted(ROOT.iterdir(), key=lambda path: path.name):
        if not _should_stage_workspace_entry(source):
            continue
        _stage_workspace_entry(source, workspace / source.name)


def _split_env_prefix(tokens: list[str]) -> tuple[list[str], list[str]]:
    return _rewrite._split_env_prefix(tokens)


def _command_tokens(argv: list[str]) -> list[str]:
    return _rewrite._command_tokens(argv)


def _is_evaluate_command(command_tokens: list[str]) -> bool:
    return _rewrite._is_evaluate_command(command_tokens)


def _is_verify_command(command_tokens: list[str]) -> bool:
    return _rewrite._is_verify_command(command_tokens)


def _is_model_loading_command(command_tokens: list[str]) -> bool:
    return _rewrite._is_model_loading_command(command_tokens)


def _is_optional_environment_command(command_tokens: list[str]) -> bool:
    return _rewrite._is_optional_environment_command(command_tokens)


def _should_skip_line_for_host_mode(stripped: str) -> bool:
    return _rewrite._should_skip_line_for_host_mode(
        stripped,
        host_supports_mps=_host_supports_mps,
    )


def _host_supports_mps() -> bool:
    return _rewrite._host_supports_mps()


def _rewrite_model_loading_tokens_for_live_smoke(argv: list[str]) -> list[str]:
    return _rewrite._rewrite_model_loading_tokens_for_live_smoke(argv)


def _rewrite_live_smoke_script_text(text: str) -> str:
    return _rewrite._rewrite_live_smoke_script_text(text)


def _insert_option_after_command(argv: list[str], option: str) -> list[str]:
    return _rewrite._insert_option_after_command(argv, option)


def _insert_tokens_after_command(argv: list[str], tokens: list[str]) -> list[str]:
    return _rewrite._insert_tokens_after_command(argv, tokens)


def _rewrite_invarlock_tokens(
    *,
    env_prefix: list[str],
    argv: list[str],
    execution_mode: str,
) -> tuple[list[str], list[str]]:
    return _rewrite._rewrite_invarlock_tokens(
        env_prefix=env_prefix,
        argv=argv,
        execution_mode=execution_mode,
    )


def _sanitize_script(
    block: BashBlock,
    *,
    execution_mode: str = "container",
    skip_model_loading: bool = False,
) -> str:
    return _rewrite._sanitize_script(
        block,
        execution_mode=execution_mode,
        skip_model_loading=skip_model_loading,
        host_supports_mps=_host_supports_mps,
        root=ROOT,
    )


def _default_env(workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(workspace / "src")
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    env.setdefault("INVARLOCK_ALLOW_NETWORK", "1")
    env.setdefault("INVARLOCK_DEDUP_TEXTS", "1")
    env.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    fixture_manifest = (
        workspace
        / "tests"
        / "fixtures"
        / "runtime_provenance"
        / "runtime.manifest.json"
    )
    try:
        fixture_payload = json.loads(fixture_manifest.read_text(encoding="utf-8"))
        runtime = fixture_payload.get("runtime")
        image_digest = (
            runtime.get("image_digest") if isinstance(runtime, dict) else None
        )
    except (OSError, json.JSONDecodeError):
        image_digest = None
    if isinstance(image_digest, str) and image_digest:
        # Markdown replay uses a staged fixture report. Treat its fixture digest
        # as the independently supplied test pin; never infer the pin from the
        # report-side manifest generated during the replay itself.
        env.setdefault("EXPECTED_RUNTIME_IMAGE_DIGEST", image_digest)
        env.setdefault("TRUSTED_RUNTIME_IMAGE_DIGEST", image_digest)
    return env


def _sync_demo_input_paths() -> None:
    _demo_inputs.ROOT = ROOT
    _demo_inputs.DEMO_EVALUATION_REPORT_FIXTURE = DEMO_EVALUATION_REPORT_FIXTURE
    _demo_inputs.DEMO_RUNTIME_MANIFEST_FIXTURE = DEMO_RUNTIME_MANIFEST_FIXTURE


def _write_json(path: Path, payload: object) -> None:
    _sync_demo_input_paths()
    _demo_inputs._write_json(path, payload)


def _write_runtime_manifest_for_report(report_path: Path) -> None:
    _sync_demo_input_paths()
    _demo_inputs._write_runtime_manifest_for_report(report_path)


def _run_logged_script(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    label: str,
) -> tuple[int, str]:
    print(f"[markdown-live] Running {label}", flush=True)
    output_tail = ""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
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
            output_tail = (output_tail + line)[-4000:]
        returncode = process.wait()
    print(f"[markdown-live] Finished {label} rc={returncode}", flush=True)
    return returncode, output_tail


def _build_demo_evaluation_report(
    run_report: dict[str, object],
    baseline_report: dict[str, object],
) -> dict[str, object]:
    _sync_demo_input_paths()
    return _demo_inputs._build_demo_evaluation_report(run_report, baseline_report)


def _demo_window_summary(section: dict[str, object]) -> tuple[float, float, int] | None:
    return _demo_inputs._demo_window_summary(section)


def _seed_demo_inputs(workspace: Path, *, fixture_mode: bool = False) -> None:
    _sync_demo_input_paths()
    _demo_inputs._seed_demo_inputs(workspace, fixture_mode=fixture_mode)


def run_blocks(
    blocks: list[BashBlock],
    *,
    output_root: Path,
    execution_mode: str = "container",
    skip_model_loading: bool = False,
) -> int:
    if output_root.exists():
        _remove_tree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / "results.jsonl"
    workspace_root = output_root / "workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

    grouped: dict[str, list[BashBlock]] = {}
    for block in blocks:
        grouped.setdefault(block.file, []).append(block)

    with results_path.open("w", encoding="utf-8") as out:
        for file_index, (file_path, file_blocks) in enumerate(
            sorted(grouped.items()), start=1
        ):
            workspace = (
                workspace_root / f"{file_index:03d}_{Path(file_path).stem}_{run_stamp}"
            )
            _prepare_workspace(workspace)
            _seed_demo_inputs(workspace)
            env = _default_env(workspace)
            for block in file_blocks:
                block_id = f"{file_index:03d}-{block.block_index:02d}"
                script_path = workspace / f".docs_live_{block_id}.sh"
                log_dir = output_root / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{block_id}.log"
                if _should_skip_block(block.text):
                    record = {
                        "id": block_id,
                        "file": block.file,
                        "line": block.line,
                        "status": "skipped-template",
                        "log_path": str(log_path),
                    }
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    continue

                script_path.write_text(
                    _sanitize_script(
                        block,
                        execution_mode=execution_mode,
                        skip_model_loading=skip_model_loading,
                    ),
                    encoding="utf-8",
                )
                expects_failure = _expects_failure(block.text)
                expected_failure_output = _expected_failure_output(block.text)
                returncode, output_tail = _run_logged_script(
                    cmd=[
                        "bash",
                        "-uo" if expects_failure else "-euo",
                        "pipefail",
                        str(script_path.name),
                    ],
                    cwd=workspace,
                    env=env,
                    log_path=log_path,
                    label=f"{block_id} {Path(block.file).name}:{block.line}",
                )
                passed = (
                    returncode != 0
                    and (
                        expected_failure_output is None
                        or expected_failure_output in output_tail
                    )
                    if expects_failure
                    else returncode == 0
                )
                record = {
                    "id": block_id,
                    "file": block.file,
                    "line": block.line,
                    "execution_mode": execution_mode,
                    "status": "ok" if passed else "failed",
                    "exit_code": int(returncode),
                    "expected_failure": expects_failure,
                    "expected_failure_output": expected_failure_output,
                    "log_path": str(log_path),
                    "stdout": output_tail,
                    "stderr": "",
                }
                out.write(json.dumps(record) + "\n")
                out.flush()

    failures = 0
    with results_path.open(encoding="utf-8") as results_file:
        for raw in results_file:
            if not raw.strip():
                continue
            record = json.loads(raw)
            if record.get("status") == "failed":
                failures += 1
    print(f"Verified {len(blocks)} bash block(s) → {results_path}")
    if failures:
        print(f"Markdown bash block failures: {failures}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Markdown files or directories to scan (default: repo-wide).",
    )
    parser.add_argument(
        "--output-root",
        default=str(TMP / "markdown_live"),
        help="Output directory for logs, workspaces, and result JSONL.",
    )
    parser.add_argument(
        "--execution-mode",
        default="container",
        choices=EXECUTION_MODES,
        help=(
            "Replay markdown commands as default runtime container commands or "
            "as explicit host commands."
        ),
    )
    parser.add_argument(
        "--skip-model-loading",
        action="store_true",
        help=(
            "Skip model-loading commands (`evaluate`, `run`, `calibrate`) while "
            "still replaying later verify/report steps against seeded demo data."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    md_files = iter_markdown_files(ROOT, paths=args.paths)
    blocks = extract_bash_blocks(md_files)
    return run_blocks(
        blocks,
        output_root=Path(args.output_root).expanduser().resolve(),
        execution_mode=args.execution_mode,
        skip_model_loading=args.skip_model_loading,
    )


if __name__ == "__main__":
    raise SystemExit(main())
