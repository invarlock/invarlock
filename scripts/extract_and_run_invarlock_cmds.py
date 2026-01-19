#!/usr/bin/env python3
"""
Extract all invarlock command examples from Markdown files and run them.

Outputs:
  - tmp/invarlock_commands.tsv               (tab-separated: id, file, line, command)
  - tmp/invarlock_command_results.jsonl      (per-command result with exit code and output snippets)

Heuristics:
  - Detect lines starting with optional env assignments and `invarlock ...` or `python -m invarlock ...`.
  - Join continuation lines ending with '\\'.
  - Strip '$ ' prompt prefixes when present.
  - Deduplicate identical commands while preserving the first location.
  - When running, apply a cautious env unless the command already sets it:
      INVARLOCK_ALLOW_NETWORK=1, INVARLOCK_DEDUP_TEXTS=1, TRANSFORMERS_NO_TORCHVISION=1,
      TOKENIZERS_PARALLELISM=false
  - Timeout per command: 180s by default; 60s for doctor/help,
    300s for run/certify, 120s for report/html/verify.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TMP = ROOT / "tmp"
TMP.mkdir(parents=True, exist_ok=True)

EXCLUDE_TOP_LEVEL_DIRS = {
    # Internal planning docs may include placeholders or future APIs.
    "plans",
    # Generated/artifact dirs.
    "tmp",
    "runs",
    "reports",
    ".certify_tmp",
    # Tooling caches / VCS.
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
}

CMD_PATTERN = re.compile(
    r"^(?P<prefix>(?:[A-Z_]+=[^\s]+\s+)*)\s*(?P<cmd>(?:invarlock\s+|python\s+-m\s+invarlock\s+).*)$"
)
ANGLE_PLACEHOLDER_PATTERN = re.compile(r"<[^>]+>")
RUN_ID_PLACEHOLDER_PATTERN = re.compile(r"\bruns/\d{8}_\d{6}\b")


def _is_code_fence(line: str) -> bool:
    return line.strip().startswith("```")


def _strip_prompt(s: str) -> str:
    s = s.lstrip()
    if s.startswith("$ "):
        return s[2:]
    return s


@dataclass
class Command:
    id: int
    file: str
    line: int
    cmd: str


SKIP_TOKENS = [
    # placeholders / templates
    "config.yaml",
    "my_config.yaml",
    "$CONFIG_FILE",
    "...",
    "…",
    "runs/latest",
    "<BASELINE_MODEL>",
    "<SUBJECT_MODEL>",
    "<model_or_id>",
    "<edited_model_or_dir>",
    "<source>",
    "<edited>",
    "<ts>",
    "<hf_dir_or_id>",
    "<cert.json>",
    "<out.html>",
    "<edited_report.json>",
    "<baseline_report.json>",
    "/path/to/",
    "/absolute/path/to/",
]


def _should_skip(cmd: str) -> bool:
    # allow disabling skip via env
    if os.getenv("SKIP_PLACEHOLDERS", "1") not in {"1", "true", "True"}:
        return False
    s = cmd.strip()
    if ANGLE_PLACEHOLDER_PATTERN.search(s):
        return True
    if RUN_ID_PLACEHOLDER_PATTERN.search(s):
        return True
    # skip obvious placeholders
    for tok in SKIP_TOKENS:
        if tok in s:
            return True
    return False


def extract_commands(paths: Iterable[Path]) -> list[tuple[str, int, str]]:
    results: list[tuple[str, int, str]] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        in_fence = False
        i = 0
        while i < len(lines):
            line = lines[i]
            if _is_code_fence(line):
                in_fence = not in_fence
                i += 1
                continue
            # Only consider inside code-fences or lines that look like shell commands
            consider = in_fence or (
                "invarlock" in line
                and (
                    line.lstrip().startswith("invarlock ")
                    or line.lstrip().startswith("python -m invarlock")
                )
            )
            if not consider:
                i += 1
                continue
            text = _strip_prompt(line.rstrip())
            m = CMD_PATTERN.match(text)
            if not m:
                i += 1
                continue
            # Join shell continuations ending with '\\' on the command line itself.
            continued = text.rstrip().endswith("\\")
            cmd = text.rstrip("\\").rstrip()
            j = i + 1
            if continued:
                while j < len(lines):
                    cont_raw = _strip_prompt(lines[j].rstrip())
                    if _is_code_fence(cont_raw):
                        break
                    cont = cont_raw.strip()
                    if not cont:
                        j += 1
                        continue
                    # Stop if the next line looks like a new command start.
                    if CMD_PATTERN.match(cont) and not cont.startswith("--"):
                        break
                    cmd += " " + cont_raw.rstrip("\\").strip()
                    continued = cont_raw.rstrip().endswith("\\")
                    j += 1
                    if not continued:
                        break
            # Skip placeholder-heavy and known invalid examples to keep audit focused
            if not _should_skip(cmd):
                results.append((str(path), i + 1, cmd))
            i = j

    # Deduplicate by command string, keeping earliest occurrence
    seen: set[str] = set()
    deduped: list[tuple[str, int, str]] = []
    for file, line, cmd in results:
        key = cmd.strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((file, line, cmd))
    return deduped


def write_commands(
    commands: list[tuple[str, int, str]], out_path: Path
) -> list[Command]:
    out: list[Command] = []
    with out_path.open("w", encoding="utf-8") as fh:
        for idx, (file, line, cmd) in enumerate(commands, start=1):
            out.append(Command(id=idx, file=file, line=line, cmd=cmd))
            fh.write(f"{idx}\t{file}\t{line}\t{cmd}\n")
    return out


def _env_for(cmd_str: str) -> dict:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(SRC))
    # Respect explicit env on the command itself (very light heuristic)
    if "INVARLOCK_ALLOW_NETWORK" not in cmd_str:
        env["INVARLOCK_ALLOW_NETWORK"] = "1"
    env.setdefault("INVARLOCK_DEDUP_TEXTS", "1")
    env.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _timeout_for(cmd_str: str) -> int:
    s = cmd_str
    if (
        " invarlock doctor" in s
        or s.strip().endswith(" --help")
        or "python -m invarlock --help" in s
    ):
        return 60
    if " invarlock calibrate" in s or s.strip().startswith("invarlock calibrate"):
        return 900
    if " invarlock run" in s or " invarlock certify" in s:
        return 600
    if " invarlock report" in s:
        return 120
    return 180


def _force_local_python(cmd_str: str) -> str:
    """Rewrite `invarlock ...` and `python -m invarlock ...` to use this interpreter.

    This ensures the command audit runs against the current checkout via
    `PYTHONPATH=src`.
    """
    m = CMD_PATTERN.match(cmd_str.strip())
    if not m:
        return cmd_str
    prefix = m.group("prefix") or ""
    cmd = (m.group("cmd") or "").strip()
    py = shlex.quote(sys.executable)
    if cmd.startswith("invarlock "):
        return f"{prefix}{py} -m invarlock {cmd[len('invarlock '):]}"
    if cmd.startswith("python -m invarlock"):
        rest = cmd[len("python -m invarlock") :].lstrip()
        return f"{prefix}{py} -m invarlock {rest}".rstrip()
    return cmd_str


def run_commands(commands: list[Command], results_path: Path) -> None:
    # Optional limit via env to keep CI/dev fast
    try:
        limit = int(os.environ.get("CMDS_LIMIT", "0"))
    except Exception:
        limit = 0
    try:
        start = int(os.environ.get("CMDS_OFFSET", "0"))
    except Exception:
        start = 0
    if start or limit:
        end = start + limit if limit and limit > 0 else None
        commands = commands[start:end]
    with results_path.open("w", encoding="utf-8") as out:
        for c in commands:
            # Build execution string; honor inline env assignments
            cmd_str = _force_local_python(c.cmd.strip())
            env = _env_for(cmd_str)
            # Execute via shell to support inline env assignments
            try:
                proc = subprocess.run(
                    cmd_str,
                    shell=True,
                    cwd=str(ROOT),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=_timeout_for(cmd_str),
                )
                record = {
                    "id": c.id,
                    "file": c.file,
                    "line": c.line,
                    "command": cmd_str,
                    "exit_code": proc.returncode,
                    "stdout": proc.stdout[-4000:],
                    "stderr": proc.stderr[-4000:],
                }
            except subprocess.TimeoutExpired as te:
                record = {
                    "id": c.id,
                    "file": c.file,
                    "line": c.line,
                    "command": cmd_str,
                    "exit_code": None,
                    "error": f"timeout after {te.timeout}s",
                    "stdout": te.stdout[-4000:] if isinstance(te.stdout, str) else "",
                    "stderr": te.stderr[-4000:] if isinstance(te.stderr, str) else "",
                }
            except Exception as e:
                record = {
                    "id": c.id,
                    "file": c.file,
                    "line": c.line,
                    "command": cmd_str,
                    "exit_code": None,
                    "error": str(e),
                }
            out.write(json.dumps(record) + "\n")
            out.flush()


def main() -> int:
    md_files: list[Path] = []
    for path in ROOT.glob("**/*.md"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(ROOT).parts
        if rel_parts and rel_parts[0] in EXCLUDE_TOP_LEVEL_DIRS:
            continue
        md_files.append(path)
    # Prefer deterministic ordering
    md_files.sort(key=lambda p: str(p))
    commands = extract_commands(md_files)
    tsv = TMP / "invarlock_commands.tsv"
    cmd_objs = write_commands(commands, tsv)
    results_path = TMP / "invarlock_command_results.jsonl"
    run_commands(cmd_objs, results_path)
    print(f"Extracted {len(cmd_objs)} commands → {tsv}")
    print(f"Ran commands → {results_path}")

    # Summarize failures for quick triage
    total_executed = 0
    failed: list[str] = []
    for raw in results_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        total_executed += 1
        try:
            rec = json.loads(raw)
        except Exception:
            continue
        exit_code = rec.get("exit_code")
        if exit_code != 0:
            loc = f"{rec.get('file')}:{rec.get('line')}"
            cmd = rec.get("command", "")
            err = rec.get("error") or rec.get("stderr", "").strip()
            failed.append(f"{loc}: {cmd}\n{err}".rstrip())

    ok = max(0, total_executed - len(failed))
    print(f"Results: executed={total_executed} · ok={ok} · failed={len(failed)}")
    if failed:
        print("--- failures (first 10) ---")
        print("\n\n".join(failed[:10]))
        if os.getenv("FAIL_ON_ERROR", "1") in {"1", "true", "True"}:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
