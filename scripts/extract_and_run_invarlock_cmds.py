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
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp"
TMP.mkdir(parents=True, exist_ok=True)


CMD_PATTERN = re.compile(
    r"^(?P<prefix>(?:[A-Z_]+=[^\s]+\s+)*)\s*(?P<cmd>(?:invarlock\s+|python\s+-m\s+invarlock\s+).*)$"
)


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
    "runs/latest",
    "<BASELINE_MODEL>",
    "<SUBJECT_MODEL>",
    "<model_or_id>",
    "<edited_model_or_dir>",
    "<source>",
    "<edited>",
    "<ts>",
    "<hf_dir_or_id>",
]


def _should_skip(cmd: str) -> bool:
    # allow disabling skip via env
    if os.getenv("SKIP_PLACEHOLDERS", "1") not in {"1", "true", "True"}:
        return False
    s = cmd.strip()
    # skip obvious placeholders
    for tok in SKIP_TOKENS:
        if tok in s:
            return True
    # skip invalid plugin subgroup examples
    if s.startswith("invarlock plugins ") and any(w in s for w in ("datasets",)):
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
            # Join continuations ending with '\\'
            cmd = text.rstrip("\\").rstrip()
            j = i + 1
            while j < len(lines) and lines[j].rstrip().endswith("\\"):
                cont = _strip_prompt(lines[j].rstrip())
                cont = cont.rstrip("\\").strip()
                if cont:
                    cmd += " " + cont
                j += 1
            # Also include the next line if it is a continuation without trailing backslash but indentation suggests it's a continued command
            if j < len(lines):
                nxt = lines[j].lstrip()
                if (
                    nxt.startswith("--")
                    and not nxt.startswith("```")
                    and not nxt.startswith("invarlock ")
                ):
                    # merge one more option line
                    cmd += " " + nxt.strip()
                    j += 1
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
    if " invarlock run" in s or " invarlock certify" in s:
        return 300
    if " invarlock report" in s:
        return 120
    return 180


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
            cmd_str = c.cmd.strip()
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
    md_files = list(ROOT.glob("**/*.md"))
    # Prefer deterministic ordering
    md_files.sort(key=lambda p: str(p))
    commands = extract_commands(md_files)
    tsv = TMP / "invarlock_commands.tsv"
    cmd_objs = write_commands(commands, tsv)
    results_path = TMP / "invarlock_command_results.jsonl"
    run_commands(cmd_objs, results_path)
    print(f"Extracted {len(cmd_objs)} commands → {tsv}")
    print(f"Ran commands → {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
