#!/usr/bin/env python3
"""
Validate `invarlock ...` CLI command examples embedded in Markdown docs.

Goal: catch stale flags/subcommands in documentation without executing expensive
workloads or requiring network/GPU access.

Strategy:
  - Extract command lines that start with `invarlock ...` or `python -m invarlock ...`
    (optionally prefixed by environment variable assignments).
  - Join line continuations ending with '\\'.
  - Parse args against the Typer/Click command tree via Click's parser, without
    invoking command callbacks.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


CMD_PATTERN = re.compile(
    r"^(?P<prefix>(?:[A-Z_][A-Z0-9_]*=[^\s]+\s+)*)\s*(?P<cmd>(?:invarlock\s+|python\s+-m\s+invarlock(?:\\.[^\\s]+)?\s+).*)$"
)


def _is_code_fence(line: str) -> bool:
    return line.strip().startswith("```")


def _strip_prompt(s: str) -> str:
    s = s.lstrip()
    if s.startswith("$ "):
        return s[2:]
    return s


def _is_env_assignment(token: str) -> bool:
    # Very small heuristic; good enough for docs examples.
    if "=" not in token:
        return False
    key, _val = token.split("=", 1)
    return bool(re.fullmatch(r"[A-Z_][A-Z0-9_]*", key))


@dataclass(frozen=True)
class Example:
    file: Path
    line: int
    command: str


def extract_examples(paths: list[Path]) -> list[Example]:
    """Extract CLI commands from Markdown, keeping first occurrence."""
    examples: list[Example] = []
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

            cmd = text.rstrip()
            j = i
            while cmd.rstrip().endswith("\\") and (j + 1) < len(lines):
                cmd = cmd.rstrip()
                cmd = cmd[:-1].rstrip()
                j += 1
                cont_raw = _strip_prompt(lines[j].rstrip())
                if cont_raw.strip():
                    cmd += " " + cont_raw.strip()

            # Optional: support one more indented option line without a backslash.
            if (j + 1) < len(lines):
                nxt = lines[j + 1].lstrip()
                if nxt.startswith("--") and not nxt.startswith("```"):
                    cmd += " " + nxt.strip()
                    j += 1

            examples.append(Example(file=path, line=i + 1, command=cmd))
            i = j + 1

    # Deduplicate by command string; keep earliest location
    seen: set[str] = set()
    out: list[Example] = []
    for ex in examples:
        key = ex.command.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def _normalize_to_invarlock_args(command: str) -> list[str] | None:
    """Return CLI args list for `invarlock`, or None when unrecognized."""
    try:
        tokens = shlex.split(command, posix=True, comments=True)
    except ValueError:
        return None
    if not tokens:
        return None

    # Drop leading env assignments
    i = 0
    while i < len(tokens) and _is_env_assignment(tokens[i]):
        i += 1
    tokens = tokens[i:]
    if not tokens:
        return None

    if tokens[0] == "invarlock":
        return tokens[1:]

    if tokens[0] in {"python", "python3"} and len(tokens) >= 3 and tokens[1] == "-m":
        mod = tokens[2]
        if mod == "invarlock" or mod.startswith("invarlock."):
            return tokens[3:]

    return None


def _parse_cli_args(args: list[str]) -> str | None:
    """Return error string if invalid, else None."""
    import click

    from invarlock.cli.app import app as typer_app

    click_cmd = click.CommandCollection(sources=[])  # type: ignore[assignment]
    try:
        import typer

        click_cmd = typer.main.get_command(typer_app)  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001
        return f"Failed to load CLI command tree: {exc}"

    # Click help/completion options are eager and will raise click.Exit during parsing.
    # We treat those as success (exit_code == 0).
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        ctx_stack: list[click.Context] = []
        cmd: click.Command = click_cmd
        remaining = list(args)
        info_name = "invarlock"
        parent: click.Context | None = None

        try:
            while True:
                try:
                    ctx = cmd.make_context(info_name, remaining, parent=parent)
                except click.exceptions.Exit as exc:
                    if exc.exit_code == 0:
                        return None
                    return f"click.Exit({exc.exit_code})"

                ctx_stack.append(ctx)

                if not isinstance(cmd, click.MultiCommand):
                    return None

                combined = [
                    *getattr(ctx, "protected_args", []),
                    *getattr(ctx, "args", []),
                ]
                if not combined:
                    return None

                name, subcmd, subargs = cmd.resolve_command(ctx, combined)
                if subcmd is None:
                    return f"Unknown command: {name}"

                cmd = subcmd
                remaining = list(subargs)
                info_name = str(name)
                parent = ctx
        except click.ClickException as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
        finally:
            for ctx in reversed(ctx_stack):
                try:
                    ctx.close()
                except Exception:
                    pass
    return None


def _iter_markdown_files(repo_root: Path, *, paths: list[str] | None) -> list[Path]:
    if paths:
        out: list[Path] = []
        for p in paths:
            pp = (Path(p) if Path(p).is_absolute() else (repo_root / p)).resolve()
            if pp.exists() and pp.is_dir():
                out.extend(sorted(pp.rglob("*.md")))
            elif pp.exists() and pp.suffix.lower() == ".md":
                out.append(pp)
        # keep deterministic order
        return sorted({p.resolve() for p in out}, key=lambda x: str(x))
    return sorted(repo_root.glob("**/*.md"), key=lambda p: str(p))


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate CLI examples in Markdown")
    ap.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Optional file/dir paths to scan (defaults to entire repo).",
    )
    args_ns = ap.parse_args()

    md_files = _iter_markdown_files(ROOT, paths=args_ns.paths)
    examples = extract_examples(md_files)

    failures: list[tuple[Example, str]] = []
    checked = 0
    for ex in examples:
        norm = _normalize_to_invarlock_args(ex.command)
        if norm is None:
            continue
        checked += 1
        err = _parse_cli_args(norm)
        if err:
            failures.append((ex, err))

    if failures:
        print(f"CLI example validation failed ({len(failures)}/{checked} invalid):")
        for ex, err in failures[:30]:
            rel = ex.file.relative_to(ROOT)
            print(f"  - {rel}:{ex.line}: {err}")
            print(f"    {ex.command}")
        if len(failures) > 30:
            print(f"  ... {len(failures) - 30} more")
        return 1

    print(f"Validated {checked} CLI example(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
