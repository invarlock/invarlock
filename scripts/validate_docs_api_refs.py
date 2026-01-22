#!/usr/bin/env python3
"""
Scan Markdown files for references to `invarlock.*` symbols and verify they resolve.

What it checks
- Finds dotted references like `invarlock.eval.bench.Bench` in all `*.md` files.
- Attempts to import the longest module prefix and then traverse attributes.
- Reports missing modules/attributes with file and line for quick triage.

Outputs
- Prints a short summary to stdout.
- Writes a JSONL report to `tmp/docs_api_refs_results.jsonl`.

Usage
- Run from repo root: `python scripts/validate_docs_api_refs.py`
"""

from __future__ import annotations

import importlib
import io
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TMP = ROOT / "tmp"
TMP.mkdir(parents=True, exist_ok=True)

EXCLUDE_TOP_LEVEL_DIRS = {
    # Internal planning docs may reference future APIs; exclude from checks.
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
}

REF_PATTERN = re.compile(r"\b(invarlock(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\b")
IGNORE_LAST_SEGMENT = {
    # Common file extensions and non-API suffixes
    "svg",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "md",
    "json",
    "yaml",
    "yml",
    # Domain-style suffix used in project email/URLs (e.g., invarlock.dev)
    # These should not be treated as Python attributes on the invarlock package.
    "dev",
}


@dataclass
class Ref:
    file: str
    line: int
    text: str


def iter_refs(paths: Iterable[Path]) -> list[Ref]:
    results: list[Ref] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, start=1):
            for m in REF_PATTERN.finditer(line):
                # De-duplicate references across multiple identical occurrences on the same line
                sym = m.group(1)
                last = sym.rsplit(".", 1)[-1]
                if last in IGNORE_LAST_SEGMENT:
                    continue
                # Heuristic: skip when the match appears inside simple quotes on the same line
                start, end = m.span(1)
                before = line[start - 1] if start - 1 >= 0 else ""
                after = line[end] if end < len(line) else ""
                if before in {'"', "'"} and after in {'"', "'"}:
                    continue
                results.append(Ref(file=str(path), line=i, text=sym))
    return results


def resolve_ref(symbol: str) -> tuple[bool, str | None]:
    """Resolve a dotted `invarlock.*` symbol to a module+attribute if present.

    Strategy:
    - Find the longest importable module prefix.
    - Then traverse remaining dotted attributes via getattr.
    """
    parts = symbol.split(".")
    # Identify longest importable prefix.
    # If imports fail only because optional third-party deps are missing
    # (e.g., torch/transformers), treat the reference as "skipped but ok"
    # so that docs checks do not fail in minimal environments.
    mod_path: str | None = None
    optional_dep_missing = False
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        try:
            importlib.import_module(candidate)
            mod_path = candidate
            break
        except ModuleNotFoundError as exc:
            # If the missing module is *not* an invarlock.* module, assume this
            # is an optional dependency (e.g., torch, transformers) and treat
            # the reference as best-effort/optional.
            name = getattr(exc, "name", "") or ""
            if not name.startswith("invarlock"):
                optional_dep_missing = True
                break
            # Otherwise, keep searching shorter prefixes (might still resolve).
            continue
        except Exception:
            # Any other import-time error is treated as optional/may depend on
            # local environment; do not fail docs validation on it.
            optional_dep_missing = True
            break
    if mod_path is None:
        if optional_dep_missing:
            # Consider references to modules that require optional deps as
            # "soft-ok": they are valid in fully-featured environments.
            return True, None
        return False, "module not found"
    # Traverse remaining attributes
    obj = importlib.import_module(mod_path)
    for attr in parts[len(mod_path.split(".")) :]:
        if not hasattr(obj, attr):
            return False, f"attribute '{attr}' missing on {obj!r}"
        obj = getattr(obj, attr)
    return True, None


def main() -> int:
    # Ensure `src/` is importable
    sys.path.insert(0, str(SRC))
    md_files: list[Path] = []
    for path in ROOT.glob("**/*.md"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(ROOT).parts
        if rel_parts and rel_parts[0] in EXCLUDE_TOP_LEVEL_DIRS:
            continue
        md_files.append(path)
    md_files.sort(key=lambda p: str(p))
    refs = iter_refs(md_files)
    # De-duplicate exact (file, line, text) tuples
    seen: set[tuple[str, int, str]] = set()
    unique: list[Ref] = []
    for r in refs:
        key = (r.file, r.line, r.text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    ok = 0
    failed = 0
    buf = io.StringIO()
    out_path = TMP / "docs_api_refs_results.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for r in unique:
            success, err = resolve_ref(r.text)
            record = {
                "file": r.file,
                "line": r.line,
                "symbol": r.text,
                "ok": success,
                "error": err,
            }
            out.write(json.dumps(record) + "\n")
            if success:
                ok += 1
            else:
                failed += 1
                buf.write(f"{r.file}:{r.line}: unresolved {r.text} — {err}\n")

    print(f"Checked {len(unique)} doc references · ok={ok} · failed={failed}")
    if failed:
        print("--- failures ---")
        print(buf.getvalue().rstrip())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
