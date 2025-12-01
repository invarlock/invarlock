#!/usr/bin/env python3
"""Check config include paths and adapter names.

This script verifies that:
- All `defaults: !include <path>` targets resolve to files.
- All adapter names referenced in configs are present in the plugin registry.

Exit code is non-zero on any failure.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def find_includes_and_adapters(
    root: Path,
) -> tuple[list[tuple[Path, str, Path]], set[str]]:
    include_re = re.compile(r"^\s*defaults:\s*!include\s+(\S+)\s*$")
    adapter_re = re.compile(r"^\s*adapter:\s*\"?([A-Za-z0-9_\-]+)\"?")
    missing_includes: list[tuple[Path, str, Path]] = []
    adapters: set[str] = set()
    for p in root.rglob("*.yaml"):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                m = include_re.match(line)
                if m:
                    inc = m.group(1)
                    target = (p.parent / inc).resolve()
                    if not target.exists():
                        missing_includes.append((p, inc, target))
                m2 = adapter_re.match(line)
                if m2:
                    adapters.add(m2.group(1))
        except Exception as e:
            print(f"ERR reading {p}: {e}")
    return missing_includes, adapters


def registry_adapters() -> set[str]:
    try:
        from invarlock.core.registry import get_registry
    except Exception:
        return set()
    reg = get_registry()
    return set(reg.list_adapters())


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("configs")
    missing_includes, adapters = find_includes_and_adapters(root)
    rc = 0
    print("Include targets:")
    if not missing_includes:
        print("  All defaults includes resolve correctly.")
    else:
        for p, inc, tgt in missing_includes:
            print(f"  MISS {p} -> {inc} (resolved {tgt})")
        rc = 1

    print("\nAdapters referenced:")
    for a in sorted(adapters):
        print(f"  {a}")
    reg = registry_adapters()
    print("\nAdapter registry availability:")
    for a in sorted(adapters):
        ok = a in reg
        print(f"  {'OK  ' if ok else 'MISS'} {a}")
        if not ok:
            rc = 1

    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
