from __future__ import annotations

import ast
from pathlib import Path

from invarlock.guards._fault_injection_seams import GUARD_FAULT_INJECTION_SEAMS

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARDS_DIR = REPO_ROOT / "src" / "invarlock" / "guards"


def _declared_fn_parameters() -> set[tuple[str, str, str]]:
    declared: set[tuple[str, str, str]] = set()
    for path in GUARDS_DIR.glob("*.py"):
        if path.name == "_fault_injection_seams.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            args = (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
            )
            for arg in args:
                if arg.arg.endswith("_fn"):
                    declared.add((path.stem, node.name, arg.arg))
    return declared


def test_guard_fn_parameters_are_documented_fault_injection_seams() -> None:
    documented = {
        (entry.module, entry.function, entry.parameter)
        for entry in GUARD_FAULT_INJECTION_SEAMS
    }
    declared = _declared_fn_parameters()

    assert declared == documented


def test_guard_fault_injection_seams_have_actionable_rationales() -> None:
    for entry in GUARD_FAULT_INJECTION_SEAMS:
        assert entry.rationale.strip()
        assert len(entry.rationale.split()) >= 6
