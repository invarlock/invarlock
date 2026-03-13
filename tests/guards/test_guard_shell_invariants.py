from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _nested_function_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nested: list[str] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if self.depth > 0:
                nested.append(node.name)
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if self.depth > 0:
                nested.append(node.name)
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

    Visitor().visit(tree)
    return nested


def _import_targets(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            targets.add(node.module)
    return targets


def test_guard_shells_have_no_nested_function_defs() -> None:
    for relpath in (
        "src/invarlock/guards/variance.py",
        "src/invarlock/guards/spectral.py",
    ):
        assert _nested_function_names(REPO_ROOT / relpath) == []


def test_guard_helper_modules_do_not_import_cli_or_reporting_layers() -> None:
    helper_files = sorted(
        (REPO_ROOT / "src/invarlock/guards").glob("variance_*.py")
    ) + sorted((REPO_ROOT / "src/invarlock/guards").glob("spectral_*.py"))
    forbidden_prefixes = ("invarlock.cli", "invarlock.reporting")
    for path in helper_files:
        imports = _import_targets(path)
        assert not any(target.startswith(forbidden_prefixes) for target in imports), (
            path
        )
