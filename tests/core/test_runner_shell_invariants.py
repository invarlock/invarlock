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


def test_runner_shell_has_no_nested_function_defs() -> None:
    path = REPO_ROOT / "src/invarlock/core/runner.py"
    assert _nested_function_names(path) == []
