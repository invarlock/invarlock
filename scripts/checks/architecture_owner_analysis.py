"""Closed declarative-expression rules for architecture owner analysis."""

from __future__ import annotations

import ast


def is_declarative_expression(
    node: ast.AST | None, *, safe_names: set[str] | None = None
) -> bool:
    safe_names = safe_names or set()
    if node is None or isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id in safe_names
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(
            is_declarative_expression(item, safe_names=safe_names) for item in node.elts
        )
    if isinstance(node, ast.Dict):
        return all(
            is_declarative_expression(key, safe_names=safe_names)
            and is_declarative_expression(value, safe_names=safe_names)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return (
            isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float, complex))
            and not isinstance(node.operand.value, bool)
        )
    return False


def is_inert_delegate_expression(node: ast.AST | None) -> bool:
    if isinstance(node, ast.BinOp):
        return False
    return is_declarative_expression(node)
