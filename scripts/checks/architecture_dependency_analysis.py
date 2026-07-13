"""Resolve imports and calls for architecture dependency rules."""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from architecture_policy_contract import optional_string_list, string_list


def _resolved_from_module(node: ast.ImportFrom, rel_path: str) -> str:
    module_parts = list(Path(rel_path).with_suffix("").parts)
    if module_parts[:1] == ["src"]:
        module_parts = module_parts[1:]
    package_parts = (
        module_parts if Path(rel_path).name == "__init__.py" else module_parts[:-1]
    )
    if not node.level:
        return node.module or ""
    keep = max(0, len(package_parts) - node.level + 1)
    resolved = [*package_parts[:keep]]
    if node.module:
        resolved.extend(node.module.split("."))
    return ".".join(resolved)


def _qualified(node: ast.AST, bindings: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _qualified(node.value, bindings)
        return f"{base}.{node.attr}" if base else None
    if isinstance(node, ast.Call):
        return _qualified(node.func, bindings)
    return None


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return {name for item in node.elts for name in _target_names(item)}
    return set()


def _parameter_names(arguments: ast.arguments) -> set[str]:
    params = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    names = {argument.arg for argument in params}
    if arguments.vararg:
        names.add(arguments.vararg.arg)
    if arguments.kwarg:
        names.add(arguments.kwarg.arg)
    return names


class _LocalNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()
        self.globals: set[str] = set()
        self.nonlocals: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(
            alias.asname or alias.name.split(".")[0] for alias in node.names
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.update(alias.asname or alias.name for alias in node.names)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Global(self, node: ast.Global) -> None:
        self.globals.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocals.update(node.names)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self.names.add(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest:
            self.names.add(node.rest)
        self.generic_visit(node)

    def _visit_comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.GeneratorExp
    ) -> None:
        for generator in node.generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        self.visit(node.elt)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        for generator in node.generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        self.visit(node.key)
        self.visit(node.value)


def _local_names(statements: list[ast.stmt]) -> set[str]:
    collector = _LocalNameCollector()
    for statement in statements:
        collector.visit(statement)
    return collector.names - collector.globals - collector.nonlocals


def _pattern_names(pattern: ast.pattern) -> set[str]:
    collector = _LocalNameCollector()
    collector.visit(pattern)
    return collector.names


class _ScopeAnalyzer:
    def __init__(self, rel_path: str) -> None:
        self.rel_path = rel_path
        self.imports: set[str] = set()
        self.calls: set[str] = set()

    def analyze(self, tree: ast.Module) -> tuple[set[str], set[str]]:
        self._statements(tree.body, {})
        return self.imports, self.calls

    def _expression(self, node: ast.AST | None, bindings: dict[str, str]) -> None:
        if node is None:
            return
        if isinstance(node, ast.Lambda):
            local = dict(bindings)
            for name in _parameter_names(node.args):
                local.pop(name, None)
            self._expression(node.body, local)
            return
        if isinstance(
            node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
        ):
            local = dict(bindings)
            for generator in node.generators:
                self._expression(generator.iter, local)
                for name in _target_names(generator.target):
                    local.pop(name, None)
                for condition in generator.ifs:
                    self._expression(condition, local)
            if isinstance(node, ast.DictComp):
                self._expression(node.key, local)
                self._expression(node.value, local)
            else:
                self._expression(node.elt, local)
            return
        if isinstance(node, ast.Call):
            called = _qualified(node.func, bindings)
            if called:
                self.calls.add(called)
        if isinstance(node, ast.NamedExpr):
            self._expression(node.value, bindings)
            for name in _target_names(node.target):
                bindings.pop(name, None)
            return
        for child in ast.iter_child_nodes(node):
            self._expression(child, bindings)

    def _import(self, node: ast.Import, bindings: dict[str, str]) -> None:
        for alias in node.names:
            self.imports.add(alias.name)
            bindings[alias.asname or alias.name.split(".")[0]] = alias.name

    def _import_from(self, node: ast.ImportFrom, bindings: dict[str, str]) -> None:
        module = _resolved_from_module(node, self.rel_path)
        if module:
            self.imports.add(module)
        for alias in node.names:
            imported = f"{module}.{alias.name}" if module else alias.name
            self.imports.add(imported)
            bindings[alias.asname or alias.name] = imported

    def _function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        bindings: dict[str, str],
    ) -> None:
        for expression in [
            *node.decorator_list,
            *node.args.defaults,
            *node.args.kw_defaults,
        ]:
            self._expression(expression, bindings)
        local = dict(bindings)
        for name in _parameter_names(node.args) | _local_names(node.body):
            local.pop(name, None)
        self._statements(node.body, local)
        bindings.pop(node.name, None)

    def _assignment(
        self,
        value: ast.AST | None,
        targets: Iterable[ast.AST],
        bindings: dict[str, str],
    ) -> None:
        self._expression(value, bindings)
        constructed = (
            _qualified(value.func, bindings) if isinstance(value, ast.Call) else None
        )
        for target in targets:
            for name in _target_names(target):
                if constructed:
                    bindings[name] = constructed
                else:
                    bindings.pop(name, None)

    def _branch(self, statements: list[ast.stmt], bindings: dict[str, str]) -> None:
        self._statements(statements, dict(bindings))

    def _statements(self, statements: list[ast.stmt], bindings: dict[str, str]) -> None:
        for statement in statements:
            if isinstance(statement, ast.Import):
                self._import(statement, bindings)
            elif isinstance(statement, ast.ImportFrom):
                self._import_from(statement, bindings)
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._function(statement, bindings)
            elif isinstance(statement, ast.ClassDef):
                for expression in statement.decorator_list:
                    self._expression(expression, bindings)
                self._branch(statement.body, bindings)
                bindings.pop(statement.name, None)
            elif isinstance(statement, ast.Assign):
                self._assignment(statement.value, statement.targets, bindings)
            elif isinstance(statement, ast.AnnAssign):
                self._assignment(statement.value, [statement.target], bindings)
            elif isinstance(statement, ast.AugAssign):
                self._expression(statement.value, bindings)
                for name in _target_names(statement.target):
                    bindings.pop(name, None)
            elif isinstance(statement, (ast.If, ast.While)):
                self._expression(statement.test, bindings)
                self._branch(statement.body, bindings)
                self._branch(statement.orelse, bindings)
            elif isinstance(statement, (ast.For, ast.AsyncFor)):
                self._expression(statement.iter, bindings)
                branch_bindings = dict(bindings)
                for name in _target_names(statement.target):
                    branch_bindings.pop(name, None)
                self._statements(statement.body, branch_bindings)
                self._branch(statement.orelse, bindings)
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                branch_bindings = dict(bindings)
                for item in statement.items:
                    self._expression(item.context_expr, bindings)
                    if item.optional_vars:
                        for name in _target_names(item.optional_vars):
                            branch_bindings.pop(name, None)
                self._statements(statement.body, branch_bindings)
            elif isinstance(statement, ast.Try):
                self._branch(statement.body, bindings)
                self._branch(statement.orelse, bindings)
                self._branch(statement.finalbody, bindings)
                for handler in statement.handlers:
                    handler_bindings = dict(bindings)
                    if handler.name:
                        handler_bindings.pop(handler.name, None)
                    self._statements(handler.body, handler_bindings)
            elif isinstance(statement, ast.Match):
                self._expression(statement.subject, bindings)
                for case in statement.cases:
                    case_bindings = dict(bindings)
                    for name in _pattern_names(case.pattern):
                        case_bindings.pop(name, None)
                    self._expression(case.guard, case_bindings)
                    self._statements(case.body, case_bindings)
            else:
                for child in ast.iter_child_nodes(statement):
                    if isinstance(child, ast.expr):
                        self._expression(child, bindings)


def _rule_findings(
    *,
    rule: dict[str, Any],
    rel_path: str,
    imports: set[str],
    calls: set[str],
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for prefix in optional_string_list(rule, "forbid_import_prefixes"):
        if any(name == prefix or name.startswith(f"{prefix}.") for name in imports):
            findings.append(
                {
                    "kind": "dependency_direction",
                    "key": f"dependency_direction:{rule['name']}:{rel_path}:{prefix}",
                    "path": rel_path,
                    "rule": rule["name"],
                    "import_prefix": prefix,
                }
            )
    for prefix in optional_string_list(rule, "forbid_call_prefixes"):
        if any(name == prefix or name.startswith(f"{prefix}.") for name in calls):
            findings.append(
                {
                    "kind": "dependency_direction",
                    "key": f"dependency_direction:{rule['name']}:{rel_path}:call:{prefix}",
                    "path": rel_path,
                    "rule": rule["name"],
                    "call_prefix": prefix,
                }
            )
    return findings


def dependency_findings(
    files: list[tuple[Path, str]],
    policy: dict[str, Any],
    matches: Callable[[str, tuple[str, ...] | list[str]], bool],
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    rules = policy["dependency_rules"]
    if not isinstance(rules, list):
        raise ValueError("dependency_rules must be an array of tables")
    for path, rel_path in files:
        if path.suffix != ".py":
            continue
        relevant = [
            rule
            for rule in rules
            if isinstance(rule, dict)
            and matches(rel_path, string_list(rule, "include"))
            and not matches(rel_path, optional_string_list(rule, "exclude"))
        ]
        if not relevant:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        imports, calls = _ScopeAnalyzer(rel_path).analyze(tree)
        for rule in relevant:
            findings.extend(
                _rule_findings(
                    rule=rule,
                    rel_path=rel_path,
                    imports=imports,
                    calls=calls,
                )
            )
    return findings
