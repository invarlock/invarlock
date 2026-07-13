#!/usr/bin/env python3
"""Enforce the repository-wide architecture policy."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from architecture_complexity import function_complexity
from architecture_dependency_analysis import dependency_findings
from architecture_owner_analysis import (
    is_declarative_expression as _is_declarative_expression,
)
from architecture_owner_analysis import (
    is_inert_delegate_expression as _is_inert_delegate_expression,
)
from architecture_policy_contract import (
    Category,
    load_allowed_names,
    load_categories,
    load_contract_owner_patterns,
    load_governed_roots,
    read_toml,
    validate_category_roots,
    validate_dependency_rules,
)

POLICY_SCHEMA = "invarlock/architecture-policy/v1"
DEBT_SCHEMA = "invarlock/architecture-debt/v1"
FORMAT_VERSION = "architecture-fragmentation-v1"
DEFAULT_POLICY = "contracts/architecture_policy.toml"
DEFAULT_DEBT = "contracts/architecture_debt.toml"
CRUFT_PATTERNS = (
    "**/.DS_Store",
    "**/.coverage/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/.ruff_cache/**",
    "**/__pycache__/**",
    "**/*.pyc",
)
CRUFT_SAMPLE_LIMIT = 50


def _matches(path: str, patterns: tuple[str, ...] | list[str]) -> bool:
    for pattern in patterns:
        variants = {pattern}
        pending = [pattern]
        while pending:
            value = pending.pop()
            start = 0
            while (index := value.find("/**/", start)) >= 0:
                collapsed = f"{value[:index]}/{value[index + 4 :]}"
                if collapsed not in variants:
                    variants.add(collapsed)
                    pending.append(collapsed)
                start = index + 1
        if any(fnmatch.fnmatch(path, value) for value in variants):
            return True
    return False


def _repo_files(repo_root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        paths = [repo_root / line for line in result.stdout.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        paths = [path for path in repo_root.rglob("*") if path.is_file()]
    return sorted(path for path in paths if path.is_file())


def _rel(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _category_for(path: str, categories: list[Category]) -> Category | None:
    matches = [
        category
        for category in categories
        if _matches(path, category.include) and not _matches(path, category.exclude)
    ]
    if len(matches) > 1:
        names = ", ".join(category.name for category in matches)
        raise ValueError(f"{path} matches multiple categories: {names}")
    return matches[0] if matches else None


def _is_type_checking_test(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == "TYPE_CHECKING") or (
        isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING"
    )


def _is_inert_node(node: ast.stmt) -> bool:
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass, ast.TypeAlias)):
        return True
    if isinstance(node, ast.Expr):
        return isinstance(node.value, ast.Constant)
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        return True
    if isinstance(node, ast.If) and _is_type_checking_test(node.test):
        return True
    return False


def _class_is_declaration(node: ast.ClassDef) -> bool:
    declaration_bases = {"Protocol", "TypedDict", "NamedTuple", "Enum", "IntEnum"}
    bases = {
        base.id if isinstance(base, ast.Name) else base.attr
        for base in node.bases
        if isinstance(base, (ast.Name, ast.Attribute))
    }
    decorators: set[str] = set()
    for decorator in node.decorator_list:
        expression = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(expression, ast.Name):
            decorators.add(expression.id)
        elif isinstance(expression, ast.Attribute):
            decorators.add(expression.attr)
    return bool(bases & declaration_bases or "dataclass" in decorators)


def _is_type_alias_declaration(node: ast.stmt) -> bool:
    return isinstance(node, ast.AnnAssign) and (
        (isinstance(node.annotation, ast.Name) and node.annotation.id == "TypeAlias")
        or (
            isinstance(node.annotation, ast.Attribute)
            and node.annotation.attr == "TypeAlias"
        )
    )


def _call_target_name(call: ast.Call) -> str | None:
    current: ast.AST = call.func
    while isinstance(current, (ast.Attribute, ast.Call)):
        current = current.value if isinstance(current, ast.Attribute) else current.func
    return current.id if isinstance(current, ast.Name) else None


def _delegate_call(
    call: ast.Call,
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    imported_names: set[str],
    visiting: set[str],
) -> bool:
    target = _call_target_name(call)
    if target not in functions:
        return target in imported_names
    if target in visiting:
        return True
    return _is_delegate_function(
        functions[target], functions, imported_names, {*visiting, target}
    )


def _is_delegate_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    imported_names: set[str],
    visiting: set[str] | None = None,
) -> bool:
    visiting = visiting or {node.name}
    result_names: set[str] = set()
    saw_delegate = False
    for statement in node.body:
        if isinstance(statement, ast.Pass) or (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
        ):
            continue
        value: ast.AST | None = None
        targets: list[ast.expr] = []
        if isinstance(statement, ast.Assign):
            value = statement.value
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            value = statement.value
            targets = [statement.target]
        elif isinstance(statement, ast.Expr):
            value = statement.value
        elif isinstance(statement, ast.Return):
            value = statement.value
            if isinstance(value, ast.Name) and value.id in result_names:
                continue
        else:
            return False
        if isinstance(value, ast.Await):
            value = value.value
        if targets and not isinstance(value, ast.Call):
            if _is_inert_delegate_expression(value):
                continue
            return False
        if not isinstance(value, ast.Call) or not _delegate_call(
            value, functions, imported_names, visiting
        ):
            return False
        saw_delegate = True
        result_names.update(
            target.id for target in targets if isinstance(target, ast.Name)
        )
    return saw_delegate


def _is_private_padding_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    if not node.name.startswith("_"):
        return False
    for statement in node.body:
        if isinstance(statement, ast.Pass) or (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
        ):
            continue
        if isinstance(statement, ast.Return) and _is_declarative_expression(
            statement.value
        ):
            continue
        if isinstance(statement, (ast.Assign, ast.AnnAssign)) and (
            _is_declarative_expression(statement.value)
        ):
            continue
        return False
    return True


def _is_delegate_class(
    node: ast.ClassDef,
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    imported_names: set[str],
) -> bool:
    saw_method = False
    for child in node.body:
        if _is_inert_node(child):
            continue
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        saw_method = True
        if not _is_delegate_function(child, functions, imported_names):
            return False
    return saw_method


def _assignment_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {target.id for target in targets if isinstance(target, ast.Name)}


def _is_strict_contract_owner(tree: ast.Module) -> bool:
    public_constants: set[str] = set()
    safe_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            safe_names.update(
                alias.asname or alias.name.split(".")[0] for alias in node.names
            )
            continue
        if isinstance(node, ast.ImportFrom):
            safe_names.update(alias.asname or alias.name for alias in node.names)
            continue
        if isinstance(node, ast.TypeAlias):
            if not _is_declarative_expression(node.value, safe_names=safe_names):
                return False
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = _assignment_names(node)
            if not names or any(
                name != "__all__" and (name.startswith("_") or not name.isupper())
                for name in names
            ):
                return False
            if not _is_declarative_expression(node.value, safe_names=safe_names):
                return False
            public_constants.update(name for name in names if name != "__all__")
            safe_names.update(names)
            continue
        if isinstance(node, ast.ClassDef) and _class_is_declaration(node):
            continue
        return False
    return bool(public_constants)


def _has_cohesive_owner(tree: ast.Module) -> bool:
    imported_names = {
        alias.asname or alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_names.update(
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    )
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for node in tree.body:
        if _is_type_alias_declaration(node):
            return True
        if _is_inert_node(node):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_private_padding_function(node) or _is_delegate_function(
                node, functions, imported_names
            ):
                continue
            return True
        if isinstance(node, ast.ClassDef):
            if _class_is_declaration(node):
                return True
            if _is_delegate_class(node, functions, imported_names):
                continue
            return True
        # Executable control flow includes entrypoint wrappers.
        return True
    return False


def _shell_has_cohesive_owner(text: str) -> bool:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("source ", ". ", "export ", "readonly ")):
            continue
        if "=" in line and line.split("=", 1)[0].replace("_", "").isalnum():
            continue
        return True
    return False


def _is_explicit_facade(tree: ast.Module) -> bool:
    has_import = any(
        isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body
    )
    return has_import and not _has_cohesive_owner(tree)


def _function_name(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
    parents: dict[ast.AST, ast.AST],
) -> str:
    names = [
        node.name if not isinstance(node, ast.Lambda) else f"<lambda>@{node.lineno}"
    ]
    current: ast.AST = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(current.name)
    return ".".join(reversed(names))


def _finding(
    kind: str,
    key: str,
    path: str,
    actual: int,
    budget: int,
    *,
    comparison: str = ">",
    **extra: object,
) -> dict[str, object]:
    return {
        "kind": kind,
        "key": key,
        "path": path,
        "actual": actual,
        "budget": budget,
        "comparison": comparison,
        **extra,
    }


def _file_findings(
    path: Path, rel_path: str, category: Category, policy: dict[str, Any]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    text = path.read_text(encoding="utf-8")
    lines = len(text.splitlines())
    hard: list[dict[str, object]] = []
    soft: list[dict[str, object]] = []
    target = hard if lines > category.hard_lines else soft
    limit = category.hard_lines if target is hard else category.soft_lines
    if lines > limit:
        target.append(
            _finding(
                "file_lines",
                f"file_lines:{rel_path}",
                rel_path,
                lines,
                limit,
                category=category.name,
            )
        )

    allowed_tiny = load_allowed_names(policy, "tiny_owners")
    if category.language != "python":
        if (
            category.role == "production"
            and path.name not in allowed_tiny
            and not _shell_has_cohesive_owner(text)
        ):
            hard.append(
                {
                    "kind": "tiny_owner",
                    "key": f"tiny_owner:{rel_path}",
                    "path": rel_path,
                    "category": category.name,
                }
            )
        return hard, soft
    try:
        tree = ast.parse(text, filename=rel_path)
    except SyntaxError as exc:
        hard.append(
            {
                "kind": "python_parse",
                "key": f"python_parse:{rel_path}",
                "path": rel_path,
                "message": str(exc),
            }
        )
        return hard, soft

    contract_patterns = load_contract_owner_patterns(policy)
    is_contract_path = _matches(rel_path, contract_patterns)
    is_contract_owner = is_contract_path and _is_strict_contract_owner(tree)
    if is_contract_path and not is_contract_owner:
        hard.append(
            {
                "kind": "invalid_contract_owner",
                "key": f"invalid_contract_owner:{rel_path}",
                "path": rel_path,
                "category": category.name,
            }
        )

    is_facade = (
        category.role == "production"
        and not is_contract_owner
        and path.name not in load_allowed_names(policy, "facades")
        and _is_explicit_facade(tree)
    )
    if is_facade:
        hard.append(
            {
                "kind": "reexport_facade",
                "key": f"reexport_facade:{rel_path}",
                "path": rel_path,
                "category": category.name,
            }
        )
    elif (
        category.role == "production"
        and path.name not in allowed_tiny
        and not is_contract_owner
        and not _has_cohesive_owner(tree)
    ):
        hard.append(
            {
                "kind": "tiny_owner",
                "key": f"tiny_owner:{rel_path}",
                "path": rel_path,
                "category": category.name,
            }
        )

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        symbol = _function_name(node, parents)
        decorators = node.decorator_list if not isinstance(node, ast.Lambda) else []
        first_line = min([node.lineno, *(decorator.lineno for decorator in decorators)])
        span = getattr(node, "end_lineno", node.lineno) - first_line + 1
        if (
            category.hard_function_lines is not None
            and span > category.hard_function_lines
        ):
            hard.append(
                _finding(
                    "function_lines",
                    f"function_lines:{rel_path}:{symbol}",
                    rel_path,
                    span,
                    category.hard_function_lines,
                    category=category.name,
                    symbol=symbol,
                )
            )
        elif (
            category.soft_function_lines is not None
            and span > category.soft_function_lines
        ):
            soft.append(
                _finding(
                    "function_lines",
                    f"function_lines:{rel_path}:{symbol}",
                    rel_path,
                    span,
                    category.soft_function_lines,
                    category=category.name,
                    symbol=symbol,
                )
            )
        complexity = function_complexity(node)
        if (
            category.hard_complexity is not None
            and complexity > category.hard_complexity
        ):
            hard.append(
                _finding(
                    "function_complexity",
                    f"function_complexity:{rel_path}:{symbol}",
                    rel_path,
                    complexity,
                    category.hard_complexity,
                    category=category.name,
                    symbol=symbol,
                )
            )
        elif (
            category.soft_complexity is not None
            and complexity > category.soft_complexity
        ):
            soft.append(
                _finding(
                    "function_complexity",
                    f"function_complexity:{rel_path}:{symbol}",
                    rel_path,
                    complexity,
                    category.soft_complexity,
                    category=category.name,
                    symbol=symbol,
                )
            )
    return hard, soft


def _apply_debt(
    findings: list[dict[str, object]], debt: dict[str, Any], today: date
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    entries = debt.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("debt entries must be an array of tables")
    by_key = {str(finding["key"]): finding for finding in findings}
    duplicate_keys: set[str] = set()
    seen: set[str] = set()
    diagnostics: list[dict[str, object]] = []
    debt_errors: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each debt entry must be a table")
        key = entry.get("key")
        owner = entry.get("owner")
        reason = entry.get("reason")
        expiry = entry.get("expires")
        ceiling = entry.get("ceiling")
        if not isinstance(key, str) or not key:
            raise ValueError("debt entry key must be non-empty")
        if key in seen:
            duplicate_keys.add(key)
        seen.add(key)
        if (
            not isinstance(owner, str)
            or not owner
            or not isinstance(reason, str)
            or not reason
        ):
            raise ValueError(f"debt {key}: owner and reason are required")
        if not isinstance(expiry, date):
            raise ValueError(f"debt {key}: expires must be an ISO date")
        if isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling < 0:
            raise ValueError(f"debt {key}: ceiling must be a non-negative integer")
        finding = by_key.get(key)
        if expiry <= today:
            debt_errors.append(
                {
                    "kind": "expired_debt",
                    "key": f"expired_debt:{key}",
                    "path": str(finding.get("path", "")) if finding else "",
                    "debt_key": key,
                    "expires": expiry.isoformat(),
                }
            )
            continue
        if finding is None:
            debt_errors.append(
                {
                    "kind": "stale_debt",
                    "key": f"stale_debt:{key}",
                    "path": "",
                    "debt_key": key,
                }
            )
            continue
        actual = finding.get("actual")
        if not isinstance(actual, int):
            debt_errors.append(
                {
                    "kind": "invalid_debt_target",
                    "key": f"invalid_debt_target:{key}",
                    "path": str(finding.get("path", "")),
                    "debt_key": key,
                }
            )
            continue
        if actual > ceiling:
            debt_errors.append(
                _finding(
                    "debt_regression",
                    f"debt_regression:{key}",
                    str(finding.get("path", "")),
                    actual,
                    ceiling,
                    debt_key=key,
                )
            )
            continue
        if actual < ceiling:
            debt_errors.append(
                _finding(
                    "stale_debt_ceiling",
                    f"stale_debt_ceiling:{key}",
                    str(finding.get("path", "")),
                    actual,
                    ceiling,
                    comparison="<",
                    debt_key=key,
                )
            )
            continue
        debt_errors.append(
            {
                "kind": "new_debt",
                "key": f"new_debt:{key}",
                "path": str(finding.get("path", "")),
                "debt_key": key,
            }
        )
        diagnostics.append(finding)
    for key in sorted(duplicate_keys):
        debt_errors.append(
            {
                "kind": "duplicate_debt",
                "key": f"duplicate_debt:{key}",
                "path": "",
                "debt_key": key,
            }
        )
    return list(by_key.values()), diagnostics, debt_errors


def _select_governed_files(
    repo_root: Path,
    repo_files: list[Path],
    categories: list[Category],
    governed_roots: tuple[str, ...],
) -> tuple[list[tuple[Path, str, Category]], list[str]]:
    governed: list[tuple[Path, str, Category]] = []
    uncategorized: list[str] = []
    root_prefixes = tuple(f"{root}/" for root in governed_roots)
    for path in repo_files:
        rel_path = _rel(path, repo_root)
        if _matches(rel_path, CRUFT_PATTERNS):
            continue
        category = _category_for(rel_path, categories)
        if category is not None:
            governed.append((path, rel_path, category))
        elif rel_path.startswith(root_prefixes) and path.suffix in {".py", ".sh"}:
            uncategorized.append(rel_path)
    return governed, uncategorized


def _collect_governed_findings(
    repo_root: Path,
    governed: list[tuple[Path, str, Category]],
    uncategorized: list[str],
    categories: list[Category],
    policy: dict[str, Any],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    Counter[str],
]:
    hard: list[dict[str, object]] = []
    soft: list[dict[str, object]] = []
    category_counts: Counter[str] = Counter()
    directory_counts: Counter[tuple[str, str]] = Counter()
    categories_by_name = {category.name: category for category in categories}
    for path, rel_path, category in governed:
        category_counts[category.name] += 1
        directory_counts[(category.name, Path(rel_path).parent.as_posix())] += 1
        file_hard, file_soft = _file_findings(path, rel_path, category, policy)
        hard.extend(file_hard)
        soft.extend(file_soft)
    for (category_name, directory), count in sorted(directory_counts.items()):
        category = categories_by_name[category_name]
        key = f"direct_files:{category_name}:{directory}"
        if count > category.hard_direct_files:
            hard.append(
                _finding(
                    "package_concentration",
                    key,
                    directory,
                    count,
                    category.hard_direct_files,
                    category=category_name,
                )
            )
        elif count > category.soft_direct_files:
            soft.append(
                _finding(
                    "package_concentration",
                    key,
                    directory,
                    count,
                    category.soft_direct_files,
                    category=category_name,
                )
            )
    paths = [(path, rel_path) for path, rel_path, _ in governed]
    hard.extend(dependency_findings(paths, policy, _matches))
    hard.extend(
        {
            "kind": "uncategorized_code",
            "key": f"uncategorized_code:{path}",
            "path": path,
        }
        for path in uncategorized
    )
    return hard, soft, category_counts


def collect_metrics(
    repo_root: Path,
    *,
    policy_path: Path | None = None,
    debt_path: Path | None = None,
    today: date | None = None,
) -> dict[str, object]:
    policy_path = policy_path or repo_root / DEFAULT_POLICY
    debt_path = debt_path or repo_root / DEFAULT_DEBT
    policy = read_toml(policy_path)
    debt = read_toml(debt_path)
    if policy.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"policy schema must be {POLICY_SCHEMA}")
    if "generated_code" in policy:
        raise ValueError("generated-code exclusions are not supported")
    if debt.get("schema") != DEBT_SCHEMA:
        raise ValueError(f"debt schema must be {DEBT_SCHEMA}")
    categories = load_categories(policy)
    governed_roots = load_governed_roots(policy)
    validate_category_roots(categories, governed_roots)
    load_allowed_names(policy, "facades")
    load_allowed_names(policy, "tiny_owners")
    load_contract_owner_patterns(policy)
    validate_dependency_rules(policy)
    repo_files = _repo_files(repo_root)
    cruft = sorted(
        _rel(path, repo_root)
        for path in repo_files
        if _matches(_rel(path, repo_root), CRUFT_PATTERNS)
    )
    governed, uncategorized = _select_governed_files(
        repo_root, repo_files, categories, governed_roots
    )
    hard, soft, category_counts = _collect_governed_findings(
        repo_root,
        governed,
        uncategorized,
        categories,
        policy,
    )

    hard.sort(key=lambda item: str(item["key"]))
    remaining, debt_diagnostics, debt_errors = _apply_debt(
        hard, debt, today or date.today()
    )
    release_blockers = sorted(
        [*remaining, *debt_errors], key=lambda item: str(item["key"])
    )
    soft.sort(key=lambda item: str(item["key"]))
    source_files = [
        item
        for item in governed
        if item[1].startswith("src/invarlock/") and item[0].suffix == ".py"
    ]
    script_files = [item for item in governed if item[1].startswith("scripts/")]
    reexport_shims = sorted(
        str(item["path"]) for item in hard if item["kind"] == "reexport_facade"
    )
    package_counts: Counter[str] = Counter()
    for _, rel_path, _ in source_files:
        rel = Path(rel_path).relative_to("src/invarlock")
        package_counts[rel.parts[0] if len(rel.parts) > 1 else "__root__"] += 1
    script_family_counts = Counter(
        Path(rel_path).parts[1] if len(Path(rel_path).parts) > 2 else "__root__"
        for _, rel_path, _ in script_files
    )
    return {
        "format_version": FORMAT_VERSION,
        "policy_schema": POLICY_SCHEMA,
        "debt_schema": DEBT_SCHEMA,
        "release_ready": not release_blockers,
        "release_blocker_count": len(release_blockers),
        "release_blockers": release_blockers,
        "soft_finding_count": len(soft),
        "soft_findings": soft,
        "suppressed_debt_count": 0,
        "suppressed_debt": [],
        "debt_diagnostic_count": len(debt_diagnostics),
        "debt_diagnostics": sorted(debt_diagnostics, key=lambda item: str(item["key"])),
        "category_file_counts": dict(sorted(category_counts.items())),
        "uncategorized_code": sorted(uncategorized),
        "source_python_files": len(source_files),
        "source_python_files_over_budget": sum(
            1
            for item in hard
            if item["kind"] == "file_lines"
            and str(item["path"]).startswith("src/invarlock/")
        ),
        "source_python_file_paths_over_budget": sorted(
            str(item["path"])
            for item in hard
            if item["kind"] == "file_lines"
            and str(item["path"]).startswith("src/invarlock/")
        ),
        "small_files_under_50_lines": sum(
            1
            for path, _, _ in source_files
            if len(path.read_text(encoding="utf-8").splitlines()) < 50
        ),
        "tiny_files_under_20_lines": sum(
            1
            for path, _, _ in source_files
            if len(path.read_text(encoding="utf-8").splitlines()) < 20
        ),
        "reexport_shim_count": len(reexport_shims),
        "reexport_shims": reexport_shims,
        "run_orchestrator_file_count": sum(
            1
            for _, rel_path, _ in source_files
            if Path(rel_path).name.startswith("run_orchestrator")
        ),
        "run_orchestrator_files": sorted(
            rel_path
            for _, rel_path, _ in source_files
            if Path(rel_path).name.startswith("run_orchestrator")
        ),
        "reporting_file_count": sum(
            1
            for _, rel_path, _ in source_files
            if rel_path.startswith("src/invarlock/reporting/")
        ),
        "largest_packages_by_file_count": [
            {"package": package, "files": count}
            for package, count in package_counts.most_common(10)
        ],
        "script_tracked_files": len(script_files),
        "script_python_files": sum(path.suffix == ".py" for path, _, _ in script_files),
        "script_shell_files": sum(path.suffix == ".sh" for path, _, _ in script_files),
        "script_generated_cruft_files": len(cruft),
        "script_generated_cruft_sample_limit": CRUFT_SAMPLE_LIMIT,
        "script_generated_cruft_sample_paths": cruft[:CRUFT_SAMPLE_LIMIT],
        "script_evidence_pack_files": sum(
            rel_path.startswith("scripts/evidence_packs/")
            for _, rel_path, _ in script_files
        ),
        "largest_script_families_by_file_count": [
            {"family": family, "files": count}
            for family, count in script_family_counts.most_common(10)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable metrics only."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--policy", help="Policy path, relative to repository root by default."
    )
    parser.add_argument(
        "--debt", help="Debt ledger path, relative to repository root by default."
    )
    parser.add_argument(
        "--as-of", help="Override today's date for deterministic debt tests."
    )
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()

    def resolve(value: str | None, default: str) -> Path:
        path = Path(value or default)
        return path if path.is_absolute() else repo_root / path

    metrics: dict[str, Any]
    try:
        metrics = collect_metrics(
            repo_root,
            policy_path=resolve(args.policy, DEFAULT_POLICY),
            debt_path=resolve(args.debt, DEFAULT_DEBT),
            today=date.fromisoformat(args.as_of) if args.as_of else None,
        )
    except (ValueError, UnicodeDecodeError) as exc:
        metrics = {
            "format_version": FORMAT_VERSION,
            "release_ready": False,
            "release_blocker_count": 1,
            "release_blockers": [
                {
                    "kind": "policy_error",
                    "key": "policy_error",
                    "path": "",
                    "message": str(exc),
                }
            ],
        }
    if args.json:
        print(json.dumps(metrics, sort_keys=True))
    else:
        print(
            f"[check_architecture_fragmentation] {'OK' if metrics['release_ready'] else 'FAIL'}"
        )
        print(
            f"release blockers: {metrics['release_blocker_count']}; "
            f"soft findings: {metrics.get('soft_finding_count', 0)}; "
            f"debt diagnostics: {metrics.get('debt_diagnostic_count', 0)}"
        )
        blockers = metrics.get("release_blockers")
        if not isinstance(blockers, list):
            raise ValueError("release_blockers must be a list")
        for finding in blockers:
            if not isinstance(finding, dict):
                raise ValueError("release blockers must be objects")
            detail = ""
            if "actual" in finding and "budget" in finding:
                comparator = finding.get("comparison", ">")
                detail = f" ({finding['actual']} {comparator} {finding['budget']})"
            print(
                f"- {finding['kind']}: {finding.get('path') or finding['key']}{detail}"
            )
    return 0 if metrics["release_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
