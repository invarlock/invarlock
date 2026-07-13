"""Parse and validate the category-based architecture policy contract."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Category:
    name: str
    language: str
    role: str
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    soft_lines: int
    hard_lines: int
    soft_direct_files: int
    hard_direct_files: int
    soft_function_lines: int | None = None
    hard_function_lines: int | None = None
    soft_complexity: int | None = None
    hard_complexity: int | None = None


def read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a TOML table")
    return payload


def positive_int(
    table: dict[str, Any], key: str, *, optional: bool = False
) -> int | None:
    value = table.get(key)
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def string_list(table: dict[str, Any], key: str) -> tuple[str, ...]:
    value = table.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"{key} must be a non-empty string array")
    return tuple(value)


def optional_string_list(table: dict[str, Any], key: str) -> tuple[str, ...]:
    value = table.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{key} must be a string array")
    return tuple(value)


def load_categories(policy: dict[str, Any]) -> list[Category]:
    raw_categories = policy.get("categories")
    if not isinstance(raw_categories, list) or not raw_categories:
        raise ValueError("policy categories must be a non-empty array of tables")
    categories: list[Category] = []
    names: set[str] = set()
    for raw in raw_categories:
        if not isinstance(raw, dict):
            raise ValueError("each policy category must be a table")
        name = raw.get("name")
        language = raw.get("language")
        role = raw.get("role")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("category names must be non-empty and unique")
        if language not in {"python", "shell"}:
            raise ValueError(f"category {name}: language must be python or shell")
        if role not in {"production", "test"}:
            raise ValueError(f"category {name}: role must be production or test")
        names.add(name)
        category = Category(
            name=name,
            language=language,
            role=role,
            include=string_list(raw, "include"),
            exclude=optional_string_list(raw, "exclude"),
            soft_lines=int(positive_int(raw, "soft_lines") or 0),
            hard_lines=int(positive_int(raw, "hard_lines") or 0),
            soft_direct_files=int(positive_int(raw, "soft_direct_files") or 0),
            hard_direct_files=int(positive_int(raw, "hard_direct_files") or 0),
            soft_function_lines=positive_int(raw, "soft_function_lines", optional=True),
            hard_function_lines=positive_int(raw, "hard_function_lines", optional=True),
            soft_complexity=positive_int(raw, "soft_complexity", optional=True),
            hard_complexity=positive_int(raw, "hard_complexity", optional=True),
        )
        limits = (
            (category.soft_lines, category.hard_lines, "lines"),
            (category.soft_direct_files, category.hard_direct_files, "direct files"),
            (
                category.soft_function_lines,
                category.hard_function_lines,
                "function lines",
            ),
            (category.soft_complexity, category.hard_complexity, "complexity"),
        )
        for soft, hard, label in limits:
            if (soft is None) != (hard is None) or (
                soft is not None and hard is not None and soft > hard
            ):
                raise ValueError(f"category {name}: invalid soft/hard {label} limits")
        python_limits = (
            category.soft_function_lines,
            category.hard_function_lines,
            category.soft_complexity,
            category.hard_complexity,
        )
        if language == "shell" and any(value is not None for value in python_limits):
            raise ValueError(f"category {name}: Python metrics cannot govern shell")
        suffix = ".py" if language == "python" else ".sh"
        if any(not pattern.endswith(suffix) for pattern in category.include):
            raise ValueError(f"category {name}: include patterns must end in {suffix}")
        if set(category.include).issubset(category.exclude):
            raise ValueError(f"category {name}: every include pattern is excluded")
        categories.append(category)
    declared_patterns = [
        pattern for category in categories for pattern in category.include
    ]
    if len(declared_patterns) != len(set(declared_patterns)):
        raise ValueError("category include patterns must be unique")
    return categories


def load_governed_roots(policy: dict[str, Any]) -> tuple[str, ...]:
    roots = string_list(policy, "governed_roots")
    if any(
        root.startswith("/") or root.endswith("/") or ".." in Path(root).parts
        for root in roots
    ):
        raise ValueError("governed_roots must be normalized repository-relative paths")
    return roots


def load_allowed_names(policy: dict[str, Any], table_name: str) -> set[str]:
    table = policy.get(table_name)
    if not isinstance(table, dict):
        raise ValueError(f"{table_name} must be a table")
    return set(string_list(table, "allowed_names"))


def load_contract_owner_patterns(policy: dict[str, Any]) -> tuple[str, ...]:
    table = policy.get("contract_owners")
    if not isinstance(table, dict):
        raise ValueError("contract_owners must be a table")
    patterns = string_list(table, "include")
    if any(not pattern.endswith(".py") for pattern in patterns):
        raise ValueError("contract owner patterns must end in .py")
    return patterns


def validate_category_roots(
    categories: list[Category], governed_roots: tuple[str, ...]
) -> None:
    covered_roots: set[str] = set()
    for category in categories:
        for pattern in category.include:
            matching_roots = {
                root for root in governed_roots if pattern.startswith(f"{root}/")
            }
            if not matching_roots:
                raise ValueError(
                    f"category {category.name}: pattern outside governed roots: {pattern}"
                )
            covered_roots.update(matching_roots)
    missing = set(governed_roots) - covered_roots
    if missing:
        raise ValueError(
            f"governed roots without a reachable category: {sorted(missing)}"
        )


def validate_dependency_rules(policy: dict[str, Any]) -> None:
    rules = policy.get("dependency_rules", [])
    if not isinstance(rules, list) or not rules:
        raise ValueError("dependency_rules must be a non-empty array of tables")
    names: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("each dependency rule must be a table")
        name = rule.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("dependency rule names must be non-empty and unique")
        names.add(name)
        string_list(rule, "include")
        optional_string_list(rule, "exclude")
        imports = optional_string_list(rule, "forbid_import_prefixes")
        calls = optional_string_list(rule, "forbid_call_prefixes")
        if not imports and not calls:
            raise ValueError(f"dependency rule {name} must forbid imports or calls")
