"""Fail-closed plugin provenance checks for strict assurance reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .builtin_plugin_catalog import BuiltinPluginSpec, builtin_plugin_specs


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, dict) else None


def _builtin_spec(plugin_type: str, name: str) -> BuiltinPluginSpec | None:
    return next(
        (spec for spec in builtin_plugin_specs(plugin_type) if spec.name == name),
        None,
    )


def _validate_plugin_entry(
    errors: list[str],
    *,
    entry: Any,
    path: str,
    plugin_type: str,
    expected_name: str | None = None,
) -> str | None:
    metadata = _mapping(entry)
    if metadata is None:
        errors.append(f"strict assurance requires {path} plugin provenance.")
        return None

    name = metadata.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append(f"{path}.name must be a non-empty string.")
        return None
    name = name.strip()
    if expected_name is not None and name != expected_name:
        errors.append(f"{path}.name must be {expected_name!r}.")

    spec = _builtin_spec(plugin_type, name)
    if spec is None:
        errors.append(
            f"{path}.name={name!r} is not a shipped plugin eligible for strict "
            "assurance."
        )
        return name

    expected = {
        "type": plugin_type,
        "module": spec.module,
        "package": "invarlock",
        "support_tier": spec.support_tier,
        "strict_assurance_allowed": spec.strict_assurance_allowed,
        "available": True,
    }
    for field, expected_value in expected.items():
        value = metadata.get(field)
        if value != expected_value or type(value) is not type(expected_value):
            errors.append(
                f"{path}.{field} must match shipped plugin metadata "
                f"({expected_value!r})."
            )
    if spec.strict_assurance_allowed is not True:
        errors.append(f"{path} is not eligible for strict assurance.")
    return name


def strict_plugin_provenance_errors(
    report: Mapping[str, Any],
    *,
    canonical_guard_chain: Sequence[str],
) -> list[str]:
    """Require exact shipped-plugin metadata and report-field reconciliation."""

    plugins = _mapping(report.get("plugins"))
    if plugins is None:
        return ["strict assurance requires a plugins provenance object."]

    errors: list[str] = []
    adapter_name = _validate_plugin_entry(
        errors,
        entry=plugins.get("adapter"),
        path="plugins.adapter",
        plugin_type="adapters",
    )
    edit_name = _validate_plugin_entry(
        errors,
        entry=plugins.get("edit"),
        path="plugins.edit",
        plugin_type="edits",
    )

    meta = _mapping(report.get("meta"))
    reported_adapter = meta.get("adapter") if meta is not None else None
    if not isinstance(reported_adapter, str) or not reported_adapter.strip():
        errors.append("strict assurance requires non-empty meta.adapter.")
    elif adapter_name is not None and adapter_name != reported_adapter:
        errors.append("plugins.adapter.name must match meta.adapter exactly.")

    edit = _mapping(report.get("edit"))
    reported_edit = edit.get("name") if edit is not None else None
    if not isinstance(reported_edit, str) or not reported_edit.strip():
        errors.append("strict assurance requires non-empty edit.name.")
    elif edit_name is not None and edit_name != reported_edit:
        errors.append("plugins.edit.name must match edit.name exactly.")

    guards = plugins.get("guards")
    if not isinstance(guards, list):
        errors.append("strict assurance requires plugins.guards provenance array.")
        return errors
    expected_guards = tuple(canonical_guard_chain)
    if len(guards) != len(expected_guards):
        errors.append(
            "plugins.guards provenance must exactly cover the canonical guard chain."
        )
    for index, expected_name in enumerate(expected_guards):
        entry = guards[index] if index < len(guards) else None
        _validate_plugin_entry(
            errors,
            entry=entry,
            path=f"plugins.guards[{index}]",
            plugin_type="guards",
            expected_name=expected_name,
        )
    return errors


__all__ = ["strict_plugin_provenance_errors"]
