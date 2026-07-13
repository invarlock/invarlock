from __future__ import annotations

from typing import Any

from invarlock.core.auto_tuning import resolve_tier_policies
from invarlock.reporting.runtime_policy_receipt import build_runtime_policy_receipt


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def bind_runtime_policy(
    report: dict[str, Any], *, profile: str | None = None
) -> dict[str, Any]:
    """Bind a CLI RunReport fixture to the same policy receipt as runtime output."""

    meta = _mapping(report.setdefault("meta", {}))
    context = _mapping(report.setdefault("context", {}))
    meta_auto = _mapping(meta.get("auto"))
    context_auto = _mapping(context.get("auto"))
    run_context = _mapping(context.get("run"))
    tier = (
        str(
            meta_auto.get("tier")
            or context_auto.get("tier")
            or run_context.get("tier")
            or context.get("tier")
            or "balanced"
        )
        .strip()
        .lower()
    )
    declared_profile = (
        str(profile or context.get("profile") or run_context.get("profile") or "dev")
        .strip()
        .lower()
    )
    meta_auto["tier"] = tier
    meta["auto"] = meta_auto
    context["profile"] = declared_profile
    report["meta"] = meta
    report["context"] = context

    edit = _mapping(report.get("edit"))
    edit_name = str(edit.get("name") or "noop").strip()
    guards = report.get("guards", [])
    resolved, receipt = build_runtime_policy_receipt(
        resolve_tier_policies(tier, edit_name, profile=declared_profile),
        guards if isinstance(guards, list) else [],
        tier=tier,
        profile=declared_profile,
        edit_name=edit_name,
    )
    report["resolved_policy"] = resolved
    report["policy_resolution"] = receipt
    return report


__all__ = ["bind_runtime_policy"]
