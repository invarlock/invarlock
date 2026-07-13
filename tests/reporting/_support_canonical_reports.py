from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, cast

from invarlock.core.auto_tuning import resolve_tier_policies
from invarlock.reporting.report_make import make_report as _make_report
from invarlock.reporting.report_types import RunReport, create_empty_report
from invarlock.reporting.runtime_policy_receipt import build_runtime_policy_receipt


def _merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            _merge(current, value)
        else:
            target[key] = copy.deepcopy(value)


def canonical_run_report(source: Mapping[str, Any]) -> RunReport:
    """Materialize an explicitly canonical test fixture as a ``RunReport``."""

    if "dataset" in source:
        raise ValueError("fixtures must use the canonical data block")
    if "ppl_final" in source or "ppl_preview" in source:
        raise ValueError("fixtures must use metrics.primary_metric")
    source_metrics = source.get("metrics")
    if not isinstance(source_metrics, Mapping):
        raise ValueError("fixtures must provide canonical metrics")
    if "ppl_final" in source_metrics or "ppl_preview" in source_metrics:
        raise ValueError("fixtures must use metrics.primary_metric")
    source_primary_metric = source_metrics.get("primary_metric")
    if not isinstance(source_primary_metric, Mapping):
        raise ValueError("fixtures must provide metrics.primary_metric")
    for field in ("kind", "preview", "final"):
        if field not in source_primary_metric:
            raise ValueError(f"fixtures must provide metrics.primary_metric.{field}")

    source_meta = source.get("meta")
    if not isinstance(source_meta, Mapping):
        raise ValueError("fixtures must provide canonical meta fields")
    for field in ("model_id", "adapter"):
        if not isinstance(source_meta.get(field), str) or not source_meta[field]:
            raise ValueError(f"fixtures must provide meta.{field}")
    source_auto = source_meta.get("auto")
    if not isinstance(source_auto, Mapping) or not isinstance(
        source_auto.get("tier"), str
    ):
        raise ValueError("fixtures must provide meta.auto.tier")

    source_context = source.get("context")
    if not isinstance(source_context, Mapping) or not isinstance(
        source_context.get("profile"), str
    ):
        raise ValueError("fixtures must provide context.profile")

    source_data = source.get("data")
    if not isinstance(source_data, Mapping):
        raise ValueError("fixtures must provide the canonical data block")
    for field in (
        "dataset",
        "split",
        "seq_len",
        "stride",
        "preview_n",
        "final_n",
    ):
        if field not in source_data:
            raise ValueError(f"fixtures must provide data.{field}")

    source_edit = source.get("edit")
    if not isinstance(source_edit, Mapping) or not isinstance(
        source_edit.get("name"), str
    ):
        raise ValueError("fixtures must provide edit.name")

    source_guards = source.get("guards")
    if not isinstance(source_guards, list):
        raise ValueError("fixtures must provide guards as a list")
    for index, guard in enumerate(source_guards):
        if not isinstance(guard, Mapping) or not isinstance(guard.get("passed"), bool):
            raise ValueError(f"fixtures must provide guards[{index}].passed")

    report = create_empty_report()
    _merge(cast(dict[str, Any], report), source)
    tier = str(source_auto["tier"]).strip().lower()
    profile = str(source_context["profile"]).strip().lower()
    runtime_policies = resolve_tier_policies(
        tier, report["edit"]["name"], profile=profile
    )
    for guard in cast(list[dict[str, Any]], report["guards"]):
        guard_name = str(guard.get("name") or "").strip().lower()
        if guard_name in runtime_policies:
            applied_policy = copy.deepcopy(runtime_policies[guard_name])
            raw_policy = guard.get("policy")
            if isinstance(raw_policy, Mapping):
                _merge(applied_policy, raw_policy)
            guard["policy"] = applied_policy
    resolved, receipt = build_runtime_policy_receipt(
        runtime_policies,
        report["guards"],
        tier=tier,
        profile=profile,
        edit_name=report["edit"]["name"],
    )
    report["resolved_policy"] = resolved
    report["policy_resolution"] = receipt
    return report


def canonical_baseline(source: Mapping[str, Any]) -> RunReport:
    """Materialize an explicitly canonical no-op baseline fixture."""

    baseline = canonical_run_report(source)
    if baseline["edit"]["name"] != "noop":
        raise ValueError("baseline fixtures must provide edit.name='noop'")
    primary_metric = baseline["metrics"]["primary_metric"]
    primary_metric.pop("ratio_vs_baseline", None)
    return baseline


def refresh_runtime_policy_receipt(report: Mapping[str, Any]) -> RunReport:
    """Rebind a fixture receipt after a test intentionally changes guard policy."""

    refreshed = copy.deepcopy(dict(report))
    resolution = refreshed.get("policy_resolution")
    if not isinstance(resolution, Mapping):
        raise ValueError("fixtures must provide policy_resolution")
    meta = refreshed.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError("fixtures must provide canonical meta fields")
    auto = meta.get("auto")
    if not isinstance(auto, Mapping) or not isinstance(auto.get("tier"), str):
        raise ValueError("fixtures must provide meta.auto.tier")
    context = refreshed.get("context")
    if not isinstance(context, Mapping) or not isinstance(context.get("profile"), str):
        raise ValueError("fixtures must provide context.profile")
    edit = refreshed.get("edit")
    if not isinstance(edit, Mapping) or not isinstance(edit.get("name"), str):
        raise ValueError("fixtures must provide edit.name")
    tier = str(auto["tier"])
    profile = str(context["profile"])
    edit_name = str(edit["name"])
    expected_resolution = {
        "tier": tier,
        "profile": profile,
        "edit_name": edit_name,
    }
    for field, expected in expected_resolution.items():
        if resolution.get(field) != expected:
            raise ValueError(f"fixtures must bind policy_resolution.{field}")
    resolved_policy = refreshed.get("resolved_policy")
    if not isinstance(resolved_policy, Mapping):
        raise ValueError("fixtures must provide resolved_policy")
    guards = refreshed.get("guards")
    if not isinstance(guards, list):
        raise ValueError("fixtures must provide guards as a list")
    resolved, receipt = build_runtime_policy_receipt(
        resolved_policy,
        guards,
        tier=tier,
        profile=profile,
        edit_name=edit_name,
    )
    refreshed["resolved_policy"] = resolved
    refreshed["policy_resolution"] = receipt
    return refreshed  # type: ignore[return-value]


def make_canonical_report(
    report: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    """Build an evaluation report from explicit canonical run fixtures."""

    return _make_report(canonical_run_report(report), canonical_baseline(baseline))


__all__ = [
    "canonical_baseline",
    "canonical_run_report",
    "make_canonical_report",
    "refresh_runtime_policy_receipt",
]
