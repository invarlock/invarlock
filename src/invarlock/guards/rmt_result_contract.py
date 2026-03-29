from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_prepare_result(
    *,
    ready: bool,
    baseline_metrics: Mapping[str, Any],
    policy_applied: Mapping[str, Any],
    preparation_time: float,
    error: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ready": bool(ready),
        "baseline_metrics": dict(baseline_metrics),
        "policy_applied": dict(policy_applied),
        "preparation_time": float(preparation_time),
    }
    if error is not None:
        result["error"] = str(error)
    return result


def build_after_edit_result(
    *,
    edge_risk_by_module: Mapping[str, Any] | None = None,
    edge_risk_by_family: Mapping[str, Any] | None = None,
    analysis_source: str = "activations_edge_risk",
    token_weight_total: int | None = None,
    batches_used: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "analysis_source": analysis_source,
        "edge_risk_by_module": dict(edge_risk_by_module or {}),
        "edge_risk_by_family": dict(edge_risk_by_family or {}),
    }
    if token_weight_total is not None:
        result["token_weight_total"] = int(token_weight_total)
    if batches_used is not None:
        result["batches_used"] = int(batches_used)
    return result
