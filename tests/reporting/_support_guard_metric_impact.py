from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from invarlock.eval.guard_metric_impact import (
    REQUIRED_GUARD_METRIC_IMPACT_CHECKS,
    build_guard_metric_bare_report,
    compute_guard_metric_impact,
    extract_guard_metric_arm_facts,
    guard_metric_schedule_digest,
)


def attach_canonical_metric_impact(cert: dict[str, Any]) -> None:
    """Attach a replayable no-regression impact surface to a report fixture."""

    metric_kind = cert["primary_metric"]["kind"]
    guarded_value = cert["primary_metric"]["final"]
    guarded_facts = extract_guard_metric_arm_facts(cert, metric_kind)
    bare_report = build_guard_metric_bare_report(cert, metric_kind)
    bare_facts = extract_guard_metric_arm_facts(bare_report, metric_kind)
    schedule_digest = guard_metric_schedule_digest(cert, metric_kind)
    measurement = compute_guard_metric_impact(metric_kind, guarded_value, guarded_value)
    assert guarded_facts is not None
    assert bare_report is not None
    bare_report["status"] = "success"
    assert bare_facts is not None
    assert schedule_digest is not None
    assert measurement is not None
    cert["guard_metric_impact"] = {
        "evaluated": True,
        "passed": True,
        **measurement.to_metrics(),
        "degradation_limit": 0.01,
        "bare_facts": bare_facts,
        "guarded_facts": guarded_facts,
        "bare_report": bare_report,
        "checks": dict.fromkeys(REQUIRED_GUARD_METRIC_IMPACT_CHECKS, True),
        "diagnostics": [],
        "source": "strict_fixture",
        "schedule_digest": schedule_digest,
    }


def ppl_arm_report(
    value: float, *, window_ids: list[int] | None = None
) -> dict[str, Any]:
    ids = list(window_ids or [1])
    counts = [100] * len(ids)
    return {
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": value}},
        "evaluation_windows": {
            "final": {
                "window_ids": ids,
                "logloss": [math.log(value)] * len(ids),
                "token_counts": counts,
            }
        },
    }


def accuracy_arm_report(
    correct: int,
    total: int,
    *,
    example_ids: list[int] | None = None,
) -> dict[str, Any]:
    ids = list(example_ids or range(total))
    value = correct / total
    return {
        "metrics": {
            "primary_metric": {"kind": "accuracy", "final": value},
            "classification": {"final": {"correct_total": correct, "total": total}},
        },
        "evaluation_windows": {"final": {"example_ids": ids}},
    }


def ppl_guard_context(
    bare_value: float,
    guarded_value: float,
    *,
    degradation_limit: float = 0.01,
) -> dict[str, Any]:
    return {
        "bare_report": ppl_arm_report(bare_value),
        "guarded_report": ppl_arm_report(guarded_value),
        "degradation_limit": degradation_limit,
    }


def canonical_ppl_impact(
    bare_value: float = 10.0,
    guarded_value: float = 10.0,
    *,
    degradation_limit: float = 0.01,
) -> dict[str, Any]:
    bare = ppl_arm_report(bare_value)
    guarded = ppl_arm_report(guarded_value)
    measurement = compute_guard_metric_impact("ppl_causal", bare_value, guarded_value)
    bare_facts = extract_guard_metric_arm_facts(bare, "ppl_causal")
    guarded_facts = extract_guard_metric_arm_facts(guarded, "ppl_causal")
    bare_report = build_guard_metric_bare_report(bare, "ppl_causal")
    schedule_digest = guard_metric_schedule_digest(guarded, "ppl_causal")
    assert measurement is not None
    assert bare_facts is not None
    assert guarded_facts is not None
    assert bare_report is not None
    bare_report["status"] = "success"
    assert schedule_digest is not None
    return {
        **measurement.to_metrics(),
        "bare_facts": bare_facts,
        "guarded_facts": guarded_facts,
        "bare_report": bare_report,
        "degradation_limit": degradation_limit,
        "evaluated": True,
        "passed": measurement.degradation <= degradation_limit,
        "checks": {
            "metric_kind_matches": True,
            "measurements_valid": True,
            "guard_metric_impact": measurement.degradation <= degradation_limit,
            "arm_facts_replay": True,
        },
        "diagnostics": [],
        "source": "test",
        "schedule_digest": schedule_digest,
    }


def final_ids_digest(ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
