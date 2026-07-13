from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from invarlock.core.assurance_contract import (
    build_assurance_section,
    strict_report_policy_errors,
)
from tests.core._support_assurance_contract import (
    strict_variance_gain_report as _strict_variance_gain_report,
)


def _variance_guard(report: dict[str, Any]) -> dict[str, Any]:
    return next(entry for entry in report["guards"] if entry["name"] == "variance")


def _set_policy(
    report: dict[str, Any],
    guard: dict[str, Any],
    **updates: object,
) -> None:
    for policy in (
        report["variance"]["policy"],
        report["resolved_policy"]["variance"],
        guard["policy"],
        guard["details"]["policy"],
    ):
        policy.update(updates)


def _set_scale_evidence(
    report: dict[str, Any],
    *,
    raw: Mapping[str, float],
    proposed: Mapping[str, float],
) -> None:
    guard = _variance_guard(report)
    raw_copy = dict(raw)
    proposed_copy = dict(proposed)
    target_names = list(raw_copy)
    metrics = guard["metrics"]
    metrics.update(
        proposed_scales=len(proposed_copy),
        target_modules=len(target_names),
        target_module_names=target_names,
        proposed_scales_pre_edit=proposed_copy,
        proposed_scales_post_edit=proposed_copy,
        raw_scales_pre_edit=raw_copy,
        raw_scales_post_edit=raw_copy,
    )
    guard["details"]["proposed_scales"] = proposed_copy
    guard["details"]["stats"].update(
        target_module_names=target_names,
        proposed_scales_pre_edit=proposed_copy,
        proposed_scales_post_edit=proposed_copy,
        raw_scales_pre_edit_normalized=raw_copy,
        raw_scales_post_edit_normalized=raw_copy,
    )
    report["variance"].update(
        proposed_scales=len(proposed_copy),
        target_modules=len(target_names),
        target_module_names=target_names,
        proposed_scales_pre_edit=proposed_copy,
        proposed_scales_post_edit=proposed_copy,
    )


def _strict_errors(report: dict[str, Any]) -> list[str]:
    report["assurance"] = build_assurance_section(report)
    return strict_report_policy_errors(report, require_strict=True)


@pytest.mark.parametrize(
    ("raw", "proposed", "policy"),
    [
        (
            {"above": 1.03, "below": 0.98, "ignored": 1.005},
            {"above": 1.02, "below": 0.98},
            {},
        ),
        (
            {"weaker": 1.006, "strongest": 0.989, "weakest": 1.002},
            {"strongest": 0.989},
            {},
        ),
        (
            {"preferred-up": 1.02, "larger-down": 0.95},
            {"preferred-up": 1.02},
            {"max_adjusted_modules": 1},
        ),
    ],
)
def test_strict_gain_accepts_exact_producer_scale_selection_replay(
    raw: dict[str, float],
    proposed: dict[str, float],
    policy: dict[str, object],
) -> None:
    report = _strict_variance_gain_report()
    guard = _variance_guard(report)
    _set_policy(report, guard, **policy)
    _set_scale_evidence(report, raw=raw, proposed=proposed)

    assert _strict_errors(report) == []


@pytest.mark.parametrize(
    ("raw", "proposed", "policy"),
    [
        (
            {"retained": 1.03, "omitted": 0.97},
            {"retained": 1.02},
            {},
        ),
        (
            {"strongest": 1.011, "forged-weaker": 0.9895},
            {"forged-weaker": 0.9895},
            {},
        ),
        (
            {"strongest": 1.011, "second": 0.9895},
            {"strongest": 1.011, "second": 0.9895},
            {"topk_backstop": 2},
        ),
        (
            {"producer-winner": 1.02, "forged-larger-delta": 0.95},
            {"forged-larger-delta": 0.98},
            {"max_adjusted_modules": 1},
        ),
    ],
    ids=(
        "omitted-qualifying-scale",
        "wrong-backstop-candidate",
        "multiple-backstop-candidates",
        "wrong-max-limit-winner",
    ),
)
def test_strict_gain_rejects_fabricated_producer_scale_selection(
    raw: dict[str, float],
    proposed: dict[str, float],
    policy: dict[str, object],
) -> None:
    report = _strict_variance_gain_report()
    guard = _variance_guard(report)
    _set_policy(report, guard, **policy)
    _set_scale_evidence(report, raw=raw, proposed=proposed)

    errors = _strict_errors(report)

    assert any("must exactly replay producer filtering" in error for error in errors), (
        "\n".join(errors)
    )


def test_variance_scale_selection_is_canonical_json_order_invariant() -> None:
    report = _strict_variance_gain_report()
    guard = _variance_guard(report)
    _set_policy(
        report,
        guard,
        min_abs_adjust=0.2,
        max_scale_step=0.02,
        topk_backstop=1,
    )
    raw = {"z.module": 1.125, "a.module": 0.875}
    proposed = {"a.module": 0.98}
    _set_scale_evidence(report, raw=raw, proposed=proposed)

    before = _strict_errors(report)
    after = _strict_errors(json.loads(json.dumps(report, sort_keys=True)))

    assert before == []
    assert after == []
