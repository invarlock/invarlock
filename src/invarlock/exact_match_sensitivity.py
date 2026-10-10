"""Bounded hypothetical subject edits for current native exact-match decisions.

This advisory does not verify evidence, mutate records, or replace a verdict.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from invarlock.evidence_pack_contract import (
    MAX_RECORDS,
    PAIRED_RECORDS_FORMAT,
    _sample_qualification,
    _side_accuracy_qualification,
    build_comparison_report,
)
from invarlock.paired_exact_match import (
    ExactMatchOutcome,
    _binary_outcomes,
    _paired_effect_confidence_interval_v2,
)


def _native_report(baseline, subject, policy) -> dict[str, object]:
    # These synthetic bindings are confined to an arithmetic oracle. They are
    # never returned as evidence or used to claim authentication of the inputs.
    return build_comparison_report(
        comparison_id="hypothetical-subject-edits",
        paired_records={
            "format": PAIRED_RECORDS_FORMAT,
            "metric": "exact_match",
            "schedule_sha256": "0" * 64,
            "records": [
                {"baseline": {"score": int(b)}, "subject": {"score": int(s)}}
                for b, s in zip(baseline, subject, strict=True)
            ],
        },
        policy={"resolved_policy": {"metrics": {"exact_match": policy}}},
        policy_digest="sha256:" + "0" * 64,
        report_format="invarlock/comparison-report-v3",
    )


def _table(n: int, b: int, a: int, c: int, policy: Mapping[str, object]):
    interval = _paired_effect_confidence_interval_v2(
        pair_count=n,
        both_pass_count=a,
        baseline_pass_subject_fail_count=b - a,
        baseline_fail_subject_pass_count=c,
    )
    sample = _sample_qualification(
        policy,
        metric="exact_match",
        scorer_binding=None,
        record_count=n,
        interval_lower=interval.lower_pp,
        interval_upper=interval.upper_pp,
    )
    side, side_passed = _side_accuracy_qualification(
        policy, baseline_mean=b / n, subject_mean=(a + c) / n
    )
    gates = {
        "effect_floor": interval.lower_pp >= cast(float, policy["delta_min_pp"]),
        "sample_qualification": sample is None or bool(sample["passed"]),
        "side_accuracy": side_passed,
    }
    return {
        "verdict": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "interval_lower_pp": interval.lower_pp,
        "interval_upper_pp": interval.upper_pp,
        "sample_qualification": sample,
        "side_accuracy": side,
        "counts": {
            "both_pass": a,
            "baseline_only": b - a,
            "subject_only": c,
            "both_fail": n - b - c,
        },
    }


def _flip_indices(baseline, subject, a_change: int, c_change: int) -> list[int]:
    indices = []
    for baseline_value, change in ((True, a_change), (False, c_change)):
        eligible = [
            i
            for i, (b, s) in enumerate(zip(baseline, subject, strict=True))
            if b == baseline_value and s == (change < 0)
        ]
        indices.extend(eligible[: abs(change)])
    return sorted(indices)


def exact_match_sensitivity(
    baseline: Sequence[ExactMatchOutcome],
    subject: Sequence[ExactMatchOutcome],
    *,
    policy: Mapping[str, object],
    max_changes: int = 8,
    max_states: int = 2048,
) -> dict[str, Any]:
    """Find the nearest opposite verdict within a bounded subject-flip search.

    Baseline, sample size, current native v3 arithmetic and policy stay fixed.
    ``policy`` contains the fields under ``metrics.exact_match``. Inputs must be
    paired binary outcomes, in the same order, with at most 10,000 records.
    A lower-bound result certifies only the fully checked radii. Witness indices
    refer to this supplied order, not to authenticated record identifiers.
    """
    for name, value, maximum in (
        ("max_changes", max_changes, 32),
        ("max_states", max_states, 5000),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    b_values = _binary_outcomes(baseline, label="baseline")
    s_values = _binary_outcomes(subject, label="subject")
    if len(b_values) != len(s_values) or len(b_values) > MAX_RECORDS:
        raise ValueError(f"paired inputs must have equal lengths at most {MAX_RECORDS}")
    # Validate all native policy fields through the same report builder used by
    # verification; the compressed search reuses its qualification helpers.
    selected = dict(policy)
    original_report = _native_report(b_values, s_values, selected)
    n = len(b_values)
    b = sum(b_values)
    a = sum(x and y for x, y in zip(b_values, s_values, strict=True))
    c = sum(not x and y for x, y in zip(b_values, s_values, strict=True))
    original = _table(n, b, a, c, selected)
    if original["verdict"] != original_report["verdict"]:
        raise RuntimeError("sensitivity arithmetic disagrees with the native report")
    binding = json.dumps(
        {"baseline": b_values, "subject": s_values, "policy": selected},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    result: dict[str, Any] = {
        "kind": "exact_match_sensitivity",
        "method": "subject_flip_distance_newcombe_v2",
        "comparison_report_format": "invarlock/comparison-report-v3",
        "input_sha256": hashlib.sha256(binding).hexdigest(),
        "policy": selected,
        "max_changes": max_changes,
        "max_states": max_states,
        "original_verdict": original["verdict"],
        "original": original,
        "status": "lower_bound",
        "minimum_changes": None,
        "checked_through_changes": 0,
        "states_examined": 0,
        "witness": None,
    }
    # With fixed baseline margins, each table is reachable with exactly this
    # Manhattan distance: the two coordinates edit disjoint baseline groups.
    for radius in range(1, min(n, max_changes) + 1):
        for dx in range(-radius, radius + 1):
            dy = radius - abs(dx)
            for y in sorted({c - dy, c + dy}):
                x = a + dx
                if not (0 <= x <= b and 0 <= y <= n - b):
                    continue
                if result["states_examined"] == max_states:
                    return result
                candidate = _table(n, b, x, y, selected)
                result["states_examined"] += 1
                if candidate["verdict"] == original["verdict"]:
                    continue
                indices = _flip_indices(b_values, s_values, x - a, y - c)
                witness = list(s_values)
                for i in indices:
                    witness[i] = not witness[i]
                replay = _native_report(b_values, witness, selected)
                if replay["verdict"] != candidate["verdict"] or len(indices) != radius:
                    raise RuntimeError("sensitivity witness failed native replay")
                result.update(
                    status="exact",
                    minimum_changes=radius,
                    witness={"subject_flip_indices": indices, **candidate},
                )
                return result
        result["checked_through_changes"] = radius
    if result["checked_through_changes"] == n:
        result["status"] = "no_flip_possible"
    return result
