"""Paired, multi-metric release checks with explicit scoring assurance."""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any, cast

from invarlock.core.scoring import MetricError, score, validate_configuration
from invarlock.evaluation_comparison.capacity import (
    DEFAULT_MAX_BOOTSTRAP_DRAWS,
    check_missing_id_capacity,
    missing_pair,
)
from invarlock.evaluation_record_contracts import contracts
from invarlock.evaluation_record_contracts.contracts import (
    EvaluationRecordsError,
    digest,
    validate,
)
from invarlock.evaluation_records.cases import validate_run_case_set
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.paired_exact_match import paired_exact_match_statistics

BOOTSTRAP_REPLICATES = 2048
_BINARY_METRICS = frozenset(
    ("exact_match", "normalized_match", "numeric_tolerance", "json_exact")
)


class LocalWorkBudgetExceeded(EvaluationRecordsError):
    """The caller's work allowance cannot cover the complete comparison."""

    reason = "local_work_budget_exceeded"


def check_bootstrap_budget(
    policy: dict[str, Any], scope_count: int, maximum: int | None
) -> int:
    required = (
        scope_count
        * sum(m["kind"] not in _BINARY_METRICS for m in policy["metrics"])
        * BOOTSTRAP_REPLICATES
    )
    if maximum is not None and required > maximum:
        raise LocalWorkBudgetExceeded(
            f"comparison plans {required} bootstrap draws; "
            f"local budget is {maximum}. Increase the local budget "
            "only on a suitable host; the complete policy has not been evaluated."
        )
    return required


def metric_summaries(comparison: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Expose completed counts/reasons without copying potentially large ID lists."""
    return tuple(
        {
            "name": metric["name"],
            "slice": metric["slice"],
            "decision": metric["decision"],
            "usable_count": metric["count"] - len(metric["missing_ids"]),
            "missing_count": len(metric["missing_ids"]),
            "reasons": list(metric["reasons"]),
        }
        for metric in comparison["metrics"]
    )


def make_run(
    records: list[dict[str, Any]],
    *,
    source: dict[str, str],
    run_id: str,
    artifact_digest: str,
    score_provenance: dict[str, Any] | None = None,
    source_digest: str | None = None,
) -> dict[str, Any]:
    """Capture existing results; never execute a model or infer a trust root."""
    normalized = []
    for row in records:
        if set(row) - {
            "id",
            "input",
            "expected",
            "output",
            "scores",
            "metadata",
            "error",
            "context",
        }:
            raise EvaluationRecordsError(
                "record has unknown fields; map them explicitly in the adapter"
            )
        normalized.append(
            {"scores": {}, "metadata": {}, "error": None, "context": {}, **row}
        )
    value = {
        "format": "invarlock/evaluation-run-v1",
        "source": source,
        "run_id": run_id,
        "artifact_digest": artifact_digest,
        "source_digest": source_digest,
        "score_provenance": score_provenance or {},
        "records": normalized,
    }
    _check_run(value)
    # Detach caller-owned dictionaries; later mutation must be an explicit new run.
    import json

    return cast(dict[str, Any], json.loads(canonical_json_bytes(value)))


def _check_run(value: dict[str, Any]) -> None:
    validate(value, "run")
    ids = [row["id"] for row in value["records"]]
    if len(ids) != len(set(ids)):
        raise EvaluationRecordsError(
            "duplicate record IDs; repeated trials need distinct paired IDs"
        )


def _check_policy(policy: dict[str, Any]) -> None:
    validate(policy, "policy")
    names = [m["name"] for m in policy["metrics"]]
    slices = [s["name"] for s in policy["slices"]]
    if (
        len(names) != len(set(names))
        or len(slices) != len(set(slices))
        or "overall" in slices
    ):
        raise EvaluationRecordsError(
            "metric and slice names must be unique; overall is reserved"
        )
    for metric in policy["metrics"]:
        if metric["kind"] == "recorded":
            if not metric.get("accepted_provenance") or not metric.get("score_key"):
                raise EvaluationRecordsError(
                    "recorded metric requires explicit accepted_provenance and score_key"
                )
            if metric["configuration"]:
                raise EvaluationRecordsError(
                    "recorded metric cannot redefine the recorded scorer"
                )
            provenance = metric["accepted_provenance"]
            if provenance["unit"] != metric["unit"]:
                raise EvaluationRecordsError(
                    "recorded metric unit differs from approved provenance"
                )
            if (
                provenance["kind"] in ("judge", "human")
                and provenance["rubric_digest"] is None
            ):
                raise EvaluationRecordsError(
                    "judge/human provenance requires an approved rubric digest"
                )
        else:
            if "score_key" in metric or "accepted_provenance" in metric:
                raise EvaluationRecordsError(
                    "recomputed metrics cannot accept recorded provenance"
                )
            if metric["direction"] != "higher" or metric["unit"] != "score":
                raise EvaluationRecordsError(
                    "deterministic scorers use higher-is-better scores in [0,1]"
                )
            try:
                validate_configuration(metric["kind"], metric["configuration"])
            except MetricError as exc:
                raise EvaluationRecordsError(str(exc)) from exc
        if metric.get("subject_minimum", -math.inf) > metric.get(
            "subject_maximum", math.inf
        ):
            raise EvaluationRecordsError("subject minimum exceeds maximum")


def _interval(
    baseline: list[float], candidate: list[float], seed: str, binary: bool
) -> dict[str, Any]:
    if binary:
        interval = paired_exact_match_statistics(
            [bool(v) for v in baseline], [bool(v) for v in candidate]
        ).effect_size_confidence_interval
        return {
            "lower": interval.lower_pp / 100,
            "upper": interval.upper_pp / 100,
            "method": interval.method,
            "mass": 0.95,
            "replicates": 0,
        }
    differences = [c - b for b, c in zip(baseline, candidate, strict=True)]
    count = len(differences)
    if min(differences) == max(differences):
        lower = upper = differences[0]
    else:
        draws = []
        # Fixed SHAKE stream and little-endian uint64 mapping make replay independent
        # of Python's random module. Rejection sampling avoids modulo bias.
        ceiling = (2**64 // count) * count
        for replicate in range(BOOTSTRAP_REPLICATES):
            values: list[float] = []
            block = 0
            while len(values) < count:
                stream = hashlib.shake_256(
                    f"{seed}:{replicate}:{block}".encode()
                ).digest(count * 8)
                values.extend(
                    differences[v % count]
                    for (v,) in struct.iter_unpack("<Q", stream)
                    if v < ceiling
                )
                block += 1
            draws.append(math.fsum(values[:count]) / count)
        draws.sort()

        def percentile(q: float) -> float:
            position = (len(draws) - 1) * q
            low = int(position)
            return draws[low] + (draws[min(low + 1, len(draws) - 1)] - draws[low]) * (
                position - low
            )

        lower, upper = percentile(0.025), percentile(0.975)
    return {
        "lower": lower,
        "upper": upper,
        "method": "paired_mean_shake256_percentile_v1",
        "mass": 0.95,
        "replicates": BOOTSTRAP_REPLICATES,
    }


def _metric_result(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    metric: dict[str, Any],
    slice_name: str,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    recorded = metric["kind"] == "recorded"
    if recorded:
        for run in (baseline, candidate):
            if (
                run["score_provenance"].get(metric["score_key"])
                != metric["accepted_provenance"]
            ):
                raise EvaluationRecordsError(
                    f"metric {metric['name']}: scorer provenance differs from approved policy"
                )
    values: list[list[float]] = [[], []]
    missing = []
    for left, right in pairs:
        if missing_pair(left, right, metric):
            missing.append(left["id"])
            continue
        for side, row in enumerate((left, right)):
            try:
                value = (
                    float(row["scores"][metric["score_key"]])
                    if recorded
                    else score(
                        metric["kind"],
                        row["expected"],
                        row["output"],
                        metric["configuration"],
                    )
                )
            except (MetricError, ValueError, OverflowError) as exc:
                raise EvaluationRecordsError(
                    f"metric {metric['name']}, record {row['id']}: {exc}"
                ) from exc
            if not math.isfinite(value):
                raise EvaluationRecordsError("record score must be finite")
            values[side].append(value)
    result: dict[str, Any] = {
        "name": metric["name"],
        "slice": slice_name,
        "kind": metric["kind"],
        "unit": metric["unit"],
        "direction": metric["direction"],
        "aggregation": "mean",
        "scoring_assurance": "recorded" if recorded else "recomputed",
        "count": len(pairs),
        "missing_ids": missing,
        "baseline_mean": None,
        "subject_mean": None,
        "delta": None,
        "interval": None,
        "decision": "insufficient_evidence",
        "reasons": [],
    }
    if missing or len(pairs) < metric["minimum_count"]:
        result["reasons"] = [
            "missing results" if missing else "minimum record count not met"
        ]
        return result
    left_values, right_values = values
    b = math.fsum(left_values) / len(left_values)
    c = math.fsum(right_values) / len(right_values)
    seed = digest(
        [
            {key: left[key] for key in ("id", "input", "expected", "metadata")}
            for left, _ in pairs
        ]
    )
    interval = _interval(
        left_values,
        right_values,
        seed,
        metric["kind"] in _BINARY_METRICS,
    )
    if not all(
        math.isfinite(v) for v in (b, c, c - b, interval["lower"], interval["upper"])
    ):
        raise EvaluationRecordsError("metric arithmetic exceeded finite numeric range")
    result.update(baseline_mean=b, subject_mean=c, delta=c - b, interval=interval)
    reasons = []
    if (
        metric["direction"] == "higher"
        and interval["lower"] < -metric["maximum_regression"]
    ):
        reasons.append("lower interval bound exceeds allowed regression")
    if (
        metric["direction"] == "lower"
        and interval["upper"] > metric["maximum_regression"]
    ):
        reasons.append("upper interval bound exceeds allowed regression")
    if c < metric.get("subject_minimum", -math.inf) or c > metric.get(
        "subject_maximum", math.inf
    ):
        reasons.append("subject mean fails absolute bounds")
    if reasons:
        result.update(decision="regression", reasons=reasons)
    elif interval["upper"] - interval["lower"] > metric["maximum_interval_width"]:
        result["reasons"] = ["interval is too wide"]
    else:
        result["decision"] = "pass"
    return result


def compare_runs(
    baseline: dict[str, Any],
    subject: dict[str, Any],
    policy: dict[str, Any],
    *,
    max_bootstrap_draws: int | None = DEFAULT_MAX_BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Check an approved policy against captured paired records without inference."""
    if max_bootstrap_draws is not None and (
        type(max_bootstrap_draws) is not int or max_bootstrap_draws < 0
    ):
        raise EvaluationRecordsError(
            "max_bootstrap_draws must be a non-negative integer"
        )
    _check_run(baseline)
    _check_run(subject)
    _check_policy(policy)
    if "expected_case_set_digest" in policy:
        for run in (baseline, subject):
            validate_run_case_set(run, policy["expected_case_set_digest"])
    baseline_rows = {r["id"]: r for r in baseline["records"]}
    candidate_rows = {r["id"]: r for r in subject["records"]}
    if baseline_rows.keys() != candidate_rows.keys():
        raise EvaluationRecordsError(
            "baseline/subject record IDs differ; export the complete paired schedule"
        )
    pairs = []
    for record_id in sorted(baseline_rows):
        left, right = baseline_rows[record_id], candidate_rows[record_id]
        for key in ("input", "expected", "metadata"):
            if canonical_json_bytes(left[key]) != canonical_json_bytes(right[key]):
                raise EvaluationRecordsError(
                    f"record {record_id}: {key} changed between runs"
                )
        pairs.append((left, right))
    scopes = []
    for subset in [{"name": "overall", "where": {}}, *policy["slices"]]:
        selected = [
            (b, c)
            for b, c in pairs
            if all(b["metadata"].get(k) == v for k, v in subset["where"].items())
        ]
        scopes.append((subset["name"], selected))
    # Charge the complete planned policy before evaluating even its first metric.
    # This conservative bound includes overlapping scopes and does not depend on
    # favorable scores, constant differences, or missing observations.
    check_bootstrap_budget(
        policy, sum(len(selected) for _, selected in scopes), max_bootstrap_draws
    )
    check_missing_id_capacity(
        scopes, policy["metrics"], byte_limit=contracts.MAX_INPUT_BYTES
    )
    results = []
    for slice_name, selected in scopes:
        for metric in policy["metrics"]:
            results.append(
                _metric_result(baseline, subject, metric, slice_name, selected)
            )
    decisions = {r["decision"] for r in results}
    decision = (
        "regression"
        if "regression" in decisions
        else "insufficient_evidence"
        if "insufficient_evidence" in decisions
        else "pass"
    )
    result = {
        "format": "invarlock/multi-metric-comparison-v1",
        "decision": decision,
        "bindings": {
            "baseline": digest(baseline),
            "subject": digest(subject),
            "policy": digest(policy),
        },
        "metrics": results,
        "limitations": [
            "Recorded scores retain source judgments; only their aggregation is recomputed.",
            "Intervals describe the paired schedule under the stated method; representativeness requires an external sampling design.",
            "Metric and slice intervals are marginal, not a simultaneous family-wide confidence guarantee.",
            "A policy pass does not establish truthful model execution, general quality, safety or compliance.",
        ],
    }
    validate(result, "comparison")
    return result
