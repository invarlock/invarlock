"""Replay captured reference-continuation likelihood facts without inference."""

from __future__ import annotations

import math
from typing import Any, cast

from invarlock.evaluation_record_contracts.contracts import (
    EvaluationRecordsError,
    digest,
)
from invarlock.evidence_pack_contract import (
    PAIRED_INTERVAL_CONFIDENCE,
    PAIRED_INTERVAL_METHOD,
    PAIRED_INTERVAL_REPLICATES,
    EvidencePackError,
    _comparison_value,
    _paired_resampling_interval,
)

LIKELIHOOD_METRIC = "normalized_nll_per_utf8_byte"


def validate_likelihood_record(
    row: dict[str, Any],
    run: dict[str, Any],
    *,
    service_identity_digest: str | None,
) -> None:
    """Cross-bind facts after schema and retained projection validation."""
    if "likelihood" not in row:
        return
    facts = row["likelihood"]
    label = f"record {row['id']}: reference-continuation likelihood"
    if not isinstance(row["expected"], str) or not row["expected"]:
        raise EvaluationRecordsError(f"{label} requires nonempty reference text")
    for key in ("token_count", "utf8_byte_count"):
        if type(facts[key]) is not int or facts[key] <= 0:
            raise EvaluationRecordsError(
                f"{label} {key} must be a positive exact integer"
            )
    if facts["utf8_byte_count"] != len(row["expected"].encode("utf-8")):
        raise EvaluationRecordsError(f"{label} utf8_byte_count differs from reference")
    # A judge's text projection does not change the input on which the upstream
    # evaluator measured likelihood. Its original input remains authenticated
    # by the projection binding, which _check_run verifies before this check.
    context = row["context"]
    captured_input = (
        context["input_projection"]["source"]["input"]
        if isinstance(context, dict) and "input_projection" in context
        else row["input"]
    )
    if "service_identity" in run:
        if (
            service_identity_digest is None
            or facts.get("service_identity_digest") != service_identity_digest
        ):
            raise EvaluationRecordsError(
                f"{label} service_identity_digest binding differs"
            )
        if (
            facts["configuration_digest"]
            != run["service_identity"]["configuration_digest"]
        ):
            raise EvaluationRecordsError(
                f"{label} service configuration binding differs"
            )
    elif "service_identity_digest" in facts:
        raise EvaluationRecordsError(
            f"{label} service identity cannot replace an artifact"
        )
    for key, expected in (
        ("input_digest", digest(captured_input)),
        ("reference_digest", digest(row["expected"])),
        ("artifact_digest", run["artifact_digest"]),
        ("source", run["source"]),
    ):
        if facts[key] != expected:
            raise EvaluationRecordsError(f"{label} {key} binding differs")


def validate_likelihood_policy(metric: dict[str, Any]) -> None:
    """Validate semantics after the closed policy schema has been checked."""
    if metric["direction"] != "lower" or metric["unit"] != "nats_per_utf8_byte":
        raise EvaluationRecordsError(
            "normalized NLL uses lower-is-better nats_per_utf8_byte"
        )
    if "maximum_regression" in metric:
        raise EvaluationRecordsError(
            "normalized NLL requires ratio_max, not maximum_regression"
        )


def validate_likelihood_bindings(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], metric: dict[str, Any]
) -> None:
    """Check facts even on errored rows before applying missingness."""
    configuration = metric["configuration"]
    for pair in pairs:
        for side, row in zip(("baseline", "subject"), pair, strict=True):
            if "likelihood" not in row:
                continue
            facts = row["likelihood"]
            if facts["configuration_digest"] != configuration["configuration_digest"]:
                raise EvaluationRecordsError(
                    f"record {row['id']}: likelihood configuration differs from policy"
                )
            if facts["tokenizer_digest"] != configuration[side + "_tokenizer_digest"]:
                raise EvaluationRecordsError(
                    f"record {row['id']}: likelihood tokenizer differs from policy"
                )


def likelihood_value(row: dict[str, Any]) -> float:
    facts = row["likelihood"]
    return -float(facts["logprob_sum"]) / cast(int, facts["utf8_byte_count"])


def likelihood_statistics(
    baseline: list[float], subject: list[float], seed: str
) -> tuple[float, dict[str, Any]]:
    """Use native normalized-NLL mean-ratio arithmetic and paired intervals."""
    try:
        ratio = _comparison_value(LIKELIHOOD_METRIC, baseline, subject)
        lower, upper = _paired_resampling_interval(
            metric=LIKELIHOOD_METRIC,
            baseline_scores=baseline,
            subject_scores=subject,
            schedule_sha256=seed.removeprefix("sha256:"),
        )
    except (EvidencePackError, ValueError, OverflowError) as exc:
        raise EvaluationRecordsError(f"captured normalized NLL: {exc}") from exc
    if not all(math.isfinite(value) for value in (ratio, lower, upper)):
        raise EvaluationRecordsError(
            "normalized NLL ratio arithmetic exceeded finite numeric range"
        )
    return ratio, {
        "lower": lower,
        "upper": upper,
        "method": PAIRED_INTERVAL_METHOD,
        "mass": PAIRED_INTERVAL_CONFIDENCE,
        "replicates": PAIRED_INTERVAL_REPLICATES,
    }
