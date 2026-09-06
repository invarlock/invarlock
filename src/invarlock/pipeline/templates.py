"""Runnable onboarding examples and reusable pipeline configuration."""

from __future__ import annotations

from typing import Any

from invarlock.pipeline.comparison import make_run
from invarlock.pipeline.contracts import digest


def example_project(kind: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if kind not in ("classification", "extraction", "judge"):
        raise ValueError("example must be classification, extraction or judge")
    provenance = {
        "kind": "judge",
        "unit": "score",
        "source": "example-judge",
        "version": "1.0.0",
        "rubric_digest": digest(
            {"rubric": "Example recorded judgment; no model was called."}
        ),
    }
    records = []
    for i in range(40):
        expected: Any = "approved" if i % 2 else "review"
        output: Any = " APPROVED " if i % 2 else "Review"
        if kind == "extraction":
            expected = {"amount": i + 10, "currency": "USD"}
            output = {"currency": "USD", "amount": i + 10}
        records.append(
            {
                "id": f"case-{i}",
                "input": f"Synthetic {kind} case {i}",
                "expected": expected,
                "output": output,
                "scores": {"quality": 0.9, "latency_ms": 120.0},
                "metadata": {"category": "routine" if i % 2 else "exception"},
            }
        )
    metric: dict[str, Any] = {
        "name": "quality",
        "kind": "normalized_match",
        "configuration": {},
        "direction": "higher",
        "unit": "score",
        "aggregation": "mean",
        "minimum_count": 10,
        "maximum_regression": 0.2,
        "maximum_interval_width": 0.4,
        "candidate_minimum": 0.8,
    }
    if kind == "extraction":
        metric.update(
            kind="json_fields", configuration={"fields": ["/amount", "/currency"]}
        )
    if kind == "judge":
        metric.update(
            kind="recorded", score_key="quality", accepted_provenance=provenance
        )
    latency_provenance = {
        "kind": "measurement",
        "unit": "milliseconds",
        "source": "example-timer",
        "version": "1.0.0",
        "rubric_digest": None,
    }
    latency = {
        "name": "latency",
        "kind": "recorded",
        "configuration": {},
        "direction": "lower",
        "unit": "milliseconds",
        "aggregation": "mean",
        "minimum_count": 10,
        "maximum_regression": 20.0,
        "maximum_interval_width": 30.0,
        "candidate_maximum": 200.0,
        "score_key": "latency_ms",
        "accepted_provenance": latency_provenance,
    }
    policy = {
        "format": "invarlock/pipeline-policy-v1",
        "metrics": [metric, latency],
        "slices": [{"name": "exceptions", "where": {"category": "exception"}}],
    }

    def run(side: str) -> dict[str, Any]:
        return make_run(
            records,
            source={"name": "synthetic-example", "version": "1.0.0"},
            run_id=side,
            artifact_digest=digest({"example_artifact": side}),
            score_provenance={"quality": provenance, "latency_ms": latency_provenance},
        )

    return run("baseline"), run("candidate"), policy
