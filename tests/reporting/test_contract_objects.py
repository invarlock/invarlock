from __future__ import annotations

from pathlib import Path

from invarlock.reporting.evaluation_report_builder import EvaluationReportBuilder
from invarlock.reporting.verify_contract import VerifyRequest
from invarlock.runtime_provenance import (
    RuntimeProvenanceResult,
    RuntimeProvenanceVerdict,
)


def test_runtime_provenance_verdict_payload_uses_shared_status_vocabulary() -> None:
    verdict = RuntimeProvenanceVerdict.from_result(
        RuntimeProvenanceResult(verified=True, skipped=False),
        declared_mode="container",
    )

    payload = verdict.as_verification_payload()["runtime_provenance"]

    assert payload["declared_mode"] == "container"
    assert payload["status"] == "verified"
    assert payload["verified"] is True
    assert payload["strict_blocking"] is False


def test_verify_request_normalizes_tolerance_and_assurance_mode() -> None:
    request = VerifyRequest.from_args(
        [Path("report.json")],
        tolerance="bad",  # type: ignore[arg-type]
        assurance_mode="STRICT",
    )

    assert request.reports == (Path("report.json"),)
    assert request.normalized_tolerance == 1e-9
    assert request.normalized_assurance_mode == "strict"


def test_evaluation_report_builder_attaches_pending_assurance() -> None:
    report = {
        "context": {
            "assurance": {"mode": "strict"},
            "profile": "ci",
            "tier": "balanced",
            "runtime": {"execution_mode": "container"},
        },
        "guards": [
            {"name": "invariants"},
            {"name": "spectral"},
            {"name": "rmt"},
            {"name": "variance"},
            {"name": "invariants"},
        ],
        "spectral": {"supported": True, "metrics": {"ok": True}},
        "rmt": {"supported": True, "metrics": {"ok": True}},
        "variance": {"supported": True, "metrics": {"ok": True}},
        "invariants": {"supported": True, "metrics": {"ok": True}},
        "report_build": {
            "synthesized_fields": [],
            "repaired_fields": [],
            "fallback_fields": [],
        },
    }

    assurance = EvaluationReportBuilder(report).finalize_assurance()

    assert assurance["mode"] == "strict"
    assert assurance["verdict"] == "pending_verifier"
    assert assurance["runtime_provenance_declared"] == "container"
    assert assurance["runtime_provenance_verification_status"] == "pending"
