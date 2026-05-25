"""Builder helpers for assurance-critical evaluation report finalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from invarlock.core.assurance_contract import (
    build_assurance_section,
    resolve_report_runtime_provenance_declared,
)

from . import report_build_evidence


@dataclass
class ReportBuildContext:
    evaluation_report: dict[str, Any]

    def ensure_evidence(self) -> dict[str, Any]:
        return report_build_evidence.ensure_report_build_evidence(
            self.evaluation_report
        )

    def has_repair_or_fallback_events(self) -> bool:
        self.ensure_evidence()
        return report_build_evidence.report_build_has_evidence_events(
            self.evaluation_report
        )

    def attach_pending_assurance(self) -> dict[str, Any]:
        self.ensure_evidence()
        assurance = build_assurance_section(
            self.evaluation_report,
            fallback_fields_used=self.has_repair_or_fallback_events(),
            runtime_provenance_verified=None,
            runtime_provenance_declared=resolve_report_runtime_provenance_declared(
                self.evaluation_report
            ),
            runtime_provenance_verification_status="pending",
        )
        self.evaluation_report["assurance"] = assurance
        return assurance


class EvaluationReportBuilder:
    def __init__(self, evaluation_report: dict[str, Any]) -> None:
        self.context = ReportBuildContext(evaluation_report=evaluation_report)

    def finalize_assurance(self) -> dict[str, Any]:
        return self.context.attach_pending_assurance()


__all__ = ["EvaluationReportBuilder", "ReportBuildContext"]
