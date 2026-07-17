"""Public facade for the canonical single-comparison evidence bundle."""

from invarlock.evidence_pack_contract import (
    COMPARISON_REPORT_FORMAT,
    EVIDENCE_PACK_FORMAT,
    EVIDENCE_PATHS,
    EvidenceObservation,
    EvidencePackError,
    InputIdentity,
    RuntimeSideEvidence,
    build_comparison_report,
    derive_paired_records,
)
from invarlock.evidence_pack_publication import (
    EvidencePublication,
    publish_comparison_evidence,
)
from invarlock.evidence_pack_verification import verify_comparison_evidence

__all__ = [
    "COMPARISON_REPORT_FORMAT",
    "EVIDENCE_PACK_FORMAT",
    "EVIDENCE_PATHS",
    "EvidencePackError",
    "EvidenceObservation",
    "InputIdentity",
    "RuntimeSideEvidence",
    "EvidencePublication",
    "build_comparison_report",
    "derive_paired_records",
    "publish_comparison_evidence",
    "verify_comparison_evidence",
]
