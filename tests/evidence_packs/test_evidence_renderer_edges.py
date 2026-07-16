from __future__ import annotations

from pathlib import Path

import pytest

from invarlock import evidence_reporting as reporting
from invarlock.evidence_pack_contract import build_comparison_report
from invarlock.evidence_reporting import EvidenceReportError, render_evidence
from tests.evidence_packs.test_evidence_reporting import _evidence, _report


def _nll_report() -> dict[str, object]:
    return build_comparison_report(
        comparison_id="model-comparison",
        paired_records={
            "format": "invarlock/paired-records-v1",
            "metric": "normalized_nll_per_utf8_byte",
            "schedule_sha256": "0" * 64,
            "records": [
                {
                    "record_id": "one",
                    "baseline": {"score": 1.0},
                    "subject": {"score": 1.05},
                },
                {
                    "record_id": "two",
                    "baseline": {"score": 1.0},
                    "subject": {"score": 1.05},
                },
            ],
            "derived_measurements": {
                "perplexity_ratio": {
                    "status": "unavailable",
                    "basis": "authenticated_target_likelihood",
                    "method": "target_token_weighted_perplexity_ratio_v1",
                    "reason": "target_token_counts_unavailable",
                }
            },
        },
        policy={
            "resolved_policy": {
                "metrics": {"normalized_nll_per_utf8_byte": {"ratio_max": 1.1}}
            }
        },
        policy_digest="sha256:" + "a" * 64,
    )


def test_closed_report_renderer_covers_normalized_nll_and_html_escaping() -> None:
    report = _nll_report()
    report["comparison_id"] = "<comparison>"

    closed = reporting._closed_comparison_report(report)
    markdown = reporting._render_markdown(
        closed,
        explain=False,
        evidence_signer="sha256:" + "a" * 64,
        observations=(),
    )
    html = reporting._render_html(
        closed,
        explain=True,
        evidence_signer="<unsafe-signer>",
        observations=(),
    )

    assert "Normalized NLL ratio" in markdown
    assert "Maximum allowed ratio" in markdown
    assert "&lt;comparison&gt;" in html
    assert "&lt;unsafe-signer&gt;" in html
    assert "manifest, checksums" in html


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(extra=True), "fields are invalid"),
        (lambda value: value.update(format="other"), "format is invalid"),
        (lambda value: value.update(comparison_id=""), "comparison_id is invalid"),
        (lambda value: value.update(record_count=True), "record_count is invalid"),
        (lambda value: value.update(baseline=[]), "baseline is invalid"),
        (
            lambda value: value["baseline"].update(mean_score=True),
            "baseline.mean_score must be a number",
        ),
        (lambda value: value.update(comparison=[]), "comparison is invalid"),
        (
            lambda value: value["comparison"].update(kind="other"),
            "comparison is invalid",
        ),
        (
            lambda value: value["comparison"].update(value=True),
            "comparison.value must be a number",
        ),
        (
            lambda value: value["comparison"].update(value=float("inf")),
            "comparison.value must be finite",
        ),
        (lambda value: value.update(verdict="unknown"), "verdict is invalid"),
    ],
)
def test_closed_report_rejects_open_or_ambiguous_display_fields(
    mutate: object, message: str
) -> None:
    report = _report()
    assert callable(mutate)
    mutate(report)

    with pytest.raises(EvidenceReportError, match=message):
        reporting._closed_comparison_report(report)


def test_closed_report_rejects_internally_contradictory_values() -> None:
    report = _report()
    report["comparison"] = {
        "kind": "exact_match_delta_pp",
        "value": 49.0,
        "minimum": -1.0,
    }
    with pytest.raises(EvidenceReportError, match="does not match the side means"):
        reporting._closed_comparison_report(report)

    report = _report()
    report["verdict"] = "fail"
    with pytest.raises(EvidenceReportError, match="verdict does not match"):
        reporting._closed_comparison_report(report)


def test_render_rejects_unsafe_evidence_root_and_html_destination(
    tmp_path: Path,
) -> None:
    evidence, _signer = _evidence(tmp_path)
    alias = tmp_path / "evidence-link"
    alias.symlink_to(evidence, target_is_directory=True)
    with pytest.raises(EvidenceReportError, match="real directory"):
        render_evidence(alias)

    with pytest.raises(EvidenceReportError, match="must name a regular file"):
        render_evidence(evidence, html_path=Path("/"))


def test_report_error_exposes_stable_exit_code() -> None:
    assert EvidenceReportError("invalid").exit_code == 2
    assert EvidenceReportError("write failed", exit_code=1).exit_code == 1
