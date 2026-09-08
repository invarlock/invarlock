"""Rendering must preserve missing context and reject signed contradictions."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from invarlock.evidence_pack_contract import build_comparison_report
from invarlock.evidence_reporting import EvidenceReportError, render_evidence
from invarlock.report_presentation import (
    CheckView,
    MetricView,
    ReportView,
    render_html,
    render_markdown,
)
from tests.cli.test_import_journey import (
    _materialize_request,
    _text_scorer_registry_and_binding,
)
from tests.evidence_packs.test_evidence_reporting import _evidence, _report


def test_display_without_optional_context_does_not_invent_evidence() -> None:
    view = ReportView(
        title="Partial comparison",
        family="Recorded results",
        decision="insufficient_evidence",
        summary="A measurement is unavailable.",
        assurance=(("Recipient acceptance", "Not performed"),),
        metrics=(
            MetricView(
                name="Accuracy",
                scope="Recorded cases",
                decision="insufficient_evidence",
                baseline="Unavailable",
                candidate="Unavailable",
                change="Unavailable",
                count="0",
                explanation="No complete pairs.",
                checks=(CheckView("Interval", "Unavailable", "Required", None),),
            ),
        ),
    )

    html = render_html(view)
    markdown = render_markdown(view)

    assert 'class="notes"' not in html
    assert 'class="limits"' not in html
    assert "Comparison and policy identities" not in html
    assert 'class="subjects"' not in html
    assert "<svg" not in html
    assert 'class="check-unknown">Unavailable' in html
    for text in (html, markdown):
        assert "More evidence needed" in text
        assert "No complete pairs" in text
        assert "Not performed" in text
        assert "Policy satisfied" not in text


@pytest.fixture
def extension_report(tmp_path: Path) -> dict:
    registry, binding = _text_scorer_registry_and_binding()
    material = _materialize_request(
        tmp_path, scorer_binding=binding, scorer_registry=registry
    )
    return build_comparison_report(
        comparison_id="model-comparison",
        paired_records=json.loads(material["records"].read_bytes()),
        policy=json.loads(material["policy"].read_bytes()),
        policy_digest="sha256:" + "a" * 64,
    )


def test_signed_extension_report_preserves_score_and_schedule_scope(
    tmp_path: Path, extension_report: dict
) -> None:
    before = deepcopy(extension_report)
    evidence, _ = _evidence(tmp_path, report_payload=extension_report)
    destination = tmp_path / "report.html"

    result = render_evidence(evidence, html_path=destination)

    assert "| 1 score | 0.5 score | -50 pp | 2 |" in result.text
    for text in (result.text, destination.read_text()):
        assert "finite-schedule resampling interval" in text
        assert "not population uncertainty" in text
        assert "Not performed by report" in text
        assert "Paired 95% confidence interval" not in text
    assert extension_report == before


@pytest.mark.parametrize("mutation", ["binding", "metric", "replay"])
def test_signed_extension_binding_contradictions_do_not_publish_html(
    tmp_path: Path, extension_report: dict, mutation: str
) -> None:
    if mutation == "binding":
        extension_report["scorer_extension"] = {}
    elif mutation == "metric":
        extension_report["metric"] = "example.different_scorer"
    else:
        extension_report["scorer_replay"].pop("subject")
    evidence, _ = _evidence(tmp_path, report_payload=extension_report)
    destination = tmp_path / "must-not-exist.html"

    with pytest.raises(EvidenceReportError, match="scorer"):
        render_evidence(evidence, html_path=destination)

    assert not destination.exists()


@pytest.mark.parametrize("invalid_count", [True, -1, 0.5])
def test_signed_paired_counts_require_nonnegative_integers(
    tmp_path: Path, invalid_count: object
) -> None:
    report = _report()
    report["paired_binary"]["discordant_pairs"] = invalid_count
    evidence, _ = _evidence(tmp_path, report_payload=report)

    with pytest.raises(EvidenceReportError, match="non-negative integer"):
        render_evidence(evidence)


def test_older_signed_report_cannot_claim_new_side_accuracy_contract(
    tmp_path: Path,
) -> None:
    report = _report()
    report["format"] = "invarlock/comparison-report-v2"
    report["side_accuracy"] = {"qualified": True}
    evidence, _ = _evidence(tmp_path, report_payload=report)

    with pytest.raises(EvidenceReportError, match="requires comparison-report-v3"):
        render_evidence(evidence)
