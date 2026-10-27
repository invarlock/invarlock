"""Client handovers keep interpretation, evidence locations and trust boundaries."""

import hashlib
import json
from dataclasses import replace

import pytest

from invarlock.evidence_reporting import render_evidence
from invarlock.record_reporting import _view
from invarlock.report_presentation import render_html, render_markdown
from tests.reporting.test_report_experience import evidence, pack_json


def test_authored_review_is_complete_separate_and_escaped_in_both_formats():
    from invarlock.report_presentation import AuthoredReview

    pack = evidence()
    original = _view(pack_json(pack, "report"), pack)
    review = AuthoredReview(
        label="Fictional worked example · Authored scenario",
        byline="Alex Example, release owner · 24 October 2026",
        context=(("Intended use", "Internal editorial evaluation <only>"),),
        recommendation="Retain the baseline. Supporting record: reports/evaluation.report.json#/metrics/0",
        decision="Replacement declined for the editorial workflow.",
        basis=(("Sample rationale", "Use the complete retained review set."),),
        next_action="Alex will review a task-specific candidate on 30 October 2026.",
    )
    view = replace(original, review=review)
    for render in (render_html, render_markdown):
        output = render(view)
        assert review.label in output
        assert review.byline in output
        assert review.decision in output
        assert output.index(review.decision) < output.index("Recorded policy result")
        assert "Intended use and business context: Not supplied" not in output
        assert "Decision owner and recommendation: Not supplied" not in output
        assert "Organizational approval: Not recorded by this report" not in output
        assert "&lt;only&gt;" in output and "<only>" not in output
        assert review.next_action in output
        assert review.basis[0][1] in output
        assert original.metrics[0].evidence_source in output
        assert "Not performed by report" in output
    assert view.metrics == original.metrics
    assert view.technical == original.technical
    assert "Intended use and business context: Not supplied" in render_html(original)


def test_decision_context_and_limitations_precede_statistics_in_both_formats():
    pack = evidence()
    view = _view(pack_json(pack, "report"), pack)
    for render in (render_html, render_markdown):
        output = render(view)
        assert output.index("Decision context") < output.index("Recorded policy result")
        assert output.index("Scope and limitations") < output.index("Decision checks")
        assert output.index("Next decision") < output.index("Decision checks")
        assert "Intended use and business context: Not supplied" in output
        assert "Decision owner and recommendation: Not supplied" in output
        assert "Organizational approval: Not recorded by this report" in output
        assert "Generated interpretation" in output
        assert "reports/evaluation.report.json#/metrics/0" in output
        for _, subject in view.subjects:
            assert output.index(subject) < output.index("Recorded policy result")


def test_sorted_findings_point_to_original_metric_positions():
    from invarlock.evaluation_records.templates import example_project
    from tests._evaluation_support import build_pack

    baseline, subject, policy = example_project("classification")
    policy["slices"] = []
    policy["metrics"] = [
        dict(policy["metrics"][0], name=name) for name in ("good", "bad")
    ]
    policy["metrics"][0].update(subject_minimum=0, maximum_regression=1)
    policy["metrics"][1]["subject_minimum"] = 1
    subject["records"][0]["output"] = "wrong"
    pack = build_pack(baseline, subject, policy)
    view = _view(pack_json(pack, "report"), pack)
    assert view.metrics[0].name == "bad"
    assert (
        view.metrics[0].evidence_source == "reports/evaluation.report.json#/metrics/1"
    )
    html = render_html(view)
    assert 'href="#metric-result-1"' in html
    assert html.index("Findings to review") < html.index("Decision checks")
    assert (
        "bad"
        in html[html.index("Findings to review") : html.index("Scope and limitations")]
    )


@pytest.mark.parametrize(
    "decision,action",
    [
        ("pass", "Confirm the intended use"),
        ("regression", "Review the unmet requirements"),
        ("insufficient_evidence", "Resolve the missing or insufficient evidence"),
        ("unknown", "Establish the recorded outcome"),
    ],
)
def test_opening_names_the_review_action_without_inventing_owner_approval(
    decision, action
):
    pack = evidence()
    view = replace(_view(pack_json(pack, "report"), pack), decision=decision)
    for render in (render_html, render_markdown):
        output = render(view)
        assert output.index("Generated interpretation") < output.index(
            "Recorded policy result"
        )
        assert output.index(action) < output.index("Findings to review")
        assert "Decision owner and recommendation: Not supplied" in output


def test_opening_keeps_recorded_change_and_points_to_remaining_context():
    pack = evidence()
    view = replace(
        _view(pack_json(pack, "report"), pack),
        changes=(
            "A different model artifact was recorded.",
            "Runtime settings differ.",
        ),
    )
    for render in (render_html, render_markdown):
        output = render(view)
        assert output.index(view.changes[0]) < output.index("Recorded policy result")
        assert "1 additional comparison note" in output
        assert view.changes[1] in output


def test_unknown_and_empty_views_do_not_visually_assert_failure_or_silently_omit_gaps():
    pack = evidence()
    view = replace(
        _view(pack_json(pack, "report"), pack),
        decision="unknown",
        metrics=(),
        limitations=(),
        assurance=(),
        next_steps=(),
    )
    html = render_html(view)
    assert 'class="hero unavailable"' in html
    for output in (html, render_markdown(view)):
        assert "Metric results are unavailable in this view" in output
        assert "Limitations were not supplied" in output
        assert "Reporting checks were not supplied" in output
    assert "<ol></ol>" not in html
    assert "### Next steps" not in render_markdown(view)


def test_findings_preview_retains_every_result_and_its_original_source():
    pack = evidence()
    view = _view(pack_json(pack, "report"), pack)
    view = replace(
        view,
        metrics=tuple(
            replace(
                view.metrics[0],
                name=f"metric_{i}",
                decision="regression",
                evidence_source=f"reports/evaluation.report.json#/metrics/{i}",
            )
            for i in range(7)
        ),
    )
    for output in (render_html(view), render_markdown(view)):
        assert "7 of 7 metric / scope results need review" in output
        assert "First 5 shown here" in output
        for metric in view.metrics:
            assert metric.evidence_source in output


def test_findings_label_outcomes_and_missing_check_details_explicitly():
    pack = evidence()
    view = _view(pack_json(pack, "report"), pack)
    view = replace(
        view,
        metrics=tuple(
            replace(view.metrics[0], name=f"metric_{i}", decision=decision, checks=())
            for i, decision in enumerate(
                ("pass", "regression", "insufficient_evidence", "unknown")
            )
        ),
    )
    html = render_html(view)
    brief = html[html.index("Findings to review") : html.index("Scope and limitations")]
    assert "Policy not met" in brief
    assert "More evidence needed" in brief
    assert "Decision unavailable" in brief
    for output in (html, render_markdown(view)):
        assert (
            "1 satisfied · 1 not met · 1 need more evidence · 1 unavailable" in output
        )
        assert "Decision check details were not supplied" in output


@pytest.mark.parametrize("family", ["native", "captured", "judge", "combined"])
def test_export_has_resolvable_sources_and_preserves_every_evidence_byte(
    tmp_path, family
):
    if family in {"native", "captured"}:
        from tests.evidence_packs.test_shared_report_v2 import _pack

        root = _pack(tmp_path, family == "captured")
    elif family == "judge":
        from tests.judge_measurements.test_evidence_acceptance import _publish

        root = _publish(tmp_path)[0].path
    else:
        from tests.evidence_sets.test_verification import fixture

        root = fixture(tmp_path)[0]
    before = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    html, markdown = tmp_path / "review.html", tmp_path / "review.md"
    render_evidence(root, html_path=html, markdown_path=markdown)
    assert before == {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    import re

    sources = re.findall(r"Supporting record: ([^<\n]+)", html.read_text())
    assert sources
    for source in sources:
        path, pointer = source.split("#", 1)
        value = json.loads((root / path).read_bytes())
        for key in pointer.strip("/").split("/") if pointer else []:
            value = value[int(key)] if isinstance(value, list) else value[key]
        assert value is not None
        assert source in markdown.read_text()
    assert "Scope and limitations" in markdown.read_text()
    # Signatures and all payload digests are covered, not just a selected manifest.
    assert all(
        hashlib.sha256(data).digest()
        == hashlib.sha256((root / path).read_bytes()).digest()
        for path, data in before.items()
    )
