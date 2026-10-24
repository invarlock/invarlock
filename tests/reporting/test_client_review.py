"""Client handovers keep interpretation, evidence locations and trust boundaries."""

import hashlib
import json

import pytest

from invarlock.evidence_reporting import render_evidence
from invarlock.record_reporting import _view
from invarlock.report_presentation import render_html, render_markdown
from tests.reporting.test_report_experience import evidence, pack_json


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
        in html[html.index("Findings to review") : html.index("What was compared")]
    )


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
