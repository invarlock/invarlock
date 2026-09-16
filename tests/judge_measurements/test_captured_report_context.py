"""Bounded external-evaluator provenance in the judge report."""

import json

from invarlock.judge_measurements.reporting import _snapshot, _view
from invarlock.report_presentation import render_html, render_markdown
from tests.judge_measurements.test_evidence_acceptance import _publish


def test_external_evaluator_identities_visible_in_all_report_forms(tmp_path):
    publication, _ = _publish(tmp_path)
    retained, artifacts = _snapshot(publication.path)
    for side in ("baseline", "subject"):
        artifacts[f"{side}_run"]["source"] = {
            "name": "existing-evaluator",
            "version": "2",
        }
        artifacts[f"{side}_run"]["source_digest"] = "sha256:" + "a" * 64
    view, facts = _view(retained, artifacts)
    for text in (render_html(view), render_markdown(view), json.dumps(facts)):
        assert "existing-evaluator" in text
        assert "sha256:" + "a" * 64 in text
    assert facts["comparison"]["baseline"]["source_digest"] == "sha256:" + "a" * 64
    assert "source assertions" in " ".join(value for _, value in view.assurance)


def test_projected_inputs_and_settings_have_bounded_provenance_previews(tmp_path):
    from invarlock.evaluation_record_contracts.contracts import digest
    from invarlock.evaluator_capture import _project_record

    publication, _ = _publish(tmp_path)
    retained, artifacts = _snapshot(publication.path)
    secret = "Original structured data excluded from report"
    projection = {"kind": "json-pointer", "pointer": "/input/prompt"}
    settings = {
        "temperature": 0,
        "injected": "<script>alert(1)</script>",
        "large": "x" * 10000,
    }
    for side in ("baseline", "subject"):
        run = artifacts[f"{side}_run"]
        for record in run["records"]:
            record["input"] = {"prompt": "Selected text", "extra": secret}
            record["context"] = {
                "model_id": "publisher/model",
                "model_revision": "frozen-revision",
                "settings": settings,
                "effective_messages": [
                    {"role": "user", "content": "private user message"}
                ],
            }
            record["metadata"] = {"workflow": "source-workflow"}
        run["records"] = [
            _project_record(record, projection) for record in run["records"]
        ]
    view, facts = _view(retained, artifacts)
    captured = facts["captured_context"]["baseline"]
    assert captured["settings"]["status"] == "common"
    assert captured["settings"]["digest"] == digest(settings)
    assert len(captured["settings"]["preview"]) < 2100
    assert captured["input_projection"]["configuration_count"] == 1
    for text in (render_html(view), render_markdown(view), json.dumps(facts)):
        assert "/input/prompt" in text
        assert digest(projection) in text
        assert "source-workflow" in text
        assert "temperature" in text
        assert secret not in text
        assert "x" * 1000 not in text
    html = render_html(view)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "publisher/model" in str(view.subjects)


def test_mixed_incomplete_settings_and_missing_context_do_not_claim_uniformity(
    tmp_path,
):
    from invarlock.judge_measurements.reporting import _captured_record_context

    assert _captured_record_context([{"context": "opaque"}]) == {}
    for contexts in ([{"settings": {}}, {}], [{"settings": 1}, {"settings": 2}]):
        result = _captured_record_context(
            [{"context": context} for context in contexts]
        )
        assert result["settings"]["status"] == "mixed_or_incomplete"
        assert "preview" not in result["settings"]
        assert "digest" not in result["settings"]
    publication, _ = _publish(tmp_path)
    retained, artifacts = _snapshot(publication.path)
    records = artifacts["baseline_run"]["records"]
    for index, record in enumerate(records):
        record["context"] = {"settings": index}
    view, facts = _view(retained, artifacts)
    assert (
        facts["captured_context"]["baseline"]["settings"]["status"]
        == "mixed_or_incomplete"
    )
    assert "Mixed or incomplete across records" in render_markdown(view)


def test_projection_configuration_inventory_is_bounded():
    from invarlock.evaluator_capture import _project_record
    from invarlock.judge_measurements.reporting import _captured_record_context

    rows = []
    for index in range(10):
        name = "field" + str(index) + "x" * 500
        configuration = {"kind": "json-pointer", "pointer": "/input/" + name}
        rows.append(_project_record({"input": {name: "Selected text"}}, configuration))
    result = _captured_record_context(rows)["input_projection"]
    assert result["configuration_count"] == 10
    assert len(result["configurations"]) == 8
    assert result["present_records"] == 10
    assert len(result["configurations"][0]["pointer"]) < 300
