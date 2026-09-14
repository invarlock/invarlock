"""Only opted-in per-case gold references receive bounded report disclosure."""

import json

import pytest

from invarlock.judge_measurements.reporting import TEXT_DETAIL_LIMIT, _snapshot, _view
from invarlock.report_presentation import render_html, render_markdown
from tests.judge_measurements.test_evidence_acceptance import _publish
from tests.judge_measurements.test_native_capture import _publication


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("mode", [None, "none", "per_case"])
def test_reference_mode_is_visible_and_reference_disclosure_is_opt_in(
    tmp_path, native, mode
):
    publication = _publication(tmp_path) if native else _publish(tmp_path)[0]
    retained, artifacts = _snapshot(publication.path)
    if mode is not None:
        artifacts["plan"]["prompt"]["reference_mode"] = mode
    reference = (
        "<script>gold reference</script>" + "r" * TEXT_DETAIL_LIMIT + "PRIVATE-TAIL"
    )
    for side in ("baseline", "subject"):
        for row in artifacts[f"{side}_run"]["records"]:
            row["expected"] = reference
    case_id = artifacts["baseline_run"]["records"][0]["id"]
    view, facts = _view(retained, artifacts, case_ids=(case_id,))
    html, markdown = render_html(view), render_markdown(view, include_details=True)
    json_text = json.dumps(facts)
    if mode == "per_case":
        assert facts["prompt"]["reference_mode"] == "per_case"
        assert facts["per_case_references"] == [
            {"case_id": case_id, "reference_excerpt": reference[:TEXT_DETAIL_LIMIT]}
        ]
        detail = dict(view.details)[case_id]
        assert detail["reference_excerpt"] == reference[:TEXT_DETAIL_LIMIT]
        assert "Per-case references" in html and "Per-case references" in markdown
        assert "&lt;script&gt;gold reference&lt;/script&gt;" in html
        assert "<script>gold reference</script>" not in html
        assert "gold reference" in markdown and "gold reference" in json_text
    else:
        assert "reference_mode" not in facts["prompt"]
        assert "per_case_references" not in facts
        assert "reference_excerpt" not in dict(view.details)[case_id]
        assert "gold reference" not in html + markdown + json_text
    assert "PRIVATE-TAIL" not in html + markdown + json_text
