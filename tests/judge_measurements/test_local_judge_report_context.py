"""Runtime-backed judging is distinguished from imported rating declarations."""

import pytest

from invarlock.judge_measurements.reporting import _snapshot, _view
from invarlock.report_presentation import render_html, render_markdown
from tests.judge_measurements.test_evidence_acceptance import _publish


@pytest.mark.parametrize("runtime_backed", [False, True])
def test_local_judge_identity_claim_requires_runtime_source(tmp_path, runtime_backed):
    publication, _ = _publish(tmp_path)
    retained, artifacts = _snapshot(publication.path)
    identity = "a" * 64
    artifacts["plan"]["judge"]["model_identity"] = {
        "kind": "local_weights",
        "weights_sha256": identity,
    }
    if runtime_backed:
        artifacts["measurements"]["source_profile"] = (
            "retained-runtime-provider-judge-v1"
        )
    view, facts = _view(retained, artifacts)
    assert facts["judge"]["model_identity"]["weights_sha256"] == identity
    for rendered in (render_html(view), render_markdown(view)):
        assert ("Judge execution evidence" in rendered) is runtime_backed
        assert ("Judge artifact identity" in rendered) is runtime_backed
    if runtime_backed:
        assert dict(view.identity)["Judge artifact identity"] == identity
        execution_note = dict(view.assurance)["Judge execution evidence"]
        assert "does not rerun the judge" in execution_note
