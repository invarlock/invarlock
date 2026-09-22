"""Runtime-backed judging is distinguished from imported rating declarations."""

import pytest

from invarlock.judge_measurements.contracts import canonical_payload
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
        assert dict(view.context)["Judge prompt format"] == "Canonical JSON"
        assert facts["prompt"]["runtime_format"] == "canonical-json-v1"
        execution_note = dict(view.assurance)["Judge execution evidence"]
        assert "does not rerun the judge" in execution_note


def test_runtime_chat_format_is_disclosed_in_report_and_json(tmp_path):
    publication, _ = _publish(tmp_path)
    retained, artifacts = _snapshot(publication.path)
    artifacts["measurements"]["source_profile"] = "retained-runtime-provider-judge-v1"
    artifacts["plan"]["prompt"]["runtime_format"] = "chatml-v1"
    artifacts["plan"]["judge"]["model_identity"] = {
        "kind": "local_weights",
        "weights_sha256": "a" * 64,
    }
    view, facts = _view(retained, artifacts)
    assert dict(view.context)["Judge prompt format"] == "ChatML"
    assert facts["prompt"]["runtime_format"] == "chatml-v1"


@pytest.mark.parametrize(
    ("service", "display"),
    [
        ("ollama", "Ollama"),
        ("lm_studio", "LM Studio"),
        ("vllm", "vLLM"),
        ("openai_compatible", "OpenAI-compatible service"),
    ],
)
@pytest.mark.parametrize("response_format", [None, "json_schema"])
def test_service_judge_report_does_not_claim_local_artifact_authentication(
    tmp_path, service, display, response_format
):
    publication, _ = _publish(tmp_path)
    retained, artifacts = _snapshot(publication.path)
    artifacts["measurements"]["source_profile"] = "retained-openai-compatible-judge-v1"
    artifacts["measurements"]["sources"][0]["content"] = canonical_payload(
        {
            "collection": {
                "service": service,
                "base_url": "http://localhost:1234/v1/",
                **({"response_format": response_format} if response_format else {}),
            },
            "service_identity": {"endpoint_sha256": "a" * 64},
        }
    ).decode()
    view, facts = _view(retained, artifacts)
    assert facts["judge_service"] == {
        "service": service,
        "base_url": "http://localhost:1234/v1/",
        "endpoint_sha256": "a" * 64,
        "model_identity_basis": "service_assertion",
        "response_format": response_format or "json_object",
    }
    assert dict(view.context)["Judge service"] == display
    assert dict(view.context)["Judge endpoint"] == "http://localhost:1234/v1/"
    assert dict(view.context)["Judge response format"] == (
        "JSON schema" if response_format else "JSON object"
    )
    assert dict(view.identity)["Judge endpoint digest"] == "a" * 64
    assert "Judge artifact identity" not in dict(view.identity)
    explanation = dict(view.assurance)["Judge service evidence"]
    assert "model files and execution environment are not authenticated" in explanation
    assert "does not call the service again" in explanation
    for rendered in (render_html(view), render_markdown(view)):
        assert display in rendered
        assert "Judge service evidence" in rendered
        assert "Judge execution evidence" not in rendered
