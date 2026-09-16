"""Resume preserves prior capture and rejects unsafe or changed workspaces."""

import base64
import json
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock.evidence_pack_contract import RuntimeSideEvidence
from invarlock.judge_measurements import native_workflow as workflow
from invarlock.judge_measurements.native_capture import validate_native_capture
from tests.judge_measurements.test_native_capture import _capture


def test_lock_excludes_second_writer_and_detects_replacement(tmp_path):
    path = tmp_path / "workspace"
    with pytest.raises(workflow.JudgeWorkflowError, match="lock changed"):
        with workflow.locked_workspace(path) as unchanged:
            with pytest.raises(workflow.JudgeWorkflowError, match="another evaluation"):
                with workflow.locked_workspace(path):
                    pytest.fail("second workspace acquired")
            (path / ".evaluation.lock").unlink()
            unchanged()


def test_private_workspace_and_no_symlink_ancestry(tmp_path):
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(workflow.JudgeWorkflowError, match="private"):
        with workflow.locked_workspace(public):
            pass
    link = tmp_path / "link"
    link.symlink_to(public, target_is_directory=True)
    with pytest.raises(OSError):
        with workflow.locked_workspace(link):
            pass


def test_identity_is_compact_and_rejects_any_changed_decision(tmp_path):
    path = tmp_path / "identity.json"
    value = {"large": "a" * (5 * 1024 * 1024)}
    workflow._retain_identity(path, value)
    assert path.stat().st_size < 1024
    workflow._retain_identity(path, value)
    with pytest.raises(workflow.JudgeWorkflowError, match="different frozen request"):
        workflow._retain_identity(path, {"large": "different"})


@pytest.fixture
def native(tmp_path, monkeypatch):
    frozen = _capture(tmp_path / "source")
    baseline, subject = validate_native_capture(frozen)
    request = SimpleNamespace(
        comparison=SimpleNamespace(
            judge=SimpleNamespace(
                workspace=tmp_path / "workspace", signer_identity="native-signer"
            )
        ),
        execution=SimpleNamespace(mode="import"),
        output=SimpleNamespace(evidence=tmp_path / "result"),
    )
    monkeypatch.setattr(workflow, "collection_preflight", lambda *_: {})
    monkeypatch.setattr(
        workflow,
        "collection_api",
        lambda: SimpleNamespace(
            CollectionOptions=lambda **kwargs: kwargs,
            prepare_collection=lambda *_args, **_kwargs: {"budget_exhausted": False},
        ),
    )
    kwargs = {
        "request": request,
        "normalized_request": frozen["normalized_request"],
        "schedule": frozen["schedule"],
        "policy_bytes": base64.b64decode(frozen["policy_base64"]),
        "signing_key": Ed25519PrivateKey.generate(),
        "expected_artifact_digests": {
            "baseline": baseline["artifact_digest"],
            "subject": subject["artifact_digest"],
        },
        "expected_runtime_digests": {
            side: run["records"][0]["context"]["runtime_digest"]
            for side, run in (("baseline", baseline), ("subject", subject))
        },
    }
    return frozen, kwargs


def test_ambiguous_capture_is_never_reexecuted(native, monkeypatch):
    _, kwargs = native
    calls = []

    def interrupted():
        calls.append(1)
        raise RuntimeError("runtime stopped after admission")

    with pytest.raises(RuntimeError, match="stopped"):
        workflow.evaluate_native_judge(**kwargs, capture=interrupted)
    with pytest.raises(
        workflow.JudgeWorkflowError, match="Previous native answer capture"
    ):
        workflow.evaluate_native_judge(**kwargs, capture=interrupted)
    assert calls == [1]
    assert not kwargs["request"].output.evidence.exists()


def test_partial_collection_resumes_without_regenerating_answers(native, monkeypatch):
    frozen, kwargs = native
    captures = []

    def capture():
        captures.append(1)
        return tuple(
            RuntimeSideEvidence(
                **{
                    name: base64.b64decode(value)
                    for name, value in frozen[side].items()
                }
            )
            for side in ("baseline", "subject")
        )

    def timed_out(**kwargs):
        kwargs["status"]["stop_reason"] = "deadline"
        return {"trials": [{"attempts": []}]}

    monkeypatch.setattr(workflow, "collect_frozen", timed_out)
    for _ in range(2):
        with pytest.raises(workflow.JudgeWorkflowError) as error:
            workflow.evaluate_native_judge(**kwargs, capture=capture)
        assert error.value.payload["resumable"]
    assert captures == [1]
    assert not kwargs["request"].output.evidence.exists()
    kwargs["expected_runtime_digests"]["subject"] = "sha256:" + "f" * 64
    with pytest.raises(workflow.JudgeWorkflowError, match="different frozen request"):
        workflow.evaluate_native_judge(**kwargs, capture=capture)


def test_malformed_retained_capture_is_controlled_error(native, monkeypatch):
    _, kwargs = native
    path = kwargs["request"].comparison.judge.workspace
    path.mkdir(mode=0o700)
    (path / "native_capture.json").write_text(json.dumps({}))
    with pytest.raises(ValueError, match="exactly the defined fields"):
        workflow.evaluate_native_judge(
            **kwargs, capture=lambda: pytest.fail("unexpected recapture")
        )
