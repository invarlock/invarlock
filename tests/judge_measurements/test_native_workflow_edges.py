"""Collection delegation and retained-workspace failures remain execution safe."""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import pytest

from invarlock.judge_measurements import native_workflow as workflow
from invarlock.judge_measurements import workflow as standalone
from invarlock.judge_measurements.contracts import canonical_payload
from invarlock.judge_measurements.evidence import replay_judge_evidence
from tests.cli.test_import_journey import _key
from tests.core.test_native_judge_transaction import _incomplete_measurements, _recipe
from tests.judge_measurements.test_native_workspace import native as native
from tests.judge_measurements.test_workflow import staged as staged


def _installed_api(monkeypatch):
    source = Path(__file__).parents[2] / "addins/inspect_judge/src"
    monkeypatch.syspath_prepend(str(source))
    return importlib.import_module("invarlock_addins.inspect_judge")


def test_missing_installed_collector_has_safe_install_diagnostic(monkeypatch):
    monkeypatch.setitem(sys.modules, "invarlock_addins.inspect_judge", None)
    with pytest.raises(
        workflow.JudgeWorkflowError, match=r"pip install .*invarlock\[judge\]=="
    ):
        workflow.collection_preflight(_recipe()["collection"])


@pytest.mark.parametrize(
    "fault", ["missing_key", "endpoint_override", "missing_dependency"]
)
def test_real_collection_environment_fails_without_loading_model(monkeypatch, fault):
    api = _installed_api(monkeypatch)
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    if fault != "missing_key":
        monkeypatch.setenv("OPENAI_API_KEY", "offline-unused-key")
    if fault == "endpoint_override":
        monkeypatch.setenv("OPENAI_API_BASE", "https://example.invalid/")
    if fault == "missing_dependency":
        monkeypatch.setattr(
            importlib.metadata,
            "version",
            Mock(side_effect=importlib.metadata.PackageNotFoundError("missing")),
        )
    with pytest.raises(api.InspectJudgeError, match="OPENAI_API_KEY|endpoint|install"):
        workflow.collection_preflight(_recipe()["collection"])


@pytest.fixture
def delegated(monkeypatch):
    real = _installed_api(monkeypatch)
    api = ModuleType("invarlock_addins.inspect_judge")
    api.CollectionOptions = real.CollectionOptions
    api.RunnerOptions = real.RunnerOptions
    api.prepare_collection = real.prepare_collection
    api.validate_collection_environment = Mock(
        return_value={"credential_available": True}
    )

    async def collected(*args, on_stop):
        on_stop("complete")
        return {"retained": True}

    api.collect_configured = AsyncMock(side_effect=collected)
    monkeypatch.setitem(sys.modules, "invarlock_addins.inspect_judge", api)
    return api


def test_collect_frozen_delegates_exact_frozen_answers_and_checkpoint(
    tmp_path, delegated
):
    recipe = _recipe()
    plan, baseline, subject = object(), {"baseline": "frozen"}, {"subject": "frozen"}
    assert workflow.collection_preflight(recipe["collection"]) == {
        "credential_available": True
    }
    status = {"stop_reason": "stale"}
    result = workflow.collect_frozen(
        plan=plan,
        collection=recipe["collection"],
        runner=recipe["runner"],
        workspace=tmp_path,
        baseline_run=baseline,
        subject_run=subject,
        status=status,
    )
    assert status == {"stop_reason": "complete"}
    assert result == {"retained": True}
    args = delegated.collect_configured.call_args.args
    assert args[0] is plan and args[3] is baseline and args[4] is subject
    assert args[1].grader == recipe["collection"]["grader"]
    assert args[2].checkpoint_directory == tmp_path / "collection"
    assert args[2].scorer_id == recipe["runner"]["scorer_id"]
    assert (
        workflow.collect_frozen(
            plan=plan,
            collection=recipe["collection"],
            runner=recipe["runner"],
            workspace=tmp_path,
            baseline_run=baseline,
            subject_run=subject,
        )
        == result
    )
    assert not list(tmp_path.iterdir())


def test_sync_collection_in_active_loop_fails_before_coroutine_creation(
    tmp_path, delegated
):
    recipe = _recipe()

    async def attempt():
        with pytest.raises(workflow.JudgeWorkflowError, match="active event loop"):
            workflow.collect_frozen(
                plan={},
                collection=recipe["collection"],
                runner=recipe["runner"],
                workspace=tmp_path,
                baseline_run={},
                subject_run={},
            )

    asyncio.run(attempt())
    delegated.collect_configured.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("unsafe", ["readable_lock", "linked_lock"])
def test_workspace_rejects_unsafe_lock_before_capture(tmp_path, unsafe):
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    lock = workspace / ".evaluation.lock"
    lock.touch(mode=0o600)
    if unsafe == "readable_lock":
        lock.chmod(0o644)
    else:
        os.link(lock, tmp_path / "alias")
    with pytest.raises(workflow.JudgeWorkflowError, match="lock is unsafe"):
        with workflow.locked_workspace(workspace):
            pytest.fail("unsafe workspace was admitted")


def test_workspace_detects_name_replacement_after_acquisition(tmp_path):
    workspace = tmp_path / "workspace"
    with pytest.raises(workflow.JudgeWorkflowError, match="ancestry changed"):
        with workflow.locked_workspace(workspace) as unchanged:
            workspace.rename(tmp_path / "detached")
            workspace.mkdir(mode=0o700)
            unchanged()


def test_workspace_detects_replaced_lock_with_same_contents(tmp_path):
    workspace = tmp_path / "workspace"
    with pytest.raises(workflow.JudgeWorkflowError, match="lock changed"):
        with workflow.locked_workspace(workspace) as unchanged:
            lock = workspace / ".evaluation.lock"
            lock.unlink()
            lock.touch(mode=0o600)
            unchanged()


def test_workspace_replacement_while_opening_fails_before_lock(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    original_stat = Path.stat
    replaced = False

    def replace_on_lookup(path, **kwargs):
        nonlocal replaced
        if path == workspace and not replaced:
            replaced = True
            workspace.rename(tmp_path / "detached")
            workspace.mkdir(mode=0o700)
        return original_stat(path, **kwargs)

    monkeypatch.setattr(Path, "stat", replace_on_lookup)
    with pytest.raises(workflow.JudgeWorkflowError, match="changed while opening"):
        with workflow.locked_workspace(workspace):
            pytest.fail("changed directory was admitted")
    assert not (workspace / ".evaluation.lock").exists()


@pytest.mark.parametrize("entry", ["preflight", "evaluate"])
def test_missing_native_configuration_never_captures(native, entry):
    _, kwargs = native
    kwargs["request"].comparison.judge = None
    with pytest.raises(workflow.JudgeWorkflowError, match="configuration is required"):
        if entry == "preflight":
            workflow.preflight_native_judge(
                kwargs["request"], kwargs["schedule"], kwargs["policy_bytes"]
            )
        else:
            workflow.evaluate_native_judge(
                **kwargs, capture=lambda: pytest.fail("capture without configuration")
            )


@pytest.mark.parametrize(
    "fault", ["list", "different_request", "artifact_pin", "runtime_pin"]
)
def test_unsafe_retained_capture_cannot_reach_collection(native, monkeypatch, fault):
    frozen, kwargs = native
    workspace = kwargs["request"].comparison.judge.workspace
    workspace.mkdir(mode=0o700)
    (workspace / "native_capture.json").write_bytes(
        canonical_payload([] if fault == "list" else frozen)
    )
    if fault == "different_request":
        kwargs["normalized_request"] = copy.deepcopy(kwargs["normalized_request"])
        kwargs["normalized_request"]["comparison"]["judge"]["signer_identity"] = (
            "different"
        )
    elif fault == "artifact_pin":
        kwargs["expected_artifact_digests"]["baseline"] = "sha256:" + "0" * 64
    elif fault == "runtime_pin":
        kwargs["expected_runtime_digests"]["baseline"] = "sha256:" + "0" * 64
    collector = Mock(side_effect=AssertionError("unsafe capture reached collection"))
    monkeypatch.setattr(workflow, "collect_frozen", collector)
    with pytest.raises(
        workflow.JudgeWorkflowError,
        match="must be an object|frozen request|current preflight",
    ):
        workflow.evaluate_native_judge(
            **kwargs, capture=lambda: pytest.fail("unexpected recapture")
        )
    collector.assert_not_called()


def _standalone_collect(staged):
    path, value = staged
    collection = json.loads(
        (
            Path(__file__).parents[2] / "examples/judge-measurements/collection.json"
        ).read_bytes()
    )
    (path.parent / "collection.json").write_text(json.dumps(collection))
    value["execution"] = {
        "mode": "judge_collect",
        "collection": {
            "integration": "inspect-judge",
            "configuration": "collection.json",
            "workspace": "workspace",
        },
    }
    value["comparison"]["measurements"] = None
    path.write_text(json.dumps(value))
    return standalone.load_judge_request(path)


def test_standalone_collection_checks_signing_key_before_any_call(staged, delegated):
    request = _standalone_collect(staged)
    invalid_key = request.root / "invalid.pem"
    invalid_key.write_text("invalid private key")
    with pytest.raises(workflow.JudgeWorkflowError, match="PEM"):
        standalone.evaluate_judge_request(
            request, signing_key=invalid_key, unsigned=False
        )
    delegated.collect_configured.assert_not_called()
    assert not request.workspace.exists()
    assert not request.evidence.exists()


@pytest.mark.parametrize("exhausted", [False, True])
def test_pending_collection_distinguishes_deadline_resume_from_retained_capacity(
    staged, delegated, monkeypatch, exhausted
):
    request = _standalone_collect(staged)
    plan = json.loads((request.root / "plan.json").read_bytes())
    completed = json.loads((request.root / "measurements.json").read_bytes())
    pending = _incomplete_measurements(plan=plan)
    outcomes = iter((pending, completed))

    async def collect(*_args, on_stop):
        value = next(outcomes)
        on_stop(
            "capacity_exhausted"
            if exhausted
            else "deadline"
            if value is pending
            else "complete"
        )
        return value

    delegated.collect_configured.side_effect = collect
    key, _ = _key(request.root / "evidence.pem")
    if exhausted:
        result = standalone.evaluate_judge_request(
            request, signing_key=key, unsigned=False
        )
        assert result.payload["collection"] == {
            "pending_trials": 2,
            "stop_reason": "retained_capacity_exhausted",
            "resumable": False,
        }
        assert result.payload["decision"] == "insufficient_evidence"
    else:
        with pytest.raises(workflow.JudgeWorkflowError) as error:
            standalone.evaluate_judge_request(request, signing_key=key, unsigned=False)
        assert (
            error.value.payload["resumable"]
            and error.value.payload["pending_trials"] == 2
        )
        assert not request.evidence.exists()
        result = standalone.evaluate_judge_request(
            request, signing_key=key, unsigned=False
        )
        assert result.payload["collection"]["stop_reason"] == "complete"
    assert (
        replay_judge_evidence(request.evidence).envelope["signer"]["identity"]
        == request.signer_identity
    )


@pytest.mark.parametrize("stop_reason", [None, "complete", "unsupported"])
def test_pending_collection_without_real_deadline_is_not_resumable(
    tmp_path, stop_reason
):
    with pytest.raises(workflow.JudgeWorkflowError) as error:
        workflow.require_completed_collection(
            {"trials": [{"attempts": []}]}, tmp_path, stop_reason=stop_reason
        )
    assert error.value.payload["resumable"] is False
    assert error.value.payload["stop_reason"] == "unknown"
