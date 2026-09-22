"""Campaign proposals and lifecycle controls use test-only local measurements."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.evaluation_records.test_live_judge import inputs

HERE = Path(__file__).resolve().parents[2] / "examples/integrations/evaluator-live"
SPEC = importlib.util.spec_from_file_location(
    "closure_judge_tests", HERE / "closure_judge.py"
)
LIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIVE)
COMMON = LIVE.common


def specification(tmp_path, *, gpu=True, profile="primary"):
    protocol, pin, index = inputs(tmp_path, gpu=gpu)
    cases = len(COMMON.read(protocol)["cases"])
    value = {
        "format": LIVE.FORMAT,
        "maximum_calls": 1288,
        "maximum_cost_microusd": 40_185_600,
        "cost_microusd_per_call": 31200,
        "groups": [
            {
                "id": "local",
                "protocol": str(protocol),
                "protocol_sha256": pin,
                "captures": str(index),
                "evaluators": ["inspect-ai"],
                "routes": ["envelope", "native-json"],
                "case_count": cases,
                "profile": profile,
                "admitted_calls": 2 if profile == "budget-control" else None,
            }
        ],
    }
    path = tmp_path / "specification.json"
    COMMON.write(path, value)
    return path, value


def frozen(tmp_path, **kwargs):
    path, spec = specification(tmp_path, **kwargs)
    root = tmp_path / "campaign"
    ledger = LIVE.freeze(path, root)
    return root, ledger, COMMON.digest(ledger), ledger["entries"][0]["id"]


def test_new_proposal_profiles_and_aggregate_do_not_change_legacy_limits(tmp_path):
    _, spec = specification(tmp_path)
    group = spec["groups"][0]
    groups = []
    for name, count, profile, evaluators, routes, limit in [
        (
            "local64",
            64,
            "primary",
            sorted(LIVE.EVALUATORS),
            ["envelope", "native-json"],
            None,
        ),
        (
            "http8",
            8,
            "primary",
            sorted(LIVE.EVALUATORS),
            ["envelope", "native-json"],
            None,
        ),
        (
            "reference",
            8,
            "reference-free",
            ["langfuse"],
            ["envelope", "native-json"],
            None,
        ),
        (
            "repeat",
            8,
            "repeat-control",
            ["langfuse"],
            ["envelope", "native-json"],
            None,
        ),
        ("budget", 8, "budget-control", sorted(LIVE.EVALUATORS), ["envelope"], 2),
    ]:
        groups.append(
            {
                **group,
                "id": name,
                "case_count": count,
                "profile": profile,
                "evaluators": evaluators,
                "routes": routes,
                "admitted_calls": limit,
            }
        )
    spec["groups"] = groups
    assert LIVE.validate_spec(spec) == 1288
    assert LIVE.BASE.MAX_CALLS == 1008 and LIVE.BASE.MAX_COST == 32_000_000
    with pytest.raises(ValueError, match="unsupported judge admission"):
        LIVE.BASE.validate_ledger({"format": LIVE.FORMAT})


@pytest.mark.parametrize(
    "fault",
    [
        "format",
        "bool",
        "ceiling",
        "cost",
        "extra",
        "duplicate",
        "selection",
        "profile",
        "digest",
        "truncation",
        "budget",
        "cases",
        "groups",
    ],
)
def test_proposal_rejects_unbounded_or_ambiguous_controls(tmp_path, fault):
    _, spec = specification(tmp_path)
    group = spec["groups"][0]
    if fault == "format":
        spec["format"] = "legacy"
    elif fault == "bool":
        spec["maximum_calls"] = True
    elif fault == "ceiling":
        spec["maximum_calls"] = 1
    elif fault == "cost":
        spec["maximum_cost_microusd"] = 1
    elif fault == "extra":
        spec["extra"] = None
    elif fault == "duplicate":
        spec["groups"].append(copy.deepcopy(group))
    elif fault == "selection":
        group["evaluators"] = ["inspect-ai", "inspect-ai"]
    elif fault == "profile":
        group["profile"] = "unknown"
    elif fault == "digest":
        group["protocol_sha256"] = "sha256:bad"
    elif fault == "truncation":
        group["admitted_calls"] = 2
    elif fault == "budget":
        group.update(profile="budget-control", admitted_calls=10**9)
    elif fault == "cases":
        group["case_count"] = 0
    else:
        spec["groups"] = []
    with pytest.raises(ValueError):
        LIVE.validate_spec(spec)


@pytest.mark.parametrize("profile", list(LIVE.PROFILES))
def test_frozen_profile_reconstructs_exact_plan_and_policy(tmp_path, profile):
    root, ledger, pin, ident = frozen(tmp_path, profile=profile)
    actual, entry, values, _ = LIVE.entry_inputs(root, pin, ident)
    assert actual == ledger
    assert entry["calls"] == values["recipe"]["collection"]["max_calls"]
    assert values["plan"]["schedule"]["repetitions"] == LIVE.PROFILES[profile][0]
    assert values["plan"]["prompt"]["reference_mode"] == LIVE.PROFILES[profile][1]
    assert set(entry["files"]) == {name + ".json" for name in LIVE.FILES}


@pytest.mark.parametrize(
    "fault", ["pin", "entry", "file", "file-inventory", "reserve", "duplicate"]
)
def test_changed_proposal_or_frozen_bytes_are_rejected(tmp_path, fault):
    root, ledger, pin, ident = frozen(tmp_path)
    if fault == "pin":
        pin = "sha256:" + "0" * 64
    elif fault == "entry":
        ident = "outside"
    elif fault == "file":
        path = root / ident / "plan.json"
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b" ")
    else:
        if fault == "file-inventory":
            ledger["entries"][0]["files"].pop("plan.json")
        elif fault == "reserve":
            ledger["entries"][0]["calls"] += 1
        else:
            ledger["entries"][1] = copy.deepcopy(ledger["entries"][0])
        (root / "admission.json").write_bytes(COMMON.encoded(ledger))
        pin = COMMON.digest(ledger)
    with pytest.raises(ValueError):
        LIVE.entry_inputs(root, pin, ident)


def measurement_collector(monkeypatch, *, stop=False):
    import invarlock.judge_measurements as api
    from tests.core import test_native_judge_transaction as test_measurements

    original = test_measurements.render_judge_request
    calls = []

    async def collect(plan, options, runner, baseline, subject, *, on_stop):
        calls.append((plan, options, runner))
        references = {row["input"]: row["expected"] for row in baseline["records"]}
        with monkeypatch.context() as patch:
            patch.setattr(
                test_measurements,
                "render_judge_request",
                lambda plan, *, input_text, answer_text: original(
                    plan,
                    input_text=input_text,
                    answer_text=answer_text,
                    reference_text=references[input_text]
                    if plan["prompt"]["reference_mode"] == "per_case"
                    else None,
                ),
            )
            measured = test_measurements._completed_measurements(
                plan=plan, baseline_run=baseline, subject_run=subject
            )
        reason = "requested" if stop and len(calls) == 1 else "complete"
        count = len(measured["trials"])
        if reason == "requested":
            count = min(options.concurrency, count - 1)
        if options.max_calls < count:
            count = options.max_calls
            reason = "capacity_exhausted"
        if count < len(measured["trials"]):
            pending = test_measurements._incomplete_measurements(plan=plan)
            pending["trials"][:count] = measured["trials"][:count]
            measured = pending
            raw = test_measurements.canonical_payload(
                {
                    "format": "invarlock/retained-judge-json-v1",
                    "trials": measured["trials"],
                }
            )
            measured["sources"][0].update(
                content=raw.decode(),
                byte_size=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
            measured["completeness"]["completed_trials"] = count
        on_stop(reason)
        return measured

    monkeypatch.setattr(api, "collect_configured", collect)
    return calls


def test_supported_lifecycle_then_signed_publication_uses_same_plan(
    tmp_path, monkeypatch
):
    root, _, pin, ident = frozen(tmp_path)
    calls = measurement_collector(monkeypatch, stop=True)
    first = LIVE.execute_entry(root, pin, ident, execute=True, stop_after_batches=1)
    assert first["resumable"] and first["stop_reason"] == "requested"
    assert not (root / ident / "evidence").exists()
    assert len(list((root / ident).glob("stopped-*/measurements.json"))) == 1
    with pytest.raises(ValueError, match="unchanged-checkpoint"):
        LIVE.execute_entry(root, pin, ident, execute=True)
    result = LIVE.execute_entry(root, pin, ident, execute=True, resume=True)
    assert not result["resumable"] and result["decision"] == "insufficient_evidence"
    assert calls[0][0] == calls[1][0] and calls[0][1] == calls[1][1]
    assert calls[0][2].checkpoint_directory == calls[1][2].checkpoint_directory
    assert (
        calls[0][2].stop_after_batches == 1 and calls[1][2].stop_after_batches is None
    )
    with pytest.raises(ValueError, match="already published"):
        LIVE.execute_entry(root, pin, ident, execute=True, resume=True)


@pytest.mark.parametrize("fault", ["authorization", "resume", "signer", "cpu", "limit"])
def test_execution_rejects_before_collection(tmp_path, monkeypatch, fault):
    root, _, pin, ident = frozen(tmp_path, gpu=fault != "cpu")
    calls = measurement_collector(monkeypatch)
    if fault == "signer":
        path = root / ident / "signer.pem"
        path.unlink()
        LIVE.RECIPIENT.key(path)
    with pytest.raises(ValueError):
        LIVE.execute_entry(
            root,
            pin,
            ident,
            execute=fault != "authorization",
            resume=fault == "resume",
            stop_after_batches=0 if fault == "limit" else None,
        )
    assert calls == []


def test_offline_verifier_uses_new_entry_validation(tmp_path, monkeypatch):
    root, _, pin, ident = frozen(tmp_path)
    seen = []
    monkeypatch.setattr(
        LIVE.BASE,
        "verify",
        lambda *args, **kwargs: seen.append((args, kwargs)) or {"verified": True},
    )
    assert LIVE.verify(root, pin, ident) == {"verified": True}
    assert seen[0][1]["input_loader"] is LIVE.entry_inputs


@pytest.mark.parametrize("stop", [True, False])
def test_child_boundary_limits_credentials_and_preserves_lifecycle(
    tmp_path, monkeypatch, stop
):
    root, _, pin, ident = frozen(tmp_path)
    commands = []
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")

    def child(command, **kwargs):
        commands.append((command, kwargs))
        if len(commands) == 1:
            assert kwargs["env"]["OPENAI_API_KEY"] == "test-only-key"
            assert kwargs["env"]["INVARLOCK_ALLOW_JUDGE_NETWORK"] == "1"
            return SimpleNamespace(
                returncode=0,
                stdout=COMMON.encoded({"resumable": stop}).decode(),
                stderr="",
            )
        assert "OPENAI_API_KEY" not in kwargs["env"]
        return SimpleNamespace(returncode=0, stdout='{"verified":true}', stderr="")

    monkeypatch.setattr(LIVE.subprocess, "run", child)
    result = LIVE.collect(
        root,
        pin,
        ident,
        "recipient-python",
        execute=True,
        resume=True,
        stop_after_batches=1 if stop else None,
    )
    assert len(commands) == (1 if stop else 2)
    assert result == ({"resumable": True} if stop else {"verified": True})
    assert "--resume" in commands[0][0]
    assert ("--stop-after-batches" in commands[0][0]) is stop


@pytest.mark.parametrize("command", ["freeze", "collect", "execute", "verify"])
def test_cli_dispatch_preserves_explicit_controls(
    tmp_path, monkeypatch, capsys, command
):
    guards, calls = [], []
    monkeypatch.setattr(
        LIVE.common, "module", lambda _: SimpleNamespace(configure=guards.append)
    )
    monkeypatch.setattr(
        LIVE.RECIPIENT, "installed_identity", lambda: guards.append("installed")
    )
    monkeypatch.setattr(LIVE, "freeze", lambda *args: {"frozen": True})
    monkeypatch.setattr(
        LIVE,
        "collect",
        lambda *args, **kwargs: calls.append(kwargs) or {"collected": True},
    )
    monkeypatch.setattr(
        LIVE,
        "execute_entry",
        lambda *args, **kwargs: calls.append(kwargs) or {"executed": True},
    )
    monkeypatch.setattr(LIVE, "verify", lambda *args: {"verified": True})
    args = [command]
    if command == "freeze":
        args += [
            "--specification",
            str(tmp_path / "spec"),
            "--output",
            str(tmp_path / "out"),
        ]
    else:
        args += [
            "--root",
            str(tmp_path),
            "--admission-sha256",
            "pin",
            "--entry",
            "entry",
        ]
    if command in {"collect", "execute"}:
        args += ["--execute-collection", "--resume", "--stop-after-batches", "1"]
    if command == "collect":
        args += ["--recipient-python", "recipient"]
    LIVE.main(args)
    assert capsys.readouterr().out
    assert guards == (
        ["judge-recipient", "installed"] if command in {"freeze", "verify"} else []
    )
    if calls:
        assert calls == [{"execute": True, "resume": True, "stop_after_batches": 1}]


def test_budget_control_publishes_incomplete_observations_without_resuming(
    tmp_path, monkeypatch
):
    from invarlock.judge_measurements.evidence import replay_judge_evidence

    root, _, pin, ident = frozen(tmp_path, profile="budget-control")
    calls = measurement_collector(monkeypatch)
    result = LIVE.execute_entry(root, pin, ident, execute=True)
    assert len(calls) == 1
    assert result["stop_reason"] == "capacity_exhausted" and not result["resumable"]
    assert result["completeness"]["completed_trials"] == 2
    assert result["completeness"]["status"] == "incomplete"
    replayed = replay_judge_evidence(root / ident / "evidence")
    assert replayed.analysis_result.to_dict()["decision"] == "insufficient_evidence"


def test_recipient_retry_does_not_repeat_completed_collection(tmp_path, monkeypatch):
    root, _, pin, ident = frozen(tmp_path)
    calls = measurement_collector(monkeypatch)
    LIVE.execute_entry(root, pin, ident, execute=True)
    commands = []

    def recipient(command, **kwargs):
        commands.append(command)
        assert "execute" not in command
        assert "OPENAI_API_KEY" not in kwargs["env"]
        return SimpleNamespace(returncode=0, stdout='{"verified":true}', stderr="")

    monkeypatch.setattr(LIVE.subprocess, "run", recipient)
    with pytest.raises(ValueError, match="original admission"):
        LIVE.collect(root, pin, ident, "recipient", execute=True)
    assert LIVE.collect(root, pin, ident, "recipient", execute=True, resume=True) == {
        "verified": True
    }
    assert len(commands) == 1 and len(calls) == 1
    COMMON.write(root / ident / "verification.json", {"verified": True})
    with pytest.raises(ValueError, match="already independently verified"):
        LIVE.collect(root, pin, ident, "recipient", execute=True, resume=True)


@pytest.mark.parametrize("phase", ["collection", "recipient"])
def test_subprocess_failures_do_not_claim_success(tmp_path, monkeypatch, phase):
    root, _, pin, ident = frozen(tmp_path)
    calls = []

    def child(*args, **kwargs):
        calls.append(args)
        failed = phase == "collection" or len(calls) == 2
        return SimpleNamespace(
            returncode=1 if failed else 0,
            stdout='{"resumable":false}',
            stderr="test failure",
        )

    monkeypatch.setattr(LIVE.subprocess, "run", child)
    with pytest.raises(ValueError, match="child stopped|recipient failed"):
        LIVE.collect(root, pin, ident, "recipient", execute=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cost_microusd_per_call", 1),
        ("cost_microusd_per_call", 31201),
        ("maximum_calls", 1289),
        ("maximum_cost_microusd", 40_185_601),
    ],
)
def test_reviewed_new_campaign_ceiling_and_price_cannot_be_raised_or_understated(
    tmp_path, field, value
):
    _, spec = specification(tmp_path)
    spec[field] = value
    with pytest.raises(ValueError, match="explicit campaign limits"):
        LIVE.validate_spec(spec)


@pytest.mark.parametrize("evaluator", sorted(LIVE.EVALUATORS))
@pytest.mark.parametrize("route", ["envelope", "native-json"])
def test_real_loopback_hosted_capture_freezes_and_verifies_judge_subject(
    tmp_path, monkeypatch, evaluator, route
):
    """Real HTTP transports test-only observations; judge answers stay synthetic."""
    import asyncio

    from typer.testing import CliRunner

    from invarlock.cli.app import app
    from invarlock.evaluation_records.identity import evaluated_subject_digest
    from invarlock.judge_measurements import CollectionOptions, RunnerOptions
    from invarlock.judge_measurements.evidence import publish_judge_evidence
    from tests.evaluation_records import test_live_http_service as http_fixture

    original_fixture = http_fixture.FIXTURE.fixture
    monkeypatch.setattr(
        http_fixture.FIXTURE,
        "fixture",
        lambda *args, **kwargs: original_fixture(
            *args, metadata={"source_cluster_id": "synthetic-http-unit"}, **kwargs
        ),
    )
    protocol, _, args = http_fixture.captured(tmp_path, monkeypatch, evaluator)
    index = tmp_path / "hosted-index.json"
    COMMON.write(
        index, {evaluator: {"baseline": str(args[2]), "subject": str(args[3])}}
    )
    spec = {
        "format": LIVE.FORMAT,
        "maximum_calls": 1288,
        "maximum_cost_microusd": 40_185_600,
        "cost_microusd_per_call": 31200,
        "groups": [
            {
                "id": "hosted",
                "protocol": str(args[0]),
                "protocol_sha256": args[1],
                "captures": str(index),
                "evaluators": [evaluator],
                "routes": [route],
                "case_count": len(protocol["cases"]),
                "profile": "primary",
                "admitted_calls": None,
            }
        ],
    }
    specification_path = tmp_path / "hosted-specification.json"
    COMMON.write(specification_path, spec)
    root = tmp_path / "hosted-campaign"
    ledger = LIVE.freeze(specification_path, root)
    pin, ident = COMMON.digest(ledger), ledger["entries"][0]["id"]
    _, _, values, _ = LIVE.entry_inputs(root, pin, ident)
    assert values["subject_run"]["artifact_digest"] is None
    assert values["subject_run"]["service_identity"]
    measurement_collector(monkeypatch)
    import invarlock.judge_measurements as api

    measured = asyncio.run(
        api.collect_configured(
            values["plan"],
            CollectionOptions(**values["recipe"]["collection"]),
            RunnerOptions(root / "unused", **values["recipe"]["runner"]),
            values["baseline_run"],
            values["subject_run"],
            on_stop=lambda _: None,
        )
    )
    publish_judge_evidence(
        root / ident / "evidence",
        plan=values["plan"],
        measurements=measured,
        baseline_run=values["baseline_run"],
        subject_run=values["subject_run"],
        analysis_policy=values["analysis_policy"],
        signing_key=root / ident / "signer.pem",
        signer_identity=LIVE.BASE.SIGNER,
    )

    def local_cli(directory, *arguments, allowed=(0,), **kwargs):
        with monkeypatch.context() as patch:
            patch.chdir(directory)
            result = CliRunner().invoke(app, list(arguments))
        assert result.exit_code in allowed, result.output
        return json.loads(result.stdout)

    monkeypatch.setattr(LIVE.BASE, "cli", local_cli)
    verified = LIVE.verify(root, pin, ident)
    assert verified["authenticated"] and verified["verified"] and verified["replayed"]
    assert verified["intended_subject"] == evaluated_subject_digest(
        values["subject_run"]
    )
    assert verified["decision"] == "insufficient_evidence"
