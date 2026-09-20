"""Judge admission and replay tests use synthetic captures; no provider is called."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tests.evaluation_records.test_live_recipient import fixture

HERE = Path(__file__).resolve().parents[2] / "examples/integrations/evaluator-live"
SPEC = importlib.util.spec_from_file_location("live_judge_example", HERE / "judge.py")
LIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIVE)
COMMON = LIVE.common


def inputs(tmp_path, *, gpu=False):
    """Synthetic file-contract fixture; the cuda label does not execute a GPU."""
    protocol, args = fixture(tmp_path, metadata={"source_cluster_id": "synthetic-unit"})
    if gpu:
        protocol["configuration"]["device"] = "cuda"
        protocol["acceptance"]["normalized_nll"]["metrics"][0]["configuration"][
            "configuration_digest"
        ] = COMMON.digest(protocol["configuration"])
        args[0].write_bytes(COMMON.encoded(protocol))
        args[1] = COMMON.digest(protocol)
        for role in ("baseline", "subject"):
            directory = tmp_path / role
            role_protocol = {**protocol, "role": role}
            (directory / "protocol.json").write_bytes(COMMON.encoded(role_protocol))
            native = COMMON.read(directory / "native.json")
            for path in list((directory / "tasks").glob("*.response.json")):
                response = COMMON.read(path)
                request = {
                    **response["request"],
                    "protocol_digest": COMMON.digest(role_protocol),
                }
                result = response["result"]
                execution = result["metadata"]["invarlock_model_execution"]
                execution.update(
                    request=request,
                    protocol_digest=request["protocol_digest"],
                    configuration=protocol["configuration"],
                )
                result["metadata"]["invarlock_likelihood"]["configuration_digest"] = (
                    COMMON.digest(protocol["configuration"])
                )
                path.unlink()
                path.with_name(path.name.replace(".response.", ".request.")).unlink()
                stem = COMMON.digest(request).removeprefix("sha256:")
                COMMON.write(directory / "tasks" / (stem + ".request.json"), request)
                COMMON.write(
                    directory / "tasks" / (stem + ".response.json"),
                    {"request": request, "result": result},
                )
                case = next(
                    case
                    for case in protocol["cases"]
                    if case["id"] == request["case_id"]
                )
                sample = next(
                    sample for sample in native["samples"] if sample["id"] == case["id"]
                )
                sample["metadata"] = {
                    **case["metadata"],
                    **LIVE.RECIPIENT.bindings.bind_result(
                        result, case, "inspect-ai", "0.3.254"
                    )["metadata"],
                }
            (directory / "native.json").write_bytes(COMMON.encoded(native))
            manifest = COMMON.read(directory / "capture.json")
            manifest.update(
                protocol_digest=COMMON.digest(role_protocol),
                native_sha256=LIVE.sha((directory / "native.json").read_bytes()),
            )
            (directory / "capture.json").write_bytes(COMMON.encoded(manifest))
    index = tmp_path / "capture-index.json"
    COMMON.write(
        index, {"inspect-ai": {"baseline": str(args[2]), "subject": str(args[3])}}
    )
    return args[0], args[1], index


def frozen(tmp_path, *, gpu=False, supplementary=False):
    args = inputs(tmp_path, gpu=gpu)
    root = tmp_path / "judge"
    ledger = LIVE.freeze(*args, root, supplementary=supplementary)
    return root, ledger, COMMON.digest(ledger)


def test_freeze_both_routes_and_controls_use_distinct_new_plans(tmp_path):
    root, ledger, pin = frozen(tmp_path, supplementary=True)
    assert len(ledger["entries"]) == 6
    assert ledger["reserved_calls"] == 40
    assert ledger["reserved_cost_microusd"] == 40 * 31200
    assert len({entry["plan_sha256"] for entry in ledger["entries"]}) == 6
    for entry in ledger["entries"]:
        _, _, values, _ = LIVE.entry_inputs(root, pin, entry["id"])
        assert values["plan"]["schedule"]["expected_trials"] == entry["calls"]
        assert values["plan"]["judge"]["config"]["reasoning_effort"] == "xhigh"
        assert "measurements" not in values["request"]["comparison"]["judge"]
    assert not list(root.rglob("*measurements*"))
    assert not list(root.rglob("collection-admission.json"))


@pytest.mark.parametrize(
    "mutation",
    ["calls", "cost", "duplicate", "negative", "boolean", "path", "ceiling", "total"],
)
def test_global_budget_and_unique_entry_inventory_fail_closed(tmp_path, mutation):
    _, ledger, _ = frozen(tmp_path)
    if mutation == "calls":
        ledger["entries"][0]["calls"] = 1009
    elif mutation == "cost":
        ledger["entries"][0]["cost_microusd"] = 32_000_001
    elif mutation == "duplicate":
        ledger["entries"][1]["id"] = ledger["entries"][0]["id"]
    elif mutation in {"negative", "boolean"}:
        ledger["entries"][0]["calls"] = -1 if mutation == "negative" else True
    elif mutation == "path":
        ledger["entries"][0]["id"] = "../elsewhere"
    elif mutation == "ceiling":
        ledger["maximum_calls"] = 2000
    else:
        ledger["reserved_calls"] += 1
    with pytest.raises(ValueError):
        LIVE.validate_ledger(ledger)


@pytest.mark.parametrize("mutation", ["pin", "plan", "protocol", "entry"])
def test_changed_frozen_inputs_cannot_reach_collection(tmp_path, monkeypatch, mutation):
    root, ledger, pin = frozen(tmp_path, gpu=True)
    ident = ledger["entries"][0]["id"]
    if mutation == "pin":
        pin = "sha256:" + "0" * 64
    elif mutation == "plan":
        path = root / ident / "plan.json"
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "protocol":
        path = root / "protocol.json"
        protocol = COMMON.read(path)
        protocol["configuration"]["max_new_tokens"] = 33
        path.write_bytes(COMMON.encoded(protocol))
    else:
        ident = "unadmitted"
    monkeypatch.setattr(LIVE, "cli", lambda *a, **k: pytest.fail("must not collect"))
    with pytest.raises(ValueError):
        LIVE.collect(root, pin, ident, "unused", execute=True)
    assert not list(root.rglob("collection-admission.json"))


def test_collection_requires_execution_flag_and_gpu_captures(tmp_path, monkeypatch):
    root, ledger, pin = frozen(tmp_path)
    ident = ledger["entries"][0]["id"]
    monkeypatch.setattr(LIVE, "cli", lambda *a, **k: pytest.fail("must not collect"))
    with pytest.raises(ValueError, match="explicit execute"):
        LIVE.collect(root, pin, ident, "unused")
    with pytest.raises(ValueError, match="GPU"):
        LIVE.collect(root, pin, ident, "unused", execute=True)


@pytest.mark.parametrize("fault", ["signer", "preexisting", "resume"])
def test_new_entry_cannot_bypass_pre_call_admission(tmp_path, monkeypatch, fault):
    root, ledger, pin = frozen(tmp_path, gpu=True)
    ident = ledger["entries"][0]["id"]
    directory = root / ident
    if fault == "signer":
        (directory / "signer.pem").unlink()
        LIVE.RECIPIENT.key(directory / "signer.pem")
    elif fault == "preexisting":
        (directory / "collection-work").mkdir()
    monkeypatch.setattr(LIVE, "cli", lambda *a, **k: pytest.fail("must not collect"))
    with pytest.raises(ValueError):
        LIVE.collect(root, pin, ident, "unused", execute=True, resume=fault == "resume")
    assert not (directory / "collection-admission.json").exists()


def test_interrupted_collection_resumes_only_same_charged_entry(tmp_path, monkeypatch):
    root, ledger, pin = frozen(tmp_path, gpu=True)
    entry = ledger["entries"][0]
    ident = entry["id"]
    calls = []

    def command(directory, *args, **kwargs):
        calls.append(args)
        assert (
            COMMON.read(directory / "collection-admission.json")["calls"]
            == entry["calls"]
        )
        if "--preflight" in args:
            return {
                "network_calls": 0,
                "planned_trials": entry["calls"],
                "collection_available": True,
            }
        raise RuntimeError("synthetic interruption before model/provider execution")

    monkeypatch.setattr(LIVE, "cli", command)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        LIVE.collect(root, pin, ident, "unused", execute=True)
    assert len(calls) == 2
    with pytest.raises(ValueError, match="resume"):
        LIVE.collect(root, pin, ident, "unused", execute=True)
    assert len(calls) == 2
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        LIVE.collect(root, pin, ident, "unused", execute=True, resume=True)
    assert len(calls) == 4
    assert len(list(root.rglob("collection-admission.json"))) == 1


@pytest.mark.parametrize("recipient_fault", [None, "authentication", "report"])
def test_source_cli_publishes_and_offline_verifies_new_synthetic_test_measurements(
    tmp_path, monkeypatch, recipient_fault
):
    from invarlock.cli.app import app
    from invarlock.judge_measurements import native_workflow
    from tests.core import test_native_judge_transaction as synthetic

    original_render = synthetic.render_judge_request

    root, ledger, pin = frozen(tmp_path, gpu=True)
    ident = ledger["entries"][0]["id"]
    calls = []
    monkeypatch.setattr(native_workflow, "collection_preflight", lambda _: {})

    def synthetic_collector(**kwargs):
        # Fresh test-only measurements for this exact new plan, never an old archive.
        calls.append(kwargs["plan"])
        references = {
            row["input"]: row["expected"] for row in kwargs["baseline_run"]["records"]
        }
        with monkeypatch.context() as context:
            context.setattr(
                synthetic,
                "render_judge_request",
                lambda plan, *, input_text, answer_text: original_render(
                    plan,
                    input_text=input_text,
                    answer_text=answer_text,
                    reference_text=references[input_text],
                ),
            )
            return synthetic._completed_measurements(
                plan=kwargs["plan"],
                baseline_run=kwargs["baseline_run"],
                subject_run=kwargs["subject_run"],
            )

    monkeypatch.setattr(native_workflow, "collect_frozen", synthetic_collector)

    failures = []

    def local_cli(directory, *args, allowed=(0,), **kwargs):
        assert kwargs.get("collection", False) == (
            args[0] == "evaluate" and "--preflight" not in args
        )
        if recipient_fault == "report" and args[0] == "report" and not failures:
            failures.append("report")
            raise RuntimeError("synthetic recipient report interruption")
        with monkeypatch.context() as context:
            context.chdir(directory)
            result = CliRunner().invoke(app, list(args))
        assert result.exit_code in allowed, result.output
        value = json.loads(result.stdout)
        if recipient_fault == "authentication" and args[0] == "verify" and not failures:
            failures.append("authentication")
            value["authenticated"] = False
        return value

    monkeypatch.setattr(LIVE, "cli", local_cli)

    def offline_recipient(command, **kwargs):
        assert "OPENAI_API_KEY" not in kwargs["env"]
        try:
            result = LIVE.verify(root, pin, ident)
        except (ValueError, RuntimeError) as exc:
            return SimpleNamespace(returncode=1, stdout="", stderr=str(exc))
        return SimpleNamespace(
            returncode=0, stdout=COMMON.encoded(result).decode(), stderr=""
        )

    monkeypatch.setattr(LIVE.subprocess, "run", offline_recipient)
    if recipient_fault:
        with pytest.raises(RuntimeError, match="offline recipient failed"):
            LIVE.collect(root, pin, ident, "test-only-recipient", execute=True)
        assert len(calls) == 1
        assert not (root / ident / "verification.json").exists()
    result = LIVE.collect(
        root,
        pin,
        ident,
        "test-only-recipient",
        execute=True,
        resume=bool(recipient_fault),
    )
    assert len(calls) == 1
    assert result["authenticated"] and result["verified"] and result["replayed"]
    assert result["decision"] == "insufficient_evidence"
    assert len(list((root / ident).glob("recipient-*"))) == 1 + bool(recipient_fault)
    directory = root / ident / result["recipient_artifacts"]
    assert all(
        (directory / name).is_file()
        for name in (
            "verification.receipt.json",
            "report.html",
            "report.md",
            "report.xml",
            "report.json",
        )
    )
    with pytest.raises(ValueError, match="already verified"):
        LIVE.collect(root, pin, ident, "unused", execute=True, resume=True)


@pytest.mark.parametrize("fault", ["model", "retry", "reserve"])
def test_unapproved_collection_configuration_is_rejected(tmp_path, fault):
    root, ledger, pin = frozen(tmp_path)
    _, _, values, _ = LIVE.entry_inputs(root, pin, ledger["entries"][0]["id"])
    if fault == "model":
        values["plan"]["judge"]["requested_model"] = "unapproved-model"
    elif fault == "retry":
        values["recipe"]["collection"]["sdk_max_retries"] = 1
    else:
        values["recipe"]["collection"]["max_calls"] += 1
    with pytest.raises(ValueError, match="approved Luna|planned call"):
        LIVE.closed_recipe(values["recipe"], values["plan"])


@pytest.mark.parametrize("fault", ["protocol", "index"])
def test_freeze_rejects_incomplete_or_unadmitted_capture_inputs(tmp_path, fault):
    protocol, pin, index = inputs(tmp_path)
    if fault == "protocol":
        pin = "sha256:" + "0" * 64
    else:
        index.write_bytes(COMMON.encoded({"inspect-ai": {"baseline": "unused"}}))
    with pytest.raises(ValueError, match="independent admission|exactly both roles"):
        LIVE.freeze(protocol, pin, index, tmp_path / "judge")
    assert not (tmp_path / "judge").exists()


def test_supplementary_excludes_nonrepresentative_evaluators(tmp_path, monkeypatch):
    monkeypatch.setattr(LIVE, "REPRESENTATIVES", set())
    _, ledger, _ = frozen(tmp_path, supplementary=True)
    assert [entry["profile"] for entry in ledger["entries"]] == ["primary", "primary"]


@pytest.mark.parametrize("fault", ["empty", "inventory", "plan-pin", "reserve"])
def test_malformed_admission_cannot_rebind_its_frozen_plan(tmp_path, fault):
    root, ledger, _ = frozen(tmp_path)
    entry = ledger["entries"][0]
    ident = entry["id"]
    if fault == "empty":
        ledger["entries"] = []
    elif fault == "inventory":
        entry["files"].pop("request.json")
    elif fault == "plan-pin":
        entry["plan_sha256"] = "sha256:" + "0" * 64
    else:
        entry["calls"] += 1
        ledger["reserved_calls"] += 1
    (root / "admission.json").write_bytes(COMMON.encoded(ledger))
    with pytest.raises(ValueError, match="inventory|control-file|admitted inputs"):
        LIVE.entry_inputs(root, COMMON.digest(ledger), ident)


@pytest.mark.parametrize(
    "fault", ["network_calls", "planned_trials", "collection_available"]
)
def test_preflight_must_match_full_admission_before_provider_execution(
    tmp_path, monkeypatch, fault
):
    root, ledger, pin = frozen(tmp_path, gpu=True)
    entry = ledger["entries"][0]
    calls = []

    def preflight(directory, *args, **kwargs):
        calls.append(args)
        assert "--preflight" in args
        result = {
            "network_calls": 0,
            "planned_trials": entry["calls"],
            "collection_available": True,
        }
        result[fault] = False if fault == "collection_available" else 99
        return result

    monkeypatch.setattr(LIVE, "cli", preflight)
    with pytest.raises(ValueError, match="preflight differs"):
        LIVE.collect(root, pin, entry["id"], "unused", execute=True)
    assert len(calls) == 1


def test_subprocess_boundary_retains_collector_credentials_only(tmp_path, monkeypatch):
    secrets = (
        "OPENAI_API_KEY",
        "HF_TOKEN",
        "SERVICE_ACCESS_TOKEN",
        "PYTHONPATH",
        "INVARLOCK_SIGNING_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "AWS_SESSION_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AZURE_CLIENT_SECRET",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "UNKNOWN_TOKEN",
        "UNKNOWN_SECRET",
        "OPENAI_BASE_URL",
        "HTTPS_PROXY",
    )
    for name in secrets:
        monkeypatch.setenv(name, "synthetic-secret")
    monkeypatch.setenv("LIVE_TEST_VISIBLE", "not-allowed")
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "1")
    monkeypatch.setenv("HOME", "/untrusted/ambient/home")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/untrusted/ambient/config")
    online = LIVE.environment(home=tmp_path / "online")
    assert online["OPENAI_API_KEY"] == "synthetic-secret"
    assert "INVARLOCK_ALLOW_NETWORK" not in online
    assert not any(name in online for name in secrets if name != "OPENAI_API_KEY")
    offline = LIVE.environment(offline=True, home=tmp_path / "offline")
    assert "LIVE_TEST_VISIBLE" not in offline
    assert not any(name in offline for name in secrets)
    assert "INVARLOCK_ALLOW_NETWORK" not in offline
    assert "INVARLOCK_ALLOW_NETWORK" not in LIVE.environment(
        offline=True, collection=True, home=tmp_path / "offline-forced"
    )
    assert offline["HOME"] == str(tmp_path / "offline")
    assert offline["XDG_CONFIG_HOME"] == str(tmp_path / "offline/config")
    results = iter(
        [
            SimpleNamespace(
                returncode=7, stdout='{"decision":"insufficient_evidence"}', stderr=""
            ),
            SimpleNamespace(returncode=4, stdout="", stderr="invalid evidence"),
        ]
    )

    def run(command, **kwargs):
        assert command[1:3] == ["-I", "-c"]
        assert command[5] == "invarlock"
        assert not any(name in kwargs["env"] for name in secrets)
        assert kwargs["cwd"] == tmp_path
        return next(results)

    monkeypatch.setattr(LIVE.subprocess, "run", run)
    assert (
        LIVE.cli(tmp_path, "verify", allowed=(0, 7))["decision"]
        == "insufficient_evidence"
    )
    with pytest.raises(RuntimeError, match="invalid evidence"):
        LIVE.cli(tmp_path, "verify")


@pytest.mark.parametrize("operation", ["freeze", "collect", "verify"])
def test_command_dispatch_preserves_explicit_admission_and_installed_recipient_check(
    monkeypatch, capsys, operation
):
    identity, dispatched = [], []
    guarded = []
    monkeypatch.setattr(
        LIVE.common, "module", lambda name: SimpleNamespace(configure=guarded.append)
    )
    monkeypatch.setattr(
        LIVE.RECIPIENT, "installed_identity", lambda: identity.append(True)
    )

    def action(*args, **kwargs):
        dispatched.append((args, kwargs))
        return {"synthetic_command_result": operation}

    monkeypatch.setattr(LIVE, operation, action)
    argv = ["judge.py", operation]
    if operation == "freeze":
        argv += [
            "--protocol",
            "protocol.json",
            "--captures",
            "captures.json",
            "--output",
            "proposal",
            "--protocol-sha256",
            "independent-pin",
            "--primary-only",
        ]
    else:
        argv += [
            "--root",
            "proposal",
            "--admission-sha256",
            "independent-pin",
            "--entry",
            "primary-inspect-ai-envelope",
        ]
        if operation == "collect":
            argv += [
                "--recipient-python",
                "recipient-python",
                "--execute-collection",
                "--resume",
            ]
    monkeypatch.setattr(LIVE.sys, "argv", argv)
    LIVE.main()
    assert len(dispatched) == 1
    assert guarded == (["judge-recipient"] if operation == "verify" else [])
    assert bool(identity) == (operation != "collect")
    assert dispatched[0][0][1] == "independent-pin"
    if operation == "freeze":
        assert dispatched[0][1] == {"supplementary": False}
        assert capsys.readouterr().out.strip().startswith("sha256:")
    else:
        if operation == "collect":
            assert dispatched[0][1] == {"execute": True, "resume": True}
        assert (
            json.loads(capsys.readouterr().out)["synthetic_command_result"] == operation
        )


def test_actual_offline_subprocess_blocks_network_and_drops_credentials(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "synthetic-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-secret")
    probe = tmp_path / "probe.py"
    probe.write_text("""
import json, os, socket, sys
assert sys.flags.isolated == 1
assert 'ANTHROPIC_AUTH_TOKEN' not in os.environ and 'OPENAI_API_KEY' not in os.environ
with socket.socket(socket.AF_INET) as channel:
    try:
        sys.audit('socket.connect', channel, ('192.0.2.1', 9))
    except RuntimeError as exc:
        assert 'local callback transport' in str(exc)
    else:
        raise AssertionError('offline guard missing')
print(json.dumps({'guarded': True, 'home': os.environ['HOME']}))
""")
    result = subprocess.run(
        LIVE.offline_command(sys.executable, probe),
        env=LIVE.environment(offline=True, home=tmp_path / "private"),
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert json.loads(result.stdout) == {
        "guarded": True,
        "home": str(tmp_path / "private"),
    }


def test_online_cli_passes_only_explicit_provider_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-explicit-key")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "excluded")
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "untrusted-ambient-value")

    def run(command, **kwargs):
        assert command[1:4] == ["-I", "-m", "invarlock"]
        assert kwargs["env"]["OPENAI_API_KEY"] == "synthetic-explicit-key"
        assert "AZURE_CLIENT_SECRET" not in kwargs["env"]
        assert kwargs["env"].get("INVARLOCK_ALLOW_NETWORK") == (
            None if "--preflight" in command else "1"
        )
        return SimpleNamespace(returncode=0, stdout='{"network_calls":0}', stderr="")

    monkeypatch.setattr(LIVE.subprocess, "run", run)
    assert LIVE.cli(tmp_path, "evaluate", "--preflight") == {"network_calls": 0}
    assert LIVE.cli(tmp_path, "evaluate", collection=True) == {"network_calls": 0}
    monkeypatch.delenv("OPENAI_API_KEY")
    assert "OPENAI_API_KEY" not in LIVE.environment(home=tmp_path / "no-key")
