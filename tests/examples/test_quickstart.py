from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/quickstart/run.py"
FIXTURE = ROOT / "examples/acceptance-handoff/golden"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("quickstart_example", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quickstart_issues_a_fresh_receipt_and_report_outside_checkout(
    tmp_path: Path,
) -> None:
    module = _module()
    output = tmp_path / "consumer-output"

    outputs = module.run_quickstart(fixture=FIXTURE, output=output)

    result = json.loads(outputs["verification"].read_bytes())
    receipt = json.loads(outputs["receipt"].read_bytes())
    assert result["ok"] is True
    assert result["policy_verdict"] == "pass"
    assert result["verifier_identity"] == "quickstart-verifier"
    assert receipt["statement"]["verdict"]["policy_verdict"] == "pass"
    assert "InvarLock comparison report" in outputs["report"].read_text(
        encoding="utf-8"
    )
    assert not output.joinpath(".verifier.private.pem").exists()


def test_quickstart_fails_closed_for_tampered_evidence(tmp_path: Path) -> None:
    module = _module()
    fixture = tmp_path / "golden"
    shutil.copytree(FIXTURE, fixture)
    report = fixture / "evidence/reports/evaluation.report.json"
    report.write_bytes(report.read_bytes() + b"\n")
    output = tmp_path / "rejected-output"

    with pytest.raises(module.QuickstartError, match="rejected the fixture"):
        module.run_quickstart(fixture=fixture, output=output)

    assert not output.exists()


def test_quickstart_rejects_reuse_and_symlinked_inputs(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(module.QuickstartError, match="must be new"):
        module.run_quickstart(fixture=FIXTURE, output=output)

    linked = tmp_path / "linked-fixture"
    linked.symlink_to(FIXTURE, target_is_directory=True)
    with pytest.raises(module.QuickstartError, match="real directory"):
        module.run_quickstart(fixture=linked, output=tmp_path / "unused")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{"schedule_digest":"one","schedule_digest":"two"}', "duplicate"),
        ("{", "strict JSON"),
        ("[]", "JSON object"),
        ('{"artifact_digests":{}}', "both artifacts"),
        (
            '{"artifact_digests":{"baseline":"sha256:'
            + "0" * 64
            + '","subject":"sha256:'
            + "1" * 64
            + '"},"runtime_digests":{}}',
            "both runtimes",
        ),
    ],
)
def test_quickstart_rejects_malformed_anchor_files(
    tmp_path: Path, payload: str, message: str
) -> None:
    module = _module()
    path = tmp_path / "anchors.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(module.QuickstartError, match=message):
        module._anchors(path)


def test_quickstart_bounds_anchor_input_and_rejects_nonfiles(tmp_path: Path) -> None:
    module = _module()
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (module._MAX_ANCHOR_BYTES + 1))
    with pytest.raises(module.QuickstartError, match="size limit"):
        module._strict_object(oversized, label="anchors")

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(module.QuickstartError, match="regular file"):
        module._strict_object(directory, label="anchors")


def test_quickstart_rejects_invalid_digest_and_incomplete_fixture(
    tmp_path: Path,
) -> None:
    module = _module()
    with pytest.raises(module.QuickstartError, match="lowercase sha256"):
        module._digest("sha256:BAD", label="anchor")

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    with pytest.raises(module.QuickstartError, match="incomplete"):
        module._real_fixture(fixture)

    shutil.copytree(FIXTURE, fixture, dirs_exist_ok=True)
    leaked = fixture / "evidence/leaked"
    leaked.symlink_to(tmp_path / "missing")
    with pytest.raises(module.QuickstartError, match="must not contain"):
        module._real_fixture(fixture)


def test_quickstart_rejects_unsafe_output_and_key_destinations(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(module.QuickstartError, match="existing directory"):
        module._new_output(tmp_path / "missing/child")

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(module.QuickstartError, match="real directory"):
        module._new_output(linked_parent / "output")

    key = tmp_path / "key.pem"
    key.write_text("occupied", encoding="utf-8")
    with pytest.raises(module.QuickstartError, match="temporary verifier key"):
        module._write_verifier_key(key)


def test_quickstart_bounds_child_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()

    def oversized(*_args: object, **kwargs: object) -> SimpleNamespace:
        kwargs["stdout"].write(b"x" * (module._MAX_COMMAND_OUTPUT + 1))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", oversized)
    with pytest.raises(module.QuickstartError, match="size limit"):
        module._run_cli(["--version"], cwd=tmp_path)

    def timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(["invarlock"], 30)

    monkeypatch.setattr(module.subprocess, "run", timeout)
    with pytest.raises(module.QuickstartError, match="complete safely"):
        module._run_cli(["--version"], cwd=tmp_path)


@pytest.mark.parametrize(
    ("stdout", "message"),
    [("not-json", "did not return JSON"), ("[]", "invalid result")],
)
def test_quickstart_rejects_invalid_command_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    message: str,
) -> None:
    module = _module()

    def command(*_args: object, **kwargs: object) -> SimpleNamespace:
        kwargs["stdout"].write(stdout.encode())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", command)
    with pytest.raises(module.QuickstartError, match=message):
        module._run_cli(["--version"], cwd=tmp_path)


def test_quickstart_rejects_nonzero_and_non_utf8_command_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()

    def rejected(*_args: object, **kwargs: object) -> SimpleNamespace:
        kwargs["stderr"].write(b"rejected")
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(module.subprocess, "run", rejected)
    with pytest.raises(module.QuickstartError, match="rejected the fixture"):
        module._run_cli(["verify"], cwd=tmp_path)

    def non_utf8(*_args: object, **kwargs: object) -> SimpleNamespace:
        kwargs["stdout"].write(b"\xff")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", non_utf8)
    with pytest.raises(module.QuickstartError, match="UTF-8"):
        module._run_cli(["verify"], cwd=tmp_path)


@pytest.mark.parametrize("failure", ["verdict", "outputs"])
def test_quickstart_cleans_partial_outputs_for_invalid_success_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    module = _module()
    output = tmp_path / "output"
    calls = 0

    def command(_arguments: list[str], *, cwd: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            if failure == "outputs":
                (cwd / "verification.receipt.json").write_text("{}", encoding="utf-8")
            return {"ok": failure != "verdict", "policy_verdict": "pass"}
        return {"ok": True}

    monkeypatch.setattr(module, "_run_cli", command)
    with pytest.raises(module.QuickstartError):
        module.run_quickstart(fixture=FIXTURE, output=output)
    assert not output.exists()


def test_quickstart_main_reports_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    output = tmp_path / "output"
    monkeypatch.setattr(
        module,
        "run_quickstart",
        lambda **_kwargs: {
            "receipt": output / "receipt.json",
            "verification": output / "result.json",
            "report": output / "report.html",
        },
    )
    assert module.main([]) == 0
    assert "Decision: pass" in capsys.readouterr().out

    def rejected(**_kwargs: object) -> None:
        raise module.QuickstartError("rejected")

    monkeypatch.setattr(module, "run_quickstart", rejected)
    assert module.main([]) == 2
    assert "FAIL rejected" in capsys.readouterr().err
