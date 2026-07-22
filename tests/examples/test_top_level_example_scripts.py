from __future__ import annotations

import hashlib
import json
import os
import runpy
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from examples import generate_keys, regenerate_fixtures
from examples import run_trust_boundary_demo as trust_demo

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples"


def test_generate_key_writes_a_private_ed25519_key_with_matching_fingerprint(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "evidence-signer.pem"

    fingerprint = generate_keys._write_key(destination)

    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    private_key = serialization.load_pem_private_key(
        destination.read_bytes(), password=None
    )
    assert isinstance(private_key, ed25519.Ed25519PrivateKey)
    assert fingerprint == generate_keys._fingerprint(private_key.public_key())


def test_generate_key_removes_a_partial_file_when_writing_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "broken.pem"
    real_fdopen = os.fdopen

    class BrokenStream:
        def __init__(self, descriptor: int) -> None:
            self.descriptor = descriptor

        def __enter__(self) -> BrokenStream:
            return self

        def __exit__(self, *_args: object) -> None:
            os.close(self.descriptor)

        def write(self, _payload: bytes) -> None:
            raise OSError("simulated storage failure")

    def broken_fdopen(descriptor: int, mode: str) -> BrokenStream:
        assert mode == "wb"
        return BrokenStream(descriptor)

    monkeypatch.setattr(generate_keys.os, "fdopen", broken_fdopen)
    try:
        with pytest.raises(OSError, match="storage failure"):
            generate_keys._write_key(destination)
    finally:
        monkeypatch.setattr(generate_keys.os, "fdopen", real_fdopen)

    assert not destination.exists()


@pytest.mark.parametrize(
    ("arguments", "expected_roles"),
    [([], {"evidence-signer", "verifier"}), (["--role", "verifier"], {"verifier"})],
)
def test_generate_keys_main_supports_default_and_single_role_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    expected_roles: set[str],
) -> None:
    destination = tmp_path / ("keys-" + "-".join(arguments or ["both"]))
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_keys.py", "--output-dir", str(destination), *arguments],
    )

    generate_keys.main()

    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert {path.stem for path in destination.glob("*.pem")} == expected_roles
    assert {path.stem for path in destination.glob("*.fingerprint")} == expected_roles
    assert all(
        path.read_text(encoding="ascii").startswith("sha256:")
        for path in destination.glob("*.fingerprint")
    )
    output = capsys.readouterr().out
    assert "never the private PEM files" in output
    for role in expected_roles:
        assert f"{role} key:" in output


def test_generate_keys_main_removes_the_output_directory_after_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "keys"
    calls = 0

    def fail_second_key(path: Path) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("second key failed")
        path.write_text("private", encoding="ascii")
        return "sha256:" + "a" * 64

    monkeypatch.setattr(generate_keys, "_write_key", fail_second_key)
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_keys.py", "--output-dir", str(destination)],
    )

    with pytest.raises(OSError, match="second key failed"):
        generate_keys.main()

    assert not destination.exists()


@pytest.mark.parametrize(
    ("returncode", "expect_success", "expected_outcome"),
    [(1, True, "succeed"), (0, False, "fail closed")],
)
def test_demo_run_reports_unexpected_command_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    expect_success: bool,
    expected_outcome: str,
) -> None:
    monkeypatch.setattr(
        trust_demo.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["invarlock"],
            returncode,
            stdout="captured stdout",
            stderr="captured stderr",
        ),
    )

    with pytest.raises(RuntimeError, match=expected_outcome) as error:
        trust_demo._run("verify", "evidence", expect_success=expect_success)

    assert "captured stdout" in str(error.value)
    assert "captured stderr" in str(error.value)


@pytest.mark.parametrize(
    ("stderr", "stdout", "message"),
    [("key stderr", "", "key stderr"), ("", "key stdout", "key stdout")],
)
def test_demo_key_generation_surfaces_subprocess_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stderr: str,
    stdout: str,
    message: str,
) -> None:
    monkeypatch.setattr(
        trust_demo.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["generate_keys.py"], 1, stdout=stdout, stderr=stderr
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        trust_demo._generate_key(EXAMPLE_ROOT, tmp_path / "keys", "verifier")


@pytest.mark.parametrize(
    ("verified", "message"),
    [
        (
            SimpleNamespace(ok=False, errors=("bad signature",), statement=None),
            "bad signature",
        ),
        (
            SimpleNamespace(ok=True, errors=(), statement=None),
            "no authenticated statement",
        ),
        (
            SimpleNamespace(ok=True, errors=(), statement={"verdict": "pass"}),
            "verdict is malformed",
        ),
        (
            SimpleNamespace(
                ok=True,
                errors=(),
                statement={"verdict": {"ok": False, "policy_verdict": "pass"}},
            ),
            "acceptance does not match",
        ),
        (
            SimpleNamespace(
                ok=True,
                errors=(),
                statement={"verdict": {"ok": True, "policy_verdict": "fail"}},
            ),
            "policy verdict does not match",
        ),
    ],
)
def test_demo_receipt_checks_fail_closed_on_invalid_authenticated_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    verified: SimpleNamespace,
    message: str,
) -> None:
    monkeypatch.setattr(
        trust_demo,
        "verify_signed_verification_receipt",
        lambda *_args, **_kwargs: verified,
    )
    anchors = {
        "baseline": "sha256:" + "a" * 64,
        "subject": "sha256:" + "b" * 64,
        "dataset": "sha256:" + "c" * 64,
    }

    with pytest.raises(RuntimeError, match=message):
        trust_demo._verify_receipt(
            tmp_path / "receipt.json",
            tmp_path / "evidence",
            tmp_path / "policy.json",
            "sha256:" + "d" * 64,
            "sha256:" + "e" * 64,
            anchors,
            expected_acceptance=True,
            expected_policy_verdict="pass",
        )


def test_demo_material_anchors_bind_the_exact_input_bytes(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    schedule = tmp_path / "schedule.json"
    baseline.write_bytes(b"baseline")
    subject.write_bytes(b"subject")
    schedule.write_bytes(b"schedule")

    anchors = trust_demo._material_anchors(baseline, subject, schedule)

    assert anchors == {
        "baseline": "sha256:" + hashlib.sha256(b"baseline").hexdigest(),
        "subject": "sha256:" + hashlib.sha256(b"subject").hexdigest(),
        "dataset": "sha256:" + hashlib.sha256(b"schedule").hexdigest(),
    }


def test_trust_boundary_demo_runs_the_real_isolated_handoff(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "trust-boundary"

    trust_demo.run_demo(EXAMPLE_ROOT, workspace)

    output = capsys.readouterr().out
    assert "PASS accepted evidence and receipt" in output
    assert "PASS human-readable report" in output
    assert "PASS authentic policy rejection" in output
    assert "PASS byte-tamper rejection" in output
    accepted = json.loads(
        (workspace / "verifier/receipts/accepted.receipt.json").read_bytes()
    )
    rejected = json.loads(
        (workspace / "verifier/receipts/policy-rejected.receipt.json").read_bytes()
    )
    tampered = json.loads(
        (workspace / "verifier/receipts/tampered.receipt.json").read_bytes()
    )
    assert accepted["statement"]["verdict"]["ok"] is True
    assert rejected["statement"]["verdict"]["policy_verdict"] == "fail"
    assert tampered["statement"]["verdict"]["integrity_ok"] is False


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_trust_boundary_demo_refuses_to_reuse_a_workspace(
    tmp_path: Path,
    kind: str,
) -> None:
    workspace = tmp_path / "workspace"
    if kind == "directory":
        workspace.mkdir()
    else:
        workspace.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises(RuntimeError, match="already exists"):
        trust_demo.run_demo(EXAMPLE_ROOT, workspace)


@pytest.mark.parametrize("use_default", [True, False])
def test_trust_boundary_main_selects_a_temporary_or_explicit_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    use_default: bool,
) -> None:
    calls: list[tuple[Path, Path]] = []
    temporary_parent = tmp_path / "temporary-parent"
    temporary_parent.mkdir()
    arguments = ["run_trust_boundary_demo.py"]
    expected = temporary_parent / "workspace"
    if not use_default:
        expected = tmp_path / "explicit"
        arguments.extend(["--workspace", str(expected)])
    monkeypatch.setattr(
        trust_demo.tempfile, "mkdtemp", lambda **_kwargs: str(temporary_parent)
    )
    monkeypatch.setattr(
        trust_demo, "run_demo", lambda root, workspace: calls.append((root, workspace))
    )
    monkeypatch.setattr(sys, "argv", arguments)

    trust_demo.main()

    assert calls == [(EXAMPLE_ROOT, expected.resolve())]


def _empty_fixture_root(root: Path) -> None:
    (root / "policy").mkdir(parents=True)
    (root / "inputs").mkdir()
    (root / "trusted-inputs").mkdir()
    (root / "policy/acceptance.json").write_bytes(
        (EXAMPLE_ROOT / "policy/acceptance.json").read_bytes()
    )


def test_regenerate_fixtures_matches_the_checked_in_golden_files(
    tmp_path: Path,
) -> None:
    generated_root = tmp_path / "examples"
    _empty_fixture_root(generated_root)

    regenerate_fixtures.regenerate(generated_root)

    generated_files = sorted(
        path.relative_to(generated_root)
        for path in generated_root.rglob("*")
        if path.is_file() and path.name != "acceptance.json"
    )
    assert len(generated_files) == 22
    for relative in generated_files:
        generated = generated_root / relative
        checked_in = EXAMPLE_ROOT / relative
        assert checked_in.is_file(), relative
        assert generated.read_bytes() == checked_in.read_bytes(), relative
        if relative.parts[0] == "import":
            assert stat.S_IMODE(generated.stat().st_mode) == 0o644

    paired = json.loads((generated_root / "import/paired-records.json").read_bytes())
    rejected = json.loads(
        (generated_root / "import/rejected-paired-records.json").read_bytes()
    )
    assert len(paired["records"]) == 50
    assert sum(record["subject"]["score"] for record in paired["records"]) == 50
    assert sum(record["subject"]["score"] for record in rejected["records"]) == 49


def test_copy_generated_replaces_an_existing_destination_and_normalizes_mode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "nested/destination.json"
    source.write_text("new", encoding="ascii")
    destination.parent.mkdir()
    destination.write_text("old", encoding="ascii")
    destination.chmod(0o600)

    regenerate_fixtures._copy_generated(source, destination)

    assert destination.read_text(encoding="ascii") == "new"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o644


def test_fixture_record_generation_rejects_output_count_drift() -> None:
    schedule = SimpleNamespace(
        records=[SimpleNamespace(record_id="record-00", input_sha256="a" * 64)]
    )

    with pytest.raises(ValueError, match="shorter than argument 1"):
        regenerate_fixtures._records(schedule, [])


def test_regenerate_fixtures_main_requires_explicit_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["regenerate_fixtures.py"])

    with pytest.raises(SystemExit) as error:
        regenerate_fixtures.main()

    assert error.value.code == 2


def test_regenerate_fixtures_main_dispatches_to_the_example_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots: list[Path] = []
    monkeypatch.setattr(sys, "argv", ["regenerate_fixtures.py", "--write"])
    monkeypatch.setattr(regenerate_fixtures, "regenerate", roots.append)

    assert regenerate_fixtures.main() == 0
    assert roots == [EXAMPLE_ROOT]


@pytest.mark.parametrize(
    "script",
    ["generate_keys.py", "regenerate_fixtures.py", "run_trust_boundary_demo.py"],
)
def test_top_level_example_scripts_expose_cli_help(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    script: str,
) -> None:
    monkeypatch.setattr(sys, "argv", [script, "--help"])

    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(EXAMPLE_ROOT / script), run_name="__main__")

    assert error.value.code == 0
    assert "usage:" in capsys.readouterr().out
