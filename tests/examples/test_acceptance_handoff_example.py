from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import jsonschema
import pytest

from invarlock.acceptance_attestation import verify_acceptance_attestation
from invarlock.evidence_pack_verification import verify_comparison_evidence
from invarlock.evidence_receipt import verify_signed_verification_receipt
from invarlock.public_contracts import load_recipient_acceptance_policy_schema

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "examples/run_acceptance_handoff.py"
GOLDEN = REPO_ROOT / "examples/acceptance-handoff/golden"
REFERENCE_POLICY = (
    REPO_ROOT / "examples/acceptance-handoff/recipient-policy.example.json"
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("acceptance_handoff", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_recipient_policy_is_schema_valid() -> None:
    policy = json.loads(REFERENCE_POLICY.read_bytes())

    jsonschema.Draft202012Validator(load_recipient_acceptance_policy_schema()).validate(
        policy
    )
    assert policy["expected_predicate_type"].endswith("/acceptance/v2")
    assert policy["required_technical_verdict"] == "pass"
    assert policy["trusted_signers"][0]["status"] == "active"
    assert policy["trusted_receipt_verifiers"][0]["status"] == "active"
    assert policy["freshness"]["max_evidence_age_seconds"] is None


def test_committed_golden_package_verifies_end_to_end() -> None:
    anchors = json.loads((GOLDEN / "technical-anchors.json").read_bytes())
    recipient_policy = json.loads((GOLDEN / "recipient-policy.json").read_bytes())
    evidence_result = verify_comparison_evidence(
        GOLDEN / "evidence",
        policy_path=GOLDEN / "evaluated-policy.json",
        expected_artifact_digests=anchors["artifact_digests"],
        expected_schedule_digest=anchors["schedule_digest"],
        expected_runtime_digests=anchors["runtime_digests"],
        expected_signer_fingerprint=anchors["evidence_signer_fingerprint"],
    )
    receipt_result = verify_signed_verification_receipt(
        GOLDEN / "verification.receipt.json",
        GOLDEN / "evidence",
        policy_path=GOLDEN / "evaluated-policy.json",
        expected_artifact_digests=anchors["artifact_digests"],
        expected_schedule_digest=anchors["schedule_digest"],
        expected_runtime_digests=anchors["runtime_digests"],
        expected_pack_signer_fingerprint=anchors["evidence_signer_fingerprint"],
        expected_verifier_identity=anchors["verifier_identity"],
        expected_verifier_fingerprint=anchors["verifier_fingerprint"],
    )
    acceptance = verify_acceptance_attestation(
        GOLDEN / "acceptance.dsse.json",
        trusted_public_keys={
            anchors["envelope_signer_fingerprint"]: (
                GOLDEN / "envelope-signer.public.pem"
            )
        },
        recipient_policy=recipient_policy,
        subject_artifact_path=GOLDEN / "artifact",
        now=datetime(2026, 7, 25, 12, 5, tzinfo=UTC),
    )

    assert evidence_result.payload["ok"] is True
    assert receipt_result.ok is True
    assert acceptance.accepted is True
    assert acceptance.envelope_authenticated is True
    assert acceptance.receipt_authenticated is True
    assert acceptance.subject_bound is True


def test_committed_golden_package_is_exact_generator_output(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "golden"
    _module().write_golden(generated)

    committed_files = {
        path.relative_to(GOLDEN): path.read_bytes()
        for path in GOLDEN.rglob("*")
        if path.is_file()
    }
    generated_files = {
        path.relative_to(generated): path.read_bytes()
        for path in generated.rglob("*")
        if path.is_file()
    }
    assert generated_files == committed_files


def test_offline_acceptance_handoff_covers_current_policy_failures(
    tmp_path: Path,
) -> None:
    module = _module()
    workspace = tmp_path / "handoff"

    module.run_handoff(workspace)

    results = json.loads((workspace / "results.json").read_bytes())
    assert results == {
        "format": "invarlock/acceptance-handoff-v1",
        "historical_technical_verification": True,
        "scenarios": {
            "accepted": True,
            "contradictory_receipt_envelope_rejected": True,
            "missing_evidence_timestamp_rejected": True,
            "revoked_signer_rejected": True,
            "stale_envelope_rejected": True,
            "stricter_policy_rejected": True,
            "tampered_envelope_rejected": True,
            "tampered_evidence_rejected": True,
            "unknown_receipt_verifier_rejected": True,
            "unknown_signer_rejected": True,
            "wrong_artifact_rejected": True,
        },
    }
    assert (workspace / "handoff/artifacts/subject/model.safetensors").is_file()
    assert (workspace / "handoff/evidence/manifest.json").is_file()
    assert (workspace / "handoff/verification.receipt.json").is_file()
    assert (workspace / "handoff/acceptance.dsse.json").is_file()
    assert (workspace / "recipient/policy.json").is_file()
    assert (workspace / "recipient/trust/envelope-signer.public.pem").is_file()


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_handoff_refuses_to_reuse_workspace(tmp_path: Path, kind: str) -> None:
    module = _module()
    workspace = tmp_path / "handoff"
    if kind == "directory":
        workspace.mkdir()
    else:
        workspace.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises(RuntimeError, match="must be new"):
        module.run_handoff(workspace)


def test_main_runs_handoff_in_explicit_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    workspace = tmp_path / "explicit"
    calls: list[Path] = []

    def run(path: Path) -> None:
        calls.append(path)
        path.mkdir(parents=True)
        path.joinpath("results.json").write_bytes(
            GOLDEN.joinpath("results.json").read_bytes()
        )

    monkeypatch.setattr(module, "run_handoff", run)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--workspace", str(workspace)])

    assert module.main() == 0
    assert calls == [workspace.resolve()]
    assert capsys.readouterr().out == (
        "PASS offline acceptance handoff\n"
        "Fixture decision: accepted\n"
        "Fail-closed scenarios rejected: 10/10\n"
        f"Signed evidence: {workspace.resolve() / 'handoff/evidence'}\n"
        "Signed verifier receipt: "
        f"{workspace.resolve() / 'handoff/verification.receipt.json'}\n"
        "Acceptance envelope: "
        f"{workspace.resolve() / 'handoff/acceptance.dsse.json'}\n"
        f"Scenario results: {workspace.resolve() / 'results.json'}\n"
        f"Workspace: {workspace.resolve()}\n"
    )


def test_main_preserves_explicit_workspace_symlink_for_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    missing = tmp_path / "missing-workspace"
    linked = tmp_path / "linked-workspace"
    linked.symlink_to(missing, target_is_directory=True)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--workspace", str(linked)])

    with pytest.raises(RuntimeError, match="must be new"):
        module.main()
    assert not missing.exists()


def test_main_uses_fresh_default_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    calls: list[Path] = []

    def run(path: Path) -> None:
        calls.append(path)
        path.mkdir(parents=True)
        path.joinpath("results.json").write_bytes(
            GOLDEN.joinpath("results.json").read_bytes()
        )

    monkeypatch.setattr(module, "run_handoff", run)
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **_kwargs: str(tmp_path))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])

    assert module.main() == 0
    assert calls == [tmp_path.resolve() / "workspace"]


def test_main_writes_golden_package(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    calls: list[bool] = []
    monkeypatch.setattr(module, "write_golden", lambda: calls.append(True))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--write-golden"])

    assert module.main() == 0
    assert calls == [True]
    assert capsys.readouterr().out == (
        f"PASS generated golden handoff package: {module.GOLDEN_ROOT}\n"
    )


def test_main_rejects_conflicting_output_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--workspace",
            str(tmp_path / "explicit"),
            "--write-golden",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        module.main()


def test_copy_golden_refuses_existing_destination(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()

    with pytest.raises(RuntimeError, match="destination must be new"):
        module._copy_golden(source, destination)


def test_contradictory_envelope_rejects_non_ed25519_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    source = tmp_path / "source.dsse.json"
    destination = tmp_path / "destination.dsse.json"
    private_key = tmp_path / "private.pem"
    statement = {
        "predicate": {"technical_verdict": {"policy_verdict": "pass"}},
    }
    source.write_bytes(
        module._canonical(
            {
                "payload": module.base64.b64encode(module._canonical(statement)).decode(
                    "ascii"
                ),
                "payloadType": module.DSSE_PAYLOAD_TYPE,
                "signatures": [{"keyid": "unused", "sig": "unused"}],
            }
        )
    )
    private_key.write_bytes(b"not-used")
    monkeypatch.setattr(
        module.serialization,
        "load_pem_private_key",
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(RuntimeError, match="not Ed25519"):
        module._contradictory_envelope(source, destination, private_key)


def test_handoff_fails_closed_when_any_scenario_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "_decision", lambda *_args, **_kwargs: False)

    with pytest.raises(RuntimeError, match="did not satisfy every scenario"):
        module.run_handoff(tmp_path / "handoff")
