from __future__ import annotations

import base64
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "policy-engine-interop"
GOLDEN = ROOT / "examples" / "acceptance-handoff" / "golden"
SUBJECT_SHA256 = "a9fcf5a7cb042b0f4db67dead3d64fad8c3775d7ea25c91ee6759b019b5603cb"


def _runner_module() -> ModuleType:
    script = EXAMPLE / "run.py"
    spec = importlib.util.spec_from_file_location("policy_engine_runner", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verifier_module() -> ModuleType:
    script = EXAMPLE / "verify_envelope.py"
    spec = importlib.util.spec_from_file_location("policy_envelope_verifier", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_builder_module() -> ModuleType:
    verifier = _verifier_module()
    previous = sys.modules.get("verify_envelope")
    sys.modules["verify_envelope"] = verifier
    try:
        script = EXAMPLE / "build_fixtures.py"
        spec = importlib.util.spec_from_file_location("policy_fixture_builder", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("verify_envelope", None)
        else:
            sys.modules["verify_envelope"] = previous


def test_standalone_verifier_reproduces_positive_fixture() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "verify_envelope.py"),
            "--envelope",
            str(GOLDEN / "acceptance.dsse.json"),
            "--envelope-key",
            str(GOLDEN / "envelope-signer.public.pem"),
            "--recipient-policy",
            str(GOLDEN / "recipient-policy.json"),
            "--expected-subject-name",
            "artifact.example/subject",
            "--expected-subject-sha256",
            SUBJECT_SHA256,
            "--now",
            "2026-07-25T12:30:00Z",
        ],
        check=True,
        capture_output=True,
    )

    assert completed.stdout == (EXAMPLE / "fixtures" / "positive.json").read_bytes()


def test_standalone_verifier_rejects_tampered_subject(tmp_path: Path) -> None:
    envelope = json.loads((GOLDEN / "acceptance.dsse.json").read_bytes())
    statement = json.loads(base64.b64decode(envelope["payload"], validate=True))
    statement["subject"][0]["digest"]["sha256"] = "0" * 64
    payload = (
        json.dumps(statement, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    envelope["payload"] = base64.b64encode(payload).decode()
    tampered = tmp_path / "tampered.dsse.json"
    tampered.write_text(
        json.dumps(envelope, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "verify_envelope.py"),
            "--envelope",
            str(tampered),
            "--envelope-key",
            str(GOLDEN / "envelope-signer.public.pem"),
            "--recipient-policy",
            str(GOLDEN / "recipient-policy.json"),
            "--expected-subject-name",
            "artifact.example/subject",
            "--expected-subject-sha256",
            SUBJECT_SHA256,
            "--now",
            "2026-07-25T12:30:00Z",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""


def test_fixtures_are_current_and_cover_six_scenarios() -> None:
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE / "build_fixtures.py"), "--check"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    expectations = json.loads((EXAMPLE / "fixtures" / "expectations.json").read_bytes())
    assert expectations["scenarios"] == {
        "policy-rejected": {
            "allow": False,
            "reasons": ["technical_verdict_rejected"],
        },
        "positive": {"allow": True, "reasons": []},
        "stale-evidence": {"allow": False, "reasons": ["stale_evidence"]},
        "tampered-subject": {
            "allow": False,
            "reasons": ["authentication_failed", "subject_rejected"],
        },
        "unsupported-contract": {
            "allow": False,
            "reasons": ["unsupported_contract"],
        },
        "untrusted-signer": {"allow": False, "reasons": ["untrusted_signer"]},
    }


def test_fixture_builder_writes_checks_and_rejects_stale_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _fixture_builder_module()
    fixtures = tmp_path / "fixtures"
    monkeypatch.setattr(module, "FIXTURES", fixtures)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: type("Args", (), {"check": True})(),
    )
    with pytest.raises(SystemExit, match="fixtures are stale"):
        module.main()

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: type("Args", (), {"check": False})(),
    )
    module.main()
    assert len(list(fixtures.glob("*.json"))) == 7

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: type("Args", (), {"check": True})(),
    )
    module.main()


def test_standalone_verifier_strict_parsing_and_crypto_helpers() -> None:
    module = _verifier_module()
    with pytest.raises(ValueError, match="duplicate key"):
        module.strict_json_bytes(b'{"a":1,"a":2}', label="test")
    with pytest.raises(ValueError, match="JSON object"):
        module.strict_json_bytes(b"[]", label="test")
    with pytest.raises(ValueError, match="base64 text"):
        module.decode_base64(7, label="test")
    with pytest.raises(ValueError, match="invalid base64"):
        module.decode_base64("***", label="test")

    key = ec.generate_private_key(ec.SECP256R1()).public_key()
    pem = key.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    with pytest.raises(ValueError, match="must be Ed25519"):
        module.load_ed25519_public_key(pem)
    with pytest.raises(ValueError, match="timezone"):
        module.unix_seconds("2026-07-25T12:00:00")
    assert module.unix_seconds("2026-07-25T12:00:00+00:00") == 1784980800


def test_standalone_verifier_rejects_malformed_envelope_structures(
    tmp_path: Path,
) -> None:
    module = _verifier_module()
    original = json.loads((GOLDEN / "acceptance.dsse.json").read_bytes())
    key = GOLDEN / "envelope-signer.public.pem"

    mutations = [
        ({**original, "extra": True}, "fields are invalid"),
        ({**original, "payloadType": "wrong"}, "payload type is unsupported"),
        ({**original, "signatures": []}, "exactly one signature"),
        (
            {**original, "signatures": [{"keyid": "wrong", "sig": "x", "extra": 1}]},
            "signature fields are invalid",
        ),
        (
            {
                **original,
                "signatures": [{**original["signatures"][0], "keyid": "wrong"}],
            },
            "key ID does not match",
        ),
    ]
    statement = json.loads(base64.b64decode(original["payload"], validate=True))
    noncanonical = json.dumps(statement, indent=2).encode()
    mutations.append(
        (
            {**original, "payload": base64.b64encode(noncanonical).decode()},
            "must use canonical JSON",
        )
    )
    for index, (envelope, message) in enumerate(mutations):
        path = tmp_path / f"envelope-{index}.json"
        path.write_text(json.dumps(envelope), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            module.verify_envelope(envelope_path=path, envelope_key_path=key)


def test_standalone_verifier_rejects_inconsistent_receipt_projections() -> None:
    module = _verifier_module()
    statement, _ = module.verify_envelope(
        envelope_path=GOLDEN / "acceptance.dsse.json",
        envelope_key_path=GOLDEN / "envelope-signer.public.pem",
    )

    bad_digest = copy.deepcopy(statement)
    bad_digest["predicate"]["receipt"]["digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="digest is invalid"):
        module.verify_receipt(bad_digest)

    bad_representation = copy.deepcopy(statement)
    raw = base64.b64decode(
        bad_representation["predicate"]["receipt"]["raw_base64"],
        validate=True,
    )
    content = json.loads(raw)
    noncanonical = json.dumps(content, indent=2).encode()
    receipt = bad_representation["predicate"]["receipt"]
    receipt["raw_base64"] = base64.b64encode(noncanonical).decode()
    receipt["digest"] = f"sha256:{module.hashlib.sha256(noncanonical).hexdigest()}"
    with pytest.raises(ValueError, match="representation is inconsistent"):
        module.verify_receipt(bad_representation)

    bad_encoding = copy.deepcopy(statement)
    bad_encoding["predicate"]["receipt"]["content"]["signature"]["public_key"][
        "encoding"
    ] = "der"
    raw_content = module.canonical_bytes(
        bad_encoding["predicate"]["receipt"]["content"]
    )
    bad_encoding["predicate"]["receipt"]["raw_base64"] = base64.b64encode(
        raw_content
    ).decode()
    bad_encoding["predicate"]["receipt"]["digest"] = (
        f"sha256:{module.hashlib.sha256(raw_content).hexdigest()}"
    )
    with pytest.raises(ValueError, match="encoding is unsupported"):
        module.verify_receipt(bad_encoding)

    bad_projection = copy.deepcopy(statement)
    bad_projection["predicate"]["signers"]["receipt"]["identity"] = "wrong"
    with pytest.raises(ValueError, match="projection is inconsistent"):
        module.verify_receipt(bad_projection)


def test_standalone_verifier_rejects_envelope_signer_projection_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _verifier_module()
    statement, verification = module.verify_envelope(
        envelope_path=GOLDEN / "acceptance.dsse.json",
        envelope_key_path=GOLDEN / "envelope-signer.public.pem",
    )
    verification["envelope_signer_fingerprint"] = "sha256:" + "0" * 64
    monkeypatch.setattr(
        module,
        "verify_envelope",
        lambda **_kwargs: (statement, verification),
    )
    with pytest.raises(ValueError, match="envelope-signer projection"):
        module.build_policy_input(
            envelope_path=GOLDEN / "acceptance.dsse.json",
            envelope_key_path=GOLDEN / "envelope-signer.public.pem",
            recipient_policy_path=GOLDEN / "recipient-policy.json",
            expected_subject_name="artifact.example/subject",
            expected_subject_sha256=SUBJECT_SHA256,
            now="2026-07-25T12:30:00Z",
        )


def test_verifier_and_policy_launchers_have_no_invarlock_or_network_dependency() -> (
    None
):
    verifier = (EXAMPLE / "verify_envelope.py").read_text(encoding="utf-8")
    launcher = (EXAMPLE / "run.py").read_text(encoding="utf-8")
    assert "from invarlock" not in verifier
    assert "import invarlock" not in verifier
    assert "requests" not in verifier
    assert "urllib" not in verifier
    assert "socket" not in verifier
    assert "http.send" not in (EXAMPLE / "policy" / "acceptance.rego").read_text(
        encoding="utf-8"
    )
    assert "invarlock-qualify" not in launcher
    assert "-m invarlock" not in launcher


def test_policy_runner_parses_engine_outputs_and_rejects_non_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner_module()

    class Completed:
        stdout = '{"allow":true,"reasons":[]}'
        stderr = ""
        returncode = 0

    calls: list[list[str]] = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command) or Completed(),
    )
    fixture = EXAMPLE / "fixtures/positive.json"
    assert module.version("opa") == Completed.stdout
    assert module.run_opa("opa", fixture) == {"allow": True, "reasons": []}
    assert module.cue_accepts("cue", fixture) is True
    assert calls[0] == ["opa", "version"]

    Completed.stdout = "[]"
    with pytest.raises(ValueError, match="must be an object"):
        module.run_opa("opa", fixture)

    Completed.returncode = 1
    assert module.cue_accepts("cue", fixture) is False


def test_policy_runner_main_checks_versions_and_scenario_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _runner_module()
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: type("Args", (), {"opa": "opa", "cue": "cue"})(),
    )
    versions = {"opa": "Version: 1.17.0", "cue": "cue version v0.16.1"}
    monkeypatch.setattr(module, "version", lambda executable: versions[executable])
    expectations = json.loads((EXAMPLE / "fixtures/expectations.json").read_bytes())
    allow = {name: value["allow"] for name, value in expectations["scenarios"].items()}
    monkeypatch.setattr(
        module,
        "run_opa",
        lambda _executable, fixture: expectations["scenarios"][fixture.stem],
    )
    monkeypatch.setattr(
        module,
        "cue_accepts",
        lambda _executable, fixture: allow[fixture.stem],
    )
    module.main()
    assert capsys.readouterr().out.count("opa=") == 6

    versions["opa"] = "Version: wrong"
    with pytest.raises(RuntimeError, match="expected OPA"):
        module.main()
    versions["opa"] = "Version: 1.17.0"
    versions["cue"] = "cue version wrong"
    with pytest.raises(RuntimeError, match="expected CUE"):
        module.main()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("allow", False, "OPA returned"),
        ("reasons", ["wrong"], "OPA returned"),
        ("cue", False, "CUE acceptance"),
    ],
)
def test_policy_runner_rejects_engine_disagreement(
    field: str,
    value: object,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner_module()
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "expectations.json").write_text(
        json.dumps(
            {
                "scenarios": {
                    "positive": {"allow": True, "reasons": []},
                }
            }
        ),
        encoding="utf-8",
    )
    (fixtures / "positive.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "FIXTURES", fixtures)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: type("Args", (), {"opa": "opa", "cue": "cue"})(),
    )
    monkeypatch.setattr(
        module,
        "version",
        lambda executable: "1.17.0" if executable == "opa" else "v0.16.1",
    )
    monkeypatch.setattr(
        module,
        "run_opa",
        lambda _executable, _fixture: {
            "allow": value if field == "allow" else True,
            "reasons": value if field == "reasons" else [],
        },
    )
    monkeypatch.setattr(
        module,
        "cue_accepts",
        lambda _executable, _fixture: value if field == "cue" else True,
    )
    with pytest.raises(AssertionError, match=message):
        module.main()


def test_pinned_opa_and_cue_conformance_when_available() -> None:
    opa = shutil.which("opa")
    cue = shutil.which("cue")
    if opa is None or cue is None:
        pytest.skip("pinned OPA and CUE are exercised by the dedicated CI gate")
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE / "run.py"), "--opa", opa, "--cue", cue],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("opa=") == 6
