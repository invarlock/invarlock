from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "policy-engine-interop"
GOLDEN = ROOT / "examples" / "acceptance-handoff" / "golden"
SUBJECT_SHA256 = "a9fcf5a7cb042b0f4db67dead3d64fad8c3775d7ea25c91ee6759b019b5603cb"


def test_standalone_verifier_reproduces_positive_fixture() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "verify_envelope.py"),
            "--envelope",
            str(GOLDEN / "acceptance.dsse.json"),
            "--envelope-key",
            str(GOLDEN / "producer.public.pem"),
            "--recipient-policy",
            str(GOLDEN / "recipient-policy.json"),
            "--expected-subject-name",
            "producer.example/subject",
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
            str(GOLDEN / "producer.public.pem"),
            "--recipient-policy",
            str(GOLDEN / "recipient-policy.json"),
            "--expected-subject-name",
            "producer.example/subject",
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
