"""Execute the receipt-reader examples with signed acceptance and rejection."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from invarlock.evidence_receipt import write_signed_verification_receipt
from tests.evidence_packs.test_evidence_receipt import (
    _input_anchor_kwargs,
    _inputs,
    _key,
    _result,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("page", ["evidence-and-verification", "key-management"])
@pytest.mark.parametrize("accepted", [True, False])
def test_documented_receipt_reader_checks_authentication_and_verdict(
    tmp_path: Path, page: str, accepted: bool
) -> None:
    # Build an authentic receipt independently of the documentation placeholders.
    pack, policy, runtimes, signer = _inputs(tmp_path)
    key, verifier = _key(tmp_path, "verifier")
    receipt = tmp_path / "verification.receipt.json"
    result = _result(pack, policy, runtimes, signer)
    result.payload.update(ok=accepted, policy_verdict="pass" if accepted else "fail")
    write_signed_verification_receipt(
        pack,
        result,
        receipt,
        policy_path=policy,
        **_input_anchor_kwargs(),
        expected_runtime_digests=runtimes,
        expected_pack_signer_fingerprint=signer,
        verifier_identity="release-verifier",
        verifier_signing_key_path=key,
    )
    text = (ROOT / f"docs/user-guide/{page}.md").read_text()
    snippets = [
        block
        for block in re.findall(r"```python\n(.*?)\n```", text, re.S)
        if "verify_signed_verification_receipt(" in block
    ]
    assert len(snippets) == 1
    code = snippets[0]
    for original, replacement in (
        ("verification.receipt.json", receipt),
        ("evidence", pack),
        ("trusted/acceptance.json", policy),
    ):
        code = code.replace(f'Path("{original}")', f"Path({str(replacement)!r})")
    placeholders = {
        "1": runtimes["baseline"],
        "2": runtimes["subject"],
        "3": signer,
        "4": verifier,
        "5": "sha256:" + "d" * 64,
        "6": "sha256:" + "e" * 64,
        "7": "sha256:" + "f" * 64,
    }
    for digit, value in placeholders.items():
        code = code.replace(f'"sha256:" + "{digit}" * 64', repr(value))
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=20
    )
    if accepted:
        assert completed.returncode == 0, completed.stderr
    else:
        assert completed.returncode != 0
        assert "authenticated receipt records a rejection" in completed.stderr
