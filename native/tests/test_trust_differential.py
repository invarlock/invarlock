"""Closed native trust snapshots against the source loader."""

import json
from pathlib import Path

import pytest

from invarlock.trust_inputs import load_trust_inputs
from native.tests.test_judge_evidence_differential import native
from tests.evidence_packs.test_trust_inputs import _material


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "request",
        "unknown",
        "traversal",
        "missing",
        "link",
        "duplicate",
        "nonobject",
    ],
)
def test_closed_profile(case, tmp_path):
    root, payload = _material(tmp_path.resolve())
    profile = root / "trust.json"
    if case == "request":
        payload["anchors"]["request_digest"] = "sha256:" + "1" * 64
    elif case == "unknown":
        payload["extra"] = 1
    elif case == "traversal":
        payload["policy"]["path"] = "../policy.json"
    elif case == "missing":
        (root / "policy.json").unlink()
    elif case == "link":
        policy = root / "policy.json"
        policy.rename(root / "original.json")
        policy.symlink_to(root / "original.json")
    profile.write_text(json.dumps(payload))
    if case == "duplicate":
        profile.write_text('{"format":1,"format":2}')
    elif case == "nonobject":
        profile.write_text("[]")
    try:
        expected = load_trust_inputs(profile)
    except ValueError as failure:
        status, actual = native("load-trust", {"trust_path": str(profile)})
        assert status != 0
        assert actual["error"] == str(failure)
    else:
        status, actual = native("load-trust", {"trust_path": str(profile)})
        assert status == 0, actual
        assert actual["trust_profile_digest"] == expected.profile_digest
        assert Path(actual["policy_path"]) == expected.policy_path
        assert actual["verifier_identity"] == expected.verifier_identity
        assert actual["expected_request_digest"] == expected.expected_request_digest
        assert "signing_key_bytes" not in actual
