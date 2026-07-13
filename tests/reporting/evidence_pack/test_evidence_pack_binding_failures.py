from __future__ import annotations

from invarlock.evidence_pack_integrity import _manual_validate_manifest


def test_manifest_rejects_malformed_verification_materials() -> None:
    errors = _manual_validate_manifest(
        {
            "verification_baselines": [{}, "not-an-object"],
            "verification_policy_pack": {
                "path": "policy/other.json",
                "digest": "sha256:" + ("a" * 64),
                "policy_digest": "invalid",
            },
        }
    )

    assert (
        "manifest verification_baselines[0].name must be a non-empty string" in errors
    )
    assert (
        "manifest verification_baselines[0].report_paths must be a non-empty list"
        in errors
    )
    assert (
        "manifest verification_policy_pack.path must point to "
        "'policy/policy-pack.json'" in errors
    )
    assert (
        "manifest verification_policy_pack.policy_digest must be a sha256:... string"
        in errors
    )
    empty_errors = _manual_validate_manifest({"verification_baselines": []})
    assert "manifest verification_baselines must be a non-empty list" in empty_errors
    wrong_type_errors = _manual_validate_manifest(
        {"verification_policy_pack": "not-an-object"}
    )
    assert "manifest verification_policy_pack must be an object" in wrong_type_errors
