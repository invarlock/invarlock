from __future__ import annotations

from copy import deepcopy

import pytest

import invarlock.policy_pack as policy_pack_mod
from invarlock.core.assurance_contract import (
    ASSURANCE_CLAIM_SET_V2,
    LEGACY_ASSURANCE_CLAIM_SET,
)
from invarlock.guards.authority import DEFAULT_GUARD_AUTHORITY
from invarlock.policy_pack import build_policy_pack
from invarlock.reporting import verify_policy
from invarlock.reporting.verify_policy import (
    append_strict_policy_authorization_errors,
)


def _policy() -> dict:
    return {
        "metrics": {
            "accuracy": {
                "delta_min_pp": -1.0,
                "min_examples": 200,
            }
        },
        "spectral": {"max_caps": 5},
        "rmt": {"epsilon_default": 0.01},
        "guard_authority": deepcopy(DEFAULT_GUARD_AUTHORITY),
    }


def _report(policy: dict | None = None, *, tier: str = "balanced") -> dict:
    return {
        "resolved_policy": deepcopy(policy if policy is not None else _policy()),
        "assurance": {"tier": tier, "claim_set": ASSURANCE_CLAIM_SET_V2},
        "auto": {"tier": tier},
        "dataset": {
            "provider": "local_jsonl",
            "dataset_name": None,
            "config_name": None,
            "revision": None,
            "split": "validation",
        },
    }


def _pack(
    policy: dict,
    *,
    tier: str = "balanced",
    support_tiers: list[str] | None = None,
) -> dict:
    return build_policy_pack(
        tier=tier,
        resolved_policy=policy,
        compatibility={
            "support_tiers": support_tiers or ["maintained_catalog"],
            "dataset_identity": deepcopy(_report()["dataset"]),
        },
    )


def _errors(report: dict, policy_pack: dict | None) -> list[str]:
    errors: list[str] = []
    append_strict_policy_authorization_errors(
        errors,
        report=report,
        policy_pack=policy_pack,
    )
    return errors


def test_strict_policy_authorization_accepts_exact_acceptance_policy() -> None:
    policy = _policy()
    pack = _pack(policy)

    assert _errors(_report(policy), pack) == []


def test_strict_policy_authorization_accepts_frozen_v1_pack() -> None:
    policy = _policy()
    policy.pop("guard_authority")
    pack = _pack(policy)
    pack["format"] = "policy-pack-v1"
    pack["resolved_policy"].pop("guard_authority")
    pack["compatibility"]["support_tiers"] = ["published_basis"]
    pack["policy_digest"] = policy_pack_mod._compute_policy_pack_digest(
        {key: value for key, value in pack.items() if key != "policy_digest"}
    )

    report = _report(policy)
    report["assurance"]["claim_set"] = LEGACY_ASSURANCE_CLAIM_SET
    assert _errors(report, pack) == []


def test_invalid_v1_policy_pack_diagnostic_names_submitted_format() -> None:
    policy = _policy()
    policy.pop("guard_authority")
    pack = _pack(policy)
    pack["format"] = "policy-pack-v1"
    pack["resolved_policy"]["guard_authority"] = deepcopy(DEFAULT_GUARD_AUTHORITY)
    pack["compatibility"]["support_tiers"] = ["published_basis"]
    pack["policy_digest"] = policy_pack_mod._compute_policy_pack_digest(
        {key: value for key, value in pack.items() if key != "policy_digest"}
    )

    errors = _errors(_report(policy), pack)

    assert any(error.startswith("Invalid policy-pack-v1:") for error in errors)
    assert not any("Invalid policy-pack-v2:" in error for error in errors)


def test_strict_policy_authorization_rejects_missing_external_pack() -> None:
    assert any(
        "independently supplied --policy-pack" in error
        for error in _errors(_report(), None)
    )


def test_strict_policy_authorization_rejects_self_loosened_threshold() -> None:
    authorized = _policy()
    submitted = deepcopy(authorized)
    submitted["metrics"]["accuracy"]["delta_min_pp"] = -100.0
    pack = _pack(authorized)

    assert any(
        "does not exactly match" in error for error in _errors(_report(submitted), pack)
    )


def test_strict_policy_authorization_rejects_tier_mismatch() -> None:
    pack = _pack(_policy(), tier="conservative")

    errors = _errors(_report(tier="balanced"), pack)

    assert any("report.assurance" in error for error in errors)
    assert any("report.auto" in error for error in errors)


def test_strict_policy_authorization_rejects_research_tier() -> None:
    pack = _pack(_policy(), tier="aggressive")

    assert any(
        "balanced or conservative" in error
        for error in _errors(_report(tier="aggressive"), pack)
    )


def test_strict_policy_authorization_rejects_tampered_pack_digest() -> None:
    pack = _pack(_policy())
    pack["policy_digest"] = "sha256:" + "0" * 64

    assert any("policy digest mismatch" in error for error in _errors(_report(), pack))


def test_strict_policy_authorization_requires_maintained_catalog_support() -> None:
    pack = _pack(_policy(), support_tiers=["community_experimental"])

    errors = _errors(_report(), pack)

    assert any("must authorize maintained_catalog" in error for error in errors)


def test_strict_policy_authorization_rejects_missing_report_policy() -> None:
    pack = _pack(_policy())
    report = _report()
    report["resolved_policy"] = None

    assert "Strict assurance requires a non-empty report.resolved_policy." in _errors(
        report, pack
    )


def test_strict_policy_authorization_rejects_missing_dataset_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = _pack(_policy())
    pack["compatibility"].pop("dataset_identity")
    monkeypatch.setattr(verify_policy, "verify_policy_pack", lambda _pack: [])

    assert any(
        "requires policy-pack compatibility.dataset_identity" in error
        for error in _errors(_report(), pack)
    )


def test_strict_policy_authorization_rejects_malformed_local_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = _pack(_policy())
    pack["compatibility"]["dataset_identity"] = {
        "provider": "",
        "dataset_name": None,
        "config_name": None,
        "revision": None,
        "split": "",
        "unexpected": "field",
    }
    monkeypatch.setattr(verify_policy, "verify_policy_pack", lambda _pack: [])

    errors = _errors(_report(), pack)

    assert any("must contain exactly" in error for error in errors)
    assert any("provider must be non-empty" in error for error in errors)
    assert any("split must be non-empty" in error for error in errors)
    assert any("for provider" in error for error in errors)
    assert any("for split" in error for error in errors)


def test_strict_policy_authorization_rejects_mutable_hosted_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()
    report["dataset"] = {
        "provider": "hf_text",
        "dataset_name": "dataset",
        "config_name": "config",
        "revision": "a" * 40,
        "split": "validation",
    }
    pack = _pack(_policy())
    pack["compatibility"]["dataset_identity"] = {
        "provider": "hf_text",
        "dataset_name": "",
        "config_name": "",
        "revision": "main",
        "split": "validation",
    }
    monkeypatch.setattr(verify_policy, "verify_policy_pack", lambda _pack: [])

    errors = _errors(report, pack)

    assert any("dataset_name must be non-empty" in error for error in errors)
    assert any("config_name must be non-empty" in error for error in errors)
    assert any("immutable lowercase hexadecimal" in error for error in errors)
    assert any("for revision" in error for error in errors)
