from __future__ import annotations

from typing import Any

from invarlock.core.assurance_contract import (
    ASSURANCE_CLAIM_SET_V2,
    LEGACY_ASSURANCE_CLAIM_SET,
)
from invarlock.core.dataset_identity import (
    DATASET_IDENTITY_FIELDS,
    canonical_dataset_revision,
    dataset_identity_from_report,
    is_hosted_dataset_provider,
)
from invarlock.policy_pack import (
    LEGACY_POLICY_PACK_FORMAT,
    verify_policy_pack,
)

STRICT_AUTHORIZED_TIERS = frozenset({"balanced", "conservative"})


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def append_strict_policy_authorization_errors(
    errors: list[str],
    *,
    report: dict[str, Any],
    policy_pack: dict[str, Any] | None,
) -> None:
    """Bind strict acceptance thresholds to a independently supplied policy pack."""

    if policy_pack is None:
        errors.append(
            "Strict assurance requires a independently supplied --policy-pack; "
            "the report cannot authorize its own acceptance thresholds."
        )
        return

    pack_errors = verify_policy_pack(policy_pack)
    if pack_errors:
        submitted_format = policy_pack.get("format")
        format_label = (
            submitted_format
            if isinstance(submitted_format, str) and submitted_format.strip()
            else "policy pack"
        )
        errors.extend(f"Invalid {format_label}: {error}" for error in pack_errors)
        return

    pack_tier = str(policy_pack.get("tier") or "").strip().lower()
    if pack_tier not in STRICT_AUTHORIZED_TIERS:
        errors.append(
            "Strict assurance policy tier must be balanced or conservative "
            f"(found {pack_tier!r})."
        )

    report_policy = report.get("resolved_policy")
    authorized_policy = policy_pack.get("resolved_policy")
    if not isinstance(report_policy, dict) or not report_policy:
        errors.append("Strict assurance requires a non-empty report.resolved_policy.")
    elif report_policy != authorized_policy:
        errors.append(
            "Strict assurance resolved_policy does not exactly match the "
            "independently supplied policy pack."
        )

    assurance_tier = (
        str(_mapping(report.get("assurance")).get("tier") or "").strip().lower()
    )
    assurance_claim = _mapping(report.get("assurance")).get("claim_set")
    expected_claim = (
        LEGACY_ASSURANCE_CLAIM_SET
        if policy_pack.get("format") == LEGACY_POLICY_PACK_FORMAT
        else ASSURANCE_CLAIM_SET_V2
    )
    if assurance_claim != expected_claim:
        errors.append(
            "Strict assurance claim_set does not match the policy-pack format "
            f"(report={assurance_claim!r}, expected={expected_claim!r})."
        )
    if assurance_tier != pack_tier:
        errors.append(
            "Strict assurance tier mismatch between report.assurance and policy pack "
            f"(report={assurance_tier!r}, policy_pack={pack_tier!r})."
        )

    auto_tier = str(_mapping(report.get("auto")).get("tier") or "").strip().lower()
    if auto_tier != pack_tier:
        errors.append(
            "Strict assurance tier mismatch between report.auto and policy pack "
            f"(report={auto_tier!r}, policy_pack={pack_tier!r})."
        )

    compatibility = _mapping(policy_pack.get("compatibility"))
    support_tiers = compatibility.get("support_tiers")
    required_support_tier = (
        "published_basis"
        if policy_pack.get("format") == LEGACY_POLICY_PACK_FORMAT
        else "maintained_catalog"
    )
    if (
        not isinstance(support_tiers, list)
        or required_support_tier not in support_tiers
    ):
        errors.append(
            "Strict assurance policy-pack compatibility.support_tiers must "
            f"authorize {required_support_tier}."
        )
    expected_identity = compatibility.get("dataset_identity")
    if not isinstance(expected_identity, dict):
        errors.append(
            "Strict assurance requires policy-pack compatibility.dataset_identity."
        )
        return
    if set(expected_identity) != set(DATASET_IDENTITY_FIELDS):
        errors.append(
            "Strict assurance policy-pack dataset_identity must contain exactly "
            f"{', '.join(DATASET_IDENTITY_FIELDS)}."
        )
    expected_provider = expected_identity.get("provider")
    expected_split = expected_identity.get("split")
    if not isinstance(expected_provider, str) or not expected_provider:
        errors.append(
            "Strict assurance policy-pack dataset_identity.provider must be non-empty."
        )
    if not isinstance(expected_split, str) or not expected_split:
        errors.append(
            "Strict assurance policy-pack dataset_identity.split must be non-empty."
        )
    if is_hosted_dataset_provider(expected_provider):
        if (
            not isinstance(expected_identity.get("dataset_name"), str)
            or not (expected_identity["dataset_name"])
        ):
            errors.append(
                "Strict hosted policy-pack dataset_identity.dataset_name must be "
                "non-empty."
            )
        if (
            not isinstance(expected_identity.get("config_name"), str)
            or not (expected_identity["config_name"])
        ):
            errors.append(
                "Strict hosted policy-pack dataset_identity.config_name must be "
                "non-empty."
            )
        if canonical_dataset_revision(expected_identity.get("revision")) is None:
            errors.append(
                "Strict hosted policy-pack dataset_identity.revision must be an "
                "immutable lowercase hexadecimal revision."
            )

    observed_identity = dataset_identity_from_report(report)
    for field in DATASET_IDENTITY_FIELDS:
        if expected_identity.get(field) != observed_identity.get(field):
            errors.append(
                "Strict assurance report dataset identity does not match the "
                f"acceptance policy pack for {field}."
            )


__all__ = [
    "STRICT_AUTHORIZED_TIERS",
    "append_strict_policy_authorization_errors",
]
