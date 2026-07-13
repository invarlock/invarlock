"""Dataset-provider policy for training receipt production and replay.

The provider binding is a policy input.  A receipt may prove that it copied a
provider identity consistently, but an independent verifier must obtain that
identity from an immutable profile or a sealed campaign policy instead.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from hashlib import sha256

from .training_contract import TrainingProfile
from .training_runtime_errors import TrainingRuntimeError


def validate_dataset_provider_binding(
    payload: object, *, label: str
) -> dict[str, object]:
    """Return one canonical, self-bound provider identity."""

    if not isinstance(payload, Mapping) or set(payload) != {
        "provider",
        "provider_sha256",
    }:
        raise TrainingRuntimeError(f"{label} is malformed")
    provider = payload.get("provider")
    if not isinstance(provider, Mapping) or not provider:
        raise TrainingRuntimeError(f"{label} coordinates are missing")
    normalized_provider = dict(provider)
    try:
        rendered = json.dumps(
            normalized_provider,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TrainingRuntimeError(
            f"{label} coordinates are not canonical JSON"
        ) from exc
    expected = "sha256:" + sha256(rendered).hexdigest()
    if payload.get("provider_sha256") != expected:
        raise TrainingRuntimeError(f"{label} digest mismatch")
    return {"provider": normalized_provider, "provider_sha256": expected}


def profile_dataset_provider_binding(profile: TrainingProfile) -> dict[str, object]:
    """Derive the safe local-provider policy from the immutable profile."""

    provider = {
        "kind": "training_jsonl",
        "path": profile.training_data.path,
        "sha256": profile.training_data.sha256,
    }
    rendered = json.dumps(
        provider,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return validate_dataset_provider_binding(
        {
            "provider": provider,
            "provider_sha256": "sha256:" + sha256(rendered).hexdigest(),
        },
        label="immutable profile dataset provider policy",
    )


def dataset_provider_binding(
    profile: TrainingProfile,
    *,
    dataset_provider_policy: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Resolve the provider used to create a new training receipt.

    The environment override remains a producer input for existing campaign
    runners.  Artifact verification never trusts it; use
    :func:`expected_dataset_provider_binding` there instead.
    """

    if dataset_provider_policy is not None:
        return validate_dataset_provider_binding(
            dataset_provider_policy, label="dataset provider policy"
        )

    raw = os.environ.get("INVARLOCK_ACCEPTANCE_DATASET_PROVIDER_SNAPSHOT_JSON")
    if raw:
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise TrainingRuntimeError(
                "acceptance dataset provider snapshot is invalid JSON"
            ) from exc
        return validate_dataset_provider_binding(
            payload, label="acceptance dataset provider binding"
        )
    return profile_dataset_provider_binding(profile)


def expected_dataset_provider_binding(
    profile: TrainingProfile,
    *,
    dataset_provider_policy: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Resolve the independently trusted provider policy for verification.

    Without a sealed campaign policy, only the immutable profile's local JSONL
    provider is acceptable.  This deliberately ignores the producer-only
    environment override so a self-consistent replacement receipt cannot set
    its own expected provider.
    """

    if dataset_provider_policy is not None:
        return validate_dataset_provider_binding(
            dataset_provider_policy, label="dataset provider policy"
        )
    return profile_dataset_provider_binding(profile)


__all__ = [
    "dataset_provider_binding",
    "expected_dataset_provider_binding",
    "profile_dataset_provider_binding",
    "validate_dataset_provider_binding",
]
