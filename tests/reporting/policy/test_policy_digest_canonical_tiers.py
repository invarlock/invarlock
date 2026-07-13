from __future__ import annotations

import pytest

from invarlock.reporting.report_enrichment import attach_policy_digest


def _payload(tier: str, _policy: dict[str, object]) -> dict[str, str]:
    return {"tier": tier}


def _digest(payload: dict[str, str]) -> str:
    return payload["tier"]


def test_policy_digest_uses_canonical_subject_and_baseline_tiers() -> None:
    output: dict[str, object] = {}

    attach_policy_digest(
        output,
        {"tier": "balanced"},
        {},
        {"meta": {"auto": {"tier": "conservative"}}},
        {},
        _payload,
        _digest,
        "policy-v1",
    )

    assert output["policy_digest"]["changed"] is True


@pytest.mark.parametrize(
    ("auto", "baseline", "expected"),
    [
        ({}, {"meta": {"auto": {"tier": "balanced"}}}, "subject"),
        ({"tier": "balanced"}, {"meta": {}}, "baseline"),
        (
            {"tier": "balanced"},
            {"context": {"auto": {"tier": "balanced"}}},
            "baseline",
        ),
    ],
)
def test_policy_digest_rejects_missing_or_alternate_tier_locations(
    auto: dict[str, object],
    baseline: dict[str, object],
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        attach_policy_digest(
            {},
            auto,
            {},
            baseline,
            {},
            _payload,
            _digest,
            "policy-v1",
        )
