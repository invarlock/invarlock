from __future__ import annotations

import copy

import pytest

from invarlock.core.auto_tuning import resolve_tier_policies
from invarlock.reporting.guards_spectral import _build_policy_output
from invarlock.reporting.policy_utils import _build_resolved_policies


@pytest.mark.parametrize("tier", ["conservative", "balanced", "aggressive"])
@pytest.mark.parametrize("guard_name", ["spectral", "rmt", "variance", "metrics"])
def test_report_resolved_policy_matches_runtime_resolver_for_every_guard_field(
    tier: str,
    guard_name: str,
) -> None:
    runtime_policy = resolve_tier_policies(tier)

    reported_policy = _build_resolved_policies(tier, {}, {}, {})

    assert reported_policy[guard_name] == runtime_policy[guard_name]


def test_balanced_spectral_output_preserves_guard_policy_semantics() -> None:
    guard_policy = copy.deepcopy(resolve_tier_policies("balanced")["spectral"])

    reported_policy = _build_policy_output(
        guard_policy,
        default_sigma_quantile=guard_policy["sigma_quantile"],
        multiple_testing=guard_policy["multiple_testing"],
        tier="balanced",
    )

    assert reported_policy == guard_policy
    assert reported_policy["correction_enabled"] is True
