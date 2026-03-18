from __future__ import annotations

import json

import pytest

from invarlock.fuzzing import exercise_policy_pack_bytes


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"{",
        b"tier: balanced\nresolved_policy:\n  metrics: []\n",
        bytes(range(256)),
        json.dumps(
            {
                "format": "policy-pack-v1",
                "tier": "balanced",
                "resolved_policy": {"metrics": {"pm_ratio": {"ratio_limit_base": 1.1}}},
                "overrides": [],
                "policy_digest": "placeholder",
                "compatibility": {"support_tiers": ["published_basis"]},
            }
        ).encode("utf-8"),
    ],
)
def test_policy_pack_target_handles_arbitrary_bytes(payload: bytes) -> None:
    exercise_policy_pack_bytes(payload)
