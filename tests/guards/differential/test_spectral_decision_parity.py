from __future__ import annotations

import pytest

from invarlock.guards.spectral import SpectralGuard
from tests.guards.property.strategies import spectral_family_decide


@pytest.mark.parametrize(
    ("z_by_name", "family_of_name", "multiple_testing"),
    [
        (
            {
                "attn.q": 3.2,
                "attn.k": 2.7,
                "mlp.up": 0.9,
                "embed.wte": 0.2,
            },
            {
                "attn.q": "attn",
                "attn.k": "attn",
                "mlp.up": "ffn",
                "embed.wte": "embed",
            },
            {"method": "bh", "alpha": 0.05},
        ),
        (
            {
                "attn.q": 1.2,
                "mlp.up": 4.5,
                "mlp.down": -4.1,
                "other.proj": 3.9,
            },
            {
                "attn.q": "attn",
                "mlp.up": "ffn",
                "mlp.down": "ffn",
                "other.proj": "other",
            },
            {"method": "bonferroni", "alpha": 0.10},
        ),
    ],
)
def test_spectral_family_selection_parity_production_vs_reference(
    z_by_name: dict[str, float],
    family_of_name: dict[str, str],
    multiple_testing: dict[str, object],
) -> None:
    guard = SpectralGuard(multiple_testing=dict(multiple_testing))
    guard.module_family_map = dict(family_of_name)
    budgeted_violations = [
        {
            "type": "family_z_cap",
            "severity": "budgeted",
            "module": name,
            "family": family_of_name[name],
            "z_score": z_score,
        }
        for name, z_score in z_by_name.items()
    ]

    selected, metrics = guard._select_budgeted_violations(budgeted_violations)
    reference = spectral_family_decide(z_by_name, family_of_name, multiple_testing)

    assert sorted(item["module"] for item in selected) == sorted(reference["selected"])
    assert metrics["families_selected"] == reference["families_selected"]
    assert metrics["method"] == reference["method"]
    assert metrics["family_violation_counts"] == reference["family_violation_counts"]
