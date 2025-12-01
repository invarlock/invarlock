from __future__ import annotations

from invarlock.reporting.guards_analysis import _extract_spectral_analysis


def test_spectral_policy_multiple_testing_merge_and_caps():
    report = {
        "meta": {"auto": {"tier": "balanced"}},
        "guards": [
            {
                "name": "spectral",
                "policy": {
                    "scope": "ffn",
                    "multiple_testing": {"method": "bh", "alpha": 0.05},
                },
                "metrics": {
                    "families": {"ffn": {"violations": 2}, "attn": {"violations": 0}},
                    "family_caps": {"ffn": {"kappa": 2.5}, "attn": {"kappa": 2.8}},
                    "sigma_ratios": [2.1, 1.9, 2.3],
                    "violations": [
                        {
                            "module": "m1",
                            "family": "ffn",
                            "kappa": 2.5,
                            "severity": "warn",
                            "z_score": 3.2,
                        },
                        {
                            "module": "m2",
                            "family": "ffn",
                            "kappa": 2.5,
                            "severity": "warn",
                            "z_score": "x",
                        },
                    ],
                },
            }
        ],
        "metrics": {
            "spectral": {"max_caps": 5, "sigma_quantile": 0.95, "deadband": 0.1}
        },
    }
    baseline = {"metrics": {}}
    out = _extract_spectral_analysis(report, baseline)
    # Expect policy block with multiple_testing merged and family caps present
    pol = out.get("policy", {})
    assert pol.get("multiple_testing", {}).get("method") == "bh"
    assert out.get("family_caps", {}).get("ffn", {}).get("kappa") == 2.5
    cap_map = out.get("caps_applied_by_family", {})
    assert isinstance(cap_map.get("ffn"), int)
