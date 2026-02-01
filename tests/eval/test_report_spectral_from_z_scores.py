from invarlock.reporting.guards_analysis import _extract_spectral_analysis


def test_extract_spectral_analysis_from_z_scores_fallback():
    report = {
        "guards": [
            {
                "name": "spectral",
                "policy": {},
                "metrics": {
                    "final_z_scores": {"m1": 3.0, "m2": 1.5, "m3": 2.0},
                    "module_family_map": {"m1": "ffn", "m2": "ffn", "m3": "attn"},
                },
            }
        ]
    }
    baseline = {"metrics": {}}
    out = _extract_spectral_analysis(report, baseline)
    # Should contain family_z_quantiles and top_z_scores synthesized from z_scores
    q = out.get("family_z_quantiles", {})
    assert "ffn" in q and "attn" in q
    tz = out.get("top_z_scores", {})
    assert isinstance(tz, dict)
