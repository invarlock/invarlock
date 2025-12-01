from invarlock.reporting.guards_analysis import _extract_spectral_analysis


def test_extract_spectral_analysis_defaults_without_guard():
    report = {"guards": [], "meta": {}}
    baseline = {}
    out = _extract_spectral_analysis(report, baseline)
    assert out["caps_applied"] == 0
    assert out["summary"]["status"] in {"stable", "capped"}
    assert "family_caps" in out
