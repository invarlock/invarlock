from invarlock.reporting.guards_analysis import _extract_spectral_analysis


def test_extract_spectral_analysis_with_empty_tier_defaults(monkeypatch):
    # Empty TIER_POLICIES should not break spectral extraction
    monkeypatch.setattr(
        "invarlock.reporting.certificate.TIER_POLICIES", {}, raising=False
    )
    report = {
        "metrics": {"spectral": {}},
        "guards": [],
        "meta": {"model_id": "m"},
    }
    baseline = {"model_id": "m"}
    out = _extract_spectral_analysis(report, baseline)
    assert isinstance(out, dict) and out.get("caps_applied", 0) == 0
    assert "summary" in out
