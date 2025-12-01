from unittest.mock import patch

import pytest

from invarlock.core.auto_tuning import TIER_POLICIES
from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.guards_analysis import _extract_spectral_analysis


def _minimal_report():
    return {
        "meta": {"model_id": "m", "seed": 1, "auto": {"tier": "balanced"}},
        "metrics": {"ppl_preview": 10.0, "ppl_final": 10.0},
        "data": {
            "dataset": "d",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
        },
        "guards": [
            {"name": "spectral", "policy": {}, "metrics": {}},
        ],
        "edit": {
            "name": "structured",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
                "sparsity": None,
            },
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }


@pytest.mark.parametrize("tier", ["balanced", "conservative", "aggressive"])
def test_spectral_summary_sigma_quantile_from_tier_defaults(tier: str):
    report = _minimal_report()
    report["meta"]["auto"]["tier"] = tier
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "ppl_final": 10.0,
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    # Ensure report passes structure validation
    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        cert = make_certificate(report, baseline)

    spectral = cert.get("spectral", {})
    # Sanity: spectral section present
    assert spectral, "spectral analysis missing from certificate"

    # summary.sigma_quantile should be present and reflect tier defaults
    summary = spectral.get("summary", {})
    expected = (
        TIER_POLICIES[tier]["spectral"].get("sigma_quantile")
        or TIER_POLICIES[tier]["spectral"].get("contraction")
        or TIER_POLICIES[tier]["spectral"].get("kappa")
    )
    assert "sigma_quantile" in summary and pytest.approx(
        summary["sigma_quantile"], rel=0, abs=1e-12
    ) == float(expected)


def test_extract_spectral_analysis_summary_sigma_quantile_present():
    # Directly test helper for branch precision
    report = _minimal_report()
    baseline = {}
    out = _extract_spectral_analysis(report, baseline)
    summary = out.get("summary", {})
    assert "sigma_quantile" in summary
