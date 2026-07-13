from copy import deepcopy
from unittest.mock import patch

import pytest

from invarlock.core.auto_tuning import TIER_POLICIES
from invarlock.reporting.guards_spectral import _extract_spectral_analysis
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def _minimal_report():
    return {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "seed": 1,
            "auto": {"tier": "balanced"},
        },
        "context": {"profile": "dev"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
            }
        },
        "data": {
            "dataset": "d",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
        },
        "guards": [
            {"name": "spectral", "passed": True, "policy": {}, "metrics": {}},
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
    baseline = deepcopy(report)
    baseline["run_id"] = "b"
    baseline["edit"]["name"] = "noop"
    baseline["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 10.0,
        "final": 10.0,
    }
    # Ensure report passes structure validation
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        cert = make_report(report, baseline)

    spectral = cert.get("spectral", {})
    # Sanity: spectral section present
    assert spectral, "spectral analysis missing from evaluation_report"

    # summary.sigma_quantile should be present and reflect tier defaults
    summary = spectral.get("summary", {})
    expected = TIER_POLICIES[tier]["spectral"].get("sigma_quantile", 0.95)
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
