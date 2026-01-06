from __future__ import annotations

from invarlock.reporting.certificate import make_certificate


def _mk_pm_only(final_val: float) -> dict:
    return {
        "meta": {"model_id": "model", "adapter": "hf_gpt2", "seed": 42},
        "metrics": {
            # No ppl_* keys; only primary_metric present
            "primary_metric": {
                "kind": "ppl_causal",
                "unit": "ppl",
                "direction": "lower",
                "aggregation_scope": "token",
                "paired": True,
                "gating_basis": "upper",
                "final": float(final_val),
            }
        },
        "dataset": {
            "provider": "wikitext2",
            "seq_len": 128,
            "windows": {"preview": 0, "final": 0},
        },
        "artifacts": {},
    }


def test_ratio_uses_primary_metric_when_ppl_missing() -> None:
    subj = _mk_pm_only(50.0)
    base = _mk_pm_only(50.0)

    cert = make_certificate(subj, base)

    # ratio_vs_baseline should be computable and equal 1.0
    pm = cert.get("primary_metric", {})
    ratio = pm.get("ratio_vs_baseline")
    assert isinstance(ratio, int | float)
    assert abs(float(ratio) - 1.0) < 1e-9

    # ppl gate should pass with default balanced tier
    assert cert["validation"]["primary_metric_acceptable"] is True
    assert cert["validation"].get("primary_metric_acceptable", True) is True
