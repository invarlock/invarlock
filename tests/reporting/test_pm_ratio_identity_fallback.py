from __future__ import annotations

from pathlib import Path

from invarlock.reporting.certificate import make_certificate


def _mk_run_with_windows(final_vals: list[float], token_counts: list[int]) -> dict:
    assert len(final_vals) == len(token_counts) and len(final_vals) > 0
    # Use identical preview/final for simplicity; only final is used for baseline pairing
    return {
        "meta": {"model_id": "model", "adapter": "hf_gpt2", "seed": 42},
        "metrics": {
            # Leave ppl_* absent to exercise pair-based recompute/identity
            "primary_metric": {
                "kind": "ppl_causal",
                "unit": "ppl",
                "direction": "lower",
                "aggregation_scope": "token",
                "paired": True,
                "gating_basis": "upper",
                # do not set ratio_vs_baseline here
                "final": 123.0,
            }
        },
        "evaluation_windows": {
            "final": {
                "window_ids": list(range(len(final_vals))),
                "logloss": [float(x) for x in final_vals],
                "token_counts": [int(w) for w in token_counts],
            }
        },
        "artifacts": {},
        "dataset": {
            "provider": "wikitext2",
            "seq_len": 128,
            "windows": {"preview": 1, "final": 1},
        },
    }


def test_pm_ratio_identity_fallback_when_windows_identical(tmp_path: Path) -> None:
    # Identical subject/baseline windows → expected ratio 1.0 even if baseline ppl is missing
    ll = [1.0, 2.0, 3.0]
    wc = [10, 20, 30]
    subj = _mk_run_with_windows(ll, wc)
    base = _mk_run_with_windows(ll, wc)

    cert = make_certificate(subj, base)
    pm = cert.get("primary_metric", {})
    ratio = pm.get("ratio_vs_baseline")
    assert isinstance(ratio, int | float)
    # Numerical identity
    assert abs(float(ratio) - 1.0) < 1e-6
