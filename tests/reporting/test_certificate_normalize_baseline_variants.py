from __future__ import annotations

from invarlock.reporting import certificate as C


def test_normalize_baseline_v1_schema() -> None:
    base_v1 = {
        "schema_version": "baseline-v1",
        "meta": {"model_id": "m", "commit_sha": "1234567890abcdef"},
        "metrics": {"ppl_final": 10.0},
        "spectral_base": {"caps": 0},
        "rmt_base": {"epsilon": {}},
    }
    norm = C._normalize_baseline(base_v1)
    assert norm.get("ppl_final") == 10.0 and norm.get("model_id") == "m"


def test_normalize_baseline_runreport_derives_ppl_from_pm() -> None:
    base = {
        "meta": {"model_id": "m"},
        "edit": {
            "name": "baseline",
            "plan_digest": "baseline_noop",
            "deltas": {"params_changed": 0},
        },
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": 12.0, "preview": 12.0}
        },
    }
    norm = C._normalize_baseline(base)
    assert norm.get("ppl_final") == 12.0
