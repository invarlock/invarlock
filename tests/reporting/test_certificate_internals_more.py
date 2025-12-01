from __future__ import annotations

import math

from invarlock.reporting import certificate as C
from invarlock.reporting.policy_utils import _build_resolved_policies
from invarlock.reporting.utils import _coerce_int, _sanitize_seed_bundle


def test_normalize_baseline_variants():
    # baseline-v1 fields
    b1 = {
        "schema_version": "baseline-v1",
        "meta": {"commit_sha": "cafebabedeadbeef", "model_id": "m"},
        "metrics": {"ppl_final": 50.0},
        "spectral_base": {"caps": 0},
        "rmt_base": {"outliers": 0},
        "invariants": {"status": "pass"},
    }
    out = C._normalize_baseline(b1)
    assert out.get("model_id") == "m"
    assert "ppl_final" in out

    # RunReport-like baseline without ppl_*, derive from PM
    base_rr = {
        "meta": {"model_id": "m"},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 49.0}},
        "edit": {"plan": {}, "plan_digest": "noop"},
    }
    out2 = C._normalize_baseline(base_rr)
    assert math.isfinite(out2.get("ppl_final", float("nan")))


def test_coerce_and_sanitize_seed_bundle():
    assert _coerce_int(1.0) == 1
    assert _coerce_int(1.5) is None
    assert _coerce_int("3") == 3
    sb = _sanitize_seed_bundle({"python": 7, "numpy": None, "torch": 9}, fallback=0)
    assert sb == {"python": 7, "numpy": None, "torch": 9}


def test_format_helpers_and_resolved_policies():
    caps = {"attn": {"kappa": 0.2}, "ffn": 0.1}
    eps_map = {"ffn": 0.1, "attn": 0.2}
    spectral = {
        "family_caps": caps,
        "multiple_testing": {"method": "bonf", "alpha": 0.05, "m": 2},
        "policy": {"correction_enabled": True},
    }
    rmt = {
        "epsilon_by_family": eps_map,
        "margin": 1.6,
        "deadband": 0.05,
        "epsilon_default": 0.1,
    }
    variance = {"predictive_gate": {"sided": "one_sided"}, "min_effect_lognll": 0.0}
    res = _build_resolved_policies("balanced", spectral, rmt, variance)
    assert res["spectral"]["family_caps"]["attn"]["kappa"] == 0.2
    assert "epsilon_by_family" in res["rmt"]
    assert isinstance(res.get("confidence", {}), dict)


def test_compute_report_and_quality_overhead():
    # Small bare/guarded PM reports
    bare = {
        "evaluation_windows": {
            "final": {"logloss": [4.0, 4.0], "token_counts": [100, 100]}
        }
    }
    guarded = {
        "evaluation_windows": {
            "final": {"logloss": [4.0, 4.1], "token_counts": [100, 100]}
        }
    }
    out = C._compute_quality_overhead_from_guard(
        {"bare_report": bare, "guarded_report": guarded}, pm_kind_hint="ppl_causal"
    )
    assert out and out["basis"] == "ratio" and out["value"] > 1.0

    # compute_report_digest minimal
    digest = C._compute_report_digest(
        {
            "meta": {"model_id": "m", "adapter": "hf", "commit": "abc", "ts": "t"},
            "edit": {"name": "noop", "plan_digest": "d"},
            "metrics": {},
        }
    )
    assert isinstance(digest, str) and len(digest) >= 8
