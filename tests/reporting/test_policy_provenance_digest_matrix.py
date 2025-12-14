from __future__ import annotations

from copy import deepcopy

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.report_types import create_empty_report


def _base_report() -> dict:
    report = create_empty_report()
    report["meta"]["model_id"] = "m"
    report["meta"]["adapter"] = "hf_gpt2"
    report["meta"]["commit"] = "deadbeef"
    report["meta"]["seed"] = 1
    report["meta"]["auto"] = {"enabled": True, "tier": "balanced"}
    report["data"].update(
        {"dataset": "d", "split": "validation", "seq_len": 8, "stride": 8}
    )
    report["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 40.0,
        "final": 44.0,
        "ratio_vs_baseline": 1.10,
        "display_ci": [1.10, 1.10],
    }
    report["evaluation_windows"] = {"final": {"window_ids": [1], "logloss": [0.1]}}
    report["guards"] = [
        {
            "name": "spectral",
            "policy": {
                "deadband": 0.10,
                "max_caps": 5,
                "family_caps": {"ffn": {"kappa": 3.0}},
                "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
            },
            "metrics": {"caps_applied": 0, "caps_exceeded": False},
        },
        {
            "name": "rmt",
            "policy": {
                "deadband": 0.10,
                "margin": 1.5,
                "epsilon_default": 0.10,
                "epsilon_by_family": {"ffn": 0.10},
            },
            "metrics": {},
        },
        {
            "name": "variance",
            "policy": {
                "deadband": 0.02,
                "min_abs_adjust": 0.012,
                "max_scale_step": 0.03,
                "min_effect_lognll": 0.0009,
                "predictive_one_sided": True,
                "topk_backstop": 1,
                "max_adjusted_modules": 1,
            },
            "metrics": {},
        },
    ]
    return report


def _baseline() -> dict:
    return {
        "run_id": "b",
        "model_id": "m",
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }


def _digest(report: dict) -> str:
    cert = make_certificate(deepcopy(report), deepcopy(_baseline()))
    prov = cert.get("policy_provenance") or {}
    digest = prov.get("policy_digest")
    assert isinstance(digest, str) and digest
    return digest


def test_policy_provenance_digest_moves_for_each_guard_knob() -> None:
    base = _base_report()
    d0 = _digest(base)

    spectral = deepcopy(base)
    spectral["guards"][0]["policy"]["multiple_testing"]["alpha"] = 0.04
    assert _digest(spectral) != d0

    rmt = deepcopy(base)
    rmt["guards"][1]["policy"]["margin"] = 1.7
    assert _digest(rmt) != d0

    variance = deepcopy(base)
    variance["guards"][2]["policy"]["min_effect_lognll"] = 0.0011
    assert _digest(variance) != d0


def test_policy_provenance_digest_includes_override_list_order() -> None:
    base = _base_report()
    base["meta"]["policy_overrides"] = ["guards.spectral.deadband", "guards.rmt.margin"]
    d1 = _digest(base)

    swapped = _base_report()
    swapped["meta"]["policy_overrides"] = [
        "guards.rmt.margin",
        "guards.spectral.deadband",
    ]
    d2 = _digest(swapped)

    assert d1 != d2
