from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from invarlock.reporting import certificate as cert


def _basic_pm(final: float) -> dict[str, object]:
    return {"metrics": {"primary_metric": {"final": final}}}


def test_compute_edit_digest_quantization_and_fallback():
    quant_report = {
        "edit": {"name": "quant_rtn", "config": {"bitwidth": 4, "scope": "ffn"}}
    }
    digest = cert._compute_edit_digest(quant_report)
    assert digest["family"] == "quantization"
    assert digest["version"] == 1

    class BadEdits(dict):
        def get(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    fallback = cert._compute_edit_digest({"edit": BadEdits()})
    assert fallback["family"] == "cert_only"


def test_compute_confidence_label_respects_custom_thresholds():
    certificate = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {"kind": "accuracy", "display_ci": (0.50, 0.55)},
        "resolved_policy": {"confidence": {"accuracy_delta_pp_width_max": 0.1}},
    }
    label = cert._compute_confidence_label(certificate)
    assert label["label"] == "High"
    certificate["primary_metric"]["unstable"] = True
    label_unstable = cert._compute_confidence_label(certificate)
    assert label_unstable["label"] == "Medium"


def test_compute_report_digest_changes_with_inputs():
    base = {
        "meta": {
            "model_id": "invarlock",
            "adapter": "hf",
            "commit": "abc",
            "ts": "now",
        },
        "edit": {"name": "noop", "plan_digest": "deadbeef"},
        "metrics": {"spectral": {"caps_applied": 1}, "rmt": {"outliers": 0}},
    }
    digest1 = cert._compute_report_digest(base)
    digest2 = cert._compute_report_digest({**base, "edit": {"name": "other"}})
    assert digest1 != digest2


def test_prepare_guard_overhead_section_with_reports():
    bare = _basic_pm(10.0)
    guarded = _basic_pm(10.05)
    payload, passed = cert._prepare_guard_overhead_section(
        {"bare_report": bare, "guarded_report": guarded, "source": "unit"}
    )
    assert passed is True
    assert payload["evaluated"] is True
    assert "overhead_ratio" in payload


def test_prepare_guard_overhead_section_ratio_fallback():
    payload, passed = cert._prepare_guard_overhead_section(
        {"bare_ppl": 10.0, "guarded_ppl": 11.0, "messages": ["note"]}
    )
    assert passed is False
    assert payload["guarded_ppl"] == 11.0
    assert payload["messages"] == ["note"]


def test_compute_quality_overhead_from_guard_handles_ratio(monkeypatch):
    bare = {"value": 10.0}
    guarded = {"value": 11.0}

    def fake_compute(report, *, kind, baseline=None):
        return {"final": report["value"], "direction": "lower"}

    monkeypatch.setattr(cert, "compute_primary_metric_from_report", fake_compute)
    monkeypatch.setattr(
        cert, "get_metric", lambda *_: SimpleNamespace(direction="lower")
    )
    info = cert._compute_quality_overhead_from_guard(
        {"bare_report": bare, "guarded_report": guarded}, pm_kind_hint="ppl_causal"
    )
    assert info == {"basis": "ratio", "value": pytest.approx(1.1), "kind": "ppl_causal"}


def test_compute_quality_overhead_from_guard_accuracy_delta(monkeypatch):
    bare = {"value": 0.9}
    guarded = {"value": 0.95}

    def fake_compute(report, *, kind, baseline=None):
        return {"final": report["value"], "direction": "higher"}

    monkeypatch.setattr(cert, "compute_primary_metric_from_report", fake_compute)
    monkeypatch.setattr(
        cert, "get_metric", lambda *_: SimpleNamespace(direction="higher")
    )
    info = cert._compute_quality_overhead_from_guard(
        {"bare_report": bare, "guarded_report": guarded}, pm_kind_hint="accuracy"
    )
    assert info["basis"] == "delta_pp"
    assert math.isclose(info["value"], 5.0)


def test_generate_run_id_uses_existing_and_hashes_otherwise():
    existing = cert._generate_run_id({"meta": {"run_id": "abc"}})
    assert existing == "abc"
    derived = cert._generate_run_id({"meta": {"model_id": "m", "ts": "now"}})
    assert len(derived) == 16 and derived != "abc"


def test_analyze_bitwidth_map_and_rank_information():
    bitwidth_map = {
        "layer1": {"bitwidth": 4, "params": 128},
        "layer2": {"bitwidth": 8, "params": 256},
    }
    stats = cert._analyze_bitwidth_map(bitwidth_map)
    assert stats["total_modules"] == 2
    assert stats["min_bitwidth"] == 4
    deltas = {
        "rank_map": {
            "layer1": {
                "rank": 8,
                "params_saved": 100,
                "energy_retained": 0.9,
                "deploy_mode": "recompose",
                "savings_mode": "realized",
                "realized_params_saved": 100,
                "theoretical_params_saved": 120,
                "realized_params": 400,
                "theoretical_params": 420,
            }
        },
        "savings": {"deploy_mode": "recompose"},
    }
    edit_cfg = {"frac": 0.5, "rank_policy": "auto"}
    rank_info = cert._extract_rank_information(edit_cfg, deltas)
    assert rank_info["rank_policy"] == "auto"
    savings = cert._compute_savings_summary(deltas)
    assert savings["mode"] == "realized"


def test_extract_compression_diagnostics_quant(monkeypatch):
    inference = {
        "flags": dict.fromkeys(("scope", "seed", "rank_policy", "frac"), False),
        "sources": {},
        "log": [],
    }
    deltas = {
        "params_changed": 5,
        "bitwidth_map": {
            "layer1": {"bitwidth": 4, "group_size": None, "params": 256},
        },
        "rank_map": {
            "layer1": {
                "rank": 8,
                "params_saved": 128,
                "energy_retained": 0.95,
                "deploy_mode": "recompose",
                "savings_mode": "realized",
                "realized_params_saved": 64,
                "theoretical_params_saved": 80,
                "realized_params": 900,
                "theoretical_params": 920,
                "skipped": False,
            }
        },
    }
    diagnostics = cert._extract_compression_diagnostics(
        "quant_rtn_rank",
        {"scope": "unknown", "clamp_ratio": 0.2},
        deltas,
        {"layers": 2},
        inference,
    )
    assert diagnostics["execution_status"] == "successful"
    assert diagnostics["target_analysis"]["modules_modified"] >= 1
    assert "algorithm_details" in diagnostics
    assert inference["flags"]["scope"] is True


def test_prepare_guard_overhead_section_empty_returns_pass():
    payload, passed = cert._prepare_guard_overhead_section({})
    assert payload == {}
    assert passed is True


def test_compute_validation_flags_respects_token_floors(monkeypatch):
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)
    flags = cert._compute_validation_flags(
        ppl={
            "preview_final_ratio": 1.02,
            "ratio_vs_baseline": 1.2,
            "ratio_ci": (1.19, 1.25),
        },
        spectral={"caps_applied": 6},
        rmt={"stable": False},
        invariants={"status": "pass"},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 1000, "final_total_tokens": 1000},
        target_ratio=1.05,
        guard_overhead={"passed": False, "evaluated": True},
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.2},
        dataset_capacity={"tokens_available": 2000},
    )
    assert flags["primary_metric_acceptable"] is False
    assert flags["spectral_stable"] is False
    assert flags["guard_overhead_acceptable"] is False
    assert flags["rmt_stable"] is False


def test_compute_validation_flags_accuracy_branch_sets_hysteresis(monkeypatch):
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)
    pm = {"kind": "accuracy", "ratio_vs_baseline": -0.5, "n_final": 50}
    flags = cert._compute_validation_flags(
        ppl={"preview_final_ratio": 1.0, "ratio_vs_baseline": 0.98},
        spectral={"caps_applied": 1},
        rmt={"stable": True},
        invariants={"status": "pass"},
        tier="balanced",
        primary_metric=pm,
        moe={"top_k": 2},
        dataset_capacity={"examples_available": 5000},
    )
    assert flags["primary_metric_acceptable"] is False
    assert flags["moe_observed"] is True
