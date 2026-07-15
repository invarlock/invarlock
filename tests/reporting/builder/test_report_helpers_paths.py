from __future__ import annotations

import json
import math

import pytest

from invarlock.reporting import report_make as report_make_mod
from invarlock.reporting import report_normalization as normalization_mod
from invarlock.reporting.report_builder_support import (
    attach_schedule_digest,
    build_moe_section,
    evaluate_primary_metric_tail,
    extract_telemetry,
    resolve_capacity_context,
    validate_retry_evaluation_report,
)
from invarlock.reporting.report_builder_support import (
    extract_report_meta as _extract_report_meta,
)
from invarlock.reporting.report_primary_metric_policy import is_ppl_kind as _is_ppl_kind
from invarlock.reporting.report_provenance import (
    compute_edit_digest as _compute_edit_digest,
)
from invarlock.reporting.utils import (
    _coerce_int,
    _coerce_interval,
    _infer_scope_from_modules,
    _sanitize_seed_bundle,
)


class _RaisingStr:
    def __str__(self) -> str:  # pragma: no cover - used to trigger exception in target
        raise RuntimeError("boom")


class _RaisingGet:
    def get(
        self, *args, **kwargs
    ):  # pragma: no cover - used to trigger exception in target
        raise RuntimeError("boom")


class _ValueErrorGet(dict):
    def get(self, *args, **kwargs):  # type: ignore[override]
        raise ValueError("bad get")


def test_is_ppl_kind_handles_str_exception() -> None:
    assert _is_ppl_kind(_RaisingStr()) is False
    assert _is_ppl_kind("ppl_causal") is True


def test_coerce_int_variants() -> None:
    assert _coerce_int(5) == 5
    # Non-integer float rejected (only near-integers accepted)
    assert _coerce_int(5.8) is None
    assert _coerce_int("7") == 7
    assert _coerce_int(None) is None
    assert _coerce_int("bad") is None


def test_sanitize_seed_bundle_partial_and_fallback() -> None:
    sanitized = _sanitize_seed_bundle({"python": 1, "numpy": None}, fallback=42)
    # Explicit/missing None entries preserve None; others use fallback
    assert (
        sanitized["python"] == 1
        and sanitized["numpy"] is None
        and sanitized["torch"] is None
    )


def test_infer_scope_from_modules_variants() -> None:
    assert _infer_scope_from_modules([]) == "unknown"
    assert _infer_scope_from_modules(["model.attn.block"]) == "attn"
    assert _infer_scope_from_modules(["decoder.mlp.fc"]) == "ffn"
    assert _infer_scope_from_modules(["wte.embedding"]) == "embed"
    mixed = _infer_scope_from_modules(["layer.attention", "mlp.ffn", "tok.embed"])
    assert mixed in {"attn+embed+ffn", "attn+ffn+embed"}


def test_coerce_interval_from_string_and_list() -> None:
    lo, hi = _coerce_interval("(1.5, 2.5)")
    assert math.isclose(lo, 1.5) and math.isclose(hi, 2.5)
    lo2, hi2 = _coerce_interval("not a tuple")
    assert math.isnan(lo2) and math.isnan(hi2)
    lo3, hi3 = _coerce_interval(["x", 2])
    assert math.isnan(lo3) and math.isnan(hi3)


def test_compute_edit_digest_quant_and_default() -> None:
    d = _compute_edit_digest({"edit": {"name": "quant_rtn", "config": {"bitwidth": 8}}})
    assert d["family"] == "quantization" and isinstance(d["impl_hash"], str)
    d2 = _compute_edit_digest({"edit": {"name": "noop"}})
    assert d2["family"] == "report_only"


def test_extract_report_meta_prefers_python_seed() -> None:
    report = {
        "meta": {
            "model_id": "demo",
            "adapter": "hf",
            "device": "cpu",
            "seed": 7,
            "seeds": {"python": 9, "numpy": None},
        }
    }
    meta = _extract_report_meta(report)
    assert meta["seed"] == 9
    assert meta["seeds"]["python"] == 9


def test_extract_report_meta_defaults_seed_to_zero() -> None:
    report = {"meta": {"model_id": "demo", "adapter": "hf", "device": "cpu"}}
    meta = _extract_report_meta(report)
    assert meta["seed"] == 0


def test_extract_report_meta_records_missing_fields() -> None:
    diagnostics: list[dict[str, object]] = []
    report = {"meta": {"adapter": "", "device": None}}
    meta = _extract_report_meta(report, diagnostics)

    assert meta["model_id"] is None
    assert meta["adapter"] is None
    assert meta["device"] is None
    codes = {entry["code"] for entry in diagnostics}
    assert "meta.model_id_unavailable" in codes
    assert "meta.adapter_unavailable" in codes
    assert "meta.device_unavailable" in codes


def test_normalize_and_validate_report_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        normalization_mod.normalize_and_validate_run_report("oops")


def test_report_builder_support_moe_capacity_and_tail_edge_paths() -> None:
    report = {
        "metrics": {
            "primary_metric": {"kind": "ppl_causal"},
            "moe": {
                "load_balance_loss": 0.7,
                "router_entropy": 0.4,
                "utilization": [1, "bad"],
            },
        },
        "evaluation_windows": {
            "final": {
                "window_ids": [1, "bad", 3],
                "logloss": [2.0, 9.0, float("inf")],
                "token_counts": [-5, "bad", 7],
            }
        },
    }
    baseline = {
        "metrics": {
            "moe": {"load_balance_loss": 0.5, "router_entropy": 0.3},
        },
        "evaluation_windows": {"final": {"window_ids": [1, 2], "logloss": [1.5, 2.5]}},
    }

    moe = build_moe_section(report, {}, baseline)
    assert moe["delta_load_balance_loss"] == pytest.approx(0.2)
    assert "utilization_mean" not in moe

    assert resolve_capacity_context(
        {}, {"windows": {"preview": "2", "final": "3"}}
    ) == (
        None,
        5,
    )
    assert resolve_capacity_context(
        {"total_tokens": 10, "candidate_limit": 4}, {"windows": {}}
    ) == (10, 4)

    captured: dict[str, object] = {}

    def _evaluate_tail(**kwargs):
        captured.update(kwargs)
        return {"mode": "strict", "evaluated": True, "passed": True}

    tail = evaluate_primary_metric_tail(
        report,
        baseline,
        {"metrics": {"pm_tail": {"max_delta": 0.1}}},
        _evaluate_tail,
    )

    assert tail["source"] == "paired_baseline.final"
    assert captured["deltas"] == [0.5]
    assert captured["weights"] == [0.0]
    assert captured["policy"] == {"max_delta": 0.1}


def test_report_builder_support_defensive_helper_edges() -> None:
    assert extract_telemetry({"metrics": []}, "cpu") == {"device": "cpu"}

    overhead_section: dict[str, object] = {}
    assert attach_schedule_digest(_ValueErrorGet(), overhead_section) is None
    assert overhead_section == {}

    assert build_moe_section(_ValueErrorGet(), {}, {}) == {}

    moe = build_moe_section(
        {
            "metrics": {
                "moe": {
                    "load_balance_loss": 0.7,
                    "router_entropy": 0.4,
                    "utilization": [0.5, 0.75],
                }
            }
        },
        _ValueErrorGet(),
        {"metrics": {"moe": {"utilization": [0.25, object()]}}},
    )
    assert moe["utilization_count"] == 2
    assert "delta_utilization_mean" not in moe

    assert resolve_capacity_context(_ValueErrorGet(), {"windows": {}}) == (None, None)


def test_report_builder_support_moe_missing_baseline_paths() -> None:
    report = {
        "metrics": {
            "moe": {
                "load_balance_loss": 0.7,
                "router_entropy": 0.4,
            }
        }
    }

    no_baseline = build_moe_section(report, [], {})
    assert no_baseline["load_balance_loss"] == pytest.approx(0.7)
    assert "delta_load_balance_loss" not in no_baseline

    raw_baseline = build_moe_section(
        report,
        {"moe": {"load_balance_loss": 0.5, "router_entropy": 0.3}},
        {},
    )
    assert raw_baseline["delta_load_balance_loss"] == pytest.approx(0.2)
    assert raw_baseline["delta_router_entropy"] == pytest.approx(0.1)

    fallback_error = build_moe_section(report, {}, _ValueErrorGet())
    assert fallback_error["load_balance_loss"] == pytest.approx(0.7)
    assert "delta_load_balance_loss" not in fallback_error


def test_report_builder_support_primary_metric_tail_pairing_edges() -> None:
    captured: dict[str, object] = {}

    def _evaluate_tail(**kwargs):
        captured.update(kwargs)
        return {"mode": "strict", "evaluated": True, "passed": True}

    report = {
        "metrics": {"primary_metric": {"kind": "ppl_causal"}},
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2, "bad", 4],
                "logloss": [2.0, float("inf"), 9.0, 4.0],
                "token_counts": ["bad", 9, 3, 4],
            }
        },
    }
    baseline = {
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2, "bad-base", 4],
                "logloss": [1.5, 1.0, 7.0, 3.5],
            }
        }
    }

    tail = evaluate_primary_metric_tail(
        report,
        baseline,
        _ValueErrorGet(),
        _evaluate_tail,
    )

    assert tail["source"] == "paired_baseline.final"
    assert captured["deltas"] == [0.5, 0.5]
    assert captured["weights"] == [0.0, 4.0]
    assert captured["policy"] == {}


def test_report_builder_support_primary_metric_tail_lookup_errors() -> None:
    captured: dict[str, object] = {}

    def _evaluate_tail(**kwargs):
        captured.update(kwargs)
        return {"mode": "strict", "evaluated": True, "passed": True}

    tail = evaluate_primary_metric_tail(
        _ValueErrorGet(),
        {},
        {"metrics": {"pm_tail": {"max_delta": 0.1}}},
        _evaluate_tail,
    )

    assert tail["source"] == "paired_baseline.final"
    assert captured["deltas"] == []
    assert captured["weights"] is None


def test_report_builder_support_primary_metric_tail_non_ppl_kind() -> None:
    captured: dict[str, object] = {}

    def _evaluate_tail(**kwargs):
        captured.update(kwargs)
        return {"mode": "strict", "evaluated": True, "passed": True}

    tail = evaluate_primary_metric_tail(
        {"metrics": {"primary_metric": []}},
        {},
        {},
        _evaluate_tail,
    )

    assert tail["source"] == "paired_baseline.final"
    assert captured["deltas"] == []
    assert captured["weights"] is None


def test_report_builder_support_tail_evaluator_exception_falls_back() -> None:
    result = evaluate_primary_metric_tail(
        {"metrics": {"primary_metric": {"kind": "ppl_causal"}}},
        {},
        {},
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("tail failed")),
    )

    assert result == {"mode": "warn", "evaluated": False, "passed": False}


def test_validate_retry_evaluation_report_loads_baseline_path_and_validation_edges(
    tmp_path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"baseline": True}), encoding="utf-8")

    result = validate_retry_evaluation_report(
        report={"run": True},
        baseline_report_data=None,
        baseline_path=baseline_path,
        build_retry_result_summary_fn=lambda validation: {
            "passed": False,
            "failures": ["primary_metric_acceptable"],
            "validation": validation,
        },
        make_report_fn=lambda _report, _baseline: {"validation": ["bad-shape"]},
        telemetry_output_enabled_fn=lambda: True,
        telemetry_summary_line_fn=lambda _report: "telemetry-summary",
    )

    assert result.status == "failed"
    assert result.validation == {}
    assert result.validation_gates == ("primary_metric_acceptable",)
    assert result.telemetry_summary == "telemetry-summary"


def test_validate_retry_evaluation_report_uses_default_report_factory(monkeypatch):
    def _fake_make_report(_report, _baseline):
        return {"validation": {"primary_metric_acceptable": True}}

    monkeypatch.setattr(report_make_mod, "make_report", _fake_make_report)
    monkeypatch.setattr(
        "invarlock.reporting.report_make.make_report", _fake_make_report
    )

    result = validate_retry_evaluation_report(
        report={"run": True},
        baseline_report_data={"baseline": True},
        baseline_path=None,
        build_retry_result_summary_fn=lambda validation: {
            "passed": True,
            "failures": [],
            "validation": validation,
        },
        make_report_fn=None,
        telemetry_output_enabled_fn=lambda: False,
    )

    assert result.status == "passed"
    assert result.validation == {"primary_metric_acceptable": True}


def test_validate_retry_evaluation_report_rejects_non_mapping_baseline_file(
    tmp_path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text("[]", encoding="utf-8")

    result = validate_retry_evaluation_report(
        report={"run": True},
        baseline_report_data=None,
        baseline_path=baseline_path,
        build_retry_result_summary_fn=lambda _validation: {
            "passed": True,
            "failures": [],
        },
        make_report_fn=lambda _report, _baseline: {"validation": {}},
    )

    assert result.status == "error"
    assert result.validation_gates == ("report_error",)
