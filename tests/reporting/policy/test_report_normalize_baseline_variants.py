from __future__ import annotations

import math

from invarlock.reporting.report_normalization import normalize_baseline


def test_normalize_baseline_v1_schema() -> None:
    base_v1 = {
        "schema_version": "baseline-v1",
        "meta": {"model_id": "m", "commit_sha": "1234567890abcdef"},
        "metrics": {"ppl_final": 10.0},
        "spectral_base": {"caps": 0},
        "rmt_base": {"epsilon": {}},
    }
    norm = normalize_baseline(base_v1)
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
    norm = normalize_baseline(base)
    assert norm.get("ppl_final") == 12.0


def test_normalize_baseline_v1_derives_weighted_ppl_from_windows() -> None:
    base_v1 = {
        "schema_version": "baseline-v1",
        "meta": {"model_id": "m", "commit_sha": "cafebabecafebabe"},
        "metrics": {},
        "evaluation_windows": {
            "final": {"logloss": [1.0, 3.0], "token_counts": [1, 3]}
        },
    }

    norm = normalize_baseline(base_v1)
    expected = math.exp((1.0 * 1.0 + 3.0 * 3.0) / 4.0)
    assert math.isclose(float(norm["ppl_final"]), expected, rel_tol=1e-12, abs_tol=0.0)


def test_normalize_baseline_v1_non_finite_window_logloss_does_not_infer_ppl() -> None:
    base_v1 = {
        "schema_version": "baseline-v1",
        "meta": {"model_id": "m", "commit_sha": "cafebabecafebabe"},
        "metrics": {},
        "evaluation_windows": {
            "final": {"logloss": [float("inf")], "token_counts": [1]}
        },
    }

    norm = normalize_baseline(base_v1)
    assert norm.get("model_id") == "m"
    assert "ppl_final" not in norm


def test_normalize_baseline_runreport_zero_weights_falls_back_to_unweighted_mean() -> (
    None
):
    baseline = {
        "meta": {"model_id": "m"},
        "edit": {
            "name": "baseline",
            "plan_digest": "baseline_noop",
            "deltas": {"params_changed": 0},
        },
        "metrics": {},
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2],
                "logloss": [0.2, 0.4],
                "token_counts": [0, 0],
            },
            "preview": {
                "window_ids": [1, 2],
                "logloss": [0.2, 0.4],
                "token_counts": [0, 0],
            },
        },
    }

    out = normalize_baseline(baseline)
    expected = math.exp((0.2 + 0.4) / 2.0)
    assert math.isclose(float(out["ppl_final"]), expected, rel_tol=1e-12, abs_tol=0.0)
    assert math.isclose(float(out["ppl_preview"]), expected, rel_tol=1e-12, abs_tol=0.0)


def test_normalize_baseline_runreport_accuracy_zero_final_does_not_raise() -> None:
    baseline = {
        "meta": {"model_id": "acc-model"},
        "edit": {
            "name": "baseline",
            "plan_digest": "baseline_noop",
            "deltas": {"params_changed": 0},
        },
        "metrics": {
            "primary_metric": {"kind": "accuracy", "final": 0.0, "preview": 0.0}
        },
        "evaluation_windows": {},
    }

    out = normalize_baseline(baseline)
    assert out.get("model_id") == "acc-model"
    assert "ppl_final" not in out


def test_normalize_baseline_v1_accuracy_zero_final_does_not_raise() -> None:
    baseline = {
        "schema_version": "baseline-v1",
        "meta": {"model_id": "acc-v1", "commit_sha": "0123456789abcdef"},
        "metrics": {"primary_metric": {"kind": "accuracy", "final": 0.0}},
    }

    out = normalize_baseline(baseline)
    assert out.get("model_id") == "acc-v1"
    assert "ppl_final" not in out


def test_normalize_baseline_normalized_accuracy_strips_invalid_ppl_fields() -> None:
    baseline = {
        "run_id": "r1",
        "model_id": "acc-model",
        "metrics": {"primary_metric": {"kind": "accuracy", "final": 0.0}},
        "ppl_final": "nan-value",
        "ppl_preview": 0.0,
    }

    out = normalize_baseline(baseline)
    assert out.get("model_id") == "acc-model"
    assert "ppl_final" not in out
    assert "ppl_preview" not in out


def test_normalize_baseline_normalized_top_level_accuracy_strips_invalid_ppl_fields() -> (
    None
):
    baseline = {
        "run_id": "r2",
        "model_id": "acc-model-top",
        "primary_metric": {"kind": "accuracy", "final": 0.0},
        "ppl_final": float("inf"),
        "ppl_preview": -1.0,
    }

    out = normalize_baseline(baseline)
    assert out.get("model_id") == "acc-model-top"
    assert "ppl_final" not in out
    assert "ppl_preview" not in out


def test_normalize_baseline_normalized_top_level_ppl_derives_ppl_fields() -> None:
    baseline = {
        "run_id": "r3",
        "model_id": "ppl-model",
        "primary_metric": {"kind": "ppl_causal", "final": 11.0, "preview": 10.5},
    }

    out = normalize_baseline(baseline)
    assert out.get("ppl_final") == 11.0
    assert out.get("ppl_preview") == 10.5


def test_normalize_baseline_missing_kind_does_not_infer_ppl_fields() -> None:
    baseline = {
        "run_id": "r4",
        "model_id": "unknown-kind-model",
        "primary_metric": {"final": 11.0, "preview": 10.5},
    }

    out = normalize_baseline(baseline)
    assert out.get("model_id") == "unknown-kind-model"
    assert "ppl_final" not in out
    assert "ppl_preview" not in out


def test_normalize_baseline_handles_unstringable_kind_value() -> None:
    class _BadKind:
        def __str__(self) -> str:  # pragma: no cover - exercised by test intent
            raise RuntimeError("boom")

    baseline = {
        "schema_version": "baseline-v1",
        "meta": {"model_id": "m", "commit_sha": "abcdef0123456789"},
        "metrics": {"primary_metric": {"kind": _BadKind(), "final": 0.0}},
    }

    norm = normalize_baseline(baseline)
    assert norm.get("model_id") == "m"
    assert "ppl_final" not in norm
