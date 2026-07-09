from __future__ import annotations

from unittest.mock import patch

from invarlock.reporting.render import render_report_markdown
from invarlock.reporting.report_make import make_report


def _make_cert(*, metrics: dict | None = None, edit_deltas: dict | None = None) -> dict:
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": metrics or {"ppl_preview": 10.0, "ppl_final": 10.0},
        "data": {
            "dataset": "d",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
        },
        "guards": [],
        "edit": {
            "name": "structured",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
                **(edit_deltas or {}),
            },
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "ppl_final": 10.0,
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        return make_report(report, baseline)


def test_drift_basis_point_only_when_no_ci():
    md = render_report_markdown(_make_cert())
    assert "| point |" in md


def test_drift_basis_includes_ci_informational_when_ci_present():
    cert = _make_cert(
        metrics={
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            "ppl_drift_ci": (0.98, 1.02),
        }
    )
    md = render_report_markdown(cert)
    assert "| point |" in md


def test_render_markdown_uses_point_basis_when_no_ratio_ci():
    cert = _make_cert(
        metrics={"ppl_preview": 10.0, "ppl_final": 10.0, "ppl_ratio": 1.0}
    )
    cert.setdefault("auto", {})["tier"] = "balanced"
    cert["auto"]["target_pm_ratio"] = 1.0

    md = render_report_markdown(cert)

    assert "Quality Gates" in md
    assert "| point |" in md
    assert "≤ 1.10x" in md
    assert "≤ 1.00x" not in md


def test_render_spectral_omits_policy_yaml_when_absent():
    cert = _make_cert()
    spectral = cert.get("spectral", {})
    spectral.pop("policy", None)
    cert["spectral"] = spectral

    md = render_report_markdown(cert)

    assert "Family κ (policy):" not in md


def test_render_omits_rmt_section_when_empty():
    cert = _make_cert()
    cert["rmt"] = {}

    md = render_report_markdown(cert)

    assert "| Family | ε_f | Bare | Guarded |" not in md


def test_render_rmt_no_baseline_outliers_row():
    cert = _make_cert(edit_deltas={"sparsity": None})
    cert["rmt"].update(
        {"outliers_bare": 0, "outliers_guarded": 0, "stable": True, "epsilon": 0.1}
    )

    md = render_report_markdown(cert)

    assert "RMT" in md


def test_render_spectral_no_tables_when_empty():
    cert = _make_cert()
    spectral = cert.get("spectral", {})
    spectral.pop("caps_applied_by_family", None)
    spectral.pop("family_z_quantiles", None)
    cert["spectral"] = spectral

    md = render_report_markdown(cert)

    assert "| Family | κ | Violations |" not in md
    assert "| Family | q95 | q99 | Max | Samples |" not in md


def test_spectral_top_z_non_numeric_formats_as_na():
    cert = _make_cert()
    cert.setdefault("spectral", {})["top_z_scores"] = {
        "ffn": [{"module": "L0", "z": "nan"}]
    }

    md = render_report_markdown(cert)

    assert "n/a" in md
