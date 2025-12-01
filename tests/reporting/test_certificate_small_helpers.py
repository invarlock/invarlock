from __future__ import annotations

from invarlock.reporting.certificate import (
    _compute_confidence_label,
    _compute_edit_digest,
    _is_ppl_kind,
)


def test_is_ppl_kind_variants() -> None:
    assert _is_ppl_kind("ppl_causal")
    assert _is_ppl_kind("ppl_seq2seq")
    assert not _is_ppl_kind("accuracy")


# Removed legacy _get_ppl_final coverage; use primary_metric in certificate outputs instead.


def test_compute_edit_digest_quantization_and_cert_only() -> None:
    rep_q = {"edit": {"name": "quant_rtn", "config": {"alpha": 0.1}}}
    d_q = _compute_edit_digest(rep_q)
    assert d_q["family"] == "quantization" and isinstance(d_q["impl_hash"], str)
    rep_c = {"edit": {"name": "foo"}}
    d_c = _compute_edit_digest(rep_c)
    assert d_c["family"] == "cert_only" and isinstance(d_c["impl_hash"], str)


def test_compute_confidence_label_high_medium_low() -> None:
    # High: pm_ok, stable, width <= threshold
    cert_high = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {
            "kind": "ppl_causal",
            "display_ci": [1.00, 1.01],
            "unstable": False,
        },
        "resolved_policy": {"confidence": {"ppl_ratio_width_max": 0.03}},
    }
    lbl_high = _compute_confidence_label(cert_high)
    assert lbl_high["label"] == "High" and lbl_high["basis"] == "ppl_ratio"

    # Medium: pm_ok but unstable
    cert_med = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {
            "kind": "accuracy",
            "display_ci": [0.80, 0.81],
            "unstable": True,
        },
        "resolved_policy": {"confidence": {"accuracy_delta_pp_width_max": 1.0}},
    }
    lbl_med = _compute_confidence_label(cert_med)
    assert lbl_med["label"] == "Medium" and lbl_med["basis"] == "accuracy"

    # Low: pm not ok or missing bounds
    cert_low = {
        "validation": {"primary_metric_acceptable": False},
        "primary_metric": {"kind": "accuracy", "display_ci": []},
    }
    lbl_low = _compute_confidence_label(cert_low)
    assert lbl_low["label"] == "Low"
