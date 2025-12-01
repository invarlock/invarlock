from __future__ import annotations

from invarlock.reporting import certificate as cert


def test_is_ppl_kind_and_get_ppl_final() -> None:
    assert cert._is_ppl_kind("ppl")
    assert cert._is_ppl_kind("ppl_causal")
    assert not cert._is_ppl_kind("accuracy")

    # Legacy _get_ppl_final removed; rely on normalized primary_metric in certificates.


def test_compute_edit_digest_quant_and_default() -> None:
    rep_quant = {"edit": {"name": "quant_rtn", "config": {"bits": 4}}}
    d1 = cert._compute_edit_digest(rep_quant)
    assert d1["family"] == "quantization"
    rep_none = {"edit": {"name": "noop"}}
    d2 = cert._compute_edit_digest(rep_none)
    assert d2["family"] == "cert_only"


def test_confidence_label_branches() -> None:
    # ppl-like, stable width => High, basis ppl_ratio
    c1 = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {"kind": "ppl_causal", "display_ci": [1.00, 1.02]},
        "resolved_policy": {"confidence": {"ppl_ratio_width_max": 0.05}},
    }
    out1 = cert._compute_confidence_label(c1)
    assert out1["label"] == "High"
    assert out1["basis"] == "ppl_ratio"

    # accuracy-like, unstable and wide => Medium/Low depending on width
    c2 = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {
            "kind": "accuracy",
            "display_ci": [0.80, 0.83],
            "unstable": True,
        },
        "resolved_policy": {"confidence": {"accuracy_delta_pp_width_max": 0.5}},
    }
    out2 = cert._compute_confidence_label(c2)
    assert out2["basis"] == "accuracy"
    assert out2["label"] in {"Medium", "Low"}


def test_validate_certificate_rejects_non_boolean_flags() -> None:
    bad = {
        "schema_version": cert.CERTIFICATE_SCHEMA_VERSION,
        "run_id": "r",
        "artifacts": {"generated_at": "t"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "p",
            "seq_len": 8,
            "windows": {"preview": 0, "final": 0},
        },
        "primary_metric": {"kind": "ppl_causal", "final": 10.0},
        "validation": {"primary_metric_acceptable": "yes"},  # invalid type
    }
    assert cert.validate_certificate(bad) is False


def test_validate_certificate_fallback_ok_and_schema_minimal() -> None:
    # Force minimal fallback path: missing many properties, but minimal fields present
    minimal = {
        "schema_version": cert.CERTIFICATE_SCHEMA_VERSION,
        "run_id": "r",
        "primary_metric": {"kind": "ppl_causal"},
    }
    # JSONSchema may reject; fallback minimal check should pass
    assert cert.validate_certificate(minimal) is True


def test_console_validation_block_guard_skipped_and_included() -> None:
    base = {
        "schema_version": cert.CERTIFICATE_SCHEMA_VERSION,
        "run_id": "x",
        "artifacts": {"generated_at": "t"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "p",
            "seq_len": 8,
            "windows": {"preview": 0, "final": 0},
        },
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
    }
    # Not evaluated: guard row omitted
    block1 = cert.compute_console_validation_block(base)
    labels1 = block1["labels"]
    assert all(label != "Guard Overhead Acceptable" for label in labels1)

    # Evaluated: include guard row
    base2 = dict(base)
    base2["guard_overhead"] = {"evaluated": True}
    block2 = cert.compute_console_validation_block(base2)
    labels2 = block2["labels"]
    assert "Guard Overhead Acceptable" in labels2
