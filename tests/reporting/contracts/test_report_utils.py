from __future__ import annotations

from invarlock.reporting import report_schema as schema_mod
from invarlock.reporting.report_enrichment import (
    compute_confidence_label as _compute_confidence_label,
)
from invarlock.reporting.report_primary_metric_policy import is_ppl_kind as _is_ppl_kind
from invarlock.reporting.report_provenance import (
    compute_edit_digest as _compute_edit_digest,
)
from invarlock.reporting.report_summary import compute_console_validation_block


def test_is_ppl_kind_and_get_ppl_final() -> None:
    assert _is_ppl_kind("ppl_causal")
    assert not _is_ppl_kind("perplexity")
    assert not _is_ppl_kind("ppl")
    assert not _is_ppl_kind("accuracy")

    # Legacy _get_ppl_final removed; rely on normalized primary_metric in evaluation_reports.


def test_compute_edit_digest_quant_and_default() -> None:
    rep_quant = {"edit": {"name": "quant_rtn", "config": {"bitwidth": 4}}}
    d1 = _compute_edit_digest(rep_quant)
    assert d1["family"] == "quantization"
    rep_none = {"edit": {"name": "noop"}}
    d2 = _compute_edit_digest(rep_none)
    assert d2["family"] == "report_only"


def test_confidence_label_paths() -> None:
    # ppl-like, stable width => High, basis ppl_ratio
    c1 = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {"kind": "ppl_causal", "display_ci": [1.00, 1.02]},
        "resolved_policy": {"confidence": {"ppl_ratio_width_max": 0.05}},
    }
    out1 = _compute_confidence_label(c1)
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
    out2 = _compute_confidence_label(c2)
    assert out2["basis"] == "accuracy"
    assert out2["label"] in {"Medium", "Low"}


def test_validate_evaluation_report_rejects_non_boolean_flags() -> None:
    bad = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
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
    assert schema_mod.validate_report(bad) is False


def test_validate_evaluation_report_rejects_minimal_payload_without_schema_success() -> (
    None
):
    minimal = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "r",
        "primary_metric": {"kind": "ppl_causal"},
    }
    assert schema_mod.validate_report(minimal) is False


def test_validate_evaluation_report_rejects_unknown_primary_metric_kind() -> None:
    bad = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "r",
        "artifacts": {"generated_at": "t"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "p",
            "seq_len": 8,
            "windows": {"preview": 0, "final": 0},
        },
        "primary_metric": {"kind": "perplexity", "final": 10.0},
        "validation": {"primary_metric_acceptable": True},
    }
    assert schema_mod.validate_report(bad) is False


def test_validate_evaluation_report_accepts_optional_evaluation_realism() -> None:
    report = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "r",
        "artifacts": {"generated_at": "t"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "p",
            "seq_len": 8,
            "windows": {"preview": 0, "final": 0, "stats": {}},
        },
        "primary_metric": {"kind": "ppl_causal", "final": 10.0},
        "validation": {"primary_metric_acceptable": True},
        "evaluation_realism": {
            "mode": "generation",
            "prompt_template_hash": "sha256:" + "a" * 64,
            "decoding_config": {"temperature": 0.2, "top_p": 0.9},
            "max_tokens": 128,
            "truncation_policy": "truncate_to_context",
            "dataset_or_task_id": "qaedit-generation-smoke",
            "metric_is_generation_realistic": True,
        },
    }

    assert schema_mod.validate_report(report) is True


def test_validate_evaluation_report_rejects_malformed_evaluation_realism() -> None:
    report = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "r",
        "artifacts": {"generated_at": "t"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "p",
            "seq_len": 8,
            "windows": {"preview": 0, "final": 0, "stats": {}},
        },
        "primary_metric": {"kind": "ppl_causal", "final": 10.0},
        "validation": {"primary_metric_acceptable": True},
        "evaluation_realism": {
            "mode": "unrealistic_mode",
            "prompt_template_hash": "not-a-digest",
            "decoding_config": "temperature=0",
            "max_tokens": -1,
            "metric_is_generation_realistic": "no",
        },
    }

    assert schema_mod.validate_report(report) is False


def test_console_validation_block_guard_skipped_and_included() -> None:
    base = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
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
    block1 = compute_console_validation_block(base)
    labels1 = block1["labels"]
    assert all(label != "Guard Overhead Acceptable" for label in labels1)

    # Evaluated: include guard row
    base2 = dict(base)
    base2["guard_overhead"] = {"evaluated": True}
    block2 = compute_console_validation_block(base2)
    labels2 = block2["labels"]
    assert "Guard Overhead Acceptable" in labels2
