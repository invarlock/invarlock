from __future__ import annotations

from pathlib import Path

import pytest

import invarlock.reporting.run_report_contract as run_report_contract_mod
import invarlock.reporting.run_report_formatters as run_report_formatters_mod
import invarlock.reporting.validate as validate_mod
import invarlock.reporting.validation.report as report_validation_mod
import invarlock.reporting.verify_check_helpers_consistency as verify_helpers_mod
import invarlock.reporting.verify_check_helpers_metrics as verify_metrics_mod
import invarlock.reporting.verify_contract as verify_contract_mod
from invarlock.reporting.report_types import RunReport, create_empty_report
from invarlock.reporting.verify_contract_types import normalize_warning_policy
from tests.reporting._support_guard_metric_impact import canonical_ppl_impact


def _valid_report() -> RunReport:
    report = create_empty_report()
    report["meta"].update(
        {
            "model_id": "gpt2",
            "adapter": "hf_causal",
            "commit": "deadbeef",
            "seed": 7,
            "device": "cpu",
            "ts": "2026-04-08T00:00:00",
        }
    )
    report["data"].update(
        {
            "dataset": "wikitext2",
            "split": "validation",
            "seq_len": 128,
            "stride": 64,
            "preview_n": 1,
            "final_n": 1,
        }
    )
    report["edit"].update(
        {
            "name": "quant_rtn",
            "plan_digest": "abc123",
            "deltas": {
                "params_changed": 1,
                "sparsity": None,
                "bitwidth_map": None,
                "layers_modified": 1,
            },
        }
    )
    report["metrics"].update(
        {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 9.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
            },
            "latency_ms_per_tok": 1.23,
            "memory_mb_peak": 256.0,
        }
    )
    return report


def test_validation_result_diagnostics_and_summary_cover_warning_and_error_paths() -> (
    None
):
    result = validate_mod.ValidationResult(
        passed=False,
        checks={"schema": True, "gates": False},
        metrics={},
        messages=["loaded"],
        warnings=["warned"],
        errors=["broken"],
    )

    diagnostics = result.diagnostics
    assert [item["severity"] for item in diagnostics] == ["info", "warning", "error"]
    summary = result.summary()
    assert "Warnings:" in summary
    assert "Errors:" in summary

    warnings_only = validate_mod.ValidationResult(
        passed=True,
        checks={"schema": True},
        metrics={},
        messages=[],
        warnings=["warn only"],
        errors=[],
    )
    warnings_summary = warnings_only.summary()
    assert "Warnings:" in warnings_summary
    assert "warn only" in warnings_summary


def test_report_validation_helpers_cover_guard_metric_impact_and_degradation_limit() -> (
    None
):
    assert (
        report_validation_mod._guard_metric_impact_has_error_diagnostic(None) is False
    )  # noqa: SLF001
    assert (
        report_validation_mod._guard_metric_impact_has_error_diagnostic(  # noqa: SLF001
            {"diagnostics": "bad"}
        )
        is False
    )
    assert (
        report_validation_mod._guard_metric_impact_has_error_diagnostic(  # noqa: SLF001
            {"diagnostics": ["bad"]}
        )
        is False
    )
    assert (
        report_validation_mod._guard_metric_impact_has_error_diagnostic(  # noqa: SLF001
            {"diagnostics": [{"severity": "error"}]}
        )
        is True
    )

    assert not report_validation_mod._resolve_guard_metric_impact_pass(  # noqa: SLF001
        {"degradation": 1.005, "degradation_limit": None},
        tiny_relax=False,
    )
    assert not report_validation_mod._resolve_guard_metric_impact_pass(  # noqa: SLF001
        {"passed": False, "evaluated": False},
        tiny_relax=True,
    )
    assert report_validation_mod._resolve_guard_metric_impact_pass(  # noqa: SLF001
        canonical_ppl_impact(10.0, 10.05, degradation_limit=0.01),
        tiny_relax=False,
    )

    class _ExplodingFloat(float):
        def __float__(self) -> float:
            raise TypeError("boom")

    flags = report_validation_mod.compute_validation_flags(
        {"preview_final_ratio": 1.0, "ratio_vs_baseline": _ExplodingFloat(1.0)},
        {"stable": True},
        {"stable": True},
        {"status": "pass"},
        pm_acceptance_range={"min": 0.95, "max": 1.10},
        get_tier_policies_fn=lambda: {
            "balanced": {"metrics": {"pm_ratio": {"ratio_limit_base": 1.10}}}
        },
    )
    assert flags["preview_final_drift_acceptable"] is True


def test_metric_impact_validation_rejects_boolean_uncoercible_and_untyped_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from invarlock.reporting.validation import metric_impact

    class _Uncoercible:
        def __float__(self) -> float:
            raise ValueError("not numeric")

    assert metric_impact._coerce_finite_float(True) is None
    assert metric_impact._coerce_finite_float(_Uncoercible()) is None

    monkeypatch.setattr(
        metric_impact, "guard_metric_impact_payload_errors", lambda *_a, **_k: []
    )
    assert (
        metric_impact.resolve_guard_metric_impact_pass(
            {
                "evaluated": True,
                "passed": "yes",
                "degradation": 0.0,
                "degradation_limit": 0.1,
                "diagnostics": [],
            },
            tiny_relax=False,
        )
        is False
    )


def test_run_report_contract_persistence_covers_missing_json_and_telemetry_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = create_empty_report()

    monkeypatch.setattr(
        run_report_contract_mod.report_bundle_module,
        "save_report",
        lambda *_args, **_kwargs: {},
    )
    with pytest.raises(RuntimeError, match="json artifact path"):
        run_report_contract_mod.persist_run_report_outputs(
            report=report,
            run_dir=tmp_path,
            run_config={},
            telemetry=False,
            save_telemetry_report_fn=lambda *_args, **_kwargs: (
                tmp_path / "telemetry.json"
            ),
        )

    monkeypatch.setattr(
        run_report_contract_mod.report_bundle_module,
        "save_report",
        lambda *_args, **_kwargs: {"json": tmp_path / "report.json"},
    )
    result = run_report_contract_mod.persist_run_report_outputs(
        report=create_empty_report(),
        run_dir=tmp_path,
        run_config={},
        telemetry=True,
        save_telemetry_report_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("telemetry boom")
        ),
    )
    assert result.report_path_out.endswith("report.json")
    assert result.telemetry_error == "telemetry boom"


def test_run_report_formatters_cover_string_diagnostics_and_html_closures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _valid_report()
    report["guards"] = [
        {
            "name": "spectral",
            "passed": True,
            "decision": "allow",
            "policy": {},
            "metrics": {},
            "diagnostics": ["note", {"severity": "warning", "message": "warn"}, 1],
            "violations": [],
        }
    ]
    markdown = run_report_formatters_mod.to_markdown(report)
    assert "- note" in markdown
    assert "- [WARNING] warn" in markdown

    monkeypatch.setattr(
        run_report_formatters_mod,
        "_generate_single_markdown",
        lambda _report: [
            "| Metric | Value |",
            "| - | - |",
            "",
            "plain text",
        ],
    )
    single_html = run_report_formatters_mod._generate_single_html(create_empty_report())  # noqa: SLF001
    assert any("<table class='metrics-table'>" in line for line in single_html)
    assert any("<p>plain text</p>" in line for line in single_html)

    monkeypatch.setattr(
        run_report_formatters_mod,
        "_generate_comparison_markdown",
        lambda _r1, _r2: [
            "| Metric | A | B |",
            "| - | - | - |",
            "",
            "comparison paragraph",
        ],
    )
    comparison_html = run_report_formatters_mod._generate_comparison_html(  # noqa: SLF001
        create_empty_report(),
        create_empty_report(),
    )
    assert any("<table class='comparison-table'>" in line for line in comparison_html)
    assert any("<p>comparison paragraph</p>" in line for line in comparison_html)


def test_verify_helpers_cover_report_loading_primary_metric_and_validation_edges(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        verify_helpers_mod._load_evaluation_report(report_path)  # noqa: SLF001

    pm_errors = verify_helpers_mod._validate_primary_metric(  # noqa: SLF001
        {
            "primary_metric": {
                "kind": "ppl_causal",
                "final": 2.0,
                "ratio_vs_baseline": 1.0,
            },
            "baseline_ref": {"primary_metric": {"final": 0.0}},
        }
    )
    assert any("baseline final must be at least 1.0" in err for err in pm_errors)
    malformed_baseline_errors = verify_helpers_mod._validate_primary_metric(  # noqa: SLF001
        {
            "primary_metric": {
                "kind": "ppl_causal",
                "final": 2.0,
                "ratio_vs_baseline": 1.0,
            },
            "baseline_ref": {"primary_metric": {"final": "bad"}},
        }
    )
    assert any("same-kind baseline" in err for err in malformed_baseline_errors)

    assert verify_helpers_mod._validate_release_gate_outcomes({}) == [  # noqa: SLF001
        "Release verification requires a validation block."
    ]
    tail_gate_errors = verify_helpers_mod._validate_release_gate_outcomes(  # noqa: SLF001
        {
            "primary_metric_tail": {"mode": "fail"},
            "validation": {
                "primary_metric_acceptable": True,
                "preview_final_drift_acceptable": True,
                "invariants_pass": True,
                "spectral_stable": True,
                "rmt_stable": True,
                "primary_metric_tail_acceptable": False,
            },
        }
    )
    assert any("rejected primary_metric_tail" in err for err in tail_gate_errors)
    fabricated_tail_flag_errors = verify_helpers_mod._validate_release_gate_outcomes(  # noqa: SLF001
        {
            "validation": {
                "primary_metric_acceptable": True,
                "preview_final_drift_acceptable": True,
                "invariants_pass": True,
                "spectral_stable": True,
                "rmt_stable": True,
                "primary_metric_tail_acceptable": True,
            }
        }
    )
    assert any(
        "without primary_metric_tail evidence" in error
        for error in fabricated_tail_flag_errors
    )
    missing_ppl_tail_errors = verify_helpers_mod._validate_release_gate_outcomes(  # noqa: SLF001
        {
            "primary_metric": {"kind": "ppl_causal"},
            "validation": {
                "primary_metric_acceptable": True,
                "preview_final_drift_acceptable": True,
                "invariants_pass": True,
                "spectral_stable": True,
                "rmt_stable": True,
            },
        }
    )
    assert any(
        "requires primary_metric_tail evidence" in error
        for error in missing_ppl_tail_errors
    )
    pairing_errors = verify_helpers_mod._validate_pairing(  # noqa: SLF001
        {
            "dataset": {
                "windows": {
                    "stats": {
                        "window_match_fraction": 0.5,
                        "window_overlap_fraction": 0.1,
                        "window_pairing_reason": "fallback",
                        "paired_windows": 0,
                    }
                }
            }
        }
    )
    assert len(pairing_errors) == 4

    counts_errors = verify_helpers_mod._validate_counts(  # noqa: SLF001
        {
            "dataset": {
                "windows": {
                    "preview": "bad",
                    "final": "bad",
                    "stats": {"paired_windows": "bad", "coverage": {}},
                }
            }
        }
    )
    assert "report has invalid dataset.windows.preview count." in counts_errors
    assert "report has invalid dataset.windows.final count." in counts_errors
    assert "report has invalid paired_windows metric." in counts_errors
    final_used_errors = verify_helpers_mod._validate_counts(  # noqa: SLF001
        {
            "dataset": {
                "windows": {
                    "preview": 2,
                    "final": 2,
                    "stats": {
                        "paired_windows": 2,
                        "coverage": {
                            "preview": {"used": 2},
                            "final": {"used": "bad"},
                        },
                    },
                }
            }
        }
    )
    assert "report has invalid coverage.final.used value." in final_used_errors

    drift_errors = verify_helpers_mod._validate_drift_band(  # noqa: SLF001
        {"primary_metric": {"preview": 1.0, "final": 2.0, "drift_band": [0.9, 1.1]}}
    )
    assert any("drift ratio out of band" in err for err in drift_errors)
    assert verify_helpers_mod._validate_drift_band({}) == [  # noqa: SLF001
        "report missing primary_metric block."
    ]

    class _BadDict(dict):
        def get(self, key, default=None):
            if key == "invalid":
                return False
            raise RuntimeError("boom")

    drift_parse_errors = verify_helpers_mod._validate_drift_band(  # noqa: SLF001
        {"primary_metric": _BadDict(preview=1.0, final=2.0, drift_band=(0.9, 1.1))}
    )
    assert drift_parse_errors == [
        "report missing preview/final to compute drift ratio."
    ]

    recomputed = verify_metrics_mod._recompute_validation_flags(  # noqa: SLF001
        {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
            },
            "dataset": {
                "windows": {
                    "stats": {
                        "bootstrap": {
                            "coverage": {
                                "preview": {"used": 1, "required": 1, "ok": True},
                                "final": {"used": 1, "required": 1, "ok": True},
                            }
                        }
                    }
                }
            },
        }
    )
    assert recomputed["primary_metric_acceptable"] is True

    tok_errors = verify_helpers_mod._validate_tokenizer_hash(  # noqa: SLF001
        {
            "meta": {},
            "dataset": {"tokenizer": {"hash": "abc"}},
            "baseline_ref": {"tokenizer_hash": "xyz"},
        }
    )
    assert tok_errors == ["Tokenizer hash mismatch between baseline and edited runs."]

    class _BrokenMeta(dict):
        def get(self, key, default=None):
            raise TypeError("bad meta")

    assert (
        verify_helpers_mod._validate_tokenizer_hash(  # noqa: SLF001
            {
                "meta": _BrokenMeta(),
                "dataset": {},
                "baseline_ref": {"tokenizer_hash": "abc"},
            }
        )
        == []
    )

    class _BadMeta:
        def get(self, *_args, **_kwargs):
            raise AttributeError("boom")

    assert (
        verify_helpers_mod._validate_tokenizer_hash(  # noqa: SLF001
            {
                "meta": _BadMeta(),
                "dataset": {},
                "baseline_ref": {"tokenizer_hash": "abc"},
            }
        )
        == []
    )

    non_dict_stats = verify_metrics_mod._recompute_validation_flags(  # noqa: SLF001
        {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
            },
            "dataset": {"windows": {"stats": "bad"}},
        }
    )
    assert non_dict_stats["primary_metric_acceptable"] is True

    invalid_band_errors = verify_helpers_mod._validate_drift_band(  # noqa: SLF001
        {
            "primary_metric": {
                "preview": 1.0,
                "final": 1.5,
                "drift_band": [2.0, 1.0],
            },
        }
    )
    assert any("drift ratio out of band" in err for err in invalid_band_errors)

    class _BadBand(list):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index):  # noqa: ANN001
            raise RuntimeError(f"boom:{index}")

    drift_parse_band_errors = verify_helpers_mod._validate_drift_band(  # noqa: SLF001
        {
            "primary_metric": {
                "preview": 1.0,
                "final": 1.5,
                "drift_band": _BadBand([0.9, 1.1]),
            },
        }
    )
    assert any("drift ratio out of band" in err for err in drift_parse_band_errors)

    class _BrokenResolvedPolicy(dict):
        def get(self, key, default=None):  # noqa: ANN001
            raise RuntimeError(f"broken:{key}")

    accuracy_default_errors = verify_helpers_mod._validate_drift_band(  # noqa: SLF001
        {
            "primary_metric": {
                "kind": "accuracy",
                "preview": 0.9,
                "final": 0.75,
            },
            "resolved_policy": _BrokenResolvedPolicy({"metrics": {}}),
        }
    )
    assert any("≤ 0.100000" in err for err in accuracy_default_errors)
    assert verify_helpers_mod._validate_drift_band(  # noqa: SLF001
        {
            "primary_metric": {
                "kind": "accuracy",
                "preview": 0.9,
                "final": 0.75,
            },
            "resolved_policy": {"metrics": {"accuracy": "not-a-dict"}},
        }
    )
    assert (
        verify_helpers_mod._validate_drift_band(  # noqa: SLF001
            {
                "primary_metric": {
                    "kind": "accuracy",
                    "preview": 0.9,
                    "final": 0.75,
                },
                "resolved_policy": {
                    "metrics": {"accuracy": {"preview_final_delta_pp_max": 0.2}}
                },
            }
        )
        == []
    )
    assert (
        verify_helpers_mod._validate_drift_band(  # noqa: SLF001
            {"primary_metric": {"kind": "bleu"}}
        )
        == []
    )


def test_verify_contract_profile_resolution_and_baseline_digest_fallbacks(
    tmp_path: Path,
) -> None:
    class _BadProfile:
        def __bool__(self) -> bool:
            raise TypeError("boom")

    assert verify_contract_mod._resolve_profile_name(_BadProfile()) == "dev"  # noqa: SLF001

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text("{not-json", encoding="utf-8")
    assert (
        verify_contract_mod._load_baseline_digest(baseline_path)  # noqa: SLF001
        is None
    )

    for removed_alias in ("strict", "warn", "advisory"):
        with pytest.raises(ValueError, match="warning_policy"):
            normalize_warning_policy(removed_alias)
    with pytest.raises(ValueError, match="warning_policy"):
        normalize_warning_policy("bad")

    diagnostics = verify_contract_mod._guard_warning_diagnostics(  # noqa: SLF001
        {
            "guard_warnings": {
                "warning_count": 2,
                "warnings": [
                    "bad-entry",
                    {
                        "guard": "spectral",
                        "kind": "new_capped_module",
                        "module": "layers.0.mlp",
                        "policy_gate": "pass",
                    },
                ],
            }
        }
    )
    assert [diagnostic.level for diagnostic in diagnostics] == ["warning", "warning"]
    assert "layers.0.mlp" in diagnostics[1].message
    for malformed_warnings in (
        {"warning_count": "2", "warnings": [{}, {}]},
        {"warning_count": 2, "warnings": "not-a-list"},
        {"warning_count": 2, "warnings": [{}]},
    ):
        with pytest.raises(ValueError, match="guard_warnings"):
            verify_contract_mod._guard_warning_diagnostics(  # noqa: SLF001
                {"guard_warnings": malformed_warnings}
            )


def test_verify_helpers_and_contract_cover_profile_parse_and_recompute_edges(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text("{}", encoding="utf-8")

    class _BadProfile(str):
        def strip(self) -> str:  # type: ignore[override]
            raise RuntimeError("boom")

    errors = verify_helpers_mod._validate_evaluation_report_payload(  # noqa: SLF001
        report_path,
        profile=_BadProfile("ci"),
        load_evaluation_report_fn=lambda _path: {},
        validate_report_fn=lambda _report: True,
        validate_report_schema_strict_fn=lambda _report: True,
        validate_primary_metric_fn=lambda _report: [],
        validate_pairing_fn=lambda _report: [],
        validate_counts_fn=lambda _report: [],
        validate_logspace_ci_identity_fn=lambda _report, profile=None: [],
        validate_drift_band_fn=lambda _report: [],
        validate_primary_metric_policy_fn=lambda _report, profile=None: [],
        apply_profile_lints_fn=lambda _report: [],
        validate_tokenizer_hash_fn=lambda _report: [],
        validate_measurement_contracts_fn=lambda _report, profile=None: [],
    )
    assert errors == []

    accuracy_warning = verify_contract_mod._append_recompute_errors(  # noqa: SLF001
        [],
        cert_obj={"primary_metric": {"kind": "accuracy"}, "metrics": {}},
        prof="dev",
        tol=1e-9,
        json_mode=False,
    )
    assert accuracy_warning.diagnostics == ()
    assert accuracy_warning.metric_mismatch is False
    assert (
        verify_contract_mod._append_recompute_errors(  # noqa: SLF001
            [],
            cert_obj={"primary_metric": {"kind": "accuracy"}, "metrics": {}},
            prof="dev",
            tol=1e-9,
            json_mode=True,
        ).diagnostics
        == ()
    )
    assert (
        verify_contract_mod._append_recompute_errors(  # noqa: SLF001
            [],
            cert_obj={
                "primary_metric": {"kind": "ppl_causal"},
                "evaluation_windows": {},
            },
            prof="dev",
            tol=1e-9,
            json_mode=True,
        ).diagnostics
        == ()
    )
    assert (
        verify_contract_mod._append_recompute_errors(  # noqa: SLF001
            [],
            cert_obj={
                "primary_metric": {"kind": "ppl_causal"},
                "evaluation_windows": {"final": {}},
            },
            prof="dev",
            tol=1e-9,
            json_mode=True,
        ).diagnostics
        == ()
    )
    assert (
        verify_contract_mod._append_recompute_errors(  # noqa: SLF001
            [],
            cert_obj={
                "primary_metric": {"kind": "ppl_causal"},
                "evaluation_windows": {"final": {"logloss": [], "token_counts": []}},
            },
            prof="ci",
            tol=1e-9,
            json_mode=False,
        ).diagnostics
        == ()
    )

    basis_errors: list[str] = []
    assert (
        verify_contract_mod._append_recompute_errors(  # noqa: SLF001
            basis_errors,
            cert_obj={
                "primary_metric": {
                    "kind": "ppl_causal",
                    "analysis_point_final": 0.0,
                },
                "evaluation_windows": {
                    "final": {"logloss": [1.0], "token_counts": [1]}
                },
            },
            prof="dev",
            tol=1e-9,
            json_mode=True,
        ).diagnostics
        == ()
    )
    assert any("Basis mismatch" in error for error in basis_errors)


def test_release_payload_rejects_report_controlled_tiny_relax(tmp_path: Path) -> None:
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text("{}", encoding="utf-8")
    report = {"context": {"run": {"tiny_relax": "on"}}}

    common = {
        "report_payload": report,
        "validate_report_fn": lambda _report: True,
        "validate_report_schema_strict_fn": lambda _report: True,
        "validate_primary_metric_fn": lambda _report: [],
        "validate_pairing_fn": lambda _report: [],
        "validate_counts_fn": lambda _report: [],
        "validate_logspace_ci_identity_fn": lambda _report, profile=None: [],
        "validate_drift_band_fn": lambda _report: [],
        "validate_primary_metric_policy_fn": lambda _report, profile=None: [],
        "apply_profile_lints_fn": lambda _report: [],
        "validate_tokenizer_hash_fn": lambda _report: [],
        "validate_measurement_contracts_fn": lambda _report, profile=None: [],
        "validate_variance_enablement_fn": lambda _report: [],
    }

    dev_errors = verify_helpers_mod._validate_evaluation_report_payload(  # noqa: SLF001
        report_path, profile="dev", **common
    )
    release_errors = verify_helpers_mod._validate_evaluation_report_payload(  # noqa: SLF001
        report_path, profile="release", **common
    )

    assert dev_errors == []
    assert (
        "Release verification forbids development-only tiny_relax policy."
        in release_errors
    )
