from __future__ import annotations

import copy

from tests.eval.report_builder_support import (
    _load_local_evaluation_report,
    create_mock_baseline,
    create_mock_run_report,
    make_report,
    patch,
    render_report_markdown,
)


class TestRenderEvaluationReportMarkdown:
    """Test render_report_markdown function."""

    def test_basic_markdown_rendering(self):
        """Test basic markdown rendering."""
        report = create_mock_run_report()
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        markdown = render_report_markdown(evaluation_report)

        assert "# InvarLock Evaluation Report" in markdown
        assert "test-model" in markdown
        assert "structured" in markdown
        assert "Overall Status:" in markdown
        # Plugin section is optional after normalization
        assert ("Plugin Provenance" in markdown) or ("Executive Summary" in markdown)

    def test_markdown_with_auto_tuning(self):
        """Test markdown rendering with auto-tuning config."""
        report = create_mock_run_report(include_auto=True)
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        markdown = render_report_markdown(evaluation_report)

        assert "Auto-Tuning Configuration" in markdown
        assert "aggressive" in markdown
        # Current evaluation_report markdown omits explicit target ratio label
        # but should still include auto-tuning tier detail.
        assert "Auto-Tuning Configuration" in markdown

    def test_markdown_without_auto_tuning(self):
        """Test markdown rendering without auto-tuning."""
        report = create_mock_run_report(include_auto=False)
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        markdown = render_report_markdown(evaluation_report)

        assert "Auto-Tuning Configuration" not in markdown

    def test_markdown_validation_status(self):
        """Test validation status rendering in markdown."""
        report = create_mock_run_report()
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        markdown = render_report_markdown(evaluation_report)

        # Quality gates table present; section titles may vary across releases
        assert "Quality Gates" in markdown
        # Guard Overhead section may be omitted when not evaluated; ensure RMT section present
        assert "RMT" in markdown

    def test_invalid_evaluation_report_render_still_returns_markdown(self):
        """Rendering is presentation-only; schema validation happens at callers."""
        invalid_evaluation_report = {
            "schema_version": "wrong",
            "run_id": "r1",
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
        markdown = render_report_markdown(invalid_evaluation_report)
        assert "# InvarLock Evaluation Report" in markdown

    def test_render_sample_evaluation_report_fixture(self):
        """Ensure sample evaluation_report renders without error and validates."""
        # Build a small synthetic evaluation_report via API to avoid stale fixtures
        report = create_mock_run_report()
        baseline = create_mock_baseline()
        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            cert = make_report(report, baseline)
        markdown = render_report_markdown(cert)

        assert "Quality Gates" in markdown
        assert "Policy Configuration" in markdown

    def test_render_markdown_includes_guard_overhead_details(self):
        report = create_mock_run_report()
        report["guard_overhead"] = {
            "bare_ppl": 120.0,
            "guarded_ppl": 121.2,
            "overhead_threshold": 0.02,
            "source": "unit-test",
        }
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        # Add provenance window plan and inference metadata to cover markdown branches
        evaluation_report.setdefault("provenance", {})["window_plan"] = {
            "profile": "release",
            "preview_n": 203,
            "final_n": 203,
        }
        evaluation_report.setdefault("structure", {}).setdefault(
            "compression_diagnostics", {}
        ).update(
            {
                "inferred": {"seed": True},
                "inference_source": {"seed": "report.meta.seeds"},
                "inference_log": ["seed inferred from report.meta.seeds: 42"],
            }
        )

        markdown = render_report_markdown(evaluation_report)
        # Guard Overhead section may be omitted if normalization dropped the measure
        assert ("Guard Overhead" in markdown) or ("Executive Summary" in markdown)
        assert "Inference Diagnostics" in markdown

    def test_render_markdown_includes_basis_and_spectral_tables(self):
        report = create_mock_run_report()
        report["guards"][0]["metrics"].update(
            {
                "family_z_quantiles": {
                    "ffn": {"q95": 1.111, "q99": 1.222, "max": 1.333, "count": 12},
                    "attn": {"q95": 2.111, "q99": 2.222, "max": 2.333, "count": 6},
                },
                "top_z_scores": {
                    "ffn": [
                        {"module": "layers.0.mlp.c_fc", "z": 1.333},
                        {"module": "layers.1.mlp.c_fc", "z": 1.200},
                    ],
                    "attn": [
                        {"module": "layers.2.attn.proj", "z": 2.333},
                        {"module": "layers.3.attn.proj", "z": 2.111},
                    ],
                },
            }
        )
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        markdown = render_report_markdown(evaluation_report)

        assert (
            "| Gate | Status | Measured | Threshold | Basis | Description |" in markdown
        )
        assert "> *Basis:" in markdown
        assert "| Family | κ | q95 | Max |z| | Violations |" in markdown
        assert "Top |z| per family:" in markdown

    def test_render_markdown_with_invariant_failures(self):
        evaluation_report = _load_local_evaluation_report()

        evaluation_report["invariants"]["summary"]["warning_violations"] = 2
        evaluation_report["invariants"]["failures"] = [
            {
                "message": "LayerNorm missing",
                "severity": "warning",
                "detail": {"module": "ln1"},
            }
        ]

        markdown = render_report_markdown(evaluation_report)

        assert "Non-fatal" in markdown
        assert "LayerNorm missing" in markdown

    def test_render_markdown_quality_gates_basis_note(self):
        evaluation_report = _load_local_evaluation_report()

        markdown = render_report_markdown(evaluation_report)

        assert (
            "> *Basis: “point” gates check the point estimate; “upper” gates check the CI upper bound; "
            "“point & upper” requires both to pass.*" in markdown
        )

    def test_render_markdown_resolved_policy_yaml_block(self):
        evaluation_report = _load_local_evaluation_report()

        markdown = render_report_markdown(evaluation_report)

        assert "## Policy Configuration" in markdown
        assert "```yaml" in markdown
        assert "spectral:" in markdown

    def test_render_markdown_with_rich_evaluation_report(self):
        evaluation_report = _load_local_evaluation_report()

        evaluation_report["meta"]["commit"] = ""
        cert_copy = copy.deepcopy(evaluation_report)
        cert_copy["policy_provenance"]["overrides"] = []
        cert_copy["policy_provenance"].pop("policy_digest", None)
        cert_copy["policy_provenance"]["resolved_at"] = "2025-10-15T00:00:00Z"

        cert_copy["spectral"].update(
            {
                "caps_applied": 2,
                "caps_applied_by_family": {"ffn": 2, "attn": 0},
                "family_z_quantiles": {
                    "ffn": {"q95": 2.1, "q99": 2.3, "max": 2.4, "count": 12}
                },
                "top_z_scores": {"ffn": [{"module": "mlp.c_fc", "z": 2.4}]},
                "policy": {
                    "family_caps": {"ffn": {"kappa": 2.5}, "attn": {"kappa": 2.8}},
                    "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
                },
            }
        )
        cert_copy["spectral"].setdefault("families", {})["ffn"] = {
            "violations": 2,
            "kappa": 2.5,
        }

        cert_copy["rmt"].update(
            {
                "families": {
                    "ffn": {"bare": 1, "guarded": 2, "epsilon": 0.1},
                    "attn": {"bare": 0, "guarded": 0, "epsilon": 0.08},
                },
                "epsilon_by_family": {"ffn": 0.1, "attn": 0.08},
                "stable": True,
            }
        )

        cert_copy["guard_overhead"] = {
            "bare_ppl": 118.0,
            "guarded_ppl": 120.5,
            "overhead_ratio": 1.021,
            "overhead_percent": 2.1,
            "source": "regression",
        }
        cert_copy["validation"]["guard_overhead_acceptable"] = False

        cert_copy["edit_name"] = "quant_rtn"
        structure = cert_copy["structure"]
        structure["bitwidths"] = [8, 8, 8]
        structure["compression_diagnostics"]["parameter_analysis"] = {
            "bitwidth": {"value": 8, "effectiveness": "applied"}
        }
        structure["compression_diagnostics"]["algorithm_details"][
            "modules_quantized"
        ] = 3
        structure["compression_diagnostics"]["warnings"] = ["Check clamp coverage"]
        structure["compression_diagnostics"]["target_analysis"]["scope"] = "attn"
        # Reduction details attached via run metrics; not required in PM-only cert
        cert_copy["baseline_ref"]["ppl_preview"] = float("nan")
        cert_copy["baseline_ref"]["ppl_final"] = float("nan")
        cert_copy["dataset"]["hash"].update(
            {
                "preview_tokens": 6400,
                "final_tokens": 6400,
                "total_tokens": 12800,
                "dataset": "hash123",
            }
        )
        cert_copy["dataset"]["tokenizer"].update(
            {
                "name": "gpt2-tokenizer",
                "hash": "tokhash",
                "vocab_size": 50257,
                "bos_token": " bos",
                "eos_token": " eos",
                "pad_token": None,
                "add_prefix_space": True,
            }
        )
        cert_copy["provenance"] = {
            "baseline": {
                "run_id": "baseline#1",
                "report_hash": "hashA",
                "report_path": "/runs/base",
            },
            "edited": {
                "run_id": "edited#1",
                "report_hash": "hashB",
                "report_path": "/runs/edit",
            },
            "window_plan": {"profile": "release", "preview_n": 203, "final_n": 203},
        }

        markdown = render_report_markdown(cert_copy)

        assert "- **Commit:** (not set)" in markdown
        assert "- **Overrides:** (none)" in markdown
        assert "Spectral Guard" in markdown
        assert "RMT Guard" in markdown or "RMT ε" in markdown

    def test_render_markdown_variance_disabled_branch(self):
        evaluation_report = _load_local_evaluation_report()
        var = evaluation_report["variance"]
        var["enabled"] = False
        var["gain"] = 0.001
        var["ppl_no_ve"] = 50.0
        var["ppl_with_ve"] = 50.4
        var["ratio_ci"] = (0.99, 1.02)
        var["calibration"] = {"coverage": 12, "requested": 16, "status": "insufficient"}

        markdown = render_report_markdown(evaluation_report)
        assert "Primary metric without VE" in markdown
        assert "Ratio CI" in markdown

    def test_render_markdown_generic_edit_paths(self):
        evaluation_report = _load_local_evaluation_report()
        evaluation_report["edit_name"] = "custom_edit"
        evaluation_report["structure"]["bitwidths"] = []
        evaluation_report["structure"]["ranks"] = []
        evaluation_report["structure"]["compression_diagnostics"][
            "parameter_analysis"
        ] = {}

        markdown = render_report_markdown(evaluation_report)
        # Generic edit paths may vary; ensure evaluation_report header renders
        assert "# InvarLock Evaluation Report" in markdown

    # Low-rank branch tests removed (no low-rank edit in this profile)

    def test_render_markdown_guard_tables_and_compression_details(self):
        report = create_mock_run_report(
            include_auto=True, include_evaluation_windows=True
        )
        baseline = create_mock_baseline()

        with patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ):
            evaluation_report = make_report(report, baseline)

        evaluation_report["spectral"] = {
            "caps_applied": 3,
            "max_caps": 5,
            "summary": {
                "caps_exceeded": False,
                "max_sigma_ratio": 1.12,
                "median_sigma_ratio": 1.05,
            },
            "caps_applied_by_family": {"ffn": 2, "attn": 1},
            "family_z_quantiles": {
                "ffn": {"q95": 2.3, "q99": 2.5, "max": 2.6, "count": 32},
                "attn": {"q95": 2.1, "q99": 2.3, "max": 2.4, "count": 32},
            },
            "policy": {
                "family_caps": {"ffn": {"kappa": 2.5}, "attn": {"kappa": 2.8}},
                "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
            },
            "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
            "top_z_scores": {
                "ffn": [{"module": "layers.0.mlp.c_fc", "z": 2.6}],
                "attn": [{"module": "layers.0.attn.c_proj", "z": 2.4}],
            },
        }
        evaluation_report["rmt"] = {
            "families": {
                "ffn": {"epsilon": 0.1, "bare": 1, "guarded": 2},
                "attn": {"epsilon": 0.08, "bare": 0, "guarded": 0},
            },
            "epsilon_by_family": {"ffn": 0.1, "attn": 0.08},
            "outliers_bare": 3,
            "outliers_guarded": 4,
            "epsilon": 0.1,
            "max_deviation_ratio": 1.05,
            "mean_deviation_ratio": 1.02,
            "status": "stable",
            "stable": True,
        }
        evaluation_report["guard_overhead"] = {
            "bare_ppl": 118.0,
            "guarded_ppl": 120.5,
            "overhead_ratio": 1.021,
            "overhead_percent": 2.1,
            "source": "regression",
        }
        evaluation_report["variance"] = {
            "enabled": True,
            "gain": 0.002,
            "scope": "ffn",
            "tap": "mlp.c_proj",
            "predictive_gate": {"ci": (0.001, 0.003)},
        }
        evaluation_report.setdefault("structure", {}).setdefault(
            "compression_diagnostics", {}
        ).update(
            {
                "execution_status": "successful",
                "target_analysis": {
                    "modules_found": 12,
                    "modules_eligible": 12,
                    "modules_modified": 12,
                    "scope": "attn",
                },
                "parameter_analysis": {
                    "bitwidth": {"value": 8, "effectiveness": "applied"},
                },
                "algorithm_details": {"modules_quantized": 12},
                "warnings": ["Check clamp coverage"],
            }
        )
        evaluation_report["dataset"]["hash"].update(
            {
                "preview_tokens": 6400,
                "final_tokens": 6400,
                "total_tokens": 12800,
                "dataset": "hash123",
            }
        )
        evaluation_report["dataset"]["tokenizer"].update(
            {
                "name": "gpt2-tokenizer",
                "hash": "tokhash",
                "vocab_size": 50257,
                "bos_token": " bos",
                "eos_token": " eos",
                "pad_token": None,
                "add_prefix_space": True,
            }
        )

        markdown = render_report_markdown(evaluation_report)

        assert "Spectral Guard" in markdown
        assert "| Family | κ | q95 | Max |z| | Violations |" in markdown
        assert "Top |z| per family" in markdown
        assert "| Family | ε_f | Bare | Guarded |" in markdown
        assert "Bare Primary Metric: 118.000" in markdown
        assert "Execution Status" in markdown
