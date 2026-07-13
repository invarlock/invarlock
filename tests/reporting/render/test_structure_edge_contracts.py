from __future__ import annotations

from invarlock.reporting.rendering import structure


def test_generated_at_falls_back_to_policy_provenance() -> None:
    assert structure.get_generated_at({"artifacts": {"generated_at": "now"}}) == "now"
    assert (
        structure.get_generated_at({"policy_provenance": {"resolved_at": "then"}})
        == "then"
    )
    assert structure.get_generated_at({}) == "(not recorded)"


def test_structural_changes_render_quantized_target_counts_and_fallbacks() -> None:
    lines: list[str] = []
    structure.append_structural_changes_section(
        lines,
        {
            "edit_name": "quant_rtn",
            "structure": {
                "params_changed": 10,
                "layers_modified": 2,
                "bitwidths": [4, 4, 4, 4],
                "compression_diagnostics": {
                    "target_analysis": {
                        "modules_eligible": 5,
                        "modules_modified": 4,
                    }
                },
            },
        },
    )
    assert "| Linear Modules Quantized | 4 of 5 targeted |" in lines

    fallback: list[str] = []
    structure.append_structural_changes_section(
        fallback,
        {
            "edit_name": "quant_rtn",
            "structure": {
                "params_changed": 1,
                "layers_modified": 2,
                "bitwidths": [4, 4, 4, 4],
                "compression_diagnostics": object(),
            },
        },
    )
    assert any("2 per block × 2 blocks" in line for line in fallback)


def test_structural_changes_are_fail_safe_for_malformed_counts() -> None:
    lines: list[str] = []
    structure.append_structural_changes_section(
        lines, {"structure": {"params_changed": object()}}
    )
    assert lines == []


def test_compression_diagnostics_render_all_evidence_shapes() -> None:
    noop: list[str] = []
    structure.append_compression_diagnostics_section(noop, {"edit_name": "noop"})
    assert "Not applicable (no parameters modified)." in noop

    lines: list[str] = []
    structure.append_compression_diagnostics_section(
        lines,
        {
            "edit_name": "quant_rtn",
            "structure": {
                "compression_diagnostics": {
                    "execution_status": "failed",
                    "target_analysis": {
                        "modules_found": 3,
                        "modules_eligible": object(),
                        "modules_modified": 2,
                        "scope": "all",
                    },
                    "parameter_analysis": {
                        "bits": {"value": 4, "effectiveness": "measured"},
                        "plain": "value",
                    },
                    "algorithm_details": {"method": "RTN"},
                    "warnings": ["warning"],
                }
            },
        },
    )
    assert "**Execution Status:** ❌ FAILED" in lines
    assert "- **bits:** 4 (measured)" in lines
    assert "- **plain:** value" in lines
    assert "- **method:** RTN" in lines
    assert "- warning" in lines


def test_moe_observability_renders_only_typed_measurements() -> None:
    lines: list[str] = []
    structure.append_moe_observability_section(lines, {"moe": "bad"})
    assert lines == []
    structure.append_moe_observability_section(
        lines,
        {
            "moe": {
                "top_k": 2,
                "utilization_count": 4,
                "utilization_mean": 0.5,
                "delta_router_entropy": -0.1,
                "delta_load_balance_loss": "bad",
            }
        },
    )
    assert "- **Utilization:** N=4; mean=0.500" in lines
    assert "- **Δ router_entropy:** -0.1000" in lines


def test_inference_and_variance_appendices_cover_enabled_and_disabled_paths() -> None:
    appendix: list[str] = []
    structure._append_inference_diagnostics_section(
        appendix,
        {
            "structure": {
                "compression_diagnostics": {
                    "inferred": {"rank": True},
                    "inference_source": {"rank": "metadata"},
                    "inference_log": ["loaded"],
                }
            }
        },
    )
    assert "  - rank: yes" in appendix
    assert "  - rank: metadata" in appendix
    assert "  - loaded" in appendix

    enabled: list[str] = []
    structure._append_variance_guard_appendix(
        enabled,
        {
            "variance": {
                "enabled": True,
                "gain": 0.1,
                "ratio_ci": [0.9, 1.1],
                "calibration": {
                    "coverage": 9,
                    "requested": 10,
                    "status": "partial",
                },
            }
        },
    )
    assert "- **Gain:** 0.100" in enabled
    assert "- **Calibration:** 9/10 windows (partial)" in enabled

    disabled: list[str] = []
    structure._append_variance_guard_appendix(
        disabled,
        {
            "variance": {"enabled": False},
            "policies": {"variance": {"min_effect_lognll": 0.01}},
        },
    )
    assert any("0.01" in line for line in disabled)


def test_structural_changes_render_bitwidth_only_and_non_quantized_edits() -> None:
    quantized: list[str] = []
    structure.append_structural_changes_section(
        quantized,
        {
            "edit_name": "quant_rtn",
            "structure": {"params_changed": 0, "layers_modified": 0, "bitwidths": [4]},
        },
    )
    assert "| Linear Modules Quantized | 1 |" in quantized

    dense: list[str] = []
    structure.append_structural_changes_section(
        dense,
        {
            "edit_name": "fine_tune",
            "structure": {"params_changed": 2, "layers_modified": 1},
        },
    )
    assert "| Layers Modified | 1 |" in dense


def test_moe_observability_covers_optional_measurement_branches() -> None:
    lines: list[str] = []
    structure.append_moe_observability_section(
        lines,
        {
            "moe": {
                "top_k": 2,
                "capacity_factor": 1.25,
                "expert_drop_rate": 0.01,
                "utilization_count": None,
                "utilization_mean": 0.75,
                "delta_load_balance_loss": -0.2,
                "delta_utilization_mean": 0.1,
            }
        },
    )
    rendered = "\n".join(lines)
    assert "capacity_factor" in rendered
    assert "mean=0.750" in rendered
    assert "Δ load_balance_loss" in rendered
    assert "Δ utilization mean" in rendered


def test_appendix_sections_render_artifacts_and_disabled_measurements() -> None:
    lines: list[str] = []
    appendix: list[str] = []
    structure.append_appendix_sections(
        lines,
        appendix,
        {
            "artifacts": {
                "events_path": "events.jsonl",
                "report_path": "report.json",
                "generated_at": "now",
            },
            "variance": {
                "enabled": False,
                "ppl_no_ve": 2.0,
                "ppl_with_ve": 1.9,
                "gain": 0.1,
                "ratio_ci": [0.9, 1.0],
            },
        },
    )
    rendered = "\n".join(lines)
    assert "## Appendix" in rendered
    assert "events.jsonl" in rendered
    assert "Gain (insufficient)" in rendered
    assert "Ratio CI" in rendered


def test_variance_appendix_tolerates_non_mapping_payload() -> None:
    lines: list[str] = []
    structure._append_variance_guard_appendix(
        lines,
        {"variance": "bad", "policies": "bad"},
    )
    assert "- **Enabled:** No" in lines


def test_structure_branch_permutations_cover_empty_provenance_and_partial_inference() -> (
    None
):
    assert (
        structure.get_generated_at({"policy_provenance": {"resolved_at": ""}})
        == "(not recorded)"
    )

    flags_only: list[str] = []
    structure._append_inference_diagnostics_section(
        flags_only,
        {"structure": {"compression_diagnostics": {"inferred": {"scope": False}}}},
    )
    assert "  - scope: no" in flags_only

    count_only: list[str] = []
    structure.append_moe_observability_section(
        count_only,
        {"moe": {"utilization_count": 2}},
    )
    assert "- **Utilization:** N=2" in count_only
