"""
Comprehensive test coverage for evaluation-report owner helpers.

Tests report assembly, validation, and rendering helper behavior.
"""

import copy
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from invarlock import __version__ as INVARLOCK_VERSION
from invarlock.reporting.dataset_hashing import (
    _compute_actual_window_hashes,
    _extract_dataset_info,
    compute_window_hashes,
)
from invarlock.reporting.guards_invariants import (
    _extract_invariants,
)
from invarlock.reporting.guards_rmt import (
    _extract_rmt_analysis,
)
from invarlock.reporting.guards_spectral import (
    _extract_spectral_analysis,
)
from invarlock.reporting.guards_variance import (
    _extract_variance_analysis,
)
from invarlock.reporting.policy_utils import (
    _build_resolved_policies,
    _compute_policy_digest,
    _compute_variance_policy_digest,
    _extract_effective_policies,
    _extract_policy_overrides,
    _format_epsilon_map,
    _format_family_caps,
    _resolve_policy_tier,
)
from invarlock.reporting.render_markdown import (
    _get_window_plan_summary,
    render_report_markdown,
)
from invarlock.reporting.report_builder_support import (
    extract_report_meta as _extract_report_meta,
)
from invarlock.reporting.report_console import (
    compute_report_hash as _compute_report_hash,
)
from invarlock.reporting.report_edit_summary import (
    analyze_bitwidth_map as _analyze_bitwidth_map,
)
from invarlock.reporting.report_edit_summary import (
    compute_savings_summary as _compute_savings_summary,
)
from invarlock.reporting.report_edit_summary import (
    extract_rank_information as _extract_rank_information,
)
from invarlock.reporting.report_edit_summary import (
    extract_structural_deltas as _extract_structural_deltas,
)
from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_normalization import (
    _generate_run_id,
)
from invarlock.reporting.report_normalization import (
    normalize_baseline as _normalize_baseline,
)
from invarlock.reporting.report_overhead import (
    prepare_guard_overhead_section as _prepare_guard_overhead_section,
)
from invarlock.reporting.report_provenance import (
    compute_report_digest as _compute_report_digest,
)
from invarlock.reporting.report_schema import REPORT_SCHEMA_VERSION, validate_report
from invarlock.reporting.report_validation import (
    compute_validation_flags as _compute_validation_flags,
)
from invarlock.reporting.utils import (
    _coerce_int,
    _coerce_interval,
    _infer_scope_from_modules,
    _iter_guard_entries,
    _pair_logloss_windows,
    _sanitize_seed_bundle,
)

__all__ = [
    "Any",
    "Mock",
    "Path",
    "SimpleNamespace",
    "INVARLOCK_VERSION",
    "REPORT_SCHEMA_VERSION",
    "_analyze_bitwidth_map",
    "_build_resolved_policies",
    "_build_spectral_guard_with_z_scores",
    "_coerce_int",
    "_coerce_interval",
    "_compute_actual_window_hashes",
    "_compute_policy_digest",
    "_compute_report_digest",
    "_compute_report_hash",
    "_compute_savings_summary",
    "_compute_validation_flags",
    "_compute_variance_policy_digest",
    "_extract_dataset_info",
    "_extract_effective_policies",
    "_extract_invariants",
    "_extract_policy_overrides",
    "_extract_rank_information",
    "_extract_rmt_analysis",
    "_extract_spectral_analysis",
    "_extract_structural_deltas",
    "_extract_variance_analysis",
    "_format_epsilon_map",
    "_format_family_caps",
    "_generate_run_id",
    "_get_window_plan_summary",
    "_infer_scope_from_modules",
    "_iter_guard_entries",
    "_load_local_evaluation_report",
    "_normalize_baseline",
    "_pair_logloss_windows",
    "_prepare_guard_overhead_section",
    "_resolve_policy_tier",
    "_sanitize_seed_bundle",
    "_extract_report_meta",
    "compute_window_hashes",
    "copy",
    "create_mock_baseline",
    "create_mock_run_report",
    "make_report",
    "math",
    "patch",
    "pytest",
    "render_report_markdown",
    "validate_report",
]


def _load_local_evaluation_report() -> dict[str, Any]:
    """Construct a representative evaluation_report locally for rendering tests.

    Avoids relying on repo-level sample artifacts under reports/.
    """
    report = create_mock_run_report(include_guards=True, include_auto=True)
    baseline = create_mock_baseline()
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        cert = make_report(report, baseline)
    # Ensure expected branches exist for rendering variations
    cert.setdefault(
        "variance",
        {
            "enabled": True,
            "gain": 1.0,
            "ratio_ci": (0.99, 1.01),
            "ppl_no_ve": None,
            "ppl_with_ve": None,
            "calibration": {"coverage": 0, "requested": 0, "status": "ok"},
        },
    )
    cert.setdefault("spectral", {"caps_applied": 0, "summary": {}})
    cert.setdefault("rmt", {"families": {}, "stable": True})
    cert.setdefault(
        "invariants", {"summary": {"warning_violations": 0}, "failures": []}
    )
    cert.setdefault("policy_provenance", {"overrides": []})
    return cert


def create_mock_run_report(
    model_id: str = "test-model",
    ppl_final: float = 10.5,
    include_guards: bool = True,
    include_auto: bool = False,
    include_evaluation_windows: bool = False,
) -> dict[str, Any]:
    """Create a mock RunReport for testing."""
    report = {
        "meta": {
            "model_id": model_id,
            "adapter": "hf_causal",
            "device": "cpu",
            "ts": "2023-10-01T12:00:00",
            "commit": "abcd1234567890abcdef",
            "seed": 42,
            "seeds": {"python": 42, "numpy": 42, "torch": 42},
            "plugins": {
                "adapter": {
                    "name": "hf_causal",
                    "module": "invarlock.adapters.hf_causal",
                    "version": INVARLOCK_VERSION,
                    "available": True,
                    "entry_point": "hf_causal",
                    "entry_point_group": "invarlock.adapters",
                },
                "edit": {
                    "name": "structured",
                    "module": "invarlock.edits.structured",
                    "version": INVARLOCK_VERSION,
                    "available": True,
                    "entry_point": "structured",
                    "entry_point_group": "invarlock.edits",
                },
                "guards": [
                    {
                        "name": "spectral",
                        "module": "invarlock.guards.spectral",
                        "version": INVARLOCK_VERSION,
                        "available": True,
                        "entry_point": "spectral",
                        "entry_point_group": "invarlock.guards",
                    }
                ],
            },
        },
        "data": {
            "dataset": "wikitext",
            "split": "test",
            "seq_len": 1024,
            "stride": 512,
            "preview_n": 10,
            "final_n": 50,
        },
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 9.8,
                "final": ppl_final,
                "ratio_vs_baseline": ppl_final / 9.0,
            },
            "logloss_delta": math.log(ppl_final) - math.log(9.8),
            "logloss_delta_ci": (
                math.log(ppl_final) - math.log(9.8) - 0.05,
                math.log(ppl_final) - math.log(9.8) + 0.05,
            ),
            "paired_delta_summary": {
                "mean": math.log(ppl_final) - math.log(9.8),
                "std": 0.01,
            },
            "invariants": {
                "weight_norm": {"passed": True},
                "activation_range": {"passed": True},
            },
            "spectral": {"sigma_ratios": [1.1, 1.2, 0.9, 1.05], "stable": True},
            "rmt": {"outliers": 2, "stable": True},
        },
        "edit": {
            "name": "structured",
            "deltas": {
                "params_changed": 1000,
                "heads_pruned": 5,
                "neurons_pruned": 100,
                "layers_modified": 3,
                "sparsity": 0.1,
            },
        },
        "artifacts": {
            "events_path": "/path/to/events.jsonl",
            "logs_path": "/path/to/logs.txt",
        },
    }

    if include_auto:
        report["meta"]["auto"] = {
            "tier": "aggressive",
            "probes_used": 5,
            "target_pm_ratio": 1.5,
        }

    if include_guards:
        report["guards"] = [
            {
                "name": "spectral",
                "policy": {
                    "sigma_quantile": 0.95,
                    "deadband": 0.1,
                    "scope": "ffn",
                    "max_caps": 5,
                    "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
                },
                "actions": ["cap_layer_2", "cap_layer_5"],
                "metrics": {
                    "violations_detected": 2,
                    "deadband": 0.1,
                    "max_caps": 5,
                    "caps_applied": 2,
                    "caps_exceeded": False,
                    "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
                },
            },
            {
                "name": "rmt",
                "policy": {"threshold": 1.5, "deadband": 0.1},
                "actions": [],
            },
            {"name": "variance", "policy": {"gain": 2.0}, "metrics": {"gain": 1.8}},
        ]

    if include_evaluation_windows:
        report["evaluation_windows"] = {
            "preview": {"input_ids": [[1, 2, 3, 4], [5, 6, 7, 8]]},
            "final": {"input_ids": [[9, 10, 11, 12], [13, 14, 15, 16]]},
        }

    return report


def create_mock_baseline(
    model_id: str = "test-model", ppl_final: float = 9.0, schema_type: str = "runreport"
) -> dict[str, Any]:
    """Create a mock baseline for testing."""
    if schema_type == "baseline-v1":
        return {
            "schema_version": "baseline-v1",
            "meta": {"model_id": model_id, "commit_sha": "baseline123456789"},
            "metrics": {
                "ppl_final": ppl_final,
                "primary_metric": {"kind": "ppl_causal", "final": ppl_final},
            },
            "spectral_base": {"sigma_ratios": [1.0, 1.0, 1.0]},
            "rmt_base": {"outliers": 1},
            "invariants": {"weight_norm": {"passed": True}},
        }
    elif schema_type == "runreport":
        return create_mock_run_report(model_id=model_id, ppl_final=ppl_final)
    else:
        # Normalized format
        return {
            "run_id": "normalized123",
            "model_id": model_id,
            "ppl_final": ppl_final,
            "metrics": {"primary_metric": {"kind": "ppl_causal", "final": ppl_final}},
            "spectral": {"sigma_ratios": [1.0, 1.0]},
            "rmt": {"outliers": 1},
            "invariants": {"all_passed": True},
        }


def _build_spectral_guard_with_z_scores() -> dict[str, Any]:
    """Return a spectral guard entry populated with module z-scores."""
    module_family_map = {
        "ffn.0.w1": "ffn",
        "ffn.0.w2": "ffn",
        "ffn.1.w1": "ffn",
        "ffn.1.w2": "ffn",
        "attn.0.wq": "attn",
        "attn.0.wk": "attn",
        "attn.0.wv": "attn",
        "attn.0.wo": "attn",
    }

    final_z_scores = {
        "ffn.0.w1": 1.0,
        "ffn.0.w2": 3.0,
        "ffn.1.w1": 2.0,
        "ffn.1.w2": 2.5,
        "attn.0.wq": 2.2,
        "attn.0.wk": 3.3,
        "attn.0.wv": 1.5,
        "attn.0.wo": 2.7,
    }

    return {
        "name": "spectral",
        "policy": {
            "sigma_quantile": 0.95,
            "deadband": 0.1,
            "scope": "all",
            "max_caps": 5,
            "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
            "max_spectral_norm": 9.9,
        },
        "metrics": {
            "families": {
                "ffn": {"violations": 2},
                "attn": {"violations": 1},
            },
            "family_z_summary": {
                "ffn": {"violations": 2, "count": 4, "max": 3.0},
                "attn": {"violations": 1, "count": 4, "max": 3.3},
            },
            "max_caps": 5,
            "caps_exceeded": False,
            "modules_checked": 8,
        },
        "violations": [
            {
                "module": "ffn.0.w2",
                "family": "ffn",
                "kappa": 2.5,
                "z_score": 3.0,
                "severity": "warn",
            },
            {
                "module": "attn.0.wk",
                "family": "attn",
                "kappa": 2.8,
                "z_score": 3.3,
            },
        ],
        "final_z_scores": final_z_scores,
        "module_family_map": module_family_map,
    }
