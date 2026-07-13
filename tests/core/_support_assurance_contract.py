from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

from invarlock.core.assurance_contract import (
    CANONICAL_GUARD_CHAIN,
    build_assurance_section,
)
from invarlock.core.builtin_plugin_catalog import builtin_plugin_specs
from tests.core._support_guard_metric_impact import bind_guard_metric_impact_evidence


def _plugin_metadata(plugin_type: str, name: str) -> dict[str, Any]:
    spec = next(item for item in builtin_plugin_specs(plugin_type) if item.name == name)
    return {
        "name": spec.name,
        "type": plugin_type,
        "module": spec.module,
        "package": "invarlock",
        "available": True,
        "support_tier": spec.support_tier,
        "strict_assurance_allowed": spec.strict_assurance_allowed,
    }


def bind_raw_guard_evidence(report: dict[str, Any]) -> dict[str, Any]:
    """Attach canonical retained observations to a strict report fixture."""

    spectral_contract = report.get("spectral", {}).get("measurement_contract") or {
        "kind": "spectral_norm_power_iter",
        "version": 1,
    }
    rmt_contract = report.get("rmt", {}).get("measurement_contract") or {
        "kind": "activation_edge_risk",
        "version": 1,
    }
    for entry in report.get("guards", []):
        name = entry.get("name")
        if name == "invariants":
            checks = {"parameter_count": 2, "structure_hash": "stable"}
            entry.update(
                {
                    "metrics": {
                        "checks_performed": 2,
                        "violations_found": 0,
                        "fatal_violations": 0,
                        "warning_violations": 0,
                    },
                    "policy": {"strict_mode": True, "on_fail": "block"},
                    "details": {
                        "baseline_checks": checks,
                        "current_checks": dict(checks),
                    },
                }
            )
        elif name == "spectral":
            family_caps = {
                "ffn": {"kappa": 3.0},
                "other": {"kappa": 3.0},
            }
            multiple_testing = {"method": "bonferroni", "alpha": 0.05, "m": 4}
            entry.update(
                {
                    "metrics": {
                        "modules_analyzed": 2,
                        "baseline_modules": 2,
                        "violations_detected": 0,
                        "candidate_violations_detected": 0,
                        "candidate_budgeted_violations": 0,
                        "budgeted_violations": 0,
                        "fatal_violations": 0,
                        "caps_applied": 0,
                        "max_caps": 5,
                        "caps_exceeded": False,
                        "selected_budgeted_findings": 0,
                        "cap_budget_exceeded": False,
                        "corrections_attempted": 0,
                        "corrections_applied": 0,
                        "correction_policy_result": "no_selected_findings",
                        "identity_changed_modules": [],
                        "measurement_exclusions": [],
                        "discovery_errors": [],
                        "max_spectral_norm": 1.0,
                        "mean_spectral_norm": 1.0,
                        "family_caps": copy.deepcopy(family_caps),
                        "multiple_testing": copy.deepcopy(multiple_testing),
                        "multiple_testing_selection": {
                            "method": "bonferroni",
                            "alpha": 0.05,
                            "m": 4,
                            "families_tested": [],
                            "families_selected": [],
                            "family_pvalues": {},
                            "family_max_abs_z": {},
                            "family_violation_counts": {},
                            "default_selected_without_pvalue": 0,
                        },
                        "measurement_contract": spectral_contract,
                    },
                    "policy": {
                        "max_caps": 5,
                        "deadband": 0.1,
                        "max_spectral_norm": None,
                        "family_caps": family_caps,
                        "multiple_testing": multiple_testing,
                        "degeneracy": {
                            "enabled": False,
                            "stable_rank": {"warn_ratio": 0.5, "fatal_ratio": 0.25},
                            "norm_collapse": {"warn_ratio": 0.25, "fatal_ratio": 0.1},
                        },
                        "correction_enabled": True,
                        "correction_cap_ratio": 2.0,
                    },
                    "baseline_metrics": {
                        "module_sigmas": {"layer.0": 1.0, "layer.1": 1.0},
                        "family_stats": {
                            "ffn": {
                                "count": 2,
                                "mean": 1.0,
                                "std": 0.0,
                                "min": 1.0,
                                "max": 1.0,
                            }
                        },
                        "baseline_degeneracy": {},
                        "measurement_contract": spectral_contract,
                    },
                    "final_metrics": {"layer.0": 1.0, "layer.1": 1.0},
                    "final_z_scores": {"layer.0": 0.0, "layer.1": 0.0},
                    "module_family_map": {"layer.0": "ffn", "layer.1": "ffn"},
                    "final_degeneracy": {},
                    "measurement_inventory": {
                        phase: {
                            "schema_version": 1,
                            "phase": phase,
                            "enumerated_modules": ["layer.0", "layer.1"],
                            "eligible_modules": ["layer.0", "layer.1"],
                            "measured_modules": ["layer.0", "layer.1"],
                            "excluded_modules": [],
                            "identity_changed_modules": [],
                            "discovery_errors": [],
                            "enumerated_count": 2,
                            "eligible_count": 2,
                            "measured_count": 2,
                            "excluded_count": 0,
                            "identity_changed_count": 0,
                            "discovery_error_count": 0,
                        }
                        for phase in ("prepare", "validate")
                    },
                    "correction_ledger": {
                        "schema_version": 1,
                        "phase": "validate",
                        "correction_enabled": True,
                        "correction_cap_ratio": 2.0,
                        "pre_correction_metrics": {
                            "layer.0": 1.0,
                            "layer.1": 1.0,
                        },
                        "pre_correction_z_scores": {
                            "layer.0": 0.0,
                            "layer.1": 0.0,
                        },
                        "pre_correction_degeneracy": {},
                        "multiple_testing_selection": {
                            "method": "bonferroni",
                            "alpha": 0.05,
                            "m": 4,
                            "families_tested": [],
                            "families_selected": [],
                            "family_pvalues": {},
                            "family_max_abs_z": {},
                            "family_violation_counts": {},
                            "default_selected_without_pvalue": 0,
                        },
                        "selected_findings": [],
                        "corrections": [],
                        "policy_result": "no_selected_findings",
                        "post_correction_metrics": {
                            "layer.0": 1.0,
                            "layer.1": 1.0,
                        },
                    },
                }
            )
        elif name == "rmt":
            entry.update(
                {
                    "metrics": {
                        "stable": True,
                        "edge_risk_by_family_base": {"ffn": 1.0},
                        "edge_risk_by_family": {"ffn": 1.0},
                        "edge_risk_by_module_base": {"layer.0.mlp": 1.0},
                        "edge_risk_by_module": {"layer.0.mlp": 1.0},
                        "module_family_map": {"layer.0.mlp": "ffn"},
                        "epsilon_by_family": {"ffn": 0.01},
                        "epsilon_violations": [],
                        "measurement_contract": rmt_contract,
                    },
                    "policy": {
                        "epsilon_default": 0.01,
                        "epsilon_by_family": {"ffn": 0.01},
                    },
                    "details": {
                        "baseline_edge_risk_by_family": {"ffn": 1.0},
                        "current_edge_risk_by_family": {"ffn": 1.0},
                    },
                }
            )
    spectral = report.get("spectral")
    if isinstance(spectral, dict):
        summary = spectral.get("summary")
        if isinstance(summary, dict):
            summary.setdefault("caps_applied", spectral.get("caps_applied", 0))
    bind_guard_metric_impact_evidence(report)
    return report


def bind_noop_variance_evidence(report: dict[str, Any]) -> dict[str, Any]:
    variance_guard = next(
        entry for entry in report["guards"] if entry["name"] == "variance"
    )
    meta = report.setdefault("meta", {})
    meta.setdefault("model_id", "strict-model")
    meta.setdefault("seed", 123)
    meta.setdefault("tokenizer_hash", "strict-tokenizer")
    dataset = report.setdefault("dataset", {})
    dataset_hashes = dataset.setdefault("hash", {})
    dataset_hashes.setdefault("dataset", "strict-dataset")
    dataset_tokenizer = dataset.setdefault("tokenizer", {})
    dataset_tokenizer.setdefault("hash", meta["tokenizer_hash"])
    existing_windows = report.get("evaluation_windows")
    if (
        isinstance(existing_windows, dict)
        and isinstance(existing_windows.get("preview"), dict)
        and isinstance(existing_windows.get("final"), dict)
        and isinstance(
            existing_windows["preview"].get("window_ids")
            or existing_windows["preview"].get("example_ids"),
            list,
        )
        and isinstance(
            existing_windows["final"].get("window_ids")
            or existing_windows["final"].get("example_ids"),
            list,
        )
    ):
        evaluation_windows = copy.deepcopy(existing_windows)
    else:
        evaluation_windows = {
            "preview": {"window_ids": list(range(4))},
            "final": {"window_ids": list(range(4, 8))},
        }
        report["evaluation_windows"] = copy.deepcopy(evaluation_windows)
    preview_ids = evaluation_windows["preview"].get("window_ids") or evaluation_windows[
        "preview"
    ].get("example_ids")
    final_ids = evaluation_windows["final"].get("window_ids") or evaluation_windows[
        "final"
    ].get("example_ids")
    report_pairing_ids = [
        *(f"preview::{index}" for index in preview_ids),
        *(f"final::{index}" for index in final_ids),
    ]
    pairing_digest = hashlib.blake2s(
        "||".join(report_pairing_ids).encode("utf-8"), digest_size=16
    ).hexdigest()
    window_ids = list(report_pairing_ids[:8])
    consumed_pairing_digest = hashlib.blake2s(
        "||".join(window_ids).encode("utf-8"), digest_size=16
    ).hexdigest()
    condition_common = {
        "tag": "post_edit",
        "window_ids": window_ids,
        "window_count": 8,
        "target_fingerprint": "strict-target",
        "pairing_digest": pairing_digest,
        "consumed_pairing_digest": consumed_pairing_digest,
        "dataset_hash": dataset_hashes["dataset"],
        "tokenizer_hash": meta["tokenizer_hash"],
        "model_id": meta["model_id"],
        "run_seed": meta["seed"],
    }
    policy = {
        "seed": 123,
        "mode": "ci",
        "alpha": 0.05,
        "predictive_gate": True,
        "calibration": {"windows": 8, "min_coverage": 6, "seed": 123},
    }
    report.setdefault("resolved_policy", {})["variance"] = copy.deepcopy(policy)
    provenance = {
        "condition_a": {
            **condition_common,
            "mode": "edited_no_ve",
            "status": "evaluated",
        },
        "condition_b": {
            **condition_common,
            "mode": "virtual_ve",
            "status": "no_scales",
        },
    }
    point_estimates = {
        "tag": "post_edit",
        "ppl_no_ve": 100.0,
        "ppl_with_ve": 100.0,
        "coverage": 8,
    }
    measurement_arm = {
        "ppl": [100.0] * 8,
        "log_loss": [math.log(100.0)] * 8,
        "token_counts": [16] * 8,
    }
    measurements = {
        "window_ids": list(window_ids),
        "condition_a": copy.deepcopy(measurement_arm),
        "condition_b": copy.deepcopy(measurement_arm),
        "ratio_bootstrap": {
            "method": "percentile_mean_ppl_ratio",
            "replicates": 500,
            "alpha": 0.05,
            "seed": 123,
        },
        "delta_log_bootstrap": {
            "method": "bca_paired_delta_log",
            "replicates": 500,
            "alpha": 0.05,
            "seed": 334,
            "weights": "condition_a_token_counts",
        },
        "ratio_ci": [1.0, 1.0],
        "delta_log_ci": [0.0, 0.0],
    }
    report["variance"].update(
        {
            "ve_enabled_during_validation": False,
            "subject_restored_after_ab": True,
            "met_threshold": False,
            "gain": 0.0,
            "ppl_no_ve": 100.0,
            "ppl_with_ve": 100.0,
            "ratio_ci": [1.0, 1.0],
            "policy": copy.deepcopy(policy),
            "ab_test": {
                "seed": 123,
                "windows_used": 8,
                "provenance": {
                    **copy.deepcopy(provenance),
                    "window_ids": list(window_ids),
                },
                "point_estimates": copy.deepcopy(point_estimates),
                "measurements": copy.deepcopy(measurements),
            },
        }
    )
    report["variance"]["calibration"].update(requested=8, seed=123)
    variance_guard["metrics"] = {
        "ve_enabled": False,
        "ve_enabled_during_validation": False,
        "subject_restored_after_ab": True,
        "met_threshold": False,
        "ab_gain": 0.0,
        "ppl_no_ve": 100.0,
        "ppl_with_ve": 100.0,
        "ratio_ci": [1.0, 1.0],
        "ab_seed_used": 123,
        "ab_windows_used": 8,
        "ab_provenance": copy.deepcopy(provenance),
        "ab_point_estimates": copy.deepcopy(point_estimates),
        "ab_measurements": copy.deepcopy(measurements),
        "mode": policy["mode"],
        "monitor_only": False,
        "predictive_gate": dict(report["variance"]["predictive_gate"]),
        "calibration": dict(report["variance"]["calibration"]),
    }
    variance_guard["policy"] = copy.deepcopy(policy)
    variance_guard["details"] = {
        "ve_tested": False,
        "ve_applied": False,
        "subject_restored_after_ab": True,
        "policy": copy.deepcopy(policy),
        "stats": {
            "ab_provenance": copy.deepcopy(provenance),
            "ab_point_estimates": copy.deepcopy(point_estimates),
            "ab_measurements": copy.deepcopy(measurements),
            "predictive_gate": dict(report["variance"]["predictive_gate"]),
            "calibration": {"window_ids": window_ids},
            "target_fingerprint": condition_common["target_fingerprint"],
            "pairing_reference": {"digest": condition_common["pairing_digest"]},
            "dataset_meta": {
                "dataset_hash": condition_common["dataset_hash"],
                "tokenizer_hash": condition_common["tokenizer_hash"],
            },
        },
    }
    return report


def strict_no_adjustment_report() -> dict[str, Any]:
    """Build a strict noop report with a fully bound no-adjustment decision."""

    report = strict_report()
    report["context"]["profile"] = "release"
    report["edit"] = {"name": "noop"}
    report["structure"] = {"params_changed": 0, "layers_modified": 0}
    report["variance"]["enabled"] = False
    report["variance"]["monitor_only"] = False
    report["variance"]["calibration"] = {
        "status": "no_scaling_required",
        "coverage": 8,
        "min_coverage": 6,
    }
    report["variance"]["predictive_gate"] = {
        "evaluated": True,
        "passed": True,
        "reason": "no_adjustment_required",
        "delta_ci": [None, None],
        "gain_ci": [None, None],
        "mean_delta": None,
    }
    bind_noop_variance_evidence(report)
    report["assurance"] = build_assurance_section(report)
    return report


def _mapping(value: object, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    nested = value.get(key)
    return nested if isinstance(nested, dict) else {}


def _sync_variance_guard_metrics(report: dict[str, Any]) -> None:
    variance = report["variance"]
    variance_guard = next(
        entry for entry in report["guards"] if entry["name"] == "variance"
    )
    metrics = {
        "ve_enabled": variance["enabled"],
        "monitor_only": variance["monitor_only"],
        "predictive_gate": dict(variance["predictive_gate"]),
    }
    for top_key, raw_key in (
        ("ve_enabled_during_validation", "ve_enabled_during_validation"),
        ("subject_restored_after_ab", "subject_restored_after_ab"),
        ("met_threshold", "met_threshold"),
        ("gain", "ab_gain"),
        ("ppl_no_ve", "ppl_no_ve"),
        ("ppl_with_ve", "ppl_with_ve"),
        ("ratio_ci", "ratio_ci"),
        ("proposed_scales", "proposed_scales"),
        ("target_modules", "target_modules"),
        ("target_module_names", "target_module_names"),
        ("proposed_scales_pre_edit", "proposed_scales_pre_edit"),
        ("proposed_scales_post_edit", "proposed_scales_post_edit"),
        ("raw_scales_pre_edit", "raw_scales_pre_edit"),
        ("raw_scales_post_edit", "raw_scales_post_edit"),
    ):
        if top_key in variance:
            metrics[raw_key] = copy.deepcopy(variance[top_key])
    if "calibration" in variance:
        metrics["calibration"] = dict(variance["calibration"])
    ab_test = variance.get("ab_test")
    if isinstance(ab_test, dict):
        metrics["ab_seed_used"] = ab_test.get("seed")
        metrics["ab_windows_used"] = ab_test.get("windows_used")
        raw_provenance = copy.deepcopy(ab_test.get("provenance"))
        if isinstance(raw_provenance, dict):
            raw_provenance.pop("window_ids", None)
        metrics["ab_provenance"] = raw_provenance
        metrics["ab_point_estimates"] = copy.deepcopy(ab_test.get("point_estimates"))
        metrics["ab_measurements"] = copy.deepcopy(ab_test.get("measurements"))
    variance_guard["metrics"] = metrics
    if "policy" in variance:
        variance_guard["policy"] = copy.deepcopy(variance["policy"])
        metrics["mode"] = variance["policy"].get("mode")
        report.setdefault("resolved_policy", {})["variance"] = copy.deepcopy(
            variance["policy"]
        )
    if "subject_restored_after_ab" not in variance:
        return
    condition_a = _mapping(metrics.get("ab_provenance"), "condition_a")
    variance_guard["details"] = {
        "ve_tested": variance["ve_enabled_during_validation"],
        "ve_applied": variance["enabled"],
        "subject_restored_after_ab": variance["subject_restored_after_ab"],
        "policy": copy.deepcopy(variance.get("policy")),
        "proposed_scales": copy.deepcopy(metrics.get("proposed_scales_post_edit", {})),
        "stats": {
            "ab_provenance": copy.deepcopy(metrics.get("ab_provenance")),
            "ab_point_estimates": copy.deepcopy(metrics.get("ab_point_estimates")),
            "ab_measurements": copy.deepcopy(metrics.get("ab_measurements")),
            "predictive_gate": copy.deepcopy(metrics.get("predictive_gate")),
            "calibration": {"window_ids": copy.deepcopy(condition_a.get("window_ids"))},
            "target_fingerprint": condition_a.get("target_fingerprint"),
            "pairing_reference": {"digest": condition_a.get("pairing_digest")},
            "dataset_meta": {
                "dataset_hash": condition_a.get("dataset_hash"),
                "tokenizer_hash": condition_a.get("tokenizer_hash"),
            },
            "target_module_names": copy.deepcopy(metrics.get("target_module_names")),
            "proposed_scales_pre_edit": copy.deepcopy(
                metrics.get("proposed_scales_pre_edit")
            ),
            "proposed_scales_post_edit": copy.deepcopy(
                metrics.get("proposed_scales_post_edit")
            ),
            "raw_scales_pre_edit_normalized": copy.deepcopy(
                metrics.get("raw_scales_pre_edit")
            ),
            "raw_scales_post_edit_normalized": copy.deepcopy(
                metrics.get("raw_scales_post_edit")
            ),
        },
    }


def strict_variance_gain_report() -> dict[str, Any]:
    """Build a strict non-noop report with complete variance-gain evidence."""

    report = strict_report()
    report["edit"] = {"name": "quant_rtn"}
    report["plugins"]["edit"] = _plugin_metadata("edits", "quant_rtn")
    report["structure"]["params_changed"] = 123
    variance_guard = next(
        entry for entry in report["guards"] if entry["name"] == "variance"
    )
    existing = variance_guard["metrics"]["ab_provenance"]
    window_ids = list(existing["condition_a"]["window_ids"])
    delta_log = math.log(98.0) - math.log(100.0)
    measurements = {
        "window_ids": list(window_ids),
        "condition_a": {
            "ppl": [100.0] * 8,
            "log_loss": [math.log(100.0)] * 8,
            "token_counts": [16] * 8,
        },
        "condition_b": {
            "ppl": [98.0] * 8,
            "log_loss": [math.log(98.0)] * 8,
            "token_counts": [16] * 8,
        },
        "ratio_bootstrap": {
            "method": "percentile_mean_ppl_ratio",
            "replicates": 500,
            "alpha": 0.05,
            "seed": 123,
        },
        "delta_log_bootstrap": {
            "method": "bca_paired_delta_log",
            "replicates": 500,
            "alpha": 0.05,
            "seed": 334,
            "weights": "condition_a_token_counts",
        },
        "ratio_ci": [0.98, 0.98],
        "delta_log_ci": [delta_log, delta_log],
    }
    condition_common = {
        "tag": "post_edit",
        "window_ids": window_ids,
        "window_count": 8,
        "target_fingerprint": "strict-target",
        "pairing_digest": existing["condition_a"]["pairing_digest"],
        "consumed_pairing_digest": existing["condition_a"]["consumed_pairing_digest"],
        "dataset_hash": "strict-dataset",
        "tokenizer_hash": "strict-tokenizer",
        "model_id": "strict-model",
        "run_seed": 123,
        "status": "evaluated",
    }
    policy = {
        "min_effect_lognll": 0.005,
        "predictive_one_sided": True,
        "predictive_gate": True,
        "alpha": 0.05,
        "min_gain": 0.0,
        "tie_breaker_deadband": 0.005,
        "min_rel_gain": 0.001,
        "seed": 123,
        "mode": "ci",
        "absolute_floor_ppl": 0.05,
        "clamp": [0.5, 2.0],
        "deadband": 0.02,
        "min_abs_adjust": 0.012,
        "max_scale_step": 0.02,
        "topk_backstop": 1,
        "max_adjusted_modules": 0,
        "calibration": {"windows": 8, "min_coverage": 6, "seed": 123},
    }
    report["variance"] = {
        "enabled": False,
        "ve_enabled_during_validation": True,
        "subject_restored_after_ab": True,
        "met_threshold": True,
        "monitor_only": False,
        "supported": True,
        "passed": True,
        "decision": "allow",
        "violations": [],
        "policy": policy,
        "gain": 0.02,
        "ppl_no_ve": 100.0,
        "ppl_with_ve": 98.0,
        "ratio_ci": [0.98, 0.98],
        "proposed_scales": 1,
        "target_modules": 1,
        "target_module_names": ["transformer.h.0.mlp.c_proj"],
        "proposed_scales_pre_edit": {"transformer.h.0.mlp.c_proj": 1.02},
        "proposed_scales_post_edit": {"transformer.h.0.mlp.c_proj": 1.02},
        "raw_scales_pre_edit": {"transformer.h.0.mlp.c_proj": 1.1},
        "raw_scales_post_edit": {"transformer.h.0.mlp.c_proj": 1.1},
        "predictive_gate": {
            "evaluated": True,
            "passed": True,
            "reason": "ci_gain_met",
            "delta_ci": [delta_log, delta_log],
            "gain_ci": [-delta_log, -delta_log],
            "mean_delta": delta_log,
        },
        "calibration": {
            "status": "complete",
            "requested": 8,
            "coverage": 8,
            "min_coverage": 6,
            "seed": 123,
        },
        "ab_test": {
            "seed": 123,
            "windows_used": 8,
            "provenance": {
                "condition_a": {**condition_common, "mode": "edited_no_ve"},
                "condition_b": {**condition_common, "mode": "virtual_ve"},
                "window_ids": window_ids,
            },
            "point_estimates": {
                "tag": "post_edit",
                "ppl_no_ve": 100.0,
                "ppl_with_ve": 98.0,
                "coverage": 8,
            },
            "measurements": measurements,
        },
    }
    _sync_variance_guard_metrics(report)
    report["assurance"] = build_assurance_section(report)
    return report


def strict_report() -> dict[str, Any]:
    guards: list[dict[str, Any]] = []
    for index, name in enumerate(CANONICAL_GUARD_CHAIN):
        entry: dict[str, Any] = {
            "name": name,
            "supported": True,
            "passed": True,
            "decision": "allow",
            "violations": [],
            "diagnostics": [],
        }
        if name == "invariants" and index == 0:
            entry["stage"] = "pre"
        if name == "invariants" and index == len(CANONICAL_GUARD_CHAIN) - 1:
            entry["stage"] = "post"
        guards.append(entry)
    report = {
        "context": {
            "profile": "ci",
            "assurance": {"mode": "strict"},
            "runtime": {"execution_mode": "container"},
            "guard_chain_observed": list(CANONICAL_GUARD_CHAIN),
        },
        "auto": {"tier": "balanced"},
        "policy_provenance": {
            "source": "runtime",
        },
        "plugins": {
            "adapter": _plugin_metadata("adapters", "hf_causal"),
            "edit": _plugin_metadata("edits", "noop"),
            "guards": [
                _plugin_metadata("guards", name) for name in CANONICAL_GUARD_CHAIN
            ],
        },
        "meta": {"adapter": "hf_causal"},
        "edit": {"name": "noop"},
        "structure": {"params_changed": 0, "layers_modified": 0},
        "primary_metric": {"kind": "ppl_causal", "final": 2.0},
        "evaluation_windows": {
            "preview": {
                "window_ids": list(range(4)),
                "logloss": [math.log(2.0)] * 4,
                "token_counts": [1] * 4,
            },
            "final": {
                "window_ids": list(range(4, 8)),
                "logloss": [math.log(2.0)] * 4,
                "token_counts": [1] * 4,
            },
        },
        "guards": guards,
        "spectral": {
            "supported": True,
            "passed": True,
            "decision": "allow",
            "violations": [],
            "evaluated": True,
            "caps_applied": 0,
            "max_caps": 5,
            "caps_exceeded": False,
            "summary": {
                "status": "stable",
                "modules_checked": 2,
                "max_caps": 5,
                "caps_exceeded": False,
            },
        },
        "rmt": {
            "supported": True,
            "passed": True,
            "decision": "allow",
            "violations": [],
            "evaluated": True,
            "status": "stable",
            "stable": True,
            "epsilon_default": 0.01,
            "epsilon_by_family": {"ffn": 0.01},
            "epsilon_violations": [],
            "edge_risk_by_family_base": {"ffn": 1.0},
            "edge_risk_by_family": {"ffn": 1.0},
            "families": {
                "ffn": {
                    "edge_base": 1.0,
                    "edge_cur": 1.0,
                    "epsilon": 0.01,
                    "allowed": 1.01,
                    "ratio": 1.0,
                    "delta": 0.0,
                }
            },
        },
        "variance": {
            "enabled": False,
            "monitor_only": False,
            "supported": True,
            "passed": True,
            "decision": "allow",
            "violations": [],
            "predictive_gate": {
                "evaluated": True,
                "passed": True,
                "reason": "no_adjustment_required",
            },
            "calibration": {
                "status": "no_scaling_required",
                "coverage": 8,
                "min_coverage": 6,
            },
        },
        "invariants": {
            "supported": True,
            "passed": True,
            "decision": "allow",
            "violations": [],
            "status": "pass",
            "pre": "pass",
            "post": "pass",
            "summary": {
                "checks_performed": 2,
                "violations_found": 0,
                "fatal_violations": 0,
                "warning_violations": 0,
            },
            "failures": [],
        },
        "guard_metric_impact": {
            "evaluated": True,
            "passed": True,
            "metric_kind": "ppl_causal",
            "bare_value": 2.0,
            "guarded_value": 2.0,
            "degradation_limit": 0.01,
            "diagnostics": [],
            "source": "strict_fixture",
        },
        "validation": {
            "preview_final_drift_acceptable": True,
            "primary_metric_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "guard_metric_impact_acceptable": True,
        },
    }
    return bind_noop_variance_evidence(bind_raw_guard_evidence(report))
