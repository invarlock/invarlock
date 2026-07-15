from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import invarlock.reporting.verify_contract as verify_mod
from invarlock.core.assurance_contract import (
    ASSURANCE_CLAIM_SET,
    CANONICAL_GUARD_CHAIN,
)
from invarlock.reporting.report_provenance import compute_report_digest
from invarlock.reporting.verify_contract import run_verify_reports
from invarlock.runtime_provenance import RuntimeProvenanceResult
from tests.cli._support_verify_runtime_provenance import (
    _matching_strict_ppl_baseline,
    _write_matching_strict_policy_pack,
    bind_runtime_policy_receipt,
)
from tests.core._support_assurance_contract import (
    _plugin_metadata,
    bind_noop_variance_evidence,
)


def _build_guard_inventory(
    guard_chain: list[str],
    *,
    spectral_contract: dict[str, object],
    rmt_contract: dict[str, object],
) -> list[dict[str, object]]:
    guard_inventory = []
    for index, name in enumerate(guard_chain):
        entry = {
            "name": name,
            "supported": True,
            "passed": True,
            "decision": "allow",
            "violations": [],
            "diagnostics": [],
        }
        if name == "invariants" and index == 0:
            entry["stage"] = "pre"
        if name == "invariants" and index == len(guard_chain) - 1:
            entry["stage"] = "post"
        if name == "invariants":
            invariant_checks = {"parameter_count": 2, "structure_hash": "stable"}
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
                        "baseline_checks": invariant_checks,
                        "current_checks": dict(invariant_checks),
                    },
                }
            )
        if name == "spectral":
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
        if name == "rmt":
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
        if name == "variance":
            entry["metrics"] = {
                "ve_enabled": False,
                "monitor_only": False,
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
            }
        guard_inventory.append(entry)
    return guard_inventory


def _report(guard_chain: list[str]) -> dict:
    window_count = 180
    preview_window_ids = list(range(window_count))
    final_window_ids = list(range(window_count, window_count * 2))
    digest = hashlib.blake2s(digest_size=16)
    for window_id in final_window_ids:
        digest.update(window_id.to_bytes(8, "little", signed=True))
    schedule_digest = digest.hexdigest()
    arm_ids_digest = hashlib.sha256(
        json.dumps(final_window_ids, separators=(",", ":")).encode()
    ).hexdigest()
    spectral_contract = {"kind": "spectral_norm_power_iter", "version": 1}
    rmt_contract = {"kind": "activation_edge_risk", "version": 1}
    guard_inventory = _build_guard_inventory(
        guard_chain,
        spectral_contract=spectral_contract,
        rmt_contract=rmt_contract,
    )
    report = {
        "schema_version": "v1",
        "run_id": "strict-test",
        "artifacts": {},
        "plugins": {
            "adapter": _plugin_metadata("adapters", "hf_causal"),
            "edit": _plugin_metadata("edits", "noop"),
            "guards": [_plugin_metadata("guards", name) for name in guard_chain],
        },
        "guards": guard_inventory,
        "spectral": {
            "supported": True,
            "passed": True,
            "decision": "allow",
            "violations": [],
            "status": "pass",
            "evaluated": True,
            "caps_applied": 0,
            "max_caps": 5,
            "caps_exceeded": False,
            "summary": {
                "status": "stable",
                "modules_checked": 2,
                "caps_applied": 0,
                "max_caps": 5,
                "caps_exceeded": False,
            },
            "measurement_contract": spectral_contract,
            "measurement_contract_hash": verify_mod._measurement_contract_digest(
                spectral_contract
            ),
            "measurement_contract_match": True,
        },
        "rmt": {
            "supported": True,
            "passed": True,
            "decision": "allow",
            "violations": [],
            "status": "stable",
            "evaluated": True,
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
            "measurement_contract": rmt_contract,
            "measurement_contract_hash": verify_mod._measurement_contract_digest(
                rmt_contract
            ),
            "measurement_contract_match": True,
        },
        "variance": {
            "supported": True,
            "passed": True,
            "decision": "allow",
            "violations": [],
            "status": "pass",
            "enabled": False,
            "monitor_only": False,
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
        "meta": {
            "profile": "ci",
            "model_id": "strict-test-model",
            "adapter": "hf_causal",
            "tokenizer_hash": "strict-tokenizer",
            "model_identity": {
                "kind": "remote_revision",
                "revision": "a" * 40,
            },
        },
        "edit": {"name": "noop"},
        "structure": {"params_changed": 0, "layers_modified": 0},
        "subject_ref": {
            "model_id": "strict-test-model",
            "adapter": "hf_causal",
            "model_identity": {
                "kind": "remote_revision",
                "revision": "a" * 40,
            },
        },
        "context": {
            "profile": "ci",
            "runtime": {"execution_mode": "container"},
            "guard_chain_observed": guard_chain,
        },
        "auto": {"tier": "balanced"},
        "resolved_policy": {
            "spectral": {"measurement_contract": spectral_contract},
            "rmt": {"measurement_contract": rmt_contract},
        },
        "policy_provenance": {
            "source": "runtime",
        },
        "provenance": {
            "provider_digest": {
                "ids_sha256": "subject-ids",
                "tokenizer_sha256": "strict-tokenizer",
            },
            "window_ids_digest": schedule_digest,
            "window_plan_digest": schedule_digest,
        },
        "dataset": {
            "provider": "local_jsonl",
            "split": "validation",
            "seq_len": 8,
            "hash": {
                "preview": "strict-preview-dataset",
                "final": "strict-final-dataset",
                "dataset": "strict-dataset",
            },
            "tokenizer": {"hash": "strict-tokenizer"},
            "windows": {
                "preview": window_count,
                "final": window_count,
                "stats": {
                    "actual_preview": window_count,
                    "actual_final": window_count,
                    "paired_windows": window_count,
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "window_pairing_reason": None,
                    "preview_final_slice_delta_summary": {
                        "mean": 0.0,
                        "ci": [0.0, 0.0],
                        "basis": "independent_disjoint_slices",
                        "paired": False,
                        "ci_method": "independent_percentile_delta_log",
                        "ci_reason": None,
                        "preview_windows": window_count,
                        "final_windows": window_count,
                        "degenerate": True,
                        "degenerate_reason": "constant_bootstrap_distribution",
                    },
                    "coverage": {
                        "tier": "balanced",
                        "preview": {
                            "used": window_count,
                            "required": window_count,
                            "ok": True,
                        },
                        "final": {
                            "used": window_count,
                            "required": window_count,
                            "ok": True,
                        },
                        "replicates": {
                            "used": 1200,
                            "required": 1200,
                            "ok": True,
                        },
                    },
                    "bootstrap": {
                        "enabled": True,
                        "method": "bca_paired_delta_log",
                        "alpha": 0.05,
                        "replicates": 1200,
                        "seed": 43,
                        "preview_final_delta_basis": ("independent_disjoint_slices"),
                        "preview_final_delta_method": (
                            "independent_percentile_delta_log"
                        ),
                        "preview_final_delta_seed": 140,
                    },
                },
            },
        },
        "baseline_ref": {
            "primary_metric": {"kind": "ppl_causal", "final": 2.0},
            "tokenizer_hash": "strict-tokenizer",
        },
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 2.0,
            "final": 2.0,
            "ratio_vs_baseline": 1.0,
            "ci": [0.0, 0.0],
            "display_ci": [1.0, 1.0],
            "analysis_basis": "mean_logloss",
            "analysis_point_preview": 0.6931471805599453,
            "analysis_point_final": 0.6931471805599453,
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": preview_window_ids,
                "logloss": [0.6931471805599453] * window_count,
                "token_counts": [10] * window_count,
            },
            "final": {
                "window_ids": final_window_ids,
                "logloss": [0.6931471805599453] * window_count,
                "token_counts": [10] * window_count,
            },
        },
        "guard_metric_impact": {
            "evaluated": True,
            "metric_kind": "ppl_causal",
            "direction": "lower",
            "bare_value": 2.0,
            "guarded_value": 2.0,
            "degradation_basis": "relative_increase",
            "degradation": 0.0,
            "display_value": 0.0,
            "display_unit": "percent",
            "degradation_limit": 0.01,
            "passed": True,
            "schedule_digest": schedule_digest,
            "bare_facts": {
                "weighted_logloss_sum": math.log(2.0) * window_count * 10,
                "token_count": window_count * 10,
                "example_ids_digest": arm_ids_digest,
            },
            "guarded_facts": {
                "weighted_logloss_sum": math.log(2.0) * window_count * 10,
                "token_count": window_count * 10,
                "example_ids_digest": arm_ids_digest,
            },
            "bare_report": {
                "primary_metric": {"kind": "ppl_causal", "final": 2.0},
                "final": {
                    "window_ids": final_window_ids,
                    "logloss": [math.log(2.0)] * window_count,
                    "token_counts": [10] * window_count,
                },
                "status": "success",
            },
            "checks": {
                "metric_kind_matches": True,
                "measurements_valid": True,
                "guard_metric_impact": True,
                "schedule_match": True,
                "measurement_complete": True,
                "arm_facts_replay": True,
            },
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
        "assurance": {
            "mode": "strict",
            "profile": "ci",
            "tier": "balanced",
            "claim_set": ASSURANCE_CLAIM_SET,
            "canonical_guard_chain": list(CANONICAL_GUARD_CHAIN),
            "guard_chain_observed": guard_chain,
            "canonical_guard_chain_enforced": guard_chain
            == list(CANONICAL_GUARD_CHAIN),
            "fallback_fields_used": False,
            "runtime_provenance_verified": False,
            "runtime_provenance_declared": "container",
            "runtime_provenance_verification_status": "pending",
            "verdict": "pending_verifier",
            "report_local_verdict": "pass",
            "verified_assurance_verdict": "pending",
            "blocking_reasons": [],
        },
    }
    return bind_runtime_policy_receipt(bind_noop_variance_evidence(report))


def _write_report(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _verified_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        verify_mod,
        "verify_runtime_provenance",
        lambda *args, **kwargs: RuntimeProvenanceResult(
            verified=True,
            skipped=False,
        ),
        raising=True,
    )


def _run_strict(path: Path, *, profile: str | None = "ci"):
    subject = json.loads(path.read_text(encoding="utf-8"))
    baseline = _matching_strict_ppl_baseline(subject)
    subject["baseline_ref"].update(
        {
            "run_id": "strict-baseline-run",
            "model_id": "strict-test-model",
            "adapter": "hf_causal",
            "tokenizer_hash": "strict-tokenizer",
            "report_hash": compute_report_digest(baseline),
            "provider_digest": copy.deepcopy(
                baseline.get("provenance", {}).get("provider_digest", {})
            ),
            "model_identity": copy.deepcopy(baseline["meta"]["model_identity"]),
        }
    )
    subject["provenance"]["baseline"] = {
        "run_id": "strict-baseline-run",
        "report_hash": compute_report_digest(baseline),
    }
    path.write_text(json.dumps(subject), encoding="utf-8")
    baseline_path = path.with_name("trusted-baseline.json")
    baseline_path.write_text(
        json.dumps(baseline),
        encoding="utf-8",
    )
    policy_path = _write_matching_strict_policy_pack(path, subject)
    return run_verify_reports(
        [path],
        baseline=baseline_path,
        policy_pack=policy_path,
        profile=profile,
        assurance_mode="strict",
    )
