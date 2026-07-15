from __future__ import annotations

import copy
import math


def _bounded_report() -> dict:
    p_value = math.erfc(2.0 / math.sqrt(2.0))
    selection = {
        "method": "bonferroni",
        "alpha": 0.05,
        "m": 1,
        "families_tested": ["ffn"],
        "families_selected": ["ffn"],
        "family_pvalues": {"ffn": p_value},
        "family_max_abs_z": {"ffn": 2.0},
        "family_violation_counts": {"ffn": 1},
        "default_selected_without_pvalue": 0,
    }
    violation = {
        "type": "family_z_cap",
        "severity": "budgeted",
        "module": "layer.0",
        "family": "ffn",
        "z_score": 2.0,
        "kappa": 1.0,
        "sigma": 1.2,
        "baseline_sigma": 1.0,
        "p_value": p_value,
        "selected": True,
        "message": "bounded family cap",
    }
    policy = {
        "deadband": 0.1,
        "max_caps": 1,
        "max_spectral_norm": None,
        "family_caps": {"ffn": {"kappa": 1.0}, "other": {"kappa": 3.0}},
        "multiple_testing": {"method": "bonferroni", "alpha": 0.05, "m": 1},
        "degeneracy": {
            "enabled": False,
            "stable_rank": {"warn_ratio": 0.5, "fatal_ratio": 0.25},
            "norm_collapse": {"warn_ratio": 0.25, "fatal_ratio": 0.1},
        },
        "correction_enabled": False,
        "correction_cap_ratio": 2.0,
    }
    metrics = {
        "modules_checked": 2,
        "baseline_modules": 2,
        "violations_found": 1,
        "budgeted_violations": 1,
        "candidate_budgeted_violations": 1,
        "fatal_violations": 0,
        "caps_applied": 1,
        "max_caps": 1,
        "caps_exceeded": False,
        "max_spectral_norm": 1.2,
        "mean_spectral_norm": 1.1,
        "family_caps": copy.deepcopy(policy["family_caps"]),
        "multiple_testing": copy.deepcopy(policy["multiple_testing"]),
        "multiple_testing_selection": selection,
    }
    entry = {
        "name": "spectral",
        "supported": True,
        "passed": True,
        "decision": "monitor",
        "policy": policy,
        "metrics": metrics,
        "violations": [violation],
        "baseline_metrics": {
            "module_sigmas": {"layer.0": 1.0, "layer.1": 1.0},
            "family_stats": {
                "ffn": {"count": 2, "mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}
            },
            "baseline_degeneracy": {},
        },
        "final_metrics": {"layer.0": 1.2, "layer.1": 1.0},
        "final_z_scores": {"layer.0": 2.0, "layer.1": 0.0},
        "module_family_map": {"layer.0": "ffn", "layer.1": "ffn"},
        "final_degeneracy": {},
    }
    finding = copy.deepcopy(violation)
    finding["finding_id"] = "finding-0001:family_z_cap:layer.0"
    entry["measurement_inventory"] = {
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
    }
    entry["correction_ledger"] = {
        "schema_version": 1,
        "phase": "validate",
        "correction_enabled": False,
        "correction_cap_ratio": 2.0,
        "pre_correction_metrics": {"layer.0": 1.2, "layer.1": 1.0},
        "pre_correction_z_scores": {"layer.0": 2.0, "layer.1": 0.0},
        "pre_correction_degeneracy": {},
        "multiple_testing_selection": copy.deepcopy(selection),
        "selected_findings": [finding],
        "corrections": [
            {
                "correction_id": "correction-0001:layer.0",
                "finding_ids": [finding["finding_id"]],
                "module": "layer.0",
                "operation": "none",
                "attempted": False,
                "mutation_applied": False,
                "outcome": "not_attempted_policy_disabled",
                "pre_sigma": 1.2,
                "baseline_sigma": 1.0,
                "post_sigma": 1.2,
                "scale_factor": 1.0,
                "pre_weight_digest": "a" * 64,
                "post_weight_digest": "a" * 64,
            }
        ],
        "policy_result": "correction_disabled",
        "post_correction_metrics": {"layer.0": 1.2, "layer.1": 1.0},
    }
    entry["metrics"].update(
        {
            "selected_budgeted_findings": 1,
            "cap_budget_exceeded": False,
            "corrections_attempted": 0,
            "corrections_applied": 0,
            "correction_policy_result": "correction_disabled",
            "identity_changed_modules": [],
            "measurement_exclusions": [],
            "discovery_errors": [],
        }
    )
    return {
        "guards": [entry],
        "spectral": {
            "supported": True,
            "passed": True,
            "decision": "monitor",
            "status": "capped",
            "evaluated": True,
            "caps_applied": 1,
            "max_caps": 1,
            "caps_exceeded": False,
            "violations": [copy.deepcopy(violation)],
            "summary": {
                "status": "capped",
                "modules_checked": 2,
                "caps_applied": 1,
                "max_caps": 1,
                "caps_exceeded": False,
            },
        },
    }


def _over_budget_report() -> dict:
    payload = _bounded_report()
    raw = payload["guards"][0]
    raw.update({"passed": False, "decision": "block"})
    raw["policy"]["max_caps"] = 0
    raw["metrics"].update(
        {"max_caps": 0, "caps_exceeded": True, "cap_budget_exceeded": True}
    )
    payload["spectral"].update(
        {
            "passed": False,
            "decision": "block",
            "status": "fail",
            "max_caps": 0,
            "caps_exceeded": True,
        }
    )
    payload["spectral"]["summary"].update(
        {"status": "fail", "max_caps": 0, "caps_exceeded": True}
    )
    return payload


__all__ = ["_bounded_report", "_over_budget_report"]
