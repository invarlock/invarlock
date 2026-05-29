from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardFaultInjectionSeam:
    module: str
    function: str
    parameter: str
    rationale: str


GUARD_FAULT_INJECTION_SEAMS: tuple[GuardFaultInjectionSeam, ...] = (
    GuardFaultInjectionSeam(
        "rmt_activation_runtime",
        "compute_activation_edge_risk",
        "classify_family_fn",
        "Injects a deterministic family classifier for activation RMT aggregation without constructing a full guard facade.",
    ),
    GuardFaultInjectionSeam(
        "spectral_control",
        "apply_weight_rescale",
        "should_process_module_fn",
        "Exercises scope-filtering and per-module mutation paths without monkeypatching global spectral detection.",
    ),
    GuardFaultInjectionSeam(
        "spectral_control",
        "apply_relative_spectral_cap",
        "should_process_module_fn",
        "Forces scoped capping and skipped-module branches while keeping tensor mutation local to the test model.",
    ),
    GuardFaultInjectionSeam(
        "spectral_control",
        "apply_relative_spectral_cap",
        "capture_baseline_sigmas_fn",
        "Injects baseline-capture failures and known baselines at the capping boundary.",
    ),
    GuardFaultInjectionSeam(
        "spectral_control",
        "apply_relative_spectral_cap",
        "compute_sigma_max_fn",
        "Injects deterministic or failing spectral estimates before in-place cap correction.",
    ),
    GuardFaultInjectionSeam(
        "spectral_control",
        "apply_spectral_control",
        "apply_relative_spectral_cap_fn",
        "Separates policy orchestration failure handling from the capping implementation.",
    ),
    GuardFaultInjectionSeam(
        "spectral_control",
        "apply_spectral_control",
        "apply_weight_rescale_fn",
        "Separates policy orchestration failure handling from the rescale implementation.",
    ),
    GuardFaultInjectionSeam(
        "spectral_detection",
        "classify_model_families",
        "should_process_module_fn",
        "Injects deterministic scope inclusion for family-map construction.",
    ),
    GuardFaultInjectionSeam(
        "spectral_detection",
        "classify_model_families",
        "classify_module_family_fn",
        "Injects deterministic family labels for policy and z-score tests.",
    ),
    GuardFaultInjectionSeam(
        "spectral_detection",
        "detect_spectral_violations",
        "compute_sigma_max_fn",
        "Forces missing-metric measurement and estimator-failure branches in violation detection.",
    ),
    GuardFaultInjectionSeam(
        "spectral_detection",
        "detect_spectral_violations",
        "classify_module_family_fn",
        "Forces unknown-module family assignment without mutating the guard's classifier.",
    ),
    GuardFaultInjectionSeam(
        "spectral_detection",
        "detect_spectral_violations",
        "compute_z_score_for_value_fn",
        "Injects z-score boundary values for direct production-vs-reference decision tests.",
    ),
    GuardFaultInjectionSeam(
        "spectral_detection",
        "detect_spectral_violations",
        "default_family_caps_fn",
        "Exercises fallback cap lookup when policy caps are malformed or missing.",
    ),
    GuardFaultInjectionSeam(
        "spectral_measurement",
        "_compute_sigma_with_optional_diagnostics",
        "compute_sigma_max_fn",
        "Centralizes custom estimator failure diagnostics for callers that pass measurement seams.",
    ),
    GuardFaultInjectionSeam(
        "spectral_measurement",
        "compute_sigma_max",
        "power_iter_sigma_max_fn",
        "Injects deterministic, non-finite, and raising estimators to verify numeric fallbacks and diagnostics.",
    ),
    GuardFaultInjectionSeam(
        "spectral_measurement",
        "auto_sigma_target",
        "compute_sigma_max_fn",
        "Injects percentile inputs and estimator failures without requiring heavyweight model modules.",
    ),
    GuardFaultInjectionSeam(
        "spectral_measurement",
        "capture_baseline_sigmas",
        "should_process_module_fn",
        "Exercises scoped baseline capture with small fake module inventories.",
    ),
    GuardFaultInjectionSeam(
        "spectral_measurement",
        "capture_baseline_sigmas",
        "compute_sigma_max_fn",
        "Injects known baseline sigmas and measurement failures for fallback tests.",
    ),
    GuardFaultInjectionSeam(
        "spectral_measurement",
        "scan_model_gains",
        "should_process_module_fn",
        "Exercises scan coverage accounting independently of global scope predicates.",
    ),
    GuardFaultInjectionSeam(
        "spectral_measurement",
        "scan_model_gains",
        "compute_sigma_max_fn",
        "Injects deterministic scan sigmas and measurement errors for diagnostic coverage.",
    ),
    GuardFaultInjectionSeam(
        "spectral_measurement",
        "capture_sigmas",
        "power_iter_sigma_max_fn",
        "Injects estimator behavior through the guard measurement contract while preserving guard state logging.",
    ),
    GuardFaultInjectionSeam(
        "spectral_runtime",
        "prepare_guard",
        "apply_policy_overrides_fn",
        "Exercises policy override failure and no-op paths without mutating global policy helpers.",
    ),
    GuardFaultInjectionSeam(
        "spectral_runtime",
        "prepare_guard",
        "classify_model_families_fn",
        "Keeps prepare-path tests deterministic without constructing real model-family heuristics.",
    ),
    GuardFaultInjectionSeam(
        "spectral_runtime",
        "prepare_guard",
        "compute_family_stats_fn",
        "Injects baseline family statistics to isolate prepare orchestration from statistics math.",
    ),
    GuardFaultInjectionSeam(
        "spectral_runtime",
        "prepare_guard",
        "summarize_sigmas_fn",
        "Injects baseline summaries to test report payload assembly independently from summary math.",
    ),
    GuardFaultInjectionSeam(
        "spectral_runtime",
        "prepare_guard",
        "percentile_fn",
        "Forces target-sigma success and failure branches without patching numpy globally.",
    ),
    GuardFaultInjectionSeam(
        "spectral_runtime",
        "before_edit_guard",
        "compute_z_scores_fn",
        "Injects pre-edit z-score maps for lifecycle-state tests.",
    ),
    GuardFaultInjectionSeam(
        "spectral_runtime",
        "after_edit_guard",
        "apply_spectral_control_fn",
        "Injects correction behavior after detected violations without mutating model weights in lifecycle tests.",
    ),
    GuardFaultInjectionSeam(
        "variance_evaluation",
        "_compute_delta_ci",
        "compute_paired_delta_log_ci_fn",
        "Centralizes paired-CI injection for complete calibration and predictive gate edge cases.",
    ),
    GuardFaultInjectionSeam(
        "variance_evaluation",
        "_handle_complete_calibration",
        "compute_paired_delta_log_ci_fn",
        "Injects CI outputs and failures while exercising complete calibration state updates.",
    ),
    GuardFaultInjectionSeam(
        "variance_evaluation",
        "evaluate_calibration_pass",
        "compute_paired_delta_log_ci_fn",
        "Injects predictive-gate CI behavior without changing calibration sample collection.",
    ),
)


__all__ = ["GUARD_FAULT_INJECTION_SEAMS", "GuardFaultInjectionSeam"]
