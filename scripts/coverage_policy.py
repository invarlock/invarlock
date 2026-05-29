#!/usr/bin/env python3
"""Canonical coverage policy shared by the checker and Makefile lanes."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

# Explicit per-file overrides. Values are branch floors unless branch-rate is
# unavailable, in which case line-rate is used as a fallback. These take
# precedence over the global/core floors below and encode the split-aware
# coverage policy for shells, pure helpers, and tensor/mutation helpers.
THRESHOLDS = {
    # Release evidence tooling
    "scripts/release/evidence_contracts.py": 0.95,
    "scripts/release/check_empirical_guard_evidence.py": 0.95,
    "scripts/release/check_release_evidence.py": 0.95,
    # Evaluation & reporting
    "src/invarlock/eval/data.py": 1.00,
    "src/invarlock/eval/bootstrap.py": 1.00,
    "src/invarlock/eval/bench_regression.py": 1.00,
    "src/invarlock/eval/probes/fft.py": 1.00,
    "src/invarlock/eval/probes/mi.py": 0.95,
    "src/invarlock/eval/probes/post_attention.py": 0.95,
    "src/invarlock/eval/providers/base.py": 1.00,
    "src/invarlock/eval/providers/seq2seq.py": 1.00,
    "src/invarlock/eval/providers/vision_text.py": 0.95,
    "src/invarlock/eval/metrics.py": 1.00,
    "src/invarlock/eval/metrics_activation.py": 0.95,
    "src/invarlock/eval/metrics_aggregation.py": 0.95,
    "src/invarlock/eval/metrics_lens.py": 1.00,
    "src/invarlock/eval/metrics_runtime.py": 0.90,
    "src/invarlock/eval/metrics_support.py": 1.00,
    # Guard-effect benchmark harness (Step 14) + primary metric core
    "src/invarlock/eval/bench.py": 1.00,
    "src/invarlock/eval/bench_policy.py": 1.00,
    "src/invarlock/eval/bench_runner.py": 0.90,
    "src/invarlock/eval/primary_metric.py": 0.95,
    "src/invarlock/eval/tail_stats.py": 1.00,
    "src/invarlock/eval/tasks/classification.py": 1.00,
    "src/invarlock/eval/tasks/qa.py": 1.00,
    "src/invarlock/eval/tasks/text_generation.py": 0.90,
    # Calibration (release artifacts / tier updates)
    "src/invarlock/calibration/spectral_null.py": 0.95,
    "src/invarlock/calibration/variance_ve.py": 0.90,
    # Reporting
    "src/invarlock/reporting/run_report_formatters.py": 0.95,
    "src/invarlock/reporting/validate.py": 0.95,
    # Reporting types
    "src/invarlock/reporting/report_types.py": 1.00,
    "src/invarlock/reporting/dataset_hashing.py": 1.00,
    "src/invarlock/reporting/report_build_evidence.py": 1.00,
    "src/invarlock/reporting/evaluation_report_builder.py": 1.00,
    "src/invarlock/reporting/report_make_output.py": 0.95,
    "src/invarlock/reporting/report_primary_metric_policy.py": 0.95,
    "src/invarlock/reporting/primary_metric_utils.py": 0.90,
    "src/invarlock/reporting/utils.py": 1.00,
    # Shell modules: lifecycle/orchestration shells should stay branch-complete.
    "src/invarlock/core/runner.py": 1.00,
    "src/invarlock/guards/variance.py": 1.00,
    "src/invarlock/guards/spectral.py": 1.00,
    # Pure contract / result / policy / selection helpers.
    "src/invarlock/reporting/report_schema.py": 1.00,
    "src/invarlock/public_contracts.py": 1.00,
    "src/invarlock/policy_pack.py": 1.00,
    # Advanced evidence-pack packaging/inspection coverage now exercises the
    # command shell branch-completely; keep the command surface held there.
    "src/invarlock/evidence_pack.py": 1.00,
    "src/invarlock/evidence_pack_edit_metadata.py": 1.00,
    "src/invarlock/reporting/evidence.py": 1.00,
    "src/invarlock/reporting/verify_output.py": 1.00,
    "src/invarlock/cli/commands/policy.py": 1.00,
    "src/invarlock/cli/commands/evidence_pack.py": 1.00,
    "src/invarlock/core/runner_lifecycle.py": 1.00,
    "src/invarlock/core/runner_pairing.py": 1.00,
    "src/invarlock/core/runner_services.py": 1.00,
    "src/invarlock/guards/variance_policy.py": 1.00,
    "src/invarlock/guards/variance_results.py": 1.00,
    "src/invarlock/guards/spectral_policy.py": 1.00,
    "src/invarlock/guards/spectral_results.py": 1.00,
    "src/invarlock/guards/spectral_selection.py": 1.00,
    "src/invarlock/guards/spectral_analysis.py": 1.00,
    # Numerical / mutation / tensor-processing helpers.
    "src/invarlock/edits/quant_rtn.py": 0.95,
    "src/invarlock/core/runner_context.py": 0.95,
    "src/invarlock/core/runner_eval_phase.py": 1.00,
    "src/invarlock/core/runner_latency.py": 1.00,
    "src/invarlock/core/runner_eval_windows.py": 0.95,
    "src/invarlock/guards/variance_batching.py": 1.00,
    "src/invarlock/guards/variance_evaluation.py": 0.95,
    "src/invarlock/guards/variance_prepare.py": 1.00,
    "src/invarlock/guards/variance_ops.py": 1.00,
    "src/invarlock/guards/variance_scaling.py": 0.95,
    "src/invarlock/guards/invariants.py": 0.95,
    "src/invarlock/guards/spectral_control.py": 0.95,
    "src/invarlock/guards/spectral_measurement.py": 0.95,
    "src/invarlock/guards/rmt.py": 0.95,
    "src/invarlock/guards/policies.py": 1.00,
    # Core orchestration & runtime
    "src/invarlock/core/registry.py": 1.00,
    "src/invarlock/core/assurance_contract.py": 1.00,
    "src/invarlock/core/bootstrap.py": 1.00,
    "src/invarlock/core/contracts.py": 1.00,
    "src/invarlock/core/auto_tuning.py": 0.95,
    # Newly added core modules to critical surface
    "src/invarlock/core/checkpoint.py": 0.90,
    "src/invarlock/core/api.py": 1.00,
    "src/invarlock/core/retry.py": 1.00,
    "src/invarlock/core/types.py": 0.95,
    "src/invarlock/core/doctor_findings.py": 0.95,
    "src/invarlock/core/evaluate_plan.py": 1.00,
    "src/invarlock/core/runtime_manifest_verify.py": 0.95,
    # CLI commands
    "src/invarlock/cli/_json.py": 1.00,
    # Simplified public-core CLI surfaces have dedicated branch-focused tests;
    # keep the hero commands and their contract enums branch-complete.
    "src/invarlock/cli/app.py": 1.00,
    "src/invarlock/cli/runtime_modes.py": 1.00,
    "src/invarlock/core/config_runtime.py": 1.00,
    "src/invarlock/core/metric_provider_resolution.py": 0.95,
    "src/invarlock/cli/commands/evaluate.py": 1.00,
    "src/invarlock/cli/commands/verify.py": 1.00,
    "src/invarlock/cli/commands/run.py": 1.00,
    "src/invarlock/reporting/report_contract.py": 1.00,
    "src/invarlock/cli/commands/calibrate.py": 0.95,
    "src/invarlock/reporting/report_files.py": 1.00,
    "src/invarlock/reporting/verify_contract.py": 0.95,
    "src/invarlock/runtime_verify.py": 1.00,
    "src/invarlock/runtime_security.py": 1.00,
    "src/invarlock/runtime_security_helpers.py": 1.00,
    "src/invarlock/adapters/hf_multimodal.py": 1.00,
    "src/invarlock/evidence_pack_integrity.py": 0.95,
    "src/invarlock/evidence_pack_manifest.py": 0.95,
    # PR-4 split modules
    "src/invarlock/cli/run_artifacts.py": 1.00,
    "src/invarlock/cli/run_config.py": 0.95,
    "src/invarlock/cli/run_overhead.py": 1.00,
    "src/invarlock/cli/run_pairing.py": 0.95,
    "src/invarlock/core/run_policy.py": 1.00,
    "src/invarlock/reporting/run_metric_utils.py": 1.00,
    # CLI determinism preset (CI/Release provenance)
    "src/invarlock/core/determinism_policy.py": 0.95,
    # Core events logger
    "src/invarlock/core/events.py": 0.95,
    # PR-5 split modules
    "src/invarlock/core/runner_eval_metrics.py": 0.90,
    "src/invarlock/core/runner_finalize.py": 0.95,
    "src/invarlock/core/runner_guards.py": 0.95,
    "src/invarlock/reporting/report_overhead.py": 1.00,
    "src/invarlock/reporting/report_policy.py": 1.00,
    "src/invarlock/reporting/report_provenance.py": 1.00,
    "src/invarlock/reporting/report_validation.py": 0.95,
    "src/invarlock/core/run_orchestrator_execute.py": 1.00,
    "src/invarlock/reporting/verify_check_helpers.py": 0.95,
    "src/invarlock/cli/run_execution_output.py": 1.0,
    "src/invarlock/cli/runtime_launch_plan.py": 1.00,
    "src/invarlock/reporting/run_report_contract.py": 0.95,
    "src/invarlock/reporting/report_builder_support.py": 1.0,
    # Existing 90% surfaces that now sustain 95%+ or 100% in repo coverage.
    "src/invarlock/core/abi.py": 1.00,
    "src/invarlock/core/adapter_auto.py": 0.95,
    "src/invarlock/core/adapter_provenance.py": 1.00,
    "src/invarlock/core/config_loader.py": 1.00,
    "src/invarlock/core/doctor_runtime.py": 1.00,
    "src/invarlock/core/error_encoding.py": 1.00,
    "src/invarlock/core/error_utils.py": 1.00,
    "src/invarlock/core/exceptions.py": 1.00,
    "src/invarlock/core/provider_config.py": 1.00,
    "src/invarlock/core/provider_parity.py": 1.00,
    "src/invarlock/core/report_inputs.py": 1.00,
    "src/invarlock/core/run_dataset_contract.py": 1.00,
    "src/invarlock/core/run_evaluation_windows_policy.py": 1.00,
    "src/invarlock/core/run_execution_request_policy.py": 1.00,
    "src/invarlock/core/run_guard_overhead_policy.py": 0.95,
    "src/invarlock/core/run_orchestrator.py": 1.00,
    "src/invarlock/core/run_orchestrator_execute_attempts.py": 0.90,
    "src/invarlock/core/run_orchestrator_execute_helpers.py": 1.00,
    "src/invarlock/core/run_orchestrator_execute_pipeline.py": 1.00,
    "src/invarlock/core/run_orchestrator_types.py": 1.00,
    "src/invarlock/core/run_provider_dataset_plan.py": 0.90,
    "src/invarlock/core/run_report_payload_policy.py": 0.95,
    "src/invarlock/core/run_retry_policy.py": 0.90,
    "src/invarlock/core/run_snapshot_policy.py": 1.00,
    "src/invarlock/core/run_snapshot_contract.py": 0.95,
    "src/invarlock/core/run_timing_policy.py": 0.95,
    "src/invarlock/core/runner_eval_metrics_multimodal.py": 1.00,
    "src/invarlock/core/runner_eval_metrics_stats.py": 1.00,
    "src/invarlock/core/runtime_provenance.py": 1.00,
    "src/invarlock/core/doctor_inventory.py": 0.90,
    "src/invarlock/core/doctor_preflight.py": 0.95,
    "src/invarlock/core/evaluate_contract.py": 0.90,
    "src/invarlock/core/plugins_inventory.py": 0.95,
    "src/invarlock/core/run_baseline_evidence.py": 0.90,
    "src/invarlock/core/run_execution_context_policy.py": 0.90,
    "src/invarlock/guards/_contracts.py": 1.00,
    "src/invarlock/guards/rmt_analysis.py": 0.95,
    "src/invarlock/guards/rmt_result_contract.py": 1.00,
    "src/invarlock/guards/rmt_types.py": 1.00,
    "src/invarlock/guards/spectral_runtime.py": 1.00,
    "src/invarlock/guards/spectral_types.py": 1.00,
    "src/invarlock/guards/tier_config.py": 0.95,
    "src/invarlock/guards/variance_types.py": 1.00,
    "src/invarlock/cli/run_pairing_baseline.py": 0.90,
    "src/invarlock/evidence_pack_metadata.py": 1.00,
}

# Every file on the enforced critical surface is branch-complete. Keep the
# per-file table explicit for auditability, but normalize any historical lower
# floors to the current ratchet.
THRESHOLDS = dict.fromkeys(THRESHOLDS, 1.00)

# Default floors (applied only to core classification; non-core modules are not
# globally enforced unless explicitly listed in THRESHOLDS).
CORE_FLOOR_DEFAULT = 0.90
DEFAULT_FLOOR_DEFAULT = 0.90

# Core module classification: files matching any of these prefixes are treated
# as part of the critical surface and must meet the core floor (unless an
# explicit override in THRESHOLDS is present).
CORE_PREFIXES = (
    # Core orchestration & runtime
    "src/invarlock/core/",
    # Guards (safety mechanisms)
    "src/invarlock/guards/",
    # Observability is now part of the enforced runtime surface.
    "src/invarlock/observability/",
)

# Individual core files outside of the broad prefixes.
CORE_FILES = (
    # Release evidence tooling
    "scripts/release/evidence_contracts.py",
    "scripts/release/check_empirical_guard_evidence.py",
    "scripts/release/check_release_evidence.py",
    # Evaluation & reporting (key entry points)
    "src/invarlock/eval/data.py",
    "src/invarlock/eval/bootstrap.py",
    "src/invarlock/eval/bench_regression.py",
    "src/invarlock/eval/probes/fft.py",
    "src/invarlock/eval/probes/mi.py",
    "src/invarlock/eval/probes/post_attention.py",
    "src/invarlock/eval/providers/base.py",
    "src/invarlock/eval/providers/seq2seq.py",
    "src/invarlock/eval/metrics.py",
    "src/invarlock/eval/metrics_activation.py",
    "src/invarlock/eval/metrics_aggregation.py",
    "src/invarlock/eval/metrics_lens.py",
    "src/invarlock/eval/metrics_runtime.py",
    "src/invarlock/eval/metrics_support.py",
    "src/invarlock/eval/bench.py",
    "src/invarlock/eval/bench_policy.py",
    "src/invarlock/eval/bench_runner.py",
    "src/invarlock/eval/primary_metric.py",
    "src/invarlock/eval/tail_stats.py",
    "src/invarlock/eval/tasks/classification.py",
    "src/invarlock/eval/tasks/qa.py",
    "src/invarlock/eval/tasks/text_generation.py",
    "src/invarlock/calibration/spectral_null.py",
    "src/invarlock/calibration/variance_ve.py",
    "src/invarlock/reporting/run_report_formatters.py",
    "src/invarlock/reporting/validate.py",
    "src/invarlock/reporting/report_types.py",
    "src/invarlock/reporting/dataset_hashing.py",
    "src/invarlock/reporting/report_schema.py",
    "src/invarlock/reporting/report_build_evidence.py",
    "src/invarlock/reporting/evaluation_report_builder.py",
    "src/invarlock/reporting/report_make_output.py",
    "src/invarlock/reporting/report_primary_metric_policy.py",
    "src/invarlock/reporting/primary_metric_utils.py",
    "src/invarlock/reporting/utils.py",
    "src/invarlock/edits/quant_rtn.py",
    # Critical CLI commands
    "src/invarlock/cli/commands/run.py",
    "src/invarlock/cli/commands/evaluate.py",
    "src/invarlock/reporting/report_contract.py",
    "src/invarlock/cli/commands/calibrate.py",
    "src/invarlock/cli/commands/policy.py",
    "src/invarlock/reporting/report_files.py",
    "src/invarlock/reporting/verify_contract.py",
    "src/invarlock/reporting/verify_output.py",
    "src/invarlock/reporting/evidence.py",
    "src/invarlock/core/determinism_policy.py",
    "src/invarlock/core/config_runtime.py",
    "src/invarlock/cli/_json.py",
    "src/invarlock/cli/app.py",
    "src/invarlock/core/doctor_findings.py",
    "src/invarlock/core/evaluate_plan.py",
    # Public contract helpers
    "src/invarlock/public_contracts.py",
    "src/invarlock/policy_pack.py",
    "src/invarlock/evidence_pack.py",
    "src/invarlock/evidence_pack_edit_metadata.py",
    "src/invarlock/runtime_verify.py",
    "src/invarlock/cli/commands/evidence_pack.py",
    "src/invarlock/runtime_security.py",
    "src/invarlock/cli/run_config.py",
    "src/invarlock/cli/run_pairing.py",
    # Newly enforced standalone surfaces.
    "src/invarlock/config.py",
    "src/invarlock/adapters/auto.py",
)

COVERAGE_MODULE_FLAGS = ("--cov",)

COVERAGE_INCLUDE_PATTERNS = (
    "scripts/release/*.py",
    "src/invarlock/eval/*",
    "src/invarlock/guards/*",
    "src/invarlock/calibration/*",
    "src/invarlock/edits/quant_rtn.py",
    "src/invarlock/cli/*",
    "src/invarlock/cli/commands/*",
    "src/invarlock/core/*",
    "src/invarlock/reporting/*",
    "src/invarlock/observability/*",
    "src/invarlock/adapters/hf_multimodal.py",
    "src/invarlock/adapters/auto.py",
    "src/invarlock/config.py",
    "src/invarlock/public_contracts.py",
    "src/invarlock/policy_pack.py",
    "src/invarlock/evidence_pack.py",
    "src/invarlock/evidence_pack_edit_metadata.py",
    "src/invarlock/evidence_pack_integrity.py",
    "src/invarlock/evidence_pack_manifest.py",
    "src/invarlock/evidence_pack_metadata.py",
    "src/invarlock/runtime_security.py",
    "src/invarlock/runtime_security_helpers.py",
    "src/invarlock/runtime_verify.py",
    "invarlock/eval/*",
    "invarlock/guards/*",
    "invarlock/calibration/*",
    "invarlock/edits/quant_rtn.py",
    "invarlock/cli/*",
    "invarlock/cli/commands/*",
    "invarlock/core/*",
    "invarlock/reporting/*",
    "invarlock/observability/*",
    "invarlock/adapters/hf_multimodal.py",
    "invarlock/adapters/auto.py",
    "invarlock/config.py",
    "invarlock/public_contracts.py",
    "invarlock/policy_pack.py",
    "invarlock/evidence_pack.py",
    "invarlock/evidence_pack_edit_metadata.py",
    "invarlock/evidence_pack_integrity.py",
    "invarlock/evidence_pack_manifest.py",
    "invarlock/evidence_pack_metadata.py",
    "invarlock/runtime_security.py",
    "invarlock/runtime_security_helpers.py",
    "invarlock/runtime_verify.py",
)


def coverage_modules() -> str:
    return " ".join(COVERAGE_MODULE_FLAGS)


def coverage_include() -> str:
    return ",".join(COVERAGE_INCLUDE_PATTERNS)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "item",
        choices=(
            "coverage-modules",
            "coverage-include",
            "threshold-count",
            "core-prefixes",
            "core-files",
        ),
        help="Policy item to print",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.item == "coverage-modules":
        print(coverage_modules())
    elif args.item == "coverage-include":
        print(coverage_include())
    elif args.item == "threshold-count":
        print(len(THRESHOLDS))
    elif args.item == "core-prefixes":
        print(" ".join(CORE_PREFIXES))
    elif args.item == "core-files":
        print(" ".join(CORE_FILES))
    else:  # pragma: no cover - argparse prevents this branch
        raise AssertionError(f"Unhandled item: {args.item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
