import json
import subprocess
import sys
from pathlib import Path


def _write_cov_xml(path: Path, class_specs: list[tuple[str, float, float]]) -> None:
    """Write a minimal coverage.xml with given (filename, branch_rate, line_rate)."""
    lines = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        "<coverage>",
        "  <packages>",
        "    <package name='pkg'>",
        "      <classes>",
    ]
    for filename, br, lr in class_specs:
        lines.append(
            f"        <class name='X' filename='{filename}' branch-rate='{br}' line-rate='{lr}'/>"
        )
    lines += [
        "      </classes>",
        "    </package>",
        "  </packages>",
        "</coverage>",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_checker(
    xml_path: Path, json_path: Path, extra_args: list[str] | None = None
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(Path("scripts") / "check_coverage_thresholds.py"),
        "--coverage",
        str(xml_path),
        "--json",
        str(json_path),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_two_tier_policy_enforced(tmp_path: Path) -> None:
    # Create a synthetic report containing a mix of core and non-core files
    xml = tmp_path / "cov.xml"
    json_out = tmp_path / "out.json"
    _write_cov_xml(
        xml,
        [
            ("src/invarlock/core/runner.py", 0.89, 0.90),  # core → FAIL (needs 0.90)
            (
                "src/invarlock/cli/commands/run.py",
                1.0,
                0.90,
            ),  # core meets floor → PASS
            (
                "src/invarlock/reporting/report_contract.py",
                1.0,
                0.90,
            ),  # explicit shell override set to 1.00 → PASS
            (
                "src/invarlock/cli/commands/plugins.py",
                0.81,
                0.90,
            ),  # non-core → not enforced (absent from THRESHOLDS)
            (
                "src/invarlock/eval/primary_metric.py",
                0.91,
                0.90,
            ),  # explicit critical file → PASS
            (
                "src/invarlock/eval/metrics.py",
                1.0,
                0.90,
            ),  # explicit critical file → PASS
            ("src/invarlock/guards/spectral.py", 0.89, 0.90),  # core (guards) → FAIL
        ],
    )

    proc = _run_checker(xml, json_out)

    # Expect non-zero due to the two intentional core failures
    assert proc.returncode != 0
    err = proc.stderr
    assert "src/invarlock/core/runner.py" in err
    assert "src/invarlock/guards/spectral.py" in err
    # Sanity: overridden core file shouldn't appear as a failure
    assert "src/invarlock/cli/commands/run.py" not in err


def test_overrides_take_precedence(tmp_path: Path) -> None:
    # Explicit overrides should win over a stricter core-floor flag.
    xml = tmp_path / "cov.xml"
    json_out = tmp_path / "out.json"
    _write_cov_xml(
        xml, [("src/invarlock/reporting/run_report_formatters.py", 0.91, 0.90)]
    )
    proc = _run_checker(xml, json_out, extra_args=["--core-floor", "0.95"])
    assert proc.returncode != 0
    assert "src/invarlock/reporting/run_report_formatters.py" in proc.stderr

    _write_cov_xml(
        xml, [("src/invarlock/reporting/run_report_formatters.py", 0.95, 0.90)]
    )
    proc = _run_checker(xml, json_out, extra_args=["--core-floor", "0.95"])

    # Should pass with the explicit 95% override applied
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(json_out.read_text())
    assert payload["status"] == "ok"
    files = {f["path"]: f for f in payload["files"]}
    assert (
        abs(
            files["src/invarlock/reporting/run_report_formatters.py"]["threshold"]
            - 0.95
        )
        < 1e-9
    )
    assert payload["configured_threshold_files"] == 139
    assert payload["evaluated_files"] == 1
    assert payload["measured_threshold_files"] == 1
    assert "src/invarlock/cli/app.py" in payload["missing_threshold_files"]


def test_new_core_cli_and_runtime_surface_thresholds_are_enforced(
    tmp_path: Path,
) -> None:
    xml = tmp_path / "cov.xml"
    json_out = tmp_path / "out.json"
    _write_cov_xml(
        xml,
        [
            ("invarlock/cli/app.py", 0.79, 0.95),
            ("invarlock/cli/commands/evaluate.py", 0.71, 0.95),
            ("invarlock/reporting/report_contract.py", 0.81, 0.90),
            ("invarlock/core/runtime_manifest_verify.py", 0.89, 0.95),
            ("invarlock/runtime_security.py", 0.71, 0.95),
        ],
    )

    proc = _run_checker(xml, json_out)

    assert proc.returncode != 0
    assert "src/invarlock/cli/app.py" in proc.stderr
    assert "src/invarlock/cli/commands/evaluate.py" in proc.stderr
    assert "src/invarlock/reporting/report_contract.py" in proc.stderr
    assert "src/invarlock/core/runtime_manifest_verify.py" in proc.stderr
    assert "src/invarlock/runtime_security.py" in proc.stderr


def test_advanced_calibrate_and_proof_pack_thresholds_are_explicit(
    tmp_path: Path,
) -> None:
    xml = tmp_path / "cov.xml"
    json_out = tmp_path / "out.json"
    _write_cov_xml(
        xml,
        [
            ("invarlock/cli/commands/calibrate.py", 0.87, 0.99),
            ("invarlock/cli/commands/proof_pack.py", 0.39, 0.90),
            ("invarlock/proof_pack.py", 0.49, 0.90),
        ],
    )

    proc = _run_checker(xml, json_out)

    assert proc.returncode != 0
    assert "src/invarlock/cli/commands/calibrate.py" in proc.stderr
    assert "src/invarlock/cli/commands/proof_pack.py" in proc.stderr
    assert "src/invarlock/proof_pack.py" in proc.stderr


def test_summary_reports_measured_vs_configured_threshold_counts(
    tmp_path: Path,
) -> None:
    xml = tmp_path / "cov.xml"
    json_out = tmp_path / "out.json"
    _write_cov_xml(xml, [("src/invarlock/reporting/report_contract.py", 1.0, 1.0)])

    proc = _run_checker(xml, json_out)

    assert proc.returncode == 0, proc.stderr
    assert (
        "Coverage OK: 1/139 threshold-listed files had coverage data and met "
        "per-file thresholds." in proc.stdout
    )
    assert (
        "138 threshold-listed files were absent from the coverage report."
        in proc.stdout
    )

    payload = json.loads(json_out.read_text())
    assert payload["status"] == "ok"
    assert payload["configured_threshold_files"] == 139
    assert payload["evaluated_files"] == 1
    assert payload["measured_threshold_files"] == 1
    assert len(payload["missing_threshold_files"]) == 138


def test_ratchets_selected_files_to_ninety_five_percent(tmp_path: Path) -> None:
    xml = tmp_path / "cov.xml"
    json_out = tmp_path / "out.json"
    _write_cov_xml(
        xml,
        [
            ("src/invarlock/cli/commands/evaluate.py", 0.949, 1.0),
            ("src/invarlock/core/determinism_policy.py", 0.949, 1.0),
            ("src/invarlock/core/bootstrap.py", 0.949, 1.0),
            ("src/invarlock/core/auto_tuning.py", 0.949, 1.0),
            ("src/invarlock/core/config_runtime.py", 0.949, 1.0),
            ("src/invarlock/core/adapter_auto.py", 0.949, 1.0),
            ("src/invarlock/core/runtime_manifest_verify.py", 0.949, 1.0),
            ("src/invarlock/core/metric_provider_resolution.py", 0.949, 1.0),
            ("src/invarlock/core/run_guard_overhead_policy.py", 0.949, 1.0),
            ("src/invarlock/core/run_snapshot_contract.py", 0.949, 1.0),
            ("src/invarlock/core/run_timing_policy.py", 0.949, 1.0),
            ("src/invarlock/eval/tail_stats.py", 0.949, 1.0),
            ("src/invarlock/eval/metrics_activation.py", 0.949, 1.0),
            ("src/invarlock/reporting/report_policy.py", 0.949, 1.0),
            ("src/invarlock/reporting/report_validation.py", 0.949, 1.0),
            ("src/invarlock/reporting/run_report_formatters.py", 0.949, 1.0),
            ("src/invarlock/reporting/validate.py", 0.949, 1.0),
            ("src/invarlock/guards/policies.py", 0.949, 1.0),
            ("src/invarlock/guards/rmt_analysis.py", 0.949, 1.0),
            ("src/invarlock/guards/tier_config.py", 0.949, 1.0),
            ("src/invarlock/eval/bench_policy.py", 0.949, 1.0),
            ("src/invarlock/cli/commands/calibrate.py", 0.949, 1.0),
            ("src/invarlock/cli/run_config.py", 0.949, 1.0),
        ],
    )

    proc = _run_checker(xml, json_out)

    assert proc.returncode != 0
    for path in (
        "src/invarlock/cli/commands/evaluate.py",
        "src/invarlock/core/determinism_policy.py",
        "src/invarlock/core/bootstrap.py",
        "src/invarlock/core/auto_tuning.py",
        "src/invarlock/core/config_runtime.py",
        "src/invarlock/core/adapter_auto.py",
        "src/invarlock/core/runtime_manifest_verify.py",
        "src/invarlock/core/metric_provider_resolution.py",
        "src/invarlock/core/run_guard_overhead_policy.py",
        "src/invarlock/core/run_snapshot_contract.py",
        "src/invarlock/core/run_timing_policy.py",
        "src/invarlock/eval/tail_stats.py",
        "src/invarlock/eval/metrics_activation.py",
        "src/invarlock/reporting/report_policy.py",
        "src/invarlock/reporting/report_validation.py",
        "src/invarlock/reporting/run_report_formatters.py",
        "src/invarlock/reporting/validate.py",
        "src/invarlock/guards/policies.py",
        "src/invarlock/guards/rmt_analysis.py",
        "src/invarlock/guards/tier_config.py",
        "src/invarlock/eval/bench_policy.py",
        "src/invarlock/cli/commands/calibrate.py",
        "src/invarlock/cli/run_config.py",
    ):
        assert path in proc.stderr


def test_calibrated_bench_runner_threshold_is_explicit(tmp_path: Path) -> None:
    xml = tmp_path / "cov.xml"
    json_out = tmp_path / "out.json"
    _write_cov_xml(xml, [("src/invarlock/eval/bench_runner.py", 0.74, 1.0)])

    proc = _run_checker(xml, json_out)

    assert proc.returncode != 0
    assert "src/invarlock/eval/bench_runner.py" in proc.stderr


def test_ratchets_selected_files_to_branch_complete(tmp_path: Path) -> None:
    xml = tmp_path / "cov.xml"
    json_out = tmp_path / "out.json"
    _write_cov_xml(
        xml,
        [
            ("src/invarlock/reporting/evidence.py", 0.999, 1.0),
            ("src/invarlock/cli/_json.py", 0.999, 1.0),
            ("src/invarlock/cli/app.py", 0.999, 1.0),
            ("src/invarlock/cli/commands/proof_pack.py", 0.999, 1.0),
            ("src/invarlock/cli/run_artifacts.py", 0.999, 1.0),
            ("src/invarlock/cli/run_overhead.py", 0.999, 1.0),
            ("src/invarlock/core/run_policy.py", 0.999, 1.0),
            ("src/invarlock/reporting/run_metric_utils.py", 0.999, 1.0),
            ("src/invarlock/core/api.py", 0.999, 1.0),
            ("src/invarlock/core/abi.py", 0.999, 1.0),
            ("src/invarlock/core/adapter_provenance.py", 0.999, 1.0),
            ("src/invarlock/core/config_dependencies.py", 0.999, 1.0),
            ("src/invarlock/core/contracts.py", 0.999, 1.0),
            ("src/invarlock/core/doctor_runtime.py", 0.999, 1.0),
            ("src/invarlock/core/error_encoding.py", 0.999, 1.0),
            ("src/invarlock/core/error_utils.py", 0.999, 1.0),
            ("src/invarlock/core/exceptions.py", 0.999, 1.0),
            ("src/invarlock/core/evaluate_plan.py", 0.999, 1.0),
            ("src/invarlock/core/provider_config.py", 0.999, 1.0),
            ("src/invarlock/core/provider_parity.py", 0.999, 1.0),
            ("src/invarlock/core/registry.py", 0.999, 1.0),
            ("src/invarlock/core/report_inputs.py", 0.999, 1.0),
            ("src/invarlock/core/retry.py", 0.999, 1.0),
            ("src/invarlock/core/run_dataset_contract.py", 0.999, 1.0),
            ("src/invarlock/core/run_evaluation_windows_policy.py", 0.999, 1.0),
            ("src/invarlock/core/run_execution_request_policy.py", 0.999, 1.0),
            ("src/invarlock/core/run_orchestrator.py", 0.999, 1.0),
            ("src/invarlock/core/run_orchestrator_types.py", 0.999, 1.0),
            ("src/invarlock/core/runtime_attestation.py", 0.999, 1.0),
            ("src/invarlock/eval/bench.py", 0.999, 1.0),
            ("src/invarlock/eval/data.py", 0.999, 1.0),
            ("src/invarlock/eval/metrics.py", 0.999, 1.0),
            ("src/invarlock/eval/metrics_lens.py", 0.999, 1.0),
            ("src/invarlock/eval/metrics_support.py", 0.999, 1.0),
            ("src/invarlock/eval/probes/fft.py", 0.999, 1.0),
            ("src/invarlock/eval/providers/seq2seq.py", 0.999, 1.0),
            ("src/invarlock/eval/providers/base.py", 0.999, 1.0),
            ("src/invarlock/guards/_contracts.py", 0.999, 1.0),
            ("src/invarlock/guards/rmt_result_contract.py", 0.999, 1.0),
            ("src/invarlock/guards/rmt_types.py", 0.999, 1.0),
            ("src/invarlock/guards/spectral_runtime.py", 0.999, 1.0),
            ("src/invarlock/guards/spectral_types.py", 0.999, 1.0),
            ("src/invarlock/guards/variance_batching.py", 0.999, 1.0),
            ("src/invarlock/guards/variance_prepare.py", 0.999, 1.0),
            ("src/invarlock/guards/variance_types.py", 0.999, 1.0),
            ("src/invarlock/proof_pack.py", 0.999, 1.0),
            ("src/invarlock/reporting/dataset_hashing.py", 0.999, 1.0),
            ("src/invarlock/reporting/report_contract.py", 0.999, 1.0),
            ("src/invarlock/reporting/report_files.py", 0.999, 1.0),
            ("src/invarlock/reporting/report_overhead.py", 0.999, 1.0),
            ("src/invarlock/reporting/report_provenance.py", 0.999, 1.0),
            ("src/invarlock/reporting/report_types.py", 0.999, 1.0),
            ("src/invarlock/reporting/utils.py", 0.999, 1.0),
            ("src/invarlock/cli/commands/run.py", 0.999, 1.0),
            ("src/invarlock/runtime_security.py", 0.999, 1.0),
            ("src/invarlock/runtime_verify.py", 0.999, 1.0),
        ],
    )

    proc = _run_checker(xml, json_out)

    assert proc.returncode != 0
    for path in (
        "src/invarlock/reporting/evidence.py",
        "src/invarlock/cli/_json.py",
        "src/invarlock/cli/app.py",
        "src/invarlock/cli/commands/proof_pack.py",
        "src/invarlock/cli/run_artifacts.py",
        "src/invarlock/cli/run_overhead.py",
        "src/invarlock/core/run_policy.py",
        "src/invarlock/reporting/run_metric_utils.py",
        "src/invarlock/core/api.py",
        "src/invarlock/core/abi.py",
        "src/invarlock/core/adapter_provenance.py",
        "src/invarlock/core/config_dependencies.py",
        "src/invarlock/core/contracts.py",
        "src/invarlock/core/doctor_runtime.py",
        "src/invarlock/core/error_encoding.py",
        "src/invarlock/core/error_utils.py",
        "src/invarlock/core/exceptions.py",
        "src/invarlock/core/evaluate_plan.py",
        "src/invarlock/core/provider_config.py",
        "src/invarlock/core/provider_parity.py",
        "src/invarlock/core/registry.py",
        "src/invarlock/core/report_inputs.py",
        "src/invarlock/core/retry.py",
        "src/invarlock/core/run_dataset_contract.py",
        "src/invarlock/core/run_evaluation_windows_policy.py",
        "src/invarlock/core/run_execution_request_policy.py",
        "src/invarlock/core/run_orchestrator.py",
        "src/invarlock/core/run_orchestrator_types.py",
        "src/invarlock/core/runtime_attestation.py",
        "src/invarlock/eval/bench.py",
        "src/invarlock/eval/data.py",
        "src/invarlock/eval/metrics.py",
        "src/invarlock/eval/metrics_lens.py",
        "src/invarlock/eval/metrics_support.py",
        "src/invarlock/eval/probes/fft.py",
        "src/invarlock/eval/providers/seq2seq.py",
        "src/invarlock/eval/providers/base.py",
        "src/invarlock/guards/_contracts.py",
        "src/invarlock/guards/rmt_result_contract.py",
        "src/invarlock/guards/rmt_types.py",
        "src/invarlock/guards/spectral_runtime.py",
        "src/invarlock/guards/spectral_types.py",
        "src/invarlock/guards/variance_batching.py",
        "src/invarlock/guards/variance_prepare.py",
        "src/invarlock/guards/variance_types.py",
        "src/invarlock/proof_pack.py",
        "src/invarlock/reporting/dataset_hashing.py",
        "src/invarlock/reporting/report_contract.py",
        "src/invarlock/reporting/report_files.py",
        "src/invarlock/reporting/report_overhead.py",
        "src/invarlock/reporting/report_provenance.py",
        "src/invarlock/reporting/report_types.py",
        "src/invarlock/reporting/utils.py",
        "src/invarlock/cli/commands/run.py",
        "src/invarlock/runtime_security.py",
        "src/invarlock/runtime_verify.py",
    ):
        assert path in proc.stderr


def test_calibrated_split_owner_thresholds_are_explicit(tmp_path: Path) -> None:
    xml = tmp_path / "cov.xml"
    json_out = tmp_path / "out.json"
    _write_cov_xml(
        xml,
        [
            ("src/invarlock/core/config_runtime.py", 0.899, 1.0),
            ("src/invarlock/core/metric_provider_resolution.py", 0.949, 1.0),
            ("src/invarlock/core/run_orchestrator.py", 0.899, 1.0),
            ("src/invarlock/core/run_provider_dataset_plan.py", 0.889, 1.0),
            ("src/invarlock/cli/run_config.py", 0.949, 1.0),
            ("src/invarlock/cli/run_pairing.py", 0.899, 1.0),
            ("src/invarlock/eval/metrics_aggregation.py", 0.899, 1.0),
            ("src/invarlock/eval/metrics_runtime.py", 0.899, 1.0),
            ("src/invarlock/core/run_orchestrator_execute.py", 0.899, 1.0),
            ("src/invarlock/proof_pack_integrity.py", 0.899, 1.0),
            ("src/invarlock/proof_pack_manifest.py", 0.899, 1.0),
            ("src/invarlock/reporting/verify_check_helpers.py", 0.949, 1.0),
            ("src/invarlock/runtime_security_helpers.py", 0.949, 1.0),
            ("src/invarlock/cli/run_execution_output.py", 0.949, 1.0),
            ("src/invarlock/cli/runtime_launch_plan.py", 0.949, 1.0),
            ("src/invarlock/reporting/run_report_contract.py", 0.949, 1.0),
            ("src/invarlock/reporting/report_builder_support.py", 0.949, 1.0),
        ],
    )

    proc = _run_checker(xml, json_out)

    assert proc.returncode != 0
    for path in (
        "src/invarlock/core/config_runtime.py",
        "src/invarlock/core/metric_provider_resolution.py",
        "src/invarlock/core/run_orchestrator.py",
        "src/invarlock/core/run_provider_dataset_plan.py",
        "src/invarlock/cli/run_config.py",
        "src/invarlock/cli/run_pairing.py",
        "src/invarlock/eval/metrics_aggregation.py",
        "src/invarlock/eval/metrics_runtime.py",
        "src/invarlock/core/run_orchestrator_execute.py",
        "src/invarlock/proof_pack_integrity.py",
        "src/invarlock/proof_pack_manifest.py",
        "src/invarlock/reporting/verify_check_helpers.py",
        "src/invarlock/runtime_security_helpers.py",
        "src/invarlock/cli/run_execution_output.py",
        "src/invarlock/cli/runtime_launch_plan.py",
        "src/invarlock/reporting/run_report_contract.py",
        "src/invarlock/reporting/report_builder_support.py",
    ):
        assert path in proc.stderr
