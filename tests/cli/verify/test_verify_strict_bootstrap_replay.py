from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.core.bootstrap import (
    INDEPENDENT_SLICE_BOOTSTRAP_SEED_OFFSET,
    PAIRED_BASELINE_BOOTSTRAP_SEED_OFFSET,
    compute_independent_delta_log_ci,
    compute_paired_delta_log_ci,
)
from invarlock.eval.guard_metric_impact import (
    build_guard_metric_bare_report,
    compute_guard_metric_impact,
    extract_guard_metric_arm_facts,
)
from invarlock.reporting.verify_bootstrap import (
    append_strict_ppl_bootstrap_replay_errors,
)
from tests.cli._support_verify_runtime_provenance import (
    _VALID_TEST_IMAGE_DIGEST,
    _final_window_schedule_digest,
    _matching_strict_ppl_baseline,
    _strict_provenance_gate_cert,
    _write_matching_strict_policy_pack,
    _write_runtime_manifest,
)


def _weighted_mean(values: list[float], weights: list[int]) -> float:
    total = sum(weights)
    return math.fsum(
        value * (weight / total) for value, weight in zip(values, weights, strict=True)
    )


def _nondegenerate_replay_case() -> tuple[dict, dict]:
    report = _strict_provenance_gate_cert()
    final_ids = list(range(180, 360))
    preview_ids = list(range(180))
    baseline_losses = [1.0] * len(final_ids)
    subject_losses = [0.7, 0.8, 0.9, 1.1, 1.05, 1.0] * 30
    token_counts = [2, 3, 5, 7, 11, 13] * 30
    bootstrap = report["dataset"]["windows"]["stats"]["bootstrap"]

    subject_mean = _weighted_mean(subject_losses, token_counts)
    baseline_mean = _weighted_mean(baseline_losses, token_counts)
    baseline_final = math.exp(baseline_mean)
    expected_ci = compute_paired_delta_log_ci(
        subject_losses,
        baseline_losses,
        weights=token_counts,
        method="bca",
        replicates=bootstrap["replicates"],
        alpha=bootstrap["alpha"],
        seed=bootstrap["seed"] + PAIRED_BASELINE_BOOTSTRAP_SEED_OFFSET,
        strict_lengths=True,
    )
    preview_final_ci = compute_independent_delta_log_ci(
        subject_losses,
        subject_losses,
        final_weights=token_counts,
        preview_weights=token_counts,
        method="percentile",
        replicates=bootstrap["replicates"],
        alpha=bootstrap["alpha"],
        seed=bootstrap["seed"] + INDEPENDENT_SLICE_BOOTSTRAP_SEED_OFFSET,
    )

    report["evaluation_windows"] = {
        "preview": {
            "window_ids": list(preview_ids),
            "logloss": list(subject_losses),
            "token_counts": list(token_counts),
        },
        "final": {
            "window_ids": list(final_ids),
            "logloss": list(subject_losses),
            "token_counts": list(token_counts),
        },
    }
    windows = report["dataset"]["windows"]
    windows["preview"] = len(preview_ids)
    windows["final"] = len(final_ids)
    stats = windows["stats"]
    stats["actual_preview"] = len(preview_ids)
    stats["actual_final"] = len(final_ids)
    stats["paired_windows"] = len(final_ids)
    stats["coverage"]["preview"]["used"] = len(preview_ids)
    stats["coverage"]["final"]["used"] = len(final_ids)
    stats["preview_final_slice_delta_summary"].update(
        {
            "mean": 0.0,
            "ci": list(preview_final_ci),
            "preview_windows": len(preview_ids),
            "final_windows": len(final_ids),
            "degenerate": math.isclose(
                preview_final_ci[0],
                preview_final_ci[1],
                rel_tol=1e-12,
                abs_tol=1e-15,
            ),
            "degenerate_reason": None,
        }
    )

    schedule_digest = _final_window_schedule_digest(final_ids)
    report["provenance"]["window_ids_digest"] = schedule_digest
    report["provenance"]["window_plan_digest"] = schedule_digest
    report["guard_metric_impact"]["schedule_digest"] = schedule_digest
    report["baseline_ref"]["primary_metric"]["final"] = baseline_final
    report["primary_metric"].update(
        {
            "preview": math.exp(subject_mean),
            "final": math.exp(subject_mean),
            "ratio_vs_baseline": math.exp(subject_mean) / baseline_final,
            "ci": list(expected_ci),
            "display_ci": [math.exp(bound) for bound in expected_ci],
            "analysis_point_preview": subject_mean,
            "analysis_point_final": subject_mean,
        }
    )
    baseline = _matching_strict_ppl_baseline(report)
    baseline["metrics"]["primary_metric"]["final"] = baseline_final
    baseline["evaluation_windows"]["final"] = {
        "window_ids": list(final_ids),
        "logloss": list(baseline_losses),
        "token_counts": list(token_counts),
    }
    bare_facts = extract_guard_metric_arm_facts(baseline, "ppl_causal")
    guarded_facts = extract_guard_metric_arm_facts(report, "ppl_causal")
    measurement = compute_guard_metric_impact(
        "ppl_causal", baseline_final, math.exp(subject_mean)
    )
    bare_report = build_guard_metric_bare_report(baseline, "ppl_causal")
    assert bare_facts is not None
    assert guarded_facts is not None
    assert measurement is not None
    assert bare_report is not None
    bare_report["status"] = "success"
    report["guard_metric_impact"].update(
        {
            **measurement.to_metrics(),
            "bare_facts": bare_facts,
            "guarded_facts": guarded_facts,
            "bare_report": bare_report,
            "passed": measurement.degradation <= 0.01,
        }
    )
    return report, baseline


def _invoke(
    tmp_path: Path,
    *,
    report: dict,
    baseline: dict | None,
    profile: str = "ci",
    assurance: str = "strict",
) -> object:
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _write_runtime_manifest(report_path)
    policy_path = _write_matching_strict_policy_pack(report_path, report)
    args = [
        "verify",
        "--profile",
        profile,
        "--assurance",
        assurance,
        "--expected-runtime-image-digest",
        _VALID_TEST_IMAGE_DIGEST,
        "--policy-pack",
        str(policy_path),
    ]
    if baseline is not None:
        baseline_path = tmp_path / "trusted-baseline.json"
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        args.extend(("--baseline", str(baseline_path)))
    args.append(str(report_path))
    return CliRunner().invoke(
        app,
        args,
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )


def _replay_errors(report: dict, baseline: dict) -> list[str]:
    errors: list[str] = []
    append_strict_ppl_bootstrap_replay_errors(
        errors,
        report=report,
        baseline_payload=baseline,
        baseline_supplied=True,
        tolerance=1e-9,
    )
    return errors


def test_strict_ppl_accepts_exact_independent_bootstrap_replay(
    tmp_path: Path,
) -> None:
    report, baseline = _nondegenerate_replay_case()

    assert _replay_errors(report, baseline) == []
    result = _invoke(tmp_path, report=report, baseline=baseline)

    assert result.exit_code == 0, result.output


def test_strict_ppl_rejects_forged_narrow_ci_that_contains_point(
    tmp_path: Path,
) -> None:
    report, baseline = _nondegenerate_replay_case()
    point = math.log(report["primary_metric"]["ratio_vs_baseline"])
    forged = [point - 1e-6, point + 1e-6]
    assert forged[0] <= point <= forged[1]
    report["primary_metric"]["ci"] = forged
    report["primary_metric"]["display_ci"] = [math.exp(bound) for bound in forged]

    errors = _replay_errors(report, baseline)
    result = _invoke(tmp_path, report=report, baseline=baseline)

    assert any("bootstrap CI mismatch" in error for error in errors)
    assert result.exit_code != 0
    assert "INVARLOCK:E602" in result.output


def test_strict_ppl_rejects_subject_baseline_weight_mismatch(
    tmp_path: Path,
) -> None:
    report, baseline = _nondegenerate_replay_case()
    baseline["evaluation_windows"]["final"]["token_counts"][2] += 1

    errors = _replay_errors(report, baseline)
    result = _invoke(tmp_path, report=report, baseline=baseline, profile="ci")

    assert any("identical subject/baseline token_counts" in error for error in errors)
    assert result.exit_code != 0


def test_strict_ppl_rejects_noncanonical_bootstrap_method(tmp_path: Path) -> None:
    report, baseline = _nondegenerate_replay_case()
    report["dataset"]["windows"]["stats"]["bootstrap"]["method"] = "percentile"

    errors = _replay_errors(report, baseline)
    result = _invoke(tmp_path, report=report, baseline=baseline, profile="ci")

    assert any("requires canonical method" in error for error in errors)
    assert result.exit_code != 0
    assert "requires canonical method" in result.output


def test_strict_ppl_replay_is_deterministic_across_repeated_verification(
    tmp_path: Path,
) -> None:
    report, baseline = _nondegenerate_replay_case()

    first_errors = _replay_errors(report, baseline)
    second_errors = _replay_errors(report, baseline)
    first = _invoke(tmp_path, report=report, baseline=baseline)
    second = _invoke(tmp_path, report=report, baseline=baseline)

    assert first_errors == second_errors == []
    assert first.exit_code == second.exit_code == 0
    assert first.output == second.output


@pytest.mark.parametrize("assurance", ["strict", "report"])
def test_strict_ppl_rejects_missing_independently_supplied_baseline(
    tmp_path: Path,
    assurance: str,
) -> None:
    report, _ = _nondegenerate_replay_case()

    result = _invoke(
        tmp_path,
        report=report,
        baseline=None,
        profile="ci",
        assurance=assurance,
    )

    assert result.exit_code != 0
    assert "requires independently supplied --baseline" in result.output


def test_strict_ppl_rejects_digest_only_baseline_without_raw_windows(
    tmp_path: Path,
) -> None:
    report, _ = _nondegenerate_replay_case()
    baseline = {
        "primary_metric": report["baseline_ref"]["primary_metric"],
        "provenance": {
            "window_ids_digest": report["provenance"]["window_ids_digest"],
        },
    }

    result = _invoke(tmp_path, report=report, baseline=baseline, profile="ci")

    assert result.exit_code != 0
    assert "complete canonical noop baseline run report" in result.output


def test_strict_ppl_rejects_baseline_metric_raw_window_fork(
    tmp_path: Path,
) -> None:
    report, baseline = _nondegenerate_replay_case()
    baseline["metrics"]["primary_metric"]["final"] *= 1.1

    errors = _replay_errors(report, baseline)
    result = _invoke(tmp_path, report=report, baseline=baseline, profile="ci")

    assert any("baseline metric/raw-window mismatch" in error for error in errors)
    assert result.exit_code != 0
