from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.reporting.verify_strict_accuracy import (
    _append_accuracy_recompute_errors,
)
from invarlock.reporting.verify_strict_ppl import _append_strict_ppl_schedule_errors
from tests.cli.verify._support_runtime_provenance import (
    _VALID_TEST_IMAGE_DIGEST,
    _final_window_schedule_digest,
    _matching_strict_accuracy_baseline,
    _matching_strict_ppl_baseline,
    _strict_accuracy_cert,
    _strict_provenance_gate_cert,
    _write_matching_strict_policy_pack,
    _write_runtime_manifest,
)


def _invoke_strict_accuracy(tmp_path: Path, payload: dict) -> object:
    baseline = _matching_strict_accuracy_baseline(payload)
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_runtime_manifest(cert_path)
    policy_path = _write_matching_strict_policy_pack(cert_path, payload)
    baseline_path = tmp_path / "trusted-accuracy-baseline.json"
    baseline_path.write_text(
        json.dumps(baseline),
        encoding="utf-8",
    )
    return CliRunner().invoke(
        app,
        [
            "verify",
            "--profile",
            "ci",
            "--assurance",
            "strict",
            "--expected-runtime-image-digest",
            _VALID_TEST_IMAGE_DIGEST,
            "--baseline",
            str(baseline_path),
            "--policy-pack",
            str(policy_path),
            str(cert_path),
        ],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )


def _invoke_strict_ppl(tmp_path: Path, payload: dict) -> object:
    baseline = _matching_strict_ppl_baseline(payload)
    cert_path = tmp_path / "evaluation.report.json"
    cert_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_runtime_manifest(cert_path)
    policy_path = _write_matching_strict_policy_pack(cert_path, payload)
    baseline_path = tmp_path / "trusted-baseline.json"
    baseline_path.write_text(
        json.dumps(baseline),
        encoding="utf-8",
    )
    return CliRunner().invoke(
        app,
        [
            "verify",
            "--profile",
            "ci",
            "--assurance",
            "strict",
            "--expected-runtime-image-digest",
            _VALID_TEST_IMAGE_DIGEST,
            "--baseline",
            str(baseline_path),
            "--policy-pack",
            str(policy_path),
            str(cert_path),
        ],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )


def test_strict_accuracy_cross_reconciled_evidence_passes(
    tmp_path: Path,
) -> None:
    result = _invoke_strict_accuracy(tmp_path, _strict_accuracy_cert())
    assert result.exit_code == 0


def test_strict_accuracy_rejects_forged_baseline_delta(tmp_path: Path) -> None:
    payload = _strict_accuracy_cert()
    payload["baseline_ref"]["primary_metric"]["final"] = 0.9
    payload["primary_metric"]["delta_vs_baseline_pp"] = 0.0
    payload["primary_metric"]["ratio_vs_baseline"] = 0.0

    result = _invoke_strict_accuracy(tmp_path, payload)

    assert result.exit_code != 0


def test_strict_accuracy_rejects_forged_preview_point(tmp_path: Path) -> None:
    payload = _strict_accuracy_cert()
    payload["primary_metric"]["preview"] = 0.95

    result = _invoke_strict_accuracy(tmp_path, payload)

    assert result.exit_code != 0
    assert "INVARLOCK:E602" in result.output


@pytest.mark.parametrize("bad_count", [0, "bad", 1.5, True])
def test_strict_ppl_rejects_invalid_token_counts(
    tmp_path: Path, bad_count: object
) -> None:
    payload = _strict_provenance_gate_cert()
    payload["evaluation_windows"]["final"]["token_counts"][0] = bad_count

    result = _invoke_strict_ppl(tmp_path, payload)

    assert result.exit_code != 0
    assert "token_counts[0] must be a positive JSON integer" in result.output


@pytest.mark.parametrize("arm", ["preview", "final"])
def test_strict_ppl_requires_window_ids(tmp_path: Path, arm: str) -> None:
    payload = _strict_provenance_gate_cert()
    payload["evaluation_windows"][arm].pop("window_ids")

    result = _invoke_strict_ppl(tmp_path, payload)

    assert result.exit_code != 0
    assert f"requires evaluation_windows.{arm}.window_ids" in result.output


@pytest.mark.parametrize("bad_id", [True, 1.5, None, [], {}, ""])
def test_strict_ppl_rejects_unstable_window_id_types(
    tmp_path: Path, bad_id: object
) -> None:
    payload = _strict_provenance_gate_cert()
    payload["evaluation_windows"]["preview"]["window_ids"] = [bad_id]

    result = _invoke_strict_ppl(tmp_path, payload)

    assert result.exit_code != 0
    assert "window_ids[0] must be a JSON integer or non-empty string" in result.output


def test_strict_ppl_rejects_raw_arm_count_fork(tmp_path: Path) -> None:
    payload = _strict_provenance_gate_cert()
    payload["dataset"]["windows"]["preview"] = 3
    payload["dataset"]["windows"]["stats"]["actual_preview"] = 3
    payload["dataset"]["windows"]["stats"]["coverage"]["preview"]["used"] = 3
    errors: list[str] = []
    _append_strict_ppl_schedule_errors(errors, cert_obj=payload)

    result = _invoke_strict_ppl(tmp_path, payload)

    assert result.exit_code != 0
    assert any(
        "PPL count mismatch: dataset.windows.preview=3 expected=180" in error
        for error in errors
    )
    assert "INVARLOCK:E602" in result.output


def test_strict_ppl_rejects_preview_final_window_overlap(tmp_path: Path) -> None:
    payload = _strict_provenance_gate_cert()
    payload["evaluation_windows"]["final"]["window_ids"] = [0]
    schedule_digest = _final_window_schedule_digest([0])
    payload["provenance"]["window_ids_digest"] = schedule_digest
    payload["provenance"]["window_plan_digest"] = schedule_digest
    payload["guard_metric_impact"]["schedule_digest"] = schedule_digest

    result = _invoke_strict_ppl(tmp_path, payload)

    assert result.exit_code != 0
    assert "preview/final window_ids must be disjoint" in result.output


@pytest.mark.parametrize(
    ("container", "field", "expected_path"),
    [
        ("provenance", "window_ids_digest", "provenance.window_ids_digest"),
        ("provenance", "window_plan_digest", "provenance.window_plan_digest"),
        (
            "guard_metric_impact",
            "schedule_digest",
            "guard_metric_impact.schedule_digest",
        ),
    ],
)
def test_strict_ppl_rejects_schedule_digest_fork(
    tmp_path: Path,
    container: str,
    field: str,
    expected_path: str,
) -> None:
    payload = _strict_provenance_gate_cert()
    payload[container][field] = "0" * 32
    errors: list[str] = []
    _append_strict_ppl_schedule_errors(errors, cert_obj=payload)

    result = _invoke_strict_ppl(tmp_path, payload)

    assert result.exit_code != 0
    assert any(
        f"PPL schedule digest differs: {expected_path}" in error for error in errors
    )


def test_strict_ppl_reconciles_preview_and_display_points(tmp_path: Path) -> None:
    payload = _strict_provenance_gate_cert()
    payload["evaluation_windows"]["preview"]["logloss"] = [math.log(9.0) + 100.0]

    result = _invoke_strict_ppl(tmp_path, payload)

    assert result.exit_code != 0
    assert "INVARLOCK:E602" in result.output


def test_strict_ppl_rejects_ci_disconnected_from_baseline_point(
    tmp_path: Path,
) -> None:
    payload = _strict_provenance_gate_cert()
    payload["primary_metric"]["ci"] = [-10.0, -9.0]
    payload["primary_metric"]["display_ci"] = [math.exp(-10.0), math.exp(-9.0)]

    result = _invoke_strict_ppl(tmp_path, payload)

    assert result.exit_code != 0
    assert "INVARLOCK:E602" in result.output


def test_strict_accuracy_cannot_omit_n_final(
    tmp_path: Path,
) -> None:
    payload = _strict_accuracy_cert()
    payload["primary_metric"].pop("n_final")
    result = _invoke_strict_accuracy(tmp_path, payload)
    assert result.exit_code == 1
    assert "requires a positive integer primary_metric.n_final" in result.output


def test_strict_accuracy_rejects_top_level_nested_count_fork(
    tmp_path: Path,
) -> None:
    payload = _strict_accuracy_cert()
    payload["metrics"]["classification"]["n_correct"] = 1
    payload["metrics"]["classification"]["n_total"] = 1
    payload["primary_metric"]["n_final"] = 1
    payload["primary_metric"]["final"] = 1.0
    result = _invoke_strict_accuracy(tmp_path, payload)
    assert result.exit_code != 0
    assert "INVARLOCK:E602" in result.output


def test_strict_accuracy_rejects_window_coverage_count_fork(
    tmp_path: Path,
) -> None:
    payload = _strict_accuracy_cert()
    payload["dataset"]["windows"]["stats"]["coverage"]["final"]["used"] = 199
    result = _invoke_strict_accuracy(tmp_path, payload)
    assert result.exit_code != 0
    assert "INVARLOCK:E602" in result.output


@pytest.mark.parametrize(
    ("path", "malformed_value", "expected_message"),
    [
        (
            ("metrics", "classification", "final", "example_correct"),
            [1, 0, "yes"],
            "metrics.classification.final.example_correct",
        ),
        (
            ("evaluation_windows", "final", "records"),
            [{"correct": True}, {"label": "missing-correct"}],
            "evaluation_windows.final.records",
        ),
        (
            ("evaluation_windows", "final", "example_ids"),
            "not-a-list",
            "evaluation_windows.final.example_ids",
        ),
        (
            (
                "dataset",
                "windows",
                "stats",
                "coverage",
                "final",
                "used",
            ),
            "200",
            "dataset.windows.stats.coverage.final.used",
        ),
    ],
)
def test_strict_accuracy_rejects_present_but_malformed_optional_evidence(
    path: tuple[str, ...],
    malformed_value: object,
    expected_message: str,
) -> None:
    payload = _strict_accuracy_cert()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = malformed_value
    errors: list[str] = []

    usable = _append_accuracy_recompute_errors(
        errors,
        cert_obj=payload,
        pm=payload["primary_metric"],
        tol=1e-9,
        require_strict=True,
    )

    assert usable is True
    assert any(expected_message in error for error in errors)


def test_strict_accuracy_rejects_counts_above_float_precision_with_correct_gt_total() -> (
    None
):
    payload = _strict_accuracy_cert()
    payload["metrics"]["classification"]["n_correct"] = (2**53) + 1
    payload["metrics"]["classification"]["n_total"] = 2**53
    payload["primary_metric"]["n_final"] = (2**53) + 1
    payload["primary_metric"]["final"] = 1.0
    # Remove smaller raw surfaces so the assertion targets exact aggregate
    # integer handling rather than failing first on a record-length mismatch.
    payload["metrics"]["classification"].pop("final")
    payload.pop("evaluation_windows")
    payload["dataset"].pop("windows")
    errors: list[str] = []

    usable = _append_accuracy_recompute_errors(
        errors,
        cert_obj=payload,
        pm=payload["primary_metric"],
        tol=1e-9,
        require_strict=True,
    )

    assert usable is True
    assert any("n_correct exceeds n_total" in error for error in errors)
