from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.reporting import verify_contract as verify_mod
from invarlock.reporting.report_provenance import compute_report_digest
from invarlock.reporting.verify_baseline import append_strict_baseline_contract_errors
from invarlock.reporting.verify_contract import VerifyOutcome
from invarlock.reporting.verify_strict_schedule import (
    _append_strict_supplied_baseline_binding_errors,
    _schedule_digest,
)
from tests.cli._support_verify_runtime_provenance import (
    _VALID_TEST_IMAGE_DIGEST,
    _final_window_schedule_digest,
    _matching_strict_accuracy_baseline,
    _matching_strict_ppl_baseline,
    _strict_accuracy_cert,
    _strict_provenance_gate_cert,
    _write_matching_strict_policy_pack,
    _write_runtime_manifest,
)


def _invoke_strict_with_baseline(
    tmp_path: Path,
    *,
    report: dict,
    baseline: dict,
    profile: str = "ci",
) -> object:
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _write_runtime_manifest(report_path)
    policy_path = _write_matching_strict_policy_pack(report_path, report)
    baseline_path = tmp_path / "trusted-baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    return CliRunner().invoke(
        app,
        [
            "verify",
            "--profile",
            profile,
            "--assurance",
            "strict",
            "--expected-runtime-image-digest",
            _VALID_TEST_IMAGE_DIGEST,
            "--baseline",
            str(baseline_path),
            "--policy-pack",
            str(policy_path),
            str(report_path),
        ],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )


def _matching_ppl_baseline() -> dict:
    return _matching_strict_ppl_baseline()


def _strict_baseline_contract_errors(report: dict, baseline: dict) -> list[str]:
    errors: list[str] = []
    append_strict_baseline_contract_errors(
        errors,
        report=report,
        baseline_payload=baseline,
        baseline_supplied=True,
        tolerance=1e-9,
    )
    return errors


def test_strict_ppl_accepts_independently_matching_baseline(tmp_path: Path) -> None:
    result = _invoke_strict_with_baseline(
        tmp_path,
        report=_strict_provenance_gate_cert(),
        baseline=_matching_ppl_baseline(),
    )

    assert result.exit_code == 0, result.output


def test_strict_ppl_accepts_matching_baseline_schedule_digest(tmp_path: Path) -> None:
    report = _strict_provenance_gate_cert()
    baseline = _matching_strict_ppl_baseline(report)
    baseline["provenance"]["window_ids_digest"] = _final_window_schedule_digest(
        baseline["evaluation_windows"]["final"]["window_ids"]
    )
    baseline_hash = compute_report_digest(baseline)
    report["baseline_ref"]["report_hash"] = baseline_hash
    report["provenance"]["baseline"]["report_hash"] = baseline_hash

    result = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=baseline,
        profile="ci",
    )

    assert result.exit_code == 0, result.output


def test_strict_ppl_rejects_subject_report_as_its_own_baseline(
    tmp_path: Path,
) -> None:
    report = _strict_provenance_gate_cert()
    report["provenance"]["provider_digest"]["tokenizer_sha256"] = "tokenizer"
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    _write_runtime_manifest(report_path)
    policy_path = _write_matching_strict_policy_pack(report_path, report)

    result = CliRunner().invoke(
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
            str(report_path),
            "--policy-pack",
            str(policy_path),
            str(report_path),
        ],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code != 0
    assert "baseline file distinct from the subject report" in result.output


def test_strict_rejects_byte_identical_subject_at_different_baseline_path(
    tmp_path: Path,
) -> None:
    report = _strict_provenance_gate_cert()
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _write_runtime_manifest(report_path)
    policy_path = _write_matching_strict_policy_pack(report_path, report)
    copied_baseline_path = tmp_path / "copied-baseline.json"
    copied_baseline_path.write_bytes(report_path.read_bytes())

    result = CliRunner().invoke(
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
            str(copied_baseline_path),
            "--policy-pack",
            str(policy_path),
            str(report_path),
        ],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code != 0
    assert "byte-identical subject copied" in result.output


def test_strict_rejects_baseline_with_subject_run_id(tmp_path: Path) -> None:
    report = _strict_provenance_gate_cert()
    baseline = _matching_ppl_baseline()
    baseline["meta"]["run_id"] = report["run_id"]

    result = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=baseline,
        profile="ci",
    )

    assert result.exit_code != 0
    assert "subject and baseline run IDs" in result.output


def test_strict_rejects_non_noop_baseline_role(tmp_path: Path) -> None:
    report = _strict_provenance_gate_cert()
    baseline = _matching_ppl_baseline()
    baseline["edit"]["name"] = "quant_rtn"

    result = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=baseline,
        profile="ci",
    )

    assert result.exit_code != 0
    assert "complete canonical noop baseline run report" in result.output


@pytest.mark.parametrize(
    ("binding_path", "replacement", "expected_message"),
    [
        (("baseline_ref", "run_id"), "other-run", "baseline_ref run_id mismatch"),
        (
            ("provenance", "baseline", "report_hash"),
            "0" * 64,
            "baseline provenance",
        ),
    ],
)
def test_strict_rejects_subject_baseline_reference_fork(
    tmp_path: Path,
    binding_path: tuple[str, ...],
    replacement: str,
    expected_message: str,
) -> None:
    report = _strict_provenance_gate_cert()
    current = report
    for segment in binding_path[:-1]:
        current = current[segment]
    current[binding_path[-1]] = replacement

    result = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=_matching_ppl_baseline(),
        profile="ci",
    )

    assert result.exit_code != 0
    assert expected_message in result.output


def test_strict_ppl_rejects_forged_embedded_baseline_final(tmp_path: Path) -> None:
    report = _strict_provenance_gate_cert()
    baseline = _matching_ppl_baseline()
    baseline["metrics"]["primary_metric"]["final"] = 99.0
    errors: list[str] = []
    _append_strict_supplied_baseline_binding_errors(
        errors,
        cert_obj=report,
        baseline_payload=baseline,
        baseline_supplied=True,
        tolerance=1e-9,
    )

    result = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=baseline,
    )

    assert result.exit_code != 0
    assert "INVARLOCK:E602" in result.output
    assert any("Supplied baseline final mismatch" in error for error in errors)


def test_strict_ppl_rejects_raw_schedule_fork_even_with_matching_digest(
    tmp_path: Path,
) -> None:
    baseline = _matching_ppl_baseline()
    baseline["evaluation_windows"]["final"]["window_ids"] = [999, 3]
    baseline["provenance"] = {
        "window_ids_digest": _final_window_schedule_digest([2, 3])
    }
    report = _strict_provenance_gate_cert()
    errors: list[str] = []
    _append_strict_supplied_baseline_binding_errors(
        errors,
        cert_obj=report,
        baseline_payload=baseline,
        baseline_supplied=True,
        tolerance=1e-9,
    )

    result = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=baseline,
    )

    assert result.exit_code != 0
    assert "INVARLOCK:E602" not in result.output
    assert "requires equal-length raw supplied_baseline" in result.output
    assert any("Supplied baseline final schedule mismatch" in error for error in errors)


def test_strict_ppl_rejects_mismatched_baseline_schedule_digest(
    tmp_path: Path,
) -> None:
    report = _strict_provenance_gate_cert()
    baseline = {
        "primary_metric": {"kind": "ppl_causal", "final": 9.0},
        "provenance": {
            "window_ids_digest": _final_window_schedule_digest([999, 3]),
        },
    }
    errors: list[str] = []
    _append_strict_supplied_baseline_binding_errors(
        errors,
        cert_obj=report,
        baseline_payload=baseline,
        baseline_supplied=True,
        tolerance=1e-9,
    )

    result = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=baseline,
    )

    assert result.exit_code != 0
    assert "INVARLOCK:E602" not in result.output
    assert "must be the complete canonical noop baseline run report" in result.output
    assert any("schedule digest mismatch" in error for error in errors)


def test_strict_ppl_rejects_ambiguous_legacy_digest_only_string_schedule(
    tmp_path: Path,
) -> None:
    report = _strict_provenance_gate_cert()
    report["evaluation_windows"]["preview"] = {
        "window_ids": ["preview-a", "preview-b"],
        "logloss": report["evaluation_windows"]["preview"]["logloss"] * 2,
        "token_counts": [1, 1],
    }
    report["evaluation_windows"]["final"] = {
        "window_ids": ["a", "bc"],
        "logloss": report["evaluation_windows"]["final"]["logloss"] * 2,
        "token_counts": [1, 1],
    }
    report["dataset"]["windows"]["preview"] = 2
    report["dataset"]["windows"]["final"] = 2
    stats = report["dataset"]["windows"]["stats"]
    stats["actual_preview"] = 2
    stats["actual_final"] = 2
    stats["paired_windows"] = 2
    stats["coverage"]["preview"]["used"] = 2
    stats["coverage"]["final"]["used"] = 2
    subject_digest = _schedule_digest(["a", "bc"])
    colliding_digest = _schedule_digest(["ab", "c"])
    assert subject_digest == colliding_digest
    report["provenance"]["window_ids_digest"] = subject_digest
    report["provenance"]["window_plan_digest"] = subject_digest
    report["guard_metric_impact"]["schedule_digest"] = subject_digest
    baseline = {
        "primary_metric": {"kind": "ppl_causal", "final": 9.0},
        "provenance": {"window_ids_digest": colliding_digest},
    }

    result = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=baseline,
    )

    assert result.exit_code != 0
    assert "INVARLOCK:E602" in result.output


@pytest.mark.parametrize(
    ("baseline", "expected_message"),
    [
        (
            {"evaluation_windows": {"final": {"window_ids": [2, 3]}}},
            "complete canonical noop baseline run report",
        ),
        (
            {"primary_metric": {"kind": "ppl_causal", "final": 9.0}},
            "complete canonical noop baseline run report",
        ),
    ],
)
def test_strict_ppl_rejects_missing_supplied_baseline_binding_evidence(
    tmp_path: Path,
    baseline: dict,
    expected_message: str,
) -> None:
    result = _invoke_strict_with_baseline(
        tmp_path,
        report=_strict_provenance_gate_cert(),
        baseline=baseline,
        profile="ci",
    )

    assert result.exit_code != 0
    assert expected_message in result.output


def test_strict_accuracy_compares_percentage_units_to_supplied_baseline(
    tmp_path: Path,
) -> None:
    report = _strict_accuracy_cert()
    matching = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=_matching_strict_accuracy_baseline(report),
    )
    assert matching.exit_code == 0, matching.output

    mismatched_baseline = _matching_strict_accuracy_baseline(report)
    mismatched_baseline["metrics"]["primary_metric"]["final"] = 0.9
    errors: list[str] = []
    _append_strict_supplied_baseline_binding_errors(
        errors,
        cert_obj=report,
        baseline_payload=mismatched_baseline,
        baseline_supplied=True,
        tolerance=1e-9,
    )
    mismatched = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=mismatched_baseline,
    )
    assert mismatched.exit_code != 0
    assert "INVARLOCK:E602" in mismatched.output
    assert any("Supplied baseline final mismatch" in error for error in errors)


def test_strict_accuracy_requires_independently_supplied_baseline(
    tmp_path: Path,
) -> None:
    report = _strict_accuracy_cert()
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _write_runtime_manifest(report_path)
    policy_path = _write_matching_strict_policy_pack(report_path, report)

    result = CliRunner().invoke(
        app,
        [
            "verify",
            "--profile",
            "ci",
            "--assurance",
            "strict",
            "--expected-runtime-image-digest",
            _VALID_TEST_IMAGE_DIGEST,
            "--policy-pack",
            str(policy_path),
            str(report_path),
        ],
        env={"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "0"},
    )

    assert result.exit_code != 0
    assert "requires a independently supplied --baseline" in result.output


def test_strict_accuracy_rejects_handwritten_metric_fragment(tmp_path: Path) -> None:
    result = _invoke_strict_with_baseline(
        tmp_path,
        report=_strict_accuracy_cert(),
        baseline={"primary_metric": {"kind": "accuracy", "final": 0.8}},
        profile="ci",
    )

    assert result.exit_code != 0
    assert "complete canonical noop baseline run report" in result.output


@pytest.mark.parametrize(
    ("path", "expected_message"),
    [
        (("provenance", "provider_digest", "ids_sha256"), "PROVIDER-DIGEST-MISSING"),
        (
            ("provenance", "provider_digest", "tokenizer_sha256"),
            "PROVIDER-DIGEST-MISSING",
        ),
        (("meta", "run_id"), "meta.run_id"),
        (("meta", "model_id"), "meta.model_id"),
        (("data", "split"), "data.split"),
        (("data", "dataset_hash"), "data.dataset_hash"),
    ],
)
def test_strict_baseline_rejects_missing_provenance_identity(
    tmp_path: Path,
    path: tuple[str, ...],
    expected_message: str,
) -> None:
    report = _strict_accuracy_cert()
    baseline = _matching_strict_accuracy_baseline(report)
    current = baseline
    for segment in path[:-1]:
        current = current[segment]
    current.pop(path[-1])

    result = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=baseline,
        profile="ci",
    )

    assert result.exit_code != 0
    assert expected_message in result.output


def test_strict_baseline_requires_report_tokenizer_hash_surface(
    tmp_path: Path,
) -> None:
    report = _strict_accuracy_cert()
    baseline = _matching_strict_accuracy_baseline(report)
    baseline["meta"].pop("tokenizer_hash")
    baseline["data"].pop("tokenizer_hash")

    result = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=baseline,
        profile="ci",
    )

    assert result.exit_code != 0
    assert "non-empty tokenizer hash" in result.output


@pytest.mark.parametrize(
    ("path", "replacement", "expected_message"),
    [
        (("meta", "model_id"), "other-model", "baseline_ref model_id"),
        (("data", "dataset"), "other-provider", "baseline dataset provider"),
        (
            ("data", "dataset_hash"),
            "other-dataset-digest",
            "baseline dataset identity",
        ),
        (
            ("provenance", "provider_digest", "ids_sha256"),
            "other-schedule",
            "IDS-DIGEST-MISMATCH",
        ),
        (("meta", "tokenizer_hash"), "other-tokenizer", "tokenizer"),
    ],
)
def test_strict_baseline_rejects_identity_parity_forks(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: object,
    expected_message: str,
) -> None:
    report = _strict_accuracy_cert()
    baseline = _matching_strict_accuracy_baseline(report)
    current = baseline
    for segment in path[:-1]:
        current = current[segment]
    current[path[-1]] = replacement

    result = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=baseline,
        profile="ci",
    )

    assert result.exit_code != 0
    assert expected_message in result.output


@pytest.mark.parametrize(
    ("path", "replacement", "expected_message"),
    [
        (("context", "profile"), "release", "profile mismatch"),
        (("context", "auto", "tier"), "minimal", "tier mismatch"),
        (
            ("context", "assurance", "mode"),
            "off",
            "context.assurance.mode='strict'",
        ),
    ],
)
def test_strict_baseline_rejects_execution_context_forks(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: str,
    expected_message: str,
) -> None:
    report = _strict_accuracy_cert()
    baseline = _matching_strict_accuracy_baseline(report)
    current = baseline
    for segment in path[:-1]:
        current = current[segment]
    current[path[-1]] = replacement

    errors = _strict_baseline_contract_errors(report, baseline)
    result = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=baseline,
        profile="ci",
    )

    assert result.exit_code != 0
    assert any(expected_message in error for error in errors)


def test_strict_baseline_rejects_incomplete_preview_raw_arm(tmp_path: Path) -> None:
    report = _strict_provenance_gate_cert()
    baseline = _matching_strict_ppl_baseline(report)
    baseline["evaluation_windows"]["preview"] = {}

    errors = _strict_baseline_contract_errors(report, baseline)
    result = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=baseline,
        profile="ci",
    )

    assert result.exit_code != 0
    assert any("preview.window_ids" in error for error in errors)


def test_strict_accuracy_rejects_baseline_raw_count_fork(tmp_path: Path) -> None:
    report = _strict_accuracy_cert()
    baseline = _matching_strict_accuracy_baseline(report)
    baseline["evaluation_windows"]["final"]["records"][0]["correct"] = False
    errors = _strict_baseline_contract_errors(report, baseline)

    result = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=baseline,
        profile="ci",
    )

    assert result.exit_code != 0
    assert any(
        "raw/count mismatch" in error or "raw evidence fork" in error
        for error in errors
    )


def test_strict_accuracy_rejects_baseline_sample_schedule_fork(tmp_path: Path) -> None:
    report = _strict_accuracy_cert()
    baseline = _matching_strict_accuracy_baseline(report)
    baseline["evaluation_windows"]["final"]["example_ids"][0] = "other-example"
    errors = _strict_baseline_contract_errors(report, baseline)

    result = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=baseline,
        profile="ci",
    )

    assert result.exit_code != 0
    assert any("identical final example_ids" in error for error in errors)


def test_strict_ppl_ci_transform_overflow_is_controlled_failure(
    tmp_path: Path,
) -> None:
    report = _strict_provenance_gate_cert()
    report["primary_metric"]["ci"] = [1000.0, 1000.0]
    report["primary_metric"]["display_ci"] = [1.0, 1.0]

    result = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=_matching_ppl_baseline(),
    )

    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "INVARLOCK:E602" in result.output


def test_strict_ppl_rejects_negative_logloss_and_underflowed_display(
    tmp_path: Path,
) -> None:
    report = _strict_provenance_gate_cert()
    for arm in ("preview", "final"):
        report["evaluation_windows"][arm]["logloss"] = [-1000.0]
    report["primary_metric"]["analysis_point_preview"] = -1000.0
    report["primary_metric"]["analysis_point_final"] = -1000.0
    report["primary_metric"]["preview"] = 5e-324
    report["primary_metric"]["final"] = 5e-324
    report["baseline_ref"]["primary_metric"]["final"] = 5e-324
    baseline = _matching_ppl_baseline()
    baseline["metrics"]["primary_metric"]["final"] = 5e-324

    result = _invoke_strict_with_baseline(
        tmp_path,
        report=report,
        baseline=baseline,
    )

    assert result.exit_code != 0
    assert "INVARLOCK:E602" in result.output


def test_report_path_swap_after_snapshot_does_not_change_verified_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = tmp_path / "evaluation.report.json"
    original_report = _strict_provenance_gate_cert()
    report_path.write_text(json.dumps(original_report), encoding="utf-8")
    _write_runtime_manifest(report_path)
    replacement = _strict_provenance_gate_cert()
    replacement["primary_metric"]["ratio_vs_baseline"] = 7.0
    original_validate = verify_mod._validate_evaluation_report_payload

    def _swap_after_snapshot(path: Path, **kwargs: object) -> list[str]:
        path.write_text(json.dumps(replacement), encoding="utf-8")
        return original_validate(path, **kwargs)

    monkeypatch.setattr(
        verify_mod,
        "_validate_evaluation_report_payload",
        _swap_after_snapshot,
    )

    baseline_path = tmp_path / "trusted-baseline.json"
    baseline_path.write_text(
        json.dumps(_matching_ppl_baseline()),
        encoding="utf-8",
    )
    policy_path = _write_matching_strict_policy_pack(report_path, original_report)

    result = verify_mod.run_verify_reports(
        [report_path],
        baseline=baseline_path,
        policy_pack=policy_path,
        profile="ci",
        assurance_mode="strict",
        expected_runtime_image_digest=_VALID_TEST_IMAGE_DIGEST,
    )

    assert result.outcome == VerifyOutcome.OK
    assert result.payload["results"][0]["ratio_vs_baseline"] == 1.0
    assert (
        json.loads(report_path.read_text(encoding="utf-8"))["primary_metric"][
            "ratio_vs_baseline"
        ]
        == 7.0
    )


def test_baseline_path_swap_after_snapshot_fails_against_original_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text(
        json.dumps(_strict_provenance_gate_cert()),
        encoding="utf-8",
    )
    _write_runtime_manifest(report_path)
    baseline_path = tmp_path / "trusted-baseline.json"
    mismatched_baseline = _matching_ppl_baseline()
    mismatched_baseline["metrics"]["primary_metric"]["final"] = 99.0
    baseline_path.write_text(json.dumps(mismatched_baseline), encoding="utf-8")
    policy_path = _write_matching_strict_policy_pack(
        report_path, _strict_provenance_gate_cert()
    )
    original_validate = verify_mod._validate_evaluation_report_payload

    def _replace_with_matching_baseline(path: Path, **kwargs: object) -> list[str]:
        baseline_path.write_text(
            json.dumps(_matching_ppl_baseline()),
            encoding="utf-8",
        )
        return original_validate(path, **kwargs)

    monkeypatch.setattr(
        verify_mod,
        "_validate_evaluation_report_payload",
        _replace_with_matching_baseline,
    )

    result = verify_mod.run_verify_reports(
        [report_path],
        baseline=baseline_path,
        policy_pack=policy_path,
        profile="ci",
        assurance_mode="strict",
        expected_runtime_image_digest=_VALID_TEST_IMAGE_DIGEST,
    )

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    assert result.error is not None
    assert "INVARLOCK:E602" in str(result.error)
    assert "metric/raw-window mismatch" in str(getattr(result.error, "details", {}))
