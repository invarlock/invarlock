from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import invarlock.core.evaluate_contract as evaluate_contract_mod
from invarlock.core.evaluate_contract import (
    apply_edited_primary_metric_policy,
    load_validated_baseline_report,
    require_run_report_artifact,
)
from invarlock.core.exceptions import ConfigError, ValidationError
from invarlock.core.report_inputs import ReportInputError


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _baseline_payload(
    *,
    adapter: str = "hf_causal",
    profile: str = "dev",
    tier: str = "balanced",
    edit_name: str = "noop",
    evaluation_windows: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "edit": {"name": edit_name},
        "meta": {"adapter": adapter},
        "context": {"profile": profile, "auto": {"tier": tier}},
        "evaluation_windows": evaluation_windows
        or {
            "preview": {"window_ids": ["preview-0"], "input_ids": [[1, 2, 3]]},
            "final": {"window_ids": ["final-0"], "input_ids": [[4, 5, 6]]},
        },
    }


def test_require_run_report_artifact_accepts_explicit_file(tmp_path: Path) -> None:
    report = _write_json(tmp_path / "report.json", {"ok": True})

    resolved = require_run_report_artifact(str(report), stage="Baseline")

    assert resolved == report.resolve()


def test_require_run_report_artifact_rejects_missing_or_non_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigError, match="did not return a report path"):
        require_run_report_artifact(None, stage="Baseline")

    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    with pytest.raises(ConfigError, match="returned a non-file report path"):
        require_run_report_artifact(report_dir, stage="Edited")


def test_load_validated_baseline_report_accepts_valid_explicit_file(
    tmp_path: Path,
) -> None:
    report = _write_json(tmp_path / "report.json", _baseline_payload())

    resolved, payload = load_validated_baseline_report(
        report,
        expected_profile="dev",
        expected_tier="balanced",
        expected_adapter="hf_causal",
    )

    assert resolved == report.resolve()
    assert payload["edit"] == {"name": "noop"}


def test_load_validated_baseline_report_accepts_multimodal_baseline_windows(
    tmp_path: Path,
) -> None:
    report = _write_json(
        tmp_path / "vision-report.json",
        _baseline_payload(
            adapter="hf_multimodal",
            evaluation_windows={
                "preview": {
                    "example_ids": ["red-square"],
                    "records": [{"id": "red-square", "correct": True}],
                },
                "final": {
                    "example_ids": ["green-square"],
                    "records": [{"id": "green-square", "correct": True}],
                },
            },
        ),
    )

    resolved, payload = load_validated_baseline_report(
        report,
        expected_profile="dev",
        expected_tier="balanced",
        expected_adapter="hf_multimodal",
    )

    assert resolved == report.resolve()
    assert payload["meta"] == {"adapter": "hf_multimodal"}


def test_load_validated_baseline_report_accepts_context_without_auto_tier(
    tmp_path: Path,
) -> None:
    report = _write_json(
        tmp_path / "baseline.json",
        {
            **_baseline_payload(),
            "context": {"profile": "dev", "auto": "skip-tier-check"},
        },
    )

    resolved, payload = load_validated_baseline_report(
        report,
        expected_profile="dev",
        expected_tier="balanced",
        expected_adapter="hf_causal",
    )

    assert resolved == report.resolve()
    assert payload["context"] == {"profile": "dev", "auto": "skip-tier-check"}


def test_load_validated_baseline_report_rejects_directory_and_bad_edit(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    with pytest.raises(
        ValidationError, match="must be an explicit report.json file path"
    ):
        load_validated_baseline_report(
            report_dir,
            expected_profile="dev",
            expected_tier="balanced",
            expected_adapter="hf_causal",
        )

    bad_edit = _write_json(
        tmp_path / "baseline.json",
        _baseline_payload(edit_name="quant_rtn"),
    )
    with pytest.raises(ValidationError, match="must be a no-op run"):
        load_validated_baseline_report(
            bad_edit,
            expected_profile="dev",
            expected_tier="balanced",
            expected_adapter="hf_causal",
        )


def test_load_validated_baseline_report_rejects_policy_and_window_mismatches(
    tmp_path: Path,
) -> None:
    mismatched = _write_json(
        tmp_path / "baseline.json",
        _baseline_payload(profile="release"),
    )
    with pytest.raises(ValidationError, match="profile mismatch"):
        load_validated_baseline_report(
            mismatched,
            expected_profile="dev",
            expected_tier="balanced",
            expected_adapter="hf_causal",
        )

    bad_windows = _write_json(
        tmp_path / "bad-windows.json",
        _baseline_payload(
            evaluation_windows={
                "preview": {"window_ids": ["preview-0"], "input_ids": [[1, 2, 3]]},
                "final": {"window_ids": ["final-0", "final-1"], "input_ids": [[4]]},
            }
        ),
    )
    with pytest.raises(ValidationError, match="inconsistent evaluation window"):
        load_validated_baseline_report(
            bad_windows,
            expected_profile="dev",
            expected_tier="balanced",
            expected_adapter="hf_causal",
        )

    missing_records = _write_json(
        tmp_path / "missing-records.json",
        _baseline_payload(
            adapter="hf_multimodal",
            evaluation_windows={
                "preview": {"example_ids": ["ex-1"]},
                "final": {
                    "example_ids": ["ex-2"],
                    "records": [{"id": "ex-2", "correct": True}],
                },
            },
        ),
    )
    with pytest.raises(
        ValidationError, match="missing evaluation_windows.preview.records"
    ):
        load_validated_baseline_report(
            missing_records,
            expected_profile="dev",
            expected_tier="balanced",
            expected_adapter="hf_multimodal",
        )

    inconsistent_records = _write_json(
        tmp_path / "inconsistent-records.json",
        _baseline_payload(
            adapter="hf_multimodal",
            evaluation_windows={
                "preview": {
                    "example_ids": ["ex-1"],
                    "records": [{"id": "ex-1", "correct": True}],
                },
                "final": {
                    "example_ids": ["ex-2", "ex-3"],
                    "records": [{"id": "ex-2", "correct": True}],
                },
            },
        ),
    )
    with pytest.raises(
        ValidationError, match="inconsistent multimodal evaluation payloads"
    ):
        load_validated_baseline_report(
            inconsistent_records,
            expected_profile="dev",
            expected_tier="balanced",
            expected_adapter="hf_multimodal",
        )


def test_load_validated_baseline_report_missing_windows_has_no_env_remediation(
    tmp_path: Path,
) -> None:
    missing_windows = _write_json(
        tmp_path / "missing-windows.json",
        _baseline_payload(evaluation_windows={"preview": {}, "final": {}}),
    )
    with pytest.raises(ValidationError) as excinfo:
        load_validated_baseline_report(
            missing_windows,
            expected_profile="dev",
            expected_tier="balanced",
            expected_adapter="hf_causal",
        )
    assert "INVARLOCK_STORE_EVAL_WINDOWS" not in str(excinfo.value)


def test_load_validated_baseline_report_rejects_non_regular_file(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo unavailable on this platform")

    fifo = tmp_path / "baseline.pipe"
    os.mkfifo(fifo)

    with pytest.raises(ValidationError, match="Baseline report not found"):
        load_validated_baseline_report(
            fifo,
            expected_profile="dev",
            expected_tier="balanced",
            expected_adapter="hf_causal",
        )


def test_apply_edited_primary_metric_policy_does_not_carry_exit_code() -> None:
    outcome = apply_edited_primary_metric_policy(
        {
            "meta": {"device": "cpu", "adapter": "hf_causal"},
            "edit": {"name": "quant_rtn"},
            "metrics": {"primary_metric": {"final": {"bad": "value"}}},
        },
        profile="ci",
    )

    assert outcome.error is not None
    assert outcome.error.code == "E111"
    assert outcome.diagnostic is not None
    assert outcome.diagnostic.code == "evaluate.primary_metric_degraded"
    assert outcome.diagnostic.details["reason"] == "non_finite_pm"
    assert outcome.payload["metrics"]["primary_metric"]["final"] == {"bad": "value"}
    assert "ratio_vs_baseline" not in outcome.payload["metrics"]["primary_metric"]
    assert not hasattr(outcome, "warning")
    assert not hasattr(outcome, "exit_code")


def test_apply_edited_primary_metric_policy_reraises_unexpected_profile_errors() -> (
    None
):
    class _BadProfile:
        def __str__(self) -> str:
            raise AssertionError("explode")

    with pytest.raises(AssertionError, match="explode"):
        apply_edited_primary_metric_policy(
            {
                "meta": {"device": "cpu", "adapter": "hf_causal"},
                "edit": {"name": "quant_rtn"},
                "metrics": {"primary_metric": {"final": 1.0}},
            },
            profile=_BadProfile(),
        )


def test_apply_edited_primary_metric_policy_treats_typeerror_profile_as_non_enforcing() -> (
    None
):
    class _TypeErrorProfile:
        def __str__(self) -> str:
            raise TypeError("bad profile")

    payload = {
        "meta": {"device": "cpu", "adapter": "hf_causal"},
        "edit": {"name": "quant_rtn"},
        "metrics": {"primary_metric": {"final": 1.0}},
    }

    outcome = apply_edited_primary_metric_policy(payload, profile=_TypeErrorProfile())

    assert outcome.error is None
    assert outcome.diagnostic is None
    assert outcome.payload == payload


def test_baseline_input_error_message_formats_unreadable_reports(
    tmp_path: Path,
) -> None:
    path = tmp_path / "baseline.json"
    exc = ReportInputError("unreadable", path, detail="permission denied")

    assert evaluate_contract_mod._baseline_input_error_message(exc) == (
        f"Baseline report is not readable: {path} (permission denied)"
    )
