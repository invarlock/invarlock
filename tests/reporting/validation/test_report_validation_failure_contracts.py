from __future__ import annotations

from pathlib import Path

from invarlock import runtime_verify
from invarlock.reporting.validation.report import (
    _apply_metric_specific_primary_metric_gate,
)


def test_tiny_relax_accuracy_rejects_fixture_without_sample_count() -> None:
    flags: dict[str, bool] = {}
    _apply_metric_specific_primary_metric_gate(
        flags,
        primary_metric={"kind": "accuracy"},
        metrics_policy={"accuracy": {"delta_min_pp": 0.0, "min_examples": 200}},
        ratio_limit_with_hyst=1.0,
        tokens_ok_eff=False,
        compression_acceptable=False,
        tiny_relax=True,
        dataset_capacity=None,
    )

    assert flags["primary_metric_acceptable"] is False


def test_runtime_digest_helpers_fail_closed_on_missing_and_invalid_values() -> None:
    assert runtime_verify._normalize_expected_image_digest(None) is None
    assert (
        runtime_verify._declared_image_digest({"runtime": {"image_digest": 7}}) is None
    )
    assert runtime_verify._expected_image_digest_errors(
        declared_image_digest=None,
        expected_image_digest="not-a-digest",
    ) == ["expected runtime image digest must match sha256:<64 lowercase hex chars>"]


def test_runtime_verify_preserves_load_and_digest_errors(tmp_path: Path) -> None:
    result = runtime_verify.verify_runtime_manifest(
        tmp_path / "missing-report.json",
        tmp_path / "missing-manifest.json",
        expected_image_digest="not-a-digest",
    )

    assert result.ok is False
    assert result.binding_verified is False
    assert any(error.startswith("unable to read report:") for error in result.errors)
    assert (
        "expected runtime image digest must match sha256:<64 lowercase hex chars>"
        in result.errors
    )
