from __future__ import annotations

from types import SimpleNamespace

import pytest

from invarlock.reporting.run_provenance_contract import (
    finalize_run_provenance,
)


def test_finalize_run_provenance_serializes_windows_and_enforces_parity() -> None:
    report: dict[str, object] = {}
    seen: dict[str, object] = {}

    result = finalize_run_provenance(
        report=report,
        core_report=SimpleNamespace(evaluation_windows={"preview": {}, "final": {}}),
        preview_records=[],
        final_records=[],
        use_mlm=False,
        preview_mask_counts=None,
        final_mask_counts=None,
        had_baseline=True,
        profile="ci",
        resolved_split="validation",
        used_fallback_split=True,
        baseline_report_data={
            "provenance": {"provider_digest": {"ids_sha256": "base"}},
        },
        serialize_evaluation_windows_fn=lambda windows: dict(windows),
        build_fallback_evaluation_windows_fn=lambda *args, **kwargs: {
            "preview": {"window_ids": [0]},
            "final": {"window_ids": [1]},
        },
        compute_provider_digest_fn=lambda payload: {"ids_sha256": "subject"},
        enforce_provider_parity_fn=lambda subject, baseline, profile=None: seen.update(
            {"subject": subject, "baseline": baseline, "profile": profile}
        ),
    )

    assert result.missing_evaluation_windows_for_baseline is False
    assert report["evaluation_windows"] == {
        "preview": {"window_ids": [0]},
        "final": {"window_ids": [1]},
    }
    assert report["provenance"] == {
        "dataset_split": "validation",
        "split_fallback": True,
        "provider_digest": {"ids_sha256": "subject"},
        "digest_version": 1,
    }
    assert seen == {
        "subject": {"ids_sha256": "subject"},
        "baseline": {"ids_sha256": "base"},
        "profile": "ci",
    }


def test_finalize_run_provenance_uses_fallback_when_serialized_windows_are_empty() -> (
    None
):
    report: dict[str, object] = {}

    finalize_run_provenance(
        report=report,
        core_report=SimpleNamespace(evaluation_windows={"preview": {}, "final": {}}),
        preview_records=[{"example_id": "ex-1", "correct": True}],
        final_records=[{"example_id": "ex-2", "correct": False}],
        use_mlm=False,
        preview_mask_counts=None,
        final_mask_counts=None,
        had_baseline=False,
        profile="dev",
        resolved_split="validation",
        used_fallback_split=False,
        baseline_report_data=None,
        serialize_evaluation_windows_fn=lambda windows: dict(windows),
        build_fallback_evaluation_windows_fn=lambda *args, **kwargs: {
            "preview": {
                "example_ids": ["ex-1"],
                "records": [{"example_id": "ex-1", "correct": True}],
            },
            "final": {
                "example_ids": ["ex-2"],
                "records": [{"example_id": "ex-2", "correct": False}],
            },
        },
        compute_provider_digest_fn=lambda payload: {"ids_sha256": "subject"},
        enforce_provider_parity_fn=lambda *args, **kwargs: None,
    )

    assert report["evaluation_windows"] == {
        "preview": {
            "example_ids": ["ex-1"],
            "records": [{"example_id": "ex-1", "correct": True}],
        },
        "final": {
            "example_ids": ["ex-2"],
            "records": [{"example_id": "ex-2", "correct": False}],
        },
    }


def test_finalize_run_provenance_returns_missing_windows_for_release_baseline() -> None:
    report: dict[str, object] = {}

    result = finalize_run_provenance(
        report=report,
        core_report=SimpleNamespace(evaluation_windows=None),
        preview_records=[],
        final_records=[],
        use_mlm=False,
        preview_mask_counts=None,
        final_mask_counts=None,
        had_baseline=True,
        profile="release",
        resolved_split="validation",
        used_fallback_split=False,
        baseline_report_data=None,
        serialize_evaluation_windows_fn=lambda windows: None,
        build_fallback_evaluation_windows_fn=lambda *args, **kwargs: {},
        compute_provider_digest_fn=lambda payload: {"ids_sha256": "subject"},
        enforce_provider_parity_fn=lambda *args, **kwargs: None,
    )

    assert result.missing_evaluation_windows_for_baseline is True
    assert "PAIRING-SCHEDULE-MISMATCH" in (
        result.missing_evaluation_windows_message or ""
    )
    assert "evaluation_windows" not in report


def test_finalize_run_provenance_builds_fallback_and_recomputes_baseline_digest() -> (
    None
):
    report: dict[str, object] = {}
    seen: dict[str, object] = {}

    def _digest(payload):
        if payload is report:
            return {"ids_sha256": "subject"}
        return {"ids_sha256": "baseline"}

    finalize_run_provenance(
        report=report,
        core_report=SimpleNamespace(evaluation_windows=None),
        preview_records=[{"window_ids": [0]}],
        final_records=[{"window_ids": [1]}],
        use_mlm=False,
        preview_mask_counts=None,
        final_mask_counts=None,
        had_baseline=False,
        profile="dev",
        resolved_split="test",
        used_fallback_split=False,
        baseline_report_data={"meta": {}},
        serialize_evaluation_windows_fn=lambda windows: None,
        build_fallback_evaluation_windows_fn=lambda *args, **kwargs: {
            "preview": {"window_ids": [0]},
            "final": {"window_ids": [1]},
        },
        compute_provider_digest_fn=_digest,
        enforce_provider_parity_fn=lambda subject, baseline, profile=None: seen.update(
            {"subject": subject, "baseline": baseline, "profile": profile}
        ),
    )

    assert report["evaluation_windows"] == {
        "preview": {"window_ids": [0]},
        "final": {"window_ids": [1]},
    }
    assert seen == {
        "subject": {"ids_sha256": "subject"},
        "baseline": {"ids_sha256": "baseline"},
        "profile": "dev",
    }


def test_finalize_run_provenance_propagates_parity_failure() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        finalize_run_provenance(
            report={},
            core_report=SimpleNamespace(
                evaluation_windows={
                    "preview": {"window_ids": [0]},
                    "final": {"window_ids": [1]},
                }
            ),
            preview_records=[],
            final_records=[],
            use_mlm=False,
            preview_mask_counts=None,
            final_mask_counts=None,
            had_baseline=True,
            profile="ci",
            resolved_split="validation",
            used_fallback_split=False,
            baseline_report_data={
                "provenance": {"provider_digest": {"ids_sha256": "b"}}
            },
            serialize_evaluation_windows_fn=lambda windows: dict(windows),
            build_fallback_evaluation_windows_fn=lambda *args, **kwargs: {},
            compute_provider_digest_fn=lambda payload: {"ids_sha256": "s"},
            enforce_provider_parity_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
        )
