from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.reporting.oss_exports import (
    VerifyResultMismatchError,
    build_report_export_context,
    render_mlflow_tags_export,
    render_model_card_evidence_block,
    render_release_review_packet,
)


def _passing_report() -> dict[str, object]:
    return {
        "schema_version": "v1",
        "run_id": "run-123",
        "meta": {"model_id": "subject-model"},
        "baseline_ref": {"model_id": "baseline-model"},
        "edit": {"name": "quant_rtn"},
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 2.91,
            "ratio_vs_baseline": 1.0246,
        },
        "validation": {
            "invariants_pass": True,
            "preview_final_drift_acceptable": True,
            "primary_metric_acceptable": True,
            "rmt_stable": True,
            "spectral_stable": True,
        },
    }


def _write_report(tmp_path: Path, report: dict[str, object]) -> Path:
    path = tmp_path / "evaluation.report.json"
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return path


def test_verify_result_overrides_report_local_status(tmp_path: Path) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)
    verify_result = {
        "format_version": "verify-v1",
        "summary": {"ok": False, "reason": "policy_fail"},
        "results": [
            {
                "id": str(report_path),
                "ok": False,
                "reason": "policy_fail",
                "verification": {
                    "runtime_provenance": {
                        "status": "failed",
                        "verified": False,
                    }
                },
            }
        ],
    }

    context = build_report_export_context(
        report_path,
        report,
        policy_profile="ci",
        verify_result=verify_result,
    )
    tags = render_mlflow_tags_export(context)["tags"]

    assert context.status == "fail"
    assert context.verifier_status == "fail"
    assert context.runtime_provenance_status == "failed"
    assert tags["invarlock.status"] == "fail"
    assert tags["invarlock.verifier_reason"] == "policy_fail"
    assert tags["invarlock.runtime_provenance_status"] == "failed"


def test_verify_result_rejects_stale_explicit_id(tmp_path: Path) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)
    other_report = tmp_path / "other-evaluation.report.json"
    verify_result = {
        "format_version": "verify-v1",
        "summary": {"ok": True, "reason": "ok"},
        "results": [
            {
                "id": str(other_report),
                "ok": True,
                "reason": "ok",
            }
        ],
    }

    with pytest.raises(
        VerifyResultMismatchError,
        match="does not contain an item for evaluation report",
    ):
        build_report_export_context(
            report_path,
            report,
            policy_profile="ci",
            verify_result=verify_result,
        )


def test_verify_result_rejects_single_idless_result(tmp_path: Path) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)
    verify_result = {
        "format_version": "verify-v1",
        "summary": {"ok": False, "reason": "policy_fail"},
        "results": [
            {
                "ok": False,
                "reason": "policy_fail",
            }
        ],
    }

    with pytest.raises(VerifyResultMismatchError, match="must include an id"):
        build_report_export_context(
            report_path,
            report,
            policy_profile="ci",
            verify_result=verify_result,
        )


def test_verify_result_rejects_non_object_result_item(tmp_path: Path) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)
    verify_result = {
        "format_version": "verify-v1",
        "summary": {"ok": False, "reason": "policy_fail"},
        "results": ["not-an-object"],
    }

    with pytest.raises(VerifyResultMismatchError, match="must include an id"):
        build_report_export_context(
            report_path,
            report,
            policy_profile="ci",
            verify_result=verify_result,
        )


def test_verify_result_rejects_missing_results(tmp_path: Path) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)
    verify_result = {
        "format_version": "verify-v1",
        "summary": {"ok": True, "reason": "ok"},
    }

    with pytest.raises(VerifyResultMismatchError, match="non-empty results list"):
        build_report_export_context(
            report_path,
            report,
            policy_profile="ci",
            verify_result=verify_result,
        )


def test_verify_result_uses_matching_item_from_batch(tmp_path: Path) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)
    other_report = tmp_path / "other-evaluation.report.json"
    verify_result = {
        "format_version": "verify-v1",
        "summary": {"ok": False, "reason": "batch_fail"},
        "results": [
            {
                "id": str(other_report),
                "ok": False,
                "reason": "other_policy_fail",
            },
            {
                "id": str(report_path),
                "ok": True,
                "reason": "ok",
            },
        ],
    }

    context = build_report_export_context(
        report_path,
        report,
        policy_profile="ci",
        verify_result=verify_result,
    )

    assert context.status == "pass"
    assert context.verifier_status == "pass"
    assert context.verifier_reason == "ok"


def test_mlflow_export_includes_registry_search_tags(tmp_path: Path) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)

    context = build_report_export_context(
        report_path,
        report,
        policy_profile="release",
        report_url="https://example.test/evaluation.report.json",
        evidence_url="https://example.test/evidence.zip",
    )
    tags = render_mlflow_tags_export(context)["tags"]

    assert tags["invarlock.edit_name"] == "quant_rtn"
    assert tags["invarlock.primary_metric_kind"] == "ppl_causal"
    assert tags["invarlock.primary_metric_final"] == "2.91"
    assert tags["invarlock.primary_metric_ratio_vs_baseline"] == "1.0246"
    assert tags["invarlock.report_url"] == "https://example.test/evaluation.report.json"
    assert tags["invarlock.evidence_url"] == "https://example.test/evidence.zip"


def test_mlflow_export_formats_optional_policy_and_metric_fields(
    tmp_path: Path,
) -> None:
    report = _passing_report()
    report["policy_provenance"] = {"policy_digest": "sha256:policy"}
    report["primary_metric"] = {
        "kind": "accuracy",
        "preview": True,
        "final": float("inf"),
    }
    report_path = _write_report(tmp_path, report)

    context = build_report_export_context(report_path, report)
    tags = render_mlflow_tags_export(context)["tags"]

    assert context.primary_metric == "accuracy preview=true final=inf"
    assert tags["invarlock.policy_digest"] == "sha256:policy"
    assert tags["invarlock.primary_metric_final"] == "inf"


def test_mlflow_export_formats_text_metric_values(tmp_path: Path) -> None:
    report = _passing_report()
    report["primary_metric"] = {
        "kind": "accuracy",
        "final": "n/a",
    }
    report_path = _write_report(tmp_path, report)

    context = build_report_export_context(report_path, report)

    assert context.primary_metric == "accuracy final=n/a"
    assert context.primary_metric_final == "n/a"


def test_context_handles_missing_primary_metric(tmp_path: Path) -> None:
    report = _passing_report()
    report.pop("primary_metric")
    report_path = _write_report(tmp_path, report)

    context = build_report_export_context(report_path, report)

    assert context.primary_metric == "unknown"


def test_release_review_counts_only_report_local_gate_failures(
    tmp_path: Path,
) -> None:
    report = _passing_report()
    report["validation"] = {
        "guard_warnings_present": False,
        "hysteresis_applied": False,
        "invariants_pass": True,
        "moe_observed": False,
        "preview_final_drift_acceptable": True,
        "primary_metric_acceptable": True,
        "rmt_stable": True,
        "spectral_stable": True,
    }
    report_path = _write_report(tmp_path, report)

    context = build_report_export_context(report_path, report)
    markdown = render_release_review_packet(context, report)

    assert context.failed_gate_count == 0
    assert "Failed report-local validation gates: `0`" in markdown
    assert "`guard_warnings_present`" not in markdown
    assert "`hysteresis_applied`" not in markdown
    assert "`moe_observed`" not in markdown


def test_release_review_handles_missing_validation_block(tmp_path: Path) -> None:
    report = _passing_report()
    report.pop("validation")
    report_path = _write_report(tmp_path, report)

    context = build_report_export_context(report_path, report)
    markdown = render_release_review_packet(context, report)

    assert "- [ ] No validation gates were present in the report." in markdown


def test_model_card_markdown_escapes_table_cells(tmp_path: Path) -> None:
    report = _passing_report()
    report["meta"] = {"model_id": "subject|model\nline"}
    report["baseline_ref"] = {"model_id": "baseline|model"}
    report_path = _write_report(tmp_path, report)

    context = build_report_export_context(report_path, report)
    markdown = render_model_card_evidence_block(context)

    assert "subject\\|model<br>line" in markdown
    assert "baseline\\|model" in markdown
