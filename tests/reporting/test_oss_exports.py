from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from invarlock.reporting import oss_exports
from invarlock.reporting.oss_exports import (
    VerifyResultMismatchError,
    VerifyResultValidationError,
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


def _verify_result_payload(
    report_path: Path,
    *,
    ok: bool = True,
    reason: str | None = None,
    runtime_status: str = "verified",
) -> dict[str, Any]:
    resolved_reason = reason or ("ok" if ok else "policy_fail")
    return {
        "format_version": "verify-v1",
        "summary": {"ok": ok, "reason": resolved_reason},
        "results": [
            {
                "id": str(report_path),
                "schema_version": "v1",
                "kind": "ppl_causal",
                "ok": ok,
                "reason": resolved_reason,
                "ci": None,
                "verification": {
                    "runtime_provenance": {
                        "status": runtime_status,
                        "verified": ok,
                    },
                    "receipt": {
                        "format_version": "invarlock.verify-receipt.v1",
                        "signed": False,
                        "subject_report_sha256": hashlib.sha256(
                            report_path.read_bytes()
                        ).hexdigest(),
                    },
                },
            }
        ],
    }


def test_verify_result_overrides_report_local_status(tmp_path: Path) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)
    verify_result = _verify_result_payload(
        report_path, ok=False, runtime_status="failed"
    )

    context = build_report_export_context(
        report_path,
        report,
        policy_profile="ci",
        verify_result=verify_result,
    )
    tags = render_mlflow_tags_export(context)["tags"]

    assert context.status == "receipt_bound_untrusted"
    assert context.report_local_status == "pass"
    assert context.verifier_status == "receipt_bound_untrusted"
    assert context.verifier_outcome == "fail"
    assert context.receipt_status == "bound_unsigned"
    assert context.runtime_provenance_status == "failed"
    assert tags["invarlock.status"] == "receipt_bound_untrusted"
    assert tags["invarlock.report_local_status"] == "pass"
    assert tags["invarlock.verifier_outcome"] == "fail"
    assert tags["invarlock.verifier_reason"] == "policy_fail"
    assert tags["invarlock.runtime_provenance_status"] == "failed"


def test_verify_result_rejects_stale_explicit_id(tmp_path: Path) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)
    other_report = tmp_path / "other-evaluation.report.json"
    verify_result = _verify_result_payload(report_path)
    verify_result["results"][0]["id"] = str(other_report)

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
    verify_result = _verify_result_payload(report_path, ok=False)
    del verify_result["results"][0]["id"]

    with pytest.raises(VerifyResultMismatchError, match="item 0.id"):
        build_report_export_context(
            report_path,
            report,
            policy_profile="ci",
            verify_result=verify_result,
        )


def test_verify_result_rejects_non_object_result_item(tmp_path: Path) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)
    verify_result = _verify_result_payload(report_path, ok=False)
    verify_result["results"] = ["not-an-object"]

    with pytest.raises(VerifyResultMismatchError, match="JSON object with an id"):
        build_report_export_context(
            report_path,
            report,
            policy_profile="ci",
            verify_result=verify_result,
        )


def test_verify_result_rejects_missing_results(tmp_path: Path) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)
    verify_result = _verify_result_payload(report_path)
    del verify_result["results"]

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
    other_report.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    verify_result = _verify_result_payload(report_path)
    other_item = _verify_result_payload(other_report, ok=False)["results"][0]
    verify_result["summary"] = {"ok": False, "reason": "policy_fail"}
    verify_result["results"] = [other_item, verify_result["results"][0]]

    context = build_report_export_context(
        report_path,
        report,
        policy_profile="ci",
        verify_result=verify_result,
    )

    assert context.status == "receipt_bound_untrusted"
    assert context.verifier_status == "receipt_bound_untrusted"
    assert context.verifier_outcome == "pass"
    assert context.verifier_reason == "ok"


def test_verify_result_rejects_receipt_for_different_report_bytes(
    tmp_path: Path,
) -> None:
    report_path = _write_report(tmp_path, _passing_report())
    verify_result = _verify_result_payload(report_path)
    verify_result["results"][0]["verification"]["receipt"]["subject_report_sha256"] = (
        "0" * 64
    )

    with pytest.raises(VerifyResultValidationError, match="does not bind"):
        build_report_export_context(
            report_path, _passing_report(), verify_result=verify_result
        )


def test_verify_result_rejects_duplicate_matching_report_items(tmp_path: Path) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)
    verify_result = _verify_result_payload(report_path)
    duplicate = _verify_result_payload(report_path)["results"][0]
    verify_result["results"].append(duplicate)

    with pytest.raises(VerifyResultValidationError, match="exactly one item"):
        build_report_export_context(report_path, report, verify_result=verify_result)


def test_verify_result_rejects_string_boolean_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)
    string_boolean = _verify_result_payload(report_path)
    string_boolean["results"][0]["ok"] = "true"

    with pytest.raises(VerifyResultValidationError, match="boolean"):
        build_report_export_context(report_path, report, verify_result=string_boolean)

    nonfinite = _verify_result_payload(report_path)
    nonfinite["results"][0]["ci"] = [float("nan"), 1.0]
    with pytest.raises(VerifyResultValidationError, match="non-finite"):
        build_report_export_context(report_path, report, verify_result=nonfinite)


def test_verify_result_rejects_unsupported_signed_claim(tmp_path: Path) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)
    verify_result = _verify_result_payload(report_path)
    verify_result["results"][0]["verification"]["receipt"]["signed"] = True

    with pytest.raises(VerifyResultValidationError, match="signed=true"):
        build_report_export_context(report_path, report, verify_result=verify_result)


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


def _assert_verify_result_rejected(
    report_path: Path,
    report: dict[str, object],
    payload: object,
    message: str,
) -> None:
    with pytest.raises(VerifyResultValidationError, match=message):
        build_report_export_context(report_path, report, verify_result=payload)


def test_verify_result_rejects_malformed_envelope_fields(tmp_path: Path) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)

    _assert_verify_result_rejected(report_path, report, [], "JSON object")

    payload = _verify_result_payload(report_path)
    payload["format_version"] = " "
    _assert_verify_result_rejected(report_path, report, payload, "must not be empty")

    payload = _verify_result_payload(report_path)
    payload["format_version"] = "verify-v0"
    _assert_verify_result_rejected(report_path, report, payload, "must be 'verify-v1'")

    payload = _verify_result_payload(report_path)
    payload["summary"] = []
    _assert_verify_result_rejected(report_path, report, payload, "summary must be")

    payload = _verify_result_payload(report_path)
    payload["summary"]["reason"] = "invented"
    _assert_verify_result_rejected(report_path, report, payload, "must be one of")


def test_verify_result_rejects_malformed_result_item_fields(tmp_path: Path) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)

    payload = _verify_result_payload(report_path)
    payload["results"][0]["schema_version"] = "v0"
    _assert_verify_result_rejected(report_path, report, payload, "must be 'v1'")

    payload = _verify_result_payload(report_path)
    payload["results"][0]["reason"] = "policy_fail"
    _assert_verify_result_rejected(report_path, report, payload, "inconsistent")

    payload = _verify_result_payload(report_path)
    del payload["results"][0]["ci"]
    _assert_verify_result_rejected(report_path, report, payload, "ci is required")

    for ci in ([1.0], [True, 1.0], [float("inf"), 1.0]):
        payload = _verify_result_payload(report_path)
        payload["results"][0]["ci"] = ci
        _assert_verify_result_rejected(report_path, report, payload, "ci")


def test_verify_result_rejects_non_json_nested_values(tmp_path: Path) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)

    payload = _verify_result_payload(report_path)
    payload["extra"] = {1: "bad-key"}
    _assert_verify_result_rejected(
        report_path, report, payload, "non-string object key"
    )

    payload = _verify_result_payload(report_path)
    payload["extra"] = object()
    _assert_verify_result_rejected(report_path, report, payload, "non-JSON value")

    payload = _verify_result_payload(report_path)
    payload["extra"] = [0.25, {"finite": 1.5}]
    context = build_report_export_context(report_path, report, verify_result=payload)
    assert context.verifier_outcome == "pass"


def test_verify_result_rejects_malformed_receipt_fields(tmp_path: Path) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)

    payload = _verify_result_payload(report_path)
    payload["results"][0]["verification"] = []
    _assert_verify_result_rejected(report_path, report, payload, "include verification")

    payload = _verify_result_payload(report_path)
    payload["results"][0]["verification"]["receipt"] = []
    _assert_verify_result_rejected(
        report_path, report, payload, "include a verification.receipt"
    )

    payload = _verify_result_payload(report_path)
    payload["results"][0]["verification"]["receipt"]["format_version"] = "old"
    _assert_verify_result_rejected(
        report_path, report, payload, "Verify receipt.format_version"
    )

    payload = _verify_result_payload(report_path)
    payload["results"][0]["verification"]["receipt"]["subject_report_sha256"] = "ABC"
    _assert_verify_result_rejected(report_path, report, payload, "lowercase SHA-256")

    payload = _verify_result_payload(report_path)
    payload["results"][0]["verification"]["runtime_provenance"] = []
    _assert_verify_result_rejected(report_path, report, payload, "runtime_provenance")


def test_export_dispatch_and_serialization_cover_every_public_format(
    tmp_path: Path,
) -> None:
    report = _passing_report()
    report_path = _write_report(tmp_path, report)
    context = build_report_export_context(
        report_path,
        report,
        report_url="https://example.test/report",
        evidence_url="https://example.test/evidence",
    )

    mlflow = oss_exports.render_report_export("mlflow-tags", context, report)
    model_card = oss_exports.render_report_export("model-card-md", context, report)
    review = oss_exports.render_report_export("release-review-md", context, report)

    assert (
        isinstance(mlflow, dict)
        and mlflow["schema_version"] == "invarlock.mlflow-tags.v1"
    )
    assert isinstance(model_card, str) and "[evidence pack]" in model_card
    assert isinstance(review, str) and "# InvarLock Release Review" in review
    assert oss_exports.serialize_report_export("line") == "line\n"
    assert oss_exports.serialize_report_export("line\n") == "line\n"
    assert '"schema_version": "invarlock.mlflow-tags.v1"' in (
        oss_exports.serialize_report_export(mlflow)
    )
    with pytest.raises(ValueError, match="Unsupported export format"):
        oss_exports.render_report_export("unknown", context, report)
