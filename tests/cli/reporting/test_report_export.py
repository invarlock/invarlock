from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from invarlock.cli.app import app as cli
from invarlock.cli.commands import report_export as report_export_command
from invarlock.reporting import oss_exports


def _evaluation_report_payload() -> dict[str, object]:
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


def _write_evaluation_report(tmp_path: Path, payload: dict[str, object]) -> Path:
    report = tmp_path / "evaluation.report.json"
    report.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return report


def _verify_result_payload(
    report: Path,
    *,
    ok: bool = True,
    reason: str | None = None,
    runtime_status: str = "verified",
) -> dict[str, Any]:
    """Build the smallest real verify-v1 result accepted by report export.

    Export accepts an unsigned receipt only as a byte-bound, untrusted handoff;
    the helper deliberately includes the exact report digest so tests cannot
    accidentally normalize the former id-only acceptance path.
    """

    resolved_reason = reason or ("ok" if ok else "policy_fail")
    return {
        "format_version": "verify-v1",
        "summary": {"ok": ok, "reason": resolved_reason},
        "results": [
            {
                "id": str(report),
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
                            report.read_bytes()
                        ).hexdigest(),
                    },
                },
            }
        ],
    }


def test_report_export_mlflow_tags_stdout(tmp_path: Path) -> None:
    report = _write_evaluation_report(tmp_path, _evaluation_report_payload())

    result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "--evaluation-report",
            str(report),
            "--format",
            "mlflow-tags",
            "--policy-profile",
            "ci",
        ],
    )

    assert result.exit_code == 0, result.stdout
    exported = json.loads(result.stdout)
    tags = exported["tags"]
    expected_sha = hashlib.sha256(report.read_bytes()).hexdigest()
    assert exported["schema_version"] == "invarlock.mlflow-tags.v1"
    assert exported["artifact"]["path"] == str(report.resolve())
    assert tags["invarlock.status"] == "report_local_pass"
    assert tags["invarlock.report_local_status"] == "pass"
    assert tags["invarlock.report_sha256"] == expected_sha
    assert tags["invarlock.policy_profile"] == "ci"
    assert tags["invarlock.baseline"] == "baseline-model"
    assert tags["invarlock.subject"] == "subject-model"
    assert tags["invarlock.verifier_status"] == "not_provided"
    assert tags["invarlock.verifier_outcome"] == "not_provided"
    assert tags["invarlock.receipt_status"] == "not_provided"


def test_report_export_model_card_writes_markdown(tmp_path: Path) -> None:
    report = _write_evaluation_report(tmp_path, _evaluation_report_payload())
    output = tmp_path / "model-card-block.md"

    result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "-i",
            str(report),
            "--format",
            "model-card-md",
            "--output",
            str(output),
            "--report-url",
            "https://example.test/evaluation.report.json",
            "--evidence-url",
            "https://example.test/evidence.zip",
        ],
    )

    assert result.exit_code == 0, result.stdout
    text = output.read_text(encoding="utf-8")
    assert "## InvarLock Evidence" in text
    assert "| Status | REPORT_LOCAL_PASS |" in text
    assert "[https://example.test/evaluation.report.json]" in text
    assert "[evidence pack](https://example.test/evidence.zip)" in text


def test_report_export_accepts_canonical_report_directory(tmp_path: Path) -> None:
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    report = _write_evaluation_report(report_dir, _evaluation_report_payload())
    verify_result = tmp_path / "verify.json"
    verify_result.write_text(
        json.dumps(_verify_result_payload(report)), encoding="utf-8"
    )

    result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "-i",
            str(report_dir),
            "--format",
            "mlflow-tags",
            "--verify-result",
            str(verify_result),
        ],
    )

    assert result.exit_code == 0, result.stdout
    exported = json.loads(result.stdout)
    assert exported["artifact"]["path"] == str(report.resolve())
    assert exported["tags"]["invarlock.status"] == "receipt_bound_untrusted"
    assert exported["tags"]["invarlock.verifier_status"] == "receipt_bound_untrusted"
    assert exported["tags"]["invarlock.verifier_outcome"] == "pass"
    assert exported["tags"]["invarlock.receipt_status"] == "bound_unsigned"


def test_report_export_rejects_missing_evaluation_report(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "-i",
            str(tmp_path / "missing.report.json"),
            "--format",
            "mlflow-tags",
        ],
    )

    assert result.exit_code == 2
    assert "FAIL" in result.stdout


def test_report_export_rejects_run_report_without_evaluation_validation(
    tmp_path: Path,
) -> None:
    report = _write_evaluation_report(
        tmp_path,
        {
            "schema_version": "v1",
            "run_id": "run-only",
            "validation": ["not", "an", "object"],
        },
    )

    result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "-i",
            str(report),
            "--format",
            "mlflow-tags",
        ],
    )

    assert result.exit_code == 2
    assert "evaluation.report.json" in result.stdout


def test_report_export_handles_exporter_import_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = _write_evaluation_report(tmp_path, _evaluation_report_payload())
    real_import = builtins.__import__

    def fail_oss_exports_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "invarlock.reporting.oss_exports":
            raise ImportError("exporter unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_oss_exports_import)

    result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "-i",
            str(report),
            "--format",
            "mlflow-tags",
        ],
    )

    assert result.exit_code == 1
    assert "Failed to load report exporter" in result.stdout


def test_report_export_release_review_refuses_overwrite(tmp_path: Path) -> None:
    report = _write_evaluation_report(tmp_path, _evaluation_report_payload())
    output = tmp_path / "release-review.md"
    output.write_text("existing", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "-i",
            str(report),
            "--format",
            "release-review-md",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert output.read_text(encoding="utf-8") == "existing"


def test_report_export_force_overwrites_existing_file(tmp_path: Path) -> None:
    report = _write_evaluation_report(tmp_path, _evaluation_report_payload())
    output = tmp_path / "release-review.md"
    output.write_text("existing", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "-i",
            str(report),
            "--format",
            "release-review-md",
            "--output",
            str(output),
            "--force",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "# InvarLock Release Review" in output.read_text(encoding="utf-8")


def test_report_export_uses_verify_result_status(tmp_path: Path) -> None:
    report = _write_evaluation_report(tmp_path, _evaluation_report_payload())
    verify_result = tmp_path / "verify.json"
    verify_result.write_text(
        json.dumps(_verify_result_payload(report, ok=False, runtime_status="failed")),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "-i",
            str(report),
            "--format",
            "mlflow-tags",
            "--verify-result",
            str(verify_result),
        ],
    )

    assert result.exit_code == 0, result.stdout
    tags = json.loads(result.stdout)["tags"]
    assert tags["invarlock.status"] == "receipt_bound_untrusted"
    assert tags["invarlock.verifier_status"] == "receipt_bound_untrusted"
    assert tags["invarlock.verifier_outcome"] == "fail"
    assert tags["invarlock.verifier_reason"] == "policy_fail"
    assert tags["invarlock.runtime_provenance_status"] == "failed"


def test_report_export_rejects_unreadable_verify_result(tmp_path: Path) -> None:
    report = _write_evaluation_report(tmp_path, _evaluation_report_payload())
    verify_result = tmp_path / "verify.json"
    verify_result.write_text("{", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "-i",
            str(report),
            "--format",
            "mlflow-tags",
            "--verify-result",
            str(verify_result),
        ],
    )

    assert result.exit_code == 2
    assert "Failed to read verify result" in result.stdout


def test_report_export_rejects_non_object_verify_result(tmp_path: Path) -> None:
    report = _write_evaluation_report(tmp_path, _evaluation_report_payload())
    verify_result = tmp_path / "verify.json"
    verify_result.write_text("[]", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "-i",
            str(report),
            "--format",
            "mlflow-tags",
            "--verify-result",
            str(verify_result),
        ],
    )

    assert result.exit_code == 2
    assert "Verify result must decode to a JSON object" in result.stdout


def test_report_export_rejects_ambiguous_or_nonfinite_verify_json(
    tmp_path: Path,
) -> None:
    report = _write_evaluation_report(tmp_path, _evaluation_report_payload())
    verify_result = tmp_path / "verify.json"
    duplicate_key = (
        '{"format_version":"verify-v1","format_version":"verify-v1",'
        '"summary":{"ok":true,"reason":"ok"},"results":[]}'
    )
    verify_result.write_text(duplicate_key, encoding="utf-8")

    duplicate_result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "-i",
            str(report),
            "--format",
            "mlflow-tags",
            "--verify-result",
            str(verify_result),
        ],
    )

    assert duplicate_result.exit_code == 2
    assert "duplicate key" in duplicate_result.stdout

    verify_result.write_text(
        '{"format_version":"verify-v1","summary":{"ok":true,'
        '"reason":"ok"},"results":[],"ignored":NaN}',
        encoding="utf-8",
    )
    nonfinite_result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "-i",
            str(report),
            "--format",
            "mlflow-tags",
            "--verify-result",
            str(verify_result),
        ],
    )

    assert nonfinite_result.exit_code == 2
    assert "non-standard constant" in nonfinite_result.stdout


def test_report_export_rejects_receipt_stale_after_report_write(tmp_path: Path) -> None:
    report = _write_evaluation_report(tmp_path, _evaluation_report_payload())
    verify_result = tmp_path / "verify.json"
    verify_result.write_text(
        json.dumps(_verify_result_payload(report)), encoding="utf-8"
    )
    revised = _evaluation_report_payload()
    revised["run_id"] = "revised-after-verification"
    report.write_text(json.dumps(revised, sort_keys=True), encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "-i",
            str(report),
            "--format",
            "mlflow-tags",
            "--verify-result",
            str(verify_result),
        ],
    )

    assert result.exit_code == 2
    assert "does not bind" in result.stdout


def test_report_export_rejects_stale_verify_result(tmp_path: Path) -> None:
    report = _write_evaluation_report(tmp_path, _evaluation_report_payload())
    other_report = tmp_path / "other-evaluation.report.json"
    output = tmp_path / "mlflow-tags.json"
    verify_result = tmp_path / "verify.json"
    stale_payload = _verify_result_payload(report)
    stale_payload["results"][0]["id"] = str(other_report)
    verify_result.write_text(json.dumps(stale_payload), encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "-i",
            str(report),
            "--format",
            "mlflow-tags",
            "--verify-result",
            str(verify_result),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "does not contain an item for evaluation report" in result.stdout
    assert not output.exists()


def test_report_export_rejects_idless_verify_result(tmp_path: Path) -> None:
    report = _write_evaluation_report(tmp_path, _evaluation_report_payload())
    output = tmp_path / "mlflow-tags.json"
    verify_result = tmp_path / "verify.json"
    verify_result.write_text(
        json.dumps(
            {
                "format_version": "verify-v1",
                "summary": {"ok": True, "reason": "ok"},
                "results": [{"ok": True, "reason": "ok"}],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "-i",
            str(report),
            "--format",
            "mlflow-tags",
            "--verify-result",
            str(verify_result),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "Verify result item 0.id" in result.stdout
    assert not output.exists()


def test_report_export_handles_render_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = _write_evaluation_report(tmp_path, _evaluation_report_payload())

    def fail_render(*args, **kwargs):
        raise RuntimeError("render boom")

    monkeypatch.setattr(oss_exports, "render_report_export", fail_render)

    result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "-i",
            str(report),
            "--format",
            "mlflow-tags",
        ],
    )

    assert result.exit_code == 1
    assert "Failed to export report: render boom" in result.stdout


def test_report_export_handles_output_write_error(tmp_path: Path) -> None:
    report = _write_evaluation_report(tmp_path, _evaluation_report_payload())
    output_parent = tmp_path / "not-a-dir"
    output_parent.write_text("blocker", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "-i",
            str(report),
            "--format",
            "mlflow-tags",
            "--output",
            str(output_parent / "mlflow-tags.json"),
        ],
    )

    assert result.exit_code == 1
    assert "Failed to write export file" in result.stdout


def test_report_export_rejects_unknown_format(tmp_path: Path) -> None:
    report = _write_evaluation_report(tmp_path, _evaluation_report_payload())

    result = CliRunner().invoke(
        cli,
        [
            "report",
            "export",
            "-i",
            str(report),
            "--format",
            "unknown",
        ],
    )

    assert result.exit_code == 2
    assert "Unsupported export format" in result.stdout


def test_export_report_command_registers_export_subcommand(monkeypatch) -> None:
    class FakeReportApp:
        def __init__(self) -> None:
            self.registered = []

        def command(self, name: str, **kwargs):
            def decorator(func):
                self.registered.append((name, kwargs, func))
                return func

            return decorator

    fake_app = FakeReportApp()
    calls = []

    def fake_export_report_command(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        report_export_command,
        "export_report_command",
        fake_export_report_command,
    )

    report_export_command.register_report_export_command(fake_app)

    assert len(fake_app.registered) == 1
    name, kwargs, registered = fake_app.registered[0]
    assert name == "export"
    assert kwargs == {
        "help": (
            "Export evaluation evidence for MLflow tags, model cards, "
            "or release review."
        )
    }

    registered(
        evaluation_report="evaluation.report.json",
        format="mlflow-tags",
        output="-",
        policy_profile="ci",
        report_url="https://example.test/report.json",
        evidence_url="https://example.test/evidence.zip",
        verify_result="verify.json",
        force=True,
    )

    assert calls == [
        {
            "evaluation_report": "evaluation.report.json",
            "format": "mlflow-tags",
            "output": "-",
            "policy_profile": "ci",
            "report_url": "https://example.test/report.json",
            "evidence_url": "https://example.test/evidence.zip",
            "verify_result": "verify.json",
            "force": True,
        }
    ]
