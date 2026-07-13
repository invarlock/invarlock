from __future__ import annotations

import json
from pathlib import Path

import invarlock.evidence_pack as evidence_pack_mod
from invarlock.reporting.verify_contract import VerifyExecutionResult, VerifyOutcome


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _policy_fail_result(
    report: Path,
    *,
    binding_verified: bool = True,
    expected_digest_matched: bool = False,
    outcome: VerifyOutcome = VerifyOutcome.POLICY_FAIL,
) -> VerifyExecutionResult:
    trust_status = (
        "expected_image_digest_matched"
        if expected_digest_matched
        else "manifest_bound"
        if binding_verified
        else "failed"
    )
    return VerifyExecutionResult(
        outcome=outcome,
        payload={
            "summary": {"ok": False, "reason": outcome.value},
            "results": [
                {
                    "id": str(report),
                    "ok": False,
                    "reason": outcome.value,
                    "verification": {
                        "runtime_provenance": {
                            "binding_verified": binding_verified,
                            "expected_digest_matched": expected_digest_matched,
                            "trust_status": trust_status,
                        }
                    },
                }
            ],
        },
        diagnostics=(),
    )


def _write_expected_failure_pack(tmp_path: Path) -> tuple[Path, Path, Path]:
    pack_dir = tmp_path / "pack"
    clean_report = (
        pack_dir / "reports/model/quant_4bit_clean/run_1/evaluation.report.json"
    )
    fail_report = (
        pack_dir / "reports/model/prune_50pct_stress/run_1/evaluation.report.json"
    )
    clean_report.parent.mkdir(parents=True, exist_ok=True)
    fail_report.parent.mkdir(parents=True, exist_ok=True)
    clean_report.write_text("{}", encoding="utf-8")
    fail_report.write_text("{}", encoding="utf-8")
    _write_json(
        pack_dir / "metadata/scenarios.json",
        {
            "schema": "evidence_pack_scenarios_v1",
            "schema_version": 1,
            "scenarios": [
                {"id": "quant_4bit_clean", "strictness": "must_pass"},
                {
                    "id": "prune_50pct_stress",
                    "strictness": "must_fail",
                    "primary_guard": "primary_metric",
                    "requirements": {
                        "detectors_any_of": [
                            {
                                "kind": "validation_flag",
                                "flag": "primary_metric_acceptable",
                                "expected": False,
                            }
                        ]
                    },
                },
            ],
        },
    )
    _write_json(
        fail_report,
        {"validation": {"primary_metric_acceptable": False}},
    )
    return pack_dir, clean_report, fail_report


def _write_mixed_error_probe_pack(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    pack_dir = tmp_path / "pack"
    clean_report = (
        pack_dir / "reports/model/quant_4bit_clean/run_1/evaluation.report.json"
    )
    hard_fail_report = (
        pack_dir / "reports/model/errors/rank_collapse/evaluation.report.json"
    )
    must_detect_report = (
        pack_dir
        / "reports/model/errors/spectral_moderate_scale_mlp_l31_up_s112/evaluation.report.json"
    )
    informational_report = (
        pack_dir
        / "reports/model/errors/rmt_norm_noise_l31_ffn_up_b030/evaluation.report.json"
    )
    for report in (
        clean_report,
        hard_fail_report,
        must_detect_report,
        informational_report,
    ):
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("{}", encoding="utf-8")
    _write_json(
        pack_dir / "metadata/scenarios.json",
        {
            "schema": "evidence_pack_scenarios_v1",
            "schema_version": 1,
            "scenarios": [
                {"id": "quant_4bit_clean", "strictness": "must_pass"},
                {
                    "id": "rank_collapse",
                    "strictness": "must_fail",
                    "primary_guard": "spectral",
                },
                {
                    "id": "spectral_moderate_scale_mlp_l31_up_s112",
                    "strictness": "must_detect",
                },
                {
                    "id": "rmt_norm_noise_l31_ffn_up_b030",
                    "strictness": "informational",
                },
            ],
        },
    )
    _write_json(
        hard_fail_report,
        {"validation": {"spectral_stable": False}},
    )
    return (
        pack_dir,
        clean_report,
        hard_fail_report,
        must_detect_report,
        informational_report,
    )


def test_verify_reports_accepts_scenario_expected_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pack_dir, clean_report, fail_report = _write_expected_failure_pack(tmp_path)
    seen: list[list[Path]] = []
    expected_digest = "sha256:" + ("a" * 64)

    def fake_run_verify_command(
        reports: list[Path],
        *,
        profile: str,
        report_assurance: str = "report",
        expected_runtime_image_digest: str | None = None,
    ) -> VerifyExecutionResult:
        assert expected_runtime_image_digest == expected_digest
        seen.append(reports)
        if reports == [fail_report]:
            return _policy_fail_result(fail_report, expected_digest_matched=True)
        return VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"ok": True, "reports": [str(path) for path in reports]},
            diagnostics=(),
        )

    monkeypatch.setattr(
        evidence_pack_mod,
        "_run_verify_command",
        fake_run_verify_command,
        raising=True,
    )

    errors, payload = evidence_pack_mod._verify_reports(
        pack_dir,
        json_out_path=None,
        profile="ci",
        report_assurance="strict",
        expected_runtime_image_digest=expected_digest,
    )

    assert errors == []
    assert seen == [[clean_report], [fail_report]]
    assert payload is not None
    assert payload["ok"] is True
    assert payload["expected_failures"]["reports"] == [
        "reports/model/prune_50pct_stress/run_1/evaluation.report.json"
    ]


def test_verify_reports_accepts_informational_error_probe_reports(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (
        pack_dir,
        clean_report,
        hard_fail_report,
        must_detect_report,
        informational_report,
    ) = _write_mixed_error_probe_pack(tmp_path)
    seen: list[list[Path]] = []

    def fake_run_verify_command(
        reports: list[Path], *, profile: str, report_assurance: str = "report"
    ) -> VerifyExecutionResult:
        seen.append(reports)
        if reports == [hard_fail_report]:
            return _policy_fail_result(hard_fail_report)
        return VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"ok": True, "reports": [str(path) for path in reports]},
            diagnostics=(),
        )

    monkeypatch.setattr(
        evidence_pack_mod,
        "_run_verify_command",
        fake_run_verify_command,
        raising=True,
    )

    errors, payload = evidence_pack_mod._verify_reports(
        pack_dir,
        json_out_path=None,
        profile="ci",
        report_assurance="off",
    )

    assert errors == []
    assert len(seen) == 2
    assert set(seen[0]) == {clean_report, must_detect_report, informational_report}
    assert seen[1] == [hard_fail_report]
    assert payload is not None
    assert payload["ok"] is True
    assert payload["expected_failures"]["reports"] == [
        "reports/model/errors/rank_collapse/evaluation.report.json"
    ]


def test_verify_reports_rejects_expected_failure_that_verifies_clean(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pack_dir, _clean_report, _fail_report = _write_expected_failure_pack(tmp_path)

    def fake_run_verify_command(
        reports: list[Path], *, profile: str, report_assurance: str = "report"
    ) -> VerifyExecutionResult:
        return VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"ok": True, "reports": [str(path) for path in reports]},
            diagnostics=(),
        )

    monkeypatch.setattr(
        evidence_pack_mod,
        "_run_verify_command",
        fake_run_verify_command,
        raising=True,
    )

    errors, _payload = evidence_pack_mod._verify_reports(
        pack_dir,
        json_out_path=None,
        profile="ci",
        report_assurance="strict",
    )

    assert errors == [
        "expected-failure report verified as passing: "
        "reports/model/prune_50pct_stress/run_1/evaluation.report.json"
    ]


def test_verify_reports_rejects_malformed_expected_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pack_dir, clean_report, fail_report = _write_expected_failure_pack(tmp_path)

    def fake_run_verify_command(
        reports: list[Path], *, profile: str, report_assurance: str = "report"
    ) -> VerifyExecutionResult:
        if reports == [fail_report]:
            return _policy_fail_result(
                fail_report,
                outcome=VerifyOutcome.MALFORMED,
            )
        return VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"ok": True, "reports": [str(clean_report)]},
            diagnostics=(),
        )

    monkeypatch.setattr(
        evidence_pack_mod, "_run_verify_command", fake_run_verify_command
    )

    errors, _payload = evidence_pack_mod._verify_reports(
        pack_dir,
        json_out_path=None,
        profile="ci",
        report_assurance="report",
    )

    assert errors == [
        "expected-failure report must produce POLICY_FAIL, not malformed: "
        "reports/model/prune_50pct_stress/run_1/evaluation.report.json"
    ]


def test_verify_reports_rejects_runtime_only_expected_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pack_dir, clean_report, fail_report = _write_expected_failure_pack(tmp_path)

    def fake_run_verify_command(
        reports: list[Path],
        *,
        profile: str,
        report_assurance: str = "report",
        expected_runtime_image_digest: str | None = None,
    ) -> VerifyExecutionResult:
        if reports == [fail_report]:
            return _policy_fail_result(fail_report, binding_verified=False)
        return VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"ok": True, "reports": [str(clean_report)]},
            diagnostics=(),
        )

    monkeypatch.setattr(
        evidence_pack_mod, "_run_verify_command", fake_run_verify_command
    )

    errors, _payload = evidence_pack_mod._verify_reports(
        pack_dir,
        json_out_path=None,
        profile="ci",
        report_assurance="off",
    )

    assert errors == [
        "expected-failure report lacks valid report/runtime binding: "
        "reports/model/prune_50pct_stress/run_1/evaluation.report.json"
    ]


def test_verify_reports_rejects_expected_image_digest_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pack_dir, clean_report, fail_report = _write_expected_failure_pack(tmp_path)
    expected_digest = "sha256:" + ("b" * 64)

    def fake_run_verify_command(
        reports: list[Path],
        *,
        profile: str,
        report_assurance: str = "report",
        expected_runtime_image_digest: str | None = None,
    ) -> VerifyExecutionResult:
        if reports == [fail_report]:
            return _policy_fail_result(fail_report, binding_verified=True)
        return VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"ok": True, "reports": [str(clean_report)]},
            diagnostics=(),
        )

    monkeypatch.setattr(
        evidence_pack_mod, "_run_verify_command", fake_run_verify_command
    )

    errors, _payload = evidence_pack_mod._verify_reports(
        pack_dir,
        json_out_path=None,
        profile="ci",
        report_assurance="strict",
        expected_runtime_image_digest=expected_digest,
    )

    assert errors == [
        "expected-failure report did not match the expected runtime image digest: "
        "reports/model/prune_50pct_stress/run_1/evaluation.report.json"
    ]


def test_verify_reports_rejects_policy_failure_without_scenario_signal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pack_dir, clean_report, fail_report = _write_expected_failure_pack(tmp_path)
    _write_json(
        fail_report,
        {"validation": {"primary_metric_acceptable": True}},
    )

    def fake_run_verify_command(
        reports: list[Path], *, profile: str, report_assurance: str = "report"
    ) -> VerifyExecutionResult:
        if reports == [fail_report]:
            return _policy_fail_result(fail_report)
        return VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"ok": True, "reports": [str(clean_report)]},
            diagnostics=(),
        )

    monkeypatch.setattr(
        evidence_pack_mod, "_run_verify_command", fake_run_verify_command
    )

    errors, _payload = evidence_pack_mod._verify_reports(
        pack_dir,
        json_out_path=None,
        profile="ci",
        report_assurance="report",
    )

    assert errors == [
        "expected-failure report lacks its intended report-local failure signal: "
        "reports/model/prune_50pct_stress/run_1/evaluation.report.json"
    ]
