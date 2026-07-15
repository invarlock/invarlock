from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import invarlock.evidence_pack_binding as binding
import invarlock.evidence_pack_report_verification as report_verification
from invarlock.reporting.verify_contract import VerifyExecutionResult, VerifyOutcome
from tests.reporting.evidence_pack._support_evidence_pack_branch_contracts import (
    _canonical_report,
    _verify_result,
    _write_json,
)


def test_verdict_binding_normalizers_and_identity_failures(tmp_path: Path) -> None:
    assert binding._normalize_binding_digest(None) is None
    assert binding._normalize_binding_digest("SHA256:" + "A" * 64) == "a" * 64
    assert binding._normalize_binding_digest("bad") is None
    for value in (
        None,
        "",
        ".",
        "bad\\path",
        "a//b",
        "..",
        "/reports/a/evaluation.report.json",
        "other/a/evaluation.report.json",
        "reports/a/not.json",
    ):
        assert binding._normalize_verdict_report_path(value) is None
    assert (
        binding._normalize_verdict_report_path("./reports/a/evaluation.report.json")
        == "reports/a/evaluation.report.json"
    )
    assert (
        binding._normalize_verdict_report_path("m/reports/a/evaluation.report.json")
        == "reports/m/a/evaluation.report.json"
    )
    assert binding._report_run_id({"meta": {"run_id": " nested "}}) == "nested"
    assert binding._report_run_id({"meta": {"run_id": ""}}) is None
    assert binding._report_id({"report_id": " top "}) == "top"
    assert binding._report_id({"meta": {"report_id": " id "}}) == "id"
    assert binding._report_id({"run_id": "fallback"}) == "fallback"

    report_path = "reports/m/s/evaluation.report.json"
    reports = {report_path: _canonical_report(report_path, run_id=None)}
    cases = [
        (
            {"path": report_path, "report_path": "reports/m/x/evaluation.report.json"},
            "disagree",
        ),
        ({}, "requires a canonical report path"),
        ({"path": "bad"}, "invalid report path"),
        ({"path": report_path}, "requires report_sha256"),
        ({"path": report_path, "report_sha256": "bad"}, "must be a SHA-256"),
        ({"path": report_path, "report_sha256": "b" * 64}, "does not match"),
        (
            {"path": report_path, "report_sha256": "a" * 64, "run_id": ""},
            "must be a non-empty string",
        ),
        (
            {"path": report_path, "report_sha256": "a" * 64, "run_id": "x"},
            "cannot be authenticated",
        ),
    ]
    for item, fragment in cases:
        _, errors = binding._validate_binding_item(
            item,
            label="Binding",
            reports_by_path=reports,
            require_path=True,
            require_digest=True,
        )
        assert any(fragment in error for error in errors)
    _, errors = binding._validate_binding_item(
        {},
        label="Binding",
        reports_by_path=reports,
        require_path=False,
        require_digest=False,
    )
    assert errors == ["Binding cannot be associated with a canonical report."]
    path, errors = binding._validate_binding_item(
        {"path": report_path},
        label="Binding",
        reports_by_path=reports,
        require_path=True,
        require_digest=False,
    )
    assert path == report_path and errors == []


def test_verdict_path_normalizer_rejects_path_object_marked_absolute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AbsolutePath:
        def __init__(self, *parts: str) -> None:
            self.parts = ("reports", "m", "evaluation.report.json")

        def is_absolute(self) -> bool:
            return True

    monkeypatch.setattr(binding, "PurePosixPath", AbsolutePath)
    assert (
        binding._normalize_verdict_report_path("reports/m/evaluation.report.json")
        is None
    )


def test_verdict_payload_binding_rejects_malformed_coverage() -> None:
    one_path = "reports/m/a/evaluation.report.json"
    other_path = "reports/m/b/evaluation.report.json"
    one = _canonical_report(one_path)
    other = _canonical_report(other_path, digest="b" * 64)
    assert binding._verify_final_verdict_payload_report_binding(
        [], [one], require_binding=True
    ) == ["Final verdict must be a JSON object."]
    assert binding._verify_final_verdict_payload_report_binding(
        {}, [], require_binding=True
    ) == ["Final verdict exists but the pack contains no canonical reports."]
    duplicate = binding._verify_final_verdict_payload_report_binding(
        {}, [one, one], require_binding=True
    )
    assert duplicate == ["Canonical report paths are not unique."]
    errors = binding._verify_final_verdict_payload_report_binding(
        {"report_bindings": "bad", "records": "bad"}, [one], require_binding=True
    )
    assert "Final verdict report_bindings must be a list." in errors
    assert "Final verdict records must be a list." in errors
    errors = binding._verify_final_verdict_payload_report_binding(
        {
            "report_bindings": [
                None,
                {"path": one_path, "report_sha256": "a" * 64},
                {"path": one_path, "report_sha256": "a" * 64},
                {
                    "path": "reports/m/missing/evaluation.report.json",
                    "report_sha256": "a" * 64,
                },
            ],
            "records": [None, {"path": one_path}, {"path": one_path}],
        },
        [one, other],
        require_binding=True,
    )
    for fragment in (
        "must be a JSON object",
        "duplicate path",
        "does not cover canonical reports",
        "non-canonical reports",
        "requires report_sha256",
        "duplicate report path",
    ):
        assert any(fragment in error for error in errors)
    errors = binding._verify_final_verdict_payload_report_binding(
        {"report_sha256": "a" * 64}, [one, other], require_binding=True
    )
    assert any("ambiguous" in error for error in errors)
    assert any("requires exact report_bindings" in error for error in errors)
    errors = binding._verify_final_verdict_payload_report_binding(
        {"report_bindings": []}, [one], require_binding=True
    )
    assert "Final verdict report_bindings must not be empty." in errors
    errors = binding._verify_final_verdict_payload_report_binding(
        {"report_bindings": [{"path": "bad"}], "records": [{"path": "bad"}]},
        [one],
        require_binding=True,
    )
    assert sum("invalid report path" in error for error in errors) == 2


def test_binding_file_discovery_rejects_unsafe_trees(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.write_text("{}", encoding="utf-8")
    assert (
        "outside the pack"
        in binding._binding_file_safety_errors(
            tmp_path / "pack", outside, label="File"
        )[0]
    )
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "reports").symlink_to(tmp_path)
    matches, errors = binding._discover_binding_files(
        pack, subtree="reports", filename="evaluation.report.json", label="Report"
    )
    assert matches == [] and "must not be a symlink" in errors[0]
    (pack / "reports").unlink()
    (pack / "reports").write_text("x", encoding="utf-8")
    assert (
        "must be a directory"
        in binding._discover_binding_files(
            pack, subtree="reports", filename="x", label="Report"
        )[1][0]
    )
    (pack / "reports").unlink()
    nested = pack / "reports/m"
    nested.mkdir(parents=True)
    (nested / "link").symlink_to(tmp_path)
    _, errors = binding._discover_binding_files(
        pack, subtree="reports", filename="evaluation.report.json", label="Report"
    )
    assert any("must not contain symlinks" in error for error in errors)
    assert (
        "must resolve to a regular file"
        in binding._binding_file_safety_errors(
            pack, pack / "reports/missing.json", label="File"
        )[0]
    )
    assert (
        "is not a regular file"
        in binding._binding_file_safety_errors(pack, nested, label="File")[0]
    )


def test_binding_frontdoor_failures_and_success(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    assert binding.verify_final_verdict_report_binding(pack) == []
    report = pack / "reports/m/s/evaluation.report.json"
    _write_json(report, {"run_id": "r"})
    assert binding.verify_final_verdict_report_binding(pack) == [
        "Canonical reports exist but final_verdict.json is missing."
    ]
    _write_json(pack / "results/a/final_verdict.json", {})
    _write_json(pack / "results/b/final_verdict.json", {})
    assert (
        "multiple final verdicts"
        in binding.verify_final_verdict_report_binding(pack)[0]
    )
    (pack / "results/b/final_verdict.json").unlink()
    (pack / "results/a/final_verdict.json").write_text("{", encoding="utf-8")
    assert "not valid JSON" in binding.verify_final_verdict_report_binding(pack)[0]
    _write_json(pack / "results/a/final_verdict.json", {"report_sha256": "bad"})
    assert any(
        "SHA-256" in error
        for error in binding.verify_final_verdict_report_binding(pack)
    )

    _write_json(pack / "manifest.json", [])
    assert not binding._pack_declares_strict_report_binding(pack)
    _write_json(pack / "manifest.json", {"verification": []})
    assert not binding._pack_declares_strict_report_binding(pack)

    _write_json(report, [])
    errors = binding.verify_final_verdict_report_binding(pack)
    assert any("must be a JSON object" in error for error in errors)
    report.write_text("{", encoding="utf-8")
    errors = binding.verify_final_verdict_report_binding(pack)
    assert any("not valid JSON" in error for error in errors)


@pytest.mark.parametrize(
    "detector,payload,expected",
    [
        ({"kind": "validation_flag", "flag": 3}, {"validation": {"x": True}}, False),
        (
            {"kind": "validation_flag", "flag": "x", "expected": False},
            {"validation": {"x": False}},
            True,
        ),
        (
            {"kind": "validation_flag", "flag": "x", "expected": "bad"},
            {"validation": {"x": "bad"}},
            True,
        ),
        ({"kind": "primary_metric", "field": 3}, {"primary_metric": {"x": 1}}, False),
        (
            {"kind": "primary_metric", "field": "x", "expected": True},
            {"primary_metric": {"x": True}},
            True,
        ),
        (
            {"kind": "primary_metric", "field": "x", "expected": 3},
            {"primary_metric": {"x": 3}},
            True,
        ),
        ({"kind": "spectral_caps_applied"}, {"spectral": []}, False),
        (
            {"kind": "spectral_caps_applied", "min": "bad"},
            {"spectral": {"caps_applied": 1}},
            False,
        ),
        (
            {"kind": "spectral_caps_applied", "min": 2},
            {"spectral": {"caps_applied": 3}},
            True,
        ),
        (
            {"kind": "invariants_status", "allowed": ["FAIL", 1]},
            {"invariants": {"status": "fail"}},
            True,
        ),
        ({"kind": "invariants_status", "allowed": "fail"}, {"invariants": {}}, False),
        ({"kind": "unknown"}, {}, False),
    ],
)
def test_failure_detector_contracts(
    detector: dict[str, Any], payload: dict[str, Any], expected: bool
) -> None:
    assert report_verification._detector_matches_report(payload, detector) is expected


def test_primary_guard_and_expected_failure_checks(tmp_path: Path) -> None:
    assert report_verification._primary_guard_failure_signal(
        {"validation": {"rmt_stable": False}}, "rmt"
    )
    assert report_verification._primary_guard_failure_signal(
        {"primary_metric": {"degraded": True}}, "primary_metric"
    )
    assert report_verification._primary_guard_failure_signal(
        {"spectral": {"status": "ok", "caps_applied": 1}}, "spectral"
    )
    assert not report_verification._primary_guard_failure_signal(
        {"spectral": {"status": "ok", "caps_applied": "bad"}}, "spectral"
    )
    assert not report_verification._primary_guard_failure_signal({}, "rmt")
    assert not report_verification._primary_guard_failure_signal(
        {"rmt": {"status": "ok"}}, "rmt"
    )
    assert report_verification._primary_guard_failure_signal(
        {"rmt": {"status": "error"}}, "rmt"
    )
    assert not report_verification._primary_guard_failure_signal(
        {"spectral": {"status": "ok", "caps_applied": 0}}, "spectral"
    )
    pack = tmp_path / "pack"
    report = pack / "reports/m/errors/fault/evaluation.report.json"
    _write_json(report, {"validation": {"rmt_stable": False}})
    assert not report_verification._report_has_intended_failure_signal(pack, report)
    report.write_text("{", encoding="utf-8")
    assert not report_verification._report_has_intended_failure_signal(pack, report)
    _write_json(report, [])
    assert not report_verification._report_has_intended_failure_signal(pack, report)
    _write_json(
        pack / "metadata/scenarios.json",
        {"scenarios": [{"id": "fault", "strictness": "must_pass"}]},
    )
    _write_json(report, {"validation": {"rmt_stable": False}})
    assert not report_verification._report_has_intended_failure_signal(pack, report)

    _write_json(
        pack / "metadata/scenarios.json",
        {
            "scenarios": [
                {
                    "id": "fault",
                    "strictness": "must_fail",
                    "primary_guard": "rmt",
                }
            ]
        },
    )
    assert report_verification._report_has_intended_failure_signal(pack, report)


def test_expected_failure_result_rejects_wrong_outcome_and_provenance(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    report = pack / "reports/m/errors/fault/evaluation.report.json"
    _write_json(report, {"validation": {"rmt_stable": False}})
    cases = [
        (_verify_result(VerifyOutcome.OK, {}), None, "must produce POLICY_FAIL"),
        (_verify_result(VerifyOutcome.POLICY_FAIL, []), None, "payload is malformed"),
        (
            _verify_result(VerifyOutcome.POLICY_FAIL, {"results": []}),
            None,
            "lacks valid report/runtime binding",
        ),
        (
            _verify_result(
                VerifyOutcome.POLICY_FAIL,
                {
                    "results": [
                        {
                            "verification": {
                                "runtime_provenance": {"binding_verified": True}
                            }
                        }
                    ]
                },
            ),
            "sha256:" + "a" * 64,
            "did not match the expected runtime image digest",
        ),
    ]
    for result, digest, fragment in cases:
        errors = report_verification._expected_failure_result_errors(
            pack, report, result, expected_runtime_image_digest=digest
        )
        assert fragment in errors[0]


def test_verify_reports_grouping_and_expected_failure_error_paths(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    assert report_verification.verify_reports(
        pack,
        json_out_path=None,
        profile="ci",
        report_assurance="off",
        run_verify_command=lambda *a, **k: _verify_result(VerifyOutcome.OK, {}),
    ) == (["No reports found in pack."], None)

    unclassified = pack / "reports/m/undeclared/evaluation.report.json"
    _write_json(unclassified, {})
    errors, payload = report_verification.verify_reports(
        pack,
        json_out_path=None,
        profile="ci",
        report_assurance="strict",
        run_verify_command=lambda *a, **k: _verify_result(VerifyOutcome.OK, {}),
    )
    assert payload is None
    assert errors == [
        "Every report must reference a scenario declared by the current "
        "scenario manifest; unclassified reports: "
        "reports/m/undeclared/evaluation.report.json"
    ]
    unclassified.unlink()

    pass_a = pack / "reports/m/a/evaluation.report.json"
    pass_b = pack / "reports/m/b/evaluation.report.json"
    fail = pack / "reports/m/errors/fault/evaluation.report.json"
    for path in (pass_a, pass_b):
        _write_json(path, {})
    _write_json(fail, {"validation": {"rmt_stable": False}})
    _write_json(
        pack / "metadata/scenarios.json",
        {
            "scenarios": [
                {"id": "a", "strictness": "must_pass"},
                {"id": "b", "strictness": "must_pass"},
                {
                    "id": "fault",
                    "strictness": "must_fail",
                    "primary_guard": "rmt",
                },
            ]
        },
    )
    baseline_a = tmp_path / "baseline-a.json"
    baseline_b = tmp_path / "baseline-b.json"
    calls: list[tuple[list[Path], dict[str, Any]]] = []

    good_runtime = {
        "results": [
            {
                "verification": {
                    "runtime_provenance": {
                        "binding_verified": True,
                        "expected_digest_matched": True,
                        "trust_status": "expected_image_digest_matched",
                    }
                }
            }
        ]
    }

    def run(reports: list[Path], **kwargs: Any) -> VerifyExecutionResult:
        calls.append((reports, kwargs))
        if reports == [fail]:
            return _verify_result(VerifyOutcome.POLICY_FAIL, good_runtime)
        return _verify_result(
            VerifyOutcome.OK,
            {
                "results": [{"path": str(path)} for path in reports],
                "summary": {"ok": True},
            },
        )

    out = tmp_path / "verify.json"
    errors, payload = report_verification.verify_reports(
        pack,
        json_out_path=out,
        profile="ci",
        report_assurance="strict",
        run_verify_command=run,
        expected_runtime_image_digest="sha256:" + "a" * 64,
        baseline_by_report={
            pass_a.resolve(): baseline_a,
            pass_b.resolve(): baseline_b,
            fail.resolve(): baseline_a,
        },
        policy_pack=tmp_path / "policy.json",
    )
    assert errors == []
    assert payload is not None and len(payload["results"]) == 2
    assert payload["expected_failures"]["reports"] == [
        "reports/m/errors/fault/evaluation.report.json"
    ]
    assert out.is_file()
    assert {kwargs["baseline"] for reports, kwargs in calls if reports != [fail]} == {
        baseline_a,
        baseline_b,
    }
    assert (
        next(kwargs for reports, kwargs in calls if reports == [fail])["baseline"]
        == baseline_a
    )

    def grouped_nonlist(reports: list[Path], **kwargs: Any) -> VerifyExecutionResult:
        return _verify_result(VerifyOutcome.OK, {"results": "not-a-list"})

    errors, payload = report_verification.verify_reports(
        pack,
        json_out_path=None,
        profile="ci",
        report_assurance="off",
        run_verify_command=grouped_nonlist,
        baseline_by_report={pass_a.resolve(): baseline_a, pass_b.resolve(): baseline_b},
    )
    assert errors and payload is not None

    def malformed_pass(reports: list[Path], **kwargs: Any) -> VerifyExecutionResult:
        return _verify_result(VerifyOutcome.OK, [])

    assert (
        "did not return a JSON object"
        in report_verification.verify_reports(
            pack,
            json_out_path=None,
            profile="ci",
            report_assurance="off",
            run_verify_command=malformed_pass,
        )[0][0]
    )

    def raising(reports: list[Path], **kwargs: Any) -> VerifyExecutionResult:
        if reports == [fail]:
            raise RuntimeError("boom")
        return _verify_result(VerifyOutcome.OK, {})

    assert (
        "failed unexpectedly"
        in report_verification.verify_reports(
            pack,
            json_out_path=None,
            profile="ci",
            report_assurance="off",
            run_verify_command=raising,
        )[0][0]
    )
