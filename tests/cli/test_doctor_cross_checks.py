from __future__ import annotations

import json

import pytest
import typer

from invarlock.core.doctor_findings import build_cross_check_findings
from tests.cli._support_doctor import (
    _install_fake_torch,
    _mk_report,
    _patch_minimal_doctor_env,
    doctor_mod,
)


def test_doctor_cross_checks_tokenizer_mismatch(monkeypatch, tmp_path, capsys):
    _install_fake_torch(monkeypatch, cuda_available=False)
    _patch_minimal_doctor_env(monkeypatch)
    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    baseline.write_text(
        json.dumps(_mk_report(tokenizer="tokA", split="validation")), encoding="utf-8"
    )
    subject.write_text(
        json.dumps(_mk_report(tokenizer="tokB", split="validation")), encoding="utf-8"
    )
    with pytest.raises(typer.Exit) as exc:
        doctor_mod.doctor_command(
            json_out=True,
            baseline_report=str(baseline),
            subject_report=str(subject),
        )
    assert exc.value.exit_code == 0
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    codes = {f["code"] for f in payload.get("findings", [])}
    assert "D009" in codes


def test_doctor_cross_checks_mask_missing(monkeypatch, tmp_path, capsys):
    _install_fake_torch(monkeypatch, cuda_available=False)
    _patch_minimal_doctor_env(monkeypatch)
    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    baseline.write_text(
        json.dumps(_mk_report(tokenizer="tokA", split="validation", pm_kind="ppl_mlm")),
        encoding="utf-8",
    )
    subject.write_text(
        json.dumps(_mk_report(tokenizer="tokA", split="validation", pm_kind="ppl_mlm")),
        encoding="utf-8",
    )
    with pytest.raises(typer.Exit) as exc:
        doctor_mod.doctor_command(
            json_out=True,
            baseline_report=str(baseline),
            subject_report=str(subject),
        )
    assert exc.value.exit_code == 0
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    codes = {f["code"] for f in payload.get("findings", [])}
    assert "D010" in codes


def test_doctor_cross_checks_split_mismatch_strict(monkeypatch, tmp_path, capsys):
    _install_fake_torch(monkeypatch, cuda_available=False)
    _patch_minimal_doctor_env(monkeypatch)
    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    baseline.write_text(json.dumps(_mk_report(split="validation")), encoding="utf-8")
    subject.write_text(json.dumps(_mk_report(split="test")), encoding="utf-8")
    with pytest.raises(typer.Exit) as exc:
        doctor_mod.doctor_command(
            json_out=True,
            baseline_report=str(baseline),
            subject_report=str(subject),
            strict=True,
        )
    assert exc.value.exit_code == 1
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    entries = [f for f in payload.get("findings", []) if f.get("code") == "D011"]
    assert entries and entries[0]["severity"] == "error"


def test_doctor_cross_checks_accuracy_pseudo_counts(monkeypatch, tmp_path, capsys):
    _install_fake_torch(monkeypatch, cuda_available=False)
    _patch_minimal_doctor_env(monkeypatch)
    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    baseline.write_text(json.dumps(_mk_report(split="validation")), encoding="utf-8")
    subject.write_text(
        json.dumps(
            _mk_report(
                split="validation",
                pm_kind="accuracy",
                counts_source="pseudo_config",
                estimated=True,
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(typer.Exit) as exc:
        doctor_mod.doctor_command(
            json_out=True,
            baseline_report=str(baseline),
            subject_report=str(subject),
            profile="ci",
        )
    assert exc.value.exit_code == 1
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    entries = [f for f in payload.get("findings", []) if f.get("code") == "D012"]
    assert entries and entries[0]["severity"] == "error"


def _run_cross_check(tmp_path, baseline_payload, subject_payload, **kwargs):
    baseline = tmp_path / "baseline_cc.json"
    subject = tmp_path / "subject_cc.json"
    baseline.write_text(json.dumps(baseline_payload), encoding="utf-8")
    subject.write_text(json.dumps(subject_payload), encoding="utf-8")
    findings, had_error = build_cross_check_findings(
        str(baseline),
        str(subject),
        cfg_metric_kind=kwargs.get("cfg_metric_kind"),
        strict=kwargs.get("strict", False),
        profile=kwargs.get("profile"),
    )
    return had_error, [(finding.code, finding.severity) for finding in findings]


def test_cross_checks_d009_tokenizer(tmp_path):
    had_error, calls = _run_cross_check(
        tmp_path,
        _mk_report(tokenizer="tokA", split="validation"),
        _mk_report(tokenizer="tokB", split="validation"),
    )
    assert not had_error
    assert ("D009", "warning") in calls


def test_cross_checks_d010_missing_mask(tmp_path):
    had_error, calls = _run_cross_check(
        tmp_path,
        _mk_report(tokenizer="tokA", split="validation", pm_kind="ppl_mlm"),
        _mk_report(tokenizer="tokA", split="validation", pm_kind="ppl_mlm"),
        cfg_metric_kind="ppl_mlm",
    )
    assert not had_error
    assert ("D010", "warning") in calls


def test_cross_checks_d011_strict(tmp_path):
    had_error, calls = _run_cross_check(
        tmp_path,
        _mk_report(tokenizer="tokA", split="validation"),
        _mk_report(tokenizer="tokA", split="test"),
        strict=True,
    )
    assert had_error
    assert ("D011", "error") in calls


def test_cross_checks_d012_profile(tmp_path):
    had_error_dev, calls_dev = _run_cross_check(
        tmp_path,
        _mk_report(split="validation"),
        _mk_report(
            split="validation",
            pm_kind="accuracy",
            counts_source="pseudo_config",
            estimated=True,
        ),
        profile=None,
    )
    assert not had_error_dev
    assert ("D012", "warning") in calls_dev

    had_error_ci, calls_ci = _run_cross_check(
        tmp_path,
        _mk_report(split="validation"),
        _mk_report(
            split="validation",
            pm_kind="accuracy",
            counts_source="pseudo_config",
            estimated=True,
        ),
        profile="ci",
    )
    assert had_error_ci
    assert ("D012", "error") in calls_ci
