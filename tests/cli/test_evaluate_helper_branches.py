from __future__ import annotations

import json

import pytest
import typer

from invarlock.cli.commands import evaluate as mod


class _RecordingConsole:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, *args: object, **_kwargs: object) -> None:
        self.lines.append(" ".join(str(arg) for arg in args))


def test_evaluate_helper_formatting_and_console_contexts(monkeypatch) -> None:
    console = _RecordingConsole()
    monkeypatch.setattr(mod, "console", console)

    assert mod._format_ratio("bad") == "N/A"
    assert mod._format_ratio(float("inf")) == "N/A"
    assert mod._phase_title(2, 3, "Validate") == "PHASE 2/3 · Validate"

    with pytest.raises(typer.Exit) as exc:
        mod._resolve_verbosity(True, True)
    assert exc.value.exit_code == 2
    assert any("mutually exclusive" in line for line in console.lines)
    assert mod._resolve_verbosity(False, True) == mod.VERBOSITY_VERBOSE

    module = type("DummyModule", (), {"console": "old"})()
    with mod._override_console(module, "new"):
        assert module.console == "new"
    assert module.console == "old"

    with mod._suppress_child_output(False) as buffer:
        assert buffer is None

    from invarlock.cli.commands import report as report_mod
    from invarlock.cli.commands import run as run_mod

    original_report_console = report_mod.console
    original_run_console = run_mod.console
    with mod._suppress_child_output(True) as buffer:
        assert buffer is not None
        assert report_mod.console is not original_report_console
        assert run_mod.console is not original_run_console
    assert report_mod.console is original_report_console
    assert run_mod.console is original_run_console


def test_evaluate_quiet_summary_variants(tmp_path, monkeypatch) -> None:
    console = _RecordingConsole()
    monkeypatch.setattr(mod, "console", console)
    report_out = tmp_path / "reports"
    report_out.mkdir()

    mod._print_quiet_summary(
        report_out=report_out,
        source="baseline",
        edited="subject",
        profile="ci",
    )
    assert any(f"Output: {report_out}" in line for line in console.lines)

    report_path = report_out / "evaluation.report.json"
    report_path.write_text("{not-json", encoding="utf-8")
    console.lines.clear()
    mod._print_quiet_summary(
        report_out=report_out,
        source="baseline",
        edited="subject",
        profile="ci",
    )
    assert any(f"Output: {report_path}" in line for line in console.lines)

    report_path.write_text(json.dumps(["not-a-dict"]), encoding="utf-8")
    console.lines.clear()
    mod._print_quiet_summary(
        report_out=report_out,
        source="baseline",
        edited="subject",
        profile="ci",
    )
    assert any(f"Output: {report_path}" in line for line in console.lines)

    report_path.write_text(
        json.dumps({"primary_metric": {"ratio_vs_baseline": 1.2345}}),
        encoding="utf-8",
    )
    console.lines.clear()
    monkeypatch.setattr(
        "invarlock.reporting.render.compute_console_validation_block",
        lambda _report: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=False,
    )
    mod._print_quiet_summary(
        report_out=report_out,
        source="baseline",
        edited="subject",
        profile="ci",
    )
    joined = "\n".join(console.lines)
    assert "Status: UNKNOWN · Gates: N/A" in joined
    assert "Primary metric ratio: 1.234" in joined


def test_evaluate_yaml_tmp_dir_and_successful_quiet_summary(
    tmp_path, monkeypatch
) -> None:
    payload = {"dataset": {"provider": "demo"}}
    yaml_path = tmp_path / "preset.yaml"
    mod._dump_yaml(yaml_path, payload)
    assert mod._load_yaml(yaml_path) == payload

    tmp_dir = tmp_path / "custom-evaluate-tmp"
    monkeypatch.setenv("INVARLOCK_EVALUATE_TMP_DIR", str(tmp_dir))
    assert mod._resolve_evaluate_tmp_dir() == tmp_dir
    assert tmp_dir.exists()

    assert mod._normalize_model_id("hf:demo/model", "hf_causal") == "demo/model"

    console = _RecordingConsole()
    monkeypatch.setattr(mod, "console", console)
    report_out = tmp_path / "reports"
    report_out.mkdir()
    report_path = report_out / "evaluation.report.json"
    report_path.write_text(
        json.dumps({"primary_metric": {"ratio_vs_baseline": 0.99}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "invarlock.reporting.render.compute_console_validation_block",
        lambda _report: {
            "rows": [{"ok": True}, {"ok": False}],
            "overall_pass": True,
        },
        raising=False,
    )
    mod._print_quiet_summary(
        report_out=report_out,
        source="baseline",
        edited="subject",
        profile="release",
    )
    joined = "\n".join(console.lines)
    assert "Status: PASS · Gates: 1/2 passed" in joined
    assert "Primary metric ratio: 0.990" in joined
