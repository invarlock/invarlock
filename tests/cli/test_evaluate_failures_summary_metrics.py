from __future__ import annotations

import builtins
import json
from contextlib import contextmanager
from pathlib import Path

import click
import pytest
import typer

from tests.cli._support_evaluate_failures import (
    RecordingConsole,
    _fake_run_command_with_paths,
    _prepare_evaluate_paths,
    _write_json,
    mod,
    run_mod,
)


def test_evaluate_ci_profile_invalid_json_exits(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    src = Path("src")
    edt = Path("edt")
    src.mkdir()
    edt.mkdir()

    baseline_report = tmp_path / "baseline.json"
    baseline_report.write_text("{}", encoding="utf-8")
    bad_report = tmp_path / "edited.json"
    bad_report.write_text("{not-json", encoding="utf-8")

    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths({"source": baseline_report, "edited": bad_report}),
        raising=False,
    )
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    with pytest.raises(click.exceptions.Exit):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            out=str(Path("runs")),
            profile="ci",
        )


def test_evaluate_ci_nonfinite_primary_metric_exits(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    src = Path("src")
    edt = Path("edt")
    src.mkdir()
    edt.mkdir()

    baseline_report = tmp_path / "baseline.json"
    baseline_report.write_text("{}", encoding="utf-8")
    edited_report = tmp_path / "edited.json"
    edited_report.write_text(
        json.dumps(
            {
                "meta": {"device": "cpu", "adapter": "hf_causal"},
                "edit": {"name": "quant_rtn"},
                "metrics": {"primary_metric": {"final": {"bad": "value"}}},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": edited_report}
        ),
        raising=False,
    )
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)
    monkeypatch.setattr(
        mod, "resolve_command_exit_code", lambda err, profile: 9, raising=False
    )

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            out=str(Path("runs")),
            profile="ci",
        )

    assert exc.value.exit_code == 9


def test_evaluate_ci_nonfinite_primary_metric_skips_report_generation(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)

    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edited_report = _write_json(
        tmp_path / "edited.json",
        {
            "status": "rollback",
            "meta": {"device": "cpu", "adapter": "hf_causal"},
            "edit": {"name": "quant_rtn"},
            "metrics": {
                "primary_metric": {
                    "preview": 1.0,
                    "final": float("nan"),
                    "ratio_vs_baseline": float("nan"),
                }
            },
        },
    )
    report_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": edited_report}
        ),
        raising=False,
    )
    monkeypatch.setattr(
        mod,
        "generate_reports",
        lambda **kwargs: report_calls.append(kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        mod, "resolve_command_exit_code", lambda err, profile: 9, raising=False
    )

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            out=str(Path("runs")),
            profile="ci",
        )

    assert exc.value.exit_code == 9
    assert report_calls == []


def test_evaluate_ci_nonfinite_primary_metric_handles_float_cast_failure(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)

    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edited_report = _write_json(tmp_path / "edited.json", {})

    class BadFloat(float):
        def __float__(self) -> float:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": edited_report}
        ),
        raising=False,
    )
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)
    monkeypatch.setattr(
        mod.json,
        "load",
        lambda _fh: {
            "meta": {"device": "cpu", "adapter": "hf_causal"},
            "edit": {"name": "quant_rtn"},
            "metrics": {
                "primary_metric": {
                    "preview": 1.0,
                    "final": BadFloat(1.0),
                }
            },
        },
        raising=False,
    )
    monkeypatch.setattr(
        mod, "resolve_command_exit_code", lambda err, profile: 9, raising=False
    )

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            out=str(Path("runs")),
            profile="ci",
        )

    assert exc.value.exit_code == 9


def test_evaluate_helper_formatting_and_console_contexts(monkeypatch) -> None:
    console = RecordingConsole()
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

    from invarlock.cli import run_execution as run_exec_mod_local
    from invarlock.cli.commands import report as report_mod

    original_report_console = report_mod.console
    original_run_console = run_exec_mod_local.console
    with mod._suppress_child_output(True) as buffer:
        assert buffer is not None
        assert report_mod.console is not original_report_console
        assert run_exec_mod_local.console is not original_run_console
    assert report_mod.console is original_report_console
    assert run_exec_mod_local.console is original_run_console


def test_evaluate_quiet_summary_variants(tmp_path: Path, monkeypatch) -> None:
    console = RecordingConsole()
    monkeypatch.setattr(mod, "console", console)
    report_out = tmp_path / "reports"
    report_out.mkdir()

    mod._print_quiet_summary(
        report_out=report_out,
        baseline="baseline",
        subject="subject",
        profile="ci",
    )
    assert any(f"Output: {report_out}" in line for line in console.lines)

    report_path = report_out / "evaluation.report.json"
    report_path.write_text("{not-json", encoding="utf-8")
    console.calls.clear()
    mod._print_quiet_summary(
        report_out=report_out,
        baseline="baseline",
        subject="subject",
        profile="ci",
    )
    assert any(f"Output: {report_path}" in line for line in console.lines)

    report_path.write_text(json.dumps(["not-a-dict"]), encoding="utf-8")
    console.calls.clear()
    mod._print_quiet_summary(
        report_out=report_out,
        baseline="baseline",
        subject="subject",
        profile="ci",
    )
    assert any(f"Output: {report_path}" in line for line in console.lines)

    report_path.write_text(
        json.dumps({"primary_metric": {"ratio_vs_baseline": 1.2345}}),
        encoding="utf-8",
    )
    console.calls.clear()
    monkeypatch.setattr(
        "invarlock.reporting.report_console.compute_console_validation_block",
        lambda _report: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=False,
    )
    mod._print_quiet_summary(
        report_out=report_out,
        baseline="baseline",
        subject="subject",
        profile="ci",
    )
    joined = console.joined()
    assert "Status: UNKNOWN · Gates: N/A" in joined
    assert "Primary metric ratio: 1.234" in joined


def test_evaluate_yaml_tmp_dir_and_successful_quiet_summary(
    tmp_path: Path, monkeypatch
) -> None:
    payload = {"dataset": {"provider": "demo"}}
    yaml_path = tmp_path / "preset.yaml"
    mod._dump_yaml(yaml_path, payload)
    assert mod._load_yaml(yaml_path) == payload

    tmp_dir = tmp_path / "custom-evaluate-tmp"
    monkeypatch.setenv("INVARLOCK_EVALUATE_TMP_DIR", str(tmp_dir))
    assert mod._resolve_evaluate_tmp_dir() == tmp_dir
    assert tmp_dir.exists()

    monkeypatch.delenv("INVARLOCK_EVALUATE_TMP_DIR")
    monkeypatch.chdir(tmp_path)
    isolated_a = mod._resolve_evaluate_tmp_dir()
    isolated_b = mod._resolve_evaluate_tmp_dir()
    expected_parent = tmp_path / "tmp" / ".evaluate"
    assert isolated_a != isolated_b
    assert isolated_a.parent == expected_parent
    assert isolated_b.parent == expected_parent
    assert isolated_a.exists()
    assert isolated_b.exists()

    assert mod._normalize_model_id("hf:demo/model", "hf_causal") == "demo/model"

    console = RecordingConsole()
    monkeypatch.setattr(mod, "console", console)
    report_out = tmp_path / "reports"
    report_out.mkdir()
    report_path = report_out / "evaluation.report.json"
    runtime_manifest = report_out / "runtime.manifest.json"
    report_path.write_text(
        json.dumps({"primary_metric": {"ratio_vs_baseline": 0.99}}), encoding="utf-8"
    )
    runtime_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "invarlock.reporting.report_console.compute_console_validation_block",
        lambda _report: {
            "rows": [{"ok": True}, {"ok": False}],
            "overall_pass": True,
        },
        raising=False,
    )
    mod._print_quiet_summary(
        report_out=report_out,
        baseline="baseline",
        subject="subject",
        profile="release",
    )
    joined = console.joined()
    assert "Status: PASS · Gates: 1/2 passed" in joined
    assert "Primary metric ratio: 0.990" in joined
    assert f"Runtime provenance: {runtime_manifest}" in joined


def test_evaluate_prints_timing_summary_when_requested(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edited_report = _write_json(tmp_path / "edited.json", {})
    summary_calls: list[tuple[dict[str, float], list[tuple[str, str]]]] = []
    perf_values = iter([100.0, 104.0])

    @contextmanager
    def fake_timed_step(*, timings, key, **_kwargs):
        yield
        timings[key] = {
            "baseline": 1.25,
            "subject": 2.0,
            "evaluation_report": 0.5,
        }[key]

    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": edited_report}
        ),
        raising=False,
    )
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)
    monkeypatch.setattr(
        "invarlock.cli.output.timed_step",
        fake_timed_step,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.output.perf_counter",
        lambda: next(perf_values),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.output.print_timing_summary",
        lambda _console, timings, *, order, **_kwargs: summary_calls.append(
            (dict(timings), list(order))
        ),
        raising=False,
    )

    mod.evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="hf_causal",
        out=str(Path("runs")),
        report_out=str(Path("reports")),
        profile="dev",
        timing=True,
    )

    assert len(summary_calls) == 1
    timings, order = summary_calls[0]
    assert timings["baseline"] == 1.25
    assert timings["subject"] == 2.0
    assert timings["evaluation_report"] == 0.5
    assert timings["total"] == 4.0
    assert order[-1] == ("Total", "total")


def test_evaluate_degraded_primary_metric_exits_without_report_generation(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edited_report = _write_json(
        tmp_path / "edited.json",
        {
            "meta": {"device": "cpu", "adapter": "hf_causal"},
            "edit": {"name": "quant_rtn"},
            "metrics": {
                "primary_metric": {
                    "preview": 0.92,
                    "final": 0.91,
                    "ratio_vs_baseline": 1.01,
                    "degraded": True,
                    "degraded_reason": "guardrail_triggered",
                }
            },
        },
    )
    report_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": edited_report}
        ),
        raising=False,
    )
    monkeypatch.setattr(
        mod,
        "generate_reports",
        lambda **kwargs: report_calls.append(kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        mod, "resolve_command_exit_code", lambda err, profile: 7, raising=False
    )

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="release",
        )

    assert exc.value.exit_code == 7
    assert report_calls == []


def test_evaluate_quiet_summary_skips_primary_metric_line_when_ratio_missing(
    tmp_path: Path, monkeypatch
) -> None:
    console = RecordingConsole()
    monkeypatch.setattr(mod, "console", console)
    report_out = tmp_path / "reports"
    report_out.mkdir()
    _write_json(report_out / "evaluation.report.json", {"primary_metric": {}})

    monkeypatch.setattr(
        "invarlock.reporting.report_console.compute_console_validation_block",
        lambda _report: {"rows": [], "overall_pass": False},
        raising=False,
    )

    mod._print_quiet_summary(
        report_out=report_out,
        baseline="baseline",
        subject="subject",
        profile="dev",
    )

    assert "Primary metric ratio:" not in console.joined()


def test_evaluate_verbose_mode_prints_debug_lines(monkeypatch, tmp_path: Path) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edited_report = _write_json(tmp_path / "edited.json", {})
    console = RecordingConsole()

    monkeypatch.setattr(
        "invarlock.cli.output.make_console", lambda **_: console, raising=False
    )
    monkeypatch.setattr(
        mod, "resolve_auto_adapter", lambda _src: "hf_causal", raising=True
    )
    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": edited_report}
        ),
        raising=False,
    )
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    mod.evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="auto",
        out=str(Path("runs")),
        report_out=str(Path("reports")),
        profile="dev",
        verbose=True,
        banner=False,
    )

    joined = console.joined()
    assert "Adapter:auto -> hf_causal" in joined
    assert "Baseline report:" in joined
    assert "Edited report:" in joined


def test_evaluate_profile_str_failure_falls_back_to_non_ci(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edited_report = _write_json(tmp_path / "edited.json", {})
    report_calls: list[dict[str, object]] = []

    class ProfileSentinel(str):
        pass

    profile = ProfileSentinel("ci")
    profile_str_calls = 0

    def fake_str(value):
        nonlocal profile_str_calls
        if value is profile:
            profile_str_calls += 1
            if profile_str_calls == 1:
                return "ci"
            raise RuntimeError("boom")
        return builtins.str(value)

    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": edited_report}
        ),
        raising=False,
    )
    monkeypatch.setattr(
        mod,
        "generate_reports",
        lambda **kwargs: report_calls.append(kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        mod, "_dump_yaml", lambda *_args, **_kwargs: None, raising=False
    )
    monkeypatch.setattr(mod, "str", fake_str, raising=False)

    mod.evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="hf_causal",
        out=str(Path("runs")),
        report_out=str(Path("reports")),
        profile=profile,
    )

    assert profile_str_calls >= 2
    assert len(report_calls) == 1


def test_evaluate_stable_text_uses_fallback_when_stringification_raises(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edited_report = _write_json(tmp_path / "edited.json", {})

    class ProfileSentinel:
        pass

    profile = ProfileSentinel()
    profile_str_calls = 0

    def fake_str(value):
        nonlocal profile_str_calls
        if value is profile:
            profile_str_calls += 1
            if profile_str_calls == 1:
                return "ci"
            raise RuntimeError("boom")
        return builtins.str(value)

    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": edited_report}
        ),
        raising=False,
    )
    monkeypatch.setattr(mod, "generate_reports", lambda **_kwargs: None, raising=False)
    monkeypatch.setattr(
        mod, "_dump_yaml", lambda *_args, **_kwargs: None, raising=False
    )
    monkeypatch.setattr(mod, "str", fake_str, raising=False)

    mod.evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="hf_causal",
        out=str(Path("runs")),
        report_out=str(Path("reports")),
        profile=profile,
    )

    assert profile_str_calls >= 2


def test_evaluate_report_validation_failure_exits_cleanly(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edited_report = _write_json(tmp_path / "edited.json", {})

    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": edited_report}
        ),
        raising=False,
    )
    monkeypatch.setattr(
        mod,
        "generate_reports",
        lambda **_kwargs: (_ for _ in ()).throw(
            mod.ValidationError(code="E231", message="Baseline normalization failed")
        ),
        raising=False,
    )

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
        )

    assert exc.value.exit_code == 1


def test_evaluate_timing_summary_uses_accumulated_total_when_style_disables_timing(
    monkeypatch, tmp_path: Path
) -> None:
    from invarlock.cli.output import OutputStyle

    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edited_report = _write_json(tmp_path / "edited.json", {})
    summary_calls: list[dict[str, float]] = []

    @contextmanager
    def fake_timed_step(*, timings, key, **_kwargs):
        yield
        timings[key] = {
            "baseline": 1.0,
            "subject": 2.0,
            "evaluation_report": 3.0,
        }[key]

    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": edited_report}
        ),
        raising=False,
    )
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)
    monkeypatch.setattr(
        "invarlock.cli.output.resolve_output_style",
        lambda **_kwargs: OutputStyle(
            name="audit", progress=False, timing=False, color=False
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.output.timed_step",
        fake_timed_step,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.output.print_timing_summary",
        lambda _console, timings, **_kwargs: summary_calls.append(dict(timings)),
        raising=False,
    )

    mod.evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="hf_causal",
        out=str(Path("runs")),
        report_out=str(Path("reports")),
        profile="dev",
        timing=True,
    )

    assert len(summary_calls) == 1
    assert summary_calls[0]["total"] == 6.0
