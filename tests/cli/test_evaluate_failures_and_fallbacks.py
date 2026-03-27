from __future__ import annotations

import builtins
import json
import os
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import click
import pytest
import typer
import yaml

import invarlock.cli.commands.run as run_mod
import invarlock.core.evaluate_contract as evaluate_contract_mod
from invarlock.cli.commands import evaluate as mod
from tests.cli.support import RecordingConsole


def _stub_run_dir(out_dir: Path, name: str = "report.json") -> Path:
    ts = out_dir / "20250101_000000"
    ts.mkdir(parents=True, exist_ok=True)
    report_path = ts / name
    report_path.write_text(
        json.dumps({"meta": {}, "metrics": {}, "data": {}}), encoding="utf-8"
    )
    return report_path


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fake_run_command_with_paths(
    path_by_out_dir: dict[str, Path | None],
    *,
    run_calls: list[dict[str, object]] | None = None,
    validator: Callable[[dict[str, object], str], None] | None = None,
) -> Callable[..., str | None]:
    def _fake_run(**kwargs):
        if run_calls is not None:
            run_calls.append(kwargs)
        out_name = Path(kwargs["out"]).name
        if validator is not None:
            validator(kwargs, out_name)
        if out_name not in path_by_out_dir:
            raise AssertionError(f"Unexpected run output dir: {kwargs['out']}")
        report_path = path_by_out_dir[out_name]
        return str(report_path) if report_path is not None else None

    return _fake_run


def _valid_baseline_report_payload(
    *,
    adapter: str = "hf_causal",
    profile: str = "dev",
    tier: str = "balanced",
    edit_name: str = "noop",
    evaluation_windows: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "edit": {"name": edit_name},
        "meta": {"adapter": adapter},
        "context": {"profile": profile, "auto": {"tier": tier}},
        "evaluation_windows": evaluation_windows
        or {
            "preview": {"window_ids": ["preview-0"], "input_ids": [[1, 2, 3]]},
            "final": {"window_ids": ["final-0"], "input_ids": [[4, 5, 6]]},
        },
    }


def _prepare_evaluate_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    monkeypatch.chdir(tmp_path)
    src = Path("src")
    edt = Path("edt")
    src.mkdir()
    edt.mkdir()
    return src, edt


def _assert_baseline_report_validation_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    payload: object | None = None,
    raw_text: str | None = None,
    profile: str = "dev",
    tier: str = "balanced",
) -> click.exceptions.Exit:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_path = Path("baseline.json")
    if raw_text is not None:
        baseline_path.write_text(raw_text, encoding="utf-8")
    else:
        assert payload is not None
        baseline_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(run_mod, "run_command", lambda **_: None, raising=False)
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            baseline_report=str(baseline_path),
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile=profile,
            tier=tier,
        )

    return exc.value


def test_evaluate_requires_explicit_runner_report_path(monkeypatch, tmp_path: Path):
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)

    def fake_run(**kwargs):
        _stub_run_dir(Path(kwargs["out"]))
        return None

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)

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


def test_evaluate_requires_existing_runner_report_path(monkeypatch, tmp_path: Path):
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)

    missing_report = tmp_path / "missing-report.json"
    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": missing_report, "edited": missing_report}
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


def test_evaluate_requires_file_runner_report_path(monkeypatch, tmp_path: Path):
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)

    report_dir = tmp_path / "runner-report-dir"
    report_dir.mkdir()
    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths({"source": report_dir, "edited": report_dir}),
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


def test_normalize_model_id_handles_bad_adapter():
    class BadAdapter:
        def __str__(self):  # pragma: no cover - invoked indirectly
            raise RuntimeError("boom")

    result = mod._normalize_model_id("hf:demo/model", BadAdapter())
    assert result == "hf:demo/model"


def test_load_yaml_rejects_non_mapping(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("- this is a list, not a mapping", encoding="utf-8")
    with pytest.raises(ValueError):
        mod._load_yaml(p)


def test_evaluate_missing_preset_exits(monkeypatch, tmp_path: Path):
    src = tmp_path / "src"
    edt = tmp_path / "edt"
    src.mkdir()
    edt.mkdir()
    # Patch to prevent actual run invocation
    monkeypatch.setattr(run_mod, "run_command", lambda **k: None, raising=False)
    with pytest.raises(click.exceptions.Exit):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            preset=str(tmp_path / "no_such_preset.yaml"),
            out=str(tmp_path / "runs"),
        )


def test_evaluate_uses_inline_preset_when_repo_preset_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    src = Path("src")
    edt = Path("edt")
    src.mkdir()
    edt.mkdir()
    runs = Path("runs")
    run_calls: list[dict[str, object]] = []
    report_calls: list[dict[str, object]] = []

    baseline_report = tmp_path / "baseline.json"
    baseline_report.write_text("{}", encoding="utf-8")
    edited_report = tmp_path / "edited.json"
    edited_report.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": edited_report},
            run_calls=run_calls,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        mod,
        "generate_reports",
        lambda **kwargs: report_calls.append(kwargs),
        raising=False,
    )

    mod.evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="hf_causal",
        out=str(runs),
        report_out=str(Path("certs")),
        profile="dev",
    )

    cfg_candidates = list((Path("tmp") / ".evaluate").rglob("baseline_noop.yaml"))
    assert len(cfg_candidates) == 1
    cfg_path = cfg_candidates[0]
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["dataset"]["provider"] == "wikitext2"
    assert len(run_calls) == 2
    assert len(report_calls) == 1


def test_evaluate_edit_config_successfully_merges_subject(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    src = Path("src")
    edt = Path("edt")
    src.mkdir()
    edt.mkdir()
    preset = Path("preset.yaml")
    preset.write_text("dataset: { provider: demo }\n", encoding="utf-8")
    edit_cfg = Path("edit_config.yaml")
    edit_cfg.write_text(
        "model:\n  id: \"<MODEL_ID>\"\n  adapter: ''\nedit:\n  name: quant_rtn\n  plan: {}\n",
        encoding="utf-8",
    )

    baseline_report = tmp_path / "baseline.json"
    baseline_report.write_text("{}", encoding="utf-8")
    edited_report = tmp_path / "edited.json"
    edited_report.write_text("{}", encoding="utf-8")

    calls = {"runs": 0}

    def validate_run(kwargs: dict[str, object], out_name: str) -> None:
        calls["runs"] += 1
        if out_name == "source":
            assert kwargs.get("baseline") is None
        else:
            assert kwargs.get("baseline") == str(baseline_report)

    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": edited_report},
            validator=validate_run,
        ),
        raising=False,
    )
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    mod.evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="hf_causal",
        preset=str(preset),
        edit_config=str(edit_cfg),
        out=str(Path("runs")),
        report_out=str(Path("certs")),
        profile="dev",
    )

    merged_candidates = list((Path("tmp") / ".evaluate").rglob("edited_merged.yaml"))
    assert len(merged_candidates) == 1
    merged = yaml.safe_load(merged_candidates[0].read_text(encoding="utf-8"))
    assert merged["model"]["id"] == str(edt)
    assert merged["model"]["adapter"] == "hf_causal"
    assert calls["runs"] == 2


def test_evaluate_uses_returned_run_report_path_over_directory_scan(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    report_calls: list[dict[str, object]] = []

    def fake_run(**kwargs):
        out = Path(kwargs["out"])
        report_path = _stub_run_dir(out)
        if out.name == "source":
            stale = out / "20260326_003657"
            stale.mkdir(parents=True, exist_ok=True)
            (stale / "events.jsonl").write_text("", encoding="utf-8")
        return str(report_path)

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(
        mod,
        "generate_reports",
        lambda **kwargs: report_calls.append(kwargs),
        raising=False,
    )

    mod.evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="hf_causal",
        out=str(Path("runs")),
        report_out=str(Path("reports")),
        profile="dev",
    )

    assert len(report_calls) == 1
    assert report_calls[0]["baseline"].endswith(
        "runs/source/20250101_000000/report.json"
    )
    assert report_calls[0]["run"].endswith("runs/edited/20250101_000000/report.json")


def test_evaluate_edit_config_invalid_yaml_exits(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    src = Path("src")
    edt = Path("edt")
    src.mkdir()
    edt.mkdir()
    edit_cfg = Path("edit_config.yaml")
    edit_cfg.write_text("- not a mapping", encoding="utf-8")
    baseline_report = tmp_path / "baseline.json"
    baseline_report.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": baseline_report}
        ),
        raising=False,
    )

    with pytest.raises(click.exceptions.Exit):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            edit_config=str(edit_cfg),
            out=str(Path("runs")),
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
        evaluate_contract_mod,
        "resolve_command_exit_code",
        lambda err, profile: 9,
        raising=False,
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
        evaluate_contract_mod,
        "resolve_command_exit_code",
        lambda err, profile: 9,
        raising=False,
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


def test_evaluate_missing_baseline_report_exits(monkeypatch, tmp_path: Path):
    src = tmp_path / "src"
    edt = tmp_path / "edt"
    src.mkdir()
    edt.mkdir()
    # Fake run does not create any reports
    monkeypatch.setattr(run_mod, "run_command", lambda **k: None, raising=False)
    with pytest.raises(click.exceptions.Exit):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            out=str(tmp_path / "runs"),
        )


def test_evaluate_missing_edited_report_exits(monkeypatch, tmp_path: Path):
    src = tmp_path / "src"
    edt = tmp_path / "edt"
    src.mkdir()
    edt.mkdir()
    runs = tmp_path / "runs"

    # Baseline run produces a report, edited run produces nothing
    def fake_run(**kwargs):
        out = Path(kwargs["out"])
        if out.name == "source":
            return str(_stub_run_dir(out))
        return None

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    with pytest.raises(click.exceptions.Exit):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            out=str(runs),
        )


def test_evaluate_edit_config_missing_exits(monkeypatch, tmp_path: Path):
    src = tmp_path / "src"
    edt = tmp_path / "edt"
    src.mkdir()
    edt.mkdir()
    baseline_report = _stub_run_dir(Path(tmp_path / "runs" / "source"))
    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": tmp_path / "edited.json"}
        ),
        raising=False,
    )
    with pytest.raises(click.exceptions.Exit):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            edit_config=str(tmp_path / "missing_edit.yaml"),
            out=str(tmp_path / "runs"),
        )


def test_evaluate_happy_path_with_preset_and_auto_adapter(monkeypatch, tmp_path: Path):
    # Create a minimal fake preset
    preset = tmp_path / "preset.yaml"
    preset.write_text(
        "model: { id: x }\nedit: { name: structured, plan: {} }\n", encoding="utf-8"
    )

    src = tmp_path / "src"
    edt = tmp_path / "edt"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )

    runs = tmp_path / "runs"
    certs = tmp_path / "certs"
    calls = {"runs": 0, "reports": 0}

    def run_stub(**kwargs):
        calls["runs"] += 1
        out = Path(kwargs["out"])
        return str(_stub_run_dir(out))

    def report_stub(**kwargs):
        calls["reports"] += 1

    with ExitStack() as stack:
        stack.enter_context(patch.object(run_mod, "run_command", run_stub))
        stack.enter_context(patch.object(mod, "generate_reports", report_stub))
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="auto",
            preset=str(preset),
            out=str(runs),
            report_out=str(certs),
        )

    assert calls["runs"] == 2 and calls["reports"] == 1


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

    from invarlock.cli.commands import report as report_mod
    from invarlock.cli.commands import run as run_mod_local

    original_report_console = report_mod.console
    original_run_console = run_mod_local.console
    with mod._suppress_child_output(True) as buffer:
        assert buffer is not None
        assert report_mod.console is not original_report_console
        assert run_mod_local.console is not original_run_console
    assert report_mod.console is original_report_console
    assert run_mod_local.console is original_run_console


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
        "invarlock.reporting.render.compute_console_validation_block",
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
        baseline="baseline",
        subject="subject",
        profile="release",
    )
    joined = console.joined()
    assert "Status: PASS · Gates: 1/2 passed" in joined
    assert "Primary metric ratio: 0.990" in joined


def test_evaluate_rejects_invalid_execution_mode() -> None:
    with pytest.raises(click.BadParameter, match="Execution mode must be one of"):
        mod.evaluate_command(baseline="baseline", subject="subject", mode="invalid")


def test_evaluate_quiet_mode_disables_progress_and_timing(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edited_report = _write_json(tmp_path / "edited.json", {})
    run_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": edited_report},
            run_calls=run_calls,
        ),
        raising=False,
    )
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    mod.evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="hf_causal",
        out=str(Path("runs")),
        report_out=str(Path("reports")),
        profile="dev",
        quiet=True,
        timing=True,
        progress=True,
    )

    assert len(run_calls) == 2
    assert all(call["progress"] is False for call in run_calls)
    assert all(call["timing"] is False for call in run_calls)


def test_evaluate_rejects_baseline_report_directory_even_with_report_json(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_dir = tmp_path / "baseline-run"
    baseline_dir.mkdir()
    _write_json(
        baseline_dir / "report.json",
        _valid_baseline_report_payload(),
    )
    run_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        run_mod,
        "run_command",
        lambda **kwargs: run_calls.append(kwargs),
        raising=False,
    )
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            baseline_report=str(baseline_dir),
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
        )

    assert exc.value.exit_code == 2
    assert run_calls == []


def test_evaluate_rejects_baseline_report_directory_without_report_json(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_dir = tmp_path / "baseline-run"
    baseline_dir.mkdir()
    _write_json(
        baseline_dir / "20250101_000000.json",
        _valid_baseline_report_payload(),
    )
    run_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        run_mod,
        "run_command",
        lambda **kwargs: run_calls.append(kwargs),
        raising=False,
    )
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            baseline_report=str(baseline_dir),
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
        )

    assert exc.value.exit_code == 2
    assert run_calls == []


def test_evaluate_baseline_report_invalid_json_exits(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        raw_text="{not-json",
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_requires_json_object(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=["not-a-dict"],
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_requires_noop_edit(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=_valid_baseline_report_payload(edit_name="quant_rtn"),
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_rejects_adapter_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=_valid_baseline_report_payload(adapter="hf_awq"),
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_rejects_profile_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=_valid_baseline_report_payload(profile="release"),
        profile="dev",
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_rejects_tier_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=_valid_baseline_report_payload(tier="strict"),
        tier="balanced",
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_requires_evaluation_windows_payload(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload={
            **_valid_baseline_report_payload(),
            "evaluation_windows": None,
        },
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_rejects_non_regular_file(
    monkeypatch, tmp_path: Path
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo unavailable on this platform")

    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_fifo = tmp_path / "baseline.pipe"
    os.mkfifo(baseline_fifo)

    monkeypatch.setattr(run_mod, "run_command", lambda **_: None, raising=False)
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            baseline_report=str(baseline_fifo),
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
        )

    assert exc.value.exit_code == 2


def test_evaluate_baseline_report_requires_preview_payload(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=_valid_baseline_report_payload(
            evaluation_windows={
                "preview": None,
                "final": {"window_ids": ["final-0"], "input_ids": [[4, 5, 6]]},
            }
        ),
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_requires_matching_window_lengths(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=_valid_baseline_report_payload(
            evaluation_windows={
                "preview": {"window_ids": ["preview-0"], "input_ids": [[1, 2, 3]]},
                "final": {
                    "window_ids": ["final-0", "final-1"],
                    "input_ids": [[4, 5, 6]],
                },
            }
        ),
    )

    assert exc.exit_code == 2


def test_evaluate_edit_config_preserves_explicit_adapter_and_guard_order(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    preset = Path("preset.yaml")
    preset.write_text("guards:\n  order:\n    - invariants\n", encoding="utf-8")
    edit_cfg = Path("edit_config.yaml")
    edit_cfg.write_text(
        "model:\n"
        "  id: hf:demo/model\n"
        "  adapter: custom_adapter\n"
        "guards:\n"
        "  order:\n"
        "    - custom_guard\n"
        "edit:\n"
        "  name: quant_rtn\n"
        "  plan: {}\n",
        encoding="utf-8",
    )
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
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    mod.evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="hf_causal",
        preset=str(preset),
        edit_config=str(edit_cfg),
        out=str(Path("runs")),
        report_out=str(Path("reports")),
        profile="dev",
    )

    merged_candidates = list((Path("tmp") / ".evaluate").rglob("edited_merged.yaml"))
    assert len(merged_candidates) == 1
    merged = yaml.safe_load(merged_candidates[0].read_text(encoding="utf-8"))
    assert merged["model"]["id"] == "demo/model"
    assert merged["model"]["adapter"] == "custom_adapter"
    assert merged["guards"]["order"] == ["custom_guard"]


def test_evaluate_edit_label_is_forwarded_to_subject_run(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edited_report = _write_json(tmp_path / "edited.json", {})
    run_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": edited_report},
            run_calls=run_calls,
        ),
        raising=False,
    )
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    mod.evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="hf_causal",
        out=str(Path("runs")),
        report_out=str(Path("reports")),
        profile="dev",
        edit_label="quantized-subject",
    )

    assert len(run_calls) == 2
    assert run_calls[1]["edit_label"] == "quantized-subject"


def test_evaluate_filters_report_kwargs_to_supported_signature(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edited_report = _write_json(tmp_path / "edited.json", {})
    report_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": edited_report}
        ),
        raising=False,
    )

    def limited_report(*, run, format, baseline, output):
        report_calls.append(
            {
                "run": run,
                "format": format,
                "baseline": baseline,
                "output": output,
            }
        )

    monkeypatch.setattr(mod, "generate_reports", limited_report, raising=False)

    mod.evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="hf_causal",
        out=str(Path("runs")),
        report_out=str(Path("reports")),
        profile="dev",
    )

    assert report_calls == [
        {
            "run": str(edited_report),
            "format": "report",
            "baseline": str(baseline_report),
            "output": str(Path("reports")),
        }
    ]


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


def test_evaluate_degraded_primary_metric_emits_report_and_exits(
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
        evaluate_contract_mod,
        "resolve_command_exit_code",
        lambda err, profile: 7,
        raising=False,
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
    assert len(report_calls) == 1
    assert report_calls[0]["run"] == str(edited_report)


def test_evaluate_quiet_summary_skips_primary_metric_line_when_ratio_missing(
    tmp_path: Path, monkeypatch
) -> None:
    console = RecordingConsole()
    monkeypatch.setattr(mod, "console", console)
    report_out = tmp_path / "reports"
    report_out.mkdir()
    _write_json(report_out / "evaluation.report.json", {"primary_metric": {}})

    monkeypatch.setattr(
        "invarlock.reporting.render.compute_console_validation_block",
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


def test_evaluate_invalid_preset_guard_order_falls_back_to_default(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    preset = Path("preset.yaml")
    preset.write_text(
        "guards:\n  order:\n    - invariants\n    - 3\n",
        encoding="utf-8",
    )
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
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    mod.evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="hf_causal",
        preset=str(preset),
        out=str(Path("runs")),
        report_out=str(Path("reports")),
        profile="dev",
    )

    baseline_candidates = list((Path("tmp") / ".evaluate").rglob("baseline_noop.yaml"))
    assert len(baseline_candidates) == 1
    baseline_cfg = yaml.safe_load(baseline_candidates[0].read_text(encoding="utf-8"))
    assert baseline_cfg["guards"]["order"] == [
        "invariants",
        "spectral",
        "rmt",
        "variance",
        "invariants",
    ]


def test_evaluate_supplied_baseline_report_path_must_exist(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(run_mod, "run_command", lambda **_: None, raising=False)
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            baseline_report="missing.json",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
        )

    assert exc.value.exit_code == 2


def test_evaluate_supplied_baseline_report_directory_requires_a_report_file(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_dir = Path("baseline-dir")
    baseline_dir.mkdir()
    monkeypatch.setattr(run_mod, "run_command", lambda **_: None, raising=False)
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            baseline_report=str(baseline_dir),
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
        )

    assert exc.value.exit_code == 2


def test_evaluate_baseline_report_accepts_non_mapping_meta_and_context(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_payload = {
        **_valid_baseline_report_payload(),
        "meta": "bad-meta",
        "context": "bad-context",
    }
    baseline_report = _write_json(tmp_path / "baseline.json", baseline_payload)
    edited_report = _write_json(tmp_path / "edited.json", {})
    run_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": edited_report, "edited": edited_report},
            run_calls=run_calls,
        ),
        raising=False,
    )
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    mod.evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="hf_causal",
        baseline_report=str(baseline_report),
        out=str(Path("runs")),
        report_out=str(Path("reports")),
        profile="dev",
    )

    assert len(run_calls) == 1
    assert run_calls[0]["baseline"] == str(baseline_report.resolve())


def test_evaluate_baseline_report_requires_nonempty_preview_window_ids(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=_valid_baseline_report_payload(
            evaluation_windows={
                "preview": {"window_ids": [], "input_ids": [[1, 2, 3]]},
                "final": {"window_ids": ["final-0"], "input_ids": [[4, 5, 6]]},
            }
        ),
    )

    assert exc.exit_code == 2


def test_evaluate_baseline_report_requires_nonempty_preview_input_ids(
    monkeypatch, tmp_path: Path
) -> None:
    exc = _assert_baseline_report_validation_exit(
        monkeypatch,
        tmp_path,
        payload=_valid_baseline_report_payload(
            evaluation_windows={
                "preview": {"window_ids": ["preview-0"], "input_ids": []},
                "final": {"window_ids": ["final-0"], "input_ids": [[4, 5, 6]]},
            }
        ),
    )

    assert exc.exit_code == 2


def test_evaluate_quiet_mode_replays_baseline_child_output_on_failure(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    console = RecordingConsole()

    def failing_run(**kwargs):
        if Path(kwargs["out"]).name == "source":
            run_mod.console.print("baseline child output", markup=False)
            raise RuntimeError("baseline boom")

    monkeypatch.setattr(
        "invarlock.cli.output.make_console", lambda **_: console, raising=False
    )
    monkeypatch.setattr(run_mod, "run_command", failing_run, raising=False)

    with pytest.raises(RuntimeError, match="baseline boom"):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            quiet=True,
        )

    assert "baseline child output" in console.joined()


def test_evaluate_quiet_mode_replays_edit_config_child_output_on_failure(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    console = RecordingConsole()
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edit_cfg = Path("edit_config.yaml")
    edit_cfg.write_text(
        "model:\n  id: <MODEL_ID>\nedit:\n  name: quant_rtn\n  plan: {}\n",
        encoding="utf-8",
    )

    def failing_run(**kwargs):
        if Path(kwargs["out"]).name == "source":
            return str(baseline_report)
        if Path(kwargs["out"]).name == "edited":
            run_mod.console.print("edited child output", markup=False)
            raise RuntimeError("edited boom")

    monkeypatch.setattr(
        "invarlock.cli.output.make_console", lambda **_: console, raising=False
    )
    monkeypatch.setattr(run_mod, "run_command", failing_run, raising=False)

    with pytest.raises(RuntimeError, match="edited boom"):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            edit_config=str(edit_cfg),
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            quiet=True,
        )

    assert "edited child output" in console.joined()


def test_evaluate_quiet_mode_replays_noop_subject_output_on_failure(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    console = RecordingConsole()
    baseline_report = _write_json(tmp_path / "baseline.json", {})

    def failing_run(**kwargs):
        if Path(kwargs["out"]).name == "source":
            return str(baseline_report)
        if Path(kwargs["out"]).name == "edited":
            run_mod.console.print("noop subject output", markup=False)
            raise RuntimeError("subject boom")

    monkeypatch.setattr(
        "invarlock.cli.output.make_console", lambda **_: console, raising=False
    )
    monkeypatch.setattr(run_mod, "run_command", failing_run, raising=False)

    with pytest.raises(RuntimeError, match="subject boom"):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            quiet=True,
        )

    assert "noop subject output" in console.joined()


def test_evaluate_quiet_mode_report_failure_bubbles_without_child_replay(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    console = RecordingConsole()
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edited_report = _write_json(tmp_path / "edited.json", {})

    def failing_report(**_kwargs):
        raise RuntimeError("report boom")

    monkeypatch.setattr(
        "invarlock.cli.output.make_console", lambda **_: console, raising=False
    )
    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": edited_report}
        ),
        raising=False,
    )
    monkeypatch.setattr(mod, "generate_reports", failing_report, raising=False)

    with pytest.raises(RuntimeError, match="report boom"):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            quiet=True,
        )

    assert "report child output" not in console.joined()


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


def test_evaluate_non_quiet_edit_config_failure_does_not_replay_buffer(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    console = RecordingConsole()
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edit_cfg = tmp_path / "edit.yaml"
    edit_cfg.write_text("edit: {}\n", encoding="utf-8")

    def failing_run(**kwargs):
        out_name = Path(kwargs["out"]).name
        if out_name == "source":
            return str(baseline_report)
        run_mod.console.print("edited child output", markup=False)
        raise RuntimeError("edited boom")

    monkeypatch.setattr(
        "invarlock.cli.output.make_console", lambda **_: console, raising=False
    )
    monkeypatch.setattr(run_mod, "run_command", failing_run, raising=False)

    with pytest.raises(RuntimeError, match="edited boom"):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            edit_config=str(edit_cfg),
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            quiet=False,
        )

    assert "edited child output" not in console.joined()


def test_evaluate_non_quiet_report_failure_bubbles_without_child_replay(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    console = RecordingConsole()
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edited_report = _write_json(tmp_path / "edited.json", {})

    def failing_report(**_kwargs):
        raise RuntimeError("report boom")

    monkeypatch.setattr(
        "invarlock.cli.output.make_console", lambda **_: console, raising=False
    )
    monkeypatch.setattr(
        run_mod,
        "run_command",
        _fake_run_command_with_paths(
            {"source": baseline_report, "edited": edited_report}
        ),
        raising=False,
    )
    monkeypatch.setattr(mod, "generate_reports", failing_report, raising=False)

    with pytest.raises(RuntimeError, match="report boom"):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            quiet=False,
        )

    assert "report child output" not in console.joined()


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
