from __future__ import annotations

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
    run_exec_mod,
    run_mod,
)


def test_evaluate_quiet_mode_replays_baseline_child_output_on_typer_exit(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    console = RecordingConsole()

    def fake_run(**kwargs):
        run_mod.console.print("baseline child output")
        raise typer.Exit(3)

    monkeypatch.setattr(
        "invarlock.cli.output.make_console", lambda **_: console, raising=False
    )
    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            quiet=True,
            assurance="off",
        )

    assert exc.value.exit_code == 3
    assert "baseline child output" in console.joined()


def test_evaluate_quiet_mode_replays_edited_child_output_on_typer_exit(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edit_config = tmp_path / "edit.yaml"
    edit_config.write_text("edit:\n  name: quant_rtn\n", encoding="utf-8")
    console = RecordingConsole()

    def fake_run(**kwargs):
        out_name = Path(kwargs["out"]).name
        if out_name == "source":
            return str(baseline_report)
        run_mod.console.print("edited child output")
        raise typer.Exit(4)

    monkeypatch.setattr(
        "invarlock.cli.output.make_console", lambda **_: console, raising=False
    )
    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            quiet=True,
            edit_config=str(edit_config),
            assurance="off",
        )

    assert exc.value.exit_code == 4
    assert "edited child output" in console.joined()


def test_evaluate_quiet_mode_replays_noop_child_output_on_runtime_error(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    console = RecordingConsole()

    def fake_run(**kwargs):
        out_name = Path(kwargs["out"]).name
        if out_name == "source":
            return str(baseline_report)
        run_mod.console.print("subject child output")
        raise RuntimeError("subject boom")

    monkeypatch.setattr(
        "invarlock.cli.output.make_console", lambda **_: console, raising=False
    )
    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)

    with pytest.raises(RuntimeError, match="subject boom"):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            quiet=True,
            assurance="off",
        )

    assert "subject child output" in console.joined()


def test_evaluate_nonquiet_edit_child_typer_exit_skips_buffer_replay(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edit_config = tmp_path / "edit.yaml"
    edit_config.write_text("edit:\n  name: quant_rtn\n", encoding="utf-8")
    console = RecordingConsole()

    def fake_run(**kwargs):
        out_name = Path(kwargs["out"]).name
        if out_name == "source":
            return str(baseline_report)
        raise typer.Exit(4)

    monkeypatch.setattr(
        "invarlock.cli.output.make_console", lambda **_: console, raising=False
    )
    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            edit_config=str(edit_config),
            assurance="off",
        )

    assert exc.value.exit_code == 4
    assert "nonquiet edited child output" not in console.joined()


def test_evaluate_quiet_mode_replays_noop_child_output_on_typer_exit(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    console = RecordingConsole()

    def fake_run(**kwargs):
        out_name = Path(kwargs["out"]).name
        if out_name == "source":
            return str(baseline_report)
        run_mod.console.print("quiet subject typer output")
        raise typer.Exit(5)

    monkeypatch.setattr(
        "invarlock.cli.output.make_console", lambda **_: console, raising=False
    )
    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            quiet=True,
            assurance="off",
        )

    assert exc.value.exit_code == 5
    assert "quiet subject typer output" in console.joined()


def test_evaluate_nonquiet_noop_child_runtime_error_skips_buffer_replay(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_report = _write_json(tmp_path / "baseline.json", {})
    console = RecordingConsole()

    def fake_run(**kwargs):
        out_name = Path(kwargs["out"]).name
        if out_name == "source":
            return str(baseline_report)
        raise RuntimeError("nonquiet subject boom")

    monkeypatch.setattr(
        "invarlock.cli.output.make_console", lambda **_: console, raising=False
    )
    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)

    with pytest.raises(RuntimeError, match="nonquiet subject boom"):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            assurance="off",
        )

    assert "nonquiet subject child output" not in console.joined()


def test_evaluate_nonquiet_subject_typer_exit_reaches_no_buffer_replay_branch(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    baseline_report = _write_json(tmp_path / "baseline-nonquiet.json", {})
    console = RecordingConsole()

    def fake_run(**kwargs):
        out_name = Path(kwargs["out"]).name
        if out_name == "source":
            return str(baseline_report)
        raise typer.Exit(6)

    monkeypatch.setattr(
        "invarlock.cli.output.make_console", lambda **_: console, raising=False
    )
    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            assurance="off",
        )

    assert exc.value.exit_code == 6


def test_evaluate_quiet_mode_replays_baseline_child_output_on_failure(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)
    console = RecordingConsole()

    def failing_run(**kwargs):
        if Path(kwargs["out"]).name == "source":
            run_exec_mod.console.print("baseline child output", markup=False)
            raise RuntimeError("baseline boom")

    monkeypatch.setattr(
        "invarlock.cli.output.make_console", lambda **_: console, raising=False
    )
    monkeypatch.setattr(run_mod, "run_command", failing_run, raising=False)

    with pytest.raises(RuntimeError, match="baseline boom"):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            quiet=True,
            assurance="off",
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
            run_exec_mod.console.print("edited child output", markup=False)
            raise RuntimeError("edited boom")

    monkeypatch.setattr(
        "invarlock.cli.output.make_console", lambda **_: console, raising=False
    )
    monkeypatch.setattr(run_mod, "run_command", failing_run, raising=False)

    with pytest.raises(RuntimeError, match="edited boom"):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            edit_config=str(edit_cfg),
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            quiet=True,
            assurance="off",
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
            run_exec_mod.console.print("noop subject output", markup=False)
            raise RuntimeError("subject boom")

    monkeypatch.setattr(
        "invarlock.cli.output.make_console", lambda **_: console, raising=False
    )
    monkeypatch.setattr(run_mod, "run_command", failing_run, raising=False)

    with pytest.raises(RuntimeError, match="subject boom"):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            quiet=True,
            assurance="off",
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
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            quiet=True,
            assurance="off",
        )

    assert "report child output" not in console.joined()


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
        run_exec_mod.console.print("edited child output", markup=False)
        raise RuntimeError("edited boom")

    monkeypatch.setattr(
        "invarlock.cli.output.make_console", lambda **_: console, raising=False
    )
    monkeypatch.setattr(run_mod, "run_command", failing_run, raising=False)

    with pytest.raises(RuntimeError, match="edited boom"):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            edit_config=str(edit_cfg),
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            quiet=False,
            assurance="off",
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
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            quiet=False,
            assurance="off",
        )

    assert "report child output" not in console.joined()
