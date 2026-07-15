from __future__ import annotations

import json
from pathlib import Path

from invarlock.cli import evaluate_output
from invarlock.cli.commands.evaluate import evaluate_command
from tests.cli._support_effective_config import preserve_effective_config


def _stub_run(out_dir: Path, *, run_kwargs: dict | None = None) -> Path:
    if run_kwargs is not None:
        preserve_effective_config(run_kwargs)
    ts_dir = out_dir / "20250101_000000"
    ts_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "meta": {"model_id": "stub", "adapter": "hf_causal"},
        "edit": {"name": "noop"},
        "metrics": {
            "primary_metric": {"preview": 1.0, "final": 1.0, "ratio_vs_baseline": 1.0}
        },
        "data": {"preview_n": 1, "final_n": 1},
    }
    report_path = ts_dir / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_evaluate_timing_block_printed(monkeypatch, tmp_path, capsys) -> None:
    src = tmp_path / "src_model"
    edt = tmp_path / "edt_model"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )
    (edt / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as cert_mod

    monkeypatch.setattr(
        run_mod,
        "run_command",
        lambda **kwargs: str(_stub_run(Path(kwargs["out"]), run_kwargs=kwargs)),
        raising=False,
    )

    def generate_reports(**kwargs) -> None:
        output = Path(kwargs["output"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "evaluation.report.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(cert_mod, "generate_reports", generate_reports, raising=False)

    # Deterministic time progression for total, plan, baseline, subject, report.
    from invarlock.cli import output as out_mod

    ticks = iter([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 3.5, 3.5])
    monkeypatch.setattr(out_mod, "perf_counter", lambda: next(ticks))

    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
        timing=True,
        style="audit",
        progress=False,
    )

    out = capsys.readouterr().out
    assert "TIMING SUMMARY" in out
    assert "Plan" in out and "0.00s" in out
    assert "Baseline" in out and "1.00s" in out
    assert "Subject" in out and "2.00s" in out
    assert "Evaluation Report" in out and "0.50s" in out
    assert "Total" in out and "3.50s" in out


def test_child_output_suppression_resolves_default_command_modules() -> None:
    import invarlock.cli.commands.report as report_mod
    import invarlock.cli.commands.run as run_mod
    import invarlock.cli.run_execution as run_execution_mod

    run_had_console = hasattr(run_mod, "console")
    original_consoles = (
        getattr(run_mod, "console", None),
        getattr(run_execution_mod, "console", None),
        getattr(report_mod, "console", None),
    )
    with evaluate_output._suppress_child_output(True) as buffer:
        assert buffer is not None
        run_mod.console.print("run output")
        run_execution_mod.console.print("execution output")
        report_mod.console.print("report output")
    assert "run output" in buffer.getvalue()
    assert "execution output" in buffer.getvalue()
    assert "report output" in buffer.getvalue()
    assert (
        getattr(run_mod, "console", None),
        getattr(run_execution_mod, "console", None),
        getattr(report_mod, "console", None),
    ) == original_consoles
    if not run_had_console:
        delattr(run_mod, "console")
