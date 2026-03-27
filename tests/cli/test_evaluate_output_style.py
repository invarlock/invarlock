from __future__ import annotations

import json
from pathlib import Path

from invarlock.cli.commands.evaluate import evaluate_command


def _stub_run(out_dir: Path) -> Path:
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
        lambda **kwargs: str(_stub_run(Path(kwargs["out"]))),
        raising=False,
    )
    monkeypatch.setattr(
        cert_mod, "generate_reports", lambda **_kwargs: None, raising=False
    )

    # Deterministic time progression for: total, baseline, subject, evaluation report.
    from invarlock.cli import output as out_mod

    ticks = iter([0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 3.5, 3.5])
    monkeypatch.setattr(out_mod, "perf_counter", lambda: next(ticks))

    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
        timing=True,
        style="audit",
        progress=False,
    )

    out = capsys.readouterr().out
    assert "TIMING SUMMARY" in out
    assert "Baseline" in out and "1.00s" in out
    assert "Subject" in out and "2.00s" in out
    assert "Evaluation Report" in out and "0.50s" in out
    assert "Total" in out and "3.50s" in out
