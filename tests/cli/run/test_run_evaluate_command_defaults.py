from __future__ import annotations

import json
from pathlib import Path

from invarlock.cli.commands.evaluate import evaluate_command
from tests.cli.run._support_run_evaluate_command_bridge import _stub_run


def test_evaluate_command_passes_concrete_run_defaults(monkeypatch, tmp_path) -> None:
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

    captured_runs: list[dict[str, object]] = []

    def fake_run(**kwargs):  # noqa: ANN001
        captured_runs.append(kwargs)
        return str(_stub_run(Path(kwargs["out"]), run_kwargs=kwargs))

    def fake_report(**_kwargs):  # noqa: ANN001
        return None

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as cert_mod

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(cert_mod, "generate_reports", fake_report, raising=False)

    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
        timing=False,
        progress=False,
        assurance="off",
    )

    assert len(captured_runs) == 2
    for call in captured_runs:
        assert call["until_pass"] is False
        assert call["max_attempts"] == 1
        assert call["timeout"] is None
        assert call["allow_network"] is False
        assert call["allow_host_execution"] is False
        assert call["allow_third_party_plugins"] is False
        assert call["allow_remote_code"] is False
