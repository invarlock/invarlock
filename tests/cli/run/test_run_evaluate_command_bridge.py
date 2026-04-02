from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from invarlock.cli.commands.evaluate import evaluate_command


def _stub_run(out_dir: Path) -> Path:
    ts_dir = out_dir / "20250101_000000"
    ts_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "meta": {"model_id": "stub", "adapter": "hf_causal", "device": "cpu"},
        "edit": {"name": "noop"},
        "metrics": {"primary_metric": {"preview": 1.0, "final": 1.0}},
        "data": {"preview_n": 1, "final_n": 1},
    }
    report_path = ts_dir / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_evaluate_command_smoke_for_bridge(monkeypatch, tmp_path) -> None:
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

    calls = {"runs": 0, "reports": 0}

    def fake_run(**kwargs):  # noqa: ANN001
        calls["runs"] += 1
        return str(_stub_run(Path(kwargs["out"])))

    def fake_report(**_kwargs):  # noqa: ANN001
        calls["reports"] += 1

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as cert_mod

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(cert_mod, "generate_reports", fake_report, raising=False)

    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
        timing=False,
        progress=False,
    )

    assert calls["runs"] == 2
    assert calls["reports"] == 1


def test_evaluate_command_reuses_baseline_report_for_bridge(monkeypatch, tmp_path):
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

    baseline_dir = tmp_path / "baseline_report_dir"
    baseline_dir.mkdir()
    baseline_report = baseline_dir / "report.json"
    baseline_report.write_text(
        json.dumps(
            {
                "meta": {"model_id": "stub", "adapter": "hf_causal", "device": "cpu"},
                "context": {"profile": "ci", "auto": {"tier": "balanced"}},
                "edit": {"name": "noop"},
                "evaluation_windows": {
                    "preview": {"window_ids": [1], "input_ids": [[1, 2]]},
                    "final": {"window_ids": [2], "input_ids": [[3, 4]]},
                },
                "metrics": {
                    "primary_metric": {
                        "kind": "ppl_causal",
                        "preview": 1.0,
                        "final": 1.0,
                        "ratio_vs_baseline": 1.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    calls = {"runs": [], "reports": []}

    def fake_run(**kwargs):  # noqa: ANN001
        calls["runs"].append(kwargs)
        return str(_stub_run(Path(kwargs["out"])))

    def fake_report(**kwargs):  # noqa: ANN001
        calls["reports"].append(kwargs)

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as cert_mod

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(cert_mod, "generate_reports", fake_report, raising=False)

    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_report=str(baseline_report),
        adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
        timing=False,
        progress=False,
    )

    assert len(calls["runs"]) == 1
    assert Path(calls["runs"][0]["baseline"]).resolve() == baseline_report.resolve()
    assert calls["reports"] and calls["reports"][0]["baseline"] == str(
        baseline_report.resolve()
    )


def test_evaluate_command_local_mode_prefers_local_files_only(monkeypatch, tmp_path):
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
        return str(_stub_run(Path(kwargs["out"])))

    def fake_report(**_kwargs):  # noqa: ANN001
        return None

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as cert_mod

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(cert_mod, "generate_reports", fake_report, raising=False)

    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="auto",
        profile="ci",
        mode="local",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
        timing=False,
        progress=False,
    )

    assert len(captured_runs) == 2
    for call in captured_runs:
        assert call["prefer_local_files_only"] is True


def test_evaluate_command_resets_runtime_security_on_success(
    monkeypatch, tmp_path
) -> None:
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

    import invarlock.cli.commands.evaluate as eval_mod
    import invarlock.cli.commands.run as run_mod
    import invarlock.cli.security_helpers as security_helpers

    state = {"active": False, "configured": None, "enter": 0, "exit": 0}

    @contextmanager
    def fake_configure(**kwargs):
        state["configured"] = kwargs
        state["active"] = True
        state["enter"] += 1
        try:
            yield
        finally:
            state["active"] = False
            state["exit"] += 1

    def fake_run(**kwargs):
        out = Path(kwargs["out"])
        return str(_stub_run(out))

    monkeypatch.setattr(security_helpers, "configure_runtime_security", fake_configure)
    monkeypatch.setattr(eval_mod, "maybe_delegate_model_command", lambda: None)
    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(
        eval_mod, "generate_reports", lambda **_kwargs: None, raising=False
    )

    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
        timing=False,
        progress=False,
        allow_network=True,
    )

    assert state["configured"]["allow_network"] is True
    assert state["active"] is False
    assert state["enter"] == 1
    assert state["exit"] == 1


def test_evaluate_command_resets_runtime_security_on_raise(
    monkeypatch, tmp_path
) -> None:
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

    import invarlock.cli.commands.evaluate as eval_mod
    import invarlock.cli.commands.run as run_mod
    import invarlock.cli.security_helpers as security_helpers

    state = {"active": False, "configured": None, "enter": 0, "exit": 0}

    @contextmanager
    def fake_configure(**kwargs):
        state["configured"] = kwargs
        state["active"] = True
        state["enter"] += 1
        try:
            yield
        finally:
            state["active"] = False
            state["exit"] += 1

    def failing_run(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(security_helpers, "configure_runtime_security", fake_configure)
    monkeypatch.setattr(eval_mod, "maybe_delegate_model_command", lambda: None)
    monkeypatch.setattr(run_mod, "run_command", failing_run, raising=False)
    monkeypatch.setattr(
        eval_mod, "generate_reports", lambda **_kwargs: None, raising=False
    )

    with pytest.raises(RuntimeError, match="boom"):
        evaluate_command(
            baseline=str(src),
            subject=str(edt),
            adapter="auto",
            profile="ci",
            out=str(tmp_path / "runs"),
            report_out=str(tmp_path / "reports"),
            timing=False,
            progress=False,
            allow_network=True,
        )

    assert state["configured"]["allow_network"] is True
    assert state["active"] is False
    assert state["enter"] == 1
    assert state["exit"] == 1


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
        return str(_stub_run(Path(kwargs["out"])))

    def fake_report(**_kwargs):  # noqa: ANN001
        return None

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as cert_mod

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(cert_mod, "generate_reports", fake_report, raising=False)

    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
        timing=False,
        progress=False,
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
