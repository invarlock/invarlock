from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from invarlock.cli.app import app
from tests.cli._support_effective_config import preserve_effective_config


@pytest.mark.parametrize(
    ("provider_flag", "provider", "expected_selection"),
    [
        ("--baseline-runtime-provider", "onnx_runtime", "baseline=onnx_runtime"),
        ("--subject-runtime-provider", "llama_cpp", "subject=llama_cpp"),
    ],
)
def test_evaluate_rejects_non_hf_runtime_provider_before_execution(
    monkeypatch,
    tmp_path: Path,
    provider_flag: str,
    provider: str,
    expected_selection: str,
) -> None:
    import invarlock.cli.commands.evaluate as evaluate_mod

    phase_calls: list[str] = []

    def _unexpected_baseline(*_args, **_kwargs):
        phase_calls.append("baseline")
        raise AssertionError("baseline evaluation must not start")

    def _unexpected_subject(*_args, **_kwargs):
        phase_calls.append("subject")
        raise AssertionError("subject evaluation must not start")

    monkeypatch.setattr(
        evaluate_mod, "run_baseline_evaluation_phase", _unexpected_baseline
    )
    monkeypatch.setattr(
        evaluate_mod, "run_subject_evaluation_phase", _unexpected_subject
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--baseline",
            "org/baseline",
            "--subject",
            "org/subject",
            "--baseline-adapter",
            "hf_causal",
            "--subject-adapter",
            "hf_causal",
            provider_flag,
            provider,
            "--profile",
            "dev",
            "--assurance",
            "off",
            "--out",
            str(tmp_path / "runs"),
            "--report-out",
            str(tmp_path / "report"),
        ],
    )

    assert result.exit_code == 2
    assert "supports only the 'hf_transformers'" in result.stdout
    assert "runtime provider" in result.stdout
    assert expected_selection in result.stdout
    assert phase_calls == []


def test_evaluate_hf_id_normalization_and_preset_fallback(monkeypatch, tmp_path: Path):
    # Patch auto adapter and run/report commands to be no-ops
    import invarlock.cli.commands.evaluate as cert_mod
    import invarlock.cli.commands.run as run_mod

    captured = []

    def _dump_yaml_capture(path: Path, data: dict):
        captured.append((Path(path), data))
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def _run_report(kwargs: dict[str, object]) -> str:
        preserve_effective_config(kwargs)
        return str(
            baseline_rep
            if Path(str(kwargs["out"])).name.endswith("source")
            else edited_rep
        )

    monkeypatch.setattr(cert_mod, "_dump_yaml", _dump_yaml_capture)
    monkeypatch.setattr(cert_mod, "resolve_auto_adapter", lambda src: "hf_causal")

    baseline_rep = tmp_path / "baseline.json"
    baseline_rep.write_text(
        json.dumps(
            {"meta": {"model_id": "m", "adapter": "hf", "seed": 0, "device": "cpu"}}
        )
    )
    edited_rep = tmp_path / "edited.json"
    edited_rep.write_text(
        json.dumps(
            {"meta": {"model_id": "m2", "adapter": "hf", "seed": 0, "device": "cpu"}}
        )
    )

    monkeypatch.setattr(
        run_mod,
        "run_command",
        lambda **kwargs: _run_report(kwargs),
    )
    # No-op report emitter
    monkeypatch.setattr(cert_mod, "generate_reports", lambda **kwargs: None)

    # Run with hf: prefix to exercise normalization; preset path is default fallback when missing
    r = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--baseline",
            "hf:org/modelA",
            "--subject",
            "hf:org/modelB",
            "--baseline-revision",
            "a" * 40,
            "--baseline-adapter",
            "auto",
            "--subject-adapter",
            "auto",
            "--baseline-runtime-provider",
            "hf_transformers",
            "--subject-runtime-provider",
            "hf_transformers",
            "--profile",
            "dev",
            "--assurance",
            "off",
            "--out",
            str(tmp_path / "runs"),
            "--report-out",
            str(tmp_path / "cert"),
        ],
    )
    assert r.exit_code == 0, r.stdout
    # First captured config is baseline; ensure hf: was stripped for HF adapter
    assert captured, "_dump_yaml should be called"
    baseline_cfg = captured[0][1]
    assert baseline_cfg["model"]["id"] == "org/modelA"
    assert baseline_cfg["model"]["model_identity"] == {
        "kind": "remote_revision",
        "revision": "a" * 40,
    }
    assert baseline_cfg["model"]["runtime_provider"] == {
        "name": "hf_transformers",
        "settings": {},
    }
    subject_cfg = captured[1][1]
    assert subject_cfg["model"]["id"] == "org/modelB"
    assert subject_cfg["model"]["runtime_provider"] == {
        "name": "hf_transformers",
        "settings": {},
    }
    assert "model_identity" not in subject_cfg["model"]


def test_evaluate_ci_aborts_on_nonfinite_pm(monkeypatch, tmp_path: Path):
    import invarlock.cli.commands.evaluate as cert_mod
    import invarlock.cli.commands.run as run_mod

    baseline_rep = tmp_path / "baseline.json"
    baseline_rep.write_text(
        json.dumps(
            {"meta": {"model_id": "m", "adapter": "hf", "seed": 0, "device": "cpu"}}
        )
    )
    edited_rep = tmp_path / "edited.json"
    # Include a primary_metric with non-finite final (None)
    edited_rep.write_text(
        json.dumps(
            {
                "meta": {"model_id": "m2", "adapter": "hf", "seed": 0, "device": "cpu"},
                "metrics": {"primary_metric": {"kind": "ppl_causal", "final": None}},
                "edit": {"name": "noop"},
            }
        )
    )

    def _run_report(kwargs: dict[str, object]) -> str:
        preserve_effective_config(kwargs)
        return str(
            baseline_rep
            if Path(str(kwargs["out"])).name.endswith("source")
            else edited_rep
        )

    monkeypatch.setattr(
        run_mod,
        "run_command",
        lambda **kwargs: _run_report(kwargs),
    )
    # No-op report emitter
    monkeypatch.setattr(cert_mod, "generate_reports", lambda **kwargs: None)
    monkeypatch.setattr(cert_mod, "resolve_auto_adapter", lambda src: "hf_causal")

    r = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--baseline",
            "hf:org/modelA",
            "--subject",
            "hf:org/modelB",
            "--baseline-revision",
            "a" * 40,
            "--subject-revision",
            "b" * 40,
            "--baseline-adapter",
            "auto",
            "--subject-adapter",
            "auto",
            "--profile",
            "ci",
            "--out",
            str(tmp_path / "runs"),
            "--report-out",
            str(tmp_path / "cert"),
        ],
    )
    # CI profile with non-finite pm should hard abort (exit 3)
    assert r.exit_code == 3
    assert "Primary metric computation failed" in r.stdout
