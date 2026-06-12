import json
from pathlib import Path

import click
import pytest
import yaml

from invarlock.cli.commands.evaluate import evaluate_command


class _StubCLIExit(Exception):
    pass


def _stub_run(out_dir: Path, baseline: Path | None = None) -> Path:
    # Create a deterministic timestamp directory and write a minimal report.json
    ts_dir = out_dir / "20250101_000000"
    ts_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "meta": {"model_id": "stub", "adapter": "hf_causal"},
        "edit": {"name": "quant_rtn"},
        "metrics": {
            "ppl_ratio": 1.0,
            "ppl_final": 10.0,
            "timings": {
                "load_model": 0.1,
                "load_dataset": 0.2,
                "eval": 0.3,
                "finalize": 0.4,
            },
        },
        "data": {"preview_n": 1, "final_n": 1},
    }
    report_path = ts_dir / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_evaluate_orchestrates_runs_and_cert(monkeypatch, tmp_path):
    # Arrange: create HF-like dirs for source/edited
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

    calls = {"runs": [], "reports": []}

    def fake_run(**kwargs):
        out = Path(kwargs.get("out"))
        calls["runs"].append(
            {k: kwargs.get(k) for k in ["config", "profile", "out", "baseline"]}
        )
        return str(_stub_run(out))

    def fake_report(**kwargs):
        calls["reports"].append(kwargs)

    # Patch in our fakes
    # Patch the lazily imported run command at its source module
    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as mod

    monkeypatch.setattr(
        run_mod, "run_command", lambda **kwargs: fake_run(**kwargs), raising=False
    )
    monkeypatch.setattr(mod, "generate_reports", fake_report, raising=False)

    # Act
    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
    )

    # Assert: two runs and one report
    assert len(calls["runs"]) == 2
    assert len(calls["reports"]) == 1

    # Baseline first, edited second with baseline set
    assert calls["runs"][0]["baseline"] is None
    assert Path(calls["runs"][0]["out"]).name == "source"
    assert Path(calls["runs"][1]["out"]).name == "edited"
    assert calls["runs"][1]["baseline"] is not None

    # Report uses the edited run and baseline
    rep = calls["reports"][0]
    assert rep["format"] == "report"
    assert Path(rep["output"]).name == "reports"
    assert rep["baseline"] is not None and rep["run"] is not None


def test_evaluate_reuses_baseline_report_skipping_baseline_run(monkeypatch, tmp_path):
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

    baseline_report = tmp_path / "baseline_report.json"
    baseline_report.write_text(
        json.dumps(
            {
                "meta": {"model_id": str(src), "adapter": "hf_causal"},
                "context": {
                    "profile": "ci",
                    "auto": {"tier": "balanced"},
                    "assurance": {"mode": "strict"},
                },
                "edit": {"name": "noop"},
                "data": {
                    "provider": "wikitext2",
                    "split": "validation",
                    "seq_len": 512,
                    "stride": 512,
                    "preview_n": 240,
                    "final_n": 240,
                    "seed": 43,
                },
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
        out = Path(kwargs.get("out"))
        calls["runs"].append(
            {k: kwargs.get(k) for k in ["config", "profile", "out", "baseline"]}
        )
        return str(_stub_run(out))

    def fake_report(**kwargs):  # noqa: ANN001
        calls["reports"].append(kwargs)

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as mod

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(mod, "generate_reports", fake_report, raising=False)

    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_report=str(baseline_report),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
    )

    assert len(calls["runs"]) == 1
    assert Path(calls["runs"][0]["out"]).name == "edited"
    assert Path(calls["runs"][0]["baseline"]).resolve() == baseline_report.resolve()

    assert len(calls["reports"]) == 1
    rep = calls["reports"][0]
    assert Path(rep["baseline"]).resolve() == baseline_report.resolve()


def test_evaluate_can_defer_optional_rendering_and_write_timing_json(
    monkeypatch, tmp_path
):
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

    calls = {"reports": []}

    def fake_run(**kwargs):  # noqa: ANN001
        return str(_stub_run(Path(kwargs.get("out"))))

    def fake_report(**kwargs):  # noqa: ANN001
        calls["reports"].append(kwargs)

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as mod

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(mod, "generate_reports", fake_report, raising=False)

    timing_json = tmp_path / "timing" / "evaluate_timing.json"
    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
        defer_report_rendering=True,
        timing_json=str(timing_json),
    )

    assert calls["reports"][0]["render_optional"] is False
    payload = json.loads(timing_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "invarlock/evaluate-timing-v1"
    assert payload["defer_report_rendering"] is True
    assert payload["baseline_report_reused"] is False
    assert payload["timings_seconds"]["plan"] >= 0.0
    assert payload["timings_seconds"]["subject"] >= 0.0
    assert payload["timings_seconds"]["evaluation_report"] >= 0.0
    assert payload["run_timings_seconds"]["baseline"]["load_model"] == 0.1
    assert payload["run_timings_seconds"]["subject"]["eval"] == 0.3
    assert payload["aggregate_run_timings_seconds"]["load_dataset"] == 0.4


def test_evaluate_timing_helpers_handle_invalid_payloads(tmp_path: Path) -> None:
    from invarlock.cli.commands import evaluate as mod

    assert mod._coerce_timing_seconds(True) is None
    assert mod._coerce_timing_seconds("not-a-number") is None
    assert mod._load_report_payload(tmp_path / "missing.json") is None

    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert mod._load_report_payload(list_payload) is None

    assert mod._extract_run_timings_seconds(None) == {}
    assert mod._extract_run_timings_seconds({"metrics": {}}) == {}


def test_evaluate_timing_json_omits_empty_run_timings(monkeypatch, tmp_path: Path):
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

    def fake_run(**kwargs):  # noqa: ANN001
        out = Path(kwargs.get("out"))
        ts_dir = out / "20250101_000000"
        ts_dir.mkdir(parents=True, exist_ok=True)
        report_path = ts_dir / "report.json"
        report_path.write_text(json.dumps({"metrics": {}}), encoding="utf-8")
        return str(report_path)

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as mod

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(mod, "generate_reports", lambda **_: None, raising=False)

    timing_json = tmp_path / "timing" / "evaluate_timing.json"
    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
        timing_json=str(timing_json),
    )

    payload = json.loads(timing_json.read_text(encoding="utf-8"))
    assert "run_timings_seconds" not in payload
    assert "aggregate_run_timings_seconds" not in payload


def test_evaluate_releases_phase_memory_between_runs(monkeypatch, tmp_path):
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
    from invarlock.cli.commands import evaluate as mod

    releases: list[str] = []

    monkeypatch.setattr(
        run_mod,
        "run_command",
        lambda **kwargs: str(_stub_run(Path(kwargs["out"]))),
        raising=False,
    )
    monkeypatch.setattr(mod, "generate_reports", lambda **kwargs: None, raising=False)
    monkeypatch.setattr(
        mod,
        "_release_phase_memory",
        lambda: releases.append("release"),
        raising=False,
    )

    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
    )

    assert releases == ["release", "release"]


def test_evaluate_baseline_report_requires_windows(monkeypatch, tmp_path):
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

    baseline_report = tmp_path / "baseline_report.json"
    baseline_report.write_text(
        json.dumps(
            {
                "meta": {"model_id": str(src), "adapter": "hf_causal"},
                "context": {
                    "profile": "ci",
                    "auto": {"tier": "balanced"},
                    "assurance": {"mode": "strict"},
                },
                "edit": {"name": "noop"},
                "data": {
                    "provider": "wikitext2",
                    "split": "validation",
                    "seq_len": 512,
                    "stride": 512,
                    "preview_n": 64,
                    "final_n": 64,
                    "seed": 43,
                },
                "evaluation_windows": {"final": {"window_ids": [1]}},
            }
        ),
        encoding="utf-8",
    )

    import invarlock.cli.commands.run as run_mod

    monkeypatch.setattr(run_mod, "run_command", lambda **_: None, raising=False)

    with pytest.raises(click.exceptions.Exit):
        evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_report=str(baseline_report),
            baseline_adapter="auto",
            subject_adapter="auto",
            profile="ci",
            out=str(tmp_path / "runs"),
            report_out=str(tmp_path / "reports"),
        )


def test_evaluate_autogen_uses_device_auto(monkeypatch, tmp_path):
    """Auto-generated evaluate presets should not hard-code CPU device."""
    monkeypatch.chdir(tmp_path)
    # Arrange HF-like source/edited dirs so auto adapter resolves to hf_causal
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

    # Stub run/report so we don't execute real pipelines but capture device
    calls: list[dict] = []

    def fake_run(**kwargs):
        calls.append(
            {k: kwargs.get(k) for k in ("config", "profile", "out", "tier", "device")}
        )
        out = Path(kwargs.get("out"))
        return str(_stub_run(out))

    def fake_report(**_kwargs):
        return None

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as mod

    monkeypatch.setattr(
        run_mod, "run_command", lambda **kwargs: fake_run(**kwargs), raising=False
    )
    monkeypatch.setattr(mod, "generate_reports", fake_report, raising=False)

    # Act
    repo_root = Path(__file__).resolve().parents[2]
    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="ci",
        preset=str(
            repo_root / "configs" / "presets" / "causal_lm" / "wikitext2_512.yaml"
        ),
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
    )

    # Assert: temp baseline config exists and does not pin device=cpu
    scratch_files = list((tmp_path / "tmp" / ".evaluate").rglob("baseline_noop.yaml"))
    assert len(scratch_files) == 1
    baseline_yaml = scratch_files[0]
    data = yaml.safe_load(baseline_yaml.read_text(encoding="utf-8")) or {}
    model_block = data.get("model") or {}
    # Ensure the preset did not pin device=cpu
    assert model_block.get("device") != "cpu"
    # And the run call saw device=None (auto resolution) by default
    assert calls and calls[0]["device"] is None


def test_evaluate_quiet_summary_emits_status(monkeypatch, tmp_path, capsys):
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

    def fake_run(**kwargs):
        out = Path(kwargs.get("out"))
        return str(_stub_run(out))

    def fake_report(**kwargs):
        output_dir = Path(kwargs.get("output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "primary_metric": {"ratio_vs_baseline": 1.01},
            "validation": {
                "primary_metric_acceptable": True,
                "preview_final_drift_acceptable": True,
                "invariants_pass": True,
                "spectral_stable": True,
                "rmt_stable": True,
            },
        }
        (output_dir / "evaluation.report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as mod

    monkeypatch.setattr(
        run_mod, "run_command", lambda **kwargs: fake_run(**kwargs), raising=False
    )
    monkeypatch.setattr(mod, "generate_reports", fake_report, raising=False)

    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
        quiet=True,
    )

    out = capsys.readouterr().out
    assert "INVARLOCK v" in out
    assert "Status: PASS" in out
    assert "Output:" in out


def test_evaluate_container_bundle_manifest_inherits_container_execution(
    monkeypatch, tmp_path
):
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

    def fake_run(**kwargs):
        out = Path(kwargs.get("out"))
        return str(_stub_run(out))

    def fake_report(**kwargs):
        output_dir = Path(kwargs.get("output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": "v1",
            "primary_metric": {"kind": "ppl_causal", "ratio_vs_baseline": 1.0},
            "validation": {
                "primary_metric_acceptable": True,
                "preview_final_drift_acceptable": True,
                "invariants_pass": True,
                "spectral_stable": True,
                "rmt_stable": True,
            },
        }
        (output_dir / "evaluation.report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import evaluate as mod

    monkeypatch.setattr(
        run_mod, "run_command", lambda **kwargs: fake_run(**kwargs), raising=False
    )
    monkeypatch.setattr(mod, "generate_reports", fake_report, raising=False)
    monkeypatch.setattr(
        "invarlock.cli.evaluate_output.resolve_runtime_image",
        lambda: "ghcr.io/invarlock/invarlock-runtime:test",
    )
    monkeypatch.setattr(
        "invarlock.cli.evaluate_output.resolve_runtime_image_digest",
        lambda: "sha256:" + ("a" * 64),
    )

    evaluate_command(
        baseline=str(src),
        subject=str(edt),
        baseline_adapter="auto",
        subject_adapter="auto",
        profile="dev",
        assurance="off",
        execution_mode="container",
        out=str(tmp_path / "runs"),
        report_out=str(tmp_path / "reports"),
    )

    manifest = json.loads(
        (tmp_path / "reports" / "runtime.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["execution_mode"] == "container"
    assert manifest["runtime"]["container_execution"] is True
    assert manifest["runtime"]["image_digest"] == "sha256:" + ("a" * 64)


def test_evaluate_fails_when_edited_report_payload_is_not_an_object(
    monkeypatch, tmp_path
):
    src = tmp_path / "src_model"
    edt = tmp_path / "edt_model"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text("{}", encoding="utf-8")
    (edt / "config.json").write_text("{}", encoding="utf-8")

    baseline_report = _stub_run(tmp_path / "runs" / "source")
    edited_dir = tmp_path / "runs" / "edited" / "20250101_000000"
    edited_dir.mkdir(parents=True, exist_ok=True)
    edited_report = edited_dir / "report.json"
    edited_report.write_text(json.dumps(["not-an-object"]), encoding="utf-8")

    import invarlock.cli.commands.run as run_mod

    def fake_run(**kwargs):
        out = Path(kwargs["out"]).name
        if out == "source":
            return str(baseline_report)
        return str(edited_report)

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)

    with pytest.raises(click.exceptions.Exit) as exc:
        evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            out=str(tmp_path / "runs"),
            report_out=str(tmp_path / "reports"),
            profile="dev",
            assurance="off",
        )

    assert exc.value.exit_code == 1


def test_evaluate_command_defaults_to_strict_assurance(monkeypatch, tmp_path, capsys):
    src = tmp_path / "src_model"
    edt = tmp_path / "edt_model"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text("{}", encoding="utf-8")
    (edt / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(click.exceptions.Exit) as exc:
        evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            profile="dev",
            out=str(tmp_path / "runs"),
            report_out=str(tmp_path / "reports"),
        )

    assert exc.value.exit_code == 2
    assert "strict assurance requires profile ci or release" in capsys.readouterr().out
