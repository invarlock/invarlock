from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import click
import pytest
import yaml

from tests.cli._support_effective_config import preserve_effective_config
from tests.cli._support_evaluate_failures import (
    _fake_run_command_with_paths,
    _materialize_test_checkpoint,
    _prepare_evaluate_paths,
    _stub_run_dir,
    _write_json,
    mod,
    run_mod,
)


def test_profile_dataset_resolution_fails_back_to_declared_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from invarlock.cli import evaluate_phases
    from invarlock.core import config_loader

    assert (
        evaluate_phases._profile_effective_dataset_config(
            {"dataset": "invalid"}, profile_name="release"
        )
        is None
    )
    declared = {"provider": "synthetic", "preview_n": 2, "final_n": 2}
    assert (
        evaluate_phases._profile_effective_dataset_config(
            {"dataset": declared}, profile_name="dev"
        )
        is declared
    )

    monkeypatch.setattr(
        config_loader,
        "apply_profile",
        lambda *_args: (_ for _ in ()).throw(ValueError("invalid profile")),
    )
    assert (
        evaluate_phases._profile_effective_dataset_config(
            {"dataset": declared}, profile_name="release"
        )
        == declared
    )


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
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            assurance="off",
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
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            assurance="off",
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
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            out=str(Path("runs")),
            report_out=str(Path("reports")),
            profile="dev",
            assurance="off",
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
    monkeypatch.setattr(run_mod, "run_command", lambda **k: None, raising=False)
    with pytest.raises(click.exceptions.Exit):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            preset=str(tmp_path / "no_such_preset.yaml"),
            out=str(tmp_path / "runs"),
            assurance="off",
        )


def test_evaluate_uses_inline_preset_when_repo_preset_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    src = Path("src")
    edt = Path("edt")
    _materialize_test_checkpoint(src)
    _materialize_test_checkpoint(edt)
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
        baseline_adapter="hf_causal",
        subject_adapter="hf_causal",
        out=str(runs),
        report_out=str(Path("certs")),
        profile="dev",
        assurance="off",
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
    _materialize_test_checkpoint(src)
    _materialize_test_checkpoint(edt)
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
        baseline_adapter="hf_causal",
        subject_adapter="hf_causal",
        preset=str(preset),
        edit_config=str(edit_cfg),
        out=str(Path("runs")),
        report_out=str(Path("certs")),
        profile="dev",
        assurance="off",
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
        preserve_effective_config(kwargs)
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
        baseline_adapter="hf_causal",
        subject_adapter="hf_causal",
        out=str(Path("runs")),
        report_out=str(Path("reports")),
        profile="dev",
        assurance="off",
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
    _materialize_test_checkpoint(src)
    _materialize_test_checkpoint(edt)
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
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            edit_config=str(edit_cfg),
            out=str(Path("runs")),
            assurance="off",
        )


def test_evaluate_missing_baseline_report_exits(monkeypatch, tmp_path: Path):
    src = tmp_path / "src"
    edt = tmp_path / "edt"
    _materialize_test_checkpoint(src)
    _materialize_test_checkpoint(edt)
    monkeypatch.setattr(run_mod, "run_command", lambda **k: None, raising=False)
    with pytest.raises(click.exceptions.Exit):
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            out=str(tmp_path / "runs"),
            assurance="off",
        )


def test_evaluate_missing_edited_report_exits(monkeypatch, tmp_path: Path):
    src = tmp_path / "src"
    edt = tmp_path / "edt"
    _materialize_test_checkpoint(src)
    _materialize_test_checkpoint(edt)
    runs = tmp_path / "runs"

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
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            out=str(runs),
            assurance="off",
        )


def test_evaluate_failed_edited_run_report_exits_before_report_generation(
    monkeypatch, tmp_path: Path
) -> None:
    src, edt = _prepare_evaluate_paths(monkeypatch, tmp_path)

    baseline_report = _write_json(tmp_path / "baseline.json", {})
    edited_report = _write_json(
        tmp_path / "edited.json",
        {
            "status": "failed",
            "error": "[INVARLOCK:E321] RTN dequantized simulation matched no target modules.",
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

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.evaluate_command(
            baseline=str(src),
            subject=str(edt),
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            out=str(Path("runs")),
            profile="dev",
            assurance="off",
        )

    assert exc.value.exit_code == 1
    assert report_calls == []


def test_evaluate_edit_config_missing_exits(monkeypatch, tmp_path: Path):
    src = tmp_path / "src"
    edt = tmp_path / "edt"
    _materialize_test_checkpoint(src)
    _materialize_test_checkpoint(edt)
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
            baseline_adapter="hf_causal",
            subject_adapter="hf_causal",
            edit_config=str(tmp_path / "missing_edit.yaml"),
            out=str(tmp_path / "runs"),
            assurance="off",
        )


def test_evaluate_happy_path_with_preset_and_auto_adapter(monkeypatch, tmp_path: Path):
    preset = tmp_path / "preset.yaml"
    preset.write_text(
        "model: { id: x }\nedit: { name: structured, plan: {} }\n", encoding="utf-8"
    )

    src = tmp_path / "src"
    edt = tmp_path / "edt"
    _materialize_test_checkpoint(src)
    _materialize_test_checkpoint(edt)
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )

    runs = tmp_path / "runs"
    certs = tmp_path / "certs"
    calls = {"runs": 0, "reports": 0}

    def run_stub(**kwargs):
        preserve_effective_config(kwargs)
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
            baseline_adapter="auto",
            subject_adapter="auto",
            preset=str(preset),
            out=str(runs),
            report_out=str(certs),
            assurance="off",
        )

    assert calls["runs"] == 2 and calls["reports"] == 1


def test_evaluate_rejects_invalid_execution_mode() -> None:
    with pytest.raises(click.BadParameter, match="Execution mode must be one of"):
        mod.evaluate_command(
            baseline="baseline",
            subject="subject",
            execution_mode="invalid",
            assurance="off",
        )


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
        baseline_adapter="hf_causal",
        subject_adapter="hf_causal",
        out=str(Path("runs")),
        report_out=str(Path("reports")),
        profile="dev",
        quiet=True,
        timing=True,
        progress=True,
        assurance="off",
    )

    assert len(run_calls) == 2
    assert all(call["progress"] is False for call in run_calls)
    assert all(call["timing"] is False for call in run_calls)


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
        baseline_adapter="hf_causal",
        subject_adapter="hf_causal",
        preset=str(preset),
        edit_config=str(edit_cfg),
        out=str(Path("runs")),
        report_out=str(Path("reports")),
        profile="dev",
        assurance="off",
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
        baseline_adapter="hf_causal",
        subject_adapter="hf_causal",
        out=str(Path("runs")),
        report_out=str(Path("reports")),
        profile="dev",
        edit_label="quantized-subject",
        assurance="off",
    )

    assert len(run_calls) == 2
    assert run_calls[1]["edit_label"] == "quantized-subject"


def test_evaluate_passes_render_optional_to_report_contract(
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

    def report_contract(*, run, format, baseline, output, render_optional):
        report_calls.append(
            {
                "run": run,
                "format": format,
                "baseline": baseline,
                "output": output,
                "render_optional": render_optional,
            }
        )

    monkeypatch.setattr(mod, "generate_reports", report_contract, raising=False)

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

    assert report_calls == [
        {
            "run": str(edited_report),
            "format": "report",
            "baseline": str(baseline_report),
            "output": str(Path("reports")),
            "render_optional": True,
        }
    ]


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
        baseline_adapter="hf_causal",
        subject_adapter="hf_causal",
        preset=str(preset),
        out=str(Path("runs")),
        report_out=str(Path("reports")),
        profile="dev",
        assurance="off",
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
