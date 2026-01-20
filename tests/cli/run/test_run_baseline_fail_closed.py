from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from invarlock.cli.commands.run import run_command


def _write_cfg(tmp_path: Path, *, preview_n: int = 1, final_n: int = 1) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"""
model:
  adapter: hf_causal
  id: gpt2
  device: cpu
edit:
  name: quant_rtn
  plan: {{}}

dataset:
  provider: synthetic
  id: synthetic
  split: validation
  seq_len: 8
  stride: 4
  preview_n: {preview_n}
  final_n: {final_n}

guards:
  order: []

eval:
  loss:
    type: auto

output:
  dir: runs
"""
    )
    return cfg


def _patch_minimal_run():
    class DummyRegistry:
        def get_adapter(self, name: str):
            return SimpleNamespace(name=name, load_model=lambda *a, **k: object())

        def get_edit(self, name: str):
            return SimpleNamespace(name=name)

        def get_guard(self, name: str):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name: str, plugin_type: str):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    class DummyRunner:
        def execute(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                edit={"plan_digest": "abcd", "deltas": {"params_changed": 0}},
                metrics={
                    "window_overlap_fraction": 0.0,
                    "window_match_fraction": 1.0,
                    "window_pairing_reason": None,
                    "paired_windows": 1,
                    "loss_type": "ce",
                    "primary_metric": {
                        "kind": "ppl_causal",
                        "preview": 10.0,
                        "final": 10.0,
                    },
                },
                guards={},
                context={"dataset_meta": {"tokenizer_hash": "tokhash123"}},
                evaluation_windows={
                    "preview": {
                        "window_ids": [0],
                        "input_ids": [[1, 2, 3]],
                        "attention_masks": [[1, 1, 1]],
                    },
                    "final": {
                        "window_ids": [1],
                        "input_ids": [[4, 5, 6]],
                        "attention_masks": [[1, 1, 1]],
                    },
                },
                status="success",
            )

    class Provider:
        def windows(self, **kwargs):  # type: ignore[no-untyped-def]
            return (
                SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
                SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
            )

    return (
        patch("invarlock.core.registry.get_registry", lambda: DummyRegistry()),
        patch("invarlock.core.runner.CoreRunner", lambda: DummyRunner()),
        patch("invarlock.eval.data.get_provider", lambda *a, **k: Provider()),
        patch(
            "invarlock.cli.commands.run.detect_model_profile",
            lambda *a, **k: SimpleNamespace(
                default_loss="ce",
                default_provider=None,
                default_metric=None,
                family="test",
                module_selectors={},
                invariants=[],
                cert_lints=[],
            ),
        ),
        patch(
            "invarlock.cli.commands.run.resolve_tokenizer",
            lambda *a, **k: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=10),
                "tokhash123",
            ),
        ),
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch(
            "invarlock.reporting.report.save_report",
            lambda report, run_dir, formats, filename_prefix: {
                "json": str(run_dir / f"{filename_prefix}.json")
            },
        ),
        patch(
            "invarlock.cli.commands.run.validate_guard_overhead",
            lambda *a, **k: SimpleNamespace(
                passed=True,
                messages=[],
                warnings=[],
                errors=[],
                checks={},
                metrics={"overhead_ratio": 1.0, "overhead_percent": 0.0},
            ),
        ),
    )


def test_run_ci_baseline_missing_file_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _write_cfg(tmp_path)
    baseline = tmp_path / "missing.json"
    with ExitStack() as stack:
        for ctx in _patch_minimal_run():
            stack.enter_context(ctx)
        with pytest.raises(click.exceptions.Exit) as ei:
            run_command(
                config=str(cfg),
                device="cpu",
                profile="ci",
                baseline=str(baseline),
                out=str(tmp_path / "runs"),
                until_pass=False,
            )
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 3
    out = capsys.readouterr().out
    assert "[INVARLOCK:E001]" in out
    assert "PAIRING-EVIDENCE-MISSING" in out


def test_run_release_baseline_invalid_json_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _write_cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{", encoding="utf-8")
    with ExitStack() as stack:
        for ctx in _patch_minimal_run():
            stack.enter_context(ctx)
        with pytest.raises(click.exceptions.Exit) as ei:
            run_command(
                config=str(cfg),
                device="cpu",
                profile="release",
                baseline=str(baseline),
                out=str(tmp_path / "runs"),
                until_pass=False,
            )
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 3
    out = capsys.readouterr().out
    assert "[INVARLOCK:E001]" in out
    assert "PAIRING-EVIDENCE-MISSING" in out


@pytest.mark.parametrize("profile", ["ci", "release"])
def test_run_baseline_missing_evaluation_windows_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], profile: str
) -> None:
    cfg = _write_cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"meta": {"tokenizer_hash": "tokhash123"}}))
    with ExitStack() as stack:
        for ctx in _patch_minimal_run():
            stack.enter_context(ctx)
        with pytest.raises(click.exceptions.Exit) as ei:
            run_command(
                config=str(cfg),
                device="cpu",
                profile=profile,
                baseline=str(baseline),
                out=str(tmp_path / "runs"),
                until_pass=False,
            )
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 3
    out = capsys.readouterr().out
    assert "[INVARLOCK:E001]" in out
    assert "PAIRING-EVIDENCE-MISSING" in out


def test_run_ci_baseline_schedule_duplicate_windows_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _write_cfg(tmp_path, preview_n=2, final_n=2)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "evaluation_windows": {
                    "preview": {
                        "window_ids": [0, 1],
                        "input_ids": [[1, 2, 3], [1, 2, 3]],
                    },
                    "final": {
                        "window_ids": [2, 3],
                        "input_ids": [[4, 5, 6], [7, 8, 9]],
                    },
                }
            }
        )
    )
    with ExitStack() as stack:
        for ctx in _patch_minimal_run():
            stack.enter_context(ctx)
        with pytest.raises(click.exceptions.Exit) as ei:
            run_command(
                config=str(cfg),
                device="cpu",
                profile="ci",
                baseline=str(baseline),
                out=str(tmp_path / "runs"),
                until_pass=False,
            )
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 3
    out = capsys.readouterr().out
    assert "[INVARLOCK:E001]" in out
    assert "PAIRING-EVIDENCE-MISSING" in out


def test_run_ci_baseline_schedule_preview_final_overlap_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _write_cfg(tmp_path, preview_n=1, final_n=1)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "evaluation_windows": {
                    "preview": {"window_ids": [0], "input_ids": [[1, 2, 3]]},
                    "final": {"window_ids": [1], "input_ids": [[1, 2, 3]]},
                }
            }
        )
    )
    with ExitStack() as stack:
        for ctx in _patch_minimal_run():
            stack.enter_context(ctx)
        with pytest.raises(click.exceptions.Exit) as ei:
            run_command(
                config=str(cfg),
                device="cpu",
                profile="ci",
                baseline=str(baseline),
                out=str(tmp_path / "runs"),
                until_pass=False,
            )
    assert getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == 3
    out = capsys.readouterr().out
    assert "[INVARLOCK:E001]" in out
    assert "PAIRING-EVIDENCE-MISSING" in out
