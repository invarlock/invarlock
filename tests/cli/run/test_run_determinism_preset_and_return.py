# ruff: noqa: I001,E402,F811
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from invarlock.cli.commands.run import run_command


def _cfg(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        """
model:
  adapter: hf_causal
  id: gpt2
  device: cpu

edit:
  name: noop
  plan: {}

dataset:
  provider: synthetic
  split: validation
  seq_len: 8
  stride: 4
  preview_n: 1
  final_n: 1
  seed: 42

guards:
  order: []

eval:
  loss:
    type: auto

output:
  dir: runs
        """,
        encoding="utf-8",
    )
    return p


def test_run_command_returns_report_path_and_emits_determinism_meta(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    captured: dict[str, object] = {}

    class DummyRegistry:
        def get_adapter(self, name):  # noqa: ARG002 - stub
            return SimpleNamespace(
                name=name,
                load_model=lambda model_id, device: object(),
                snapshot=lambda model: b"x",
                restore=lambda model, blob: None,
            )

        def get_edit(self, name):  # noqa: ARG002 - stub
            return SimpleNamespace(name=name)

        def get_guard(self, name):  # noqa: ARG002 - stub
            raise KeyError(name)

        def get_plugin_metadata(self, name, t):  # noqa: ARG002 - stub
            return {"name": name, "module": f"{t}.{name}", "version": "test"}

    def _runner_exec(**_kwargs):
        return SimpleNamespace(
            edit={"plan_digest": "abcd", "deltas": {"params_changed": 0}},
            metrics={"latency_ms_per_tok": 0.0, "memory_mb_peak": 0.0},
            guards={},
            context={"dataset_meta": {}},
            evaluation_windows={
                "preview": {"logloss": [2.0], "token_counts": [8]},
                "final": {"logloss": [2.0], "token_counts": [8]},
            },
            status="success",
        )

    def _fake_emit(*, report, out_dir, filename_prefix, console):  # noqa: ARG001
        captured["report"] = report
        return {"json": str(out_dir / f"{filename_prefix}.json")}

    fake_pm = lambda *a, **k: {  # noqa: E731
        "kind": "ppl_causal",
        "preview": 10.0,
        "final": 10.0,
        "ratio_vs_baseline": 1.0,
        "display_ci": [1.0, 1.0],
    }

    with ExitStack() as stack:
        stack.enter_context(
            patch("invarlock.core.registry.get_registry", lambda: DummyRegistry())
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=_runner_exec),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(
                            input_ids=[[1, 2, 3, 4]], attention_masks=[[1, 1, 1, 1]]
                        ),
                        SimpleNamespace(
                            input_ids=[[5, 6, 7, 8]], attention_masks=[[1, 1, 1, 1]]
                        ),
                    )
                ),
            )
        )
        stack.enter_context(
            patch("invarlock.cli.device.resolve_device", lambda d: "cpu")
        )
        stack.enter_context(
            patch(
                "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.detect_model_profile",
                lambda *a, **k: SimpleNamespace(
                    default_loss="ce",
                    invariants=[],
                    cert_lints=[],
                    module_selectors={},
                    family="test",
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.resolve_tokenizer",
                lambda *a, **k: (
                    SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=10),
                    "tokhash123",
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.eval.primary_metric.compute_primary_metric_from_report",
                fake_pm,
            )
        )
        stack.enter_context(
            patch("invarlock.cli.commands.run._emit_run_artifacts", _fake_emit)
        )

        report_path = run_command(
            config=str(cfg),
            device="cpu",
            profile="ci",
            out=str(tmp_path / "runs"),
        )

    assert isinstance(report_path, str) and report_path.endswith(".json")
    report_obj = captured.get("report")
    assert isinstance(report_obj, dict)
    meta = report_obj.get("meta", {})
    assert isinstance(meta, dict)
    det = meta.get("determinism")
    assert isinstance(det, dict)
    assert det.get("level") in {"strict", "tolerance", "off"}


def test_run_command_does_not_include_determinism_when_preset_empty(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    captured: dict[str, object] = {}

    class DummyRegistry:
        def get_adapter(self, name):  # noqa: ARG002 - stub
            return SimpleNamespace(
                name=name,
                load_model=lambda model_id, device: object(),
                snapshot=lambda model: b"x",
                restore=lambda model, blob: None,
            )

        def get_edit(self, name):  # noqa: ARG002 - stub
            return SimpleNamespace(name=name)

        def get_guard(self, name):  # noqa: ARG002 - stub
            raise KeyError(name)

        def get_plugin_metadata(self, name, t):  # noqa: ARG002 - stub
            return {"name": name, "module": f"{t}.{name}", "version": "test"}

    def _runner_exec(**_kwargs):
        return SimpleNamespace(
            edit={"plan_digest": "abcd", "deltas": {"params_changed": 0}},
            metrics={"latency_ms_per_tok": 0.0, "memory_mb_peak": 0.0},
            guards={},
            context={"dataset_meta": {}},
            evaluation_windows={
                "preview": {"logloss": [2.0], "token_counts": [8]},
                "final": {"logloss": [2.0], "token_counts": [8]},
            },
            status="success",
        )

    def _fake_emit(*, report, out_dir, filename_prefix, console):  # noqa: ARG001
        captured["report"] = report
        return {"json": str(out_dir / f"{filename_prefix}.json")}

    fake_pm = lambda *a, **k: {  # noqa: E731
        "kind": "ppl_causal",
        "preview": 10.0,
        "final": 10.0,
        "ratio_vs_baseline": 1.0,
        "display_ci": [1.0, 1.0],
    }

    with ExitStack() as stack:
        stack.enter_context(
            patch("invarlock.core.registry.get_registry", lambda: DummyRegistry())
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=_runner_exec),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(
                            input_ids=[[1, 2, 3, 4]], attention_masks=[[1, 1, 1, 1]]
                        ),
                        SimpleNamespace(
                            input_ids=[[5, 6, 7, 8]], attention_masks=[[1, 1, 1, 1]]
                        ),
                    )
                ),
            )
        )
        stack.enter_context(
            patch("invarlock.cli.device.resolve_device", lambda d: "cpu")
        )
        stack.enter_context(
            patch(
                "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.detect_model_profile",
                lambda *a, **k: SimpleNamespace(
                    default_loss="ce",
                    invariants=[],
                    cert_lints=[],
                    module_selectors={},
                    family="test",
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.resolve_tokenizer",
                lambda *a, **k: (
                    SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=10),
                    "tokhash123",
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.eval.primary_metric.compute_primary_metric_from_report",
                fake_pm,
            )
        )
        stack.enter_context(
            patch("invarlock.cli.commands.run._emit_run_artifacts", _fake_emit)
        )
        stack.enter_context(
            patch("invarlock.cli.determinism.apply_determinism_preset", lambda **_k: {})
        )

        report_path = run_command(
            config=str(cfg),
            device="cpu",
            profile="ci",
            out=str(tmp_path / "runs"),
        )

    assert isinstance(report_path, str) and report_path.endswith(".json")
    report_obj = captured.get("report")
    assert isinstance(report_obj, dict)
    meta = report_obj.get("meta", {})
    assert isinstance(meta, dict)
    assert "determinism" not in meta


def test_run_command_emits_overhead_skip_marker_in_release_when_env_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_SKIP_OVERHEAD_CHECK", "1")
    cfg = _cfg(tmp_path)
    captured: dict[str, object] = {}

    class DummyRegistry:
        def get_adapter(self, name):  # noqa: ARG002 - stub
            return SimpleNamespace(
                name=name,
                load_model=lambda model_id, device: object(),
                snapshot=lambda model: b"x",
                restore=lambda model, blob: None,
            )

        def get_edit(self, name):  # noqa: ARG002 - stub
            return SimpleNamespace(name=name)

        def get_guard(self, name):  # noqa: ARG002 - stub
            raise KeyError(name)

        def get_plugin_metadata(self, name, t):  # noqa: ARG002 - stub
            return {"name": name, "module": f"{t}.{name}", "version": "test"}

    def _runner_exec(**_kwargs):
        return SimpleNamespace(
            edit={"plan_digest": "abcd", "deltas": {"params_changed": 0}},
            metrics={"latency_ms_per_tok": 0.0, "memory_mb_peak": 0.0},
            guards={},
            context={"dataset_meta": {}},
            evaluation_windows={
                "preview": {"logloss": [2.0], "token_counts": [8]},
                "final": {"logloss": [2.0], "token_counts": [8]},
            },
            status="success",
        )

    def _fake_emit(*, report, out_dir, filename_prefix, console):  # noqa: ARG001
        captured["report"] = report
        return {"json": str(out_dir / f"{filename_prefix}.json")}

    fake_pm = lambda *a, **k: {  # noqa: E731
        "kind": "ppl_causal",
        "preview": 10.0,
        "final": 10.0,
        "ratio_vs_baseline": 1.0,
        "display_ci": [1.0, 1.0],
    }

    with ExitStack() as stack:
        stack.enter_context(
            patch("invarlock.core.registry.get_registry", lambda: DummyRegistry())
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=_runner_exec),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(
                            input_ids=[[1, 2, 3, 4]], attention_masks=[[1, 1, 1, 1]]
                        ),
                        SimpleNamespace(
                            input_ids=[[5, 6, 7, 8]], attention_masks=[[1, 1, 1, 1]]
                        ),
                    )
                ),
            )
        )
        stack.enter_context(
            patch("invarlock.cli.device.resolve_device", lambda d: "cpu")
        )
        stack.enter_context(
            patch(
                "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.detect_model_profile",
                lambda *a, **k: SimpleNamespace(
                    default_loss="ce",
                    invariants=[],
                    cert_lints=[],
                    module_selectors={},
                    family="test",
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.resolve_tokenizer",
                lambda *a, **k: (
                    SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=10),
                    "tokhash123",
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.eval.primary_metric.compute_primary_metric_from_report",
                fake_pm,
            )
        )
        stack.enter_context(
            patch("invarlock.cli.commands.run._emit_run_artifacts", _fake_emit)
        )
        stack.enter_context(
            patch("invarlock.cli.determinism.apply_determinism_preset", lambda **_k: {})
        )

        report_path = run_command(
            config=str(cfg),
            device="cpu",
            profile="release",
            out=str(tmp_path / "runs"),
        )

    assert isinstance(report_path, str) and report_path.endswith(".json")
    report_obj = captured.get("report")
    assert isinstance(report_obj, dict)
    overhead = report_obj.get("guard_overhead")
    assert isinstance(overhead, dict)
    assert overhead.get("skipped") is True
    assert overhead.get("mode") == "skipped"
    assert overhead.get("source") == "env:INVARLOCK_SKIP_OVERHEAD_CHECK"
    assert overhead.get("skip_reason") == "INVARLOCK_SKIP_OVERHEAD_CHECK"
