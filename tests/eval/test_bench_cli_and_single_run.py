"""
Test coverage for bench.py - targeting missing coverage areas.

This module provides comprehensive tests for the InvarLock benchmark module,
focusing on areas likely to be uncovered to push coverage from 76% to 80%+.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

from invarlock.cli.bench import main
from invarlock.eval.bench import (
    ConfigurationManager,
    ScenarioConfig,
    execute_single_run,
)
from invarlock.reporting.report_types import create_empty_report


def _report_with_artifacts(report_path: str = "report.json") -> dict[str, object]:
    report = create_empty_report()
    report["artifacts"]["report_path"] = report_path
    return report


class TestCLIAndMain:
    """Test CLI argument parsing and main function."""

    @patch("sys.argv", ["bench.py", "--edits", "quant_rtn", "--profile", "ci"])
    @patch("invarlock.cli.bench.run_guard_effect_benchmark")
    def test_main_basic_invocation(self, mock_benchmark):
        """Test basic main function invocation."""
        mock_benchmark.return_value = {"overall_pass": True}

        assert main() == 0
        mock_benchmark.assert_called_once()

    @patch("sys.argv", ["bench.py", "--edits", "invalid_edit"])
    def test_main_invalid_edit_type(self):
        """Test main function with invalid edit type."""
        assert main() == 1

    @patch("sys.argv", ["bench.py", "--edits", "quant_rtn", "--tiers", "invalid_tier"])
    def test_main_invalid_tier(self):
        """Test main function with invalid tier."""
        assert main() == 1

    @patch("sys.argv", ["bench.py", "--edits", "quant_rtn", "--probes", "-1"])
    def test_main_invalid_probe_count(self):
        """Test main function with invalid probe count."""
        assert main() == 1

    @patch("sys.argv", ["bench.py", "--edits", "quant_rtn", "--profile", "ci"])
    @patch("invarlock.cli.bench.run_guard_effect_benchmark")
    def test_main_benchmark_failure(self, mock_benchmark):
        """Test main function when benchmark fails gates."""
        mock_benchmark.return_value = {"overall_pass": False}

        assert main() == 1

    @patch("sys.argv", ["bench.py", "--edits", "quant_rtn", "--profile", "ci"])
    @patch("invarlock.cli.bench.run_guard_effect_benchmark")
    def test_main_keyboard_interrupt(self, mock_benchmark):
        """Test main function with keyboard interrupt."""
        mock_benchmark.side_effect = KeyboardInterrupt()

        assert main() == 1

    @patch("sys.argv", ["bench.py", "--edits", "quant_rtn", "--profile", "ci"])
    @patch("invarlock.cli.bench.run_guard_effect_benchmark")
    def test_main_exception_handling(self, mock_benchmark):
        """Test main function exception handling."""
        mock_benchmark.side_effect = RuntimeError("Test error")

        assert main() == 1

    @patch(
        "sys.argv",
        ["bench.py", "--edits", "quant_rtn", "--profile", "ci", "--verbose"],
    )
    @patch("invarlock.cli.bench.run_guard_effect_benchmark")
    def test_main_exception_verbose_traces(self, mock_benchmark):
        mock_benchmark.side_effect = RuntimeError("boom")
        assert main() == 1


class TestExecuteSingleRun:
    """Test execute_single_run function."""

    def test_execute_single_run_success(self, monkeypatch):
        """Test successful single run execution."""
        from types import SimpleNamespace

        from invarlock.eval.data import EvaluationWindow

        provider_kwargs_seen: list[dict[str, object]] = []

        class DummyProfile:
            def make_tokenizer(self):
                return object(), "tokhash"

        class DummyProvider:
            def windows(  # noqa: PLR0913
                self,
                tokenizer,  # noqa: ARG002
                *,
                seq_len: int,
                stride: int,  # noqa: ARG002
                preview_n: int,
                final_n: int,
                seed: int,  # noqa: ARG002
                split: str,  # noqa: ARG002
            ):
                preview = EvaluationWindow(
                    input_ids=[[1] * seq_len for _ in range(preview_n)],
                    attention_masks=[[1] * seq_len for _ in range(preview_n)],
                    indices=list(range(preview_n)),
                )
                final = EvaluationWindow(
                    input_ids=[[2] * seq_len for _ in range(final_n)],
                    attention_masks=[[1] * seq_len for _ in range(final_n)],
                    indices=list(range(final_n)),
                )
                return preview, final

        class DummyAdapter:
            def load_model(self, model_id: str, device: str = "auto", **_kwargs):
                return SimpleNamespace(name=f"{model_id}:{device}")

            def snapshot(self, _model):
                return b"snapshot"

            def restore(self, _model, _blob):
                return None

        class DummyEdit:
            name = "quant_rtn"

        class DummyRegistry:
            def get_adapter(self, _name: str):
                return DummyAdapter()

            def get_edit(self, _name: str):
                return DummyEdit()

            def get_guard(self, _name: str):
                return SimpleNamespace(name=_name)

        class DummyCoreReport:
            def __init__(self):
                self.meta = {"duration": 0.01, "guard_recovered": False}
                self.edit = {
                    "plan_digest": "pd",
                    "deltas": {"params_changed": 0, "layers_modified": 0},
                }
                self.metrics = {
                    "primary_metric": {
                        "kind": "ppl_causal",
                        "preview": 1.0,
                        "final": 1.0,
                    },
                    "latency_ms_per_tok": 1.0,
                    "memory_mb_peak": 1.0,
                }
                self.guards = {}
                self.evaluation_windows = {"preview": {}, "final": {}}
                self.status = "success"

        def _fake_execute(*_a, **_k):
            return DummyCoreReport()

        monkeypatch.setattr(
            "invarlock.model_profile.detect_model_profile",
            lambda *_a, **_k: DummyProfile(),
        )
        monkeypatch.setattr(
            "invarlock.eval.data.get_provider",
            lambda *_a, **_k: (
                provider_kwargs_seen.append(dict(_k)) or DummyProvider()
            ),
        )
        monkeypatch.setattr(
            "invarlock.core.registry.get_registry", lambda: DummyRegistry()
        )
        monkeypatch.setattr("invarlock.core.runner.CoreRunner.execute", _fake_execute)
        monkeypatch.setattr(
            "invarlock.eval.bench_runner.rmt_analysis.capture_baseline_mp_stats",
            lambda *_a, **_k: {},
        )
        monkeypatch.setattr(
            "invarlock.eval.bench_runner.rmt_detection.rmt_detect",
            lambda *_a, **_k: {"n_layers_flagged": 0},
        )

        scenario = ScenarioConfig(
            edit="quant_rtn",
            tier="balanced",
            probes=2,
            device="cpu",
        )
        run_config = ConfigurationManager.create_bare_config(scenario)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_single_run(run_config, scenario, "bare", Path(temp_dir))

        assert result.success is True
        assert result.run_type == "bare"
        assert result.report["meta"]["model_id"] == "gpt2"
        assert result.report["edit"]["name"] == "quant_rtn"
        assert provider_kwargs_seen == [{"device_hint": "cpu"}]

    def test_execute_single_run_exception_handling(self, monkeypatch):
        """Test exception handling in single run execution."""
        scenario = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=2)
        run_config = ConfigurationManager.create_bare_config(scenario)

        monkeypatch.setattr(
            "invarlock.model_profile.detect_model_profile",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_single_run(run_config, scenario, "bare", Path(temp_dir))

        assert result.success is False
        assert result.error_message is not None
        assert "boom" in result.error_message

    def test_execute_single_run_guard_construction_failure_surfaces_error(
        self, monkeypatch
    ):
        """Guard construction failures should stop guarded execution immediately."""
        from types import SimpleNamespace

        from invarlock.eval.data import EvaluationWindow

        class DummyProfile:
            def make_tokenizer(self):
                return object(), "tokhash"

        class DummyProvider:
            def windows(
                self,
                tokenizer,
                *,
                seq_len: int,
                stride: int,
                preview_n: int,
                final_n: int,
                seed: int,
                split: str,
            ):
                preview = EvaluationWindow(
                    input_ids=[[1] * seq_len for _ in range(preview_n)],
                    attention_masks=[[1] * seq_len for _ in range(preview_n)],
                    indices=list(range(preview_n)),
                )
                final = EvaluationWindow(
                    input_ids=[[2] * seq_len for _ in range(final_n)],
                    attention_masks=[[1] * seq_len for _ in range(final_n)],
                    indices=list(range(final_n)),
                )
                return preview, final

        class DummyAdapter:
            def load_model(self, model_id: str, device: str = "auto", **_kwargs):
                return SimpleNamespace(name=f"{model_id}:{device}")

            def snapshot(self, _model):
                return b"snapshot"

            def restore(self, _model, _blob):
                return None

        class DummyEdit:
            name = "quant_rtn"

        class GuardingRegistry:
            def get_adapter(self, _name: str):
                return DummyAdapter()

            def get_edit(self, _name: str):
                return DummyEdit()

            def get_guard(self, _name: str):
                raise RuntimeError("guard boom")

        calls = {"execute": 0}

        def _execute(*_args, **_kwargs):
            calls["execute"] += 1
            raise AssertionError("CoreRunner.execute should not be called")

        monkeypatch.setattr(
            "invarlock.model_profile.detect_model_profile",
            lambda *_a, **_k: DummyProfile(),
        )
        monkeypatch.setattr(
            "invarlock.eval.data.get_provider", lambda *_a, **_k: DummyProvider()
        )
        monkeypatch.setattr(
            "invarlock.core.registry.get_registry", lambda: GuardingRegistry()
        )
        monkeypatch.setattr("invarlock.core.runner.CoreRunner.execute", _execute)

        scenario = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=2)
        run_config = ConfigurationManager.create_guarded_config(scenario)

        runtime = {
            "adapter": DummyAdapter(),
            "model": SimpleNamespace(name="m"),
            "baseline_snapshot": b"snapshot",
            "pairing_schedule": {"preview": {}, "final": {}},
            "calibration_data": [],
            "tokenizer_hash": "tokhash",
            "split": "validation",
            "dataset_name": "wikitext2",
            "rmt_baseline_mp_stats": {"layer": {}},
            "rmt_baseline_sigmas": {"layer": 0.1},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_single_run(
                run_config, scenario, "guarded", Path(temp_dir), runtime=runtime
            )

        assert result.success is False
        assert result.error_message is not None
        assert "guard boom" in result.error_message
        assert calls["execute"] == 0

    def test_execute_single_run_reuses_runtime_without_recomputing_baselines(
        self, monkeypatch
    ):
        """When runtime is pre-populated, heavy setup branches are skipped."""
        from types import SimpleNamespace

        from invarlock.eval.data import EvaluationWindow

        class DummyProfile:
            def make_tokenizer(self):
                return object(), "tokhash"

        class DummyProvider:
            def windows(  # noqa: PLR0913
                self,
                tokenizer,  # noqa: ARG002
                *,
                seq_len: int,
                stride: int,  # noqa: ARG002
                preview_n: int,
                final_n: int,
                seed: int,  # noqa: ARG002
                split: str,  # noqa: ARG002
            ):
                preview = EvaluationWindow(
                    input_ids=[[1] * seq_len for _ in range(preview_n)],
                    attention_masks=[[1] * seq_len for _ in range(preview_n)],
                    indices=list(range(preview_n)),
                )
                final = EvaluationWindow(
                    input_ids=[[2] * seq_len for _ in range(final_n)],
                    attention_masks=[[1] * seq_len for _ in range(final_n)],
                    indices=list(range(final_n)),
                )
                return preview, final

        class DummyAdapter:
            def load_model(self, model_id: str, device: str = "auto", **_kwargs):
                return SimpleNamespace(name=f"{model_id}:{device}")

            def snapshot(self, _model):
                return b"snapshot"

            def restore(self, _model, _blob):
                return None

        class DummyEdit:
            name = "quant_rtn"

        class DummyRegistry:
            def get_adapter(self, _name: str):
                return DummyAdapter()

            def get_edit(self, _name: str):
                return DummyEdit()

            def get_guard(self, _name: str):
                return SimpleNamespace(name=_name)

        class DummyCoreReport:
            def __init__(self):
                self.meta = {"duration": 0.01, "guard_recovered": False}
                self.edit = {
                    "plan_digest": "pd",
                    "deltas": {"params_changed": 0, "layers_modified": 0},
                }
                self.metrics = {
                    "primary_metric": {
                        "kind": "ppl_causal",
                        "preview": 1.0,
                        "final": 1.0,
                    },
                    "latency_ms_per_tok": 1.0,
                    "memory_mb_peak": 1.0,
                }
                self.guards = {}
                self.evaluation_windows = {"preview": {}, "final": {}}
                self.status = "success"

        def _fake_execute(*_a, **_k):
            return DummyCoreReport()

        calls = {"capture": 0, "provider": 0}

        def _fake_capture(*_a, **_k):
            calls["capture"] += 1
            return {}

        def _fake_provider(*_a, **_k):
            calls["provider"] += 1
            return DummyProvider()

        monkeypatch.setattr(
            "invarlock.model_profile.detect_model_profile",
            lambda *_a, **_k: DummyProfile(),
        )
        monkeypatch.setattr("invarlock.eval.data.get_provider", _fake_provider)
        monkeypatch.setattr(
            "invarlock.core.registry.get_registry", lambda: DummyRegistry()
        )
        monkeypatch.setattr("invarlock.core.runner.CoreRunner.execute", _fake_execute)
        monkeypatch.setattr(
            "invarlock.eval.bench_runner.rmt_analysis.capture_baseline_mp_stats",
            _fake_capture,
        )
        monkeypatch.setattr(
            "invarlock.eval.bench_runner.rmt_detection.rmt_detect",
            lambda *_a, **_k: {"n_layers_flagged": 0},
        )

        scenario = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=2)
        run_config = ConfigurationManager.create_guarded_config(scenario)

        runtime = {
            "adapter": DummyAdapter(),
            "model": SimpleNamespace(name="m"),
            "baseline_snapshot": b"snapshot",
            "pairing_schedule": {"preview": {}, "final": {}},
            "calibration_data": [],
            "tokenizer_hash": "tokhash",
            "split": "validation",
            "dataset_name": "wikitext2",
            "rmt_baseline_mp_stats": {"layer": {}},
            "rmt_baseline_sigmas": {"layer": 0.1},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_single_run(
                run_config, scenario, "guarded", Path(temp_dir), runtime=runtime
            )

        assert result.success is True
        # When runtime is pre-populated, provider and capture helpers are never called.
        assert calls["provider"] == 0
        assert calls["capture"] == 0

    def test_execute_single_run_rmt_detection_failure_surfaces_error(self, monkeypatch):
        from types import SimpleNamespace

        from invarlock.eval.data import EvaluationWindow

        class DummyProfile:
            def make_tokenizer(self):
                return object(), "tokhash"

        class DummyProvider:
            def windows(
                self,
                tokenizer,
                *,
                seq_len: int,
                stride: int,
                preview_n: int,
                final_n: int,
                seed: int,
                split: str,
            ):
                preview = EvaluationWindow(
                    input_ids=[[1] * seq_len for _ in range(preview_n)],
                    attention_masks=[[1] * seq_len for _ in range(preview_n)],
                    indices=list(range(preview_n)),
                )
                final = EvaluationWindow(
                    input_ids=[[2] * seq_len for _ in range(final_n)],
                    attention_masks=[[1] * seq_len for _ in range(final_n)],
                    indices=list(range(final_n)),
                )
                return preview, final

        class DummyAdapter:
            def load_model(self, model_id: str, device: str = "auto", **_kwargs):
                return SimpleNamespace(name=f"{model_id}:{device}")

            def snapshot(self, _model):
                return b"snapshot"

            def restore(self, _model, _blob):
                return None

        class DummyEdit:
            name = "quant_rtn"

        class DummyRegistry:
            def get_adapter(self, _name: str):
                return DummyAdapter()

            def get_edit(self, _name: str):
                return DummyEdit()

            def get_guard(self, _name: str):
                return SimpleNamespace(name=_name)

        class DummyCoreReport:
            def __init__(self):
                self.meta = {"duration": 0.01, "guard_recovered": False}
                self.edit = {
                    "plan_digest": "pd",
                    "deltas": {"params_changed": 0, "layers_modified": 0},
                }
                self.metrics = {
                    "primary_metric": {
                        "kind": "ppl_causal",
                        "preview": 1.0,
                        "final": 1.0,
                    }
                }
                self.guards = {}
                self.evaluation_windows = {"preview": {}, "final": {}}
                self.status = "success"

        monkeypatch.setattr(
            "invarlock.model_profile.detect_model_profile",
            lambda *_a, **_k: DummyProfile(),
        )
        monkeypatch.setattr(
            "invarlock.eval.data.get_provider", lambda *_a, **_k: DummyProvider()
        )
        monkeypatch.setattr(
            "invarlock.core.registry.get_registry", lambda: DummyRegistry()
        )
        monkeypatch.setattr(
            "invarlock.core.runner.CoreRunner.execute",
            lambda *_a, **_k: DummyCoreReport(),
        )
        monkeypatch.setattr(
            "invarlock.eval.bench_runner.rmt_analysis.capture_baseline_mp_stats",
            lambda *_a, **_k: {},
        )
        monkeypatch.setattr(
            "invarlock.eval.bench_runner.rmt_detection.rmt_detect",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("detect boom")),
        )

        scenario = ScenarioConfig(edit="quant_rtn", tier="balanced", probes=2)
        run_config = ConfigurationManager.create_bare_config(scenario)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_single_run(run_config, scenario, "bare", Path(temp_dir))

        assert result.success is False
        assert result.error_message is not None
        assert "RMT detection failed for quant_rtn (bare)" in result.error_message
