from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from tests.cli.run._internal_cli import internal_run_app as cli
from tests.cli.run._support_run_common import canonical_ppl_metrics


def _cfg(tmp_path: Path, *, skip_guard_metric_impact: bool = False) -> str:
    p = tmp_path / "cfg.yaml"
    context_yaml = ""
    if skip_guard_metric_impact:
        context_yaml = """
context:
  run:
    skip_guard_metric_impact_check: true
"""
    p.write_text(
        """
model:
  adapter: hf_causal
  id: gpt2
  device: auto
edit:
  name: quant_rtn
  plan: {}

dataset:
  provider: synthetic
  id: synthetic
  split: validation
  seq_len: 8
  stride: 4
  preview_n: 1
  final_n: 1

guards:
  order: []

eval:
  metric: { kind: ppl_causal }
  loss: { type: auto }

output:
  dir: runs
"""
        + context_yaml
    )
    return str(p)


def _stub_env(
    monkeypatch,
    tmp_path: Path,
    *,
    with_snapshot: bool = True,
    broken_snapshot: bool = False,
):
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
    # Device
    monkeypatch.setattr("invarlock.cli.device.resolve_device", lambda d: "cpu")
    monkeypatch.setattr(
        "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
    )

    # Registry and runner
    class DummyRegistry:
        def get_adapter(self, name):
            adapter = SimpleNamespace(
                name=name,
                load_model=lambda *a, **k: object(),
            )
            if with_snapshot:
                if broken_snapshot:
                    adapter.snapshot = lambda _m=None: (_ for _ in ()).throw(
                        AssertionError("snapshot should not be called")
                    )
                    adapter.restore = lambda _m, _b=None: None
                    adapter.snapshot_chunked = lambda _m=None: (_ for _ in ()).throw(
                        AssertionError("snapshot_chunked should not be called")
                    )
                    adapter.restore_chunked = lambda _m, _d=None: None
                else:
                    # Provide snapshot capabilities so snapshot_mode line is printed deterministically
                    adapter.snapshot = lambda _m=None: b"blob"
                    adapter.restore = lambda _m, _b=None: None
                    adapter.snapshot_chunked = lambda _m=None: str(tmp_path / "snapdir")
                    adapter.restore_chunked = lambda _m, _d=None: None
            return adapter

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())

    def _exec(**kwargs):
        return SimpleNamespace(
            edit={"deltas": {"params_changed": 0}},
            metrics=canonical_ppl_metrics(
                window_overlap_fraction=0.0,
                window_match_fraction=1.0,
            ),
            guards={},
            context={"dataset_meta": {}},
            evaluation_windows={},
            status="success",
        )

    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: SimpleNamespace(execute=_exec)
    )
    # Provider
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda *a, **k: SimpleNamespace(
            windows=lambda **kw: (
                SimpleNamespace(
                    input_ids=[
                        [idx + 1, idx + 2] for idx in range(int(kw.get("preview_n", 1)))
                    ],
                    attention_masks=[
                        [1, 1] for _ in range(int(kw.get("preview_n", 1)))
                    ],
                ),
                SimpleNamespace(
                    input_ids=[
                        [10_000 + idx + 1, 10_000 + idx + 2]
                        for idx in range(int(kw.get("final_n", 1)))
                    ],
                    attention_masks=[[1, 1] for _ in range(int(kw.get("final_n", 1)))],
                ),
            ),
            estimate_capacity=lambda **kw: {
                "available_unique": 5000,
                "available_nonoverlap": 5000,
                "total_tokens": 50000,
                "dedupe_rate": 0.0,
            },
        ),
    )
    # Profile and tokenizer
    monkeypatch.setattr(
        "invarlock.cli.run_runtime_exec.detect_model_profile",
        lambda *a, **k: SimpleNamespace(
            default_loss="ce",
            invariants=[],
            cert_lints=[],
            module_selectors={},
            family="test",
        ),
    )
    monkeypatch.setattr(
        "invarlock.cli.run_runtime_exec.resolve_tokenizer",
        lambda *a, **k: (
            SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=10),
            "tokhash123",
        ),
    )


def test_snapshot_and_cleanup_lines(tmp_path: Path, monkeypatch):
    _stub_env(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    r = CliRunner().invoke(cli, ["run", "-c", cfg, "--profile", "dev"])
    s = r.stdout
    assert "Snapshot mode:" in s
    assert ("Cleanup: removed" in s) or ("Cleanup: skipped" in s)


def test_no_cleanup_flag_skips_deletion(tmp_path: Path, monkeypatch):
    _stub_env(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    r = CliRunner().invoke(cli, ["run", "-c", cfg, "--no-cleanup", "--profile", "dev"])
    s = r.stdout
    assert "Cleanup: skipped" in s


def test_ci_skip_guard_metric_impact_reuses_loaded_model_without_snapshot(
    tmp_path: Path, monkeypatch
):
    _stub_env(monkeypatch, tmp_path, with_snapshot=False)
    monkeypatch.setattr(
        "invarlock.cli.run_metric_impact.RELEASE_MIN_WINDOWS_PER_ARM", 1
    )

    loaded_model = SimpleNamespace(name="loaded-model")
    load_calls: list[dict[str, object] | None] = []

    def _fake_load_model_with_cfg(*args, **kwargs):
        load_calls.append(kwargs.get("warning_context"))
        if len(load_calls) > 1:
            raise AssertionError("unexpected second model load")
        return loaded_model

    def _exec(**kwargs):
        assert kwargs["model"] is loaded_model
        return SimpleNamespace(
            edit={"deltas": {"params_changed": 0}},
            metrics=canonical_ppl_metrics(
                window_overlap_fraction=0.0,
                window_match_fraction=1.0,
            ),
            guards={},
            context={"dataset_meta": {}},
            evaluation_windows={},
            status="success",
        )

    monkeypatch.setattr(
        "invarlock.cli.run_runtime_exec.load_model_with_cfg", _fake_load_model_with_cfg
    )
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: SimpleNamespace(execute=_exec)
    )

    cfg = _cfg(tmp_path, skip_guard_metric_impact=True)
    result = CliRunner().invoke(cli, ["run", "-c", cfg, "--profile", "ci"])

    assert result.exit_code == 0, result.stdout
    assert len(load_calls) == 1
    assert "Reusing initially loaded model for guarded execution." in result.stdout


def test_ci_skip_guard_metric_impact_reuses_loaded_model_before_snapshot_setup(
    tmp_path: Path, monkeypatch
):
    _stub_env(monkeypatch, tmp_path, with_snapshot=True, broken_snapshot=True)
    monkeypatch.setattr(
        "invarlock.cli.run_metric_impact.RELEASE_MIN_WINDOWS_PER_ARM", 1
    )

    loaded_model = SimpleNamespace(name="loaded-model")
    load_calls: list[dict[str, object] | None] = []

    def _fake_load_model_with_cfg(*args, **kwargs):
        load_calls.append(kwargs.get("warning_context"))
        if len(load_calls) > 1:
            raise AssertionError("unexpected second model load")
        return loaded_model

    def _exec(**kwargs):
        assert kwargs["model"] is loaded_model
        return SimpleNamespace(
            edit={"deltas": {"params_changed": 0}},
            metrics=canonical_ppl_metrics(
                window_overlap_fraction=0.0,
                window_match_fraction=1.0,
            ),
            guards={},
            context={"dataset_meta": {}},
            evaluation_windows={},
            status="success",
        )

    monkeypatch.setattr(
        "invarlock.cli.run_runtime_exec.load_model_with_cfg", _fake_load_model_with_cfg
    )
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: SimpleNamespace(execute=_exec)
    )

    cfg = _cfg(tmp_path, skip_guard_metric_impact=True)
    result = CliRunner().invoke(cli, ["run", "-c", cfg, "--profile", "ci"])

    assert result.exit_code == 0, result.stdout
    assert len(load_calls) == 1
    assert "Reusing initially loaded model for guarded execution." in result.stdout
