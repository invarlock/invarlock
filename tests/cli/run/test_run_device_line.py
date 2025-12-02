from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from invarlock.cli.app import app as cli


def _cfg(tmp_path: Path) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        """
model:
  adapter: hf_gpt2
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
    )
    return str(p)


def _common_stubs(monkeypatch):
    # Registry and runner
    class DummyRegistry:
        def get_adapter(self, name):
            return SimpleNamespace(
                name=name,
                load_model=lambda *a, **k: object(),
                snapshot=lambda _m=None: b"blob",
                restore=lambda _m, _b=None: None,
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner",
        lambda: SimpleNamespace(
            execute=lambda **k: SimpleNamespace(
                edit={"deltas": {"params_changed": 0}},
                metrics={"window_overlap_fraction": 0.0, "window_match_fraction": 1.0},
                guards={},
                context={"dataset_meta": {}},
                evaluation_windows={},
                status="success",
            )
        ),
    )
    # Provider minimal
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda *a, **k: SimpleNamespace(
            windows=lambda **kw: (
                SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
            ),
            estimate_capacity=lambda **kw: {
                "available_unique": 100,
                "available_nonoverlap": 100,
                "total_tokens": 1000,
                "dedupe_rate": 0.0,
            },
        ),
    )
    # Profile + tokenizer
    monkeypatch.setattr(
        "invarlock.cli.commands.run.detect_model_profile",
        lambda *a, **k: SimpleNamespace(
            default_loss="ce",
            invariants=[],
            cert_lints=[],
            module_selectors={},
            family="test",
        ),
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.run.resolve_tokenizer",
        lambda *a, **k: (
            SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=10),
            "tokhash123",
        ),
    )


def test_device_line_cpu(tmp_path: Path, monkeypatch):
    _common_stubs(monkeypatch)
    # Force device resolution to CPU
    monkeypatch.setattr("invarlock.cli.device.resolve_device", lambda d: "cpu")

    cfg = _cfg(tmp_path)
    r = CliRunner().invoke(cli, ["run", "-c", cfg, "--profile", "dev"])
    s = r.stdout
    assert "Device: cpu (requested=auto, resolved=cpu)" in s


def test_device_line_cuda0(tmp_path: Path, monkeypatch):
    _common_stubs(monkeypatch)
    # Force device resolution to cuda:0
    monkeypatch.setattr("invarlock.cli.device.resolve_device", lambda d: "cuda:0")

    cfg = _cfg(tmp_path)
    r = CliRunner().invoke(cli, ["run", "-c", cfg, "--profile", "dev"])
    s = r.stdout
    assert "Device: cuda:0 (requested=auto, resolved=cuda:0)" in s
