from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from invarlock.cli.app import app as cli


def _cfg(tmp_path: Path, *, provider: str = "synthetic") -> str:
    p = tmp_path / "cfg.yaml"
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
  provider: __PROVIDER__
  id: __PROVIDER__
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
""".replace("__PROVIDER__", provider)
    )
    return str(p)


def _common_stubs(monkeypatch) -> None:
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


def test_run_ci_uses_semantic_prefixes_no_emojis(tmp_path: Path, monkeypatch) -> None:
    _common_stubs(monkeypatch)
    monkeypatch.setattr("invarlock.cli.device.resolve_device", lambda _d: "cpu")

    cfg = _cfg(tmp_path)
    r = CliRunner().invoke(
        cli, ["run", "-c", cfg, "--profile", "ci", "--style", "audit"]
    )
    assert r.exit_code == 0
    s = r.stdout
    assert "[INIT]" in s or "[EXEC]" in s or "[DATA]" in s
    for emoji in [
        "🚀",
        "📋",
        "🔧",
        "🛡️",
        "📜",
        "✅",
        "❌",
        "⚠️",
        "📊",
        "📚",
        "💾",
        "🧹",
        "✂️",
        "🧪",
        "🏁",
    ]:
        assert emoji not in s


def test_run_audit_routes_provider_events_without_emojis(
    tmp_path: Path, monkeypatch
) -> None:
    _common_stubs(monkeypatch)
    monkeypatch.setattr("invarlock.cli.device.resolve_device", lambda _d: "cpu")

    def _provider_factory(kind: str, *, emit=None, **kwargs):
        _ = kwargs
        assert kind == "wikitext2"
        assert emit is not None

        def _windows(**kw):
            _ = kw
            emit("DATA", "WikiText-2 validation: loading split...", "📚")
            emit("DATA", "Creating evaluation windows:", "📊")
            return (
                SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
            )

        return SimpleNamespace(windows=_windows)

    monkeypatch.setattr("invarlock.eval.data.get_provider", _provider_factory)

    cfg = _cfg(tmp_path, provider="wikitext2")
    r = CliRunner().invoke(
        cli, ["run", "-c", cfg, "--profile", "ci", "--style", "audit"]
    )
    assert r.exit_code == 0
    s = r.stdout
    assert "[DATA] WikiText-2 validation: loading split..." in s
    assert "[DATA] Creating evaluation windows:" in s
    assert "📚" not in s
    assert "📊" not in s
