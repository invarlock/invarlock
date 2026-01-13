import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from invarlock.cli.app import app as cli


def test_run_report_has_dataset_windows_stats(tmp_path: Path, monkeypatch):
    # Minimal environment
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
    monkeypatch.setattr("invarlock.cli.device.resolve_device", lambda d: "cpu")
    monkeypatch.setattr(
        "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
    )

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

    def _exec(**kwargs):
        return SimpleNamespace(
            edit={"deltas": {"params_changed": 0}},
            metrics={
                "window_overlap_fraction": 0.0,
                "window_match_fraction": 1.0,
                "loss_type": "ce",
            },
            guards={},
            context={"dataset_meta": {}},
            evaluation_windows={
                "preview": {"logloss": [3.0, 3.2], "token_counts": [10, 10]},
                "final": {"logloss": [3.1, 3.3], "token_counts": [10, 10]},
            },
            status="success",
        )

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: SimpleNamespace(execute=_exec)
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

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""
            model:
              adapter: hf_gpt2
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
              preview_n: 2
              final_n: 2

            guards:
              order: []

            eval:
              metric: {{ kind: ppl_causal, reps: 10 }}
              loss: {{ type: auto }}

            output: {{ dir: "{tmp_path}" }}
            """
        )
    )

    r = CliRunner().invoke(cli, ["run", "-c", str(cfg), "--profile", "dev"])
    # Allow generic fail if stubs; we only assert structure presence
    assert r.exit_code in (0, 1)

    rpt = next(tmp_path.rglob("report.json"))
    j = json.loads(rpt.read_text())
    stats = j.get("dataset", {}).get("windows", {}).get("stats", {})
    assert "coverage" in stats
    assert "window_match_fraction" in stats
    assert "window_overlap_fraction" in stats
    pm = j.get("metrics", {}).get("primary_metric") or {}
    assert "kind" in pm
