from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from invarlock.cli.commands.run import run_command


def test_run_command_metrics_loss_type_fallback_set(tmp_path: Path):
    # Config
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
model:
  adapter: hf_gpt2
  id: gpt2
  device: cpu
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
  loss:
    type: auto

output:
  dir: runs
        """
    )

    # Capture the report passed to save_report
    captured = {}

    def capture_save_report(report, run_dir, formats, filename_prefix):  # noqa: D401
        captured["report"] = report
        return {"json": str(run_dir / (filename_prefix + ".json"))}

    class DummyRegistry:
        def get_adapter(self, name):
            return SimpleNamespace(
                name=name, load_model=lambda model_id, device: object()
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    class DummyRunner:
        def execute(self, **kwargs):
            # No loss_type in metrics to trigger fallback
            return SimpleNamespace(
                edit={
                    "plan_digest": "abcd",
                    "deltas": {
                        "params_changed": 0,
                        "heads_pruned": 0,
                        "neurons_pruned": 0,
                        "layers_modified": 0,
                    },
                },
                metrics={"ppl_preview": 10.0, "ppl_final": 10.0, "ppl_ratio": 1.0},
                guards={},
                context={"dataset_meta": {}},
                evaluation_windows={},
                status="success",
            )

    class Provider:
        def windows(self, **kwargs):
            prev = SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]])
            fin = SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]])
            return prev, fin

    outdir = tmp_path / "runs"
    with (
        patch("invarlock.core.registry.get_registry", lambda: DummyRegistry()),
        patch("invarlock.core.runner.CoreRunner", lambda: DummyRunner()),
        patch("invarlock.eval.data.get_provider", lambda *args, **kwargs: Provider()),
        patch(
            "invarlock.cli.commands.run.detect_model_profile",
            lambda model_id=None, adapter=None: SimpleNamespace(
                default_loss="ce",
                model_id=model_id,
                adapter=adapter,
                module_selectors={},
                invariants=[],
                cert_lints=[],
                family="gpt2",
                make_tokenizer=lambda: (
                    SimpleNamespace(
                        eos_token="</s>", pad_token="</s>", vocab_size=50000
                    ),
                    "tokhash123",
                ),
            ),
        ),
        patch(
            "invarlock.cli.commands.run.resolve_tokenizer",
            lambda model_profile: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
                "tokhash123",
            ),
        ),
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch("invarlock.reporting.report.save_report", capture_save_report),
    ):
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(outdir),
            edit=None,
            tier=None,
            probes=0,
            until_pass=False,
        )

    assert captured.get("report") is not None
    lt = captured["report"]["metrics"].get("loss_type")
    assert lt in {"ce", "causal", "mlm"}
