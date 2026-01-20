from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from invarlock.cli.commands.run import run_command


def test_run_command_release_profile_with_capacity(tmp_path: Path):
    # Minimal config YAML
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  adapter: hf_causal
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
  preview_n: 2
  final_n: 2

guards:
  order: []

eval:
  loss:
    type: auto

output:
  dir: runs
        """
    )

    # Registry and runner stubs
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
                metrics={
                    "ppl_preview": 10.0,
                    "ppl_final": 10.0,
                    "ppl_ratio": 1.0,
                    "window_overlap_fraction": 0.0,
                    "window_match_fraction": 1.0,
                    "loss_type": "ce",
                },
                guards={},
                context={"dataset_meta": {}},
                evaluation_windows={},
                status="success",
            )

    # Data provider with capacity estimator and windows
    class Provider:
        def estimate_capacity(self, **kwargs):
            # Provide enough unique tokens to satisfy buffer/reserve math
            return {
                "available_unique": 2000,
                "available_nonoverlap": 2000,
                "total_tokens": 1000000,
                "dedupe_rate": 0.1,
            }

        def windows(self, **kwargs):
            n_prev = kwargs.get("preview_n", 2)
            n_final = kwargs.get("final_n", 2)
            prev_ids = [[i, i + 1, i + 2, i + 3] for i in range(0, n_prev * 4, 4)]
            fin_ids = [
                [1000 + i, 1001 + i, 1002 + i, 1003 + i]
                for i in range(0, n_final * 4, 4)
            ]
            prev = SimpleNamespace(
                input_ids=prev_ids, attention_masks=[[1] * 4] * n_prev
            )
            fin = SimpleNamespace(
                input_ids=fin_ids, attention_masks=[[1] * 4] * n_final
            )
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
        patch(
            "invarlock.reporting.report.save_report",
            lambda report, run_dir, formats, filename_prefix: {
                "json": str(run_dir / (filename_prefix + ".json"))
            },
        ),
        patch(
            "invarlock.cli.commands.run.validate_guard_overhead",
            lambda *args, **kwargs: SimpleNamespace(
                passed=True,
                overhead_ratio=0.0,
                overhead_percent=0.0,
                threshold=0.01,
                errors=[],
            ),
        ),
    ):
        run_command(
            config=str(config_path),
            device="cpu",
            profile="release",
            out=str(outdir),
            edit=None,
            tier=None,
            probes=0,
            until_pass=False,
        )
