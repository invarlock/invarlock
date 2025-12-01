from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from invarlock import security
from invarlock.cli.commands.run import run_command


def test_run_command_happy_path(tmp_path: Path):
    # Minimal config via patch; avoid reading YAML
    class DummyCfg:
        def __init__(self, outdir):
            self.model = SimpleNamespace(id="gpt2", adapter="hf_gpt2", device="cpu")
            self.edit = SimpleNamespace(name="quant_rtn", plan={})
            self.auto = SimpleNamespace(
                enabled=False, tier="balanced", probes=0, target_pm_ratio=None
            )
            self.guards = SimpleNamespace(order=[])
            self.dataset = SimpleNamespace(
                provider=None,
                seq_len=8,
                stride=4,
                preview_n=1,
                final_n=1,
                seed=42,
                split="validation",
            )
            self.eval = SimpleNamespace(
                spike_threshold=2.0, loss=SimpleNamespace(type="ce")
            )
            self.output = SimpleNamespace(dir=outdir)

        def model_dump(self):
            return {
                "model": {
                    "id": self.model.id,
                    "adapter": self.model.adapter,
                    "device": self.model.device,
                },
                "edit": {"name": self.edit.name, "plan": self.edit.plan},
                "auto": {
                    "enabled": self.auto.enabled,
                    "tier": self.auto.tier,
                    "probes": self.auto.probes,
                    "target_pm_ratio": self.auto.target_pm_ratio,
                },
                "guards": {"order": []},
                "dataset": {
                    "provider": None,
                    "seq_len": 8,
                    "stride": 4,
                    "preview_n": 1,
                    "final_n": 1,
                    "seed": 42,
                    "split": "validation",
                },
                "eval": {"spike_threshold": 2.0, "loss": {"type": "ce"}},
                "output": {"dir": str(self.output.dir)},
            }

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
        def __init__(self):
            self.executions = 0

        def execute(self, **kwargs):
            self.executions += 1
            # Return a minimal report
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

    outdir = tmp_path / "runs"

    # Build minimal YAML config instead of patching load_config
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
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

    class StubProvider:
        def windows(self, **kwargs):
            prev = SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]])
            fin = SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]])
            return prev, fin

    # Ensure network guard is disabled for this isolated run (even though we stub all IO).
    security.enforce_network_policy(True)

    with (
        patch("invarlock.core.registry.get_registry", lambda: DummyRegistry()),
        patch("invarlock.core.runner.CoreRunner", lambda: DummyRunner()),
        patch(
            "invarlock.eval.data.get_provider", lambda *args, **kwargs: StubProvider()
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
            "invarlock.cli.commands.run.detect_model_profile",
            lambda model_id, adapter: SimpleNamespace(
                default_loss="ce",
                default_provider=None,
                default_metric=None,
                model_id=model_id,
                adapter=adapter,
                family="gpt2",
                module_selectors={},
                invariants=[],
                cert_lints=[],
            ),
        ),
        patch(
            "invarlock.cli.commands.run.resolve_tokenizer",
            lambda model_profile: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
                "tokhash123",
            ),
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
            profile=None,
            out=str(outdir),
            edit=None,
            tier=None,
            probes=0,
            until_pass=False,
        )
