from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SNS = SimpleNamespace


def write_base_run_config(
    tmp_path: Path,
    preview: int = 1,
    final: int = 1,
    *,
    edit_name: str = "quant_rtn",
    edit_plan: str = "{}",
    eval_fields: str = "",
    loss_type: str = "auto",
) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        f"""
model:
  adapter: hf_causal
  id: gpt2
  device: cpu
edit:
  name: {edit_name}
  plan: {edit_plan}

dataset:
  provider: synthetic
  id: synthetic
  split: validation
  seq_len: 8
  stride: 4
  preview_n: {preview}
  final_n: {final}

guards:
  order: []

eval:
{eval_fields}  loss:
    type: {loss_type}

output:
  dir: runs
        """
    )
    return p


def common_ce_patches(
    *,
    include_profile: bool = True,
    include_registry: bool = False,
    include_save_report: bool = False,
    tokenizer_name_or_path: bool = False,
    tokenizer_vocab_size: int = 50000,
):
    tokenizer_fields: dict[str, object] = {
        "eos_token": "</s>",
        "pad_token": "</s>",
        "vocab_size": tokenizer_vocab_size,
    }
    if tokenizer_name_or_path:
        tokenizer_fields["name_or_path"] = "tok"

    patches = []
    if include_profile:
        patches.append(
            patch(
                "invarlock.cli.run_runtime_exec.detect_model_profile",
                lambda model_id=None, adapter=None: SimpleNamespace(
                    default_loss="ce",
                    model_id=model_id,
                    adapter=adapter,
                    module_selectors={},
                    invariants=set(),
                    cert_lints=[],
                    family="gpt",
                ),
            )
        )
    patches.extend(
        (
            patch(
                "invarlock.cli.run_runtime_exec.resolve_tokenizer",
                lambda *_a, **_k: (SimpleNamespace(**tokenizer_fields), "tokhash123"),
            ),
            patch("invarlock.cli.device.resolve_device", lambda d: d),
            patch(
                "invarlock.cli.device.validate_device_for_config",
                lambda d: (True, ""),
            ),
        )
    )
    if include_save_report:
        patches.append(
            patch(
                "invarlock.reporting.report_files.save_report",
                lambda report, run_dir, formats, filename_prefix: {
                    "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
                },
            )
        )
    if include_registry:
        patches.append(
            patch(
                "invarlock.core.registry.get_registry",
                lambda: SimpleNamespace(
                    get_adapter=lambda name: SimpleNamespace(
                        name=name,
                        load_model=lambda model_id, device=None: object(),
                    ),
                    get_edit=lambda name: SimpleNamespace(name=name),
                    get_guard=lambda name: SimpleNamespace(name=name),
                    get_plugin_metadata=lambda n, t: {
                        "name": n,
                        "module": f"{t}.{n}",
                        "version": "test",
                    },
                ),
            )
        )
    return tuple(patches)


def offline_registry():
    class _Registry:
        def get_adapter(self, name):
            return SimpleNamespace(
                name=name,
                load_model=lambda model_id, device=None: SimpleNamespace(
                    named_parameters=lambda: [], named_buffers=lambda: []
                ),
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            return SimpleNamespace(name=name)

        def get_plugin_metadata(self, name, kind):
            return {"name": name, "module": f"{kind}.{name}", "version": "test"}

    return _Registry()


def offline_registry_patch():
    return patch("invarlock.core.registry.get_registry", offline_registry)


def common_ce_detect_ce_patches():
    return (
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        offline_registry_patch(),
        patch(
            "invarlock.cli.run_runtime_exec.detect_model_profile",
            lambda model_id=None, adapter=None: SimpleNamespace(
                default_loss="ce",
                model_id=model_id,
                adapter=adapter,
                module_selectors={},
                invariants=set(),
                cert_lints=[],
                family="gpt",
            ),
        ),
        patch(
            "invarlock.cli.run_runtime_exec.resolve_tokenizer",
            lambda profile: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
                "tokhash123",
            ),
        ),
        patch(
            "invarlock.reporting.report_files.save_report",
            lambda report, run_dir, formats, filename_prefix: {
                "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
            },
        ),
    )


def synthetic_provider_min():
    return SimpleNamespace(
        windows=lambda **kw: (
            SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
            SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
        )
    )


def runner_success():
    return SimpleNamespace(
        execute=lambda **k: SimpleNamespace(
            edit={},
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context={"dataset_meta": {}},
            status="success",
        )
    )


def assert_single_run_report_artifact(
    tmp_path: Path, *, profile: str | None = None
) -> dict[str, object]:
    runs_dir = tmp_path / "runs"
    run_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    assert len(run_dirs) == 1

    run_dir = run_dirs[0]
    report_path = run_dir / "report.json"
    manifest_path = run_dir / "runtime.manifest.json"
    assert report_path.is_file()
    assert manifest_path.is_file()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["meta"]["adapter"] == "hf_causal"
    assert payload["metrics"]["primary_metric"]["kind"] == "ppl_causal"
    if profile is not None:
        assert payload["context"]["profile"] == profile
    return payload
