from __future__ import annotations

import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

SNS = SimpleNamespace


def configure_guard_metric_impact_skip(path: Path) -> Path:
    """Declare metric-impact measurement out of scope for an unrelated CLI test."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    context = payload.setdefault("context", {})
    assert isinstance(context, dict)
    run = context.setdefault("run", {})
    assert isinstance(run, dict)
    run["skip_guard_metric_impact_check"] = True
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def canonical_ppl_metrics(
    *, preview: float = 1.0, final: float = 1.0, **extra: object
) -> dict[str, object]:
    """Build finite CoreRunner PPL metrics for unrelated CLI success fixtures."""
    ratio = final / preview
    return {
        "ppl_preview": preview,
        "ppl_final": final,
        "ppl_ratio": ratio,
        "logloss_final": math.log(final),
        "final_total_tokens": 1,
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": preview,
            "final": final,
            "invalid": False,
            "degraded": False,
            "degraded_reason": None,
        },
        **extra,
    }


def measured_guard_metric_impact_result(
    *, degradation: float = 0.0, degradation_limit: float = 0.01
) -> SimpleNamespace:
    """Build canonical successful validator evidence for unrelated CLI tests."""
    bare_value = 1.0
    guarded_value = 1.0 + degradation
    display_value = degradation * 100.0
    return SimpleNamespace(
        passed=True,
        messages=[],
        warnings=[],
        errors=[],
        diagnostics=[],
        checks={
            "metric_kind_matches": True,
            "measurements_valid": True,
            "guard_metric_impact": True,
        },
        metrics={
            "metric_kind": "ppl_causal",
            "direction": "lower",
            "bare_value": bare_value,
            "guarded_value": guarded_value,
            "degradation_basis": "relative_increase",
            "degradation": degradation,
            "display_value": display_value,
            "display_unit": "percent",
        },
    )


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
            patch(
                "invarlock.cli.run_runtime_exec.validate_guard_metric_impact",
                lambda *_args, **_kwargs: measured_guard_metric_impact_result(),
            ),
        )
    )
    if include_save_report:
        patches.append(
            patch(
                "invarlock.reporting.report_bundle.save_report",
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
            "invarlock.reporting.report_bundle.save_report",
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
            metrics=canonical_ppl_metrics(),
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


def assert_single_run_output_artifacts(
    tmp_path: Path, *, expected_count: int | None = 1
) -> Path:
    runs_dir = tmp_path / "runs"
    run_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    if expected_count is None:
        assert run_dirs
    else:
        assert len(run_dirs) == expected_count

    for run_dir in run_dirs:
        assert re.fullmatch(r"\d{8}_\d{6}", run_dir.name)

        events_path = run_dir / "events.jsonl"
        if events_path.is_file():
            assert events_path.read_text(encoding="utf-8").strip()

        report_path = run_dir / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            assert report["meta"]["adapter"] == "hf_causal"

        manifest_path = run_dir / "runtime.manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert manifest
            context = manifest.get("context")
            if isinstance(context, dict) and "command" in context:
                assert context["command"]
            if "command" in manifest:
                assert manifest["command"]
            execution_mode = manifest.get("execution_mode")
            if execution_mode is None and isinstance(manifest.get("environment"), dict):
                execution_mode = manifest["environment"].get("execution_mode")
            if execution_mode is not None:
                assert execution_mode in {"container", "host-bypass"}
    return run_dirs[-1]
