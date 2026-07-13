from __future__ import annotations

import hashlib
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.cli.run._support_run_common import (
    measured_guard_metric_impact_result,
    write_base_run_config,
)


def baseline_pairing_write_base_cfg(
    tmp_path: Path, preview_n: int = 2, final_n: int = 2
) -> Path:
    return write_base_run_config(tmp_path, preview_n, final_n)


def baseline_pairing_common_patches_ce():
    return (
        patch(
            "invarlock.core.registry.get_registry",
            lambda: SimpleNamespace(
                get_adapter=lambda name: SimpleNamespace(
                    name=name, load_model=lambda model_id, device: object()
                ),
                get_edit=lambda name: SimpleNamespace(name=name),
                get_guard=lambda name: (_ for _ in ()).throw(KeyError("no guard")),
                get_plugin_metadata=lambda n, t: {
                    "name": n,
                    "module": f"{t}.{n}",
                    "version": "test",
                },
            ),
        ),
        patch(
            "invarlock.cli.run_runtime_exec.detect_model_profile",
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
            "invarlock.cli.run_runtime_exec.resolve_tokenizer",
            lambda model_profile: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
                "tokhash123",
            ),
        ),
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch(
            "invarlock.cli.run_runtime_exec.validate_guard_metric_impact",
            lambda *_args, **_kwargs: measured_guard_metric_impact_result(),
        ),
        patch(
            "invarlock.reporting.report_bundle.save_report",
            lambda report, run_dir, formats, filename_prefix: {
                "json": str(run_dir / (filename_prefix + ".json"))
            },
        ),
    )


def baseline_pairing_compute_seq_hash(seqs: list[list[int]]) -> str:
    h = hashlib.blake2s(digest_size=16)
    for seq in seqs:
        h.update(len(seq).to_bytes(4, "little", signed=False))
        arr = array("I", (int(tok) & 0xFFFFFFFF for tok in seq))
        h.update(arr.tobytes())
    return h.hexdigest()


def supplemental_cfg(tmp_path: Path, preview: int = 1, final: int = 1) -> Path:
    return write_base_run_config(tmp_path, preview, final)


def supplemental_common_patches_detect_ce():
    return (
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch(
            "invarlock.reporting.report_bundle.save_report",
            lambda report, run_dir, formats, filename_prefix=None: {
                "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
            },
        ),
        patch(
            "invarlock.cli.run_runtime_exec.detect_model_profile",
            lambda model_id=None, adapter=None: SimpleNamespace(
                default_loss="ce",
                model_id=model_id,
                adapter=adapter,
                module_selectors={},
                invariants=set(),
                cert_lints=[],
                family="gpt2",
            ),
        ),
    )


def supplemental_provider_min():
    return SimpleNamespace(
        windows=lambda **kw: (
            SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
            SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
        )
    )
