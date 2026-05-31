from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.cli.run._support_run_common import common_ce_patches, write_base_run_config


def snapshot_cfg(tmp_path: Path, preview: int = 2, final: int = 2) -> Path:
    return write_base_run_config(
        tmp_path,
        preview,
        final,
        edit_plan="{ heads: { mask_only: true, mask_auto: true, materialize: true } }",
        eval_fields="  spike_threshold: 2.0\n",
    )


def snapshot_common_ce():
    return common_ce_patches(include_profile=False, include_save_report=True)


class FakeTensor:
    def __init__(self, bytes_count: int):
        self._bytes = bytes_count

    def element_size(self):
        return 1

    def nelement(self):
        return self._bytes


class LargeModel:
    def named_parameters(self):
        return [("p", FakeTensor(500_000_000))]

    def named_buffers(self):
        return [("b", FakeTensor(100_000_000))]


class SmallModel:
    def named_parameters(self):
        return [("p", FakeTensor(1_000_000))]

    def named_buffers(self):
        return []


def provider_windows(
    preview_ids: list[list[int]] | None = None,
    final_ids: list[list[int]] | None = None,
    preview_masks: list[list[int]] | None = None,
    final_masks: list[list[int]] | None = None,
):
    preview_ids = preview_ids or [[1, 2]]
    final_ids = final_ids or [[3, 4]]
    preview_masks = preview_masks or [[1] * len(preview_ids[0])]
    final_masks = final_masks or [[1] * len(final_ids[0])]
    return SimpleNamespace(
        windows=lambda **kw: (
            SimpleNamespace(input_ids=preview_ids, attention_masks=preview_masks),
            SimpleNamespace(input_ids=final_ids, attention_masks=final_masks),
        )
    )


def provider_windows_patch(
    preview_ids: list[list[int]] | None = None,
    final_ids: list[list[int]] | None = None,
    preview_masks: list[list[int]] | None = None,
    final_masks: list[list[int]] | None = None,
):
    return patch(
        "invarlock.eval.data.get_provider",
        lambda *a, **k: provider_windows(
            preview_ids=preview_ids,
            final_ids=final_ids,
            preview_masks=preview_masks,
            final_masks=final_masks,
        ),
    )


def registry_with_adapter_patch(adapter):
    return patch(
        "invarlock.core.registry.get_registry",
        lambda: SimpleNamespace(
            get_adapter=lambda n: adapter,
            get_edit=lambda n: SimpleNamespace(name=n),
            get_guard=lambda n: SimpleNamespace(name=n),
            get_plugin_metadata=lambda n, t: {
                "name": n,
                "module": f"{t}.{n}",
                "version": "test",
            },
        ),
    )


def psutil_vm(available_mb: float):
    return SimpleNamespace(available=int(available_mb * 1024 * 1024))


def disk_usage(free_mb: float):
    DU = namedtuple("DU", ["total", "used", "free"])
    return DU(total=10 * 1024 * 1024 * 1024, used=0, free=int(free_mb * 1024 * 1024))
