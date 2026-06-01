from __future__ import annotations

import importlib.machinery
import sys
import types
from types import SimpleNamespace

from invarlock.cli.commands import doctor as doctor_mod


class DummyConsole:
    def __init__(self):
        self.lines: list[str] = []

    def print(self, *args, **kwargs):
        self.lines.append(" ".join(str(arg) for arg in args))


def _install_fake_torch(monkeypatch, *, cuda_available: bool) -> None:
    torch_mod = types.ModuleType("torch")
    torch_mod.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)

    class FakeProps:
        def __init__(self, total_memory=8 * 1e9, name="FakeGPU"):
            self.total_memory = total_memory
            self.device_name = name
            self.memory_total = f"{total_memory / 1e9:.1f} GB"

    class FakeCuda:
        def is_available(self):
            return cuda_available

        def device_count(self):
            return 1

        def get_device_properties(self, idx):
            return FakeProps()

    torch_mod.__version__ = "0.0.0"
    torch_mod.cuda = FakeCuda()
    torch_mod.version = SimpleNamespace(cuda=None)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)


def _patch_minimal_doctor_env(monkeypatch) -> None:
    fake_registry = SimpleNamespace(
        list_adapters=lambda: [],
        list_edits=lambda: [],
        list_guards=lambda: [],
        get_plugin_info=lambda name, kind: {
            "module": "invarlock.adapters",
            "entry_point": name,
        },
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.doctor.get_registry",
        lambda: fake_registry,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.core.registry.get_registry",
        lambda: fake_registry,
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.plugins._check_plugin_extras",
        lambda *args, **kwargs: "",
        raising=False,
    )
    monkeypatch.setattr(
        "invarlock.eval.data.list_providers",
        lambda: [],
        raising=False,
    )
    monkeypatch.setattr(
        doctor_mod.importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace(name=name),
        raising=False,
    )
    monkeypatch.setattr(
        doctor_mod,
        "get_device_info",
        lambda: {"auto_selected": "cpu", "cpu": {"available": True, "info": "ok"}},
        raising=False,
    )


def _mk_report(
    *,
    tokenizer=None,
    masking=None,
    split=None,
    pm_kind=None,
    counts_source=None,
    estimated=None,
) -> dict:
    prov: dict[str, object] = {}
    if tokenizer is not None or masking is not None:
        prov["provider_digest"] = {}
        if tokenizer is not None:
            prov["provider_digest"]["tokenizer_sha256"] = tokenizer
        if masking is not None:
            prov["provider_digest"]["masking_sha256"] = masking
    if split is not None:
        prov["dataset_split"] = split
    metrics = {}
    if pm_kind is not None:
        metrics = {"primary_metric": {"kind": pm_kind}}
        if counts_source is not None:
            metrics["primary_metric"]["counts_source"] = counts_source
        if estimated is not None:
            metrics["primary_metric"]["estimated"] = estimated
    return {"provenance": prov, "metrics": metrics}
