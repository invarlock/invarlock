from __future__ import annotations

from types import SimpleNamespace

from invarlock.core import plugins_inventory as plugins_inventory_mod
from invarlock.core.plugins_inventory import (
    adapter_inventory_json_items,
    combined_plugins_json_items,
    dataset_inventory_json_items,
    detect_cuda_available,
    filter_inventory_rows,
    gather_adapter_inventory_rows,
    gather_generic_inventory_rows,
    is_minimal_plugins_view,
)


class _Registry:
    def __init__(self) -> None:
        self._adapters = {
            "hf_auto": {"module": "invarlock.adapters.hf", "entry_point": "auto"},
            "hf_bnb": {
                "module": "invarlock.plugins.bitsandbytes",
                "entry_point": "bnb",
            },
        }
        self._guards = {
            "shape_guard": {"module": "invarlock.guards.shape", "entry_point": "shape"},
            "remote_guard": {"module": "vendor.guard", "entry_point": "remote"},
        }
        self._edits = {
            "quant_rtn": {"module": "invarlock.edits.quant", "entry_point": "quant"}
        }

    def list_adapters(self) -> list[str]:
        return list(self._adapters)

    def list_guards(self) -> list[str]:
        return list(self._guards)

    def list_edits(self) -> list[str]:
        return list(self._edits)

    def get_plugin_info(self, name: str, kind: str) -> dict[str, str]:
        mapping = {
            "adapters": self._adapters,
            "guards": self._guards,
            "edits": self._edits,
        }
        return mapping[kind][name]


def test_is_minimal_plugins_view_and_cuda_detection() -> None:
    assert is_minimal_plugins_view("1") is True
    assert is_minimal_plugins_view("false") is False
    assert (
        detect_cuda_available(
            SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        )
        is True
    )
    assert detect_cuda_available(object()) is False
    assert (
        detect_cuda_available(
            SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
                )
            )
        )
        is False
    )


def test_safe_import_success_and_attribute_detection(monkeypatch) -> None:
    monkeypatch.setattr(
        plugins_inventory_mod.importlib,
        "import_module",
        lambda _name: SimpleNamespace(present=object()),
    )

    assert plugins_inventory_mod._safe_import("demo") is True
    assert plugins_inventory_mod._safe_import("demo", "present") is True
    assert plugins_inventory_mod._safe_import("demo", "missing") is False


def test_get_adapter_rows_keeps_bitsandbytes_ready_when_runtime_available(
    monkeypatch,
) -> None:
    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: _Registry())
    monkeypatch.setattr(
        plugins_inventory_mod,
        "bitsandbytes_runtime_available",
        lambda: True,
    )

    rows = plugins_inventory_mod.get_adapter_rows()
    bnb_row = next(row for row in rows if row["name"] == "hf_bnb")

    assert bnb_row["backend"] == "bitsandbytes"
    assert bnb_row["status"] == "ready"


def test_gather_adapter_inventory_rows_and_json_payloads() -> None:
    registry = _Registry()

    rows = gather_adapter_inventory_rows(
        registry=registry,
        minimal=False,
        has_cuda=False,
        is_linux=True,
        extras_checker=lambda name, _kind: (
            "⚠️ missing invarlock[gpu]" if name == "hf_bnb" else ""
        ),
        provenance_extractor=lambda name: (
            SimpleNamespace(library="bitsandbytes", version=None)
            if name == "hf_bnb"
            else SimpleNamespace(library="transformers", version="1.0")
        ),
        bitsandbytes_runtime_available=lambda: False,
    )

    assert [row["name"] for row in filter_inventory_rows(rows, "optional")] == [
        "hf_bnb"
    ]
    json_items = adapter_inventory_json_items(
        [{**row, "capability": {"kind": row["name"]}} for row in rows]
    )
    bnb_item = next(item for item in json_items if item["name"] == "hf_bnb")
    assert bnb_item["status"] == "needs_extra"
    assert bnb_item["backend"] == {"name": "bitsandbytes", "present": False}
    assert bnb_item["support_tier"] == "optional_backend_loader"
    assert bnb_item["deployment_claim"] is False


def test_gather_adapter_inventory_rows_marks_quanto_missing_extra() -> None:
    class _QuantoRegistry(_Registry):
        def __init__(self) -> None:
            super().__init__()
            self._adapters = {
                "hf_quanto": {
                    "module": "invarlock.plugins.quanto",
                    "entry_point": "quanto",
                }
            }

    rows = gather_adapter_inventory_rows(
        registry=_QuantoRegistry(),
        minimal=False,
        has_cuda=False,
        is_linux=True,
        extras_checker=lambda name, _kind: (
            "⚠️ missing invarlock[quanto]" if name == "hf_quanto" else ""
        ),
        provenance_extractor=lambda _name: SimpleNamespace(
            library="optimum-quanto",
            version=None,
        ),
        bitsandbytes_runtime_available=lambda: True,
    )

    item = adapter_inventory_json_items(rows)[0]
    assert item["name"] == "hf_quanto"
    assert item["status"] == "needs_extra"
    assert item["backend"] == {"name": "optimum-quanto", "present": False}


def test_gather_adapter_inventory_rows_marks_ct_missing_extra() -> None:
    class _CompressedTensorsRegistry(_Registry):
        def __init__(self) -> None:
            super().__init__()
            self._adapters = {
                "hf_ct": {
                    "module": "invarlock.plugins.ct",
                    "entry_point": "ct",
                }
            }

    rows = gather_adapter_inventory_rows(
        registry=_CompressedTensorsRegistry(),
        minimal=False,
        has_cuda=False,
        is_linux=True,
        extras_checker=lambda name, _kind: (
            "⚠️ missing invarlock[compressed-tensors]" if name == "hf_ct" else ""
        ),
        provenance_extractor=lambda _name: SimpleNamespace(
            library="compressed-tensors",
            version=None,
        ),
        bitsandbytes_runtime_available=lambda: True,
    )

    item = adapter_inventory_json_items(rows)[0]
    assert item["name"] == "hf_ct"
    assert item["status"] == "needs_extra"
    assert item["backend"] == {"name": "compressed-tensors", "present": False}


def test_gather_adapter_inventory_rows_marks_multimodal_missing_extra() -> None:
    class _MultimodalRegistry(_Registry):
        def __init__(self) -> None:
            super().__init__()
            self._adapters = {
                "hf_multimodal": {
                    "module": "invarlock.adapters.hf_multimodal",
                    "entry_point": "hf_multimodal",
                }
            }

    rows = gather_adapter_inventory_rows(
        registry=_MultimodalRegistry(),
        minimal=False,
        has_cuda=False,
        is_linux=True,
        extras_checker=lambda name, _kind: (
            "⚠️ missing invarlock[multimodal]" if name == "hf_multimodal" else ""
        ),
        provenance_extractor=lambda _name: SimpleNamespace(
            library="transformers",
            version="5.5.0",
        ),
        bitsandbytes_runtime_available=lambda: True,
    )

    row = rows[0]
    assert row["name"] == "hf_multimodal"
    assert row["support"] == "core"
    assert row["status"] == "needs_extra"
    assert row["enable"] == "pip install 'invarlock[multimodal]'"


def test_filter_inventory_rows_support_tier_modes() -> None:
    rows = [
        {"name": "spectral", "support_tier": "core_supported", "status": "ready"},
        {"name": "hello", "support_tier": "demo_only", "status": "ready"},
    ]

    assert filter_inventory_rows(rows, "demo_only") == [rows[1]]


def test_gather_generic_and_combined_inventory_payloads() -> None:
    registry = _Registry()
    guards = gather_generic_inventory_rows(
        registry=registry,
        plugin_type="guards",
        extras_checker=lambda _name, _kind: "",
    )
    edits = gather_generic_inventory_rows(
        registry=registry,
        plugin_type="edits",
        extras_checker=lambda _name, _kind: "",
    )

    combined = combined_plugins_json_items(
        adapter_rows=[],
        guard_rows=guards,
        edit_rows=edits,
    )

    assert any(item["kind"] == "guard" for item in combined)
    assert any(item["origin"] == "third_party" for item in combined)
    quant_item = next(item for item in combined if item["name"] == "quant_rtn")
    assert quant_item["support_tier"] == "validation_simulation"


def test_gather_adapter_inventory_rows_tolerates_probe_failures() -> None:
    registry = _Registry()

    rows = gather_adapter_inventory_rows(
        registry=registry,
        minimal=False,
        has_cuda=True,
        is_linux=True,
        extras_checker=lambda _name, _kind: (_ for _ in ()).throw(ValueError("bad")),
        provenance_extractor=lambda _name: (_ for _ in ()).throw(
            RuntimeError("missing provenance")
        ),
        bitsandbytes_runtime_available=lambda: True,
    )

    bnb_row = next(row for row in rows if row["name"] == "hf_bnb")
    assert bnb_row["backend"] == ""
    assert bnb_row["status"] == "needs_extra"
    assert bnb_row["enable"] == "pip install 'invarlock[gpu]'"


def test_gather_adapter_inventory_rows_handles_empty_missing_hint_and_cuda_runtime_gaps() -> (
    None
):
    class _RegistryWithOptional(_Registry):
        def __init__(self) -> None:
            super().__init__()
            self._adapters["custom_optional"] = {
                "module": "vendor.optional",
                "entry_point": "custom",
            }

    registry = _RegistryWithOptional()

    def _provenance(name: str) -> SimpleNamespace:
        if name == "hf_bnb":
            return SimpleNamespace(library="bitsandbytes", version="1.0")
        return SimpleNamespace(library="vendor-lib", version=None)

    rows = gather_adapter_inventory_rows(
        registry=registry,
        minimal=False,
        has_cuda=True,
        is_linux=True,
        extras_checker=lambda name, _kind: (
            "⚠️ missing" if name == "custom_optional" else ""
        ),
        provenance_extractor=_provenance,
        bitsandbytes_runtime_available=lambda: False,
    )

    custom_row = next(row for row in rows if row["name"] == "custom_optional")
    bnb_row = next(row for row in rows if row["name"] == "hf_bnb")

    assert custom_row["status"] == "needs_extra"
    assert custom_row["enable"] == ""
    assert bnb_row["status"] == "unsupported"
    assert bnb_row["enable"] == "bitsandbytes unavailable on this host"


def test_dataset_inventory_json_items_preserve_provider_module() -> None:
    items = dataset_inventory_json_items(
        ["wikitext2"],
        {"wikitext2": SimpleNamespace(__module__="invarlock.eval.data")},
    )

    assert items == [
        {
            "name": "wikitext2",
            "module": "invarlock.eval.data",
            "status": "available",
        }
    ]
