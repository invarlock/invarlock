from __future__ import annotations

from types import SimpleNamespace

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


def test_gather_adapter_inventory_rows_and_json_payloads() -> None:
    registry = _Registry()

    rows = gather_adapter_inventory_rows(
        registry=registry,
        minimal=False,
        has_cuda=False,
        is_linux=True,
        extras_checker=lambda name, _kind: "⚠️ missing invarlock[gpu]"
        if name == "hf_bnb"
        else "",
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
