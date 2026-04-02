from __future__ import annotations

import importlib.metadata as importlib_metadata
from types import SimpleNamespace

import invarlock.core.doctor_inventory as mod
from invarlock.core.doctor_inventory import (
    build_adapter_inventory_rows,
    build_dataset_inventory_rows,
    build_generic_inventory_rows,
    summarize_inventory_rows,
)


class _FakeRegistry:
    def __init__(self, *, adapters: list[str], edits: list[str], guards: list[str]):
        self._adapters = adapters
        self._edits = edits
        self._guards = guards

    def list_adapters(self) -> list[str]:
        return list(self._adapters)

    def list_edits(self) -> list[str]:
        return list(self._edits)

    def list_guards(self) -> list[str]:
        return list(self._guards)

    def get_plugin_info(self, name: str, kind: str) -> dict[str, str]:
        if name == "hf_bnb":
            return {"module": "thirdparty.adapters.hf_bnb", "entry_point": name}
        return {"module": f"invarlock.{kind}.{name}", "entry_point": name}


def test_build_adapter_inventory_rows_classifies_optional_backend_states() -> None:
    registry = _FakeRegistry(adapters=["hf_causal", "hf_bnb"], edits=[], guards=[])

    def _find_spec(name: str) -> object | None:
        if name == "bitsandbytes":
            return None
        if name == "transformers":
            return SimpleNamespace(name=name)
        return None

    rows = build_adapter_inventory_rows(
        registry,
        has_cuda=False,
        is_linux=True,
        find_spec_safe=_find_spec,
        bitsandbytes_runtime_ready=False,
    )

    assert [row.name for row in rows] == ["hf_causal", "hf_bnb"]
    assert rows[0].origin == "core"
    assert rows[0].mode == "adapter"
    assert rows[0].backend == "transformers"
    assert rows[1].status == "needs_extra"
    assert rows[1].required_extra == "invarlock[gpu]"


def test_build_adapter_inventory_rows_marks_bitsandbytes_unsupported_without_runtime() -> (
    None
):
    registry = _FakeRegistry(adapters=["hf_bnb"], edits=[], guards=[])

    rows = build_adapter_inventory_rows(
        registry,
        has_cuda=True,
        is_linux=True,
        find_spec_safe=lambda name: SimpleNamespace(name=name),
        bitsandbytes_runtime_ready=False,
    )

    assert len(rows) == 1
    assert rows[0].status == "unsupported"
    assert rows[0].detail == "bitsandbytes unavailable on this host"


def test_build_adapter_inventory_rows_marks_auto_adapter_and_linux_only_quantizers(
    monkeypatch,
) -> None:
    registry = _FakeRegistry(adapters=["hf_auto", "hf_awq"], edits=[], guards=[])

    monkeypatch.setattr(
        mod.importlib_metadata,
        "version",
        lambda package_name: "4.0.0" if package_name == "transformers" else "0.0.0",
    )

    rows = build_adapter_inventory_rows(
        registry,
        has_cuda=False,
        is_linux=False,
        find_spec_safe=lambda name: SimpleNamespace(name=name),
        bitsandbytes_runtime_ready=True,
    )

    assert rows[0].mode == "auto-matcher"
    assert rows[0].origin == "core"
    assert rows[0].version == "4.0.0"
    assert rows[1].status == "unsupported"
    assert rows[1].detail == "Linux-only"


def test_package_version_tolerates_missing_and_generic_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        mod.importlib_metadata,
        "version",
        lambda package_name: (_ for _ in ()).throw(
            importlib_metadata.PackageNotFoundError(package_name)
        ),
    )
    assert mod._package_version("transformers") is None

    monkeypatch.setattr(
        mod.importlib_metadata,
        "version",
        lambda package_name: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert mod._package_version("transformers") is None


def test_build_generic_inventory_rows_uses_plugin_extra_hints() -> None:
    registry = _FakeRegistry(adapters=[], edits=["quant_rtn"], guards=["spectral"])

    rows = build_generic_inventory_rows(
        registry,
        kind="edits",
        check_plugin_extras=lambda name, kind: "⚠️ missing invarlock[quant]"
        if name == "quant_rtn"
        else "",
    )

    assert len(rows) == 1
    assert rows[0].name == "quant_rtn"
    assert rows[0].status == "needs_extra"
    assert rows[0].required_extra == "invarlock[quant]"


def test_build_generic_inventory_rows_tolerates_extra_lookup_errors() -> None:
    registry = _FakeRegistry(adapters=[], edits=[], guards=["spectral"])

    rows = build_generic_inventory_rows(
        registry,
        kind="guards",
        check_plugin_extras=lambda name, kind: (_ for _ in ()).throw(
            RuntimeError("boom")
        ),
    )

    assert len(rows) == 1
    assert rows[0].status == "ready"
    assert rows[0].mode == "guard"


def test_dataset_inventory_and_summary_rows_are_deterministic() -> None:
    rows = build_dataset_inventory_rows(
        ["synthetic", "wikitext2"],
        provider_network={"synthetic": "no", "wikitext2": "cache"},
        provider_params={"synthetic": "-", "wikitext2": "split,seq_len"},
    )
    summary = summarize_inventory_rows(
        build_generic_inventory_rows(
            _FakeRegistry(adapters=[], edits=["noop"], guards=[]),
            kind="edits",
            check_plugin_extras=lambda *_args, **_kwargs: "",
        )
    )

    assert rows[0].network_mode == "no"
    assert rows[1].network_mode == "cache"
    assert rows[0].available is True
    assert rows[1].params == "split,seq_len"
    assert summary == {
        "total": 1,
        "ready": 1,
        "needs_extra": 0,
        "unsupported": 0,
        "auto": 0,
    }


def test_dataset_inventory_rows_handles_unknown_network_labels() -> None:
    rows = build_dataset_inventory_rows(
        ["custom"],
        provider_network={"custom": "sometimes"},
        provider_params={},
    )

    assert rows == [
        mod.DoctorDatasetRow(
            provider="custom",
            network_mode="unknown",
            available=True,
            params="-",
        )
    ]
