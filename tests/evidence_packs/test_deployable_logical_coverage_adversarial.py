from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import invarlock.evidence_pack_contracts.deployable_coverage as coverage_mod
from invarlock.evidence_pack_contracts.deployable_coverage import (
    DenseParameterCatalog,
    canonical_names_sha256,
    dense_parameter_catalog,
    inspect_bitsandbytes_modules,
    logical_coverage_from_inventory,
    require_inventory_logical_binding,
    require_inventory_runtime_facts,
    require_logical_coverage,
)
from invarlock.evidence_pack_deployable_validation import (
    _deployable_sidecar_consistency_errors,
)
from scripts.evidence_packs.python.editing.validate_deployable import (
    _deployable_metadata_issues,
    _deployable_sidecar_issues,
)


class _AllowedWeight(torch.nn.Parameter):
    pass


class _AllowedLinear8bitLt:
    weight: object


class _SpoofedLinear8bitLt:
    __module__ = "bitsandbytes.nn.evil"
    weight: object


class _SpoofedInt8Params:
    __module__ = "bitsandbytes.nn.evil"

    def numel(self) -> int:
        return 6


def _authenticate_test_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        coverage_mod,
        "_bitsandbytes_type_contract",
        lambda bits: (_AllowedLinear8bitLt, _AllowedWeight),
    )


def _model_with_modules(entries: list[tuple[str, object]]) -> object:
    return type(
        "PackedModel",
        (),
        {"named_modules": lambda self: [("", self), *entries]},
    )()


def _logical_payload() -> dict[str, object]:
    names = ["layer.weight"]
    return {
        "basis": "dense_baseline_unique_parameters",
        "weight_tensor_names": names,
        "weight_tensor_names_sha256": canonical_names_sha256(names),
        "weight_tensor_count": 1,
        "parameter_elements": 12,
        "total_unique_parameter_elements": 15,
    }


def test_spoofed_bitsandbytes_fqcn_does_not_authenticate_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _authenticate_test_types(monkeypatch)
    module = _SpoofedLinear8bitLt()
    module.weight = _SpoofedInt8Params()

    try:
        inspect_bitsandbytes_modules(_model_with_modules([("layer", module)]), bits=8)
    except RuntimeError as exc:
        assert "no bitsandbytes packed linear modules" in str(exc)
    else:  # pragma: no cover - explicit adversarial failure signal
        raise AssertionError("spoofed bitsandbytes class identity was accepted")


def test_distinct_dense_parameter_wrappers_sharing_storage_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _authenticate_test_types(monkeypatch)
    dense = torch.nn.Module()
    dense.layer = torch.nn.Linear(4, 3, bias=False)
    dense.alias = torch.nn.Linear(4, 3, bias=False)
    dense.alias.weight = torch.nn.Parameter(dense.layer.weight.data)
    assert dense.alias.weight is not dense.layer.weight
    assert (
        dense.alias.weight.untyped_storage().data_ptr()
        == dense.layer.weight.untyped_storage().data_ptr()
    )

    packed = _AllowedLinear8bitLt()
    packed.weight = _AllowedWeight(torch.randn(3, 4), requires_grad=False)
    try:
        logical_coverage_from_inventory(
            dense_parameter_catalog(dense),
            inspect_bitsandbytes_modules(
                _model_with_modules([("layer", packed)]), bits=8
            ),
        )
    except RuntimeError as exc:
        assert "tied or ambiguous" in str(exc)
    else:  # pragma: no cover - explicit adversarial failure signal
        raise AssertionError("shared dense storage was counted twice")


def test_distinct_packed_weight_wrappers_sharing_storage_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _authenticate_test_types(monkeypatch)
    first_weight = _AllowedWeight(torch.randn(3, 4), requires_grad=False)
    second_weight = _AllowedWeight(first_weight.data, requires_grad=False)
    first = _AllowedLinear8bitLt()
    second = _AllowedLinear8bitLt()
    first.weight = first_weight
    second.weight = second_weight

    try:
        inspect_bitsandbytes_modules(
            _model_with_modules([("first", first), ("second", second)]), bits=8
        )
    except RuntimeError as exc:
        assert "share one storage weight" in str(exc)
    else:  # pragma: no cover - explicit adversarial failure signal
        raise AssertionError("shared packed storage was counted twice")


def test_bias_is_excluded_from_edited_names_but_included_in_dense_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _authenticate_test_types(monkeypatch)
    dense = torch.nn.Module()
    dense.layer = torch.nn.Linear(4, 3, bias=True)
    packed = _AllowedLinear8bitLt()
    packed.weight = _AllowedWeight(torch.randn(3, 4), requires_grad=False)

    logical = logical_coverage_from_inventory(
        dense_parameter_catalog(dense),
        inspect_bitsandbytes_modules(_model_with_modules([("layer", packed)]), bits=8),
    )

    assert logical["basis"] == "dense_baseline_unique_parameters"
    assert logical["weight_tensor_names"] == ["layer.weight"]
    assert logical["parameter_elements"] == 12
    assert logical["total_unique_parameter_elements"] == 15


def test_false_mapping_and_sidecar_runtime_mapping_drift_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _authenticate_test_types(monkeypatch)
    dense = torch.nn.Module()
    dense.layer = torch.nn.Linear(4, 3, bias=True)
    packed = _AllowedLinear8bitLt()
    packed.weight = _AllowedWeight(torch.randn(3, 4), requires_grad=False)
    inventory = inspect_bitsandbytes_modules(
        _model_with_modules([("missing", packed)]), bits=8
    )
    try:
        logical_coverage_from_inventory(dense_parameter_catalog(dense), inventory)
    except RuntimeError as exc:
        assert "no dense baseline weight" in str(exc)
    else:  # pragma: no cover - explicit adversarial failure signal
        raise AssertionError("false module-to-dense mapping was accepted")

    runtime = {
        "quantized_module_count": 1,
        "quantized_module_names": ["other"],
        "quantized_module_names_sha256": canonical_names_sha256(["other"]),
        "quantized_module_types": ["authenticated.test.Linear8bitLt"],
        "packed_weight_storage_elements": 12,
    }
    try:
        require_inventory_logical_binding(runtime, _logical_payload())
    except ValueError as exc:
        assert "do not match packed module names" in str(exc)
    else:  # pragma: no cover - explicit adversarial failure signal
        raise AssertionError("sidecar/runtime logical mapping drift was accepted")


def test_metadata_and_memory_ratios_are_recomputed_canonically() -> None:
    logical = _logical_payload()
    metadata = {
        "artifact_class": "deployable_optimized_subject",
        "optimized_deployment_backend": True,
        "packed_quantized_storage": True,
        "backend": "bitsandbytes",
        "logical_coverage": logical,
        "coverage": {
            "edited_tensors": 1,
            "edited_params": 12,
            "total_params": 15,
            "coverage_ratio": 0.9,
        },
    }
    assert "edit_metadata.coverage does not bind logical coverage" in (
        _deployable_metadata_issues(metadata, "bitsandbytes")
    )

    for invalid_ratio in (0.5, float("nan"), float("inf"), True):
        memory = {
            "schema": "invarlock/deployable-memory-report-v1",
            "ok": True,
            "baseline_reported_bytes": 100,
            "quantized_reported_bytes": 60,
            "reduction_bytes": 40,
            "reduction_ratio": invalid_ratio,
            "runtime_memory_reduction_observed": True,
        }
        assert any(
            "reduction_ratio must equal" in issue
            for issue in _deployable_sidecar_issues(
                "memory_report.json", memory, backend="bitsandbytes"
            )
        )
        assert any(
            "reduction_ratio does not match" in issue
            for issue in _deployable_sidecar_consistency_errors(
                scenario_id="adversarial",
                sidecar="memory_report.json",
                payload=memory,
            )
        )


def test_storage_and_backend_contracts_fail_closed_on_unusable_runtime_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Storage:
        def __init__(self, *, handle: int, pointer: int = 0, size: int = 0) -> None:
            self._cdata = handle
            self._pointer = pointer
            self._size = size

        def data_ptr(self) -> int:
            return self._pointer

        def nbytes(self) -> int:
            return self._size

    class Value:
        device = "cuda:0"

        def __init__(self, storage: Storage | None, *, fail: bool = False) -> None:
            self.storage = storage
            self.fail = fail

        def untyped_storage(self) -> Storage:
            if self.fail:
                raise TypeError("unavailable")
            assert self.storage is not None
            return self.storage

    assert coverage_mod._storage_identity(Value(Storage(handle=7))) == (
        "torch_storage",
        7,
    )
    assert coverage_mod._storage_identity(
        Value(Storage(handle=0, pointer=11, size=12))
    ) == ("torch_storage_address", "cuda:0", 11, 12)
    assert coverage_mod._storage_identity(Value(None, fail=True))[0] == "python_object"

    def missing_backend(_name: str) -> object:
        raise ImportError("not installed")

    monkeypatch.setattr(coverage_mod.importlib, "import_module", missing_backend)
    with pytest.raises(RuntimeError, match="bitsandbytes is required"):
        coverage_mod._bitsandbytes_type_contract(8)

    class Linear4:
        pass

    class Params4:
        pass

    class Linear8:
        pass

    class Params8:
        pass

    monkeypatch.setattr(
        coverage_mod.importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            Linear4bit=Linear4,
            Params4bit=Params4,
            Linear8bitLt=Linear8,
            Int8Params=Params8,
        ),
    )
    assert coverage_mod._bitsandbytes_type_contract(4) == (Linear4, Params4)
    assert coverage_mod._bitsandbytes_type_contract(8) == (Linear8, Params8)
    with pytest.raises(ValueError, match="bits must be"):
        coverage_mod._bitsandbytes_type_contract(16)
    monkeypatch.setattr(
        coverage_mod.importlib,
        "import_module",
        lambda _name: SimpleNamespace(Linear4bit=object(), Params4bit=Params4),
    )
    with pytest.raises(RuntimeError, match="type contract is unavailable"):
        coverage_mod._bitsandbytes_type_contract(4)


def test_dense_and_packed_inventory_contracts_reject_ambiguous_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Parameter:
        def __init__(self, count: int, marker: int) -> None:
            self.count = count
            self.marker = marker

        def numel(self) -> int:
            return self.count

        def untyped_storage(self) -> object:
            return SimpleNamespace(_cdata=self.marker)

    class Model:
        def __init__(self, entries: list[tuple[object, Parameter]]) -> None:
            self.entries = entries

        def named_parameters(
            self, *, remove_duplicate: bool
        ) -> list[tuple[object, Parameter]]:
            assert remove_duplicate is False
            return self.entries

    with pytest.raises(RuntimeError, match="does not expose"):
        dense_parameter_catalog(object())

    class LegacyModel:
        def named_parameters(self) -> list[object]:
            return []

    with pytest.raises(RuntimeError, match="cannot expose aliases"):
        dense_parameter_catalog(LegacyModel())
    with pytest.raises(RuntimeError, match="has no named"):
        dense_parameter_catalog(Model([]))
    with pytest.raises(RuntimeError, match="names are not canonical"):
        dense_parameter_catalog(Model([("", Parameter(1, 1))]))
    with pytest.raises(RuntimeError, match="parameter is empty"):
        dense_parameter_catalog(Model([("weight", Parameter(0, 1))]))
    with pytest.raises(RuntimeError, match="disagree on logical size"):
        dense_parameter_catalog(
            Model([("first", Parameter(1, 1)), ("second", Parameter(2, 1))])
        )

    _authenticate_test_types(monkeypatch)
    no_weight = _AllowedLinear8bitLt()
    with pytest.raises(RuntimeError, match="no canonical name"):
        inspect_bitsandbytes_modules(_model_with_modules([(None, no_weight)]), bits=8)
    with pytest.raises(RuntimeError, match="no packed backend weight"):
        inspect_bitsandbytes_modules(
            _model_with_modules([("layer", no_weight)]), bits=8
        )
    first = _AllowedLinear8bitLt()
    first.weight = _AllowedWeight(torch.ones(1), requires_grad=False)
    second = _AllowedLinear8bitLt()
    second.weight = _AllowedWeight(torch.ones(1), requires_grad=False)
    with pytest.raises(RuntimeError, match="names are ambiguous"):
        inspect_bitsandbytes_modules(
            _model_with_modules([("layer", first), ("layer", second)]), bits=8
        )
    empty = _AllowedLinear8bitLt()
    empty.weight = _AllowedWeight(torch.empty(0), requires_grad=False)
    with pytest.raises(RuntimeError, match="storage element count is not positive"):
        inspect_bitsandbytes_modules(_model_with_modules([("layer", empty)]), bits=8)


def test_logical_and_runtime_coverage_payloads_reject_each_invalid_binding() -> None:
    valid_logical = _logical_payload()
    valid_runtime = {
        "quantized_module_names": ["layer"],
        "quantized_module_names_sha256": canonical_names_sha256(["layer"]),
        "quantized_module_count": 1,
        "packed_weight_storage_elements": 12,
        "quantized_module_types": ["authenticated.test.Linear8bitLt"],
    }
    assert require_logical_coverage(valid_logical) == valid_logical
    assert require_inventory_runtime_facts(valid_runtime) == valid_runtime

    malformed_logical = [
        {},
        {**valid_logical, "weight_tensor_names": ["layer"]},
        {**valid_logical, "weight_tensor_names_sha256": "sha256:" + "0" * 64},
        {**valid_logical, "weight_tensor_count": True},
        {**valid_logical, "weight_tensor_count": 2},
        {**valid_logical, "parameter_elements": 16},
    ]
    for payload in malformed_logical:
        with pytest.raises(ValueError):
            require_logical_coverage(payload)

    malformed_runtime = [
        None,
        {**valid_runtime, "quantized_module_names": []},
        {**valid_runtime, "quantized_module_names_sha256": "sha256:" + "0" * 64},
        {**valid_runtime, "quantized_module_count": True},
        {**valid_runtime, "packed_weight_storage_elements": 0},
        {**valid_runtime, "quantized_module_types": []},
    ]
    for payload in malformed_runtime:
        with pytest.raises(ValueError):
            require_inventory_runtime_facts(payload)

    catalog = DenseParameterCatalog(
        by_name={
            "first.weight": (("shared", 1), 2),
            "second.weight": (("shared", 1), 2),
        },
        aliases={
            ("shared", 1): ("first.weight",),
        },
        total_unique_elements=4,
    )
    with pytest.raises(RuntimeError, match="same dense baseline weight"):
        logical_coverage_from_inventory(
            catalog,
            {"names": ["first", "second"]},
        )
    with pytest.raises(RuntimeError, match="names are not canonical"):
        logical_coverage_from_inventory(catalog, {"names": ["first", 1]})

    mismatched_runtime = {**valid_runtime, "quantized_module_names": ["other"]}
    mismatched_runtime["quantized_module_names_sha256"] = canonical_names_sha256(
        ["other"]
    )
    with pytest.raises(ValueError, match="do not match packed module names"):
        require_inventory_logical_binding(mismatched_runtime, valid_logical)
