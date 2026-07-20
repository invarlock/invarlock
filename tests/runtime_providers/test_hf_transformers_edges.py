from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.runtime_providers import hf_transformers as provider


def _native_model() -> object:
    model_class = type(
        "NativeModel",
        (),
        {"__module__": "transformers.models.fixture.modeling_fixture"},
    )
    return model_class()


def test_native_conversion_metadata_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _native_model()
    monkeypatch.setattr(
        provider.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("missing")),
    )
    with pytest.raises(RuntimeError, match="conversion metadata is unavailable"):
        provider._authoritative_checkpoint_key_targets(
            {"weight"}, live_state={}, model=model
        )

    class Renaming:
        pass

    class Converter:
        pass

    conversion = SimpleNamespace(get_model_conversion_mapping=lambda _model: ())
    loading = SimpleNamespace(
        rename_source_key=lambda *_args: ("weight", None),
        WeightRenaming=Renaming,
        WeightConverter=Converter,
    )
    monkeypatch.setattr(
        provider.importlib,
        "import_module",
        lambda name: conversion if name.endswith("conversion_mapping") else loading,
    )
    with pytest.raises(RuntimeError, match="conversion metadata is invalid"):
        provider._authoritative_checkpoint_key_targets(
            {"weight"}, live_state={}, model=model
        )

    conversion.get_model_conversion_mapping = lambda _model: [Renaming()]
    loading.rename_source_key = lambda *_args: (_ for _ in ()).throw(ValueError())
    with pytest.raises(RuntimeError, match="key conversion failed"):
        provider._authoritative_checkpoint_key_targets(
            {"weight"}, live_state={}, model=model
        )

    loading.rename_source_key = lambda *_args: ("", None)
    with pytest.raises(RuntimeError, match="conversion metadata is invalid"):
        provider._authoritative_checkpoint_key_targets(
            {"weight"}, live_state={}, model=model
        )


def test_legacy_gpt2_mask_helpers_validate_config_prefix_and_tensor_shape() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="gpt2",
            max_position_embeddings=8,
            num_hidden_layers=2,
        )
    )
    assert provider._is_legacy_gpt2_causal_mask_key(
        "transformer.h.1.attn.bias", model=model, prefix="transformer"
    )
    assert not provider._is_legacy_gpt2_causal_mask_key(
        "h.2.attn.bias", model=model, prefix=None
    )

    model.config.num_hidden_layers = True
    assert not provider._is_legacy_gpt2_causal_mask_key(
        "h.0.attn.bias", model=model, prefix=None
    )
    model.config.num_hidden_layers = 2
    model.config.max_position_embeddings = True
    assert not provider._is_authenticated_legacy_gpt2_causal_mask(
        "h.0.attn.bias", object(), model=model, prefix=None
    )
    model.config.max_position_embeddings = 8
    assert not provider._is_authenticated_legacy_gpt2_causal_mask(
        "h.0.attn.bias", SimpleNamespace(shape=(1,)), model=model, prefix=None
    )


def test_tensor_storage_identity_and_alias_helpers_fail_closed() -> None:
    assert provider._tensor_storage_identity(object()) is None

    class EmptyStorage:
        def data_ptr(self) -> int:
            return 0

    class EmptyTensor:
        def detach(self) -> EmptyTensor:
            return self

        def untyped_storage(self) -> EmptyStorage:
            return EmptyStorage()

    empty = EmptyTensor()
    assert provider._tensor_storage_identity(empty) is None
    assert provider._tensors_share_exact_storage(empty, empty)

    class BrokenTensor(EmptyTensor):
        def untyped_storage(self) -> EmptyStorage:
            raise RuntimeError("unavailable")

    assert provider._tensor_storage_identity(BrokenTensor()) is None


def test_strict_loader_requires_complete_loading_information(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="did not return loading information"):
        provider.load_hf_model_with_strict_loading_info(
            lambda *_args, **_kwargs: object(), tmp_path
        )

    with pytest.raises(RuntimeError, match="invalid loading information"):
        provider.load_hf_model_with_strict_loading_info(
            lambda *_args, **_kwargs: (SimpleNamespace(), {}), tmp_path
        )
