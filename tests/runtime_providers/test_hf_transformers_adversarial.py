from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import invarlock.runtime_providers.hf_transformers as hf
from invarlock.core.runtime_provider import (
    ModelRuntimeSpec,
    RuntimeArtifactResources,
    RuntimeExecutionContext,
)

_BARE_DIGEST = "a" * 64
_IMAGE_DIGEST = "sha256:" + "b" * 64


@pytest.mark.parametrize("value", [7, "", " value"])
def test_optional_text_rejects_non_string_empty_or_untrimmed_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="non-empty trimmed string"):
        hf._optional_text({"value": value}, "value")  # type: ignore[dict-item]


def test_optional_digest_accepts_prefix_but_rejects_malformed_value() -> None:
    assert hf._optional_sha256({"digest": f"sha256:{_BARE_DIGEST}"}, "digest") == (
        _BARE_DIGEST
    )
    with pytest.raises(ValueError, match="sha256 digest"):
        hf._optional_sha256({"digest": "sha256:short"}, "digest")


def test_malformed_host_path_is_treated_as_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Path,
        "exists",
        lambda _path: (_ for _ in ()).throw(OSError("path rejected")),
    )
    assert hf._is_local_path_like("model-id") is True


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"batch_size": 0}, "positive integer"),
        ({"context_length": True}, "positive integer"),
        ({"seed": -1}, "non-negative integer"),
        ({"offline": "yes"}, "offline must be boolean"),
    ],
)
def test_setting_validation_rejects_ambiguous_scalar_values(
    settings: dict[str, str | int | float | bool | None],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        hf._validate_setting_values(
            ModelRuntimeSpec("hf_transformers", "model", settings)
        )


@pytest.mark.parametrize(
    ("value", "positive", "message"),
    [
        (None, True, "positive integer"),
        (True, True, "positive integer"),
        (0, True, "positive integer"),
        (-1, False, "non-negative integer"),
    ],
)
def test_required_integer_enforces_strict_receipt_domain(
    value: object,
    positive: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        hf._required_integer(
            {"value": value},  # type: ignore[dict-item]
            "value",
            positive=positive,
        )


def test_image_reference_can_be_the_digest_itself() -> None:
    assert hf._image_ref_matches_digest(_IMAGE_DIGEST, _IMAGE_DIGEST) is True
    assert hf._image_ref_matches_digest("runtime:latest", _IMAGE_DIGEST) is False


def test_strict_runtime_boundary_requires_pinned_context_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hf, "network_allowed", lambda: False)
    monkeypatch.setattr(hf, "remote_code_allowed", lambda: False)
    monkeypatch.setattr(hf, "third_party_plugins_allowed", lambda: False)
    monkeypatch.setattr(hf, "strict_container_boundary_present", lambda: True)
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=None,
        device_kind="cpu",
    )

    with pytest.raises(ValueError, match="pinned outer container image"):
        hf._require_strict_runtime_boundary(context)


def test_safetensors_binding_requires_callable_mapping_state_dict(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="does not expose state_dict"):
        hf._require_safetensors_match(tmp_path, model=object())

    raising = SimpleNamespace(
        state_dict=lambda: (_ for _ in ()).throw(RuntimeError("unavailable"))
    )
    with pytest.raises(RuntimeError, match="state is unavailable"):
        hf._require_safetensors_match(tmp_path, model=raising)

    nonmapping = SimpleNamespace(state_dict=lambda: [])
    with pytest.raises(RuntimeError, match="state is unavailable"):
        hf._require_safetensors_match(tmp_path, model=nonmapping)


def test_safetensors_binding_rejects_missing_and_non_tensor_live_values(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    safetensors_torch.save_file(
        {"weight": torch.tensor([1])},
        tmp_path / "model.safetensors",
    )
    with pytest.raises(ValueError, match="missing authenticated"):
        hf._require_safetensors_match(
            tmp_path,
            model=SimpleNamespace(state_dict=lambda: {}),
        )

    with pytest.raises(RuntimeError, match="non-tensor value"):
        hf._require_safetensors_match(
            tmp_path,
            model=SimpleNamespace(state_dict=lambda: {"weight": object()}),
        )


def test_safetensors_binding_authenticates_base_prefix_and_live_buffers(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    weight = torch.tensor([1.0, 2.0])
    attention_mask = torch.tensor([[True, False], [True, True]])
    safetensors_torch.save_file(
        {"attention.bias": attention_mask, "weight": weight},
        tmp_path / "model.safetensors",
    )
    buffers = {"transformer.attention.bias": attention_mask.clone()}
    model = SimpleNamespace(
        base_model_prefix="transformer",
        state_dict=lambda: {"transformer.weight": weight.clone()},
        get_buffer=lambda name: buffers[name],
    )

    hf._require_safetensors_match(tmp_path, model=model)

    buffers["transformer.attention.bias"] = torch.logical_not(attention_mask)
    with pytest.raises(ValueError, match="tensors do not match"):
        hf._require_safetensors_match(tmp_path, model=model)


def test_safetensors_binding_authenticates_language_component_mapping(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    weight = torch.tensor([1.0, 2.0])
    safetensors_torch.save_file(
        {"model.weight": weight},
        tmp_path / "model.safetensors",
    )
    live = {"model.language_model.weight": weight.clone()}
    model = SimpleNamespace(
        base_model_prefix="model",
        state_dict=lambda: live,
    )

    hf._require_safetensors_match(tmp_path, model=model)

    live["model.language_model.weight"] = torch.tensor([2.0, 1.0])
    with pytest.raises(ValueError, match="tensors do not match"):
        hf._require_safetensors_match(tmp_path, model=model)


def test_safetensors_binding_authenticates_removed_gpt2_causal_masks(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    weight = torch.tensor([1.0, 2.0])
    causal_mask = torch.ones((1, 1, 4, 4), dtype=torch.float32).tril()
    model = SimpleNamespace(
        base_model_prefix="transformer",
        config=SimpleNamespace(
            max_position_embeddings=4,
            model_type="gpt2",
            num_hidden_layers=1,
        ),
        state_dict=lambda: {"transformer.weight": weight.clone()},
    )
    safetensors_torch.save_file(
        {"h.0.attn.bias": causal_mask, "weight": weight},
        tmp_path / "model.safetensors",
    )

    hf._require_safetensors_match(tmp_path, model=model)

    invalid_mask = causal_mask.clone()
    invalid_mask[0, 0, 0, 1] = 1
    safetensors_torch.save_file(
        {"h.0.attn.bias": invalid_mask, "weight": weight},
        tmp_path / "model.safetensors",
    )
    with pytest.raises(ValueError, match="missing authenticated"):
        hf._require_safetensors_match(tmp_path, model=model)


def test_safetensors_binding_rejects_untrusted_base_prefix_mapping(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    weight = torch.tensor([1.0])
    safetensors_torch.save_file(
        {"weight": weight},
        tmp_path / "model.safetensors",
    )

    with pytest.raises(ValueError, match="missing authenticated"):
        hf._require_safetensors_match(
            tmp_path,
            model=SimpleNamespace(
                base_model_prefix="../transformer",
                state_dict=lambda: {"../transformer.weight": weight},
            ),
        )


def test_safetensors_binding_detects_inventory_change_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(hf, "safetensors_storage_keys", lambda _path: {"weight"})

    with pytest.raises(RuntimeError, match="inventory changed"):
        hf._require_safetensors_match(
            tmp_path,
            model=SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([1])}),
        )


def _native_transformers_model() -> Any:
    model_class = type("NativeModel", (), {})
    model_class.__module__ = "transformers.models.test.modeling_test"
    model = model_class()
    model.base_model_prefix = "model"
    return model


def _install_checkpoint_conversion_api(
    monkeypatch: pytest.MonkeyPatch,
    *,
    conversions: list[object],
    rename: object,
    weight_renaming: type[object],
    weight_converter: type[object],
) -> None:
    real_import = importlib.import_module
    conversion_module = SimpleNamespace(
        get_model_conversion_mapping=lambda _model: conversions
    )
    core_module = SimpleNamespace(
        rename_source_key=rename,
        WeightRenaming=weight_renaming,
        WeightConverter=weight_converter,
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, package=None: (
            conversion_module
            if name == "transformers.conversion_mapping"
            else core_module
            if name == "transformers.core_model_loading"
            else real_import(name, package)
        ),
    )


def test_safetensors_binding_uses_authoritative_pure_renames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    class Renaming:
        pass

    class Converter:
        pass

    _install_checkpoint_conversion_api(
        monkeypatch,
        conversions=[Renaming()],
        rename=lambda source, *_args: (
            "model.current.weight" if source == "model.legacy.weight" else source,
            None,
        ),
        weight_renaming=Renaming,
        weight_converter=Converter,
    )
    model = _native_transformers_model()
    live_weight = torch.tensor([1.0, 2.0])
    live_state = {"model.current.weight": live_weight}
    model.state_dict = lambda: live_state
    safetensors_torch.save_file(
        {"model.legacy.weight": live_weight.clone()},
        tmp_path / "model.safetensors",
    )

    targets = hf._authoritative_checkpoint_key_targets(
        {"model.legacy.weight"},
        live_state=live_state,
        model=model,
    )
    bindings = hf._bind_authenticated_live_tensors(
        {"model.legacy.weight"},
        live_state=live_state,
        model=model,
        prefix="model",
        authoritative_targets=targets,
    )

    assert targets == {"model.legacy.weight": "model.current.weight"}
    assert bindings == {"model.legacy.weight": (("model.current.weight", live_weight),)}
    hf._require_safetensors_match(tmp_path, model=model)


def test_safetensors_binding_rejects_unmapped_authoritative_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    class Renaming:
        pass

    class Converter:
        pass

    _install_checkpoint_conversion_api(
        monkeypatch,
        conversions=[Renaming()],
        rename=lambda _source, *_args: ("model.absent.weight", None),
        weight_renaming=Renaming,
        weight_converter=Converter,
    )
    model = _native_transformers_model()
    model.state_dict = lambda: {"model.other.weight": torch.tensor([1.0])}
    safetensors_torch.save_file(
        {"model.legacy.weight": torch.tensor([1.0])},
        tmp_path / "model.safetensors",
    )

    with pytest.raises(ValueError, match="missing authenticated"):
        hf._require_safetensors_match(tmp_path, model=model)


def test_safetensors_binding_rejects_conversion_and_rename_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Renaming:
        pass

    class Converter:
        pass

    model = _native_transformers_model()
    _install_checkpoint_conversion_api(
        monkeypatch,
        conversions=[Converter()],
        rename=lambda source, *_args: (source, None),
        weight_renaming=Renaming,
        weight_converter=Converter,
    )
    with pytest.raises(ValueError, match="unsupported tensor conversion"):
        hf._authoritative_checkpoint_key_targets(
            {"a.weight"}, live_state={"a.weight": object()}, model=model
        )

    _install_checkpoint_conversion_api(
        monkeypatch,
        conversions=[Renaming()],
        rename=lambda _source, *_args: ("model.weight", None),
        weight_renaming=Renaming,
        weight_converter=Converter,
    )
    with pytest.raises(ValueError, match="not one-to-one"):
        hf._authoritative_checkpoint_key_targets(
            {"a.weight", "b.weight"},
            live_state={"model.weight": object()},
            model=model,
        )


_EXPECTED_QWEN3_5_MTP_KEYS = {
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
}


def _qwen3_5_test_model() -> object:
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["full_attention"],
    )
    return transformers.Qwen3_5ForCausalLM(config)


def _qwen3_5_multimodal_test_model() -> object:
    transformers = pytest.importorskip("transformers")
    text_config = transformers.Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["full_attention"],
    )
    vision_config = transformers.Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=16,
        num_position_embeddings=16,
    )
    config = transformers.Qwen3_5Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=30,
        video_token_id=29,
        vision_start_token_id=28,
        vision_end_token_id=27,
    )
    return transformers.Qwen3_5ForConditionalGeneration(config)


def _qwen3_5_linear_multimodal_test_model() -> object:
    transformers = pytest.importorskip("transformers")
    text_config = transformers.Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["linear_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        dtype="bfloat16",
    )
    vision_config = transformers.Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=16,
        num_position_embeddings=16,
    )
    config = transformers.Qwen3_5Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=30,
        video_token_id=29,
        vision_start_token_id=28,
        vision_end_token_id=27,
    )
    return transformers.Qwen3_5ForConditionalGeneration(config)


def _authenticated_qwen_config(checkpoint: Path) -> object:
    transformers = pytest.importorskip("transformers")
    return transformers.AutoConfig.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )


def test_qwen3_5_exact_mtp_inventory_is_explicitly_non_executing() -> None:
    model = _qwen3_5_test_model()

    assert hf._qwen3_5_non_executing_checkpoint_keys(
        set(_EXPECTED_QWEN3_5_MTP_KEYS) | {"model.weight"},
        live_state={"model.weight": object()},
        model=model,
        authenticated_config=model.config,
    ) == set(_EXPECTED_QWEN3_5_MTP_KEYS)


def test_qwen3_5_multimodal_exact_mtp_inventory_is_explicitly_non_executing() -> None:
    model = _qwen3_5_multimodal_test_model()

    assert hf._qwen3_5_non_executing_checkpoint_keys(
        set(_EXPECTED_QWEN3_5_MTP_KEYS) | {"model.weight"},
        live_state={"model.weight": object()},
        model=model,
        authenticated_config=model.config,
    ) == set(_EXPECTED_QWEN3_5_MTP_KEYS)


@pytest.mark.parametrize("mutation", ["forged", "partial", "live"])
def test_qwen3_5_mtp_exception_rejects_forged_partial_or_live_state(
    mutation: str,
) -> None:
    keys = set(_EXPECTED_QWEN3_5_MTP_KEYS)
    if mutation == "forged":
        keys.add("mtp.forged.weight")
    elif mutation == "partial":
        keys.remove("mtp.fc.weight")
    model = _qwen3_5_test_model()
    live_state = {"model.weight": object()}
    if mutation == "live":
        live_state["mtp.hidden"] = object()

    message = "overlap live" if mutation == "live" else "unsupported non-executing"
    with pytest.raises(ValueError, match=message):
        hf._qwen3_5_non_executing_checkpoint_keys(
            keys,
            live_state=live_state,
            model=model,
            authenticated_config=model.config,
        )


def test_real_qwen_causal_checkpoint_accepts_exact_nonexecuting_mtp(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["full_attention"],
    )
    transformers.Qwen3_5ForCausalLM(config).save_pretrained(
        tmp_path,
        safe_serialization=True,
    )
    shard = tmp_path / "model.safetensors"
    tensors = safetensors_torch.load_file(shard)
    for index, key in enumerate(sorted(_EXPECTED_QWEN3_5_MTP_KEYS)):
        tensors[key] = torch.tensor([float(index)])
    safetensors_torch.save_file(tensors, shard, metadata={"format": "pt"})

    model = hf.load_hf_model_with_strict_loading_info(
        transformers.AutoModelForCausalLM.from_pretrained,
        tmp_path,
    )
    model.eval()

    assert type(model) is transformers.Qwen3_5ForCausalLM
    assert model.config.model_type == "qwen3_5_text"
    assert not any(key.startswith("mtp.") for key in model.state_dict())
    hf._require_safetensors_match(
        tmp_path,
        model=model,
        authenticated_config=_authenticated_qwen_config(tmp_path),
    )


def test_real_qwen_multimodal_checkpoint_accepts_exact_nonexecuting_mtp(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    transformers = pytest.importorskip("transformers")
    _qwen3_5_multimodal_test_model().save_pretrained(
        tmp_path,
        safe_serialization=True,
    )
    shard = tmp_path / "model.safetensors"
    tensors = safetensors_torch.load_file(shard)
    for index, key in enumerate(sorted(_EXPECTED_QWEN3_5_MTP_KEYS)):
        tensors[key] = torch.tensor([float(index)])
    safetensors_torch.save_file(tensors, shard, metadata={"format": "pt"})

    model = hf.load_hf_model_with_strict_loading_info(
        transformers.AutoModelForImageTextToText.from_pretrained,
        tmp_path,
    )
    model.eval()

    assert type(model) is transformers.Qwen3_5ForConditionalGeneration
    assert model.config.model_type == "qwen3_5"
    assert not any(key.startswith("mtp.") for key in model.state_dict())
    hf._require_safetensors_match(
        tmp_path,
        model=model,
        authenticated_config=_authenticated_qwen_config(tmp_path),
    )


def test_real_qwen_multimodal_checkpoint_authenticates_causal_projection(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    transformers = pytest.importorskip("transformers")
    _qwen3_5_multimodal_test_model().save_pretrained(
        tmp_path,
        safe_serialization=True,
    )
    shard = tmp_path / "model.safetensors"
    tensors = safetensors_torch.load_file(shard)
    for index, key in enumerate(sorted(_EXPECTED_QWEN3_5_MTP_KEYS)):
        tensors[key] = torch.tensor([float(index)])
    safetensors_torch.save_file(tensors, shard, metadata={"format": "pt"})

    model = hf.load_hf_model_with_strict_loading_info(
        transformers.AutoModelForCausalLM.from_pretrained,
        tmp_path,
    )
    model.eval()

    assert type(model) is transformers.Qwen3_5ForCausalLM
    assert model.config.model_type == "qwen3_5_text"
    authenticated_config = hf._require_model_config_match(tmp_path, model=model)
    hf._require_safetensors_match(
        tmp_path,
        model=model,
        authenticated_config=authenticated_config,
    )


def test_qwen_multimodal_causal_projection_authenticates_native_bfloat16_cast(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    transformers = pytest.importorskip("transformers")
    _qwen3_5_linear_multimodal_test_model().to(torch.bfloat16).save_pretrained(
        tmp_path,
        safe_serialization=True,
    )
    config_path = tmp_path / "config.json"
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_payload.pop("dtype", None)
    config_path.write_text(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    shard = tmp_path / "model.safetensors"
    tensors = safetensors_torch.load_file(shard)
    for key in (
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.norm.weight",
    ):
        tensors[key] = tensors[key].to(torch.float32)
    safetensors_torch.save_file(tensors, shard, metadata={"format": "pt"})

    model = hf.load_hf_model_with_strict_loading_info(
        transformers.AutoModelForCausalLM.from_pretrained,
        tmp_path,
    )
    model.eval()
    authenticated_config = hf._require_model_config_match(tmp_path, model=model)

    assert model.config.dtype == torch.bfloat16
    hf._require_safetensors_match(
        tmp_path,
        model=model,
        authenticated_config=authenticated_config,
    )


def test_qwen_multimodal_causal_projection_rejects_top_level_profile_drift(
    tmp_path: Path,
) -> None:
    transformers = pytest.importorskip("transformers")
    _qwen3_5_multimodal_test_model().save_pretrained(
        tmp_path,
        safe_serialization=True,
    )
    model = hf.load_hf_model_with_strict_loading_info(
        transformers.AutoModelForCausalLM.from_pretrained,
        tmp_path,
    )
    config_path = tmp_path / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["architectures"] = ["Qwen3_5ForCausalLM"]
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Qwen3.5 causal projection"):
        hf._require_model_config_match(tmp_path, model=model)


def test_qwen_multimodal_causal_projection_rejects_live_text_config_drift(
    tmp_path: Path,
) -> None:
    transformers = pytest.importorskip("transformers")
    _qwen3_5_multimodal_test_model().save_pretrained(
        tmp_path,
        safe_serialization=True,
    )
    model = hf.load_hf_model_with_strict_loading_info(
        transformers.AutoModelForCausalLM.from_pretrained,
        tmp_path,
    )
    model.config.hidden_size += 1

    with pytest.raises(ValueError, match="does not match"):
        hf._require_model_config_match(tmp_path, model=model)


def test_real_qwen_multimodal_checkpoint_accepts_exact_native_bfloat16_cast(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    transformers = pytest.importorskip("transformers")
    _qwen3_5_linear_multimodal_test_model().to(torch.bfloat16).save_pretrained(
        tmp_path,
        safe_serialization=True,
    )
    config_path = tmp_path / "config.json"
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_payload.pop("dtype", None)
    config_path.write_text(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    shard = tmp_path / "model.safetensors"
    tensors = safetensors_torch.load_file(shard)
    for key in (
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.norm.weight",
    ):
        tensors[key] = tensors[key].to(torch.float32)
    safetensors_torch.save_file(tensors, shard, metadata={"format": "pt"})

    model = hf.load_hf_model_with_strict_loading_info(
        transformers.AutoModelForImageTextToText.from_pretrained,
        tmp_path,
    )
    model.eval()

    assert model.config.dtype == torch.bfloat16
    assert model.config.text_config.dtype == torch.bfloat16
    hf._require_safetensors_match(
        tmp_path,
        model=model,
        authenticated_config=_authenticated_qwen_config(tmp_path),
    )


def test_real_qwen_multimodal_checkpoint_accepts_all_bfloat16_storage(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    _qwen3_5_linear_multimodal_test_model().to(torch.bfloat16).save_pretrained(
        tmp_path,
        safe_serialization=True,
    )

    model = hf.load_hf_model_with_strict_loading_info(
        transformers.AutoModelForImageTextToText.from_pretrained,
        tmp_path,
    )
    model.eval()

    hf._require_safetensors_match(
        tmp_path,
        model=model,
        authenticated_config=_authenticated_qwen_config(tmp_path),
    )


@pytest.mark.parametrize("mutation", ["partial", "extra"])
def test_qwen_native_bfloat16_cast_rejects_unexpected_storage_inventory(
    tmp_path: Path,
    mutation: str,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    transformers = pytest.importorskip("transformers")
    _qwen3_5_linear_multimodal_test_model().to(torch.bfloat16).save_pretrained(
        tmp_path,
        safe_serialization=True,
    )
    shard = tmp_path / "model.safetensors"
    tensors = safetensors_torch.load_file(shard)
    cast_keys = (
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.norm.weight",
    )
    for key in cast_keys:
        tensors[key] = tensors[key].to(torch.float32)
    if mutation == "partial":
        tensors[cast_keys[1]] = tensors[cast_keys[1]].to(torch.bfloat16)
    else:
        extra_key = "model.language_model.layers.0.linear_attn.dt_bias"
        tensors[extra_key] = tensors[extra_key].to(torch.float32)
    safetensors_torch.save_file(tensors, shard, metadata={"format": "pt"})

    model = hf.load_hf_model_with_strict_loading_info(
        transformers.AutoModelForImageTextToText.from_pretrained,
        tmp_path,
    )
    model.eval()

    message = "profile is incomplete" if mutation == "partial" else "do not match"
    with pytest.raises(ValueError, match=message):
        hf._require_safetensors_match(
            tmp_path,
            model=model,
            authenticated_config=_authenticated_qwen_config(tmp_path),
        )


@pytest.mark.parametrize("mutation", ["value", "live_float32"])
def test_qwen_native_bfloat16_cast_rejects_live_tensor_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    transformers = pytest.importorskip("transformers")
    _qwen3_5_linear_multimodal_test_model().to(torch.bfloat16).save_pretrained(
        tmp_path,
        safe_serialization=True,
    )
    shard = tmp_path / "model.safetensors"
    tensors = safetensors_torch.load_file(shard)
    for key in (
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.norm.weight",
    ):
        tensors[key] = tensors[key].to(torch.float32)
    safetensors_torch.save_file(tensors, shard, metadata={"format": "pt"})
    model = hf.load_hf_model_with_strict_loading_info(
        transformers.AutoModelForImageTextToText.from_pretrained,
        tmp_path,
    )
    model.eval()
    if mutation == "value":
        with torch.no_grad():
            model.model.language_model.layers[0].linear_attn.A_log.add_(1)
    else:
        model.float()

    with pytest.raises(ValueError, match="do not match"):
        hf._require_safetensors_match(
            tmp_path,
            model=model,
            authenticated_config=_authenticated_qwen_config(tmp_path),
        )


@pytest.mark.parametrize("dtype_name", ["float16", "float64"])
def test_qwen_native_bfloat16_profile_rejects_other_exact_floating_dtypes(
    tmp_path: Path,
    dtype_name: str,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    transformers = pytest.importorskip("transformers")
    _qwen3_5_linear_multimodal_test_model().to(torch.bfloat16).save_pretrained(
        tmp_path,
        safe_serialization=True,
    )
    shard = tmp_path / "model.safetensors"
    dtype = getattr(torch, dtype_name)
    tensors = safetensors_torch.load_file(shard)
    for key, tensor in tuple(tensors.items()):
        if tensor.is_floating_point():
            tensors[key] = torch.zeros_like(tensor, dtype=dtype)
    safetensors_torch.save_file(tensors, shard, metadata={"format": "pt"})
    model = hf.load_hf_model_with_strict_loading_info(
        transformers.AutoModelForImageTextToText.from_pretrained,
        tmp_path,
    )
    model.eval().to(dtype)

    with pytest.raises(ValueError, match="do not match"):
        hf._require_safetensors_match(
            tmp_path,
            model=model,
            authenticated_config=_authenticated_qwen_config(tmp_path),
        )


def test_qwen_native_bfloat16_profile_requires_authenticated_dtype(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    _qwen3_5_linear_multimodal_test_model().to(torch.bfloat16).save_pretrained(
        tmp_path,
        safe_serialization=True,
    )
    model = hf.load_hf_model_with_strict_loading_info(
        transformers.AutoModelForImageTextToText.from_pretrained,
        tmp_path,
    )
    model.eval()
    authenticated_config = _authenticated_qwen_config(tmp_path)
    authenticated_config.dtype = None
    authenticated_config.text_config.dtype = None

    with pytest.raises(ValueError, match="not authorized"):
        hf._require_safetensors_match(
            tmp_path,
            model=model,
            authenticated_config=authenticated_config,
        )


def test_qwen_native_bfloat16_cast_rejects_loader_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    model = _qwen3_5_linear_multimodal_test_model()
    model.config.dtype = torch.bfloat16
    authenticated_config = copy.deepcopy(model.config)
    monkeypatch.setattr(model, "_get_dtype_plan", lambda _dtype: {"weight": "float32"})
    with pytest.raises(ValueError, match="conversion plan is unsupported"):
        hf._qwen3_5_native_float32_to_bfloat16_keys(
            model,
            authenticated_config=authenticated_config,
        )


@pytest.mark.parametrize("marker", ["config", "is_quantized", "hf_quantizer"])
def test_qwen_native_bfloat16_cast_rejects_quantization(marker: str) -> None:
    torch = pytest.importorskip("torch")
    model = _qwen3_5_linear_multimodal_test_model()
    model.config.dtype = torch.bfloat16
    authenticated_config = copy.deepcopy(model.config)
    if marker == "config":
        authenticated_config.quantization_config = {"quant_method": "unsupported"}
    elif marker == "is_quantized":
        model.is_quantized = True
    else:
        model.hf_quantizer = object()
    with pytest.raises(ValueError, match="requires an unquantized checkpoint"):
        hf._qwen3_5_native_float32_to_bfloat16_keys(
            model,
            authenticated_config=authenticated_config,
        )


def test_model_eval_binding_rejects_missing_unavailable_or_training_modules() -> None:
    with pytest.raises(RuntimeError, match="does not expose module state"):
        hf._require_model_eval_mode(object())
    with pytest.raises(RuntimeError, match="module state is unavailable"):
        hf._require_model_eval_mode(
            SimpleNamespace(
                modules=lambda: (_ for _ in ()).throw(RuntimeError("unavailable"))
            )
        )
    with pytest.raises(RuntimeError, match="requires model.eval"):
        hf._require_model_eval_mode(SimpleNamespace(modules=lambda: ()))
    with pytest.raises(RuntimeError, match="requires model.eval"):
        hf._require_model_eval_mode(
            SimpleNamespace(modules=lambda: (SimpleNamespace(training=True),))
        )


def test_model_config_binding_requires_loader_and_mapping_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(AutoConfig=object()),
    )
    with pytest.raises(RuntimeError, match="configuration is unavailable"):
        hf._require_model_config_match(
            tmp_path,
            model=SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {})),
        )

    class Config:
        def __init__(self, payload: object) -> None:
            self.payload = payload

        def to_dict(self) -> object:
            return self.payload

    loader = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: Config([]))
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(AutoConfig=loader),
    )
    with pytest.raises(RuntimeError, match="configuration is unavailable"):
        hf._require_model_config_match(
            tmp_path, model=SimpleNamespace(config=Config({}))
        )

    loader.from_pretrained = lambda *_args, **_kwargs: object()
    with pytest.raises(RuntimeError, match="could not be authenticated"):
        hf._require_model_config_match(
            tmp_path, model=SimpleNamespace(config=Config({}))
        )


def test_model_config_binding_rejects_class_and_payload_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LiveConfig:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def to_dict(self) -> dict[str, object]:
            return self.payload

    class OtherConfig(LiveConfig):
        pass

    loader = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: OtherConfig({"layers": 2})
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(AutoConfig=loader),
    )
    with pytest.raises(ValueError, match="config class"):
        hf._require_model_config_match(
            tmp_path,
            model=SimpleNamespace(config=LiveConfig({"layers": 2})),
        )

    loader.from_pretrained = lambda *_args, **_kwargs: LiveConfig({"layers": 3})
    with pytest.raises(ValueError, match="does not match"):
        hf._require_model_config_match(
            tmp_path,
            model=SimpleNamespace(config=LiveConfig({"layers": 2})),
        )

    loader.from_pretrained = lambda *_args, **_kwargs: LiveConfig(
        {"layers": 2, "dtype": "bfloat16"}
    )
    with pytest.raises(ValueError, match="does not match"):
        hf._require_model_config_match(
            tmp_path,
            model=SimpleNamespace(config=LiveConfig({"layers": 2, "dtype": "float32"})),
        )


def test_model_config_binding_accepts_dtype_inferred_from_bound_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Config:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def to_dict(self) -> dict[str, object]:
            return self.payload

    loader = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: Config({"layers": 2, "dtype": None})
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(AutoConfig=loader),
    )

    hf._require_model_config_match(
        tmp_path,
        model=SimpleNamespace(config=Config({"layers": 2, "dtype": "float32"})),
    )

    loader.from_pretrained = lambda *_args, **_kwargs: Config(
        {"vision_config": {"dtype": None, "layers": 4}}
    )
    hf._require_model_config_match(
        tmp_path,
        model=SimpleNamespace(
            config=Config({"vision_config": {"dtype": "bfloat16", "layers": 4}})
        ),
    )

    loader.from_pretrained = lambda *_args, **_kwargs: Config(
        {"vision_config": {"dtype": "bfloat16", "layers": 4}}
    )
    with pytest.raises(ValueError, match="does not match"):
        hf._require_model_config_match(
            tmp_path,
            model=SimpleNamespace(
                config=Config({"vision_config": {"dtype": "float32", "layers": 4}})
            ),
        )


class _TestModelConfig:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def to_dict(self) -> dict[str, object]:
        return self.payload


_AUTHENTICATED_FP8_CONFIG: dict[str, object] = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}
_LIVE_FP8_CONFIG: dict[str, object] = {
    "activation_scheme": "dynamic",
    "dequantize": False,
    "modules_to_not_convert": None,
    "quant_method": "fp8",
    "scale_fmt": "float",
    "weight_block_size": [128, 128],
}


def _install_model_config_loader(
    monkeypatch: pytest.MonkeyPatch,
    authenticated_quantization: dict[str, object],
) -> None:
    real_import = importlib.import_module
    loader = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: _TestModelConfig(
            {
                "layers": 40,
                "quantization_config": authenticated_quantization,
            }
        )
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, package=None: (
            SimpleNamespace(AutoConfig=loader)
            if name == "transformers"
            else real_import(name, package)
        ),
    )


def test_model_config_binding_accepts_runtime_canonical_fp8_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_model_config_loader(monkeypatch, _AUTHENTICATED_FP8_CONFIG)

    hf._require_model_config_match(
        tmp_path,
        model=SimpleNamespace(
            config=_TestModelConfig(
                {"layers": 40, "quantization_config": _LIVE_FP8_CONFIG}
            )
        ),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("activation_scheme", "static"),
        ("dequantize", True),
        ("modules_to_not_convert", ["lm_head"]),
        ("scale_fmt", "ue8m0"),
        ("weight_block_size", [64, 128]),
    ],
)
def test_model_config_binding_rejects_runtime_fp8_semantic_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    live_quantization = dict(_LIVE_FP8_CONFIG)
    live_quantization[field] = value
    _install_model_config_loader(monkeypatch, _AUTHENTICATED_FP8_CONFIG)

    with pytest.raises(ValueError, match="does not match"):
        hf._require_model_config_match(
            tmp_path,
            model=SimpleNamespace(
                config=_TestModelConfig(
                    {"layers": 40, "quantization_config": live_quantization}
                )
            ),
        )


def test_model_config_binding_rejects_quantization_config_class_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_model_config_loader(monkeypatch, _AUTHENTICATED_FP8_CONFIG)
    live_quantization = {"quant_method": "fbgemm_fp8"}

    with pytest.raises(ValueError, match="quantization config class"):
        hf._require_model_config_match(
            tmp_path,
            model=SimpleNamespace(
                config=_TestModelConfig(
                    {"layers": 40, "quantization_config": live_quantization}
                )
            ),
        )


@pytest.mark.parametrize("side", ["live", "authenticated", "both"])
def test_model_config_binding_rejects_unknown_fp8_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    side: str,
) -> None:
    authenticated_quantization = dict(_AUTHENTICATED_FP8_CONFIG)
    live_quantization = dict(_LIVE_FP8_CONFIG)
    if side in {"authenticated", "both"}:
        authenticated_quantization["future_behavior_switch"] = True
    if side in {"live", "both"}:
        live_quantization["future_behavior_switch"] = True
    _install_model_config_loader(monkeypatch, authenticated_quantization)

    with pytest.raises(ValueError, match="unsupported fields"):
        hf._require_model_config_match(
            tmp_path,
            model=SimpleNamespace(
                config=_TestModelConfig(
                    {"layers": 40, "quantization_config": live_quantization}
                )
            ),
        )


def test_model_config_binding_rejects_unknown_quantization_method(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_model_config_loader(monkeypatch, _AUTHENTICATED_FP8_CONFIG)

    with pytest.raises(ValueError, match="quantization config class"):
        hf._require_model_config_match(
            tmp_path,
            model=SimpleNamespace(
                config=_TestModelConfig(
                    {"layers": 40, "quantization_config": {"quant_method": "unknown"}}
                )
            ),
        )


@pytest.mark.parametrize("legacy_format", [None, "other"])
def test_model_config_binding_rejects_unsupported_fp8_legacy_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_format: object,
) -> None:
    authenticated = dict(_AUTHENTICATED_FP8_CONFIG)
    authenticated["fmt"] = legacy_format
    _install_model_config_loader(monkeypatch, authenticated)

    with pytest.raises(ValueError, match="legacy format is unsupported"):
        hf._require_model_config_match(
            tmp_path,
            model=SimpleNamespace(
                config=_TestModelConfig(
                    {"layers": 40, "quantization_config": _LIVE_FP8_CONFIG}
                )
            ),
        )


def test_model_config_binding_rejects_live_fp8_legacy_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = dict(_LIVE_FP8_CONFIG)
    live["fmt"] = "e4m3"
    _install_model_config_loader(monkeypatch, _AUTHENTICATED_FP8_CONFIG)

    with pytest.raises(ValueError, match="live fine-grained FP8 config"):
        hf._require_model_config_match(
            tmp_path,
            model=SimpleNamespace(
                config=_TestModelConfig({"layers": 40, "quantization_config": live})
            ),
        )


def test_scorer_identity_and_checkpoint_binding_are_immutable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="artifact_identity_sha256"):
        hf.HFTransformersCausalScorer(object(), object(), "bad")
    with pytest.raises(ValueError, match="checkpoint_path must be absolute"):
        hf.HFTransformersCausalScorer(
            object(),
            object(),
            _BARE_DIGEST,
            Path("relative"),
        )

    model = object()
    scorer = hf.HFTransformersCausalScorer(model, object(), _BARE_DIGEST, tmp_path)
    with pytest.raises(ValueError, match="exact native model"):
        scorer.require_binding(model=object(), artifact_identity_sha256=_BARE_DIGEST)
    with pytest.raises(ValueError, match="artifact identity"):
        scorer.require_binding(model=model, artifact_identity_sha256="c" * 64)


def test_artifact_identity_requires_tokenizer_digest() -> None:
    spec = ModelRuntimeSpec(
        "hf_transformers",
        "org/model",
        {"immutable_revision": "d" * 40},
    )
    with pytest.raises(ValueError, match="requires tokenizer_metadata_sha256"):
        hf.HFTransformersProvider().identify_artifact(spec)


def _resources(tmp_path: Path, artifact: str) -> RuntimeArtifactResources:
    return RuntimeArtifactResources(
        root=tmp_path,
        primary_artifact=artifact,
        support_resources={},
        device_kind="cpu",
        container_image_digest=_IMAGE_DIGEST,
    )


def test_provider_preparation_rejects_file_and_unbound_checkpoint_tree(
    tmp_path: Path,
) -> None:
    artifact_file = tmp_path / "model.bin"
    artifact_file.write_bytes(b"model")
    with pytest.raises(ValueError, match="must be a directory"):
        hf.HFTransformersProvider().prepare_execution(
            ModelRuntimeSpec("hf_transformers", "org/model"),
            _resources(tmp_path, artifact_file.name),
        )

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="requires checkpoint_tree_sha256"):
        hf.HFTransformersProvider().prepare_execution(
            ModelRuntimeSpec(
                "hf_transformers",
                "org/model",
                {
                    "immutable_revision": "d" * 40,
                    "tokenizer_metadata_sha256": _BARE_DIGEST,
                },
            ),
            _resources(tmp_path, checkpoint.name),
        )


def test_provider_preparation_rejects_unreadable_or_mismatched_checkpoint_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    spec = ModelRuntimeSpec(
        "hf_transformers",
        "org/model",
        {
            "checkpoint_tree_sha256": _BARE_DIGEST,
            "tokenizer_metadata_sha256": _BARE_DIGEST,
        },
    )
    resources = _resources(tmp_path, checkpoint.name)
    monkeypatch.setattr(
        hf,
        "checkpoint_tree_sha256",
        lambda _path: (_ for _ in ()).throw(OSError("unreadable")),
    )
    with pytest.raises(ValueError, match="could not be authenticated"):
        hf.HFTransformersProvider().prepare_execution(spec, resources)

    monkeypatch.setattr(hf, "checkpoint_tree_sha256", lambda _path: "c" * 64)
    with pytest.raises(ValueError, match="does not match"):
        hf.HFTransformersProvider().prepare_execution(spec, resources)
