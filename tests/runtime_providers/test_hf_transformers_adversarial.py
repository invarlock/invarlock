from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hf, "safetensors_storage_keys", lambda _path: {"weight"})
    with pytest.raises(ValueError, match="missing authenticated"):
        hf._require_safetensors_match(
            tmp_path,
            model=SimpleNamespace(state_dict=lambda: {}),
        )

    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    safetensors_torch.save_file(
        {"weight": torch.tensor([1])},
        tmp_path / "model.safetensors",
    )
    with pytest.raises(RuntimeError, match="non-tensor value"):
        hf._require_safetensors_match(
            tmp_path,
            model=SimpleNamespace(state_dict=lambda: {"weight": object()}),
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
