from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from invarlock_addins.multimodal import provider as provider_module
from invarlock_addins.multimodal.provider import (
    HFVisionTextProvider,
    HFVisionTextScorer,
    processor_contract_sha256,
)

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import (
    EvaluationBatch,
    EvaluationRecord,
    ModelRuntimeSpec,
    RuntimeArtifactResources,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionContext,
    RuntimeExecutionSettings,
    RuntimeScoringRecord,
    ScoringObservation,
)
from invarlock.core.runtime_provider.behavioral_observation import (
    runtime_scoring_records_sha256,
)
from invarlock.runtime_provider_evidence import encode_scoring_observation

_IMAGE_DIGEST = "sha256:" + "9" * 64


def _spec(**overrides: object) -> ModelRuntimeSpec:
    settings: dict[str, object] = {
        "batch_size": 1,
        "checkpoint_tree_sha256": "b" * 64,
        "context_length": 32,
        "max_output_tokens": 4,
        "offline": True,
        "processor_metadata_sha256": "c" * 64,
        "seed": 7,
        "timeout_seconds": 30,
        "tokenizer_metadata_sha256": "d" * 64,
    }
    settings.update(overrides)
    return ModelRuntimeSpec(
        provider_name="hf_vision_text",
        model_id="org/vision-text-model",
        settings=settings,  # type: ignore[arg-type]
    )


class _Tokenizer:
    special_tokens_map = {"pad_token": "<pad>", "eos_token": "</s>"}
    chat_template = "{{ messages }}"

    def get_vocab(self) -> dict[str, int]:
        return {"<pad>": 0, "cat": 1, "dog": 2}


class _ImageProcessor:
    def to_dict(self) -> dict[str, object]:
        return {
            "do_resize": True,
            "size": {"height": 2, "width": 2},
            "mean": [0.5, 0.5, 0.5],
        }


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.image_processor = _ImageProcessor()
        self.chat_template = "{{ image }} {{ prompt }}"


class _Model:
    def __init__(self) -> None:
        self.training = True
        self.device = SimpleNamespace(type="cpu", index=None)
        self.moves: list[str] = []

    def to(self, device: str) -> _Model:
        self.moves.append(device)
        self.device = SimpleNamespace(type=device, index=None)
        return self

    def eval(self) -> _Model:
        self.training = False
        return self

    def modules(self) -> tuple[object, ...]:
        return (self,)

    def parameters(self) -> tuple[object, ...]:
        return (SimpleNamespace(device=self.device),)

    def buffers(self) -> tuple[object, ...]:
        return ()

    def generate(self, **_kwargs: object) -> object:
        raise AssertionError("the lifecycle test replaces scoring")


def _batch() -> EvaluationBatch:
    input_text = "What animal is shown?"
    return EvaluationBatch(
        schedule_sha256=hashlib.sha256(b"vision-schedule").hexdigest(),
        records=(
            EvaluationRecord(
                record_id="vision/1",
                input_text=input_text,
                input_sha256=hashlib.sha256(input_text.encode()).hexdigest(),
                expected_output="cat",
            ),
        ),
        metric="exact_match",
        task="vision_text_generation",
    )


def _observation(batch: EvaluationBatch, artifact_sha256: str) -> ScoringObservation:
    output = "cat"
    records = (
        RuntimeScoringRecord(
            record_id=batch.records[0].record_id,
            input_sha256=batch.records[0].input_sha256,
            status="ok",
            output_text=output,
            output_sha256=hashlib.sha256(output.encode()).hexdigest(),
        ),
    )
    return ScoringObservation(
        provider_name="hf_vision_text",
        artifact_identity_sha256=artifact_sha256,
        schedule_sha256=batch.schedule_sha256,
        records=records,
        aggregate_source_sha256=runtime_scoring_records_sha256(
            [
                {
                    "record_id": records[0].record_id,
                    "input_sha256": records[0].input_sha256,
                    "status": records[0].status,
                    "output_text": records[0].output_text,
                    "output_sha256": records[0].output_sha256,
                    "logprob_sum": None,
                    "token_count": None,
                    "utf8_byte_count": None,
                    "error_code": None,
                }
            ]
        ),
    )


def test_processor_contract_is_deterministic_and_binds_live_configuration() -> None:
    processor = _Processor()

    first = processor_contract_sha256(processor)
    second = processor_contract_sha256(processor)
    processor.image_processor.to_dict = lambda: {  # type: ignore[method-assign]
        "do_resize": False,
        "size": {"height": 2, "width": 2},
        "mean": [0.5, 0.5, 0.5],
    }
    changed = processor_contract_sha256(processor)

    assert len(first) == 64
    assert first == second
    assert changed != first


@pytest.mark.parametrize(
    ("processor", "message"),
    [
        (SimpleNamespace(), "tokenizer vocabulary"),
        (
            SimpleNamespace(
                tokenizer=SimpleNamespace(get_vocab=lambda: {}),
                image_processor=_ImageProcessor(),
                chat_template="template",
            ),
            "vocabulary is unavailable",
        ),
        (
            SimpleNamespace(
                tokenizer=SimpleNamespace(get_vocab=lambda: {"bad": True}),
                image_processor=_ImageProcessor(),
                chat_template="template",
            ),
            "vocabulary is invalid",
        ),
        (
            SimpleNamespace(
                tokenizer=_Tokenizer(), image_processor=object(), chat_template="x"
            ),
            "configuration is unavailable",
        ),
        (
            SimpleNamespace(
                tokenizer=_Tokenizer(),
                image_processor=SimpleNamespace(to_dict=lambda: {}),
                chat_template="x",
            ),
            "configuration is unavailable",
        ),
        (
            SimpleNamespace(
                tokenizer=SimpleNamespace(
                    get_vocab=lambda: {"cat": 1}, chat_template=None
                ),
                image_processor=_ImageProcessor(),
                chat_template=None,
            ),
            "chat template is unavailable",
        ),
        (
            SimpleNamespace(
                tokenizer=_Tokenizer(),
                image_processor=SimpleNamespace(to_dict=lambda: {"bad": {1, 2}}),
                chat_template="x",
            ),
            "non-JSON value",
        ),
    ],
)
def test_processor_contract_rejects_incomplete_or_ambiguous_inputs(
    processor: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        processor_contract_sha256(processor)


def test_image_decode_authenticates_media_type_and_normalizes_rgb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Decoded:
        format = "PNG"
        n_frames = 1
        size = (2, 3)
        mode = "RGBA"

        def __enter__(self) -> _Decoded:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def verify(self) -> None:
            return None

        def convert(self, mode: str) -> _Decoded:
            self.mode = mode
            return self

        def copy(self) -> _Decoded:
            return self

        def close(self) -> None:
            return None

    image_module = SimpleNamespace(
        DecompressionBombWarning=Warning,
        open=lambda _stream: _Decoded(),
    )
    real_import = importlib.import_module
    monkeypatch.setattr(
        provider_module.importlib,
        "import_module",
        lambda name: image_module if name == "PIL.Image" else real_import(name),
    )
    payload = b"authenticated image bytes"

    decoded = provider_module._decode_image(payload, media_type="image/png")

    assert decoded.mode == "RGB"
    assert decoded.size == (2, 3)
    decoded.close()
    with pytest.raises(ValueError, match="media type is unsupported"):
        provider_module._decode_image(payload, media_type="image/gif")
    with pytest.raises(ValueError, match="could not be decoded safely"):
        provider_module._decode_image(payload, media_type="image/jpeg")


def test_image_decode_requires_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> object:
        raise ImportError("Pillow unavailable")

    monkeypatch.setattr(provider_module.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="Pillow is required"):
        provider_module._decode_image(b"image", media_type="image/png")


@pytest.mark.parametrize(
    ("parts", "message"),
    [
        ((), "structured input_parts"),
        ((SimpleNamespace(kind="text", role="prompt", text="p"),), "exactly one"),
        (
            (
                SimpleNamespace(
                    kind="text",
                    role="prompt",
                    text="",
                    sha256=hashlib.sha256(b"").hexdigest(),
                ),
                SimpleNamespace(kind="content", role="image"),
            ),
            "prompt must be non-empty",
        ),
        (
            (
                SimpleNamespace(
                    kind="text", role="prompt", text="prompt", sha256="0" * 64
                ),
                SimpleNamespace(kind="content", role="image"),
            ),
            "prompt digest does not match",
        ),
    ],
)
def test_record_material_rejects_noncanonical_structured_inputs(
    parts: tuple[object, ...], message: str
) -> None:
    record = SimpleNamespace(input_parts=parts, input_sha256="1" * 64)
    with pytest.raises(ValueError, match=message):
        provider_module._record_material(record)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (
            ModelRuntimeSpec(
                provider_name="hf_transformers",
                model_id="org/model",
                settings=_spec().settings,
            ),
            "provider_name",
        ),
        (
            ModelRuntimeSpec(
                provider_name="hf_vision_text",
                model_id="org/model",
                settings={
                    key: value
                    for key, value in _spec().settings.items()
                    if key != "timeout_seconds"
                },
            ),
            "missing",
        ),
        (_spec(checkpoint_tree_sha256=7), "sha256 digest"),
        (_spec(immutable_revision=" bad "), "trimmed text"),
        (_spec(context_length=True), "positive integer"),
        (_spec(context_length=0), "positive integer"),
        (_spec(seed=-1), "non-negative integer"),
    ],
)
def test_provider_config_failures_identify_the_invalid_contract(
    spec: ModelRuntimeSpec, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        HFVisionTextProvider().validate_config(spec)


def test_model_and_module_state_helpers_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="execution tensors"):
        provider_module._model_device(object())
    with pytest.raises(RuntimeError, match="has no execution tensors"):
        provider_module._model_device(
            SimpleNamespace(parameters=lambda: (), buffers=lambda: ())
        )
    with pytest.raises(RuntimeError, match="module state"):
        provider_module._require_eval_mode(object())
    with pytest.raises(RuntimeError, match="evaluation mode"):
        provider_module._require_eval_mode(
            SimpleNamespace(modules=lambda: (SimpleNamespace(training=True),))
        )

    module_file = tmp_path / "backend.py"
    module_file.write_bytes(b"backend identity\n")
    assert (
        provider_module._module_file_sha256(
            SimpleNamespace(__file__=str(module_file)), label="test"
        )
        == hashlib.sha256(module_file.read_bytes()).hexdigest()
    )
    with pytest.raises(RuntimeError, match="identity is unavailable"):
        provider_module._module_file_sha256(object(), label="test")


def test_backend_identity_binds_each_runtime_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modules: dict[str, object] = {}
    for name, version in (
        ("transformers", "5.14.1"),
        ("torch", "2.11.0"),
        ("PIL", "12.3.0"),
    ):
        module_path = tmp_path / f"{name}.py"
        module_path.write_text(name + "\n", encoding="utf-8")
        modules[name] = SimpleNamespace(__file__=str(module_path), __version__=version)
    monkeypatch.setattr(
        provider_module.importlib, "import_module", lambda name: modules[name]
    )

    identity = provider_module._backend_identity()

    assert identity.name == "huggingface-vision-text"
    assert identity.version == "pillow=12.3.0;torch=2.11.0;transformers=5.14.1"
    assert identity.binary_sha256 is not None

    modules["PIL"].__version__ = ""  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="versions are unavailable"):
        provider_module._backend_identity()


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("network", "offline"),
        ("remote", "remote code"),
        ("boundary", "container boundary"),
        ("digest", "image digest"),
        ("reference", "image reference"),
    ],
)
def test_runtime_boundary_rejects_each_untrusted_authority(
    failure: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=failure == "network",
        container_image_digest=_IMAGE_DIGEST,
        device_kind="cpu",
    )
    monkeypatch.setattr(provider_module, "network_allowed", lambda: False)
    monkeypatch.setattr(
        provider_module, "remote_code_allowed", lambda: failure == "remote"
    )
    monkeypatch.setattr(provider_module, "third_party_plugins_allowed", lambda: False)
    monkeypatch.setattr(
        provider_module,
        "strict_container_boundary_present",
        lambda: failure != "boundary",
    )
    monkeypatch.setattr(
        provider_module,
        "resolve_runtime_image_digest",
        lambda: "sha256:" + "8" * 64 if failure == "digest" else _IMAGE_DIGEST,
    )
    monkeypatch.setattr(
        provider_module,
        "resolve_runtime_image",
        lambda: (
            "registry.invalid/runtime:mutable"
            if failure == "reference"
            else f"registry.invalid/runtime@{_IMAGE_DIGEST}"
        ),
    )

    with pytest.raises(ValueError, match=message):
        provider_module._require_runtime_boundary(context)


def test_vision_text_model_loader_prefers_multimodal_auto_model() -> None:
    multimodal_loader = SimpleNamespace(from_pretrained=lambda: None)
    image_text_loader = SimpleNamespace(from_pretrained=lambda: None)
    vision_seq_loader = SimpleNamespace(from_pretrained=lambda: None)
    transformers = SimpleNamespace(
        AutoModelForMultimodalLM=multimodal_loader,
        AutoModelForImageTextToText=image_text_loader,
        AutoModelForVision2Seq=vision_seq_loader,
    )

    selected = provider_module._resolve_vision_text_model_loader(transformers)

    assert selected is multimodal_loader.from_pretrained


@pytest.mark.parametrize(
    ("transformers", "expected"),
    [
        pytest.param(
            SimpleNamespace(
                AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=None),
                AutoModelForImageTextToText=SimpleNamespace(
                    from_pretrained=lambda: "image-text"
                ),
                AutoModelForVision2Seq=SimpleNamespace(
                    from_pretrained=lambda: "vision-seq"
                ),
            ),
            "image-text",
            id="image-text-fallback",
        ),
        pytest.param(
            SimpleNamespace(
                AutoModelForVision2Seq=SimpleNamespace(
                    from_pretrained=lambda: "vision-seq"
                )
            ),
            "vision-seq",
            id="vision-seq-fallback",
        ),
    ],
)
def test_vision_text_model_loader_uses_available_fallback(
    transformers: SimpleNamespace, expected: str
) -> None:
    selected = provider_module._resolve_vision_text_model_loader(transformers)

    assert selected() == expected


def test_vision_text_model_loader_failure_names_supported_apis() -> None:
    with pytest.raises(
        RuntimeError,
        match=(
            "AutoModelForMultimodalLM.*AutoModelForImageTextToText"
            ".*AutoModelForVision2Seq"
        ),
    ):
        provider_module._resolve_vision_text_model_loader(SimpleNamespace())


def test_prepare_open_score_receipt_and_close_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    checkpoint.joinpath("config.json").write_text(
        '{"model_type":"vision-test"}\n', encoding="utf-8"
    )
    content = tmp_path / "images"
    content.mkdir()
    processor = _Processor()
    model = _Model()
    tree_digest = checkpoint_tree_sha256(checkpoint).removeprefix("sha256:")
    processor_digest = processor_contract_sha256(processor)
    spec = _spec(
        checkpoint_tree_sha256=tree_digest,
        processor_metadata_sha256=processor_digest,
    )
    observed_loads: list[tuple[str, dict[str, object]]] = []

    def load_processor(path: str, **kwargs: object) -> _Processor:
        observed_loads.append((path, kwargs))
        return processor

    def load_model(path: str, **kwargs: object) -> _Model:
        observed_loads.append((path, kwargs))
        return model, {
            "missing_keys": set(),
            "unexpected_keys": set(),
            "mismatched_keys": set(),
            "error_msgs": [],
        }

    transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=load_model),
    )
    real_import = importlib.import_module
    monkeypatch.setattr(
        provider_module.importlib,
        "import_module",
        lambda name: transformers if name == "transformers" else real_import(name),
    )
    monkeypatch.setattr(
        provider_module, "hf_tokenizer_contract_sha256", lambda _t: "d" * 64
    )
    bindings: list[Path] = []
    monkeypatch.setattr(
        provider_module,
        "require_loaded_hf_checkpoint_binding",
        lambda **values: bindings.append(values["checkpoint"]),
    )
    resources = RuntimeArtifactResources(
        root=tmp_path,
        primary_artifact=checkpoint.name,
        support_resources={"content_store": content.name},
        device_kind="cpu",
        container_image_digest=_IMAGE_DIGEST,
    )
    provider = HFVisionTextProvider()

    context = provider.prepare_execution(spec, resources)

    assert model.moves == ["cpu"]
    assert model.training is False
    assert len(observed_loads) == 2
    assert all(load[1]["local_files_only"] is True for load in observed_loads)
    assert observed_loads[1][1]["output_loading_info"] is True
    assert context.scorer is not None
    assert context.provider_state == (model, processor, checkpoint)

    monkeypatch.setattr(provider_module, "network_allowed", lambda: False)
    monkeypatch.setattr(provider_module, "remote_code_allowed", lambda: False)
    monkeypatch.setattr(provider_module, "third_party_plugins_allowed", lambda: False)
    monkeypatch.setattr(
        provider_module, "strict_container_boundary_present", lambda: True
    )
    monkeypatch.setattr(
        provider_module, "resolve_runtime_image_digest", lambda: _IMAGE_DIGEST
    )
    monkeypatch.setattr(
        provider_module,
        "resolve_runtime_image",
        lambda: f"registry.invalid/runtime@{_IMAGE_DIGEST}",
    )
    monkeypatch.setattr(
        provider_module,
        "_backend_identity",
        lambda: RuntimeBackendIdentity(
            name="vision-backend",
            version="1",
            source_sha256="1" * 64,
            binary_sha256=None,
            build_sha256=None,
        ),
    )
    monkeypatch.setattr(
        provider_module,
        "_device_facts",
        lambda _model, *, expected_kind: RuntimeDeviceFacts(
            device_kind=expected_kind, device_name="test-cpu"
        ),
    )
    monkeypatch.setattr(
        HFVisionTextScorer,
        "__call__",
        lambda self, batch, _settings: _observation(
            batch, self.artifact_identity_sha256
        ),
    )

    session = provider.open(spec, context)
    with pytest.raises(RuntimeError, match="unavailable before scoring"):
        session.runtime_receipt()

    observation = session.score(_batch())
    receipt = session.runtime_receipt()

    assert len(bindings) == 3
    assert receipt.outer_image_digest == _IMAGE_DIGEST
    assert (
        receipt.scoring_observation_sha256
        == hashlib.sha256(encode_scoring_observation(observation)).hexdigest()
    )
    assert receipt.plugin.distribution == "invarlock-runtime-hf-vision-text"

    session.close()
    with pytest.raises(RuntimeError, match="session is closed"):
        session.score(_batch())
    with pytest.raises(RuntimeError, match="session is closed"):
        session.runtime_receipt()


def test_prepare_execution_rejects_missing_checkpoint(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="could not be authenticated"):
        HFVisionTextProvider().authenticate_artifact(_spec(), missing)


@pytest.mark.parametrize(
    ("batch", "settings", "message"),
    [
        (
            EvaluationBatch(
                schedule_sha256="a" * 64,
                records=_batch().records,
                task="text_causal",
            ),
            RuntimeExecutionSettings(0, 8, 1, 1, 1),
            "vision_text_generation",
        ),
        (
            EvaluationBatch(
                schedule_sha256="a" * 64,
                records=_batch().records,
                task="vision_text_generation",
                metric="normalized_nll_per_utf8_byte",
            ),
            RuntimeExecutionSettings(0, 8, 1, 1, 1),
            "exact_match only",
        ),
        (
            _batch(),
            RuntimeExecutionSettings(0, 8, 2, 1, 1),
            "batch_size=1",
        ),
    ],
)
def test_scorer_rejects_unsupported_task_metric_and_batching(
    batch: EvaluationBatch,
    settings: RuntimeExecutionSettings,
    message: str,
    tmp_path: Path,
) -> None:
    scorer = HFVisionTextScorer(
        model=object(),
        processor=object(),
        content_store=tmp_path,
        artifact_identity_sha256="e" * 64,
    )
    with pytest.raises(ValueError, match=message):
        scorer(batch, settings)


def test_scorer_constructor_rejects_unbound_identity_and_store(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="artifact_identity_sha256"):
        HFVisionTextScorer(object(), object(), tmp_path, "bad")
    with pytest.raises(ValueError, match="content_store must be absolute"):
        HFVisionTextScorer(object(), object(), Path("relative"), "e" * 64)


def test_deadline_criterion_fails_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    criterion = provider_module._deadline_criterion(object, deadline=10.0)
    monkeypatch.setattr(provider_module.time, "monotonic", lambda: 9.0)
    assert criterion() is False
    monkeypatch.setattr(provider_module.time, "monotonic", lambda: 10.0)
    with pytest.raises(TimeoutError, match="timed out"):
        criterion()
