from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import invarlock_addins.multimodal.provider as provider_module
import pytest
from invarlock_addins.multimodal.provider import (
    HFVisionTextProvider,
    HFVisionTextScorer,
    _read_content_bytes,
)

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import (
    EvaluationBatch,
    EvaluationInputPart,
    EvaluationRecord,
    ModelRuntimeSpec,
    RuntimeArtifactResources,
    RuntimeBehavioralSchedule,
    RuntimeExecutionSettings,
    build_runtime_behavioral_schedule_from_material,
    evaluation_input_parts_sha256,
)
from invarlock.core.runtime_provider.behavioral_observation import (
    verify_runtime_behavioral_observation,
)
from invarlock.runtime_provider_evidence import encode_scoring_observation

_DIGEST = "a" * 64


def _png_bytes() -> bytes:
    image_module = importlib.import_module("PIL.Image")
    output = io.BytesIO()
    image = image_module.new("RGB", (2, 2), color=(255, 0, 0))
    image.save(output, format="PNG")
    image.close()
    return output.getvalue()


def _vision_schedule(
    payload: bytes,
    *,
    content_id: str = "image_001",
    media_type: str = "image/png",
) -> RuntimeBehavioralSchedule:
    return _vision_schedule_for_bindings(((content_id, media_type, payload),))


def _vision_schedule_for_bindings(
    bindings: tuple[tuple[str, str, bytes], ...],
) -> RuntimeBehavioralSchedule:
    prompt = "What animal is shown?"
    records: list[dict[str, object]] = []
    for index, (content_id, media_type, payload) in enumerate(bindings, start=1):
        parts = (
            EvaluationInputPart(
                kind="content",
                role="image",
                content_id=content_id,
                media_type=media_type,
                byte_length=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            ),
            EvaluationInputPart(
                kind="text",
                role="prompt",
                text=prompt,
                sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            ),
        )
        records.append(
            {
                "record_id": f"vision/{index}",
                "input_parts": [part.to_payload() for part in parts],
                "expected_output": "cat",
            }
        )
    return build_runtime_behavioral_schedule_from_material(
        dataset_identity={
            "provider": "local",
            "dataset_name": None,
            "config_name": None,
            "revision": None,
            "split": "qualification",
        },
        records=records,
        task="vision_text_generation",
    )


def _spec(**overrides: object) -> ModelRuntimeSpec:
    settings: dict[str, object] = {
        "batch_size": 1,
        "checkpoint_tree_sha256": "b" * 64,
        "context_length": 512,
        "max_output_tokens": 32,
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


def test_provider_declares_only_implemented_behavior() -> None:
    provider = HFVisionTextProvider()

    assert provider.capabilities().tasks == ("vision_text_generation",)
    assert provider.capabilities().metrics == ("exact_match",)
    assert provider.capabilities().execution_modes == ("container",)
    assert provider.identify_artifact(_spec()).checkpoint_tree_sha256 == "b" * 64


def test_total_pixel_ceiling_admits_a_diverse_400_record_suite() -> None:
    assert provider_module._MAX_TOTAL_IMAGE_PIXELS >= 400 * 4_000_000
    assert provider_module._MAX_TOTAL_IMAGE_PIXELS <= 2_000_000_000


def test_provider_authenticates_checkpoint_without_loading_model(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = checkpoint / "config.json"
    config.write_text('{"model_type":"vision-test"}\n', encoding="utf-8")
    digest = checkpoint_tree_sha256(checkpoint).removeprefix("sha256:")
    spec = _spec(checkpoint_tree_sha256=digest)
    provider = HFVisionTextProvider()

    assert provider.authenticate_artifact(spec, checkpoint) == (
        provider.identify_artifact(spec)
    )

    config.write_text('{"model_type":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="tree digest does not match"):
        provider.authenticate_artifact(spec, checkpoint)


def test_cuda_device_facts_allow_torch_without_private_driver_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = SimpleNamespace(type="cuda", index=0)
    model = SimpleNamespace(
        parameters=lambda: iter((SimpleNamespace(device=device),)),
        buffers=lambda: iter(()),
    )
    torch = SimpleNamespace(
        _C=SimpleNamespace(),
        cuda=SimpleNamespace(
            get_device_name=lambda _index: "Test GPU",
            get_device_capability=lambda _index: (9, 0),
        ),
        version=SimpleNamespace(cuda="12.8"),
    )
    real_import = provider_module.importlib.import_module
    monkeypatch.setattr(
        provider_module.importlib,
        "import_module",
        lambda name: torch if name == "torch" else real_import(name),
    )

    facts = provider_module._device_facts(model, expected_kind="cuda")

    assert facts.device_name == "Test GPU"
    assert facts.compute_capability == "9.0"
    assert facts.driver_version is None
    assert facts.cuda_runtime_version == "12.8"


def test_device_facts_reject_wrong_device_and_describe_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = SimpleNamespace(type="cpu", index=None)
    model = SimpleNamespace(
        parameters=lambda: iter((SimpleNamespace(device=device),)),
        buffers=lambda: iter(()),
    )
    with pytest.raises(ValueError, match="does not match selected device"):
        provider_module._device_facts(model, expected_kind="cuda")

    monkeypatch.setattr(provider_module.platform, "processor", lambda: "Test CPU")
    facts = provider_module._device_facts(model, expected_kind="cpu")
    assert facts.device_kind == "cpu"
    assert facts.device_name == "Test CPU"


def test_module_identity_rejects_missing_nonregular_and_changed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RuntimeError, match="module identity is unavailable"):
        provider_module._module_file_sha256(  # noqa: SLF001
            SimpleNamespace(__file__=str(tmp_path / "missing")),
            label="test",
        )
    with pytest.raises(RuntimeError, match="module identity is unavailable"):
        provider_module._module_file_sha256(  # noqa: SLF001
            SimpleNamespace(__file__=str(tmp_path)),
            label="test",
        )

    module_file = tmp_path / "module.py"
    module_file.write_bytes(b"module")
    identities = iter(((1,), (2,)))
    monkeypatch.setattr(
        provider_module, "_stat_identity", lambda _value: next(identities)
    )
    with pytest.raises(RuntimeError, match="changed while being identified"):
        provider_module._module_file_sha256(  # noqa: SLF001
            SimpleNamespace(__file__=str(module_file)),
            label="test",
        )


def test_vision_session_rejects_missing_binding_and_closed_use() -> None:
    session = provider_module._VisionTextSession(  # noqa: SLF001
        scorer=lambda *_args: None,  # type: ignore[arg-type]
        provenance=SimpleNamespace(
            execution_settings=RuntimeExecutionSettings(
                seed=1,
                context_length=32,
                batch_size=1,
                max_output_tokens=8,
                timeout_seconds=2,
            )
        ),  # type: ignore[arg-type]
        binding_check=None,
    )
    batch = EvaluationBatch(
        schedule_sha256="a" * 64,
        records=(
            EvaluationRecord(
                record_id="record",
                input_text="prompt",
                input_sha256=hashlib.sha256(b"prompt").hexdigest(),
            ),
        ),
        task="vision_text_generation",
    )
    with pytest.raises(RuntimeError, match="binding check is unavailable"):
        session.score(batch)
    with pytest.raises(RuntimeError, match="unavailable before scoring"):
        session.runtime_receipt()
    session.close()
    with pytest.raises(RuntimeError, match="session is closed"):
        session.score(batch)
    with pytest.raises(RuntimeError, match="session is closed"):
        session.runtime_receipt()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"batch_size": 2}, "batch_size=1"),
        ({"offline": False}, "offline=true"),
        ({"processor_metadata_sha256": "bad"}, "sha256 digest"),
        ({"unexpected": True}, "unsupported"),
    ],
)
def test_config_rejects_unimplemented_or_ambiguous_behavior(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        HFVisionTextProvider().validate_config(_spec(**change))


def test_content_store_rechecks_length_and_digest(tmp_path: Path) -> None:
    payload = _png_bytes()
    tmp_path.joinpath("image_001").write_bytes(payload)

    assert (
        _read_content_bytes(
            tmp_path,
            content_id="image_001",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_byte_length=len(payload),
        )
        == payload
    )
    with pytest.raises(ValueError, match="identity does not match"):
        _read_content_bytes(
            tmp_path,
            content_id="image_001",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_byte_length=len(payload) + 1,
        )
    with pytest.raises(ValueError, match="digest does not match"):
        _read_content_bytes(
            tmp_path,
            content_id="image_001",
            expected_sha256=_DIGEST,
            expected_byte_length=len(payload),
        )


@pytest.mark.parametrize(
    ("content_id", "digest", "byte_length", "message"),
    [
        (".", _DIGEST, 1, "safe basename"),
        ("image", "BAD", 1, "lowercase sha256"),
        ("image", _DIGEST, True, "outside the supported range"),
        ("image", _DIGEST, 0, "outside the supported range"),
    ],
)
def test_content_store_rejects_invalid_bindings_before_open(
    tmp_path: Path,
    content_id: str,
    digest: str,
    byte_length: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _read_content_bytes(
            tmp_path,
            content_id=content_id,
            expected_sha256=digest,
            expected_byte_length=byte_length,  # type: ignore[arg-type]
        )


def test_content_store_reports_missing_store_and_object(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="content store is unavailable"):
        _read_content_bytes(
            tmp_path / "missing",
            content_id="image",
            expected_sha256=_DIGEST,
            expected_byte_length=1,
        )

    with pytest.raises(ValueError, match="content object is unavailable"):
        _read_content_bytes(
            tmp_path,
            content_id="missing",
            expected_sha256=_DIGEST,
            expected_byte_length=1,
        )


def test_content_store_detects_truncation_and_growth_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"abc"
    tmp_path.joinpath("image").write_bytes(payload)
    real_read = provider_module.os.read

    monkeypatch.setattr(provider_module.os, "read", lambda _fd, _size: b"")
    with pytest.raises(ValueError, match="changed while being read"):
        _read_content_bytes(
            tmp_path,
            content_id="image",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_byte_length=len(payload),
        )

    calls = 0

    def growing_read(descriptor: int, size: int) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            return b"x"
        return real_read(descriptor, size)

    monkeypatch.setattr(provider_module.os, "read", growing_read)
    with pytest.raises(ValueError, match="changed while being read"):
        _read_content_bytes(
            tmp_path,
            content_id="image",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_byte_length=len(payload),
        )


def test_optional_text_rejects_untrimmed_or_empty_values() -> None:
    for value in ("", " untrimmed"):
        with pytest.raises(ValueError, match="non-empty trimmed text"):
            provider_module._optional_text({"revision": value}, "revision")
    assert provider_module._optional_text({"revision": "stable"}, "revision") == (
        "stable"
    )


def test_schedule_content_rejects_wrong_task(tmp_path: Path) -> None:
    schedule = SimpleNamespace(task="text_causal", records=())
    with pytest.raises(ValueError, match="requires vision_text_generation"):
        provider_module._validate_schedule_content(  # type: ignore[arg-type]
            schedule,
            content_store=tmp_path,
        )


def _unchecked_vision_record(**image_changes: object) -> object:
    prompt = "prompt"
    image = SimpleNamespace(
        kind="content",
        role="image",
        content_id="image",
        media_type="image/png",
        byte_length=3,
        sha256=hashlib.sha256(b"abc").hexdigest(),
    )
    for name, value in image_changes.items():
        setattr(image, name, value)
    prompt_part = SimpleNamespace(
        kind="text",
        role="prompt",
        text=prompt,
        sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    )
    return SimpleNamespace(
        record_id="record",
        input_text=prompt,
        input_sha256="a" * 64,
        input_parts=(image, prompt_part),
    )


def test_schedule_content_rejects_incomplete_and_unclosable_decodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    malformed = SimpleNamespace(
        task="vision_text_generation",
        records=(_unchecked_vision_record(byte_length=True),),
    )
    with pytest.raises(ValueError, match="binding is incomplete"):
        provider_module._validate_schedule_content(  # type: ignore[arg-type]
            malformed,
            content_store=tmp_path,
        )

    monkeypatch.setattr(
        provider_module, "_read_content_bytes", lambda *_a, **_k: b"abc"
    )
    monkeypatch.setattr(
        provider_module,
        "_decode_image",
        lambda *_a, **_k: SimpleNamespace(size=(1, 1)),
    )
    valid = SimpleNamespace(
        task="vision_text_generation",
        records=(_unchecked_vision_record(),),
    )
    with pytest.raises(ValueError, match="cannot be safely closed"):
        provider_module._validate_schedule_content(  # type: ignore[arg-type]
            valid,
            content_store=tmp_path,
        )


def test_schedule_content_rejects_invalid_decoded_dimensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []
    monkeypatch.setattr(
        provider_module, "_read_content_bytes", lambda *_a, **_k: b"abc"
    )
    monkeypatch.setattr(
        provider_module,
        "_decode_image",
        lambda *_a, **_k: SimpleNamespace(
            size=(True, 1), close=lambda: closed.append(True)
        ),
    )
    schedule = SimpleNamespace(
        task="vision_text_generation",
        records=(_unchecked_vision_record(),),
    )
    with pytest.raises(ValueError, match="dimensions are invalid"):
        provider_module._validate_schedule_content(  # type: ignore[arg-type]
            schedule,
            content_store=tmp_path,
        )
    assert closed == [True]


def test_input_preflight_authenticates_schedule_content_without_loading_model(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    content_store = tmp_path / "images"
    content_store.mkdir()
    payload = _png_bytes()
    content_store.joinpath("image_001").write_bytes(payload)
    schedule = _vision_schedule(payload)
    resources = RuntimeArtifactResources(
        root=tmp_path,
        primary_artifact="checkpoint",
        support_resources={"content_store": "images"},
        device_kind="cuda",
        container_image_digest="sha256:" + "9" * 64,
    )
    provider = HFVisionTextProvider()

    provider.validate_evaluation_inputs(_spec(), resources, schedule)

    content_store.joinpath("image_001").unlink()
    with pytest.raises(ValueError, match="content object is unavailable"):
        provider.validate_evaluation_inputs(_spec(), resources, schedule)


@pytest.mark.parametrize(
    ("payload", "media_type", "message"),
    [
        (b"not an image", "image/png", "could not be decoded safely"),
        (None, "image/jpeg", "could not be decoded safely"),
        (None, "image/gif", "media type is unsupported"),
    ],
)
def test_input_preflight_rejects_malformed_or_mislabeled_media_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes | None,
    media_type: str,
    message: str,
) -> None:
    observed_imports: list[str] = []
    real_import = provider_module.importlib.import_module

    def record_import(name: str, package: str | None = None) -> object:
        observed_imports.append(name)
        return real_import(name, package)

    monkeypatch.setattr(provider_module.importlib, "import_module", record_import)
    actual_payload = _png_bytes() if payload is None else payload
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    content_store = tmp_path / "images"
    content_store.mkdir()
    content_store.joinpath("image_001").write_bytes(actual_payload)
    resources = RuntimeArtifactResources(
        root=tmp_path,
        primary_artifact="checkpoint",
        support_resources={"content_store": "images"},
        device_kind="cuda",
        container_image_digest="sha256:" + "9" * 64,
    )

    with pytest.raises(ValueError, match=message):
        HFVisionTextProvider().validate_evaluation_inputs(
            _spec(), resources, _vision_schedule(actual_payload, media_type=media_type)
        )

    assert "transformers" not in observed_imports
    assert "torch" not in observed_imports


@pytest.mark.parametrize(
    ("limit_name", "limit", "message"),
    [
        ("_MAX_UNIQUE_IMAGES", 0, "unique image limit"),
        ("_MAX_TOTAL_IMAGE_BYTES", 1, "total image byte limit"),
        ("_MAX_TOTAL_IMAGE_PIXELS", 1, "total decoded pixel limit"),
    ],
)
def test_input_preflight_enforces_aggregate_media_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit: int,
    message: str,
) -> None:
    payload = _png_bytes()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    content_store = tmp_path / "images"
    content_store.mkdir()
    content_store.joinpath("image_001").write_bytes(payload)
    resources = RuntimeArtifactResources(
        root=tmp_path,
        primary_artifact="checkpoint",
        support_resources={"content_store": "images"},
        device_kind="cuda",
        container_image_digest="sha256:" + "9" * 64,
    )
    monkeypatch.setattr(provider_module, limit_name, limit)

    with pytest.raises(ValueError, match=message):
        HFVisionTextProvider().validate_evaluation_inputs(
            _spec(), resources, _vision_schedule(payload)
        )


def test_input_preflight_deduplicates_identical_bindings_and_counts_unique_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _png_bytes()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    content_store = tmp_path / "images"
    content_store.mkdir()
    content_store.joinpath("image_001").write_bytes(payload)
    content_store.joinpath("image_002").write_bytes(payload)
    resources = RuntimeArtifactResources(
        root=tmp_path,
        primary_artifact="checkpoint",
        support_resources={"content_store": "images"},
        device_kind="cuda",
        container_image_digest="sha256:" + "9" * 64,
    )
    provider = HFVisionTextProvider()
    monkeypatch.setattr(provider_module, "_MAX_TOTAL_IMAGE_BYTES", len(payload))

    provider.validate_evaluation_inputs(
        _spec(),
        resources,
        _vision_schedule_for_bindings(
            (
                ("image_001", "image/png", payload),
                ("image_001", "image/png", payload),
            )
        ),
    )

    with pytest.raises(ValueError, match="total image byte limit"):
        provider.validate_evaluation_inputs(
            _spec(),
            resources,
            _vision_schedule_for_bindings(
                (
                    ("image_001", "image/png", payload),
                    ("image_002", "image/png", payload),
                )
            ),
        )


def test_input_preflight_enforces_per_image_pixel_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _png_bytes()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    content_store = tmp_path / "images"
    content_store.mkdir()
    content_store.joinpath("image_001").write_bytes(payload)
    resources = RuntimeArtifactResources(
        root=tmp_path,
        primary_artifact="checkpoint",
        support_resources={"content_store": "images"},
        device_kind="cuda",
        container_image_digest="sha256:" + "9" * 64,
    )
    monkeypatch.setattr(provider_module, "_MAX_IMAGE_PIXELS", 1)

    with pytest.raises(ValueError, match="could not be decoded safely"):
        HFVisionTextProvider().validate_evaluation_inputs(
            _spec(), resources, _vision_schedule(payload)
        )


def test_input_preflight_rejects_conflicting_duplicate_content_bindings(
    tmp_path: Path,
) -> None:
    payload = _png_bytes()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    content_store = tmp_path / "images"
    content_store.mkdir()
    content_store.joinpath("image_001").write_bytes(payload)
    resources = RuntimeArtifactResources(
        root=tmp_path,
        primary_artifact="checkpoint",
        support_resources={"content_store": "images"},
        device_kind="cuda",
        container_image_digest="sha256:" + "9" * 64,
    )
    schedule = _vision_schedule_for_bindings(
        (
            ("image_001", "image/png", payload),
            ("image_001", "image/jpeg", payload),
        )
    )

    with pytest.raises(ValueError, match="conflicting authenticated bindings"):
        HFVisionTextProvider().validate_evaluation_inputs(_spec(), resources, schedule)


def test_input_preflight_rejects_non_regular_content_object(tmp_path: Path) -> None:
    payload = _png_bytes()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    content_store = tmp_path / "images"
    content_store.mkdir()
    content_store.joinpath("image_001").mkdir()
    resources = RuntimeArtifactResources(
        root=tmp_path,
        primary_artifact="checkpoint",
        support_resources={"content_store": "images"},
        device_kind="cuda",
        container_image_digest="sha256:" + "9" * 64,
    )

    with pytest.raises(ValueError, match="content object identity does not match"):
        HFVisionTextProvider().validate_evaluation_inputs(
            _spec(), resources, _vision_schedule(payload)
        )


def test_input_preflight_rejects_animated_media(tmp_path: Path) -> None:
    image_module = importlib.import_module("PIL.Image")
    output = io.BytesIO()
    first = image_module.new("RGB", (2, 2), color=(255, 0, 0))
    second = image_module.new("RGB", (2, 2), color=(0, 0, 255))
    first.save(
        output,
        format="WEBP",
        save_all=True,
        append_images=[second],
        duration=100,
        loop=0,
    )
    first.close()
    second.close()
    payload = output.getvalue()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    content_store = tmp_path / "images"
    content_store.mkdir()
    content_store.joinpath("image_001").write_bytes(payload)
    resources = RuntimeArtifactResources(
        root=tmp_path,
        primary_artifact="checkpoint",
        support_resources={"content_store": "images"},
        device_kind="cuda",
        container_image_digest="sha256:" + "9" * 64,
    )

    with pytest.raises(ValueError, match="could not be decoded safely"):
        HFVisionTextProvider().validate_evaluation_inputs(
            _spec(), resources, _vision_schedule(payload, media_type="image/webp")
        )


def test_image_decode_fails_closed_if_format_changes_between_validation_and_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Candidate:
        n_frames = 1
        size = (2, 2)

        def __init__(self, image_format: str) -> None:
            self.format = image_format

        def __enter__(self) -> Candidate:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def verify(self) -> None:
            return None

    images = iter((Candidate("PNG"), Candidate("JPEG")))
    fake_image_module = SimpleNamespace(
        DecompressionBombWarning=Warning,
        open=lambda _payload: next(images),
    )
    monkeypatch.setattr(
        provider_module.importlib,
        "import_module",
        lambda _name: fake_image_module,
    )

    with pytest.raises(ValueError, match="could not be decoded safely"):
        provider_module._decode_image(b"authenticated image", media_type="image/png")


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_content_store_rejects_links_and_path_syntax(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-image"
    outside.write_bytes(b"outside")
    tmp_path.joinpath("linked_image").symlink_to(outside)

    with pytest.raises(ValueError, match="unavailable"):
        _read_content_bytes(
            tmp_path,
            content_id="linked_image",
            expected_sha256=hashlib.sha256(b"outside").hexdigest(),
            expected_byte_length=len(b"outside"),
        )


def test_scorer_executes_authenticated_vision_text_exact_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    image_bytes = b"test image bytes"
    tmp_path.joinpath("image_001").write_bytes(image_bytes)
    prompt = "What animal is shown?"
    parts = (
        EvaluationInputPart(
            kind="content",
            role="image",
            content_id="image_001",
            media_type="image/png",
            byte_length=len(image_bytes),
            sha256=hashlib.sha256(image_bytes).hexdigest(),
        ),
        EvaluationInputPart(
            kind="text",
            role="prompt",
            text=prompt,
            sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        ),
    )
    record = EvaluationRecord(
        record_id="record_001",
        input_text=prompt,
        input_sha256=evaluation_input_parts_sha256(parts),
        expected_output="cat",
        input_parts=parts,
    )

    class _Image:
        closed = False

        def close(self) -> None:
            self.closed = True

    image = _Image()
    monkeypatch.setattr(
        provider_module, "_decode_image", lambda *_args, **_kwargs: image
    )

    class _Processor:
        def apply_chat_template(self, messages, **kwargs):  # noqa: ANN001
            assert messages[0]["content"][1]["text"] == prompt
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return "<image>What animal is shown?"

        def __call__(self, **kwargs):  # noqa: ANN003
            assert kwargs["images"] is image
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "pixel_values": torch.ones((1, 3, 2, 2)),
            }

        def batch_decode(self, sequences, **kwargs):  # noqa: ANN001
            assert sequences.tolist() == [[9]]
            assert kwargs["clean_up_tokenization_spaces"] is False
            return ["cat"]

    class _Module:
        training = False

    class _Model:
        def parameters(self):
            return iter((torch.ones(1),))

        def buffers(self):
            return iter(())

        def modules(self):
            return iter((_Module(),))

        def generate(self, **kwargs):  # noqa: ANN003
            assert kwargs["do_sample"] is False
            assert kwargs["max_new_tokens"] == 4
            return torch.tensor([[1, 2, 3, 9]])

    scorer = HFVisionTextScorer(
        model=_Model(),
        processor=_Processor(),
        content_store=tmp_path,
        artifact_identity_sha256="e" * 64,
    )
    batch = EvaluationBatch(
        schedule_sha256="f" * 64,
        records=(record,),
        metric="exact_match",
        task="vision_text_generation",
    )
    settings = RuntimeExecutionSettings(
        seed=1,
        context_length=128,
        batch_size=1,
        max_output_tokens=4,
        timeout_seconds=10,
    )

    observation = scorer(batch, settings)

    assert image.closed is True
    assert observation.records[0].output_text == "cat"
    result = verify_runtime_behavioral_observation(
        json.loads(encode_scoring_observation(observation)),
        expected_provider_name="hf_vision_text",
        expected_artifact_identity_sha256="e" * 64,
        expected_batch=batch,
        metric="exact_match",
    )
    assert result.metric == "exact_match"
    assert result.value == 1.0
    tmp_path.joinpath("image_001").write_bytes(image_bytes[:-1] + b"X")
    with pytest.raises(ValueError, match="content object digest does not match"):
        scorer(batch, settings)
    with pytest.raises(ValueError, match="safe basename"):
        _read_content_bytes(
            tmp_path,
            content_id="../outside-image",
            expected_sha256=hashlib.sha256(b"outside").hexdigest(),
            expected_byte_length=len(b"outside"),
        )


@pytest.mark.parametrize(
    "missing_api",
    ["stopping_criteria", "processor", "model"],
)
def test_scorer_requires_the_pinned_framework_api_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_api: str,
) -> None:
    class Context:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    fake_torch = SimpleNamespace(
        are_deterministic_algorithms_enabled=lambda: False,
        inference_mode=Context,
        is_deterministic_algorithms_warn_only_enabled=lambda: False,
        random=SimpleNamespace(fork_rng=Context),
        use_deterministic_algorithms=lambda *_args, **_kwargs: None,
    )
    stopping = SimpleNamespace(
        StoppingCriteria=None
        if missing_api == "stopping_criteria"
        else type("Stop", (), {}),
        StoppingCriteriaList=lambda values: values,
    )

    class Processor:
        def __call__(self, **_kwargs: object) -> dict[str, object]:
            return {}

        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            return "prompt"

        def batch_decode(self, *_args: object, **_kwargs: object) -> list[str]:
            return ["answer"]

    processor: object = object() if missing_api == "processor" else Processor()
    model: object = (
        object()
        if missing_api == "model"
        else SimpleNamespace(generate=lambda **_kwargs: None)
    )
    monkeypatch.setattr(
        provider_module.importlib,
        "import_module",
        lambda name: fake_torch if name == "torch" else stopping,
    )
    scorer = HFVisionTextScorer(
        model=model,
        processor=processor,
        content_store=tmp_path,
        artifact_identity_sha256="e" * 64,
    )
    record_text = "prompt"
    batch = EvaluationBatch(
        schedule_sha256="f" * 64,
        records=(
            EvaluationRecord(
                record_id="record",
                input_text=record_text,
                input_sha256=hashlib.sha256(record_text.encode()).hexdigest(),
            ),
        ),
        metric="exact_match",
        task="vision_text_generation",
    )
    settings = RuntimeExecutionSettings(
        seed=1,
        context_length=128,
        batch_size=1,
        max_output_tokens=4,
        timeout_seconds=10,
    )

    message = {
        "stopping_criteria": "stopping criteria",
        "processor": "processor APIs",
        "model": "model generation API",
    }[missing_api]
    with pytest.raises(RuntimeError, match=message):
        scorer(batch, settings)


def test_cuda_scoring_restores_deterministic_state_when_record_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class Context:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    fake_torch = SimpleNamespace(
        are_deterministic_algorithms_enabled=lambda: False,
        cuda=SimpleNamespace(
            manual_seed_all=lambda seed: calls.append(("cuda_seed", seed))
        ),
        inference_mode=Context,
        is_deterministic_algorithms_warn_only_enabled=lambda: True,
        manual_seed=lambda seed: calls.append(("seed", seed)),
        random=SimpleNamespace(fork_rng=Context),
        use_deterministic_algorithms=lambda enabled, *, warn_only: calls.append(
            ("deterministic", enabled, warn_only)
        ),
    )
    stopping = SimpleNamespace(
        StoppingCriteria=type("Stop", (), {}),
        StoppingCriteriaList=lambda values: values,
    )
    monkeypatch.setattr(
        provider_module.importlib,
        "import_module",
        lambda name: fake_torch if name == "torch" else stopping,
    )
    monkeypatch.setattr(
        provider_module,
        "_record_material",
        lambda _record: (_ for _ in ()).throw(ValueError("invalid record")),
    )

    class Processor:
        def __call__(self, **_kwargs: object) -> dict[str, object]:
            return {}

        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            return "prompt"

        def batch_decode(self, *_args: object, **_kwargs: object) -> list[str]:
            return ["answer"]

    model = SimpleNamespace(
        buffers=lambda: iter(()),
        generate=lambda **_kwargs: None,
        modules=lambda: iter((SimpleNamespace(training=False),)),
        parameters=lambda: iter(
            (SimpleNamespace(device=SimpleNamespace(type="cuda")),)
        ),
    )
    scorer = HFVisionTextScorer(
        model=model,
        processor=Processor(),
        content_store=tmp_path,
        artifact_identity_sha256="e" * 64,
    )
    settings = RuntimeExecutionSettings(
        seed=7,
        context_length=128,
        batch_size=1,
        max_output_tokens=4,
        timeout_seconds=10,
    )

    record_text = "prompt"
    batch = EvaluationBatch(
        schedule_sha256="f" * 64,
        records=(
            EvaluationRecord(
                record_id="record",
                input_text=record_text,
                input_sha256=hashlib.sha256(record_text.encode()).hexdigest(),
            ),
        ),
        metric="exact_match",
        task="vision_text_generation",
    )

    with pytest.raises(ValueError, match="invalid record"):
        scorer(batch, settings)

    assert calls == [
        ("deterministic", True, False),
        ("seed", 7),
        ("cuda_seed", 7),
        ("deterministic", False, True),
    ]
