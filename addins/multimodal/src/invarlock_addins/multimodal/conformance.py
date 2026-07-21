"""Machine-readable ABI conformance check for the vision-text add-in."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import tempfile
from pathlib import Path
from typing import cast

from invarlock.core.runtime_provider import (
    INVARLOCK_RUNTIME_PROVIDER_ABI,
    EvaluationInputPart,
    ModelRuntimeSpec,
    RuntimeArtifactResources,
    RuntimeProvider,
    build_runtime_behavioral_schedule_from_material,
)

from .provider import HFVisionTextProvider


def _exercise_host_input_preflight(provider: HFVisionTextProvider) -> None:
    """Prove the base install can decode authenticated media without model code."""

    image_module = importlib.import_module("PIL.Image")
    encoded = io.BytesIO()
    image = image_module.new("RGB", (1, 1), color=(255, 0, 0))
    image.save(encoded, format="PNG")
    image.close()
    payload = encoded.getvalue()
    prompt = "Describe the image."
    parts = (
        EvaluationInputPart(
            kind="content",
            role="image",
            content_id="conformance_png",
            media_type="image/png",
            byte_length=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
        EvaluationInputPart(
            kind="text",
            role="prompt",
            text=prompt,
            sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        ),
    )
    schedule = build_runtime_behavioral_schedule_from_material(
        dataset_identity={
            "provider": "local",
            "dataset_name": None,
            "config_name": None,
            "revision": None,
            "split": "conformance",
        },
        records=[
            {
                "record_id": "conformance/1",
                "input_parts": [part.to_payload() for part in parts],
                "expected_output": "red",
            }
        ],
        task="vision_text_generation",
    )
    spec = ModelRuntimeSpec(
        provider_name="hf_vision_text",
        model_id="local/conformance",
        settings={
            "batch_size": 1,
            "checkpoint_tree_sha256": "a" * 64,
            "context_length": 32,
            "max_output_tokens": 4,
            "offline": True,
            "processor_metadata_sha256": "b" * 64,
            "seed": 0,
            "timeout_seconds": 30,
            "tokenizer_metadata_sha256": "c" * 64,
        },
    )
    with tempfile.TemporaryDirectory(prefix="invarlock-vision-conformance-") as root:
        resources_root = Path(root)
        resources_root.joinpath("checkpoint").mkdir()
        content_store = resources_root / "images"
        content_store.mkdir()
        content_store.joinpath("conformance_png").write_bytes(payload)
        resources = RuntimeArtifactResources(
            root=resources_root,
            primary_artifact="checkpoint",
            support_resources={"content_store": "images"},
            device_kind="cpu",
            container_image_digest="sha256:" + "d" * 64,
        )
        provider.validate_evaluation_inputs(spec, resources, schedule)


def conformance_payload() -> dict[str, object]:
    """Return provider identity and ABI conformance without loading backends."""

    candidate: object = HFVisionTextProvider()
    errors: list[str] = []
    if not isinstance(candidate, RuntimeProvider):
        errors.append("provider does not implement RuntimeProvider")
        return {
            "abi_version": None,
            "errors": errors,
            "format_version": "invarlock/runtime-provider-conformance-v1",
            "ok": False,
            "provider": None,
        }
    provider = cast(HFVisionTextProvider, candidate)
    if provider.name != "hf_vision_text":
        errors.append("provider name must be hf_vision_text")
    if provider.abi_version != INVARLOCK_RUNTIME_PROVIDER_ABI:
        errors.append("provider ABI does not match the installed core")
    capabilities = provider.capabilities()
    if capabilities.tasks != ("vision_text_generation",):
        errors.append("provider must expose only vision_text_generation")
    if capabilities.metrics != ("exact_match",):
        errors.append("provider must expose only exact_match")
    try:
        _exercise_host_input_preflight(provider)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"host input preflight failed: {exc}")
    return {
        "abi_version": provider.abi_version,
        "errors": errors,
        "format_version": "invarlock/runtime-provider-conformance-v1",
        "ok": not errors,
        "provider": provider.name,
    }


def main() -> int:
    payload = conformance_payload()
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0 if payload["ok"] else 1


__all__ = ["conformance_payload", "main"]


if __name__ == "__main__":  # pragma: no cover - package smoke
    raise SystemExit(main())
