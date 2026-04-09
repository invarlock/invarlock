from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from invarlock.eval.providers.vision_text import (
    VisionTextProvider,
    _normalize_answers,
    _resolve_image_path,
)


def _sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def test_vision_text_provider_digest_and_schedule_stable(tmp_path):
    images = []
    for index in range(1, 4):
        image_path = tmp_path / f"img-{index:03d}.bin"
        image_path.write_bytes(f"image-{index:03d}-bytes".encode())
        images.append(image_path)
    manifest = tmp_path / "vision.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"img-{index:03d}",
                    "image_path": image_path.name,
                    "prompt": f"prompt {index}",
                    "answer": f"answer {index}",
                }
            )
            for index, image_path in enumerate(images, start=1)
        ),
        encoding="utf-8",
    )
    pipeline = "resize-224-center-crop-normalize(mean=0.5,std=0.5)"

    p1 = VisionTextProvider(path=str(manifest), transform_pipeline=pipeline, seed=42)
    p2 = VisionTextProvider(path=str(manifest), transform_pipeline=pipeline, seed=42)

    # Pairing schedule must be a stable, sorted list of ids
    sched1 = p1.pairing_schedule()
    sched2 = p2.pairing_schedule()
    assert sched1 == ["img-001", "img-002", "img-003"]
    assert sched1 == sched2

    # Digest must be stable and include ids/image hashes and the transform pipeline
    d1 = p1.digest()
    d2 = p2.digest()
    assert d1 == d2
    assert d1["provider"] == "vision_text"
    assert d1["version"] >= 1
    assert d1["transform_pipeline"] == pipeline
    # ids hash is sha256 over the sorted ids
    ids_concat = "".join(["img-001", "img-002", "img-003"]).encode()
    assert d1["ids_sha256"] == _sha256_hex(ids_concat)
    # images hash is sha256 over concatenated per-image hashes in schedule order
    per_img_hashes = b"".join(
        _sha256_hex(image_path.read_bytes()).encode() for image_path in images
    )
    assert d1["images_sha256"] == _sha256_hex(per_img_hashes)
    assert isinstance(d1["prompts_sha256"], str)
    assert isinstance(d1["answers_sha256"], str)

    # Changing the pipeline should change the digest
    p3 = VisionTextProvider(
        path=str(manifest),
        transform_pipeline=pipeline + "+brightness(0.1)",
        seed=42,
    )
    d3 = p3.digest()
    assert d3["transform_pipeline"] != d1["transform_pipeline"]


def test_vision_text_provider_handles_missing_bytes():
    items = [
        {"id": "img-100", "prompt": "prompt 1", "answer": "answer 1"},
        {
            "id": "img-200",
            "prompt": "prompt 2",
            "answer": "answer 2",
            "image_bytes": b"",
        },
    ]
    provider = VisionTextProvider(items=items)
    digest = provider.digest()
    assert "seed" not in digest
    # When bytes missing, sha256 of empty bytes is used
    empty_hash = _sha256_hex(b"").encode()
    combined = _sha256_hex(empty_hash + empty_hash)
    assert digest["images_sha256"] == combined


def test_vision_text_provider_raises_for_missing_image(tmp_path):
    manifest = tmp_path / "missing.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "missing",
                "image_path": "nope.png",
                "prompt": "what is here?",
                "answer": "nothing",
            }
        ),
        encoding="utf-8",
    )

    provider = VisionTextProvider(path=str(manifest))

    with pytest.raises(Exception, match="image file is missing"):
        provider.examples()


def test_vision_text_provider_batches_and_max_samples(tmp_path):
    images = []
    for index in range(1, 4):
        image_path = tmp_path / f"img-{index:03d}.ppm"
        image_path.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
        images.append(image_path)

    manifest = tmp_path / "vision.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"img-{index:03d}",
                    "image_path": image_path.name,
                    "prompt": f"prompt {index}",
                    "answer": f"answer {index}",
                }
            )
            for index, image_path in enumerate(images, start=1)
        ),
        encoding="utf-8",
    )

    provider = VisionTextProvider(path=str(manifest), max_samples=2)

    assert provider.available_splits() == ["validation"]
    assert len(provider.examples()) == 2

    batches = list(provider.batches(seed=123, batch_size=2))
    assert batches == [
        {
            "records": [
                provider.examples()[0],
                provider.examples()[1],
            ]
        }
    ]


def test_vision_text_provider_items_override_caches_and_normalizes_bytes() -> None:
    provider = VisionTextProvider(
        items=[
            "ignore-me",
            {
                "prompt": "What?",
                "answers": [" cat ", ""],
                "image_bytes": bytearray(b"img-bytes"),
            },
        ]
    )

    examples = provider.examples()
    assert examples == provider.examples()
    assert examples[0]["id"] == "memory:2"
    assert examples[0]["answers"] == ["cat"]
    assert examples[0]["image_sha256"] == _sha256_hex(b"img-bytes")

    batches = list(provider.batches(seed=0, batch_size=0))
    assert batches == [examples[0]]


def test_vision_text_provider_answer_normalization_paths() -> None:
    assert _normalize_answers({"answers": [" cat ", ""]}) == ["cat"]
    assert _normalize_answers({"answers": [], "answer": " dog "}) == ["dog"]

    with pytest.raises(Exception, match="missing answer/answers"):
        _normalize_answers({})


def test_vision_text_provider_raises_for_empty_sources_and_bad_manifest_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(Exception, match="produced no samples"):
        VisionTextProvider().examples()

    manifest = tmp_path / "vision.jsonl"
    manifest.write_text("", encoding="utf-8")

    original_read_text = Path.read_text

    def _broken_read_text(self, *args, **kwargs):  # noqa: ANN001
        if self == manifest:
            raise OSError("boom")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _broken_read_text)

    with pytest.raises(Exception, match="failed to read vision_text manifest"):
        VisionTextProvider(path=str(manifest)).examples()


def test_vision_text_provider_manifest_validation_errors(tmp_path: Path) -> None:
    image_path = tmp_path / "img.ppm"
    image_path.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")

    invalid_manifest = tmp_path / "invalid.jsonl"
    invalid_manifest.write_text("{bad json}\n", encoding="utf-8")
    with pytest.raises(Exception, match="failed to parse vision_text manifest"):
        VisionTextProvider(path=str(invalid_manifest)).examples()

    missing_prompt = tmp_path / "missing-prompt.jsonl"
    missing_prompt.write_text(
        json.dumps({"image_path": image_path.name, "answer": "cat"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="missing prompt"):
        VisionTextProvider(path=str(missing_prompt)).examples()

    missing_image_path = tmp_path / "missing-image-path.jsonl"
    missing_image_path.write_text(
        json.dumps({"prompt": "what?", "answer": "cat"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="missing image_path"):
        VisionTextProvider(path=str(missing_image_path)).examples()

    empty_manifest = tmp_path / "empty.jsonl"
    empty_manifest.write_text('\n"ignore"\n', encoding="utf-8")
    with pytest.raises(Exception, match="produced no samples"):
        VisionTextProvider(path=str(empty_manifest)).examples()


def test_vision_text_provider_resolve_path_accepts_absolute_and_items_short_circuit(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "absolute.ppm"
    image_path.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")

    assert _resolve_image_path(str(image_path), base_dir=tmp_path) == image_path

    provider = VisionTextProvider(items=[{"prompt": "What?", "answer": "cat"}])
    assert provider._resolve_files() == []
