from __future__ import annotations

import hashlib

from invarlock.eval.providers.vision_text import VisionTextProvider


def _sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def test_vision_text_provider_digest_and_schedule_stable():
    # Prepare three tiny fake images and ids
    items = [
        {"id": "img-001", "image_bytes": b"image-001-bytes"},
        {"id": "img-002", "image_bytes": b"image-002-bytes"},
        {"id": "img-003", "image_bytes": b"image-003-bytes"},
    ]
    pipeline = "resize-224-center-crop-normalize(mean=0.5,std=0.5)"

    p1 = VisionTextProvider(items=items, transform_pipeline=pipeline, seed=42)
    p2 = VisionTextProvider(items=items, transform_pipeline=pipeline, seed=42)

    # Pairing schedule must be a stable, sorted list of ids
    sched1 = p1.pairing_schedule()
    sched2 = p2.pairing_schedule()
    assert sched1 == sorted([i["id"] for i in items])
    assert sched1 == sched2

    # Digest must be stable and include ids/image hashes and the transform pipeline
    d1 = p1.digest()
    d2 = p2.digest()
    assert d1 == d2
    assert d1["provider"] == "vision_text"
    assert d1["version"] >= 1
    assert d1["transform_pipeline"] == pipeline
    # ids hash is sha256 over the sorted ids
    ids_concat = "".join(sorted([i["id"] for i in items])).encode()
    assert d1["ids_sha256"] == _sha256_hex(ids_concat)
    # images hash is sha256 over concatenated per-image hashes in schedule order
    per_img_hashes = b"".join(
        _sha256_hex(i["image_bytes"]).encode()
        for i in sorted(items, key=lambda x: x["id"])
    )
    assert d1["images_sha256"] == _sha256_hex(per_img_hashes)

    # Changing the pipeline should change the digest
    p3 = VisionTextProvider(
        items=items, transform_pipeline=pipeline + "+brightness(0.1)", seed=42
    )
    d3 = p3.digest()
    assert d3 != d1


def test_vision_text_provider_handles_missing_bytes():
    items = [{"id": "img-100"}, {"id": "img-200", "image_bytes": b""}]
    provider = VisionTextProvider(items=items)
    digest = provider.digest()
    assert "seed" not in digest
    # When bytes missing, sha256 of empty bytes is used
    empty_hash = _sha256_hex(b"").encode()
    combined = _sha256_hex(empty_hash + empty_hash)
    assert digest["images_sha256"] == combined
