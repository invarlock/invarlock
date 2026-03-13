from __future__ import annotations

from invarlock.eval.providers.vision_text import VisionTextProvider


def test_vtext_provider_digest_and_schedule_extra_case():
    items = [
        {"id": "b", "image_bytes": b"img2"},
        {"id": "a", "image_bytes": b"img1"},
        {"id": "c", "image_bytes": b""},
    ]
    p = VisionTextProvider(items=items, transform_pipeline="t", seed=123)
    sched = p.pairing_schedule()
    assert sched == ["a", "b", "c"]
    d = p.digest()
    assert d.get("seed") == 123
    assert isinstance(d.get("ids_sha256"), str) and isinstance(
        d.get("images_sha256"), str
    )
