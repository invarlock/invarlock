"""
Vision-Text provider (Phase 4): deterministic pairing and reproducibility digest.

This lightweight provider focuses on stable IDs and a reproducibility digest for
multimodal (image+text) tasks such as image captioning and VQA. It does not
perform actual batching or transforms; those belong to adapters/inference.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any

from .base import EvaluationProvider


def _sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


class VisionTextProvider(EvaluationProvider):
    """
    Minimal provider that exposes:
    - a stable pairing schedule (sorted example IDs), and
    - a digest with ids and image hashes plus transform pipeline string.

    Args:
        items: sequence of records with at least keys {"id": str, "image_bytes": bytes}
        transform_pipeline: human-readable transform pipeline description
        seed: integer seed recorded for determinism breadcrumbs
    """

    def __init__(
        self,
        *,
        items: Sequence[dict[str, Any]],
        transform_pipeline: str = "",
        seed: int | None = None,
    ) -> None:
        self._items = list(items)
        self._transform_pipeline = str(transform_pipeline)
        self._seed = int(seed) if seed is not None else None

        # Build sorted schedule once
        try:
            self._schedule: list[str] = sorted(str(x["id"]) for x in self._items)
        except Exception:  # pragma: no cover - defensive
            self._schedule = []

        # Precompute digest components
        ids_concat = "".join(self._schedule).encode()
        self._ids_sha256 = _sha256_hex(ids_concat)
        # Compute per-image hashes in schedule order; missing bytes treated as empty
        per_img_hashes = []
        for rec_id in self._schedule:
            # find the record with matching id (first match)
            img_bytes = b""
            for rec in self._items:
                if str(rec.get("id")) == rec_id:
                    ib = rec.get("image_bytes")
                    if isinstance(ib, bytes | bytearray):
                        img_bytes = bytes(ib)
                    break
            per_img_hashes.append(_sha256_hex(img_bytes).encode())
        self._images_sha256 = _sha256_hex(b"".join(per_img_hashes))

    def pairing_schedule(self) -> list[str]:
        return list(self._schedule)

    def digest(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "provider": "vision_text",
            "version": 1,
            "ids_sha256": self._ids_sha256,
            "images_sha256": self._images_sha256,
            "transform_pipeline": self._transform_pipeline,
        }
        if self._seed is not None:
            d["seed"] = int(self._seed)
        return d

    def batches(
        self, *, seed: int, batch_size: int
    ) -> Iterable[dict[str, Any]]:  # pragma: no cover - not used in Phase 4 tests
        raise NotImplementedError(
            "VisionTextProvider batches are adapter/task-specific and not implemented here"
        )


__all__ = ["VisionTextProvider"]
