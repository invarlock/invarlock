from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

_ENC = "utf-8"


class _HashLike(Protocol):
    def update(self, data: bytes, /) -> None: ...
    def hexdigest(self, /) -> str: ...


def _h() -> _HashLike:
    return hashlib.blake2s(digest_size=32)


def hash_bytes(b: bytes, *, salt: bytes | None = None) -> str:
    h = _h()
    if salt:
        h.update(salt)
    h.update(b)
    return h.hexdigest()


def hash_json(obj: Any, *, salt: str | None = None) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hash_bytes(s.encode(_ENC), salt=salt.encode(_ENC) if salt else None)


def hash_int_array(arr, *, salt: str | None = None, byteorder: str = "little") -> str:
    # Accept numpy arrays to avoid extra copies
    try:
        import numpy as _np
    except Exception:  # pragma: no cover - numpy always present in tests
        # Fallback: best-effort conversion
        b = bytes(int(x) & 0xFFFFFFFF for x in arr)
        return hash_bytes(b, salt=salt.encode(_ENC) if salt else None)

    a = _np.asarray(arr, dtype=_np.int32, order="C")
    return hash_bytes(a.tobytes(order="C"), salt=salt.encode(_ENC) if salt else None)


__all__ = ["hash_bytes", "hash_json", "hash_int_array"]
