"""Pure-ish helper utilities for run pairing, masking digests, and token coercion."""

from __future__ import annotations

import hashlib
from array import array
from collections.abc import Iterable, Sequence
from typing import Any

import click
import numpy as np
import typer

_IMPORT_UNSET = object()
torch: Any = _IMPORT_UNSET


def _get_torch() -> Any:
    global torch
    if torch is _IMPORT_UNSET:
        try:
            import torch as _torch
        except ImportError:
            torch = None
        else:
            torch = _torch
    return None if torch is _IMPORT_UNSET else torch


def _to_int_list(values: Sequence[int] | Iterable[int]) -> list[int]:
    return [int(v) for v in values]


def _tensor_or_list_to_ints(values: Any) -> list[int]:
    """Coerce possible tensor/list-like inputs to a list[int]."""
    torch_mod = _get_torch()
    if torch_mod is not None and hasattr(values, "tolist"):
        raw = values.tolist()
        if isinstance(raw, list):
            return _to_int_list(raw)
        try:
            return _to_int_list(list(raw))
        except (typer.Exit, SystemExit, click.exceptions.Exit):
            raise
        except (TypeError, ValueError):
            return []
    if isinstance(values, np.ndarray | list | tuple):
        return _to_int_list(list(values))
    if isinstance(values, Iterable):
        return _to_int_list(values)
    return []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hash_sequences(seqs: Sequence[Sequence[int]] | Iterable[Sequence[int]]) -> str:
    """Compute a stable digest for a sequence of integer token sequences."""
    hasher = hashlib.blake2s(digest_size=16)
    for seq in seqs:
        try:
            seq_len = len(seq)
        except TypeError:
            seq = list(seq)
            seq_len = len(seq)
        hasher.update(seq_len.to_bytes(4, "little", signed=False))
        arr = array("I", (int(token) & 0xFFFFFFFF for token in seq))
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def _compute_mask_positions_digest(windows: dict[str, Any]) -> str | None:
    """Compute a rolled hash of MLM mask positions across windows."""
    hasher = hashlib.blake2s(digest_size=16)
    any_masked = False
    for arm in ("preview", "final"):
        sec = windows.get(arm)
        if not isinstance(sec, dict):
            continue
        labels = sec.get("labels")
        if not isinstance(labels, list) or not labels:
            continue
        hasher.update(arm.encode("utf-8"))
        for row in labels:
            row_list = _tensor_or_list_to_ints(row)
            if not row_list:
                continue
            found = False
            for idx, value in enumerate(row_list):
                if int(value) != -100:
                    hasher.update(b"1")
                    hasher.update(idx.to_bytes(4, "little", signed=False))
                    found = True
            if found:
                any_masked = True
            hasher.update(b"|")
    if not any_masked:
        return None
    digest = hasher.hexdigest()
    return digest if digest else None
