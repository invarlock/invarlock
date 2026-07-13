"""Canonical source and output shard planning for transformation replay."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _source_shard_plan(
    checkpoint: Path,
    *,
    weight_map: Mapping[str, Path],
    tensor_bytes: Mapping[str, int],
) -> dict[str, object]:
    by_shard: dict[Path, list[str]] = defaultdict(list)
    for name, path in weight_map.items():
        by_shard[path].append(name)
    return {
        "source_shards": [
            {
                "path": path.relative_to(checkpoint).as_posix(),
                "sha256": _file_sha256(path),
                "tensor_names": sorted(names),
                "byte_count": sum(tensor_bytes[name] for name in names),
            }
            for path, names in sorted(
                by_shard.items(), key=lambda item: item[0].as_posix()
            )
        ]
    }


def _expected_output_shard_plan(
    checkpoint: Path,
    *,
    weight_map: Mapping[str, Path],
    tensor_bytes: Mapping[str, int],
    source_shard_plan_sha256: str,
    target_manifest_sha256: str,
    max_output_shard_bytes: int,
) -> tuple[dict[str, object], dict[str, str]]:
    """Rebuild the materializer's bounded, canonical output topology.

    The receipt is not allowed to choose arbitrary shard names or assignments.
    This mirrors the materializer's source-shard ordering and greedy bounded
    chunking exactly, then returns the required artifact index mapping too.
    """

    by_shard: dict[Path, list[str]] = defaultdict(list)
    for name, path in weight_map.items():
        by_shard[path].append(name)
    source_digests = {
        path: _file_sha256(path)
        for path in sorted(by_shard, key=lambda item: item.as_posix())
    }
    chunk_specs: list[tuple[Path, tuple[str, ...], int]] = []
    for source_path in sorted(by_shard, key=lambda path: path.as_posix()):
        current_names: list[str] = []
        current_bytes = 0
        for name in sorted(by_shard[source_path]):
            byte_count = tensor_bytes[name]
            if current_names and current_bytes + byte_count > max_output_shard_bytes:
                chunk_specs.append((source_path, tuple(current_names), current_bytes))
                current_names = []
                current_bytes = 0
            current_names.append(name)
            current_bytes += byte_count
        if current_names:
            chunk_specs.append((source_path, tuple(current_names), current_bytes))

    chunks: list[dict[str, object]] = []
    expected_weight_map: dict[str, str] = {}
    total_chunks = len(chunk_specs)
    for ordinal, (source_path, names, byte_count) in enumerate(chunk_specs, start=1):
        name = f"model-{ordinal:05d}-of-{total_chunks:05d}.safetensors"
        chunks.append(
            {
                "name": name,
                "source_path": source_path.relative_to(checkpoint).as_posix(),
                "source_sha256": source_digests[source_path],
                "tensor_names": list(names),
                "byte_count": byte_count,
            }
        )
        for tensor_name in names:
            expected_weight_map[tensor_name] = name
    return (
        {
            "source_shard_plan_sha256": source_shard_plan_sha256,
            "target_manifest_sha256": target_manifest_sha256,
            "chunks": chunks,
        },
        expected_weight_map,
    )
