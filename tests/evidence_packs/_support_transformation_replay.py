from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.evidence_packs.python.editing.streaming_transform import (
    materialize_transformation_artifact,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_baseline(path: Path) -> None:
    path.mkdir(parents=True)
    _write_json(path / "config.json", {"model_type": "qwen2", "num_hidden_layers": 1})
    _write_json(path / "tokenizer_config.json", {"model_max_length": 128})
    _write_json(path / "tokenizer.json", {"version": "1.0"})
    (path / "vocab.json").write_bytes(b'{"a": 0}\n')
    save_file(
        {
            "model.layers.0.mlp.up_proj.weight": torch.tensor(
                [
                    [0.37, 0.72, -1.11, 2.43],
                    [-0.31, 1.79, 0.16, -2.07],
                    [1.31, -0.54, 3.27, 0.91],
                    [-1.77, 0.43, -0.81, 2.14],
                ],
                dtype=torch.float32,
            ),
            "model.layers.0.self_attn.q_proj.weight": torch.tensor(
                [[-0.0, 1.0], [2.0, 3.0]], dtype=torch.float32
            ),
            "model.visual.blocks.0.mlp.up_proj.weight": torch.tensor(
                [[4.0, 5.0], [6.0, 7.0]], dtype=torch.float32
            ),
        },
        path / "model.safetensors",
        metadata={"format": "pt"},
    )


def _baseline_tensors(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(
        str(path / "model.safetensors"), framework="pt", device="cpu"
    ) as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}


def _artifact_tensors(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    index = json.loads(
        (path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    tensors: dict[str, torch.Tensor] = {}
    for name, shard_name in index["weight_map"].items():
        with safe_open(str(path / shard_name), framework="pt", device="cpu") as handle:
            tensors[name] = handle.get_tensor(name)
    return tensors, dict(index["weight_map"])


def _rewrite_artifact_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    _, weight_map = _artifact_tensors(path)
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard_name in weight_map.items():
        by_shard[shard_name].append(name)
    for shard_name, names in by_shard.items():
        save_file(
            {name: tensors[name].contiguous() for name in names},
            path / shard_name,
            metadata={"format": "pt"},
        )


def _materialize(
    tmp_path: Path,
    *,
    edit_type: str = "quant_rtn",
    parameters: dict[str, object] | None = None,
    scope: str = "ffn",
) -> tuple[Path, Path, dict[str, object]]:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    _write_baseline(baseline)
    resolved_parameters = parameters or {"bits": 4, "group_size": 2}
    materialize_transformation_artifact(
        baseline_path=baseline,
        output_path=artifact,
        edit_type=edit_type,
        parameters=resolved_parameters,
        scope=scope,
    )
    return baseline, artifact, resolved_parameters
