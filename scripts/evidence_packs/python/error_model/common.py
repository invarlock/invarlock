from __future__ import annotations

import gc
import json
import os
import re
import shutil
import sys
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

try:
    from ..runtime_tools import load_causal_model
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime_tools import load_causal_model

_IMPORT_OR_LOAD_ERRORS = (ImportError, ModuleNotFoundError, OSError, RuntimeError)
_NUMERIC_COERCION_ERRORS = (TypeError, ValueError, OverflowError)
_CONFIG_ATTR_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)
_OVERLAY_FALLBACK_ERRORS = (
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
    KeyError,
)


def fix_layer_drop_config(
    config: Any,
    *,
    total_layers: int,
    kept_layers: int,
    baseline_config: Mapping[str, Any] | None = None,
) -> None:
    if config is None:
        return

    if not isinstance(total_layers, int) or not isinstance(kept_layers, int):
        return
    if total_layers < 1 or kept_layers < 1 or kept_layers > total_layers:
        return

    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(config, key):
            try:
                setattr(config, key, int(kept_layers))
            except _CONFIG_ATTR_ERRORS:
                continue

    # Some architectures (e.g., Qwen2) store per-layer config lists such as
    # `layer_types`. If we shrink the transformer stack, these lists must be
    # truncated to match the new `num_hidden_layers` or model loading fails.
    try:
        items = list(vars(config).items())
    except TypeError:
        items = []

    for name, value in items:
        if "layer" not in name:
            continue
        if not isinstance(value, list):
            continue
        if len(value) != total_layers:
            continue
        try:
            setattr(config, name, value[:kept_layers])
        except _CONFIG_ATTR_ERRORS:
            continue

    if baseline_config is None:
        return

    # Some configs can lose optional attributes during save/load (custom
    # transformers configs + remote-code-backed configs). Preserve baseline
    # settings when they are present but become null on the mutated config.
    if (
        hasattr(config, "sliding_window")
        and getattr(config, "sliding_window", None) is None
    ):
        sliding_window = baseline_config.get("sliding_window")
        if isinstance(sliding_window, int) and sliding_window > 0:
            try:
                config.sliding_window = sliding_window
            except _CONFIG_ATTR_ERRORS:
                pass


def _is_norm_module(module: torch.nn.Module) -> bool:
    """Check if module is a normalization layer (LayerNorm, RMSNorm, etc.)."""
    class_name = module.__class__.__name__.lower()
    # Match LayerNorm, RMSNorm, LlamaRMSNorm, MistralRMSNorm, Qwen2RMSNorm, etc.
    return "norm" in class_name and any(
        kw in class_name for kw in ("layer", "rms", "group", "batch")
    )


def _get_norm_weight(module: torch.nn.Module) -> torch.Tensor | None:
    """Get the weight parameter from a norm module."""
    # Standard LayerNorm uses .weight
    if hasattr(module, "weight") and module.weight is not None:
        return module.weight
    # Some RMSNorm implementations use .scale
    if hasattr(module, "scale") and module.scale is not None:
        return module.scale
    return None


def _parse_layer_indices(spec: str, max_layers: int) -> list[int]:
    """Parse layer specification like '8,9,10,11' or 'all' or empty for all."""
    if not spec or spec.lower() in ("all", "*", ""):
        return list(range(max_layers))
    indices = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            # Range like "8-12"
            start, end = part.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return [i for i in indices if 0 <= i < max_layers]


def _default_last_layers(all_layer_indices: list[int], n: int) -> list[int]:
    if not all_layer_indices:
        return []
    return all_layer_indices[-n:] if len(all_layer_indices) >= n else all_layer_indices


def _stable_crc32(text: str) -> int:
    """Stable hash for deterministic per-parameter RNG across runs/processes."""
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def _select_row_indices(
    w: torch.Tensor,
    *,
    rows: int,
    selection: str,
    seed: int,
    name: str,
) -> torch.Tensor:
    import torch

    """Select row indices deterministically (per-param) for row-wise probes."""
    rows = max(0, min(int(rows), int(w.shape[0])))
    if rows <= 0:
        return torch.empty((0,), dtype=torch.long, device=w.device)

    mode = (selection or "first").strip().lower()
    if mode in {"first", "head"}:
        return torch.arange(rows, device=w.device)
    if mode in {"last", "tail"}:
        start = max(0, int(w.shape[0]) - rows)
        return torch.arange(start, start + rows, device=w.device)

    # Per-parameter deterministic RNG.
    seed_u32 = (int(seed) + _stable_crc32(name)) & 0xFFFFFFFF
    g = torch.Generator(device="cpu")
    g.manual_seed(seed_u32)

    if mode in {"random_block", "rand_block", "block"}:
        max_start = max(0, int(w.shape[0]) - rows)
        start = (
            0
            if max_start == 0
            else int(torch.randint(0, max_start + 1, (1,), generator=g).item())
        )
        return torch.arange(start, start + rows, device=w.device)

    if mode in {"random", "rand"}:
        idx = torch.randperm(int(w.shape[0]), generator=g)[:rows]
        return idx.to(device=w.device)

    if mode in {"top_norm", "top", "largest_norm"}:
        norms = torch.linalg.vector_norm(w.float(), ord=2, dim=1)
        return torch.topk(norms, k=rows, largest=True).indices

    # Default to first rows for unknown modes.
    return torch.arange(rows, device=w.device)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _safe_copy(src: Path, dest: Path) -> None:
    try:
        if src.is_file() and not dest.exists():
            shutil.copy2(src, dest)
    except OSError:
        return


def _safe_symlink(src: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        return
    try:
        os.symlink(src, dest)
    except OSError:
        # Fall back to a copy for filesystems that disallow symlinks.
        _safe_copy(src, dest)


def _shape_mismatch_overlay_safetensors(
    *,
    baseline_path: Path,
    output_path: Path,
    tokenizer: Any,
    delta: int,
) -> dict[str, Any] | None:
    """Create a vocab-size mismatch error model without re-saving full weights.

    For very large sharded checkpoints, `save_pretrained()` can be killed (OOM)
    while writing shards. Instead, we:
    - symlink the baseline shards into the output dir
    - write a small safetensors override shard containing resized embedding + lm_head
    - write a new `model.safetensors.index.json` pointing those tensors to the override
    """
    index_path = baseline_path / "model.safetensors.index.json"
    if not index_path.is_file():
        return None

    import torch

    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except (ImportError, ModuleNotFoundError):
        return None

    baseline_index = _read_json(index_path)
    weight_map = baseline_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return None

    def _pick_key(candidates: tuple[str, ...]) -> str | None:
        for suffix in candidates:
            for key in weight_map:
                if not isinstance(key, str):
                    continue
                if key.endswith(suffix):
                    return key
        return None

    embed_key = _pick_key(
        ("embed_tokens.weight", "wte.weight", "word_embeddings.weight")
    )
    if embed_key is None:
        return None

    head_key = _pick_key(("lm_head.weight",))

    embed_file = weight_map.get(embed_key)
    head_file = weight_map.get(head_key) if head_key else None
    if not isinstance(embed_file, str) or not embed_file:
        return None
    if head_key and (not isinstance(head_file, str) or not head_file):
        head_key = None
        head_file = None

    # Load only the tensors we need.
    def _load_tensor(filename: str, key: str) -> torch.Tensor:
        with safe_open(
            str(baseline_path / filename), framework="pt", device="cpu"
        ) as f:
            return f.get_tensor(key)

    embed_weight = _load_tensor(embed_file, embed_key)
    old_vocab = int(embed_weight.shape[0])
    hidden = int(embed_weight.shape[1]) if embed_weight.ndim == 2 else None
    if hidden is None or embed_weight.ndim != 2:
        return None

    new_vocab = old_vocab + int(delta)
    if new_vocab <= old_vocab:
        return None

    # Preserve the old embeddings exactly; pad new rows with zeros. This keeps
    # the scenario stable (dataset won't hit the new token IDs).
    new_embed = torch.zeros(
        (new_vocab, hidden), dtype=embed_weight.dtype, device=embed_weight.device
    )
    new_embed[:old_vocab, :] = embed_weight
    del embed_weight

    overrides: dict[str, torch.Tensor] = {embed_key: new_embed}
    if head_key and head_file:
        head_weight = _load_tensor(head_file, head_key)
        if head_weight.ndim == 2 and int(head_weight.shape[0]) == old_vocab:
            new_head = torch.zeros(
                (new_vocab, int(head_weight.shape[1])),
                dtype=head_weight.dtype,
                device=head_weight.device,
            )
            new_head[:old_vocab, :] = head_weight
            overrides[head_key] = new_head
        del head_weight

    output_path.mkdir(parents=True, exist_ok=True)

    # Save tokenizer assets for later evaluation.
    try:
        tokenizer.save_pretrained(output_path)
    except (OSError, RuntimeError):
        pass

    # Copy minimal config artifacts and bump vocab_size.
    cfg_path = baseline_path / "config.json"
    cfg: dict[str, Any] = _read_json(cfg_path) if cfg_path.is_file() else {}
    cfg["vocab_size"] = int(new_vocab)
    _write_json(output_path / "config.json", cfg)

    for extra in ("generation_config.json", "chat_template.jinja"):
        _safe_copy(baseline_path / extra, output_path / extra)

    # Symlink all baseline shards referenced by the index into the output dir.
    for filename in sorted({v for v in weight_map.values() if isinstance(v, str)}):
        _safe_symlink(baseline_path / filename, output_path / filename)

    override_name = "shape_mismatch_overrides.safetensors"
    override_path = output_path / override_name
    save_file(overrides, str(override_path))

    updated_weight_map = {
        str(k): str(v) for k, v in weight_map.items() if isinstance(k, str)
    }
    updated_weight_map[embed_key] = override_name
    if head_key:
        updated_weight_map[head_key] = override_name

    _write_json(
        output_path / "model.safetensors.index.json",
        {
            "metadata": baseline_index.get("metadata")
            if isinstance(baseline_index.get("metadata"), dict)
            else {},
            "weight_map": updated_weight_map,
        },
    )

    return {
        "error_type": "shape_mismatch",
        "injected": True,
        "mode": "overlay_safetensors",
        "old_vocab_size": int(old_vocab),
        "new_vocab_size": int(new_vocab),
        "delta": int(delta),
        "overrides_file": override_name,
        "overrides_keys": sorted(overrides),
    }


def _load_error_model(
    *, baseline_path: Path, trust_remote_code: bool
) -> tuple[torch.nn.Module, bool]:
    import torch

    # trust_remote_code is gated by require_remote_code_opt_in /
    # INVARLOCK_ALLOW_REMOTE_CODE in task_tools.py create-error-model before it
    # reaches this shared helper.
    try:
        model, _ = load_causal_model(
            baseline_path,
            dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        return model, True
    except (OSError, RuntimeError, ValueError) as gpu_err:
        print(
            f"GPU loading failed ({gpu_err}), falling back to CPU (may be slow for large models)"
        )
        model, _ = load_causal_model(
            baseline_path,
            dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        return model, False


def _collect_block_params(
    model: torch.nn.Module,
) -> tuple[dict[int, list[tuple[str, torch.Tensor]]], int]:
    block_params: dict[int, list[tuple[str, torch.Tensor]]] = {}
    block_pattern = re.compile(r"(?:layers|blocks|h)\.(\d+)\.")
    for name, param in model.named_parameters():
        match = block_pattern.search(name)
        if match:
            idx = int(match.group(1))
            block_params.setdefault(idx, []).append((name, param))
    num_blocks = max(block_params.keys()) + 1 if block_params else 0
    return block_params, num_blocks


def _shrink_layer_stack(container: object, attr: str) -> tuple[bool, int, int]:
    import torch

    layers = getattr(container, attr, None)
    if layers is None:
        return False, 0, 0
    try:
        total = len(layers)
    except TypeError:
        return False, 0, 0
    if total < 2:
        return False, total, total
    keep = total - 1
    try:
        if isinstance(layers, torch.nn.ModuleList):
            new_layers = torch.nn.ModuleList(list(layers)[:keep])
        else:
            new_layers = list(layers)[:keep]
        setattr(container, attr, new_layers)
        return True, total, keep
    except (AttributeError, TypeError):
        return False, total, total


def _save_error_model(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    output_path: Path,
    error_info: dict[str, object],
    use_gpu: bool,
) -> None:
    if use_gpu:
        import torch

        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)
    (output_path / "error_metadata.json").write_text(json.dumps(error_info, indent=2))
