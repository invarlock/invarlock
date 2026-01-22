from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _parse_scope(raw_scope: str) -> tuple[str, int | None, int | None]:
    base = (raw_scope or "").strip()
    layer_limit: int | None = None
    layer_exact: int | None = None
    if "@" in base:
        base, rest = base.split("@", 1)
        base = base.strip()
        for item in (s.strip() for s in rest.split(",") if s.strip()):
            if item.startswith("layers="):
                try:
                    layer_limit = int(item.split("=", 1)[1])
                except Exception:
                    layer_limit = None
            elif item.startswith("layer="):
                try:
                    layer_exact = int(item.split("=", 1)[1])
                except Exception:
                    layer_exact = None
    return base, layer_limit, layer_exact


def _extract_layer_index(name: str) -> int | None:
    marker = ".layers."
    pos = name.find(marker)
    if pos < 0:
        return None
    start = pos + len(marker)
    end = start
    while end < len(name) and name[end].isdigit():
        end += 1
    if end == start:
        return None
    try:
        return int(name[start:end])
    except Exception:
        return None


def _should_lowrank(name: str, base_scope: str) -> bool:
    name_lower = name.lower()
    if base_scope == "all":
        return "weight" in name_lower and any(
            x in name_lower
            for x in ("linear", "proj", "fc", "mlp", "wqkv", "query_key_value")
        )
    if base_scope == "ffn":
        return "weight" in name_lower and any(
            x in name_lower
            for x in (
                "mlp",
                "fc",
                "gate",
                "up_proj",
                "down_proj",
                "dense_h_to_4h",
                "dense_4h_to_h",
            )
        )
    if base_scope == "attn":
        return "weight" in name_lower and any(
            x in name_lower
            for x in (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "wqkv",
                "out_proj",
                "query_key_value",
            )
        )
    return False


@torch.no_grad()
def _truncated_svd(weight: torch.Tensor, rank: int) -> torch.Tensor:
    if weight.dim() < 2:
        return weight

    original_shape = weight.shape
    weight_2d = weight.view(weight.shape[0], -1).float()

    max_rank = min(weight_2d.shape)
    effective_rank = min(rank, max_rank)

    u, s, v = torch.svd_lowrank(weight_2d, q=effective_rank, niter=2)
    lowrank = (u * s) @ v.T
    return lowrank.to(weight.dtype).view(original_shape)


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(
            "Usage: create_lowrank_model.py <baseline_path> <output_path> <rank> "
            "<scope>",
            file=sys.stderr,
        )
        return 2

    baseline_path = Path(argv[1])
    output_path = Path(argv[2])
    rank = int(argv[3])
    scope = argv[4]

    print(f"Loading baseline from {baseline_path}...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        baseline_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    base_scope, layer_limit, layer_exact = _parse_scope(scope)
    if base_scope != scope:
        print(
            f"Parsed scope={scope} -> base_scope={base_scope}, layer_limit={layer_limit}, layer={layer_exact}"
        )

    print(f"Applying low-rank SVD with rank={rank} (scope={scope})...")
    modified_count = 0
    total_energy_retained = 0.0
    num_matrices = 0
    total_model_params = sum(p.numel() for p in model.parameters())
    edited_params = 0

    for name, param in model.named_parameters():
        if layer_limit is not None or layer_exact is not None:
            idx = _extract_layer_index(name)
            if idx is None:
                continue
            if layer_exact is not None and idx != layer_exact:
                continue
            if layer_limit is not None and idx >= layer_limit:
                continue

        if _should_lowrank(name, base_scope) and param.dim() >= 2:
            original_norm = param.data.norm()
            param.data = _truncated_svd(param.data, rank)
            new_norm = param.data.norm()
            energy_retained = (
                (new_norm / original_norm).item() if original_norm > 0 else 1.0
            )
            modified_count += 1
            total_energy_retained += energy_retained
            num_matrices += 1
            edited_params += param.numel()
            if modified_count <= 3:
                print(f"  Low-rank: {name}, energy retained: {energy_retained:.4f}")

    avg_energy = total_energy_retained / num_matrices if num_matrices > 0 else 1.0
    coverage_pct = (
        100.0 * edited_params / total_model_params if total_model_params else 0.0
    )
    print(
        f"Modified {modified_count} matrices "
        f"({edited_params:,} / {total_model_params:,} = {coverage_pct:.1f}% coverage)"
    )
    print(f"Average energy retained: {avg_energy:.2%}")

    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)

    metadata = {
        "edit_type": "lowrank_svd",
        "rank": rank,
        "scope": scope,
        "modified_matrices": modified_count,
        "avg_energy_retained": avg_energy,
    }
    (output_path / "edit_metadata.json").write_text(json.dumps(metadata, indent=2))

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved low-rank model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
