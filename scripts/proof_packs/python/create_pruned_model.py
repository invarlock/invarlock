from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _should_prune(name: str, scope: str) -> bool:
    name_lower = name.lower()
    if scope == "all":
        return "weight" in name_lower and any(
            x in name_lower
            for x in (
                "linear",
                "proj",
                "fc",
                "mlp",
                "attn",
                "wqkv",
                "query_key_value",
            )
        )
    if scope == "ffn":
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
    if scope == "attn":
        return "weight" in name_lower and any(
            x in name_lower
            for x in (
                "attn",
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
def _magnitude_prune(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    flat = weight.abs().flatten()
    k = int(flat.numel() * sparsity)
    if k == 0:
        return weight
    threshold = torch.kthvalue(flat, k).values
    mask = weight.abs() >= threshold
    return weight * mask.to(weight.dtype)


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(
            "Usage: create_pruned_model.py <baseline_path> <output_path> <sparsity> "
            "<scope>",
            file=sys.stderr,
        )
        return 2

    baseline_path = Path(argv[1])
    output_path = Path(argv[2])
    sparsity = float(argv[3])
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

    print(f"Pruning with sparsity={sparsity} (scope={scope})...")
    pruned_count = 0
    total_zeros = 0
    total_edited_params = 0
    total_model_params = sum(p.numel() for p in model.parameters())

    for name, param in model.named_parameters():
        if _should_prune(name, scope) and param.dim() >= 2:
            original_zeros = (param == 0).sum().item()
            param.data = _magnitude_prune(param.data, sparsity)
            new_zeros = (param == 0).sum().item()
            pruned_count += 1
            total_zeros += new_zeros
            total_edited_params += param.numel()
            if pruned_count <= 3:
                print(f"  Pruned: {name} ({original_zeros} -> {new_zeros} zeros)")

    actual_sparsity = (
        total_zeros / total_edited_params if total_edited_params > 0 else 0.0
    )
    coverage_pct = (
        100.0 * total_edited_params / total_model_params if total_model_params else 0.0
    )
    print(
        f"Pruned {pruned_count} parameters "
        f"({total_edited_params:,} / {total_model_params:,} = {coverage_pct:.1f}% coverage)"
    )
    print(f"Actual sparsity within edited params: {actual_sparsity:.2%}")

    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)

    metadata = {
        "edit_type": "magnitude_prune",
        "target_sparsity": sparsity,
        "actual_sparsity": actual_sparsity,
        "scope": scope,
        "pruned_params": pruned_count,
    }
    (output_path / "edit_metadata.json").write_text(json.dumps(metadata, indent=2))

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved pruned model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
