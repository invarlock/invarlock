from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch

try:
    from edit_implementations import apply_dense_lowrank_approximation
    from edit_metadata import build_validation_edit_metadata
    from hf_causal_loader import load_causal_model
    from runtime_tools import require_remote_code_opt_in
    from save_subject_artifact import save_edited_subject_artifact
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from edit_implementations import apply_dense_lowrank_approximation
    from edit_metadata import build_validation_edit_metadata
    from hf_causal_loader import load_causal_model
    from runtime_tools import require_remote_code_opt_in
    from save_subject_artifact import save_edited_subject_artifact
from transformers import AutoTokenizer


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
    trust_remote_code = require_remote_code_opt_in("create_lowrank_model.py")
    tokenizer = AutoTokenizer.from_pretrained(
        baseline_path, trust_remote_code=trust_remote_code
    )
    model, _ = load_causal_model(
        baseline_path,
        dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print(f"Applying low-rank SVD with rank={rank} (scope={scope})...")
    stats = apply_dense_lowrank_approximation(model, rank=rank, scope=scope)
    avg_energy = float(stats.details.get("avg_energy_retained") or 1.0)
    coverage_pct = 100.0 * stats.coverage_ratio
    print(
        f"Modified {stats.edited_tensors} matrices "
        f"({stats.edited_params:,} / {stats.total_params:,} = {coverage_pct:.1f}% coverage)"
    )
    print(f"Average energy retained: {avg_energy:.2%}")

    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    metadata = build_validation_edit_metadata(
        edit_type="lowrank_svd",
        scope=scope,
        parameters={"rank": rank},
        coverage=stats.coverage_payload(),
        extra={
            "rank": rank,
            "modified_matrices": stats.edited_tensors,
            "avg_energy_retained": avg_energy,
            "base_scope": stats.details.get("base_scope"),
            "layer_limit": stats.details.get("layer_limit"),
            "layer": stats.details.get("layer"),
        },
    )
    save_edited_subject_artifact(
        model=model,
        tokenizer=tokenizer,
        output_path=output_path,
        metadata=metadata,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved low-rank model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
