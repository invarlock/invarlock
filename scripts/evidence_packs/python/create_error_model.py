from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

try:
    from create_error_model_helpers import (
        _OVERLAY_FALLBACK_ERRORS,
        _apply_error_injection,
        _collect_block_params,
        _load_error_model,
        _save_error_model,
        _shape_mismatch_overlay_safetensors,
    )
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from create_error_model_helpers import (
        _OVERLAY_FALLBACK_ERRORS,
        _apply_error_injection,
        _collect_block_params,
        _load_error_model,
        _save_error_model,
        _shape_mismatch_overlay_safetensors,
    )

try:
    from runtime_tools import require_remote_code_opt_in
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runtime_tools import require_remote_code_opt_in

from transformers import AutoTokenizer


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            "Usage: create_error_model.py <baseline_path> <output_path> <error_type>",
            file=sys.stderr,
        )
        return 2

    baseline_path = Path(argv[1])
    output_path = Path(argv[2])
    error_type = argv[3]

    print(f"Loading baseline from {baseline_path}...")
    trust_remote_code = require_remote_code_opt_in("create_error_model.py")
    tokenizer = AutoTokenizer.from_pretrained(
        baseline_path, trust_remote_code=trust_remote_code
    )

    if error_type == "shape_mismatch":
        # Large sharded models can be OOM-killed during save_pretrained() shard writes.
        # Prefer an index-based overlay that only rewrites the embedding + lm_head tensors.
        delta = 8
        try:
            error_info = _shape_mismatch_overlay_safetensors(
                baseline_path=baseline_path,
                output_path=output_path,
                tokenizer=tokenizer,
                delta=delta,
            )
        except _OVERLAY_FALLBACK_ERRORS as exc:
            error_info = None
            print(f"WARNING: shape_mismatch overlay failed ({exc}); falling back")

        if error_info is not None:
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "error_metadata.json").write_text(
                json.dumps(error_info, indent=2, sort_keys=True) + "\n"
            )
            print(f"Saved error model to {output_path}")
            return 0

    model, use_gpu = _load_error_model(
        baseline_path=baseline_path, trust_remote_code=trust_remote_code
    )
    error_info: dict[str, object] = {"error_type": error_type, "injected": False}
    block_params, num_blocks = _collect_block_params(model)
    print(f"Detected {num_blocks} transformer blocks")

    if error_type == "shape_mismatch":
        try:
            emb = model.get_input_embeddings()
            old_vocab = int(getattr(emb, "num_embeddings", emb.weight.shape[0]))
            delta = 8
            new_vocab = old_vocab + delta
            model.resize_token_embeddings(new_vocab)
            error_info.update(
                {
                    "injected": True,
                    "old_vocab_size": old_vocab,
                    "new_vocab_size": int(new_vocab),
                    "delta": int(delta),
                }
            )
            print(f"Resized token embeddings: {old_vocab} -> {new_vocab}")
        except (RuntimeError, TypeError, ValueError) as exc:
            print(f"WARNING: shape_mismatch not injected ({exc})")
    else:
        _apply_error_injection(
            error_type=error_type,
            model=model,
            baseline_path=baseline_path,
            block_params=block_params,
            error_info=error_info,
        )

    _save_error_model(
        model=model,
        tokenizer=tokenizer,
        output_path=output_path,
        error_info=error_info,
        use_gpu=use_gpu,
    )

    del model
    gc.collect()
    print(f"Saved error model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
