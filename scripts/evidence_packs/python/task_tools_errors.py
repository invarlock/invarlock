from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

try:
    from .runtime_tools import require_remote_code_opt_in
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runtime_tools import require_remote_code_opt_in


def _create_error_model(args: argparse.Namespace) -> int:
    try:
        from .error_model.common import (
            _OVERLAY_FALLBACK_ERRORS,
            _collect_block_params,
            _load_error_model,
            _save_error_model,
            _shape_mismatch_overlay_safetensors,
        )
        from .error_model.probe_injections import _apply_error_injection
    except ImportError:  # pragma: no cover - direct script execution
        from error_model.common import (
            _OVERLAY_FALLBACK_ERRORS,
            _collect_block_params,
            _load_error_model,
            _save_error_model,
            _shape_mismatch_overlay_safetensors,
        )
        from error_model.probe_injections import _apply_error_injection

    from transformers import AutoTokenizer

    baseline_path = Path(args.baseline_path)
    output_path = Path(args.output_path)
    error_type = str(args.error_type)

    print(f"Loading baseline from {baseline_path}...")
    trust_remote_code = require_remote_code_opt_in("task_tools.py create-error-model")
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
