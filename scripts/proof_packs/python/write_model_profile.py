from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _get(cfg: dict[str, Any], key: str, *fallbacks: str) -> Any:
    value = cfg.get(key)
    if value is not None:
        return value
    for fb in fallbacks:
        value = cfg.get(fb)
        if value is not None:
            return value
    return None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(
            "Usage: write_model_profile.py <baseline_dir> <model_id>", file=sys.stderr
        )
        return 2

    baseline_dir = Path(argv[0])
    model_id = argv[1]
    profile_path = baseline_dir / "model_profile.json"
    if profile_path.exists():
        return 0

    config_path = baseline_dir / "config.json"
    if not config_path.exists():
        return 0

    try:
        cfg = json.loads(config_path.read_text())
    except Exception:
        return 0

    if not isinstance(cfg, dict):
        return 0

    weights_bytes = 0
    for pat in ("*.safetensors", "*.bin"):
        for fp in baseline_dir.glob(pat):
            try:
                weights_bytes += fp.stat().st_size
            except OSError:
                pass

    profile = {
        "model_id": model_id,
        "weights_bytes": weights_bytes,
        "weights_gb": round(weights_bytes / (1024**3), 3),
        "hidden_size": _get(cfg, "hidden_size", "n_embd", "d_model"),
        "num_layers": _get(cfg, "num_hidden_layers", "n_layer"),
        "num_heads": _get(cfg, "num_attention_heads", "n_head"),
        "num_kv_heads": _get(cfg, "num_key_value_heads", "num_key_value_groups"),
        "max_position_embeddings": _get(
            cfg, "max_position_embeddings", "max_seq_len", "seq_length"
        ),
        "dtype_bytes": 2,
    }

    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
