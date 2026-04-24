from __future__ import annotations

import json
import sys
from pathlib import Path

from error_injection_config import fix_layer_drop_config_json


def _layer_count(cfg: dict) -> int | None:
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        value = cfg.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "Usage: repair_missing_tensors_config.py <baseline_config.json> <error_config.json>",
            file=sys.stderr,
        )
        return 2

    baseline_path = Path(argv[1])
    error_path = Path(argv[2])

    baseline_cfg = json.loads(baseline_path.read_text(encoding="utf-8"))
    error_cfg = json.loads(error_path.read_text(encoding="utf-8"))
    if not isinstance(baseline_cfg, dict) or not isinstance(error_cfg, dict):
        return 2

    total_layers = _layer_count(baseline_cfg)
    kept_layers = _layer_count(error_cfg)
    if total_layers is None or kept_layers is None:
        return 0

    before = json.dumps(error_cfg, sort_keys=True)
    fix_layer_drop_config_json(
        error_cfg,
        total_layers=total_layers,
        kept_layers=kept_layers,
        baseline_config=baseline_cfg,
    )
    after = json.dumps(error_cfg, sort_keys=True)
    if before != after:
        error_path.write_text(json.dumps(error_cfg, indent=2) + "\n", encoding="utf-8")
        print(
            f"Repaired missing_tensors config: total_layers={total_layers} kept_layers={kept_layers}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
