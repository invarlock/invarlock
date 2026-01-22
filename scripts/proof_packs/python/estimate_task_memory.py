from __future__ import annotations

import json
import math
import os
from pathlib import Path


def parse_batch(val: str | None, default: int) -> int:
    if not val:
        return default
    val = str(val).strip()
    if val.startswith("auto:"):
        try:
            return int(val.split(":", 1)[1])
        except Exception:
            return default
    try:
        return int(val)
    except Exception:
        return default


def main() -> int:
    profile_path = Path(os.environ["PROFILE_PATH"])
    task_type = os.environ.get("TASK_TYPE", "")
    model_id = (os.environ.get("MODEL_ID") or "").lower()

    try:
        profile = json.loads(profile_path.read_text())
    except Exception:
        return 0

    if not isinstance(profile, dict):
        return 0

    if not model_id:
        model_id = str(profile.get("model_id", "")).lower()

    weights_gb = profile.get("weights_gb") or 0.0
    if not weights_gb:
        weights_bytes = profile.get("weights_bytes") or 0
        weights_gb = weights_bytes / (1024**3) if weights_bytes else 0.0

    hidden_size = profile.get("hidden_size")
    num_layers = profile.get("num_layers")
    num_heads = profile.get("num_heads")
    num_kv_heads = profile.get("num_kv_heads") or num_heads
    dtype_bytes = profile.get("dtype_bytes") or 2

    def size_category() -> str:
        if any(t in model_id for t in ("mixtral", "moe", "8x7b")):
            return "moe"
        if weights_gb >= 120:
            return "70"
        if weights_gb >= 80:
            return "40"
        if weights_gb >= 60:
            return "30"
        if weights_gb >= 24:
            return "13"
        return "7"

    category = size_category()

    invarlock_cfg: dict[str, tuple[int, int]] = {
        "7": (2048, 96),
        "13": (1536, 64),
        "30": (1024, 48),
        "40": (1024, 32),
        "moe": (1024, 24),
        "70": (128, 2),
    }

    seq_len_invarlock, batch_invarlock = invarlock_cfg.get(category, (1024, 32))

    def kv_cache_gb(batch: int, seq_len: int) -> float:
        if not all(
            isinstance(x, int) and x > 0 for x in (hidden_size, num_layers, num_heads)
        ):
            return 0.0
        kv_heads = (
            num_kv_heads
            if isinstance(num_kv_heads, int) and num_kv_heads > 0
            else num_heads
        )
        head_dim = hidden_size // num_heads if num_heads else 0
        if head_dim == 0:
            return 0.0
        elems = 2 * num_layers * batch * seq_len * kv_heads * head_dim
        return elems * dtype_bytes / (1024**3)

    load_overhead = float(os.environ.get("MODEL_LOAD_OVERHEAD_GB", "4"))
    edit_overhead = float(os.environ.get("EDIT_OVERHEAD_GB", "8"))
    batch_overhead = float(os.environ.get("BATCH_EDIT_OVERHEAD_GB", "8"))
    inv_overhead = float(os.environ.get("INVARLOCK_OVERHEAD_GB", "6"))

    if task_type == "GENERATE_PRESET":
        required = 5.0
    elif task_type == "SETUP_BASELINE":
        required = float(weights_gb) + load_overhead
    elif task_type == "CREATE_EDITS_BATCH":
        required = (float(weights_gb) * 2.0) + batch_overhead
    elif task_type in ("CREATE_EDIT", "CREATE_ERROR"):
        required = float(weights_gb) + edit_overhead
    elif task_type in ("CALIBRATION_RUN", "CERTIFY_EDIT", "CERTIFY_ERROR"):
        required = (
            float(weights_gb)
            + kv_cache_gb(int(batch_invarlock), int(seq_len_invarlock))
            + inv_overhead
        )
    else:
        required = float(weights_gb) + inv_overhead

    per_device = int(os.environ.get("GPU_MEMORY_PER_DEVICE", "180"))
    max_gpus = int(os.environ.get("NUM_GPUS", "8"))
    required_mem = int(math.ceil(required))
    required_gpus = max(1, int(math.ceil(required_mem / per_device)))
    if max_gpus > 0:
        required_gpus = min(required_gpus, max_gpus)

    print(f"{required_mem} {required_gpus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
