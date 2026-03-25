from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _maybe_number(value: str | None) -> int | float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except Exception:
        return None
    if number.is_integer():
        return int(number)
    return number


def _load_run_state_environment(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / "state" / "environment.json"
    if not state_path.is_file():
        return {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_environment_payload(run_dir: Path | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if run_dir is not None:
        payload.update(_load_run_state_environment(run_dir))

    payload.setdefault("recorded_at", _utc_now())
    payload.setdefault("platform", platform.platform())
    payload.setdefault("python_version", platform.python_version())
    payload.setdefault("gpu_name", os.environ.get("PACK_GPU_NAME", ""))
    payload.setdefault("gpu_count", _maybe_number(os.environ.get("PACK_GPU_COUNT")))
    payload.setdefault(
        "gpu_memory_gb", _maybe_number(os.environ.get("PACK_GPU_MEM_GB"))
    )
    payload.setdefault(
        "fp8_native_support",
        _truthy(os.environ.get("FP8_NATIVE_SUPPORT")),
    )
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write proof-pack environment metadata."
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-dir")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = Path(args.run_dir) if args.run_dir else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(build_environment_payload(run_dir), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
