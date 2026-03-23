from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as exc:  # pragma: no cover - proof-pack preflight enforces PyYAML
    raise SystemExit(
        "PyYAML is required to normalize staged proof-pack presets"
    ) from exc


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _normalized_payload(
    payload: Any,
    *,
    seq_len: int,
    stride: int,
    preview_n: int,
    final_n: int,
    skip_overhead_check: bool,
) -> dict[str, Any]:
    normalized = _mapping(payload)

    dataset = _mapping(normalized.get("dataset"))
    dataset["seq_len"] = seq_len
    dataset["stride"] = stride
    dataset["preview_n"] = preview_n
    dataset["final_n"] = final_n
    normalized["dataset"] = dataset

    if skip_overhead_check:
        context = _mapping(normalized.get("context"))
        run = _mapping(context.get("run"))
        run["skip_overhead_check"] = True
        context["run"] = run
        normalized["context"] = context

    return normalized


def normalize_staged_preset(
    preset_path: Path,
    *,
    seq_len: int,
    stride: int,
    preview_n: int,
    final_n: int,
    skip_overhead_check: bool,
) -> None:
    raw = preset_path.read_text()
    loaded = yaml.safe_load(raw) if raw.strip() else {}
    normalized = _normalized_payload(
        loaded,
        seq_len=seq_len,
        stride=stride,
        preview_n=preview_n,
        final_n=final_n,
        skip_overhead_check=skip_overhead_check,
    )
    preset_path.write_text(yaml.safe_dump(normalized, sort_keys=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize a proof-pack staged preset for evaluate runtime."
    )
    parser.add_argument("--preset", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--preview-n", type=int, required=True)
    parser.add_argument("--final-n", type=int, required=True)
    parser.add_argument("--skip-overhead-check", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    normalize_staged_preset(
        Path(args.preset),
        seq_len=args.seq_len,
        stride=args.stride,
        preview_n=args.preview_n,
        final_n=args.final_n,
        skip_overhead_check=args.skip_overhead_check,
    )


if __name__ == "__main__":
    main()
