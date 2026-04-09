from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except (
    ModuleNotFoundError
) as exc:  # pragma: no cover - proof-pack preflight enforces PyYAML
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


def _coerce_required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SystemExit(
            f"baseline report is missing numeric data.{key} needed to normalize preset"
        )
    return int(value)


def schedule_from_baseline_report(report_path: Path) -> tuple[int, int, int, int]:
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to load baseline report JSON: {report_path}") from exc

    source: dict[str, Any] = {}
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            source = data
        else:
            dataset = payload.get("dataset")
            if isinstance(dataset, dict):
                source = dataset
    if not source:
        raise SystemExit(
            f"baseline report does not contain data/dataset schedule fields: {report_path}"
        )

    return (
        _coerce_required_int(source, "seq_len"),
        _coerce_required_int(source, "stride"),
        _coerce_required_int(source, "preview_n"),
        _coerce_required_int(source, "final_n"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize a proof-pack staged preset for evaluate runtime."
    )
    parser.add_argument("--preset", required=True)
    parser.add_argument("--baseline-report")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--preview-n", type=int)
    parser.add_argument("--final-n", type=int)
    parser.add_argument("--skip-overhead-check", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.baseline_report:
        seq_len, stride, preview_n, final_n = schedule_from_baseline_report(
            Path(args.baseline_report)
        )
    else:
        values = (args.seq_len, args.stride, args.preview_n, args.final_n)
        if any(value is None for value in values):
            raise SystemExit(
                "either --baseline-report or all of "
                "--seq-len/--stride/--preview-n/--final-n is required"
            )
        seq_len, stride, preview_n, final_n = (
            int(args.seq_len),
            int(args.stride),
            int(args.preview_n),
            int(args.final_n),
        )
    normalize_staged_preset(
        Path(args.preset),
        seq_len=seq_len,
        stride=stride,
        preview_n=preview_n,
        final_n=final_n,
        skip_overhead_check=args.skip_overhead_check,
    )


if __name__ == "__main__":
    main()
