from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_ratio(cert: dict) -> float | None:
    verdict = cert.get("verdict") or {}
    metrics = cert.get("metrics") or {}
    for candidate in (
        verdict.get("primary_metric_ratio"),
        verdict.get("primary_metric_ratio_raw"),
        verdict.get("primary_metric_ratio_mean"),
        metrics.get("primary_metric_ratio"),
        metrics.get("primary_metric_ratio_mean"),
    ):
        if isinstance(candidate, (int, float)):
            return float(candidate)
    return None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 7:
        print(
            "Usage: determinism_repeats_summary.py <out_path> <model_id> <edit_name> "
            "<requested_repeats> <mode> <suite> <cert_path> [cert_path...]",
            file=sys.stderr,
        )
        return 2

    out_path = Path(argv[0])
    model_id = argv[1]
    edit_name = argv[2]
    try:
        requested = int(argv[3])
    except Exception:
        requested = 0
    mode = argv[4]
    suite = argv[5]
    cert_paths = [Path(p) for p in argv[6:]]

    hashes: list[str] = []
    ratios: list[float] = []
    errors: list[str] = []

    for path in cert_paths:
        try:
            raw = path.read_bytes()
            hashes.append(hashlib.sha256(raw).hexdigest())
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue

        if isinstance(data, dict):
            ratio = _extract_ratio(data)
            if ratio is not None:
                ratios.append(ratio)

    hashes_match = bool(hashes) and len(set(hashes)) == 1
    ratio_summary = None
    if ratios:
        ratio_summary = {
            "min": min(ratios),
            "max": max(ratios),
            "delta": max(ratios) - min(ratios),
        }

    payload: dict[str, object] = {
        "requested": requested,
        "completed": len(cert_paths),
        "mode": str(mode),
        "suite": str(suite),
        "model_id": str(model_id),
        "edit_name": str(edit_name),
        "cert_hashes_match": hashes_match,
        "cert_hashes": hashes,
        "primary_metric_ratio": ratio_summary,
        "errors": errors,
        "generated_at": _utc_now(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
