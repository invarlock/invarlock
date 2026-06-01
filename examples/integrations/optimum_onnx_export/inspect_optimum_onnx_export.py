#!/usr/bin/env python3
"""Inspect an Optimum ONNX export and record compatibility boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FORMAT_VERSION = "invarlock-optimum-onnx-compat-v1"
EXPECTED_HF_LOAD_ERROR_FILES = (
    "pytorch_model.bin",
    "model.safetensors",
    "tf_model.h5",
    "model.ckpt.index",
    "flax_model.msgpack",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect an Optimum ONNX export for integration compatibility."
    )
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_file_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def probe_onnxruntime_sessions(
    export_dir: Path, onnx_files: list[Path]
) -> dict[str, Any]:
    try:
        import onnxruntime as ort

        sessions: dict[str, Any] = {}
        for onnx_path in onnx_files:
            session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            sessions[str(onnx_path.relative_to(export_dir))] = {
                "ok": True,
                "session_providers": session.get_providers(),
                "inputs": [
                    {"name": item.name, "shape": item.shape, "type": item.type}
                    for item in session.get_inputs()
                ],
                "outputs": [
                    {"name": item.name, "shape": item.shape, "type": item.type}
                    for item in session.get_outputs()
                ],
            }

        return {
            "ok": True,
            "available_providers": ort.get_available_providers(),
            "sessions": sessions,
        }
    except Exception as exc:  # pragma: no cover - exercised by integration smoke.
        return probe_error(exc)


def probe_optimum_ort_model(export_dir: Path) -> dict[str, Any]:
    try:
        from optimum.onnxruntime import ORTModelForCausalLM

        model = ORTModelForCausalLM.from_pretrained(export_dir)
        return {
            "ok": True,
            "model_class": f"{type(model).__module__}.{type(model).__name__}",
            "providers": getattr(model, "providers", None),
        }
    except Exception as exc:  # pragma: no cover - exercised by integration smoke.
        return probe_error(exc)


def probe_hf_pytorch_load(export_dir: Path) -> dict[str, Any]:
    try:
        from transformers import AutoModelForCausalLM

        AutoModelForCausalLM.from_pretrained(export_dir)
        return {"ok": True}
    except Exception as exc:
        result = probe_error(exc)
        message = str(result.get("error_message", ""))
        result["expected_for_onnx_export"] = all(
            file_name in message for file_name in EXPECTED_HF_LOAD_ERROR_FILES
        )
        return result


def probe_error(exc: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "error_type": type(exc).__name__,
        "error_message": str(exc).splitlines()[0],
    }


def main() -> int:
    args = parse_args()
    export_dir = args.export_dir
    if not export_dir.is_dir():
        raise SystemExit(f"Export directory does not exist: {export_dir}")

    onnx_files = sorted(export_dir.glob("*.onnx"))
    if not onnx_files:
        raise SystemExit(f"No ONNX model files found under: {export_dir}")

    file_rows = collect_file_rows(export_dir)
    total_bytes = sum(row["bytes"] for row in file_rows)

    summary = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "toolchain": "hugging-face-optimum-onnx",
        "baseline_model_requested": args.baseline_model,
        "export_dir": str(export_dir),
        "onnx_files": [str(path.relative_to(export_dir)) for path in onnx_files],
        "artifact_file_count": len(file_rows),
        "artifact_total_bytes": total_bytes,
        "artifact_total_mib": round(total_bytes / math.pow(1024, 2), 3),
        "artifact_files": file_rows,
        "runtime_load_probes": {
            "onnxruntime_sessions": probe_onnxruntime_sessions(export_dir, onnx_files),
            "optimum_ort_model": probe_optimum_ort_model(export_dir),
            "hf_pytorch_auto_model": probe_hf_pytorch_load(export_dir),
        },
        "invarlock_compatibility": {
            "status": "compatibility-investigation",
            "shared_compare_ready": False,
            "reason": (
                "Optimum ONNX exports are ONNX Runtime artifacts, not HF "
                "PyTorch checkpoint directories accepted by the current "
                "shared InvarLock compare wrapper."
            ),
            "recommended_pairing": (
                "Use the shared InvarLock wrapper on HF-loadable baseline and "
                "subject checkpoints, and keep this report beside those "
                "artifacts as Optimum deployment evidence."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote compatibility probe: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
