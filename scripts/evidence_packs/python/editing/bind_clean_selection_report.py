# ruff: noqa: UP045  # Evidence-pack shell hosts still include Python 3.9.
"""Bind a real candidate evaluation report to a v1 clean-selection schedule.

Run this only after InvarLock has emitted ``evaluation.report.json`` and the
candidate's exact replay/runtime proofs exist.  The tool recomputes quality
loss from the report, verifies the strict local assurance result, and refuses
to overwrite an incompatible binding.  It is deliberately not a free-form
JSON writer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

if __package__ in {None, ""}:  # pragma: no cover - direct shell execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from invarlock.clean_selection.binding import build_candidate_report_binding
from invarlock.clean_selection.common import (
    CleanSelectionEvidenceError,
    strict_json_object_snapshot,
)
from invarlock.evidence_pack_json import sha256_prefixed


def _write_report(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise CleanSelectionEvidenceError(
            f"could not atomically write candidate report: {exc}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def bind_candidate_report(
    *,
    report_path: Path,
    replay_path: Path,
    runtime_path: Path,
    selection_config_path: Path,
    execution_receipt_path: Path,
    model_key: str,
    candidate_id: str,
    repeat_index: int,
) -> dict[str, object]:
    """Create or verify one report-local selection binding."""

    _, report = strict_json_object_snapshot(
        report_path, label="candidate evaluation report"
    )
    _, replay = strict_json_object_snapshot(
        replay_path, label="candidate replay sidecar"
    )
    _, runtime = strict_json_object_snapshot(
        runtime_path, label="candidate runtime sidecar"
    )
    _, selection_config = strict_json_object_snapshot(
        selection_config_path, label="candidate selection config"
    )
    execution_bytes, execution_receipt = strict_json_object_snapshot(
        execution_receipt_path, label="candidate selection execution receipt"
    )
    transformation = {
        "edit_type": replay.get("edit_type"),
        "parameters": replay.get("parameters"),
        "scope": replay.get("scope"),
    }
    binding = build_candidate_report_binding(
        report=report,
        replay=replay,
        runtime=runtime,
        original_model_key=model_key,
        candidate_id=candidate_id,
        transformation=transformation,
        selection_config=selection_config,
        execution_receipt=execution_receipt,
        execution_receipt_sha256=sha256_prefixed(execution_bytes),
        repeat_index=repeat_index,
    )
    existing = report.get("clean_selection")
    if existing is not None:
        if existing != binding:
            raise CleanSelectionEvidenceError(
                "refusing to overwrite a different candidate report selection binding"
            )
        return binding
    report["clean_selection"] = binding
    _write_report(report_path, report)
    return binding


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--selection-config", required=True, type=Path)
    parser.add_argument("--execution-receipt", required=True, type=Path)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--repeat-index", required=True, type=int)
    args = parser.parse_args(argv)
    bind_candidate_report(
        report_path=args.report,
        replay_path=args.replay,
        runtime_path=args.runtime,
        selection_config_path=args.selection_config,
        execution_receipt_path=args.execution_receipt,
        model_key=args.model_key,
        candidate_id=args.candidate_id,
        repeat_index=args.repeat_index,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
