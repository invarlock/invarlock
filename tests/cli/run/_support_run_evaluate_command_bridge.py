from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tests.cli._support_effective_config import preserve_effective_config


def _stub_run(out_dir: Path, *, run_kwargs: Mapping[str, Any] | None = None) -> Path:
    if run_kwargs is not None:
        preserve_effective_config(run_kwargs)
    ts_dir = out_dir / "20250101_000000"
    ts_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "meta": {"model_id": "stub", "adapter": "hf_causal", "device": "cpu"},
        "edit": {"name": "noop"},
        "metrics": {"primary_metric": {"preview": 1.0, "final": 1.0}},
        "data": {"preview_n": 1, "final_n": 1},
    }
    report_path = ts_dir / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path
