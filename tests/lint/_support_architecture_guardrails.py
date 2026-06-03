from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = REPO_ROOT / "src/invarlock/eval/metrics.py"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")
