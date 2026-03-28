from __future__ import annotations

from .render_markdown import render_report_markdown
from .report_console import compute_report_hash as _compute_report_hash

__all__ = ["render_report_markdown", "_compute_report_hash"]
