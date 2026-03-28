from __future__ import annotations

from .render_helpers import _fmt_by_kind, _fmtv, _p, _short_digest
from .render_markdown import (
    _append_accuracy_subgroups,
    _append_executive_summary_section,
    _append_policy_configuration_section,
    _append_primary_metric_section,
    _append_quality_gates_section,
    _append_report_header,
    _append_safety_dashboard_section,
    _append_system_overhead_section,
    _format_plugin,
    _get_generated_at,
    _render_executive_dashboard,
    render_report_markdown,
)
from .report_console import compute_report_hash as _compute_report_hash
from .report_summary import (
    build_quality_gates_summary,
    build_safety_dashboard_summary,
)

__all__ = [
    "build_quality_gates_summary",
    "build_safety_dashboard_summary",
    "render_report_markdown",
    "_append_executive_summary_section",
    "_append_accuracy_subgroups",
    "_append_primary_metric_section",
    "_append_policy_configuration_section",
    "_append_quality_gates_section",
    "_append_safety_dashboard_section",
    "_append_system_overhead_section",
    "_append_report_header",
    "_compute_report_hash",
    "_format_plugin",
    "_fmt_by_kind",
    "_fmtv",
    "_get_generated_at",
    "_p",
    "_render_executive_dashboard",
    "_short_digest",
]
