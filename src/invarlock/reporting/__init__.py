"""
Evaluation report tooling (`invarlock.reporting`).

Provides the evaluation report schema, builder, and renderers.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from typing import Any, cast

from .render_markdown import render_report_markdown as _render_report_markdown
from .report_schema import REPORT_JSON_SCHEMA, REPORT_SCHEMA_VERSION, validate_report
from .report_summary import (
    build_quality_gates_summary as _build_quality_gates_summary,
)
from .report_summary import (
    build_safety_dashboard_summary as _build_safety_dashboard_summary,
)
from .telemetry import (
    telemetry_output_enabled as _telemetry_output_enabled,
)
from .telemetry import (
    telemetry_summary_line as _telemetry_summary_line,
)


def _measurement_contract_digest(contract):
    from .verify_check_helpers_consistency import (
        _measurement_contract_digest as _impl,
    )

    return _impl(contract)


def _baseline_guard_payload(baseline, guard_name):
    from .verify_check_helpers_consistency import _baseline_guard_payload as _impl

    return _impl(baseline, guard_name)


def _install_compat_module(name: str, exports: dict[str, object]) -> types.ModuleType:
    module_name = f"{__name__}.{name}"
    module = types.ModuleType(module_name)
    module.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None)
    module.__dict__.update(exports)
    cast(Any, module).__all__ = tuple(exports)
    sys.modules[module_name] = module
    globals()[name] = module
    return module


_install_compat_module(
    "render",
    {
        "build_quality_gates_summary": _build_quality_gates_summary,
        "build_safety_dashboard_summary": _build_safety_dashboard_summary,
        "render_report_markdown": _render_report_markdown,
    },
)
_install_compat_module(
    "report_telemetry",
    {
        "telemetry_output_enabled": _telemetry_output_enabled,
        "telemetry_summary_line": _telemetry_summary_line,
    },
)
_install_compat_module(
    "guards_common",
    {
        "_baseline_guard_payload": _baseline_guard_payload,
        "_measurement_contract_digest": _measurement_contract_digest,
    },
)


def make_report(*args, **kwargs):
    from .report_make import make_report as _make_report

    return _make_report(*args, **kwargs)


def render_report_markdown(*args, **kwargs):
    return _render_report_markdown(*args, **kwargs)


def render_report_html(*args, **kwargs):
    from .html import render_report_html as _render_report_html

    return _render_report_html(*args, **kwargs)


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "REPORT_JSON_SCHEMA",
    "make_report",
    "render_report_markdown",
    "render_report_html",
    "validate_report",
]
