"""
Evaluation report tooling (`invarlock.reporting`).

Provides the evaluation report schema, builder, and renderers.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from typing import Any, cast

from invarlock.core.assurance_contract import (
    REPORT_BUILD_EVENT_CATEGORIES as _REPORT_BUILD_EVENT_CATEGORIES,
)

from .render_markdown import render_report_markdown as _render_report_markdown
from .report_schema import REPORT_JSON_SCHEMA, REPORT_SCHEMA_VERSION, validate_report
from .report_summary import (
    ReportManifestSummary as _ReportManifestSummary,
)
from .report_summary import (
    build_quality_gates_summary as _build_quality_gates_summary,
)
from .report_summary import (
    build_report_manifest_summary as _build_report_manifest_summary,
)
from .report_summary import (
    build_safety_dashboard_summary as _build_safety_dashboard_summary,
)
from .report_summary import (
    derive_report_manifest_evidence_level as _derive_report_manifest_evidence_level,
)


def _measurement_contract_digest(contract):
    from .verify_check_helpers_consistency import (
        _measurement_contract_digest as _impl,
    )

    return _impl(contract)


def _baseline_guard_payload(baseline, guard_name):
    from .verify_check_helpers_consistency import _baseline_guard_payload as _impl

    return _impl(baseline, guard_name)


def _render_evaluation_bundle_reviewer_summary(*args, **kwargs):
    from .report_bundle import render_evaluation_bundle_reviewer_summary as _impl

    return _impl(*args, **kwargs)


def _write_report_manifest(*args, **kwargs):
    from .report_bundle import write_report_manifest as _impl

    return _impl(*args, **kwargs)


def _ensure_report_build_evidence(*args, **kwargs):
    from .report_builder_support import ensure_report_build_evidence as _impl

    return _impl(*args, **kwargs)


def _record_report_build_event(*args, **kwargs):
    from .report_builder_support import record_report_build_event as _impl

    return _impl(*args, **kwargs)


def _report_build_has_evidence_events(*args, **kwargs):
    from .report_builder_support import report_build_has_evidence_events as _impl

    return _impl(*args, **kwargs)


def _format_debug_metric_diffs(*args, **kwargs):
    from .run_report_metrics_contract import format_debug_metric_diffs as _impl

    return _impl(*args, **kwargs)


def _merge_primary_metric_health(*args, **kwargs):
    from .run_report_metrics_contract import merge_primary_metric_health as _impl

    return _impl(*args, **kwargs)


def _install_compat_module(name: str, exports: dict[str, object]) -> types.ModuleType:
    module_name = f"{__name__}.{name}"
    existing = sys.modules.get(module_name)
    module = (
        existing
        if isinstance(existing, types.ModuleType)
        else types.ModuleType(module_name)
    )
    module.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None)
    for export_name, export_value in exports.items():
        module.__dict__.setdefault(export_name, export_value)
    cast(Any, module).__all__ = tuple(exports)
    sys.modules[module_name] = module
    globals()[name] = module
    return module


def _install_lazy_compat_module(
    name: str,
    *,
    target_module: str,
    exports: tuple[str, ...],
) -> types.ModuleType:
    module_name = f"{__name__}.{name}"
    existing = sys.modules.get(module_name)
    module = (
        existing
        if isinstance(existing, types.ModuleType)
        else types.ModuleType(module_name)
    )
    module.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None)

    def __getattr__(attr: str) -> object:
        if attr not in exports:
            raise AttributeError(attr)
        target = __import__(target_module, fromlist=[attr])
        value = getattr(target, attr)
        module.__dict__[attr] = value
        return value

    module.__dict__["__getattr__"] = __getattr__
    module.__dict__["__all__"] = exports
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
    "guards_common",
    {
        "_baseline_guard_payload": _baseline_guard_payload,
        "_measurement_contract_digest": _measurement_contract_digest,
    },
)
_install_compat_module(
    "report_manifest",
    {
        "ReportManifestSummary": _ReportManifestSummary,
        "build_report_manifest_summary": _build_report_manifest_summary,
        "derive_report_manifest_evidence_level": _derive_report_manifest_evidence_level,
        "render_evaluation_bundle_reviewer_summary": _render_evaluation_bundle_reviewer_summary,
        "write_report_manifest": _write_report_manifest,
    },
)
_install_compat_module(
    "report_build_evidence",
    {
        "REPORT_BUILD_EVENT_CATEGORIES": _REPORT_BUILD_EVENT_CATEGORIES,
        "ensure_report_build_evidence": _ensure_report_build_evidence,
        "record_report_build_event": _record_report_build_event,
        "report_build_has_evidence_events": _report_build_has_evidence_events,
    },
)
_install_compat_module(
    "run_metric_utils",
    {
        "format_debug_metric_diffs": _format_debug_metric_diffs,
        "merge_primary_metric_health": _merge_primary_metric_health,
    },
)
_install_lazy_compat_module(
    "report_files",
    target_module=__name__ + ".report_bundle",
    exports=("save_report",),
)
_install_lazy_compat_module(
    "report_build_context",
    target_module=__name__ + ".report_" + "builder_support",
    exports=(
        "ReportBuildContext",
        "EvaluationReportBuilder",
        "extract_telemetry",
        "build_artifacts_payload",
        "attach_schedule_digest",
        "build_moe_section",
        "resolve_capacity_context",
        "evaluate_primary_metric_tail",
    ),
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
