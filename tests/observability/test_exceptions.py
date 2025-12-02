from __future__ import annotations

import time

import pytest

from invarlock.observability.exporters import ExportedMetric, MetricsExporter
from invarlock.observability.exporters import (
    export_or_raise as _export_or_raise,  # type: ignore[attr-defined]
)


class _FailingExporter(MetricsExporter):
    def __init__(self, name: str = "failing"):
        super().__init__(name)

    def export(self, metrics):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")


class _FalseExporter(MetricsExporter):
    def __init__(self, name: str = "false"):
        super().__init__(name)

    def export(self, metrics):  # type: ignore[no-untyped-def]
        return False


def _make_metric():
    return [ExportedMetric(name="m", value=1.0, timestamp=time.time())]


@pytest.mark.unit
def test_export_or_raise_raises_observability_error_on_exception():
    from invarlock.core.exceptions import ObservabilityError

    with pytest.raises(ObservabilityError) as ei:
        _export_or_raise(_FailingExporter(), _make_metric())
    assert ei.value.code == "E801"
    assert ei.value.details and ei.value.details.get("exporter") == "failing"


@pytest.mark.unit
def test_export_or_raise_raises_observability_error_on_false():
    from invarlock.core.exceptions import ObservabilityError

    with pytest.raises(ObservabilityError) as ei:
        _export_or_raise(_FalseExporter(), _make_metric())
    assert ei.value.code == "E801"
