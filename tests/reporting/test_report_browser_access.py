"""Optional browser checks of report navigation, reflow and offline presentation.

Set INVARLOCK_REPORT_PLAYWRIGHT_MODULE to an installed Playwright Node module.
The fixture deliberately uses synthetic stress labels, not real evaluation data.
"""

import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from invarlock.record_reporting import _view
from invarlock.report_presentation import render_html
from tests.reporting.test_report_experience import evidence, pack_json

pytestmark = pytest.mark.manual


def test_keyboard_narrow_spacing_contrast_and_offline_report(tmp_path):
    module = os.environ.get("INVARLOCK_REPORT_PLAYWRIGHT_MODULE")
    if not module:
        pytest.skip("Set the report browser module path for this check")
    pack = evidence()
    original = _view(pack_json(pack, "report"), pack)
    metrics = tuple(
        replace(
            original.metrics[0],
            name=("layout-stress-" + "x" * 180) if i == 2 else f"metric_{i}",
            scope="Synthetic scope " + "long-identifier-" * 12,
            decision=("pass", "regression", "insufficient_evidence")[i],
            evidence_source=f"synthetic-layout-only/{'long-source/' * 12}result.json#/metrics/{i}",
        )
        for i in range(3)
    )
    view = replace(
        original,
        title="Synthetic layout fixture: not evaluation evidence",
        decision="insufficient_evidence",
        summary="Synthetic presentation stress test with three metric groups.",
        metrics=metrics,
    )
    html = tmp_path / "stress.html"
    html.write_text(render_html(view))
    result = subprocess.run(
        [
            "node",
            str(Path(__file__).with_name("report_browser_access.cjs")),
            module,
            html.as_uri(),
            str(tmp_path / "access.png"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
