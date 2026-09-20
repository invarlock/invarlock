"""Presentation preferences do not change report facts or trust policy."""

import base64
import hashlib
import re

from invarlock.report_presentation import ReportView, render_html


def test_report_defaults_to_light_and_authorizes_only_its_local_scripts():
    view = ReportView(
        title="Report",
        family="Captured",
        decision="pass",
        summary="A result",
        metrics=(),
        assurance=(),
    )
    html = render_html(view)
    assert "prefers-color-scheme" not in html
    assert ':root[data-report-theme="dark"]' in html
    assert 'id="report-theme-toggle"' in html
    assert 'aria-pressed="false" hidden>Dark mode</button>' in html
    assert "matchMedia" not in html
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    assert scripts
    for script in scripts:
        digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
        assert f"'sha256-{digest}'" in html
    assert "script-src 'unsafe-inline'" not in html
    assert "default-src 'none'" in html
    assert "@media print{.report-appearance{display:none}}" in html
