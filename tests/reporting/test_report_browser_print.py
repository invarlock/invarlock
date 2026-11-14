"""Optional real-browser regression for print, closed details and metric tabs.

Set INVARLOCK_REPORT_PLAYWRIGHT_MODULE to an installed Playwright Node module
and INVARLOCK_REPORT_PDF_PYTHON to a Python with pypdf. No downloads are made.
"""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from invarlock.evidence_reporting import render_evidence

pytestmark = pytest.mark.manual


@pytest.mark.parametrize("family", ["native", "captured", "judge", "combined"])
@pytest.mark.parametrize("paper", ["A4", "Letter"])
def test_actual_pdf_retains_review_and_closed_evidence_details(tmp_path, family, paper):
    module = os.environ.get("INVARLOCK_REPORT_PLAYWRIGHT_MODULE")
    pdf_python = os.environ.get("INVARLOCK_REPORT_PDF_PYTHON")
    if not module or not pdf_python:
        pytest.skip("Set the report browser and PDF reader paths for this check")
    if family == "native":
        root = (
            Path(__file__).resolve().parents[2]
            / "examples/acceptance-handoff/golden/evidence"
        )
        exact_key = "paired_binary"
    elif family == "captured":
        from invarlock.evaluation_records.templates import example_project
        from tests._evaluation_support import build_pack, write_snapshot

        baseline, subject, policy = example_project("classification")
        root = tmp_path / "evidence"
        write_snapshot(root, build_pack(baseline, subject, policy))
        exact_key = "missing_count"
    elif family == "judge":
        from tests.judge_measurements.test_evidence_acceptance import _publish

        root = _publish(tmp_path)[0].path
        exact_key = "subject_bound"
    else:
        from tests.evidence_sets.test_verification import fixture

        root = fixture(tmp_path)[0]
        exact_key = "statistical_scope"
    html = tmp_path / "review.html"
    render_evidence(root, html_path=html)
    script = tmp_path / "print.cjs"
    script.write_text("""
const { chromium } = require(process.argv[2]);
const fs = require('node:fs');
(async () => {
  const browser = await chromium.launch({headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1280, height: 900}});
    await page.goto(process.argv[3]);
    if (!(await page.locator('details:not([open])').count())) throw Error('No closed details exercised');
    const expected = await page.locator('.review-context li,.review-context dt,.review-context dd,.review-context p,.interpretation-note,.change-summary,.review-action,.result-reading,.decision-summary,.assurance dt,.assurance dd,.findings li,.findings p,.limits li,.next-steps p,.next-steps li,.metric-explanation,.interval-reading dt,.interval-reading dd,.interval-basis p,.notes li,.shared-context dd').allTextContents();
    fs.writeFileSync(process.argv[6], JSON.stringify(expected));
    await page.pdf({path: process.argv[4], format: process.argv[7], printBackground: process.argv[7] === 'A4',
      margin: {top: '14mm', bottom: '14mm', left: '14mm', right: '14mm'}});
    await page.locator('.evidence-detail').evaluateAll(details => details.forEach(item => item.open = true));
    await page.pdf({path: process.argv[4] + '.expanded.pdf', format: process.argv[7], printBackground: true});
    await page.setViewportSize({width: 320, height: 844});
    if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw Error('Mobile page overflows');
    await page.screenshot({path: process.argv[5], fullPage: true});
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exit(1);});
""")
    pdf = tmp_path / "review.pdf"
    subprocess.run(
        [
            "node",
            str(script),
            module,
            html.as_uri(),
            str(pdf),
            str(tmp_path / "mobile.png"),
            str(tmp_path / "expected.json"),
            paper,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        [
            pdf_python,
            "-c",
            "from pypdf import PdfReader; import json,sys; print(json.dumps([p.extract_text() for p in PdfReader(sys.argv[1]).pages]))",
            str(pdf),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    pages = json.loads(result.stdout)
    assert "Review action" in pages[0]
    assert all(f"Page {i} of {len(pages)}" in page for i, page in enumerate(pages, 1))
    # Running page furniture can interrupt a paragraph split across two pages.
    # Check it above, then compare the complete body text independently.
    body_pages = [
        re.sub(r"Page \d+ of \d+|InvarLock · Model change review", "", page)
        for page in pages
    ]
    text = re.sub(r"\s+", " ", " ".join(body_pages))
    for phrase in (
        "Model change review",
        "Decision context",
        "Findings to review",
        "Scope and limitations",
        "Next decision",
        "Generated interpretation",
        "Intended use and business context: Not supplied",
        "Decision checks",
        "Exact comparison data",
    ):
        assert phrase.casefold() in text.casefold(), phrase

    def compact(value):
        return re.sub(r"\s+", "", value).casefold()

    for explanation in json.loads((tmp_path / "expected.json").read_text()):
        assert compact(explanation) in compact(text), explanation[:120]
    sources = re.findall(r"Supporting record: ([^<\n]+)", html.read_text())
    for source in sources:
        assert re.sub(r"\s+", "", source) in re.sub(r"\s+", "", text)
    assert f'"{exact_key}"' not in text
    expanded = subprocess.run(
        [
            pdf_python,
            "-c",
            "from pypdf import PdfReader; import sys; print(' '.join(p.extract_text() for p in PdfReader(sys.argv[1]).pages))",
            str(pdf) + ".expanded.pdf",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert f'"{exact_key}"' in expanded.stdout
