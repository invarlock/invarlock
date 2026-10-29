"""Complete worked scenarios preserve the source pack and exported commentary."""

import hashlib
import json
import os
import re
import runpy
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/client-review"


def test_worked_examples_keep_original_evidence_and_populate_the_review(tmp_path):
    build = runpy.run_path(str(EXAMPLE / "build.py"))["build"]
    output = tmp_path / "worked"
    inventories = build(output)
    scenarios = json.loads((EXAMPLE / "scenarios.json").read_text())
    for name, scenario in scenarios.items():
        destination = output / name
        source = ROOT / scenario["reference"]
        original_files = [p for p in (source / "evidence").rglob("*") if p.is_file()]
        for original in original_files:
            relative = original.relative_to(source)
            assert (destination / relative).read_bytes() == original.read_bytes()
            assert (
                inventories[name][str(relative)]
                == hashlib.sha256(original.read_bytes()).hexdigest()
            )
        with ZipFile(output / f"{name}.zip") as archive:
            for relative, expected in inventories[name].items():
                assert hashlib.sha256(archive.read(relative)).hexdigest() == expected
            assert (
                archive.read("review.html")
                == (destination / "review.html").read_bytes()
            )
        for suffix in ("html", "md"):
            review = (destination / f"review.{suffix}").read_text()
            assert scenario["label"] in review
            assert scenario["decision"] in review
            assert "Decision owner and recommendation: Not supplied" not in review
            assert "Organizational approval: Not recorded by this report" not in review
            assert scenario["next_action"] in review
            assert "evidence/reports/evaluation.report.json#/metrics/0" in review
            from html import unescape

            brief = (destination / f"brief.{suffix}").read_text()
            for _, value in scenario["summary"]:
                assert value in unescape(review)
                assert value in unescape(brief)
        assert (
            "Legacy unstructured commentary"
            in (destination / "technical-report.html").read_text()
        )
        for match in re.finditer(
            r"((?:evidence/|verification\.)[\w/.-]+\.json)#(/[^\s.]+)",
            json.dumps(scenario),
        ):
            path, pointer = match.groups()
            node = json.loads((destination / path).read_text())
            for key in pointer.rstrip('"').split("/")[1:]:
                node = node[int(key)] if isinstance(node, list) else node[key]
    with pytest.raises(FileExistsError):
        build(output)


def test_worked_case_illustrations_match_retained_outputs_and_selection_rule():
    scenario = json.loads((EXAMPLE / "scenarios.json").read_text())["hosted-pass"]
    root = ROOT / scenario["reference"] / "evidence/records"
    baseline = json.loads((root / "baseline.json").read_text())["records"]
    candidate = json.loads((root / "subject.json").read_text())["records"]
    assert [row["id"] for row in baseline] == [row["id"] for row in candidate]
    pairs = [
        (b["output"] == b["expected"], c["output"] == c["expected"])
        for b, c in zip(baseline, candidate, strict=True)
    ]
    assert len(pairs) == 400
    assert sum(b for b, _ in pairs) == 20
    assert sum(c for _, c in pairs) == 24
    for index, expected, old, new, outcome in (
        (34, "ultrasound", "ultrasound.", "ultrasound", (False, True)),
        (56, "Hank", "Hank", "a", (True, False)),
    ):
        assert pairs.index(outcome) == index
        assert baseline[index]["expected"] == candidate[index]["expected"] == expected
        assert baseline[index]["output"] == old and candidate[index]["output"] == new
        illustration = next(
            value for _, value in scenario["basis"] if f"#/records/{index}" in value
        )
        assert baseline[index]["id"] in illustration
        for word in (expected, old, new):
            assert f"“{word}”" in illustration


@pytest.mark.manual
@pytest.mark.parametrize("name", ["likelihood-adverse", "hosted-pass"])
@pytest.mark.parametrize("paper", ["A4", "Letter"])
def test_worked_review_context_survives_actual_print(tmp_path, name, paper):
    module = os.environ.get("INVARLOCK_REPORT_PLAYWRIGHT_MODULE")
    pdf_python = os.environ.get("INVARLOCK_REPORT_PDF_PYTHON")
    if not module or not pdf_python:
        pytest.skip("Set the report browser and PDF reader paths for this check")
    build = runpy.run_path(str(EXAMPLE / "build.py"))["build"]
    output = tmp_path / "worked"
    build(output)
    root = output / name
    pdf = root / "review.pdf"
    subprocess.run(
        ["node", str(EXAMPLE / "print.cjs"), module, str(root), paper],
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
    scenario = json.loads((root / "authored-scenario.json").read_text())

    def compact(text):
        return re.sub(r"\s+", "", text).casefold()

    assert compact(scenario["decision"]) in compact(pages[0])
    body = compact(
        " ".join(
            re.sub(r"Page \d+ of \d+|InvarLock · Model change review", "", page)
            for page in pages
        )
    )
    values = [
        scenario[key]
        for key in ("label", "byline", "recommendation", "decision", "next_action")
    ]
    values.extend(
        value
        for _, value in scenario["context"] + scenario["basis"] + scenario["summary"]
    )
    for value in values:
        assert compact(value) in body, value
    assert "Decisionownerandrecommendation:Notsupplied".casefold() not in body

    brief_result = subprocess.run(
        [
            pdf_python,
            "-c",
            "from pypdf import PdfReader; import json,sys; print(json.dumps([p.extract_text() for p in PdfReader(sys.argv[1]).pages]))",
            str(root / "brief.pdf"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    brief_pages = json.loads(brief_result.stdout)
    assert brief_pages
    assert "Recipient verification" in brief_pages[0]
    brief_text = compact(" ".join(brief_pages))
    for _, value in scenario["summary"]:
        assert compact(value) in brief_text
    for value in (
        scenario["decision"],
        scenario["next_action"],
        "Recorded policy result",
        "Not performed for this recipient",
        "See result 1: evidence and checks",
    ):
        assert compact(value) in brief_text
