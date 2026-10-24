"""Build complete fictional reviews around unchanged retained evidence, offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import replace
from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from invarlock.captured_contracts import captured_snapshot, load_payloads
from invarlock.record_reporting import _view
from invarlock.report_presentation import AuthoredReview, render_html, render_markdown

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = Path(__file__).with_name("scenarios.json")


def build(output: Path) -> dict[str, dict[str, str]]:
    """Use a fresh destination; retain original files and a byte inventory."""
    output.mkdir(parents=True, exist_ok=False)
    scenarios = json.loads(SCENARIOS.read_text())
    inventory = {}
    for name, scenario in scenarios.items():
        reference = ROOT / scenario["reference"]
        destination = output / name
        shutil.copytree(reference / "evidence", destination / "evidence")
        with captured_snapshot(reference / "evidence") as snapshot:
            _, inputs, _ = load_payloads(snapshot)
            technical = _view(inputs["report"], snapshot)
        originals = {
            str(path.relative_to(destination)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in destination.rglob("*")
            if path.is_file()
        }
        for source_name, target_name in (
            ("verification.receipt.json", "verification.receipt.json"),
            ("reference.json", "source-reference.json"),
            ("README.md", "source-reference.md"),
        ):
            shutil.copyfile(reference / source_name, destination / target_name)
            originals[target_name] = hashlib.sha256(
                (reference / source_name).read_bytes()
            ).hexdigest()
        for path in sorted(reference.glob("*.pem")):
            shutil.copyfile(path, destination / path.name)
            originals[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for suffix in ("html", "md", "xml"):
            path = reference / f"report.{suffix}"
            if path.exists():
                target = f"retained-report.{suffix}"
                shutil.copyfile(path, destination / target)
                originals[target] = hashlib.sha256(path.read_bytes()).hexdigest()
        for suffix, render in (("html", render_html), ("md", render_markdown)):
            (destination / f"technical-report.{suffix}").write_text(render(technical))
        authored = AuthoredReview(
            **{
                key: scenario[key]
                for key in (
                    "label",
                    "byline",
                    "recommendation",
                    "decision",
                    "next_action",
                )
            },
            context=tuple(tuple(row) for row in scenario["context"]),
            basis=tuple(tuple(row) for row in scenario["basis"]),
        )
        review = replace(
            technical,
            review=authored,
            metrics=tuple(
                replace(metric, evidence_source="evidence/" + metric.evidence_source)
                for metric in technical.metrics
            ),
        )
        (destination / "review.html").write_text(render_html(review))
        (destination / "review.md").write_text(render_markdown(review))
        (destination / "authored-scenario.json").write_text(
            json.dumps(scenario, indent=2, ensure_ascii=False) + "\n"
        )
        (destination / "original-file-hashes.json").write_text(
            json.dumps(originals, indent=2, sort_keys=True) + "\n"
        )
        (destination / "README.txt").write_text(
            "Start with review.html or review.md for the completed fictional client review.\n"
            "authored-scenario.json contains the fictional business context and decisions.\n"
            "technical-report.html and .md are current technical renderings without authored context.\n"
            "retained-report files, where present, are the original reference renderings.\n"
            "The evidence/ directory retains the original pack's paths and exact bytes.\n"
            "original-file-hashes.json inventories those files and the copied source reference and receipt.\n"
            "Source reference: "
            + scenario["reference"]
            + " in the InvarLock repository.\n"
            "source-reference.md is copied unchanged; its original relative links refer to that repository location.\n"
            "Verify the canonical pack inside evidence/ using independently approved trust inputs.\n"
        )
        with ZipFile(output / f"{name}.zip", "w", compression=ZIP_DEFLATED) as archive:
            for path in sorted(destination.rglob("*")):
                if path.is_file():
                    archive.write(path, str(path.relative_to(destination)))
        inventory[name] = originals
    links = "".join(
        f'<li><a href="{name}/review.html">{escape(scenario["decision"])}</a> '
        f'· <a href="{name}.zip">Complete review package</a> '
        f'· <a href="{name}/technical-report.html">Technical report (evidence only)</a></li>'
        for name, scenario in scenarios.items()
    )
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Completed InvarLock review examples</title>"
        "<style>body{font:18px/1.6 system-ui;max-width:850px;margin:40px auto;padding:24px;color:#20251f}li{margin:24px 0}a{color:#145d7b}</style>"
        "<h1>Completed client review examples</h1>"
        "<p>Real retained comparisons, complete fictional business reviews. Each example chooses a purpose, rationale, owner, recommendation, decision and next action.</p>"
        "<ul>" + links + "</ul></html>\n"
    )
    return inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output.resolve())
    print(args.output / "index.html")
