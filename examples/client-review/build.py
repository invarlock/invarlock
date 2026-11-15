"""Build the worked scenarios through the installed client-handoff API, offline."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from html import escape
from pathlib import Path

from invarlock.engine import CLIENT_REVIEW_CONTEXT_FORMAT, build_client_review

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = Path(__file__).with_name("scenarios.json")


def build(output: Path) -> dict[str, dict[str, str]]:
    """Keep reference originals; clients use the same builder with their own context."""
    output.mkdir(parents=True, exist_ok=False)
    scenarios = json.loads(SCENARIOS.read_text())
    inventory = {}
    for name, scenario in scenarios.items():
        reference = ROOT / scenario["reference"]
        destination = output / name
        attachments = {
            "source-reference.json": reference / "reference.json",
            "source-reference.md": reference / "README.md",
        }
        attachments.update(
            {path.name: path for path in sorted(reference.glob("*.pem"))}
        )
        for suffix in ("html", "md", "xml"):
            path = reference / f"report.{suffix}"
            if path.exists():
                attachments[f"retained-report.{suffix}"] = path
        with tempfile.TemporaryDirectory(prefix="client-review-context-") as directory:
            temporary = Path(directory).resolve()
            scenario_path = temporary / "authored-scenario.json"
            scenario_path.write_text(
                json.dumps(scenario, indent=2, ensure_ascii=False) + "\n"
            )
            attachments["authored-scenario.json"] = scenario_path
            context = {
                key: value for key, value in scenario.items() if key != "reference"
            }
            context.update(
                format=CLIENT_REVIEW_CONTEXT_FORMAT,
                review_date="2026-10-24",
                title=(
                    "Editorial model replacement"
                    if name == "likelihood-adverse"
                    else "Model service handover rehearsal"
                ),
            )
            context_path = temporary / "context.json"
            context_path.write_text(json.dumps(context, ensure_ascii=False))
            build_client_review(
                reference / "evidence",
                context_path=context_path,
                output=destination,
                retained_receipt=reference / "verification.receipt.json",
                supporting_files=attachments,
            )
        inventory[name] = json.loads(
            (destination / "original-file-hashes.json").read_text()
        )
        shutil.copyfile(destination / "handoff.zip", output / f"{name}.zip")
    links = "".join(
        f'<li><a href="{name}/brief.html">{escape(scenario["decision"])}</a> '
        f'· <a href="{name}.zip">Review package ZIP</a> '
        f'· <a href="{name}/verification.html">Recipient guide</a> '
        f'· <a href="{name}/review.html">Full authored review</a></li>'
        for name, scenario in scenarios.items()
    )
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>InvarLock client review examples</title>"
        "<style>body{font:18px/1.6 system-ui;max-width:850px;margin:40px auto;padding:24px;color:#20251f}li{margin:24px 0}a{color:#145d7b}</style>"
        "<h1>Client review examples</h1>"
        "<p>Real retained comparisons and attributed fictional business reviews. Start with the decision brief, then follow the guide using independently approved recipient inputs.</p>"
        "<ul>" + links + "</ul></html>\n"
    )
    return inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output.resolve())
    print(args.output / "index.html")
