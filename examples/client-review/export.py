"""Export a verified package inventory to a fresh PDF delivery directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from invarlock.report_delivery import export_client_review

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--browser-module", type=Path, required=True)
    parser.add_argument("--paper", choices=["A4", "Letter"], default="A4")
    args = parser.parse_args()
    print(
        json.dumps(
            export_client_review(
                args.package,
                output=args.output,
                browser_module=args.browser_module,
                paper=args.paper,
            ),
            indent=2,
        )
    )
