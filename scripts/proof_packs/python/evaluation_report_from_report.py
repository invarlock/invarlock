from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate evaluation.report.json from report.json."
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report_path = Path(args.report)
    out_path = Path(args.out)

    try:
        from invarlock.reporting.report_make import make_report
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"Evaluation report generation warning: {exc}", file=sys.stderr)
        return 1

    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Evaluation report generation warning: {exc}", file=sys.stderr)
        return 1

    try:
        evaluation_report = make_report(report, report)
    except (RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(f"Evaluation report generation warning: {exc}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(evaluation_report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
