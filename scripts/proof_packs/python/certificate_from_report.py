from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate evaluation.cert.json from report.json."
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report_path = Path(args.report)
    cert_path = Path(args.out)

    try:
        from invarlock.reporting.certificate import make_certificate
    except Exception as exc:
        print(f"Certificate generation warning: {exc}", file=sys.stderr)
        return 1

    try:
        report = json.loads(report_path.read_text())
    except Exception as exc:
        print(f"Certificate generation warning: {exc}", file=sys.stderr)
        return 1

    try:
        cert = make_certificate(report, report)
    except Exception as exc:
        print(f"Certificate generation warning: {exc}", file=sys.stderr)
        return 1

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    cert_path.write_text(json.dumps(cert, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
