from __future__ import annotations

import argparse
import json
from pathlib import Path

from invarlock.core.runtime_manifest_verify import (
    verify_report_manifest as _verify_report_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="invarlock-runtime-verify",
        description="Verify runtime.manifest.json against an evaluation report.",
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report_path = Path(args.report)
    manifest_path = Path(args.manifest)
    errors = _verify_report_manifest(report_path, manifest_path)
    ok = not errors

    if args.json:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "errors": errors,
                    "report": str(report_path),
                    "manifest": str(manifest_path),
                }
            )
        )
    elif ok:
        print(f"runtime verify ok report={report_path} manifest={manifest_path}")
    else:
        for error in errors:
            print(error)
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
