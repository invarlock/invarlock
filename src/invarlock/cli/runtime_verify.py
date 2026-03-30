from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from invarlock.runtime_verify import verify_runtime_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify an evaluation report against its runtime manifest."
    )
    parser.add_argument("--report", required=True, help="Path to evaluation report")
    parser.add_argument(
        "--manifest", required=True, help="Path to runtime.manifest.json"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON result",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = verify_runtime_manifest(args.report, args.manifest)

    payload = {
        "ok": result.ok,
        "errors": list(result.errors),
        "report": result.report,
        "manifest": result.manifest,
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    elif result.ok:
        print(f"OK {result.report}")
    else:
        print(f"FAIL {result.report}")
        for error in result.errors:
            print(f"  - {error}")

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
