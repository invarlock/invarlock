from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from source_repo_metadata import SourceRepoMetadataError, build_source_repo_payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write proof-pack source repository metadata."
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = build_source_repo_payload()
    except SourceRepoMetadataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
