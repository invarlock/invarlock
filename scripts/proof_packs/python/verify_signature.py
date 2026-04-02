from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from invarlock.proof_pack_integrity import verify_signature


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a package-native proof-pack signature bundle."
    )
    parser.add_argument("pack_dir", help="Path to the proof-pack directory.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail closed when manifest.signature.json is missing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    errors, warnings, signer_fingerprint = verify_signature(
        Path(args.pack_dir),
        strict=args.strict,
    )
    for warning in warnings:
        print(warning, file=sys.stderr)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    if signer_fingerprint:
        print(signer_fingerprint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
