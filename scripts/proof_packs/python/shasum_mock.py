from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-a", dest="algo", default="")
    parser.add_argument("-c", dest="check_file", default="")
    parser.add_argument("files", nargs="*")
    return parser.parse_args(argv)


def _check(check_file: Path) -> int:
    lines = check_file.read_text().splitlines()
    ok = True
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        expected = parts[0]
        filename = parts[-1]
        actual = _sha256(Path(filename))
        if actual != expected:
            ok = False
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check_file:
        return _check(Path(args.check_file))

    for filename in args.files:
        digest = _sha256(Path(filename))
        print(f"{digest}  {filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
