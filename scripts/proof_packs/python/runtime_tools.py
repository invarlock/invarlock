from __future__ import annotations

import argparse
import datetime
import os

UTC = getattr(datetime, "UTC", datetime.timezone.utc)  # noqa: UP017


def iso_to_epoch(iso: str) -> int:
    iso = iso.strip()
    if not iso or iso == "null":
        return 0
    try:
        dt = datetime.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except Exception:
        return 0
    return int(dt.timestamp())


def now_iso_plus_seconds(seconds: int) -> str:
    dt = datetime.datetime.now(UTC) + datetime.timedelta(seconds=seconds)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def require_remote_code_opt_in(context: str) -> bool:
    if env_truthy("INVARLOCK_ALLOW_REMOTE_CODE"):
        return True
    raise RuntimeError(
        f"{context} requires INVARLOCK_ALLOW_REMOTE_CODE=1 before using "
        "trust_remote_code=True."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proof-pack runtime helpers.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    iso_parser = subparsers.add_parser(
        "iso-to-epoch", help="Convert ISO8601 UTC to epoch."
    )
    iso_parser.add_argument("iso", help="e.g. 2025-01-01T00:00:10Z")

    now_parser = subparsers.add_parser(
        "now-iso-plus-seconds", help="Return now() + delta seconds as ISO8601 UTC."
    )
    now_parser.add_argument("seconds", type=int)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.cmd == "iso-to-epoch":
        print(iso_to_epoch(str(args.iso)))
        return 0
    if args.cmd == "now-iso-plus-seconds":
        print(now_iso_plus_seconds(int(args.seconds)))
        return 0
    raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
