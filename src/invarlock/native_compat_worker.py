"""Explicit, bounded entry point for Python-only model and extension compatibility.

The Rust control command owns process lifetime. This worker retains the installed
Python CLI and SDK contracts while explicitly selecting the native portable core.
It is never discovered or launched by an offline verification operation.
"""

from __future__ import annotations

import sys


def main() -> None:
    from invarlock.evidence_pack_json import parse_json_bytes

    raw = sys.stdin.buffer.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("compatibility request exceeds its byte limit")
    request = parse_json_bytes(raw, label="Python compatibility request")
    if (
        not isinstance(request, dict)
        or set(request) != {"format", "argv"}
        or request["format"] != "invarlock/python-worker-v1"
    ):
        raise ValueError("invalid Python compatibility request")
    argv = request["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or len(argv) > 4096
        or any(not isinstance(x, str) or "\0" in x for x in argv)
        or argv[0] not in {"evaluate", "verify", "report"}
    ):
        raise ValueError("invalid Python compatibility arguments")
    if any(x.split("=", 1)[0] in {"--backend", "--execution-backend"} for x in argv):
        raise ValueError(
            "compatibility worker owns the explicit native backend selection"
        )
    from invarlock.native_backend import require_native

    require_native()
    option = "--execution-backend" if argv[0] == "evaluate" else "--backend"
    from invarlock.cli.app import app

    app(args=[*argv, option, "rust"], prog_name="invarlock")


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        print(f"Python compatibility failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
