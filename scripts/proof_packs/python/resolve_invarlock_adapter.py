from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("Usage: resolve_invarlock_adapter.py <model_id_or_path>", file=sys.stderr)
        return 2

    model_id = argv[0].strip()
    if not model_id:
        return 0

    from invarlock.cli.adapter_auto import resolve_auto_adapter

    print(resolve_auto_adapter(model_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
