from __future__ import annotations

import argparse
from collections.abc import Sequence

from invarlock.cli.config_execution import ConfigExecutionRequest, run_request


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m invarlock.cli.internal_config_run",
        description="Package-internal delegated config runner.",
    )
    parser.add_argument(
        "--invoked-command",
        default="run",
        help="Logical command name recorded in runtime manifests.",
    )
    ConfigExecutionRequest.add_argparse_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_request(
        ConfigExecutionRequest.from_argparse(args),
        command_name=str(args.invoked_command),
        delegate=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
