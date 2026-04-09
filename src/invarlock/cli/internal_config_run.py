from __future__ import annotations

import argparse
from collections.abc import Sequence

from invarlock.cli.config_execution import run_from_config


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
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--device")
    parser.add_argument("--profile")
    parser.add_argument("--out")
    parser.add_argument("--edit")
    parser.add_argument("--edit-label")
    parser.add_argument("--tier")
    parser.add_argument("--metric-kind")
    parser.add_argument("--probes", type=int)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--baseline")
    parser.add_argument("--style")
    parser.add_argument("--until-pass", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--prefer-local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_from_config(
        config=args.config,
        device=args.device,
        profile=args.profile,
        out=args.out,
        edit=args.edit,
        edit_label=args.edit_label,
        tier=args.tier,
        metric_kind=args.metric_kind,
        probes=args.probes,
        until_pass=bool(args.until_pass),
        max_attempts=int(args.max_attempts),
        timeout=args.timeout,
        baseline=args.baseline,
        no_cleanup=bool(args.no_cleanup),
        style=args.style,
        progress=bool(args.progress),
        timing=bool(args.timing),
        telemetry=bool(args.telemetry),
        no_color=bool(args.no_color),
        prefer_local_files_only=bool(args.prefer_local_files_only),
        command_name=str(args.invoked_command),
        delegate=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
