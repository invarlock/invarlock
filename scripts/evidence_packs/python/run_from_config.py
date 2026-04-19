from __future__ import annotations

import argparse
import sys
from pathlib import Path

from invarlock.cli.config_execution import RuntimeDelegationError, run_from_config
from invarlock.cli.runtime_launch_plan import (
    build_current_process_container_launch_plan,
)
from invarlock.cli.security_helpers import resolve_shell_runtime_security_policy
from invarlock.runtime_security import (
    apply_runtime_allowances,
    delegate_python_script_to_container,
    host_execution_allowed,
    running_inside_container,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repo-only evidence-pack config runner."
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
    parser.add_argument("--until-pass", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--baseline")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--style")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--allow-host-execution", action="store_true")
    parser.add_argument("--allow-third-party-plugins", action="store_true")
    parser.add_argument("--allow-remote-code", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    return parser.parse_args(argv)


def _delegated_argv(args: argparse.Namespace, argv: list[str]) -> list[str]:
    delegated = list(argv)
    if args.device is not None:
        return delegated
    if "--device" in delegated or any(
        token.startswith("--device=") for token in delegated
    ):
        return delegated
    delegated.extend(["--device", "auto"])
    return delegated


def _delegate_if_needed(args: argparse.Namespace, argv: list[str]) -> int | None:
    policy = resolve_shell_runtime_security_policy(
        allow_network=bool(args.allow_network),
        allow_host_execution=bool(args.allow_host_execution),
        allow_third_party_plugins=bool(args.allow_third_party_plugins),
        allow_remote_code=bool(args.allow_remote_code),
    )
    apply_runtime_allowances(policy=policy)
    if running_inside_container() or host_execution_allowed():
        return None
    delegated_argv = _delegated_argv(args, argv)
    try:
        return delegate_python_script_to_container(
            Path(__file__),
            build_current_process_container_launch_plan(delegated_argv),
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parsed_argv = list(argv) if argv is not None else sys.argv[1:]
    args = _parse_args(parsed_argv)

    delegated_exit = _delegate_if_needed(args, parsed_argv)
    if delegated_exit is not None:
        return delegated_exit

    try:
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
            allow_network=bool(args.allow_network),
            allow_host_execution=bool(args.allow_host_execution),
            allow_third_party_plugins=bool(args.allow_third_party_plugins),
            allow_remote_code=bool(args.allow_remote_code),
            command_name="evidence-pack-run",
            delegate=False,
        )
    except RuntimeDelegationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
