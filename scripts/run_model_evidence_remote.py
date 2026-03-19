#!/usr/bin/env python3
"""Launch the shipped-model evidence sweep on a remote GPU host via tmux."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE_REPO = "/root/invarlock-public"
DEFAULT_SUITE = "current-supported-experimental"
DEFAULT_REMOTE_VENV_CANDIDATES = (
    "{remote_repo}/.venv/bin/python",
    "/root/venvs/invarlock/bin/python",
)


@dataclass(frozen=True)
class RemoteLaunch:
    session: str
    gpu: str
    shard_index: int
    shard_count: int
    output_root: str
    remote_command: str
    ssh_command: list[str]

    def to_payload(self) -> dict[str, object]:
        return {
            "session": self.session,
            "gpu": self.gpu,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "output_root": self.output_root,
            "remote_command": self.remote_command,
            "ssh_command": self.ssh_command,
        }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the maintained shipped-model evidence sweep on a remote host."
    )
    parser.add_argument("--host", required=True, help="Remote SSH target.")
    parser.add_argument(
        "--remote-repo",
        default=DEFAULT_REMOTE_REPO,
        help="Remote repository checkout path.",
    )
    parser.add_argument(
        "--branch",
        default="staging/next",
        help="Git branch to fast-forward on the remote checkout.",
    )
    parser.add_argument(
        "--remote-python",
        default=None,
        help="Remote Python executable. Defaults to <remote-repo>/.venv/bin/python.",
    )
    parser.add_argument(
        "--remote-output-root",
        default=None,
        help="Remote output root. Defaults to /root/model-evidence/<stamp>.",
    )
    parser.add_argument(
        "--suite",
        default=DEFAULT_SUITE,
        help="Lane suite name passed through to scripts/model_evidence_sweep.py.",
    )
    parser.add_argument(
        "--slug",
        action="append",
        default=[],
        help="Repeat to restrict the sweep to specific manifest slugs.",
    )
    parser.add_argument(
        "--lane-id",
        action="append",
        default=[],
        help="Repeat to restrict the sweep to specific support_matrix lane_ids.",
    )
    parser.add_argument(
        "--profile",
        default="ci",
        help="Verify/evaluate profile passed through to the sweep script.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed through to the sweep script.",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated GPU ids. One tmux shard is launched per GPU id.",
    )
    parser.add_argument(
        "--session-prefix",
        default="model-evidence",
        help="tmux session name prefix.",
    )
    parser.add_argument(
        "--stamp",
        default=None,
        help="Deterministic timestamp suffix for sessions/output roots.",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip remote git fast-forward and packaged-contract sync check.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launch plan as JSON and exit.",
    )
    return parser.parse_args(argv)


def _remote_python(
    remote_repo: str, remote_python: str | None
) -> tuple[str, list[str], list[str]]:
    if remote_python:
        return remote_python, [], [remote_python]
    candidate_paths = [
        template.format(remote_repo=remote_repo.rstrip("/"))
        for template in DEFAULT_REMOTE_VENV_CANDIDATES
    ]
    setup = [
        'PYTHON_BIN=""',
        "for candidate in "
        + " ".join(shlex.quote(path) for path in candidate_paths)
        + '; do if [ -x "$candidate" ]; then PYTHON_BIN="$candidate"; break; fi; done',
        'if [ -z "$PYTHON_BIN" ] && command -v python3.12 >/dev/null 2>&1; then PYTHON_BIN="$(command -v python3.12)"; fi',
        'if [ -z "$PYTHON_BIN" ] && command -v python3 >/dev/null 2>&1; then PYTHON_BIN="$(command -v python3)"; fi',
        'if [ -z "$PYTHON_BIN" ]; then echo "No remote Python runtime found" >&2; exit 127; fi',
    ]
    return "$PYTHON_BIN", setup, candidate_paths


def _stamp(value: str | None) -> str:
    return value or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _parse_gpus(raw: str) -> list[str]:
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU id must be supplied via --gpus")
    return gpus


def _shell_join(args: list[str]) -> str:
    return shlex.join(args)


def _shell_command(args: list[str]) -> str:
    rendered: list[str] = []
    for idx, arg in enumerate(args):
        if idx == 0 and arg.startswith("$"):
            rendered.append(arg)
        else:
            rendered.append(shlex.quote(arg))
    return " ".join(rendered)


def build_sync_command(
    *,
    remote_repo: str,
    branch: str,
    remote_python: str,
    python_setup: list[str],
) -> str:
    return " && ".join(
        [
            f"cd {shlex.quote(remote_repo)}",
            "git fetch origin",
            f"git checkout {shlex.quote(branch)}",
            f"git pull --ff-only origin {shlex.quote(branch)}",
            *python_setup,
            _shell_command(
                [remote_python, "scripts/sync_packaged_contracts.py", "--check"]
            ),
        ]
    )


def build_launches(
    *,
    host: str,
    remote_repo: str,
    remote_python: str,
    remote_output_root: str,
    suite: str,
    slugs: list[str],
    lane_ids: list[str],
    profile: str,
    device: str,
    gpus: list[str],
    session_prefix: str,
    stamp: str,
    python_setup: list[str],
) -> list[RemoteLaunch]:
    launches: list[RemoteLaunch] = []
    shard_count = len(gpus)
    for shard_index, gpu in enumerate(gpus):
        session = f"{session_prefix}-{stamp}-g{gpu}"
        shard_output_root = (
            f"{remote_output_root.rstrip('/')}/shard-{shard_index:02d}-gpu-{gpu}"
        )
        sweep_cmd = [
            remote_python,
            "scripts/model_evidence_sweep.py",
            "--suite",
            suite,
            "--profile",
            profile,
            "--device",
            device,
            "--output-root",
            shard_output_root,
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(shard_count),
        ]
        for slug in slugs:
            sweep_cmd.extend(["--slug", slug])
        for lane_id in lane_ids:
            sweep_cmd.extend(["--lane-id", lane_id])

        inner_command = " && ".join(
            [
                f"cd {shlex.quote(remote_repo)}",
                f"mkdir -p {shlex.quote(shard_output_root)}",
                *python_setup,
                (
                    "export PYTHONPATH=src INVARLOCK_ALLOW_NETWORK=1 "
                    f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)}"
                ),
                _shell_command(sweep_cmd),
            ]
        )
        remote_command = (
            f"tmux new-session -d -s {shlex.quote(session)} "
            f"bash -lc {shlex.quote(inner_command)}"
        )
        launches.append(
            RemoteLaunch(
                session=session,
                gpu=gpu,
                shard_index=shard_index,
                shard_count=shard_count,
                output_root=shard_output_root,
                remote_command=remote_command,
                ssh_command=["ssh", host, remote_command],
            )
        )
    return launches


def _default_remote_output_root(stamp: str) -> str:
    return f"/root/model-evidence/{stamp}"


def _monitor_commands(host: str, launches: list[RemoteLaunch]) -> dict[str, object]:
    sessions = [launch.session for launch in launches]
    return {
        "tmux_list": ["ssh", host, "tmux list-sessions"],
        "captures": [
            {
                "session": session,
                "command": [
                    "ssh",
                    host,
                    f"tmux capture-pane -pt {shlex.quote(session)} -S -80",
                ],
            }
            for session in sessions
        ],
    }


def run_remote(args: argparse.Namespace) -> int:
    stamp = _stamp(args.stamp)
    remote_repo = args.remote_repo
    remote_python, python_setup, remote_python_candidates = _remote_python(
        remote_repo, args.remote_python
    )
    remote_output_root = args.remote_output_root or _default_remote_output_root(stamp)
    gpus = _parse_gpus(args.gpus)

    sync_command = None
    if not args.skip_sync:
        sync_command = build_sync_command(
            remote_repo=remote_repo,
            branch=args.branch,
            remote_python=remote_python,
            python_setup=python_setup,
        )

    launches = build_launches(
        host=args.host,
        remote_repo=remote_repo,
        remote_python=remote_python,
        remote_output_root=remote_output_root,
        suite=args.suite,
        slugs=args.slug,
        lane_ids=args.lane_id,
        profile=args.profile,
        device=args.device,
        gpus=gpus,
        session_prefix=args.session_prefix,
        stamp=stamp,
        python_setup=python_setup,
    )
    payload = {
        "host": args.host,
        "branch": args.branch,
        "remote_repo": remote_repo,
        "remote_python": args.remote_python or "auto",
        "remote_python_candidates": remote_python_candidates,
        "remote_output_root": remote_output_root,
        "suite": args.suite,
        "gpus": gpus,
        "sync_command": sync_command,
        "launches": [launch.to_payload() for launch in launches],
        "monitor": _monitor_commands(args.host, launches),
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if sync_command is not None:
        sync_proc = subprocess.run(
            ["ssh", args.host, sync_command],
            cwd=REPO_ROOT,
            check=False,
            text=True,
        )
        if sync_proc.returncode != 0:
            return sync_proc.returncode

    for launch in launches:
        proc = subprocess.run(launch.ssh_command, cwd=REPO_ROOT, check=False, text=True)
        if proc.returncode != 0:
            return proc.returncode

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        return run_remote(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
