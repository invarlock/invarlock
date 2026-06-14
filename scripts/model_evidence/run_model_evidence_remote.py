#!/usr/bin/env python3
"""Launch the shipped-model evidence sweep on a remote GPU host via tmux."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REMOTE_REPO = "/root/invarlock-public"
DEFAULT_REMOTE_REPO_FALLBACKS = ("/root/invarlock-public-a100",)
DEFAULT_SUITE = "current-supported-experimental"
DEFAULT_REMOTE_VENV_CANDIDATES = (
    "{remote_repo}/.venv/bin/python",
    "/root/venvs/invarlock/bin/python",
)
EXECUTION_MODES = ("container", "host")
REMOTE_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class RemoteLaunch:
    session: str
    gpu: str
    gpu_group: tuple[str, ...]
    shard_index: int
    shard_count: int
    output_root: str
    remote_command: str
    ssh_command: list[str]

    def to_payload(self) -> dict[str, object]:
        return {
            "session": self.session,
            "gpu": self.gpu,
            "gpu_group": list(self.gpu_group),
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "output_root": self.output_root,
            "remote_command": self.remote_command,
            "ssh_command": self.ssh_command,
        }


def _shell_path(arg: str) -> str:
    return arg if arg.startswith("$") else shlex.quote(arg)


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
        help="Lane suite name passed through to scripts/model_evidence/model_evidence_sweep.py.",
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
        "--preset-override",
        action="append",
        default=[],
        metavar="SLUG=PATH",
        help=(
            "Pass a SLUG=PATH preset override through to the remote sweep. "
            "Repeat for multiple lanes."
        ),
    )
    parser.add_argument(
        "--profile",
        default=None,
        help=(
            "Optional verify/evaluate profile override passed through to the sweep "
            "script. Defaults to the sweep's lane-specific profile resolution."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed through to the sweep script.",
    )
    parser.add_argument(
        "--execution-mode",
        default="container",
        choices=EXECUTION_MODES,
        help=(
            "Execution mode passed through to scripts/model_evidence/model_evidence_sweep.py. "
            "'container' keeps the default container path; 'host' uses the "
            "explicit host-bypass matrix."
        ),
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated GPU ids. One tmux shard is launched per GPU id.",
    )
    parser.add_argument(
        "--gpu-group",
        action="append",
        default=[],
        metavar="IDS",
        help=(
            "Comma-separated GPU ids that should be exposed to one tmux shard, "
            "for example 0,1,2,3 for a sharded MoE load. Repeat for multiple "
            "groups. When set, these groups define the shards and --gpus is "
            "used only for payload visibility/backwards-compatible dry runs."
        ),
    )
    parser.add_argument(
        "--remote-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Repeat to export an extra environment variable inside each remote "
            "tmux shard, for example HF_HUB_DISABLE_XET=1."
        ),
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


def _remote_repo(remote_repo: str) -> tuple[str, list[str], list[str]]:
    candidate_paths: list[str] = []
    for candidate in (remote_repo, *DEFAULT_REMOTE_REPO_FALLBACKS):
        if candidate and candidate not in candidate_paths:
            candidate_paths.append(candidate)

    if len(candidate_paths) == 1:
        return remote_repo, [], candidate_paths

    setup = [
        'REPO_DIR=""',
        "for candidate in "
        + " ".join(shlex.quote(path) for path in candidate_paths)
        + '; do if [ -d "$candidate/.git" ] || [ -f "$candidate/.git" ]; then REPO_DIR="$candidate"; break; fi; done',
        f'if [ -z "$REPO_DIR" ]; then REPO_DIR={shlex.quote(remote_repo)}; fi',
    ]
    return "$REPO_DIR", setup, candidate_paths


def _remote_python(
    remote_repo: str,
    remote_repo_candidates: list[str],
    remote_python: str | None,
) -> tuple[str, list[str], list[str]]:
    if remote_python:
        return remote_python, [], [remote_python]
    candidate_paths: list[str] = []
    for template in DEFAULT_REMOTE_VENV_CANDIDATES:
        if "{remote_repo}" in template:
            candidate_paths.append(template.format(remote_repo=remote_repo.rstrip("/")))
        else:
            candidate_paths.append(template)
    display_candidates = [
        f"{candidate.rstrip('/')}/.venv/bin/python"
        for candidate in remote_repo_candidates
    ]
    display_candidates.append("/root/venvs/invarlock/bin/python")
    display_candidates = list(dict.fromkeys(display_candidates))
    setup = [
        'PYTHON_BIN=""',
        "for candidate in "
        + " ".join(_shell_path(path) for path in candidate_paths)
        + '; do if [ -x "$candidate" ]; then PYTHON_BIN="$candidate"; break; fi; done',
        'if [ -z "$PYTHON_BIN" ] && command -v python3.12 >/dev/null 2>&1; then PYTHON_BIN="$(command -v python3.12)"; fi',
        'if [ -z "$PYTHON_BIN" ] && command -v python3 >/dev/null 2>&1; then PYTHON_BIN="$(command -v python3)"; fi',
        'if [ -z "$PYTHON_BIN" ]; then echo "No remote Python runtime found" >&2; exit 127; fi',
    ]
    return "$PYTHON_BIN", setup, display_candidates


def _stamp(value: str | None) -> str:
    return value or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _parse_gpus(raw: str) -> list[str]:
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU id must be supplied via --gpus")
    return gpus


def _parse_gpu_groups(
    raw_items: list[str], fallback_gpus: list[str]
) -> list[tuple[str, ...]]:
    if not raw_items:
        return [(gpu,) for gpu in fallback_gpus]
    groups: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for raw in raw_items:
        group = tuple(item.strip() for item in raw.split(",") if item.strip())
        if not group:
            raise ValueError("--gpu-group entries must contain at least one GPU id")
        if group in seen:
            raise ValueError("--gpu-group entries must be unique")
        seen.add(group)
        groups.append(group)
    return groups


def _gpu_group_label(group: tuple[str, ...]) -> str:
    return "-".join(re.sub(r"[^A-Za-z0-9_.-]+", "-", item) for item in group)


def _flatten_gpu_groups(gpu_groups: list[tuple[str, ...]]) -> list[str]:
    return list(dict.fromkeys(gpu for group in gpu_groups for gpu in group))


def _parse_remote_env(raw_items: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for raw in raw_items:
        if "=" not in raw:
            raise ValueError("--remote-env entries must use KEY=VALUE")
        key, value = raw.split("=", 1)
        if not REMOTE_ENV_KEY_RE.fullmatch(key):
            raise ValueError(f"--remote-env key is not a valid shell name: {key!r}")
        parsed.append((key, value))
    return parsed


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
    repo_setup: list[str],
    python_setup: list[str],
) -> str:
    quoted_branch = shlex.quote(branch)
    origin_ref = f"origin/{branch}"
    quoted_origin_ref = shlex.quote(origin_ref)
    quoted_fetch_refspec = shlex.quote(
        f"refs/heads/{branch}:refs/remotes/{origin_ref}"
    )
    return " && ".join(
        [
            *repo_setup,
            f"cd {_shell_path(remote_repo)}",
            f"git fetch origin {quoted_fetch_refspec}",
            (
                f"git checkout {quoted_branch} "
                f"|| git checkout -b {quoted_branch} --track {quoted_origin_ref}"
            ),
            f"git merge --ff-only {quoted_origin_ref}",
            *python_setup,
            _shell_command(
                [remote_python, "scripts/checks/sync_packaged_contracts.py", "--check"]
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
    preset_overrides: list[str],
    profile: str | None,
    device: str,
    execution_mode: str,
    gpu_groups: list[tuple[str, ...]],
    remote_env: list[tuple[str, str]],
    session_prefix: str,
    stamp: str,
    repo_setup: list[str],
    python_setup: list[str],
) -> list[RemoteLaunch]:
    launches: list[RemoteLaunch] = []
    shard_count = len(gpu_groups)
    for shard_index, gpu_group in enumerate(gpu_groups):
        gpu = ",".join(gpu_group)
        gpu_label = _gpu_group_label(gpu_group)
        session = f"{session_prefix}-{stamp}-g{gpu_label}"
        shard_output_root = (
            f"{remote_output_root.rstrip('/')}/shard-{shard_index:02d}-gpu-{gpu_label}"
        )
        sweep_cmd = [
            remote_python,
            "scripts/model_evidence/model_evidence_sweep.py",
            "--suite",
            suite,
            "--device",
            device,
            "--execution-mode",
            execution_mode,
            "--output-root",
            shard_output_root,
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(shard_count),
        ]
        if profile:
            sweep_cmd[4:4] = ["--profile", profile]
        for slug in slugs:
            sweep_cmd.extend(["--slug", slug])
        for lane_id in lane_ids:
            sweep_cmd.extend(["--lane-id", lane_id])
        for preset_override in preset_overrides:
            sweep_cmd.extend(["--preset-override", preset_override])

        export_pairs = [
            ("PYTHONPATH", "src"),
            ("INVARLOCK_ALLOW_NETWORK", "1"),
            *remote_env,
            ("CUDA_VISIBLE_DEVICES", gpu),
        ]
        export_command = "export " + " ".join(
            f"{key}={shlex.quote(value)}" for key, value in export_pairs
        )

        inner_command = " && ".join(
            [
                *repo_setup,
                f"cd {_shell_path(remote_repo)}",
                f"mkdir -p {shlex.quote(shard_output_root)}",
                *python_setup,
                export_command,
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
                gpu_group=gpu_group,
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
    remote_repo, repo_setup, remote_repo_candidates = _remote_repo(args.remote_repo)
    remote_python, python_setup, remote_python_candidates = _remote_python(
        remote_repo, remote_repo_candidates, args.remote_python
    )
    remote_output_root = args.remote_output_root or _default_remote_output_root(stamp)
    gpus = _parse_gpus(args.gpus)
    gpu_groups = _parse_gpu_groups(args.gpu_group, gpus)
    payload_gpus = _flatten_gpu_groups(gpu_groups)
    remote_env = _parse_remote_env(args.remote_env)

    sync_command = None
    if not args.skip_sync:
        sync_command = build_sync_command(
            remote_repo=remote_repo,
            branch=args.branch,
            remote_python=remote_python,
            repo_setup=repo_setup,
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
        preset_overrides=args.preset_override,
        profile=args.profile,
        device=args.device,
        execution_mode=args.execution_mode,
        gpu_groups=gpu_groups,
        remote_env=remote_env,
        session_prefix=args.session_prefix,
        stamp=stamp,
        repo_setup=repo_setup,
        python_setup=python_setup,
    )
    payload = {
        "host": args.host,
        "branch": args.branch,
        "remote_repo": args.remote_repo,
        "remote_repo_candidates": remote_repo_candidates,
        "remote_python": args.remote_python or "auto",
        "remote_python_candidates": remote_python_candidates,
        "remote_output_root": remote_output_root,
        "suite": args.suite,
        "execution_mode": args.execution_mode,
        "gpus": payload_gpus,
        "gpu_groups": [list(group) for group in gpu_groups],
        "remote_env": [{"name": key, "value": value} for key, value in remote_env],
        "preset_overrides": args.preset_override,
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
