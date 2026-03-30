from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from invarlock.cli.runtime_launch_plan import build_request_container_launch_plan
from invarlock.runtime_security import (
    delegate_container_command,
    host_execution_allowed,
    running_inside_container,
    runtime_allowances_scope,
    write_runtime_manifest,
)

from .run_execution import execute_config_run_request
from .security_helpers import resolve_shell_runtime_security_policy


class RuntimeDelegationError(RuntimeError):
    """Raised when secure-default container delegation cannot start."""


@dataclass(frozen=True)
class ConfigExecutionRequest:
    config: str
    device: str | None = None
    profile: str | None = None
    out: str | None = None
    edit: str | None = None
    edit_label: str | None = None
    tier: str | None = None
    metric_kind: str | None = None
    probes: int | None = None
    until_pass: bool = False
    max_attempts: int = 3
    timeout: int | None = None
    baseline: str | None = None
    no_cleanup: bool = False
    style: str | None = None
    progress: bool = False
    timing: bool = False
    telemetry: bool = False
    no_color: bool = False
    allow_network: bool = False
    allow_host_execution: bool = False
    allow_third_party_plugins: bool = False
    allow_remote_code: bool = False
    prefer_local_files_only: bool = False


def run_from_config(
    *,
    config: str,
    device: str | None = None,
    profile: str | None = None,
    out: str | None = None,
    edit: str | None = None,
    edit_label: str | None = None,
    tier: str | None = None,
    metric_kind: str | None = None,
    probes: int | None = None,
    until_pass: bool = False,
    max_attempts: int = 3,
    timeout: int | None = None,
    baseline: str | None = None,
    no_cleanup: bool = False,
    style: str | None = None,
    progress: bool = False,
    timing: bool = False,
    telemetry: bool = False,
    no_color: bool = False,
    allow_network: bool = False,
    allow_host_execution: bool = False,
    allow_third_party_plugins: bool = False,
    allow_remote_code: bool = False,
    prefer_local_files_only: bool = False,
    command_name: str = "run",
    delegate: bool = True,
) -> Path:
    """Run a config-driven job and return the emitted report path."""
    policy = resolve_shell_runtime_security_policy(
        allow_network=allow_network,
        allow_host_execution=allow_host_execution,
        allow_third_party_plugins=allow_third_party_plugins,
        allow_remote_code=allow_remote_code,
    )
    with runtime_allowances_scope(policy=policy):
        if delegate and not running_inside_container() and not host_execution_allowed():
            try:
                exit_code = delegate_container_command(
                    build_request_container_launch_plan(
                        command_name,
                        ConfigExecutionRequest(
                            config=config,
                            device=device,
                            profile=profile,
                            out=out,
                            edit=edit,
                            edit_label=edit_label,
                            tier=tier,
                            metric_kind=metric_kind,
                            probes=probes,
                            until_pass=until_pass,
                            max_attempts=max_attempts,
                            timeout=timeout,
                            baseline=baseline,
                            no_cleanup=no_cleanup,
                            style=style,
                            progress=progress,
                            timing=timing,
                            telemetry=telemetry,
                            no_color=no_color,
                            prefer_local_files_only=prefer_local_files_only,
                        ),
                    )
                )
            except RuntimeError as exc:
                raise RuntimeDelegationError(str(exc)) from exc
            raise SystemExit(exit_code)

        request = ConfigExecutionRequest(
            config=config,
            device=device,
            profile=profile,
            out=out,
            edit=edit,
            edit_label=edit_label,
            tier=tier,
            metric_kind=metric_kind,
            probes=probes,
            until_pass=until_pass,
            max_attempts=max_attempts,
            timeout=timeout,
            baseline=baseline,
            no_cleanup=no_cleanup,
            style=style,
            progress=progress,
            timing=timing,
            telemetry=telemetry,
            no_color=no_color,
            prefer_local_files_only=prefer_local_files_only,
        )

        report_path = execute_config_run_request(request)

        if report_path is None:
            raise RuntimeError("run execution did not return a report path")

        report = Path(report_path).resolve()
        if report.exists():
            write_runtime_manifest(
                report,
                config_path=config,
                extra={
                    "command": command_name,
                    "profile": profile,
                    "allow_network": policy.allow_network,
                    "allow_remote_code": policy.allow_remote_code,
                    "allow_third_party_plugins": policy.allow_third_party_plugins,
                },
            )

        return report
