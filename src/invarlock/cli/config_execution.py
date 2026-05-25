from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar

from invarlock.runtime_security import (
    delegate_python_module_to_container,
    host_execution_allowed,
    running_inside_container,
    runtime_allowances_scope,
    write_runtime_manifest,
)

from .security_helpers import resolve_shell_runtime_security_policy


class RuntimeDelegationError(RuntimeError):
    """Raised when default runtime container delegation cannot start."""


@dataclass(frozen=True)
class ConfigExecutionRequest:
    """Canonical request object for config-driven run execution."""

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
    allow_unverified_provenance: bool = False
    prefer_local_files_only: bool = False

    VALUE_ARG_SPECS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("config", "--config"),
        ("device", "--device"),
        ("profile", "--profile"),
        ("out", "--out"),
        ("edit", "--edit"),
        ("edit_label", "--edit-label"),
        ("tier", "--tier"),
        ("metric_kind", "--metric-kind"),
        ("probes", "--probes"),
        ("max_attempts", "--max-attempts"),
        ("timeout", "--timeout"),
        ("baseline", "--baseline"),
        ("style", "--style"),
    )
    VALUE_ARG_KWARGS: ClassVar[dict[str, dict[str, Any]]] = {
        "config": {"aliases": ("-c",), "required": True},
        "probes": {"type": int},
        "max_attempts": {"type": int, "default": 3},
        "timeout": {"type": int},
    }
    BOOL_ARG_SPECS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("until_pass", "--until-pass"),
        ("no_cleanup", "--no-cleanup"),
        ("progress", "--progress"),
        ("timing", "--timing"),
        ("telemetry", "--telemetry"),
        ("no_color", "--no-color"),
        ("prefer_local_files_only", "--prefer-local-files-only"),
    )
    POLICY_FIELDS: ClassVar[tuple[str, ...]] = (
        "allow_network",
        "allow_host_execution",
        "allow_third_party_plugins",
        "allow_remote_code",
        "allow_unverified_provenance",
    )
    INTERNAL_ARG_DEFAULTS: ClassVar[dict[str, object]] = {"device": "auto"}
    INTERNAL_OMIT_DEFAULTS: ClassVar[dict[str, object]] = {"max_attempts": 3}

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(cls))

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> ConfigExecutionRequest:
        field_names = set(cls.field_names())
        unknown = sorted(set(kwargs) - field_names)
        if unknown:
            joined = ", ".join(unknown)
            raise TypeError(f"unknown config execution request field(s): {joined}")
        return cls(
            **{name: kwargs[name] for name in cls.field_names() if name in kwargs}
        )

    @classmethod
    def add_argparse_arguments(cls, parser: argparse.ArgumentParser) -> None:
        for attr, flag in cls.VALUE_ARG_SPECS:
            kwargs = dict(cls.VALUE_ARG_KWARGS.get(attr, {}))
            aliases = tuple(kwargs.pop("aliases", ()))
            parser.add_argument(flag, *aliases, dest=attr, **kwargs)
        for attr, flag in cls.BOOL_ARG_SPECS:
            parser.add_argument(flag, dest=attr, action="store_true")

    @classmethod
    def from_argparse(
        cls,
        args: argparse.Namespace,
    ) -> ConfigExecutionRequest:
        data: dict[str, Any] = {}
        for attr, _flag in (*cls.VALUE_ARG_SPECS, *cls.BOOL_ARG_SPECS):
            if hasattr(args, attr):
                data[attr] = getattr(args, attr)
        return cls.from_kwargs(**data)

    def runtime_policy_kwargs(self) -> dict[str, bool]:
        return {name: bool(getattr(self, name)) for name in self.POLICY_FIELDS}

    def to_internal_argv(self, command_name: str | Iterable[str]) -> list[str]:
        argv: list[str] = []
        _append_option(argv, "--invoked-command", _command_name_string(command_name))
        for attr, flag in self.VALUE_ARG_SPECS:
            value = getattr(self, attr)
            if value is None and attr in self.INTERNAL_ARG_DEFAULTS:
                value = self.INTERNAL_ARG_DEFAULTS[attr]
            if attr in self.INTERNAL_OMIT_DEFAULTS:
                default = self.INTERNAL_OMIT_DEFAULTS[attr]
                if value is None or str(value) == str(default):
                    continue
            _append_option(argv, flag, value)
        for attr, flag in self.BOOL_ARG_SPECS:
            _append_bool_flag(argv, flag, getattr(self, attr))
        return argv


def _command_name_tokens(command_name: str | Iterable[str]) -> list[str]:
    if isinstance(command_name, str):
        return [command_name]
    return [str(token) for token in command_name]


def _command_name_string(command_name: str | Iterable[str]) -> str:
    return " ".join(_command_name_tokens(command_name))


def _append_option(argv: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def _append_bool_flag(argv: list[str], flag: str, enabled: object) -> None:
    if bool(enabled):
        argv.append(flag)


def build_request_container_launch_plan(
    command_name: str | Iterable[str],
    request: ConfigExecutionRequest,
):
    from invarlock.cli.runtime_launch_plan import (
        build_request_container_launch_plan as _build_request_container_launch_plan,
    )

    return _build_request_container_launch_plan(command_name, request)


def execute_config_run_request(request: ConfigExecutionRequest) -> str | None:
    from .run_execution import (
        execute_config_run_request as _execute_config_run_request,
    )

    return _execute_config_run_request(request)


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
    allow_unverified_provenance: bool = False,
    prefer_local_files_only: bool = False,
    command_name: str | Iterable[str] = "run",
    delegate: bool = True,
) -> Path:
    """Run a config-driven job and return the emitted report path."""
    request = ConfigExecutionRequest.from_kwargs(
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
        allow_network=allow_network,
        allow_host_execution=allow_host_execution,
        allow_third_party_plugins=allow_third_party_plugins,
        allow_remote_code=allow_remote_code,
        allow_unverified_provenance=allow_unverified_provenance,
        prefer_local_files_only=prefer_local_files_only,
    )
    return run_request(request, command_name=command_name, delegate=delegate)


def run_request(
    request: ConfigExecutionRequest,
    *,
    command_name: str | Iterable[str] = "run",
    delegate: bool = True,
) -> Path:
    """Run a canonical config execution request and return the emitted report path."""
    policy = resolve_shell_runtime_security_policy(**request.runtime_policy_kwargs())
    with runtime_allowances_scope(policy=policy):
        if delegate and not running_inside_container() and not host_execution_allowed():
            try:
                exit_code = delegate_python_module_to_container(
                    "invarlock.cli.internal_config_run",
                    build_request_container_launch_plan(command_name, request),
                )
            except RuntimeError as exc:
                raise RuntimeDelegationError(str(exc)) from exc
            raise SystemExit(exit_code)

        report_path = execute_config_run_request(request)

        if report_path is None:
            raise RuntimeError("run execution did not return a report path")

        report = Path(report_path).resolve()
        if report.exists():
            manifest_command = (
                " ".join(str(token) for token in command_name)
                if not isinstance(command_name, str)
                else command_name
            )
            write_runtime_manifest(
                report,
                config_path=request.config,
                extra={
                    "command": manifest_command,
                    "profile": request.profile,
                    "allow_network": policy.allow_network,
                    "allow_remote_code": policy.allow_remote_code,
                    "allow_third_party_plugins": policy.allow_third_party_plugins,
                },
            )

        return report
