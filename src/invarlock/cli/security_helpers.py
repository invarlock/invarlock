from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, cast

import typer

from invarlock.runtime_provenance import (
    configure_runtime_security as _configure_runtime_security_core,
)
from invarlock.runtime_provenance import (
    verify_runtime_provenance as _verify_runtime_provenance_core,
)
from invarlock.runtime_security import (
    ALLOW_HOST_EXECUTION_ENV,
    ALLOW_NETWORK_ENV,
    ALLOW_REMOTE_CODE_ENV,
    ALLOW_THIRD_PARTY_PLUGINS_ENV,
    ALLOW_UNVERIFIED_PROVENANCE_ENV,
    RuntimeManifestExecution,
    RuntimeSecurityPolicy,
    build_runtime_security_policy,
    current_runtime_security_policy,
    delegate_container_command,
    running_inside_container,
    write_runtime_manifest,
)


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_shell_runtime_security_policy(
    *,
    allow_network: bool = False,
    allow_host_execution: bool = False,
    allow_third_party_plugins: bool = False,
    allow_remote_code: bool = False,
    allow_unverified_provenance: bool = False,
) -> RuntimeSecurityPolicy:
    policy = current_runtime_security_policy()
    if policy is not None:
        return policy
    return build_runtime_security_policy(
        allow_network=allow_network or _env_truthy(ALLOW_NETWORK_ENV),
        allow_host_execution=allow_host_execution
        or _env_truthy(ALLOW_HOST_EXECUTION_ENV),
        allow_third_party_plugins=allow_third_party_plugins
        or _env_truthy(ALLOW_THIRD_PARTY_PLUGINS_ENV),
        allow_remote_code=allow_remote_code or _env_truthy(ALLOW_REMOTE_CODE_ENV),
        allow_unverified_provenance=allow_unverified_provenance
        or _env_truthy(ALLOW_UNVERIFIED_PROVENANCE_ENV),
    )


@contextmanager
def configure_runtime_security(
    *,
    allow_network: bool = False,
    allow_host_execution: bool = False,
    allow_third_party_plugins: bool = False,
    allow_remote_code: bool = False,
    allow_unverified_provenance: bool = False,
) -> Iterator[None]:
    policy = resolve_shell_runtime_security_policy(
        allow_network=allow_network,
        allow_host_execution=allow_host_execution,
        allow_third_party_plugins=allow_third_party_plugins,
        allow_remote_code=allow_remote_code,
        allow_unverified_provenance=allow_unverified_provenance,
    )
    with _configure_runtime_security_core(
        allow_network=policy.allow_network,
        allow_host_execution=policy.allow_host_execution,
        allow_third_party_plugins=policy.allow_third_party_plugins,
        allow_remote_code=policy.allow_remote_code,
        allow_unverified_provenance=policy.allow_unverified_provenance,
    ):
        yield


def runtime_security_scoped(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        execution_mode = str(kwargs.get("execution_mode", "") or "").strip().lower()
        host_mode = execution_mode == "host"
        with configure_runtime_security(
            allow_network=bool(kwargs.get("allow_network", False)),
            allow_host_execution=bool(kwargs.get("allow_host_execution", False))
            or host_mode,
            allow_third_party_plugins=bool(
                kwargs.get("allow_third_party_plugins", False)
            ),
            allow_remote_code=bool(kwargs.get("allow_remote_code", False)),
            allow_unverified_provenance=bool(
                kwargs.get("allow_unverified_provenance", False)
            )
            or host_mode,
        ):
            return func(*args, **kwargs)

    return cast(Callable[..., Any], wrapper)


def build_current_process_container_launch_plan(
    argv: list[str] | None = None,
) -> Any:
    from invarlock.cli.runtime_launch_plan import (
        build_current_process_container_launch_plan as _build_current_process_container_launch_plan,
    )

    return _build_current_process_container_launch_plan(argv)


def maybe_delegate_model_command() -> None:
    policy = resolve_shell_runtime_security_policy()
    if running_inside_container() or policy.allow_host_execution:
        return
    try:
        code = delegate_container_command(build_current_process_container_launch_plan())
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    raise typer.Exit(code)


def emit_runtime_manifest(
    report_path: str | Path | None,
    *,
    config_path: str | Path | None = None,
    config_payload: Any | None = None,
    extra: dict[str, Any] | None = None,
    execution: RuntimeManifestExecution | None = None,
) -> Path | None:
    if not report_path:
        return None
    path = Path(report_path)
    if not path.exists():
        return None
    return write_runtime_manifest(
        path,
        config_path=config_path,
        config_payload=config_payload,
        extra=extra,
        execution=execution,
    )


def verify_runtime_provenance(
    report_path: str | Path,
    *,
    allow_unverified: bool = False,
) -> list[str]:
    return _verify_runtime_provenance_core(
        report_path,
        allow_unverified=allow_unverified,
    )
