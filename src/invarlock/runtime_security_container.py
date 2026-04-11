from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _helpers():
    from invarlock import runtime_security_helpers as helpers

    return helpers


def build_container_command(plan: Any) -> list[str]:
    helpers = _helpers()
    context = helpers._resolve_container_launch_context(plan)
    command: list[str] = helpers._compose_container_run_args(
        context,
        plan,
        argv=tuple(plan.argv),
    )
    return command


def delegate_container_command(plan: Any) -> int:
    helpers = _helpers()
    command = helpers.build_container_command(plan)
    try:
        completed = helpers.subprocess.run(
            command,
            check=False,
            timeout=helpers._CONTAINER_EXECUTION_TIMEOUT_SECONDS,
        )
    except helpers.subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Container execution timed out after "
            f"{helpers._CONTAINER_EXECUTION_TIMEOUT_SECONDS} seconds."
        ) from exc
    return int(completed.returncode)


def build_container_python_command(
    script_path: str | os.PathLike[str],
    plan: Any,
) -> list[str]:
    helpers = _helpers()
    context = helpers._resolve_container_launch_context(plan)
    cwd = context.cwd
    script_host_path = helpers._absolute_host_path(Path(script_path), cwd=cwd)
    script_mounts: set[Path] = set()
    if helpers._record_path_dependencies(script_host_path, script_mounts, cwd=cwd):
        container_script = helpers._workspace_path(script_host_path, cwd=cwd)
    else:
        container_script = str(script_host_path)
    command: list[str] = helpers._compose_container_run_args(
        context,
        plan,
        entrypoint=("--entrypoint", "python"),
        extra_mounts=tuple(script_mounts),
        argv=(container_script, *plan.argv),
    )
    return command


def build_container_python_module_command(
    module_name: str,
    plan: Any,
) -> list[str]:
    helpers = _helpers()
    context = helpers._resolve_container_launch_context(plan)
    command: list[str] = helpers._compose_container_run_args(
        context,
        plan,
        entrypoint=("--entrypoint", "python"),
        argv=("-m", module_name, *plan.argv),
    )
    return command


def delegate_python_script_to_container(
    script_path: str | os.PathLike[str],
    plan: Any,
) -> int:
    helpers = _helpers()
    command = helpers.build_container_python_command(script_path, plan)
    try:
        completed = helpers.subprocess.run(
            command,
            check=False,
            timeout=helpers._CONTAINER_EXECUTION_TIMEOUT_SECONDS,
        )
    except helpers.subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Container execution timed out after "
            f"{helpers._CONTAINER_EXECUTION_TIMEOUT_SECONDS} seconds."
        ) from exc
    return int(completed.returncode)


def delegate_python_module_to_container(
    module_name: str,
    plan: Any,
) -> int:
    helpers = _helpers()
    command = helpers.build_container_python_module_command(module_name, plan)
    try:
        completed = helpers.subprocess.run(
            command,
            check=False,
            timeout=helpers._CONTAINER_EXECUTION_TIMEOUT_SECONDS,
        )
    except helpers.subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Container execution timed out after "
            f"{helpers._CONTAINER_EXECUTION_TIMEOUT_SECONDS} seconds."
        ) from exc
    return int(completed.returncode)
