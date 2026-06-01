from __future__ import annotations

import os
from pathlib import Path

import pytest

import invarlock.runtime_security as runtime_launch_plan
import invarlock.runtime_security as runtime_security
import invarlock.runtime_security_helpers as runtime_security_helpers


def _env_value(command: list[str], key: str) -> str:
    needle = f"{key}="
    for idx, token in enumerate(command[:-1]):
        if token == "-e" and command[idx + 1].startswith(needle):
            return command[idx + 1][len(needle) :]
    raise AssertionError(f"environment variable {key} not found")


def _canonical_path(path: str | Path) -> Path:
    return Path(os.path.realpath(os.path.abspath(str(path))))


def _mounted_roots(command: list[str]) -> list[Path]:
    roots: list[Path] = []
    for idx, token in enumerate(command[:-1]):
        if token != "-v":
            continue
        host_root, _, _ = command[idx + 1].partition(":")
        roots.append(_canonical_path(host_root))
    return roots


def _path_is_mounted(command: list[str], path: str | Path) -> bool:
    target = _canonical_path(path)
    for root in _mounted_roots(command):
        try:
            target.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _build_container_command(argv: list[str]) -> list[str]:
    return runtime_security.build_container_command(
        runtime_launch_plan.build_current_process_container_launch_plan(argv)
    )


def _stub_container_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "0")
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_container_engine",
        lambda: "docker",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "container_image_available_locally",
        lambda image=None, *, engine=None: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image",
        lambda: "invarlock-runtime:local",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image_digest",
        lambda: "sha256:test",
        raising=True,
    )
