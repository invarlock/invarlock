"""CLI command package.

Lightweight namespace re-exports for programmatic access in tests and tooling.
Submodules are imported lazily so importing one command does not eagerly import
the full CLI surface.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "doctor_command": ("invarlock.cli.commands.doctor", "doctor_command"),
    "evaluate_command": ("invarlock.cli.commands.evaluate", "evaluate_command"),
    "explain_gates_command": (
        "invarlock.cli.commands.explain_gates",
        "explain_gates_command",
    ),
    "export_html_command": (
        "invarlock.cli.commands.export_html",
        "export_html_command",
    ),
    "plugins_command": ("invarlock.cli.commands.plugins", "plugins_command"),
    "policy_build_command": ("invarlock.cli.commands.policy", "build_command"),
    "policy_verify_command": ("invarlock.cli.commands.policy", "verify_command"),
    "proof_pack_build_command": (
        "invarlock.cli.commands.proof_pack",
        "build_command",
    ),
    "proof_pack_inspect_command": (
        "invarlock.cli.commands.proof_pack",
        "inspect_command",
    ),
    "proof_pack_verify_command": (
        "invarlock.cli.commands.proof_pack",
        "verify_command",
    ),
    "report_command": ("invarlock.cli.commands.report", "report_command"),
    "run_command": ("invarlock.cli.commands.run", "run_command"),
    "verify_command": ("invarlock.cli.commands.verify", "verify_command"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
