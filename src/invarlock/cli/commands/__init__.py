"""CLI command package.

Lightweight namespace re-exports for programmatic access in tests and tooling.
Import-time work is minimal; subcommands themselves may perform heavier imports
only when invoked.
"""

from .doctor import doctor_command
from .evaluate import evaluate_command
from .explain_gates import explain_gates_command
from .export_html import export_html_command
from .plugins import plugins_command
from .policy import build_command as policy_build_command
from .policy import verify_command as policy_verify_command
from .report import report_command
from .run import run_command
from .verify import verify_command

__all__ = [
    "evaluate_command",
    "doctor_command",
    "explain_gates_command",
    "export_html_command",
    "policy_build_command",
    "policy_verify_command",
    "plugins_command",
    "run_command",
    "verify_command",
    "report_command",
]
