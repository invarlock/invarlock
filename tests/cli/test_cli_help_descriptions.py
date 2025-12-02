from collections.abc import Iterable

import typer
from typer.main import get_command


def _iter_commands(cmd: "typer.main.TyperCommand") -> Iterable[tuple[str, object]]:
    # Root level
    for name, sub in cmd.commands.items():  # type: ignore[attr-defined]
        yield name, sub
        # Recurse into groups
        try:
            subcommands = getattr(sub, "commands", None)
            if isinstance(subcommands, dict):
                for subname, subsub in subcommands.items():
                    yield f"{name} {subname}", subsub
        except Exception:  # pragma: no cover - defensive
            continue


def test_all_commands_have_help(monkeypatch):
    # Keep import light to avoid plugin discovery
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")

    # Import the app lazily after setting env
    from invarlock.cli.app import app as invarlock_app

    # Convert Typer app to Click command tree
    root = get_command(invarlock_app)

    # Sanity: root must be a group with help
    assert getattr(root, "help", None), "Root invarlock app should have help text"

    missing: list[str] = []
    for name, sub in _iter_commands(root):
        help_text = getattr(sub, "help", None) or getattr(sub, "short_help", None)
        if not (isinstance(help_text, str) and help_text.strip()):
            missing.append(name)

    assert not missing, "Commands missing help: " + ", ".join(sorted(missing))
