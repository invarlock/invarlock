import sys

import pytest

from invarlock import security
from invarlock.cli.app import main


def test_cli_main_invokes_app(monkeypatch):
    # Ensure no args so Typer shows help and exits
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    argv_backup = sys.argv[:]
    sys.argv = [argv_backup[0]]
    try:
        with pytest.raises(SystemExit):
            main()
    finally:
        # Restore argv and network policy so subsequent tests are unaffected
        sys.argv = argv_backup
        security.enforce_network_policy(True)
