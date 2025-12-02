def test_cli_errors_reexport_matches_core() -> None:
    from invarlock.cli.errors import InvarlockError as CLIInvarlockError
    from invarlock.core.exceptions import InvarlockError as CoreInvarlockError

    assert CLIInvarlockError is CoreInvarlockError
