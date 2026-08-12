from importlib import import_module

cli_app = import_module("invarlock.cli.app")


def test_installed_entrypoint_delegates_to_the_supported_cli(monkeypatch) -> None:  # noqa: ANN001
    calls = 0

    def fake_app() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(cli_app, "app", fake_app)

    cli_app.main()

    assert calls == 1
