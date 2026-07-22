from __future__ import annotations

from invarlock import __main__ as module_entrypoint


def test_module_entrypoint_delegates_to_the_supported_cli(monkeypatch) -> None:  # noqa: ANN001
    called = False

    def fake_app() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(module_entrypoint, "app", fake_app)

    module_entrypoint.main()

    assert called is True
