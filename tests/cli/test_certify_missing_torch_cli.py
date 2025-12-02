from __future__ import annotations

import builtins

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_certify_missing_torch_shows_extra_hint_via_cli(
    monkeypatch,
) -> None:
    """Ensure CLI surfaces a clear extras hint when torch is missing."""
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[override]
        if name == "torch":
            raise ModuleNotFoundError("torch not available in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "certify",
            "--baseline",
            "gpt2",
            "--subject",
            "gpt2",
        ],
    )

    assert result.exit_code != 0
    out = result.stdout
    assert "Torch is required for this command." in out
    assert "invarlock[hf]" in out
    assert "invarlock[adapters]" in out
