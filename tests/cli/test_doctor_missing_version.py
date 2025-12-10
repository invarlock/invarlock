import types

import pytest

from invarlock.cli.commands import doctor


def test_doctor_handles_torch_without_version(monkeypatch):
    """Ensure doctor tolerates torch modules lacking __version__."""

    # Stub a minimal torch module without __version__
    fake_torch = types.SimpleNamespace()
    monkeypatch.setitem(doctor.sys.modules, "torch", fake_torch)

    # Avoid pulling real device info (which expects functional torch)
    monkeypatch.setattr("invarlock.cli.device.get_device_info", lambda: {})

    # Should exit cleanly (typer.Exit wraps SystemExit with code 0)
    import typer

    with pytest.raises(typer.Exit) as excinfo:
        doctor.doctor_command(json_out=True)
    assert excinfo.value.exit_code == 0
