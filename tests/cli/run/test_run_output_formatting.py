import warnings
from types import SimpleNamespace

import pytest

from invarlock.cli.commands import certify as certify_mod
from invarlock.cli.commands import run as run_mod


def test_format_guard_chain_dedupes_names() -> None:
    guards = [
        SimpleNamespace(name="invariants"),
        SimpleNamespace(name="spectral"),
        SimpleNamespace(name="invariants"),
    ]
    assert run_mod._format_guard_chain(guards) == "invariants \u2192 spectral"


def test_device_resolution_note_variants() -> None:
    assert run_mod._device_resolution_note("auto", "cpu") == "auto-resolved"
    assert run_mod._device_resolution_note("cpu", "cpu") == "requested"
    assert run_mod._device_resolution_note("cuda", "cuda:0") == "resolved from cuda"


def test_format_kv_line_alignment() -> None:
    assert run_mod._format_kv_line("Device", "cpu") == "  Device    : cpu"


def test_suppress_noisy_warnings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("INVARLOCK_SUPPRESS_WARNINGS", "1")
    with run_mod._suppress_noisy_warnings("dev"):
        warnings.warn("noisy", UserWarning, stacklevel=2)


def test_suppress_noisy_warnings_passthrough(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_SUPPRESS_WARNINGS", raising=False)
    with run_mod._suppress_noisy_warnings("dev"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.warn("noisy", UserWarning, stacklevel=2)


def test_suppress_noisy_warnings_dev_filters_known_messages(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_SUPPRESS_WARNINGS", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with run_mod._suppress_noisy_warnings("dev"):
            warnings.warn(
                "loss_type=None is unrecognized by this model",
                UserWarning,
                stacklevel=2,
            )
            with pytest.raises(UserWarning):
                warnings.warn("some other warning", UserWarning, stacklevel=2)


def test_certify_helpers_cover_banner_and_ratio() -> None:
    lines = certify_mod._render_banner_lines("Title", "Context")
    assert len(lines) == 4
    assert "Title" in lines[1]
    assert "Context" in lines[2]
    assert len(lines[0]) == len(lines[1]) == len(lines[2]) == len(lines[3])
    assert certify_mod._format_ratio(1.23456) == "1.235"
    assert certify_mod._resolve_verbosity(False, False) == certify_mod.VERBOSITY_DEFAULT
    assert certify_mod._resolve_verbosity(True, False) == certify_mod.VERBOSITY_QUIET
