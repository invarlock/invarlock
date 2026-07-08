import io
import logging
import sys
import warnings
from types import SimpleNamespace

import pytest

from invarlock.cli import run_shell_output as run_output_mod
from invarlock.cli.commands import evaluate as evaluate_mod
from invarlock.cli.run_runtime_exec import suppress_noisy_warnings


def test_format_guard_chain_preserves_configured_order() -> None:
    guards = [
        SimpleNamespace(name="invariants"),
        SimpleNamespace(name="spectral"),
        SimpleNamespace(name="invariants"),
    ]
    assert (
        run_output_mod._format_guard_chain(guards)
        == "invariants \u2192 spectral \u2192 invariants"
    )


def test_device_resolution_note_variants() -> None:
    assert run_output_mod._device_resolution_note("auto", "cpu") == "auto-resolved"
    assert run_output_mod._device_resolution_note("cpu", "cpu") == "requested"
    assert (
        run_output_mod._device_resolution_note("cuda", "cuda:0") == "resolved from cuda"
    )


def test_format_kv_line_alignment() -> None:
    assert run_output_mod._format_kv_line("Device", "cpu") == "  Device    : cpu"


def test_suppress_noisy_warnings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("INVARLOCK_SUPPRESS_WARNINGS", "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with suppress_noisy_warnings("dev"):
            warnings.warn("noisy", UserWarning, stacklevel=2)
    assert caught == []


def test_suppress_noisy_warnings_passthrough(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_SUPPRESS_WARNINGS", raising=False)
    with pytest.warns(UserWarning, match="noisy"):
        with suppress_noisy_warnings("dev"):
            warnings.warn("noisy", UserWarning, stacklevel=2)


def test_suppress_noisy_warnings_dev_filters_known_messages(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_SUPPRESS_WARNINGS", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning):
            with suppress_noisy_warnings("dev"):
                warnings.warn(
                    "loss_type=None is unrecognized by this model",
                    UserWarning,
                    stacklevel=2,
                )


def test_suppress_noisy_warnings_release_suppresses_transformers_logs(
    monkeypatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_SUPPRESS_WARNINGS", raising=False)

    stream = io.StringIO()
    logger = logging.getLogger("transformers")
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.WARNING)

    prev_level = logger.level
    prev_propagate = logger.propagate
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        with suppress_noisy_warnings("release"):
            logger.warning("loss_type=None is unrecognized by this model")
        handler.flush()
        assert stream.getvalue() == ""
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
        logger.propagate = prev_propagate


def test_suppress_noisy_warnings_release_filters_stderr_output(
    monkeypatch, capsys
) -> None:
    monkeypatch.delenv("INVARLOCK_SUPPRESS_WARNINGS", raising=False)
    with suppress_noisy_warnings("release"):
        print(
            "`loss_type=None` was set in the config but it is unrecognized. "
            "Using the default loss: `ForCausalLMLoss`.",
            file=sys.stderr,
        )
    captured = capsys.readouterr()
    assert "loss_type=None" not in captured.err


def test_evaluate_helpers_cover_banner_and_ratio() -> None:
    lines = evaluate_mod._render_banner_lines("Title", "Context")
    assert len(lines) == 3
    assert lines[0] == "Title"
    assert lines[1] == "Context"
    assert set(lines[2]) == {"-"}
    assert evaluate_mod._format_ratio(1.23456) == "1.235"
    assert (
        evaluate_mod._resolve_verbosity(False, False) == evaluate_mod.VERBOSITY_DEFAULT
    )
    assert evaluate_mod._resolve_verbosity(True, False) == evaluate_mod.VERBOSITY_QUIET
