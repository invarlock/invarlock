import io
import logging
import warnings
from types import SimpleNamespace

import pytest

from invarlock.cli.commands import evaluate as evaluate_mod
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
        with pytest.raises(UserWarning):
            with run_mod._suppress_noisy_warnings("dev"):
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
        with run_mod._suppress_noisy_warnings("release"):
            logger.warning("loss_type=None is unrecognized by this model")
        handler.flush()
        assert stream.getvalue() == ""
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
        logger.propagate = prev_propagate


def test_evaluate_helpers_cover_banner_and_ratio() -> None:
    lines = evaluate_mod._render_banner_lines("Title", "Context")
    assert len(lines) == 4
    assert "Title" in lines[1]
    assert "Context" in lines[2]
    assert len(lines[0]) == len(lines[1]) == len(lines[2]) == len(lines[3])
    assert evaluate_mod._format_ratio(1.23456) == "1.235"
    assert (
        evaluate_mod._resolve_verbosity(False, False) == evaluate_mod.VERBOSITY_DEFAULT
    )
    assert evaluate_mod._resolve_verbosity(True, False) == evaluate_mod.VERBOSITY_QUIET
