from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

from invarlock.cli.run_warning_filters import (
    _apply_warning_filters,
    suppress_noisy_warnings,
)


class _CaptureHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        # No-op: we only care about attached filters being exercised.
        return


def test_warning_suppression_filters_and_event_logging(
    tmp_path: Path, monkeypatch
) -> None:
    # Cover suppress-all branch in _apply_warning_filters().
    with warnings.catch_warnings():
        monkeypatch.setenv("INVARLOCK_SUPPRESS_WARNINGS", "1")
        assert _apply_warning_filters("ci") is True
    monkeypatch.delenv("INVARLOCK_SUPPRESS_WARNINGS", raising=False)

    # Cover handler de-duplication in _iter_handlers() by attaching the same handler
    # to multiple loggers.
    handler = _CaptureHandler()
    root_logger = logging.getLogger()
    tf_logger = logging.getLogger("transformers")
    root_logger.addHandler(handler)
    tf_logger.addHandler(handler)
    try:
        event_path = tmp_path / "events.jsonl"
        # Ensure the "pass-through" warning path doesn't register as a test warning
        # under pytest's warnings capture.
        real_showwarning = warnings.showwarning

        def _noop_showwarning(*_a, **_k) -> None:  # noqa: ANN001
            return

        warnings.showwarning = _noop_showwarning
        try:
            with suppress_noisy_warnings(
                "ci",
                event_path=event_path,
                context={"source": "tests"},
            ):
                noisy_filters = [
                    f
                    for f in handler.filters
                    if f.__class__.__name__ == "_NoisyLogFilter"
                ]
                assert noisy_filters
                log_filter = noisy_filters[-1]

                rec_match = logging.LogRecord(
                    name="transformers",
                    level=logging.WARNING,
                    pathname=__file__,
                    lineno=1,
                    msg="loss_type=None is unrecognized",
                    args=(),
                    exc_info=None,
                )
                rec_no_match = logging.LogRecord(
                    name="transformers",
                    level=logging.WARNING,
                    pathname=__file__,
                    lineno=1,
                    msg="all good",
                    args=(),
                    exc_info=None,
                )
                assert log_filter.filter(rec_match) is False
                assert log_filter.filter(rec_no_match) is True

                # Exercise bytes path in _FilteredStream.write().
                sys.stdout.write(b"loss_type=None unrecognized\n")
                sys.stdout.write("hello\n")

                # Exercise warning filtering branch (match + pass-through).
                warnings.showwarning(
                    "loss_type=None is unrecognized",
                    UserWarning,
                    __file__,
                    1,
                    file=None,
                    line=None,
                )
                warnings.showwarning(
                    "just a warning",
                    UserWarning,
                    __file__,
                    1,
                    file=None,
                    line=None,
                )
        finally:
            warnings.showwarning = real_showwarning
    finally:
        root_logger.removeHandler(handler)
        tf_logger.removeHandler(handler)

    lines = event_path.read_text(encoding="utf-8").splitlines()
    assert lines
    payload = json.loads(lines[-1])
    assert payload["component"] == "warnings"
    assert payload["operation"] == "suppressed"
    assert payload["data"]["count"] >= 1
