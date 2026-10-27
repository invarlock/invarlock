"""Byte ownership and corrected arithmetic at the installed native boundary."""

import json

import invarlock_native as native
import pytest

from invarlock.evaluation_comparison.comparison import compare_runs
from tests.evaluation_comparison.test_numeric_range import (
    OVERFLOW_CASES,
    scalar_request,
)


@pytest.mark.parametrize("baseline,subject", OVERFLOW_CASES)
def test_corrected_numeric_range_errors(baseline, subject):
    b, s, policy = scalar_request(baseline, subject)
    with pytest.raises(
        ValueError, match="metric arithmetic exceeded finite numeric range"
    ):
        compare_runs(b, s, policy)
    raw = json.dumps({"baseline": b, "subject": s, "policy": policy}).encode()
    with pytest.raises(
        ValueError, match="metric arithmetic exceeded finite numeric range"
    ) as raised:
        native.compare_runs_json(raw)
    assert raised.value.reason == "invalid_evaluation_records"


@pytest.mark.parametrize("factory", [bytes, bytearray, list, memoryview])
def test_buffer_input_compatibility(factory):
    raw = b'{"z":2,"a":[true,"hello"]}'
    assert native.canonicalize_json(factory(raw)) == b'{"a":[true,"hello"],"z":2}\n'
    assert (
        native.canonicalize_json(factory(raw), newline=False)
        == b'{"a":[true,"hello"],"z":2}'
    )
    assert json.loads(native.dispatch_json("canonicalize", factory(raw)))[
        "payload"
    ] == json.loads(raw)


def test_large_buffer_retains_exact_output():
    raw = b'{"value":"' + b"x" * (2 * 1024 * 1024) + b'"}'
    assert native.canonicalize_json(raw) == raw + b"\n"


@pytest.mark.parametrize("raw", [b"{", bytearray(b"{"), [123], memoryview(b"{")])
def test_buffer_errors_preserve_message(raw):
    with pytest.raises(ValueError) as expected:
        native.canonicalize_json(b"{")
    with pytest.raises(ValueError) as actual:
        native.canonicalize_json(raw)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize(
    "baseline,subject",
    [
        ([1, 2], [2, 4]),
        ([0, 0], [1e307, -1e307]),
        ([-5e307, -5e307], [5e307, 5e307]),
        ([2**60 + 1], [2**60]),
        ([2**53, 2**53 + 2], [2**53, 2**53]),
    ],
)
def test_finite_control_report_bytes(baseline, subject):
    from invarlock.evaluation_record_contracts.contracts import _canonical_chunks

    b, s, policy = scalar_request(baseline, subject)
    raw = json.dumps({"baseline": b, "subject": s, "policy": policy}).encode()
    assert native.compare_runs_json(raw) == b"".join(
        _canonical_chunks(compare_runs(b, s, policy))
    )


@pytest.mark.parametrize("baseline,subject", OVERFLOW_CASES[:4])
def test_corrected_cli_refuses_without_publishing(tmp_path, baseline, subject):
    from typer.testing import CliRunner

    from invarlock.cli.app import app
    from tests._evaluation_support import materialize_captured_request

    outcomes = []
    for backend in ("python", "rust"):
        directory = (tmp_path / backend).resolve()
        directory.mkdir()
        request = materialize_captured_request(
            directory, *scalar_request(baseline, subject)
        )
        result = CliRunner().invoke(
            app,
            ["evaluate", str(request), "--unsigned", "--json", "--backend", backend],
        )
        assert result.exit_code == 2, result.output
        assert not (directory / "evidence").exists()
        payload = json.loads(result.stdout)
        assert payload["ok"] is False
        assert "finite numeric range" in " ".join(payload["errors"])
        outcomes.append(payload)
    assert outcomes[0] == outcomes[1]
