from __future__ import annotations

import pytest

from invarlock.core.runtime_provider import (
    ExactMatchOutputError,
    exact_match_output_text,
)


def test_exact_match_output_contract_preserves_user_visible_bytes() -> None:
    text = "  caf\N{LATIN SMALL LETTER E WITH ACUTE}\r\nanswer\t "

    assert exact_match_output_text(text) == text
    assert exact_match_output_text(text).encode("utf-8") == (
        b"  caf\xc3\xa9\r\nanswer\t "
    )


@pytest.mark.parametrize("value", [None, b"answer", 7, "\ud800"])
def test_exact_match_output_contract_rejects_non_utf8_text(value: object) -> None:
    with pytest.raises(ExactMatchOutputError):
        exact_match_output_text(value)


def test_exact_match_output_contract_rejects_ambiguous_backend_marker() -> None:
    with pytest.raises(ExactMatchOutputError, match="backend-control marker"):
        exact_match_output_text(
            "answer [end of text]\n",
            forbidden_backend_markers=(" [end of text]\n",),
        )


@pytest.mark.parametrize("markers", [("",), ("marker", "marker")])
def test_exact_match_output_contract_rejects_invalid_marker_contract(
    markers: tuple[str, ...],
) -> None:
    with pytest.raises(ExactMatchOutputError, match="backend-control markers"):
        exact_match_output_text("answer", forbidden_backend_markers=markers)
