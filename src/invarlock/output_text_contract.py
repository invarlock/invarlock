"""Provider-neutral contract for exact-match output text."""

from __future__ import annotations

from collections.abc import Iterable


class ExactMatchOutputError(ValueError):
    """Raised when backend output cannot be treated as user-visible text."""


def exact_match_output_text(
    value: object,
    *,
    forbidden_backend_markers: Iterable[str] = (),
) -> str:
    """Return byte-exact user-visible text or fail closed.

    Providers must remove termination and other backend-control tokens before
    calling this function.  The function deliberately performs no trimming,
    Unicode normalization, newline conversion, or other content rewriting.
    A provider that can only observe an ambiguous textual control marker must
    name that marker and reject the result instead of guessing whether to strip
    it.
    """

    if not isinstance(value, str):
        raise ExactMatchOutputError("exact-match output must be text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ExactMatchOutputError(
            "exact-match output must be representable as strict UTF-8"
        ) from exc
    observed_markers: set[str] = set()
    for marker in forbidden_backend_markers:
        if not isinstance(marker, str) or not marker:
            raise ExactMatchOutputError(
                "backend-control markers must be non-empty text"
            )
        if marker in observed_markers:
            raise ExactMatchOutputError(
                "backend-control markers must not contain duplicates"
            )
        observed_markers.add(marker)
        if marker in value:
            raise ExactMatchOutputError(
                "exact-match output contains an ambiguous backend-control marker"
            )
    return value


__all__ = ["ExactMatchOutputError", "exact_match_output_text"]
