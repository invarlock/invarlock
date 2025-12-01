from __future__ import annotations

from invarlock.reporting.certificate import validate_certificate


def test_validate_certificate_returns_false_for_invalid_payload():
    # Missing required blocks should yield False (no exception)
    assert validate_certificate({}) is False
