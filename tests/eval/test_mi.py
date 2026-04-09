"""Compatibility shim for split MI probe tests."""

if __name__ == "__main__":
    import pytest

    raise SystemExit(
        pytest.main(
            [
                "tests/eval/test_mi_compute.py",
                "tests/eval/test_mi_scores.py",
                "tests/eval/test_mi_coverage.py",
            ]
        )
    )
