from __future__ import annotations

import pytest

from invarlock.runtime_providers import _hf_transformers_identity as identity


@pytest.mark.parametrize(
    ("imported", "installed"),
    [
        ("2.11.0", "2.11.0"),
        ("2.11.0+cpu", "2.11.0"),
        ("2.11.0", "2.11.0+cpu"),
        ("2.11.0+CUDA_12-8", "2.11.0+cuda.12.8"),
    ],
)
def test_runtime_version_accepts_only_equivalent_local_build_suffixes(
    imported: str, installed: str
) -> None:
    assert identity._runtime_version_matches_distribution(imported, installed)


@pytest.mark.parametrize(
    ("imported", "installed"),
    [
        ("2.11.1+cpu", "2.11.0"),
        ("2.11.0+cu128", "2.11.0+cpu"),
        ("2.11.0+", "2.11.0"),
        ("2.11.0+cuda@12", "2.11.0"),
        ("2.11.0+cu128+debug", "2.11.0"),
        (" 2.11.0", "2.11.0"),
        (None, "2.11.0"),
    ],
)
def test_runtime_version_rejects_non_equivalent_or_invalid_versions(
    imported: object, installed: str
) -> None:
    assert not identity._runtime_version_matches_distribution(imported, installed)
