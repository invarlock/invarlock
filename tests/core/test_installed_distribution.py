from __future__ import annotations

from collections import Counter

import pytest

from invarlock.core import installed_distribution as versions
from invarlock.core.backend_inventory import build_backend_inventory_for_adapter
from invarlock.core.runtime_quantization_proof import (
    build_runtime_quantization_proof,
)


@pytest.fixture(autouse=True)
def clear_version_cache() -> None:
    versions._clear_installed_distribution_version_cache()
    yield
    versions._clear_installed_distribution_version_cache()


def test_repeated_proof_and_inventory_builds_share_version_cache(monkeypatch) -> None:
    calls: Counter[str] = Counter()

    def fake_version(distribution: str) -> str:
        calls[distribution] += 1
        return {"bitsandbytes": "1.2.3", "transformers": "4.5.6"}[distribution]

    monkeypatch.setattr(versions.importlib_metadata, "version", fake_version)

    for _ in range(3):
        build_runtime_quantization_proof(adapter="hf_bnb", model=None)
        build_backend_inventory_for_adapter(adapter="hf_bnb", model=None)

    assert calls == Counter({"bitsandbytes": 1, "transformers": 1})


def test_missing_distribution_result_is_cached(monkeypatch) -> None:
    calls = 0

    def missing(distribution: str) -> str:
        nonlocal calls
        calls += 1
        raise versions.importlib_metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(versions.importlib_metadata, "version", missing)

    assert versions.installed_distribution_version("not-installed") is None
    assert versions.installed_distribution_version("not-installed") is None
    assert calls == 1


def test_cache_is_separated_by_process_id(monkeypatch) -> None:
    process_id = 101
    calls: list[int] = []

    def fake_version(_distribution: str) -> str:
        calls.append(process_id)
        return f"version-for-{process_id}"

    monkeypatch.setattr(versions.os, "getpid", lambda: process_id)
    monkeypatch.setattr(versions.importlib_metadata, "version", fake_version)

    assert versions.installed_distribution_version("backend") == "version-for-101"
    assert versions.installed_distribution_version("backend") == "version-for-101"
    process_id = 202
    assert versions.installed_distribution_version("backend") == "version-for-202"
    assert calls == [101, 202]
