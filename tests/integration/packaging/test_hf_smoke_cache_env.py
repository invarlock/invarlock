from __future__ import annotations

import sys
from pathlib import Path

from tests.integration.packaging import _support_installed_wheel as support


def test_resolve_hf_smoke_env_prefers_ready_shared_cache(
    tmp_path: Path, monkeypatch
) -> None:
    shared_cache = tmp_path / "shared-hf"
    monkeypatch.setenv("HF_HOME", str(shared_cache))
    monkeypatch.setattr(
        support, "_hf_cache_root_candidates", lambda: [shared_cache, tmp_path / "other"]
    )
    monkeypatch.setattr(support, "_hf_cache_root_is_writable", lambda root: True)
    monkeypatch.setattr(
        support,
        "_local_hf_smoke_cache_ready",
        lambda python_exe, hf_home: hf_home == shared_cache,
    )

    env, local_cache_ready = support._resolve_hf_smoke_env(
        Path(sys.executable), tmp_path
    )

    assert local_cache_ready is True
    assert env["HF_HOME"] == str(shared_cache)
    assert env["HF_HUB_CACHE"] == str(shared_cache / "hub")
    assert env["HF_DATASETS_CACHE"] == str(shared_cache / "datasets")
    assert env["DISABLE_SAFETENSORS_CONVERSION"] == "1"


def test_resolve_hf_smoke_env_reuses_writable_shared_cache_before_tmp_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    shared_cache = tmp_path / "shared-hf"
    monkeypatch.setattr(support, "_hf_cache_root_candidates", lambda: [shared_cache])
    monkeypatch.setattr(
        support, "_local_hf_smoke_cache_ready", lambda python_exe, hf_home: False
    )
    monkeypatch.setattr(support, "_hf_cache_root_is_writable", lambda root: True)

    env, local_cache_ready = support._resolve_hf_smoke_env(
        Path(sys.executable), tmp_path
    )

    assert local_cache_ready is False
    assert env["HF_HOME"] == str(shared_cache)
    assert env["HF_DATASETS_CACHE"] == str(shared_cache / "datasets")
    assert env["DISABLE_SAFETENSORS_CONVERSION"] == "1"


def test_resolve_hf_smoke_env_falls_back_to_tmp_when_no_shared_cache_is_writable(
    tmp_path: Path, monkeypatch
) -> None:
    read_only = tmp_path / "read-only"
    monkeypatch.setattr(support, "_hf_cache_root_candidates", lambda: [read_only])
    monkeypatch.setattr(
        support, "_local_hf_smoke_cache_ready", lambda python_exe, hf_home: False
    )
    monkeypatch.setattr(support, "_hf_cache_root_is_writable", lambda root: False)

    env, local_cache_ready = support._resolve_hf_smoke_env(
        Path(sys.executable), tmp_path
    )

    assert local_cache_ready is False
    assert env["HF_HOME"] == str(tmp_path / ".hf")
    assert env["HF_HUB_CACHE"] == str(tmp_path / ".hf" / "hub")
    assert env["HF_DATASETS_CACHE"] == str(tmp_path / ".hf" / "datasets")
    assert env["DISABLE_SAFETENSORS_CONVERSION"] == "1"
