from __future__ import annotations

from pathlib import Path

import pytest

from tests.cli.test_container_default_contract import (
    _build_container_command,
    _env_value,
    _path_is_mounted,
    _stub_container_launch,
)


def test_container_launch_forwards_reviewed_runtime_env_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    config_path = repo_dir / "config.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")
    export_dir = repo_dir / "exports"
    export_dir.mkdir()
    eval_tmp_dir = repo_dir / "tmp-eval"
    eval_tmp_dir.mkdir()
    config_root = tmp_path / "config-root"
    config_root.mkdir()
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    hub_cache = hf_home / "hub"
    hub_cache.mkdir()
    datasets_cache = hf_home / "datasets"
    datasets_cache.mkdir()
    transformers_cache = tmp_path / "transformers-cache"
    transformers_cache.mkdir()
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()

    monkeypatch.chdir(repo_dir)
    _stub_container_launch(monkeypatch)
    monkeypatch.setenv("INVARLOCK_CONFIG_ROOT", str(config_root))
    monkeypatch.setenv("INVARLOCK_EVALUATE_TMP_DIR", str(eval_tmp_dir))
    monkeypatch.setenv("INVARLOCK_EXPORT_DIR", str(export_dir))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(datasets_cache))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(transformers_cache))
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setenv("INVARLOCK_STORE_EVAL_WINDOWS", "1")
    monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "auto")
    monkeypatch.setenv("INVARLOCK_SKIP_OVERHEAD_CHECK", "1")
    monkeypatch.setenv("INVARLOCK_DETERMINISM", "strict")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")

    command = _build_container_command(["run", "--config", str(config_path)])

    assert _path_is_mounted(command, config_root)
    assert _path_is_mounted(command, hf_home)
    assert _path_is_mounted(command, transformers_cache)
    assert _path_is_mounted(command, tmpdir)

    assert _env_value(command, "INVARLOCK_CONFIG_ROOT") == str(config_root)
    assert _env_value(command, "INVARLOCK_EVALUATE_TMP_DIR") == "/workspace/tmp-eval"
    assert _env_value(command, "INVARLOCK_EXPORT_DIR") == "/workspace/exports"
    assert _env_value(command, "HF_HOME") == str(hf_home)
    assert _env_value(command, "HF_HUB_CACHE") == str(hub_cache)
    assert _env_value(command, "HF_DATASETS_CACHE") == str(datasets_cache)
    assert _env_value(command, "TRANSFORMERS_CACHE") == str(transformers_cache)
    assert _env_value(command, "TMPDIR") == str(tmpdir)
    assert _env_value(command, "INVARLOCK_STORE_EVAL_WINDOWS") == "1"
    assert _env_value(command, "INVARLOCK_SNAPSHOT_MODE") == "auto"
    assert _env_value(command, "INVARLOCK_SKIP_OVERHEAD_CHECK") == "1"
    assert _env_value(command, "INVARLOCK_DETERMINISM") == "strict"
    assert _env_value(command, "HF_DATASETS_OFFLINE") == "1"


def test_container_launch_scans_config_includes_and_absolute_references(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    dataset_dir = external_root / "dataset"
    dataset_dir.mkdir()
    include_path = external_root / "include.yaml"
    include_path.write_text(
        "dataset:\n"
        f"  file: {dataset_dir / 'corpus.jsonl'}\n"
        "model:\n"
        "  id: sshleifer/tiny-gpt2\n"
        "  adapter: hf_causal\n",
        encoding="utf-8",
    )
    config_path = repo_dir / "config.yaml"
    config_path.write_text(
        f"defaults: !include ../external/{include_path.name}\nedit:\n  name: noop\n  plan: {{}}\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo_dir)
    _stub_container_launch(monkeypatch)
    monkeypatch.setenv("INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE", "1")

    command = _build_container_command(["run", "--config", "config.yaml"])

    assert _path_is_mounted(command, external_root)
    assert _env_value(command, "INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE") == "1"


def test_container_launch_fails_closed_when_config_scan_rejects_include(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    include_path = external_root / "include.yaml"
    include_path.write_text("model: {}\n", encoding="utf-8")
    config_path = repo_dir / "config.yaml"
    config_path.write_text(
        f"defaults: !include ../external/{include_path.name}\nedit:\n  name: noop\n  plan: {{}}\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo_dir)
    _stub_container_launch(monkeypatch)

    with pytest.raises(RuntimeError, match="Delegated runtime config"):
        _build_container_command(["run", "--config", "config.yaml"])
