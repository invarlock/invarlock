from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_download_baseline_module():
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root / "scripts" / "evidence_packs" / "python" / "download_baseline.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evidence_pack_download_baseline", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_huggingface_hub(
    monkeypatch,
    *,
    list_repo_files,
    snapshot_download,
) -> None:
    module = types.ModuleType("huggingface_hub")
    module.list_repo_files = list_repo_files
    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


def test_download_snapshot_copy_prefers_safetensors(
    monkeypatch, tmp_path: Path
) -> None:
    download_baseline = _load_download_baseline_module()
    calls: list[dict[str, object]] = []

    def fake_list_repo_files(
        repo_id: str, *, repo_type: str, revision: str | None
    ) -> list[str]:
        assert repo_id == "org/model"
        assert repo_type == "model"
        assert revision is None
        return [
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        ]

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        local_dir = Path(str(kwargs["local_dir"]))
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}\n", encoding="utf-8")
        return str(local_dir)

    _stub_huggingface_hub(
        monkeypatch,
        list_repo_files=fake_list_repo_files,
        snapshot_download=fake_snapshot_download,
    )

    output_dir = tmp_path / "baseline"
    weight_format = download_baseline.download_snapshot(
        "org/model", output_dir, "snapshot_copy", None
    )

    assert weight_format == "safetensors"
    assert len(calls) == 1
    call = calls[0]
    assert call["local_dir"] == str(output_dir)
    assert call["ignore_patterns"] == ["*.bin", "*.bin.index.json"]
    assert "local_dir_use_symlinks" not in call


def test_download_snapshot_symlink_uses_cache_tree_and_copy_on_write_generation_config(
    monkeypatch, tmp_path: Path
) -> None:
    download_baseline = _load_download_baseline_module()
    calls: list[dict[str, object]] = []
    snapshot_dir = tmp_path / "hf-cache" / "snapshots" / "abc123"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot_dir / "weights.safetensors").write_text("weights\n", encoding="utf-8")
    (snapshot_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "do_sample": False,
                "temperature": 0.7,
                "top_p": 0.8,
            }
        ),
        encoding="utf-8",
    )

    def fake_list_repo_files(
        repo_id: str, *, repo_type: str, revision: str | None
    ) -> list[str]:
        assert repo_id == "org/model"
        assert repo_type == "model"
        assert revision == "rev1"
        return ["config.json", "generation_config.json", "weights.safetensors"]

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot_dir)

    _stub_huggingface_hub(
        monkeypatch,
        list_repo_files=fake_list_repo_files,
        snapshot_download=fake_snapshot_download,
    )

    output_dir = tmp_path / "baseline"
    weight_format = download_baseline.download_snapshot(
        "org/model", output_dir, "snapshot_symlink", "rev1"
    )

    assert weight_format == "safetensors"
    assert len(calls) == 1
    call = calls[0]
    assert "local_dir" not in call
    assert "local_dir_use_symlinks" not in call
    assert (output_dir / "config.json").is_symlink()
    assert (output_dir / "weights.safetensors").is_symlink()
    assert (output_dir / "generation_config.json").is_symlink()

    download_baseline.sanitize_generation_config(output_dir)

    sanitized = json.loads((output_dir / "generation_config.json").read_text())
    cached = json.loads((snapshot_dir / "generation_config.json").read_text())
    assert not (output_dir / "generation_config.json").is_symlink()
    assert sanitized["temperature"] is None
    assert sanitized["top_p"] is None
    assert cached["temperature"] == 0.7
    assert cached["top_p"] == 0.8
