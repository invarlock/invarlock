from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from invarlock.adapters.hf_mixin import HFAdapterMixin


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        state[name] = tensor.detach().cpu().clone()
    return state


def _assert_state_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key])


def test_restore_chunked_missing_tensor_file_raises_and_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    adapter = HFAdapterMixin()
    model = torch.nn.Linear(4, 3)

    snapshot_dir = Path(adapter.snapshot_chunked(model, prefix="test-snap-"))
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    missing_file = next(iter(manifest["params"].values()))
    (snapshot_dir / missing_file).unlink()

    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    modified = _clone_state(model)

    with pytest.raises(FileNotFoundError):
        adapter.restore_chunked(model, str(snapshot_dir))

    _assert_state_equal(_clone_state(model), modified)


def test_restore_chunked_missing_parameter_key_raises_and_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    adapter = HFAdapterMixin()
    source = torch.nn.Linear(4, 3, bias=True)
    snapshot_dir = Path(adapter.snapshot_chunked(source, prefix="test-snap-"))

    # Restore into a mismatched model (missing bias) must fail closed.
    target = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        for param in target.parameters():
            param.add_(1.0)
    modified = _clone_state(target)

    with pytest.raises(KeyError):
        adapter.restore_chunked(target, str(snapshot_dir))

    _assert_state_equal(_clone_state(target), modified)


def test_restore_bytes_shape_mismatch_raises_before_mutation() -> None:
    adapter = HFAdapterMixin()
    source = torch.nn.Linear(4, 3, bias=True)
    snapshot = adapter.snapshot(source)
    target = torch.nn.Linear(5, 3, bias=True)
    modified = _clone_state(target)

    with pytest.raises(ValueError, match="shape mismatch"):
        adapter.restore(target, snapshot)

    _assert_state_equal(_clone_state(target), modified)


def test_restore_chunked_rejects_manifest_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    adapter = HFAdapterMixin()
    model = torch.nn.Linear(4, 3)
    snapshot_dir = Path(adapter.snapshot_chunked(model, prefix="test-snap-"))
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_param = next(iter(manifest["params"]))
    manifest["params"][first_param] = "../outside.safetensors"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    modified = _clone_state(model)

    with pytest.raises(ValueError, match="Invalid snapshot manifest filename"):
        adapter.restore_chunked(model, str(snapshot_dir))

    _assert_state_equal(_clone_state(model), modified)


def test_restore_chunked_rejects_absolute_manifest_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    adapter = HFAdapterMixin()
    model = torch.nn.Linear(4, 3)
    snapshot_dir = Path(adapter.snapshot_chunked(model, prefix="test-snap-"))
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_param = next(iter(manifest["params"]))
    manifest["params"][first_param] = str(tmp_path / "outside.safetensors")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    modified = _clone_state(model)

    with pytest.raises(ValueError, match="Invalid snapshot manifest filename"):
        adapter.restore_chunked(model, str(snapshot_dir))

    _assert_state_equal(_clone_state(model), modified)


def test_restore_chunked_rejects_duplicate_manifest_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    adapter = HFAdapterMixin()
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    snapshot_dir = Path(adapter.snapshot_chunked(model, prefix="test-snap-"))
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params = list(manifest["params"])
    manifest["params"][params[1]] = manifest["params"][params[0]]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    modified = _clone_state(model)

    with pytest.raises(ValueError, match="Duplicate snapshot manifest filename"):
        adapter.restore_chunked(model, str(snapshot_dir))

    _assert_state_equal(_clone_state(model), modified)
