from __future__ import annotations

from pathlib import Path

import pytest

from invarlock.cli import run_config as run_config_mod
from invarlock.cli import run_runtime_exec as run_runtime_exec_mod
from invarlock.core.checkpoint_identity import (
    CheckpointIdentityError,
    checkpoint_tree_sha256,
)
from invarlock.core.config_runtime import InvarLockConfig
from invarlock.core.exceptions import InvarlockError


class DummyKwAdapter:
    def __init__(self):
        self.calls: list[tuple[str, str | None, dict]] = []

    def load_model(self, model_id: str, device: str | None = None, **kwargs):
        self.calls.append((model_id, device, kwargs))
        return object()


class DummyNoKwAdapter:
    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []

    def load_model(self, model_id: str, device: str | None = None):
        self.calls.append((model_id, device))
        return object()


class ExplodingAdapter:
    def load_model(self, model_id: str, device: str | None = None, **kwargs):
        raise RuntimeError(f"load failed for {model_id} on {device}")


@pytest.mark.unit
def test_extract_model_load_kwargs_excludes_core_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("INVARLOCK_ALLOW_REMOTE_CODE", "1")
    cfg = InvarLockConfig(
        {
            "model": {
                "id": "foo",
                "adapter": "dummy",
                "device": "cuda",
                "baseline_id": "baseline/model",
                "subject_id": "subject/model",
                "dtype": "float16",
                "trust_remote_code": True,
            }
        }
    )

    assert run_config_mod.extract_model_load_kwargs(
        cfg,
        invarlock_error_cls=InvarlockError,
    ) == {
        "dtype": "float16",
        "trust_remote_code": True,
    }


@pytest.mark.unit
def test_extract_model_load_kwargs_rejects_removed_keys():
    cfg = InvarLockConfig(
        {
            "model": {
                "id": "foo",
                "adapter": "dummy",
                "device": "cuda",
                "torch_dtype": "float16",
            }
        }
    )

    with pytest.raises(InvarlockError) as excinfo:
        _ = run_config_mod.extract_model_load_kwargs(
            cfg,
            invarlock_error_cls=InvarlockError,
        )

    assert excinfo.value.code == "E007"
    assert excinfo.value.details.get("removed_keys") == ["torch_dtype"]


@pytest.mark.unit
def test_extract_model_load_kwargs_rejects_removed_dtype_alias_values() -> None:
    cfg = InvarLockConfig(
        {
            "model": {
                "id": "foo",
                "adapter": "dummy",
                "device": "cuda",
                "dtype": "bf16",
            }
        }
    )

    with pytest.raises(InvarlockError) as excinfo:
        _ = run_config_mod.extract_model_load_kwargs(
            cfg,
            invarlock_error_cls=InvarlockError,
        )

    assert excinfo.value.code == "E007"
    assert excinfo.value.details == {
        "removed_values": ["model.dtype=bf16"],
        "replacement": "model.dtype=bfloat16",
    }


@pytest.mark.unit
def test_load_model_with_cfg_passes_all_kwargs_to_var_kw_adapter(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("INVARLOCK_ALLOW_REMOTE_CODE", "1")
    cfg = InvarLockConfig(
        {"model": {"id": "foo", "adapter": "dummy", "trust_remote_code": True}}
    )
    adapter = DummyKwAdapter()

    _ = run_runtime_exec_mod.load_model_with_cfg(adapter, cfg, "cpu")

    assert adapter.calls == [("foo", "cpu", {"trust_remote_code": True})]


@pytest.mark.unit
def test_load_model_with_cfg_derives_remote_revision_from_typed_identity() -> None:
    revision = "a" * 40
    cfg = InvarLockConfig(
        {
            "model": {
                "id": "org/model",
                "adapter": "dummy",
                "model_identity": {
                    "kind": "remote_revision",
                    "revision": revision,
                },
            }
        }
    )
    adapter = DummyKwAdapter()

    _ = run_runtime_exec_mod.load_model_with_cfg(adapter, cfg, "cpu")

    assert adapter.calls == [("org/model", "cpu", {"revision": revision})]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("legacy_field", "legacy_value"),
    [
        ("revision", "a" * 40),
        ("model_revision", "a" * 40),
        ("model_checkpoint_tree_sha256", "sha256:" + "b" * 64),
    ],
)
def test_load_model_with_cfg_rejects_legacy_identity_fields(
    legacy_field: str,
    legacy_value: str,
) -> None:
    cfg = InvarLockConfig(
        {
            "model": {
                "id": "org/model",
                "adapter": "dummy",
                legacy_field: legacy_value,
                "model_identity": {
                    "kind": "remote_revision",
                    "revision": "a" * 40,
                },
            }
        }
    )

    with pytest.raises(CheckpointIdentityError, match="legacy model identity field"):
        run_runtime_exec_mod.load_model_with_cfg(DummyKwAdapter(), cfg, "cpu")


@pytest.mark.unit
def test_load_model_with_cfg_passes_local_files_only_to_var_kw_adapter():
    cfg = InvarLockConfig({"model": {"id": "foo", "adapter": "dummy"}})
    adapter = DummyKwAdapter()

    _ = run_runtime_exec_mod.load_model_with_cfg(
        adapter,
        cfg,
        "cpu",
        prefer_local_files_only=True,
    )

    assert adapter.calls == [("foo", "cpu", {"prefer_local_files_only": True})]


@pytest.mark.unit
def test_load_model_with_cfg_filters_unknown_kwargs_for_strict_adapter(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("INVARLOCK_ALLOW_REMOTE_CODE", "1")
    cfg = InvarLockConfig(
        {"model": {"id": "foo", "adapter": "dummy", "trust_remote_code": True}}
    )
    adapter = DummyNoKwAdapter()

    _ = run_runtime_exec_mod.load_model_with_cfg(adapter, cfg, "cpu")

    assert adapter.calls == [("foo", "cpu")]


@pytest.mark.unit
def test_load_model_with_cfg_omits_local_files_only_for_strict_adapter():
    cfg = InvarLockConfig({"model": {"id": "foo", "adapter": "dummy"}})
    adapter = DummyNoKwAdapter()

    _ = run_runtime_exec_mod.load_model_with_cfg(
        adapter,
        cfg,
        "cpu",
        prefer_local_files_only=True,
    )

    assert adapter.calls == [("foo", "cpu")]


@pytest.mark.unit
def test_load_model_with_cfg_propagates_adapter_load_failures() -> None:
    cfg = InvarLockConfig({"model": {"id": "foo", "adapter": "dummy"}})

    with pytest.raises(RuntimeError, match="load failed for foo on cpu"):
        run_runtime_exec_mod.load_model_with_cfg(
            ExplodingAdapter(),
            cfg,
            "cpu",
            prefer_local_files_only=True,
        )


@pytest.mark.unit
def test_load_model_with_cfg_rechecks_local_tree_after_model_load(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"before")
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    digest = checkpoint_tree_sha256(checkpoint)
    cfg = InvarLockConfig(
        {
            "model": {
                "id": str(checkpoint),
                "adapter": "dummy",
                "model_identity": {
                    "kind": "local_checkpoint_tree",
                    "sha256": digest,
                },
            }
        }
    )

    class MutatingAdapter:
        def load_model(self, model_id: str, device: str | None = None, **kwargs):
            del model_id, device, kwargs
            weights.write_bytes(b"after")
            return object()

    with pytest.raises(CheckpointIdentityError, match="changed during model loading"):
        run_runtime_exec_mod.load_model_with_cfg(MutatingAdapter(), cfg, "cpu")


@pytest.mark.unit
def test_load_model_with_cfg_preserves_bound_tree_for_noop_loader(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"stable")
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "model_profile.json").write_text(
        '{"model_id":"local/model"}\n', encoding="utf-8"
    )
    digest = checkpoint_tree_sha256(checkpoint)
    cfg = InvarLockConfig(
        {
            "model": {
                "id": str(checkpoint),
                "adapter": "dummy",
                "model_identity": {
                    "kind": "local_checkpoint_tree",
                    "sha256": digest,
                },
            }
        }
    )

    adapter = DummyKwAdapter()
    loaded = run_runtime_exec_mod.load_model_with_cfg(adapter, cfg, "cpu")

    assert loaded is not None
    assert adapter.calls == [(str(checkpoint), "cpu", {})]
    assert checkpoint_tree_sha256(checkpoint) == digest


@pytest.mark.unit
def test_load_model_with_cfg_rejects_preload_tree_substitution(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"before")
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    digest = checkpoint_tree_sha256(checkpoint)
    weights.write_bytes(b"substituted")
    cfg = InvarLockConfig(
        {
            "model": {
                "id": str(checkpoint),
                "adapter": "dummy",
                "model_identity": {
                    "kind": "local_checkpoint_tree",
                    "sha256": digest,
                },
            }
        }
    )

    with pytest.raises(CheckpointIdentityError, match="before model loading"):
        run_runtime_exec_mod.load_model_with_cfg(DummyKwAdapter(), cfg, "cpu")


@pytest.mark.unit
def test_load_model_with_cfg_rejects_tree_swap_and_revert_during_load(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"trusted")
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    digest = checkpoint_tree_sha256(checkpoint)
    held = tmp_path / "held-checkpoint"
    malicious = tmp_path / "malicious-checkpoint"
    malicious.mkdir()
    (malicious / "model.safetensors").write_bytes(b"malicious")
    (malicious / "config.json").write_text("{}\n", encoding="utf-8")
    cfg = InvarLockConfig(
        {
            "model": {
                "id": str(checkpoint),
                "adapter": "dummy",
                "model_identity": {
                    "kind": "local_checkpoint_tree",
                    "sha256": digest,
                },
            }
        }
    )

    class SwapAndRevertAdapter:
        def load_model(self, model_id: str, device: str | None = None, **kwargs):
            del device, kwargs
            model_path = Path(model_id)
            model_path.replace(held)
            malicious.replace(model_path)
            assert (model_path / "model.safetensors").read_bytes() == b"malicious"
            model_path.replace(malicious)
            held.replace(model_path)
            return object()

    with pytest.raises(
        CheckpointIdentityError,
        match="checkpoint tree changed during model loading",
    ):
        run_runtime_exec_mod.load_model_with_cfg(SwapAndRevertAdapter(), cfg, "cpu")


@pytest.mark.unit
def test_extract_model_load_kwargs_rejects_remote_code_without_explicit_allow(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("INVARLOCK_ALLOW_REMOTE_CODE", raising=False)
    cfg = InvarLockConfig(
        {"model": {"id": "foo", "adapter": "dummy", "trust_remote_code": True}}
    )

    with pytest.raises(InvarlockError) as excinfo:
        run_config_mod.extract_model_load_kwargs(
            cfg,
            invarlock_error_cls=InvarlockError,
        )

    assert excinfo.value.code == "E008"


@pytest.mark.unit
def test_extract_model_load_kwargs_allows_remote_code_with_explicit_allow(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("INVARLOCK_ALLOW_REMOTE_CODE", "1")
    cfg = InvarLockConfig(
        {"model": {"id": "foo", "adapter": "dummy", "trust_remote_code": True}}
    )

    assert run_config_mod.extract_model_load_kwargs(
        cfg,
        invarlock_error_cls=InvarlockError,
    ) == {"trust_remote_code": True}
