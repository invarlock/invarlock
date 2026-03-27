from __future__ import annotations

import pytest

from invarlock.cli.commands import run as run_mod
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
                "dtype": "float16",
                "trust_remote_code": True,
            }
        }
    )

    assert run_mod._extract_model_load_kwargs(cfg) == {
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
        _ = run_mod._extract_model_load_kwargs(cfg)

    assert excinfo.value.code == "E007"
    assert excinfo.value.details.get("removed_keys") == ["torch_dtype"]


@pytest.mark.unit
def test_load_model_with_cfg_passes_all_kwargs_to_var_kw_adapter(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("INVARLOCK_ALLOW_REMOTE_CODE", "1")
    cfg = InvarLockConfig(
        {"model": {"id": "foo", "adapter": "dummy", "trust_remote_code": True}}
    )
    adapter = DummyKwAdapter()

    _ = run_mod._load_model_with_cfg(adapter, cfg, "cpu")

    assert adapter.calls == [("foo", "cpu", {"trust_remote_code": True})]


@pytest.mark.unit
def test_load_model_with_cfg_filters_unknown_kwargs_for_strict_adapter(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("INVARLOCK_ALLOW_REMOTE_CODE", "1")
    cfg = InvarLockConfig(
        {"model": {"id": "foo", "adapter": "dummy", "trust_remote_code": True}}
    )
    adapter = DummyNoKwAdapter()

    _ = run_mod._load_model_with_cfg(adapter, cfg, "cpu")

    assert adapter.calls == [("foo", "cpu")]


@pytest.mark.unit
def test_extract_model_load_kwargs_rejects_remote_code_without_explicit_allow(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("INVARLOCK_ALLOW_REMOTE_CODE", raising=False)
    cfg = InvarLockConfig(
        {"model": {"id": "foo", "adapter": "dummy", "trust_remote_code": True}}
    )

    with pytest.raises(InvarlockError) as excinfo:
        run_mod._extract_model_load_kwargs(cfg)

    assert excinfo.value.code == "E008"


@pytest.mark.unit
def test_extract_model_load_kwargs_allows_remote_code_with_explicit_allow(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("INVARLOCK_ALLOW_REMOTE_CODE", "1")
    cfg = InvarLockConfig(
        {"model": {"id": "foo", "adapter": "dummy", "trust_remote_code": True}}
    )

    assert run_mod._extract_model_load_kwargs(cfg) == {"trust_remote_code": True}
