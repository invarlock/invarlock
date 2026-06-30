from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.unit
def test_text_only_loader_disables_optional_torchvision(monkeypatch) -> None:
    import invarlock.adapters.hf_loading as hf_loading

    fake_utils = SimpleNamespace(
        _torchvision_available=True,
        _torchvision_version="0.0",
        is_torchvision_available=lambda: True,
    )
    fake_import_utils = SimpleNamespace(
        _torchvision_available=True,
        _torchvision_version="0.0",
        is_torchvision_available=lambda: True,
    )

    def fake_import_module(name: str):
        modules = {
            "transformers.utils": fake_utils,
            "transformers.utils.import_utils": fake_import_utils,
        }
        return modules[name]

    monkeypatch.setattr(hf_loading.importlib, "import_module", fake_import_module)

    assert hf_loading._disable_torchvision_for_text_only_task("causal") is True
    assert fake_utils.is_torchvision_available() is False
    assert fake_import_utils.is_torchvision_available() is False
    assert fake_utils._torchvision_available is False
    assert fake_import_utils._torchvision_available is False
    assert fake_utils._torchvision_version is None
    assert fake_import_utils._torchvision_version is None

    fake_utils._torchvision_available = True
    assert hf_loading._disable_torchvision_for_text_only_task("multimodal") is False
    assert fake_utils._torchvision_available is True
