from __future__ import annotations

import importlib

import pytest


def test_get_provider_invalid_name_maps_to_validation_error():
    data_mod = importlib.import_module("invarlock.eval.data")
    with pytest.raises(Exception) as ei:
        data_mod.get_provider("nope")
    err = ei.value
    from invarlock.core.exceptions import ValidationError

    assert isinstance(err, ValidationError)
    assert getattr(err, "code", "") == "E308"
    assert "PROVIDER-NOT-FOUND" in str(err)


def test_wikitext2_missing_dependency_maps_to_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
):
    data_mod = importlib.import_module("invarlock.eval.data")
    # Simulate datasets missing by forcing HAS_DATASETS = False
    monkeypatch.setattr(data_mod, "HAS_DATASETS", False, raising=True)
    with pytest.raises(Exception) as ei:
        data_mod.WikiText2Provider()
    err = ei.value
    from invarlock.core.exceptions import DependencyError

    assert isinstance(err, DependencyError)
    assert getattr(err, "code", "") == "E301"
    assert "DEPENDENCY-MISSING" in str(err)


def test_windows_not_enough_samples_maps_to_data_error(monkeypatch: pytest.MonkeyPatch):
    # Force texts list short to trigger error path
    import importlib

    data_mod = importlib.import_module("invarlock.eval.data")
    prov = data_mod.WikiText2Provider()

    def _load_small(split: str = "validation", max_samples: int = 2000, **kwargs):  # type: ignore[no-untyped-def]
        return ["short sample"]

    monkeypatch.setattr(prov, "load", _load_small)

    class Tok:
        pad_token_id = 0

        def encode(self, text, truncation=True, max_length=128):  # type: ignore[no-untyped-def]
            return [1, 2, 3]

    with pytest.raises(Exception) as ei:
        prov.windows(Tok(), seq_len=8, stride=8, preview_n=10, final_n=10, seed=0)
    err = ei.value
    from invarlock.core.exceptions import DataError

    assert isinstance(err, DataError)
    # Accept any of E303/E304 since different checks may trigger first
    assert getattr(err, "code", "") in {"E303", "E304"}
    assert any(
        tag in str(err) for tag in ("CAPACITY-INSUFFICIENT", "TOKENIZE-INSUFFICIENT")
    )
