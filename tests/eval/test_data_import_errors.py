from __future__ import annotations

import pytest

import invarlock.eval.data as data_mod


@pytest.mark.skipif(
    getattr(data_mod, "HAS_DATASETS", False),
    reason="datasets installed; skip import-error checks",
)
def test_wikitext2_provider_requires_datasets():
    from invarlock.core.exceptions import DependencyError

    with pytest.raises(DependencyError):
        _ = data_mod.get_provider("wikitext2")


@pytest.mark.skipif(
    getattr(data_mod, "HAS_DATASETS", False),
    reason="datasets installed; skip import-error checks",
)
def test_hf_text_provider_requires_datasets():
    from invarlock.core.exceptions import DependencyError

    with pytest.raises(DependencyError):
        _ = data_mod.get_provider("hf_text")
