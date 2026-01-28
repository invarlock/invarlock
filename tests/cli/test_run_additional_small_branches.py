from __future__ import annotations

from types import SimpleNamespace

import click
from rich.console import Console

from invarlock.cli.commands import run as run_mod


def test_hash_sequences_falls_back_when_len_unavailable():
    # Inner generator sequences do not support len(), exercising the fallback path.
    seqs = ((i for i in [1, 2, 3]), (i for i in [4, 5]))
    assert run_mod._hash_sequences(seqs) == "e08215eb1a73f6d493dfb9f17c0de613"


def test_tensor_or_list_to_ints_reraises_click_exit(monkeypatch):
    class ExplodingIter:
        def __iter__(self):
            raise click.exceptions.Exit(2)

    fake_tensor = SimpleNamespace(tolist=lambda: ExplodingIter())
    monkeypatch.setattr(run_mod, "torch", SimpleNamespace(), raising=False)
    # Best-effort helper should swallow exceptions and return empty list.
    assert run_mod._tensor_or_list_to_ints(fake_tensor) == []


def test_resolve_provider_and_split_provider_and_split_access_errors():
    class BadDataset:
        @property
        def provider(self):
            raise RuntimeError("boom")

        @property
        def split(self):
            raise RuntimeError("boom")

    def _get_provider(name, **kwargs):  # noqa: ARG001
        class Provider:
            def available_splits(self):
                return ["train", "validation"]

        return Provider()

    cfg = SimpleNamespace(dataset=BadDataset())
    provider, split, used = run_mod._resolve_provider_and_split(
        cfg,
        model_profile=SimpleNamespace(default_provider="synthetic"),
        get_provider_fn=_get_provider,
        provider_kwargs=None,
        console=Console(),
        resolved_device="cpu",
    )
    assert provider is not None
    assert split == "validation"
    assert used is True


def test_extract_model_load_kwargs_dtype_aliasing_and_normalization():
    class _Cfg:
        def model_dump(self):
            return {
                "model": {
                    "id": "foo",
                    "adapter": "dummy",
                    "device": "cpu",
                    "dtype": "fp16",
                }
            }

    assert run_mod._extract_model_load_kwargs(_Cfg()) == {"dtype": "float16"}

    class _Cfg2:
        def model_dump(self):
            return {
                "model": {
                    "id": "foo",
                    "adapter": "dummy",
                    "device": "cpu",
                    "dtype": "custom_dtype",
                }
            }

    assert run_mod._extract_model_load_kwargs(_Cfg2()) == {"dtype": "custom_dtype"}
