from __future__ import annotations

from dataclasses import dataclass

import pytest

from invarlock.core.exceptions import ConfigError
from invarlock.core.orchestration.environment import (
    _attach_tokenizer_load_kwargs,
    _detect_model_profile_with_tokenizer_kwargs,
    _extract_tokenizer_load_kwargs_from_cfg,
    _validate_removed_edit_keys,
)


class _ExplodingEditConfig:
    def model_dump(self) -> dict:
        return {}

    @property
    def edit(self):
        raise RuntimeError("unavailable")


class _ExplodingModelConfig:
    def model_dump(self) -> dict:
        return {}

    @property
    def model(self):
        raise TypeError("unavailable")


class _ExplodingModelFields:
    @property
    def trust_remote_code(self):
        raise TypeError("unavailable")

    @property
    def revision(self):
        raise TypeError("unavailable")


class _ObjectModelConfig:
    def model_dump(self) -> dict:
        return {}

    model = _ExplodingModelFields()


def test_environment_config_observation_errors_fail_closed() -> None:
    _validate_removed_edit_keys(
        _ExplodingEditConfig(), config_value_exceptions=(RuntimeError,)
    )
    assert _extract_tokenizer_load_kwargs_from_cfg(_ExplodingModelConfig()) == {}
    assert _extract_tokenizer_load_kwargs_from_cfg(_ObjectModelConfig()) == {}


@pytest.mark.parametrize("removed", ["kind", "parameters"])
def test_removed_edit_configuration_never_reenters_runtime(removed: str) -> None:
    class RemovedConfig:
        def model_dump(self) -> dict:
            return {"edit": {removed: "legacy"}}

        edit = None

    with pytest.raises(ConfigError, match="CONFIG-KEY-REMOVED"):
        _validate_removed_edit_keys(
            RemovedConfig(), config_value_exceptions=(Exception,)
        )


@dataclass
class _ProfileWithTokenizerKwargs:
    tokenizer_load_kwargs: dict | None = None


@dataclass
class _MutableProfileWithoutField:
    name: str = "profile"


@dataclass(slots=True)
class _SlottedProfileWithoutField:
    name: str = "profile"


def test_tokenizer_kwargs_attach_through_dataclass_or_object_fallback() -> None:
    kwargs = {"revision": "a" * 40}
    untouched = _MutableProfileWithoutField()
    assert _attach_tokenizer_load_kwargs(untouched, {}) is untouched
    replaced = _attach_tokenizer_load_kwargs(_ProfileWithTokenizerKwargs(), kwargs)
    assert replaced.tokenizer_load_kwargs == kwargs

    mutable = _MutableProfileWithoutField()
    assert _attach_tokenizer_load_kwargs(mutable, kwargs) is mutable
    assert mutable.tokenizer_load_kwargs == kwargs

    slotted = _SlottedProfileWithoutField()
    assert _attach_tokenizer_load_kwargs(slotted, kwargs) is slotted
    assert not hasattr(slotted, "tokenizer_load_kwargs")


class _OpaqueProfileDetector:
    @property
    def __signature__(self):
        raise ValueError("opaque callable")

    def __call__(self, *, model_id: str, adapter: str):
        return _MutableProfileWithoutField(name=f"{model_id}:{adapter}")


def test_profile_detection_falls_back_when_callable_signature_is_opaque() -> None:
    profile = _detect_model_profile_with_tokenizer_kwargs(
        _OpaqueProfileDetector(),
        model_id="model",
        adapter="hf_causal",
        tokenizer_load_kwargs={"trust_remote_code": False},
    )

    assert profile.name == "model:hf_causal"
    assert profile.tokenizer_load_kwargs == {"trust_remote_code": False}
