from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from invarlock.training_model_load import (
    TRAINING_MODEL_LOAD_DIAGNOSTICS_SCHEMA,
    TrainingModelLoadError,
    configure_causal_lm_loss,
    load_diagnostics_sha256,
    load_model_with_diagnostics,
    normalize_load_diagnostics,
)

EXPECTED_UNEXPECTED = (
    "transformer.h.0.attn.masked_bias",
    "transformer.h.1.attn.masked_bias",
)


def _diagnostics(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "missing_keys": [],
        "unexpected_keys": list(EXPECTED_UNEXPECTED),
        "mismatched_keys": [],
        "error_msgs": [],
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"missing_keys": ["weight"]}, "missing_keys"),
        ({"unexpected_keys": [*EXPECTED_UNEXPECTED, "injected"]}, "unexpected keys"),
        ({"unexpected_keys": [EXPECTED_UNEXPECTED[0]]}, "unexpected keys"),
        ({"mismatched_keys": [("weight", (2,), (3,))]}, "mismatched_keys"),
        ({"error_msgs": ["loader failed"]}, "error_msgs"),
    ],
)
def test_baseline_diagnostics_fail_on_every_undeclared_outcome(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(TrainingModelLoadError, match=message):
        normalize_load_diagnostics(
            _diagnostics(**overrides),
            expected_unexpected_keys=EXPECTED_UNEXPECTED,
            label="upstream baseline model",
        )


def test_baseline_diagnostics_are_order_normalized_and_digest_bound() -> None:
    normalized = normalize_load_diagnostics(
        _diagnostics(unexpected_keys=list(reversed(EXPECTED_UNEXPECTED))),
        expected_unexpected_keys=EXPECTED_UNEXPECTED,
        label="upstream baseline model",
    )
    assert normalized == {
        "schema": TRAINING_MODEL_LOAD_DIAGNOSTICS_SCHEMA,
        "policy": "exact_source_key_migration",
        "missing_keys": [],
        "unexpected_keys": list(EXPECTED_UNEXPECTED),
        "mismatched_keys": [],
        "error_msgs": [],
    }
    assert load_diagnostics_sha256(normalized).startswith("sha256:")


def test_saved_subject_requires_completely_clean_loading_diagnostics() -> None:
    with pytest.raises(TrainingModelLoadError, match="unexpected keys"):
        normalize_load_diagnostics(
            _diagnostics(),
            expected_unexpected_keys=(),
            label="saved training subject",
        )


def test_loader_requires_a_complete_diagnostic_tuple() -> None:
    class BareLoader:
        @staticmethod
        def from_pretrained(_source: object, **_options: object) -> object:
            return object()

    with pytest.raises(TrainingModelLoadError, match="model and loading diagnostics"):
        load_model_with_diagnostics(
            BareLoader,
            "baseline",
            load_options={},
            expected_unexpected_keys=EXPECTED_UNEXPECTED,
            label="upstream baseline model",
        )


def test_loss_semantics_are_explicit_and_fail_closed() -> None:
    class Model:
        loss_type: str | None = None

        @property
        def loss_function(self) -> Any:
            if self.loss_type != "ForCausalLM":
                raise ValueError("invalid loss")
            return lambda: None

    model = Model()
    configure_causal_lm_loss(model, loss_function="ForCausalLM")
    assert model.loss_type == "ForCausalLM"
    with pytest.raises(TrainingModelLoadError, match="must be ForCausalLM"):
        configure_causal_lm_loss(model, loss_function="custom")
    with pytest.raises(TrainingModelLoadError, match="does not expose"):
        configure_causal_lm_loss(
            SimpleNamespace(loss_function=lambda: None),
            loss_function="ForCausalLM",
        )
