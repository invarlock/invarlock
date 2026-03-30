from types import SimpleNamespace

import pytest

from invarlock.eval import metrics as M


def test_measure_latency_early_exit_when_no_long_sequences():
    # No sequence longer than 10 -> invalid latency measurement input.
    window = SimpleNamespace(
        input_ids=[[1] * 5, [1] * 7], attention_masks=[[1] * 5, [1] * 7]
    )

    class DummyLM:
        def parameters(self):  # pragma: no cover
            yield from ()

        def eval(self):  # pragma: no cover
            return self

        def __call__(self, *args, **kwargs):  # pragma: no cover
            return SimpleNamespace(logits=None)

    with pytest.raises(M.ValidationError) as exc_info:
        M.measure_latency(DummyLM(), window, device="cpu")
    assert (
        exc_info.value.details["reason"]
        == "latency measurement requires at least one sequence longer than 10 tokens"
    )
