from types import SimpleNamespace

from invarlock.eval import metrics as M


def test_measure_latency_early_exit_when_no_long_sequences():
    # No sequence longer than 10 → early return 0.0
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

    lat = M.measure_latency(DummyLM(), window, device="cpu")
    assert isinstance(lat, float) and lat == 0.0
