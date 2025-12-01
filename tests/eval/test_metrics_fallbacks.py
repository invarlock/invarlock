import torch

from invarlock.eval import metrics as M


def test_compute_perplexity_tuple_fallback():
    class DummyLM:
        def __init__(self, vocab=8):
            self.out = torch.nn.Linear(4, vocab)

        def parameters(self):
            yield from self.out.parameters()

        def eval(self):  # pragma: no cover
            return self

        def __call__(self, input_ids=None, attention_mask=None, return_dict=True):
            B, T = input_ids.shape
            logits = self.out(torch.zeros(B, T, 4))
            return (logits,)

    model = DummyLM()
    batch = {"input_ids": torch.ones(1, 4, dtype=torch.long)}
    ppl = M.compute_perplexity(model, [batch])
    assert ppl >= 1.0


def test_pre_eval_checks_warnings():
    class DummyLM:
        def parameters(self):
            # Provide at least one parameter
            yield torch.nn.Parameter(torch.tensor(1.0))

        def eval(self):  # pragma: no cover
            return self

    class BadData:
        def __iter__(self):  # pragma: no cover
            # Iterator that raises to exercise warning paths
            def _gen():
                raise StopIteration
                yield  # pragma: no cover

            return _gen()

    # Should not raise
    M._perform_pre_eval_checks(
        DummyLM(), BadData(), torch.device("cpu"), M.MetricsConfig()
    )
