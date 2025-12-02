import torch.nn as nn

from invarlock.guards.variance import VarianceGuard, _iter_transformer_layers


def test_iter_transformer_layers_llama_and_bert_styles():
    class LLaMA(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module()])
            self.model.layers[0].attn = nn.Module()
            self.model.layers[0].mlp = nn.Module()

    class BERT(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([nn.Module()])
            self.encoder.layer[0].attn = nn.Module()
            self.encoder.layer[0].mlp = nn.Module()

    llama_layers = list(_iter_transformer_layers(LLaMA()))
    bert_layers = list(_iter_transformer_layers(BERT()))
    assert len(llama_layers) == 1 and len(bert_layers) == 1


def test_materialize_batch_deepcopy_fallback():
    class NonCopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("nope")

    g = VarianceGuard()
    out = g._materialize_batch(NonCopyable())
    # Should return the original object as-is when deepcopy fails
    assert isinstance(out, NonCopyable)
