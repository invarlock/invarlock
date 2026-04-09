import pytest

from invarlock.eval import metrics_runtime as runtime_mod


def test_sanitize_token_ids_clamps_out_of_range():
    torch = pytest.importorskip("torch")

    input_ids = torch.tensor([[0, 5, 10]])
    attention_mask = torch.tensor([[1, 1, 1]])
    labels = torch.tensor([[0, 10, 11]])

    cleaned_ids, cleaned_mask, cleaned_labels = (
        runtime_mod._sanitize_token_ids_for_model(
            input_ids,
            attention_mask,
            labels,
            vocab_size=10,
            pad_token_id=0,
        )
    )

    assert cleaned_ids.tolist() == [[0, 5, 0]]
    assert cleaned_mask.tolist() == [[1, 1, 0]]
    assert cleaned_labels.tolist() == [[0, -100, -100]]


def test_sanitize_token_ids_handles_missing_attention_mask_with_labels():
    torch = pytest.importorskip("torch")

    input_ids = torch.tensor([[9, 1]])
    labels = torch.tensor([[9, 1]])

    cleaned_ids, cleaned_mask, cleaned_labels = (
        runtime_mod._sanitize_token_ids_for_model(
            input_ids,
            None,
            labels,
            vocab_size=5,
            pad_token_id=0,
        )
    )

    assert cleaned_ids.tolist() == [[0, 1]]
    assert cleaned_mask is None
    assert cleaned_labels.tolist() == [[-100, 1]]


def test_infer_model_vocab_size_falls_back_when_embedding_weight_is_unusable():
    class _WeightWithoutShape:
        pass

    class _EmbeddingWithoutShape:
        weight = _WeightWithoutShape()

    class _Model:
        config = type("Cfg", (), {"vocab_size": 7})()

        def get_input_embeddings(self):
            return _EmbeddingWithoutShape()

    assert runtime_mod._infer_model_vocab_size(_Model()) == 7
