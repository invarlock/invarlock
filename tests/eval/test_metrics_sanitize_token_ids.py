import pytest

from invarlock.eval import metrics as metrics_mod


def test_sanitize_token_ids_clamps_out_of_range():
    torch = pytest.importorskip("torch")

    input_ids = torch.tensor([[0, 5, 10]])
    attention_mask = torch.tensor([[1, 1, 1]])
    labels = torch.tensor([[0, 10, 11]])

    cleaned_ids, cleaned_mask, cleaned_labels = (
        metrics_mod._sanitize_token_ids_for_model(
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
