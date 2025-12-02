from __future__ import annotations

from invarlock.eval.primary_metric import (
    compute_accuracy_counts,
    infer_binary_label_from_ids,
)


def test_infer_binary_label_from_ids_parity():
    # Sum([1,2,3])=6 -> 0; Sum([1,1])=2 -> 0; Sum([1])=1 -> 1
    assert infer_binary_label_from_ids([1, 2, 3]) == 0
    assert infer_binary_label_from_ids([1, 1]) == 0
    assert infer_binary_label_from_ids([1]) == 1


def test_compute_accuracy_counts_perfect_prediction():
    recs = [{"input_ids": [1, 2]}, {"input_ids": [5]}]
    correct, total = compute_accuracy_counts(recs)
    assert total == 2
    assert correct == 2
