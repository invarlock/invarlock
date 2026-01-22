from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _iter_pairs(record: dict[str, Any]) -> list[tuple[Any, Any]]:
    if "correct" in record:
        return [(bool(record.get("correct")), True)]

    label = record.get("label")
    pred = record.get("prediction")
    if label is None:
        label = record.get("labels")
    if pred is None:
        pred = record.get("pred")
    if pred is None:
        pred = record.get("predictions")

    if isinstance(label, list) and isinstance(pred, list):
        return list(zip(label, pred, strict=False))
    if label is None or pred is None:
        return []
    return [(label, pred)]


def accuracy_from_records(records: Iterable[dict[str, Any]]) -> float:
    """Compute accuracy from records with labels/predictions.

    Accepted record shapes:
    - {"label": <label>, "prediction": <label>}
    - {"labels": [...], "predictions": [...]}
    - {"correct": <bool>}
    """
    total = 0
    correct = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        for label, pred in _iter_pairs(record):
            total += 1
            if isinstance(label, bool):
                correct += int(label is pred)
            else:
                correct += int(label == pred)
    if total == 0:
        return float("nan")
    return float(correct / total)
