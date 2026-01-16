from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def exact_match_from_records(records: Iterable[dict[str, Any]]) -> float:
    """Compute exact-match accuracy for QA-style records.

    Accepted record shapes:
    - {"prediction": "...", "answer": "..."}
    - {"prediction": "...", "answers": ["...", ...]}
    """
    total = 0
    correct = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        pred = record.get("prediction")
        answers = record.get("answers")
        if answers is None and "answer" in record:
            answers = [record.get("answer")]
        if pred is None or answers is None:
            continue
        pred_norm = _normalize(pred)
        answer_list = answers if isinstance(answers, list) else [answers]
        total += 1
        if any(_normalize(a) == pred_norm for a in answer_list if a is not None):
            correct += 1
    if total == 0:
        return float("nan")
    return float(correct / total)
