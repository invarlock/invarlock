from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any


def _tokenize(text: str) -> list[str]:
    return [tok for tok in str(text).strip().lower().split() if tok]


def _bleu1(pred: str, ref: str) -> float:
    pred_tokens = _tokenize(pred)
    ref_tokens = _tokenize(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum(min(pred_counts[tok], ref_counts.get(tok, 0)) for tok in pred_counts)
    precision = overlap / float(len(pred_tokens))
    bp = 1.0
    if len(pred_tokens) < len(ref_tokens):
        bp = pow(2.718281828, 1.0 - (len(ref_tokens) / float(len(pred_tokens))))
    return float(precision * bp)


def bleu1_from_records(records: Iterable[dict[str, Any]]) -> float:
    """Compute BLEU-1 from records with predictions and references."""
    scores: list[float] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        pred = record.get("prediction")
        refs = record.get("references")
        if pred is None:
            continue
        if refs is None and "reference" in record:
            refs = [record.get("reference")]
        if refs is None:
            continue
        ref_list = refs if isinstance(refs, list) else [refs]
        best = 0.0
        for ref in ref_list:
            if ref is None:
                continue
            best = max(best, _bleu1(str(pred), str(ref)))
        scores.append(best)
    if not scores:
        return float("nan")
    return float(sum(scores) / float(len(scores)))


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, tok_a in enumerate(a, start=1):
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def _rouge_l(pred: str, ref: str) -> float:
    pred_tokens = _tokenize(pred)
    ref_tokens = _tokenize(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, ref_tokens)
    prec = lcs / float(len(pred_tokens))
    rec = lcs / float(len(ref_tokens))
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def rouge_l_from_records(records: Iterable[dict[str, Any]]) -> float:
    """Compute ROUGE-L (F1) from records with predictions and references."""
    scores: list[float] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        pred = record.get("prediction")
        refs = record.get("references")
        if pred is None:
            continue
        if refs is None and "reference" in record:
            refs = [record.get("reference")]
        if refs is None:
            continue
        ref_list = refs if isinstance(refs, list) else [refs]
        best = 0.0
        for ref in ref_list:
            if ref is None:
                continue
            best = max(best, _rouge_l(str(pred), str(ref)))
        scores.append(best)
    if not scores:
        return float("nan")
    return float(sum(scores) / float(len(scores)))
