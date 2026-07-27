"""Importable deterministic metric used through TruLens' Metric wrapper."""


def exact_match(output: str, expected: str) -> float:
    return float(output == expected)
