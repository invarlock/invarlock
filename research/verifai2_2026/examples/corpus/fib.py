"""Tiny example file for building a release-safe local_jsonl canary."""


def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
