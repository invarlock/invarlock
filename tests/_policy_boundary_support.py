"""Acceptance properties over legal neighboring policies on fixed evidence."""

from __future__ import annotations

import itertools
import math
from decimal import Decimal

import pytest


def check_boundaries(centers, accepts, *, relaxing_down):
    single = {}
    for key, center in centers.items():
        if isinstance(center, Decimal):
            quantum = Decimal("0.000000000000001")
            points = [center - quantum, center, center + quantum]
        elif isinstance(center, int):
            points = [center - 1, center, center + 1]
        else:
            points = [
                math.nextafter(center, -math.inf),
                center,
                math.nextafter(center, math.inf),
            ]
        outcomes = [accepts({key: point}) for point in points]
        assert outcomes[1], (key, "equality must pass")
        relaxed = list(reversed(outcomes)) if key in relaxing_down else outcomes
        assert all(not a or b for a, b in zip(relaxed, relaxed[1:], strict=False)), key
        single[key] = points, outcomes
    for left, right in itertools.combinations(single, 2):
        left_values, left_outcomes = single[left]
        right_values, right_outcomes = single[right]
        for i, j in itertools.product(range(3), repeat=2):
            changes = {left: left_values[i], right: right_values[j]}
            if changes.get("subject_minimum", -math.inf) > changes.get(
                "subject_maximum", math.inf
            ):
                with pytest.raises(ValueError, match="minimum exceeds maximum"):
                    accepts(changes)
                continue
            assert accepts(changes) == (left_outcomes[i] and right_outcomes[j]), changes
