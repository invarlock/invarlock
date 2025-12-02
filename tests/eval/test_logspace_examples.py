"""Examples referenced in documentation should stay executable."""

from __future__ import annotations

import math


def test_logspace_vs_ratio_of_means_example() -> None:
    weights = [512, 256]
    ppl_preview = [40.0, 220.0]
    ppl_final = [38.0, 260.0]

    delta_sum = sum(
        w * (math.log(b) - math.log(a))
        for w, a, b in zip(weights, ppl_preview, ppl_final, strict=True)
    )
    ratio_log = math.exp(delta_sum / sum(weights))

    ratio_means = sum(w * b for w, b in zip(weights, ppl_final, strict=True)) / sum(
        w * a for w, a in zip(weights, ppl_preview, strict=True)
    )

    assert math.isclose(ratio_log, 1.02172172, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(ratio_means, 1.12, rel_tol=0.0, abs_tol=1e-6)
    assert ratio_log < ratio_means
