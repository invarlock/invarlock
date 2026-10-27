"""Small independent arithmetic checks, not a replacement production verifier.

Only --diagnose imports production comparison code. The NLL replay follows the
specified hash schedule but uses exact integer accumulation and rational quantiles.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path


def exact_mean(values):
    values = tuple(map(Fraction, values))
    if not values:
        raise ValueError("mean requires observations")
    return sum(values, Fraction()) / len(values)


def exact_ratio(baseline, subject):
    baseline, subject = tuple(baseline), tuple(subject)
    if not baseline or len(baseline) != len(subject):
        raise ValueError("a nonempty paired sample is required")
    denominator = exact_mean(baseline)
    if denominator <= 0:
        raise ValueError("baseline mean must be positive")
    return exact_mean(subject) / denominator


def exhaustive_ratios(baseline, subject):
    """All equally likely paired bootstrap samples; intentionally tiny only."""
    if not 1 <= len(baseline) <= 5 or len(baseline) != len(subject):
        raise ValueError("exhaustive checks require one to five pairs")
    return sorted(
        exact_ratio([baseline[i] for i in indexes], [subject[i] for i in indexes])
        for indexes in itertools.product(range(len(baseline)), repeat=len(baseline))
    )


def exact_percentile(values, probability):
    ordered = sorted(map(Fraction, values))
    if not ordered or not 0 <= probability <= 1:
        raise ValueError("a sample and probability in [0,1] are required")
    position = (len(ordered) - 1) * Fraction(probability)
    low = position.numerator // position.denominator
    fraction = position - low
    return (
        ordered[low] * (1 - fraction)
        + ordered[min(low + 1, len(ordered) - 1)] * fraction
    )


def sign_enumeration_probability(losses, gains):
    """Exact conditional two-sided probability from equally likely sign strings."""
    count = losses + gains
    if count > 12 or min(losses, gains) < 0:
        raise ValueError("sign enumeration is bounded to twelve discordant pairs")
    signs = itertools.product((0, 1), repeat=count)
    tail = sum(min(sum(s), count - sum(s)) <= min(losses, gains) for s in signs)
    return Fraction(tail, 2**count)


def hoeffding_family_tail(radius, width, units, comparisons):
    """Substitute radius into the tail inequality; no log/sqrt interval formula."""
    with localcontext() as context:
        context.prec = 150
        return (
            Decimal(2 * comparisons)
            * (-Decimal(2 * units) * (Decimal(radius) / Decimal(width)) ** 2).exp()
        )


def scheduled_nll_interval(baseline, subject, seed, replicates=2048):
    """Specified SHA-256 indexes; independent exact arithmetic on paired values."""
    if len(seed) != 32 or not baseline or len(baseline) != len(subject):
        raise ValueError("a SHA-256 seed and nonempty paired values are required")
    sides = [list(map(Fraction, baseline)), list(map(Fraction, subject))]
    denominator = math.lcm(*(v.denominator for side in sides for v in side))
    integers = [
        [v.numerator * (denominator // v.denominator) for v in side] for side in sides
    ]
    distribution = []
    count = len(baseline)
    for replicate in range(replicates):
        frequencies = Counter(
            int.from_bytes(
                hashlib.sha256(
                    seed + replicate.to_bytes(8, "big") + draw.to_bytes(8, "big")
                ).digest()[:8],
                "big",
            )
            % count
            for draw in range(count)
        )
        totals = [
            sum(side[i] * multiplicity for i, multiplicity in frequencies.items())
            for side in integers
        ]
        if totals[0] <= 0:
            raise ValueError("sampled baseline sum must be positive")
        distribution.append(Fraction(totals[1], totals[0]))
    return exact_percentile(distribution, Fraction(1, 40)), exact_percentile(
        distribution, Fraction(39, 40)
    )


def scalar_request(baseline, subject):
    """Portable valid public-API inputs for diagnosing numerical limitations."""
    provenance = {
        "kind": "measurement",
        "source": "synthetic-arithmetic",
        "version": "1",
        "unit": "score",
        "rubric_digest": None,
    }

    def run(values, name):
        return {
            "format": "invarlock/evaluation-run-v1",
            "source": {"name": "synthetic-arithmetic", "version": "1"},
            "run_id": name,
            "artifact_digest": "sha256:" + "a" * 64,
            "source_digest": None,
            "score_provenance": {"value": provenance},
            "records": [
                {
                    "id": f"case-{i}",
                    "input": f"question {i}",
                    "expected": "reference",
                    "output": None,
                    "scores": {"value": value},
                    "metadata": {},
                    "error": None,
                    "context": {},
                }
                for i, value in enumerate(values)
            ],
        }

    policy = {
        "format": "invarlock/comparison-policy-v1",
        "slices": [],
        "metrics": [
            {
                "name": "value",
                "kind": "recorded",
                "score_key": "value",
                "accepted_provenance": provenance,
                "configuration": {},
                "unit": "score",
                "direction": "higher",
                "aggregation": "mean",
                "minimum_count": 1,
                "maximum_regression": 0,
                "maximum_interval_width": 2,
            }
        ],
    }
    return {
        "baseline": run(baseline, "baseline"),
        "subject": run(subject, "subject"),
        "policy": policy,
    }


def diagnose():
    from invarlock.evaluation_comparison.comparison import compare_runs

    findings = []
    for name, baseline, subject in [
        ("integer_conversion", [2**60 + 1], [2**60]),
        ("representable_mean_rounding", [2**53, 2**53 + 2], [2**53, 2**53]),
        ("finite_sum_overflow", [1e308, 1e308], [1e308, 1e308]),
        ("difference_overflow", [1e308, -1e308], [-1e308, 1e308]),
    ]:
        item = {
            "case": name,
            "mathematical_delta": str(exact_mean(subject) - exact_mean(baseline)),
        }
        try:
            item["observed"] = compare_runs(**scalar_request(baseline, subject))[
                "metrics"
            ][0]
            item["delta_matches_exact_arithmetic"] = Fraction(
                item["observed"]["delta"]
            ) == Fraction(item["mathematical_delta"])
        except Exception as error:
            item.update(exception=type(error).__name__, message=str(error))
        findings.append(item)
    return {
        "scope": "diagnostic observations, not passing mathematical correctness assertions",
        "findings": findings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--subject", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.diagnose:
        print(json.dumps(diagnose(), indent=2))
        return
    if not all((args.baseline, args.subject, args.report)):
        parser.error("provide --baseline, --subject and --report, or --diagnose")
    baseline = json.loads(args.baseline.read_text())
    subject = json.loads(args.subject.read_text())
    left = {r["id"]: r for r in baseline["records"]}
    right = {r["id"]: r for r in subject["records"]}
    if left.keys() != right.keys() or not 1 <= len(left) <= 1000:
        parser.error("bounded replay requires matching IDs and one to 1,000 pairs")
    ids = sorted(left)
    projection = [
        {k: left[i][k] for k in ("id", "input", "expected", "metadata")} for i in ids
    ]
    seed = hashlib.sha256(
        (
            json.dumps(
                projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            + "\n"
        ).encode()
    ).digest()
    values = [
        [
            -Fraction(rows[i]["likelihood"]["logprob_sum"])
            / rows[i]["likelihood"]["utf8_byte_count"]
            for i in ids
        ]
        for rows in (left, right)
    ]
    lower, upper = scheduled_nll_interval(*values, seed)
    metric = json.loads(args.report.read_text())["metrics"][0]
    actual = [metric["interval"][key] for key in ("lower", "upper")]
    print(
        json.dumps(
            {
                "pairs": len(ids),
                "replicates": 2048,
                "seed_sha256": seed.hex(),
                "mathematical_endpoints": [str(lower), str(upper)],
                "rounded_endpoints": [float(lower), float(upper)],
                "stored_endpoints": actual,
                "rounded_endpoints_equal": [float(lower), float(upper)] == actual,
                "input_sha256": {
                    name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in (
                        ("baseline", args.baseline),
                        ("subject", args.subject),
                        ("report", args.report),
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
