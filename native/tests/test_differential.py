"""Bitwise differential checks against this checkout's unchanged Python helper.

Run: python -m unittest discover -s native/tests -v
Requires the separately built invarlock-native-experiment wheel to be installed.
No optional root package dependencies are needed: compile the exact three helper
AST nodes from the checked-out source, with their original algorithm constants.
"""

from __future__ import annotations

import ast
import hashlib
import math
import random
import re
import struct
import unittest
from collections.abc import Sequence
from pathlib import Path

import invarlock_native as native

SOURCE = Path(__file__).resolve().parents[2] / "src/invarlock/evidence_pack_contract.py"


def reference_namespace():
    source = SOURCE.read_text()
    tree = ast.parse(source, filename=str(SOURCE))
    names = {"_comparison_value", "_percentile", "_paired_resampling_interval"}
    constants = {"PAIRED_INTERVAL_CONFIDENCE", "PAIRED_INTERVAL_REPLICATES"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
        or isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in constants
            for target in node.targets
        )
    ]
    assert len(nodes) == 5, "reference helper/constant shape changed"
    namespace = {
        "math": math,
        "hashlib": hashlib,
        "re": re,
        "Sequence": Sequence,
        "EvidencePackError": ValueError,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace
    )
    return namespace


REF = reference_namespace()


def bits(value):
    return struct.pack(">d", value)


def draws(seed, count, replicate):
    seed = bytes.fromhex(seed)
    return [
        int.from_bytes(
            hashlib.sha256(
                seed + replicate.to_bytes(8, "big") + draw.to_bytes(8, "big")
            ).digest()[:8],
            "big",
        )
        % count
        for draw in range(count)
    ]


def reference_distribution(metric, baseline, subject, seed):
    comparison = REF["_comparison_value"]
    if len(baseline) == 1:
        return [comparison(metric, baseline, subject)]
    result = []
    for replicate in range(2048):
        indexes = draws(seed, len(baseline), replicate)
        result.append(
            comparison(
                metric, [baseline[i] for i in indexes], [subject[i] for i in indexes]
            )
        )
    return result


class DifferentialTests(unittest.TestCase):
    def assert_float_equal(self, observed, expected):
        if math.isnan(expected):
            self.assertTrue(math.isnan(observed))
        else:
            self.assertEqual(bits(observed), bits(expected))

    def check_interval(self, metric, baseline, subject, seed):
        kwargs = {
            "metric": metric,
            "baseline_scores": baseline,
            "subject_scores": subject,
            "schedule_sha256": seed,
        }
        expected = REF["_paired_resampling_interval"](**kwargs)
        actual = native.paired_resampling_interval(**kwargs)
        for observed, wanted in zip(actual, expected, strict=True):
            self.assert_float_equal(observed, wanted)
        expected_distribution = reference_distribution(metric, baseline, subject, seed)
        actual_distribution = native.resampling_distribution(**kwargs)
        self.assertEqual(len(actual_distribution), len(expected_distribution))
        for replicate, (observed, wanted) in enumerate(
            zip(actual_distribution, expected_distribution, strict=True)
        ):
            with self.subTest(replicate=replicate):
                self.assert_float_equal(observed, wanted)

    def test_every_replicate_small_nonconstant_and_numerical_boundaries(self):
        tiny = math.ulp(0.0)
        cases = [
            (
                "normalized_nll_per_utf8_byte",
                [0.125, 1.0, 3.0, 9.0],
                [0.2, 0.75, 4.0, 5.0],
            ),
            (
                "custom_score",
                [-1e100, 1.0, 1e100, 1e-100],
                [1.0, 1e100, -1e100, 3e-100],
            ),
            ("custom_score", [0.0] * 3, [1e-16, 1.0, 1e16]),
            ("custom_score", [tiny, 2 * tiny, 4 * tiny], [2 * tiny, tiny, 8 * tiny]),
            (
                "normalized_nll_per_utf8_byte",
                [2 * tiny, 4 * tiny, 8 * tiny],
                [tiny, 8 * tiny, 4 * tiny],
            ),
            ("custom_score", [-0.0, 0.0, -0.0], [0.0, -0.0, -0.0]),
            ("normalized_nll_per_utf8_byte", [tiny, tiny], [1.0, 1.0]),
            ("custom_score", [-1e308, -1e308], [1e308, 1e308]),
        ]
        for i, (metric, baseline, subject) in enumerate(cases):
            with self.subTest(case=i):
                self.check_interval(
                    metric,
                    baseline,
                    subject,
                    hashlib.sha256(str(i).encode()).hexdigest(),
                )

    def test_single_pair_shortcut(self):
        for metric, baseline, subject in [
            ("normalized_nll_per_utf8_byte", [2.0], [3.0]),
            ("extension_score", [0.9], [0.2]),
            ("custom_score", [-0.0], [0.0]),
            ("normalized_nll_per_utf8_byte", [math.ulp(0.0)], [1.0]),
        ]:
            self.check_interval(metric, baseline, subject, "a" * 64)

    def test_all_draws_and_golden(self):
        self.assertEqual(native.sample_indexes("a" * 64, 7, 3), [2, 0, 5, 4, 1, 6, 5])
        for seed, count in [("0" * 64, 3), ("f" * 64, 17)]:
            for replicate in range(2048):
                self.assertEqual(
                    native.sample_indexes(seed, count, replicate),
                    draws(seed, count, replicate),
                )
        for replicate in [2**32 + 1, 2**64 - 1]:
            self.assertEqual(
                native.sample_indexes("a" * 64, 7, replicate),
                draws("a" * 64, 7, replicate),
            )

    def test_fsum_partial_rounding_and_random_finite_inputs(self):
        rng = random.Random(105743)
        cases = [
            [1e-16, 1.0, 1e16],
            [1e16, 1.0, -1e16],
            [-0.0],
            [],
            [math.ulp(0.0)] * 3,
            [1e308, 1e308, -1e308],
            [math.inf],
            [math.inf, -math.inf],
            [math.nan],
        ]
        for _ in range(3000):
            cases.append(
                [
                    math.ldexp(rng.uniform(-1, 1), rng.randrange(-1074, 1024))
                    for _ in range(rng.randrange(1, 65))
                ]
            )
        for values in cases:
            try:
                expected = math.fsum(values)
            except (OverflowError, ValueError) as exc:
                with self.assertRaisesRegex(type(exc), re.escape(str(exc))):
                    native._diagnostic_fsum(values)
            else:
                self.assert_float_equal(native._diagnostic_fsum(values), expected)

    def test_error_type_and_message_matches_validated_reference_domain(self):
        for kwargs in [
            {
                "metric": "custom",
                "baseline_scores": [1.0],
                "subject_scores": [1.0],
                "schedule_sha256": "A" * 64,
            },
            {
                "metric": "normalized_nll_per_utf8_byte",
                "baseline_scores": [0.0],
                "subject_scores": [1.0],
                "schedule_sha256": "a" * 64,
            },
            {
                "metric": "normalized_nll_per_utf8_byte",
                "baseline_scores": [-1.0],
                "subject_scores": [1.0],
                "schedule_sha256": "a" * 64,
            },
            {
                "metric": "normalized_nll_per_utf8_byte",
                "baseline_scores": [1e308] * 2,
                "subject_scores": [1.0] * 2,
                "schedule_sha256": "a" * 64,
            },
            {
                "metric": "custom",
                "baseline_scores": [-1e308, 1e308],
                "subject_scores": [1e308, -1e308],
                "schedule_sha256": "a" * 64,
            },
        ]:
            with self.subTest(kwargs=kwargs):
                try:
                    REF["_paired_resampling_interval"](**kwargs)
                except (ValueError, OverflowError) as exc:
                    with self.assertRaisesRegex(type(exc), re.escape(str(exc))):
                        native.paired_resampling_interval(**kwargs)
                else:
                    self.fail("reference fixture must raise")

    def test_native_boundary_rejects_unvalidated_inputs(self):
        for baseline, subject in [
            ([], []),
            ([1.0], []),
            ([math.inf], [1.0]),
            ([1.0], [math.nan]),
        ]:
            with self.assertRaises(ValueError):
                native.paired_resampling_interval(
                    metric="custom",
                    baseline_scores=baseline,
                    subject_scores=subject,
                    schedule_sha256="a" * 64,
                )
        for seed in ["", "0" * 63, "0" * 65, "sha256:" + "0" * 64, "á" * 64]:
            with self.assertRaisesRegex(
                ValueError, "paired records schedule_sha256 is invalid"
            ):
                native.paired_resampling_interval(
                    metric="custom",
                    baseline_scores=[0.0],
                    subject_scores=[1.0],
                    schedule_sha256=seed,
                )


if __name__ == "__main__":
    unittest.main()
