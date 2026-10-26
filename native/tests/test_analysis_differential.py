"""Whole-report and binary64 parity against the unchanged Python reference.

Build core/examples/analysis and set INVARLOCK_ANALYSIS_BIN to that executable.
The diagnostic process runs Rust only; Python supplies independent expectations.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import random
import struct
import subprocess
import unicodedata
import unittest
from pathlib import Path

from invarlock.core.scoring import score
from invarlock.evaluation_comparison.comparison import compare_runs, make_run
from invarlock.evaluation_record_contracts.contracts import digest
from invarlock.paired_exact_match import paired_exact_match_statistics

ROOT = Path(__file__).resolve().parents[2]


def assert_exact(test, actual, expected, path=""):
    if isinstance(expected, float):
        test.assertEqual(struct.pack(">d", actual), struct.pack(">d", expected), path)
    elif isinstance(expected, dict):
        test.assertEqual(set(actual), set(expected), path)
        for key in expected:
            assert_exact(test, actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, (list, tuple)):
        test.assertEqual(len(actual), len(expected), path)
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_exact(test, a, e, f"{path}/{i}")
    else:
        test.assertEqual(actual, expected, path)


def metric(kind="exact_match", configuration=None, **kw):
    return dict(
        name="quality",
        kind=kind,
        configuration=configuration or {},
        direction="higher",
        unit="score",
        aggregation="mean",
        minimum_count=1,
        maximum_regression=0.2,
        maximum_interval_width=2,
        **kw,
    )


def policy(m=None):
    return {
        "format": "invarlock/comparison-policy-v1",
        "metrics": [m or metric()],
        "slices": [],
    }


def run(rows, run_id="test"):
    return make_run(
        rows,
        source={"name": "retained", "version": "1"},
        run_id=run_id,
        artifact_digest=digest("artifact"),
    )


class AnalysisDifferential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary = os.environ.get(
            "INVARLOCK_ANALYSIS_BIN",
            str(ROOT / "native/target/debug/examples/analysis"),
        )
        if not Path(binary).exists():
            raise RuntimeError(
                "Build core/examples/analysis and set INVARLOCK_ANALYSIS_BIN"
            )
        cls.process = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

    @classmethod
    def tearDownClass(cls):
        cls.process.stdin.close()
        cls.process.wait(timeout=10)
        cls.process.stdout.close()

    def native(self, **request):
        self.process.stdin.write(
            json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n"
        )
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        self.assertTrue(line, "Rust diagnostic terminated")
        return json.loads(line)

    def comparison(self, b, s, p, **kw):
        actual = self.native(baseline=b, subject=s, policy=p, **kw)
        expected = compare_runs(b, s, p, **kw)
        self.assertIn("ok", actual, actual)
        assert_exact(self, actual["ok"], expected)

    def error(self, b, s, p, **kw):
        with self.assertRaises(ValueError) as raised:
            compare_runs(b, s, p, **kw)
        actual = self.native(baseline=b, subject=s, policy=p, **kw)
        self.assertEqual(actual["error"], str(raised.exception))

    def test_exact_complete_reports_and_slices(self):
        rng = random.Random(148)
        for n in [1, 2, 3, 7, 23, 100]:
            rows = [
                {
                    "id": f"id-{i}",
                    "input": f"prompt-{i}",
                    "expected": "yes",
                    "output": "yes",
                    "metadata": {"group": str(i % 2)},
                }
                for i in range(n)
            ]
            b, s = run(rows), run(rows)
            for row in b["records"] + s["records"]:
                row["output"] = rng.choice(["yes", "no"])
            p = policy()
            p["slices"] = [
                {"name": "one", "where": {"group": "1"}},
                {"name": "missing", "where": {"group": "5"}},
            ]
            self.comparison(b, s, p)

    def test_scalar_reports_use_complete_shake_distribution(self):
        rng = random.Random(91)
        provenance = {
            "kind": "measurement",
            "source": "retained",
            "version": "1",
            "unit": "ms",
            "rubric_digest": None,
        }
        m = metric("recorded", score_key="latency", accepted_provenance=provenance)
        m.update(direction="lower", unit="ms")
        for n in [1, 2, 5, 11, 29]:
            rows = [
                {
                    "id": str(i),
                    "input": "x",
                    "expected": "y",
                    "output": "z",
                    "scores": {"latency": rng.uniform(-10, 10)},
                }
                for i in range(n)
            ]
            b, s = run(rows), run(rows)
            for r in s["records"]:
                r["scores"]["latency"] += rng.uniform(-1, 1)
            for r in [b, s]:
                r["score_provenance"] = {"latency": provenance}
            self.comparison(b, s, policy(m))
            self.error(b, s, policy(m), max_bootstrap_draws=n * 2048 - 1)

    def test_all_deterministic_scorers(self):
        cases = [
            ("exact_match", "", "", {}),
            ("exact_match", "a", None, {}),
            (
                "normalized_match",
                "ＳＴＲＡＳＳＥ\u00a0\tÉ",
                "strasse é",
                {"unicode_version": "15.0.0"},
            ),
            (
                "normalized_match",
                "각\u0315\u0300",
                "각\u0300\u0315",
                {"unicode_version": "15.0.0"},
            ),
            ("token_f1", "one one two", "ONE TWO THREE", {"unicode_version": "15.0.0"}),
            ("token_f1", "\u001c", "", {"unicode_version": "15.0.0"}),
            (
                "json_fields",
                {"x": [1, 2], "a/b": {"~": True}},
                '{"x":[1,3],"a/b":{"~":true}}',
                {"fields": ["/x/0", "/x/1", "/a~1b/~0"]},
            ),
            ("json_exact", '{"a":1}', {"a": 1.0}, {}),
            ("json_exact", '{"a":1}', '{"a":1,"a":1}', {}),
            ("numeric_tolerance", "1e1000", "1.0001e1000", {"relative": 0.0001}),
            ("numeric_tolerance", "0.1", 0.3 - 0.2, {"absolute": 0}),
            ("numeric_tolerance", 123, "1_2_3", {}),
            ("numeric_tolerance", "1", "nan", {}),
        ]
        for kind, expected, output, configuration in cases:
            with self.subTest(kind=kind, expected=expected):
                actual = self.native(
                    operation="score",
                    kind=kind,
                    expected=expected,
                    output=output,
                    configuration=configuration,
                )
                self.assertIn("ok", actual, actual)
                assert_exact(
                    self, actual["ok"], score(kind, expected, output, configuration)
                )

    def test_unicode_normalization_tables(self):
        rng = random.Random(401)
        points = [
            cp
            for cp in range(0x110000)
            if not 0xD800 <= cp <= 0xDFFF
            and (
                unicodedata.decomposition(chr(cp))
                or chr(cp).casefold() != chr(cp)
                or unicodedata.combining(chr(cp))
            )
        ]
        for _ in range(80):
            value = "".join(chr(rng.choice(points)) for _ in range(25))
            normalized = " ".join(
                unicodedata.normalize("NFKC", value).casefold().split()
            )
            actual = self.native(
                operation="score",
                kind="normalized_match",
                expected=value,
                output=normalized,
                configuration={"unicode_version": "15.0.0"},
            )
            assert_exact(
                self,
                actual["ok"],
                score(
                    "normalized_match", value, normalized, {"unicode_version": "15.0.0"}
                ),
            )

    def test_exact_statistics_both_methods(self):
        rng = random.Random(811)
        for method in [
            "newcombe_hybrid_score_paired_v1",
            "newcombe_hybrid_score_paired_v2",
        ]:
            for n in [1, 2, 3, 10, 27, 100, 1000, 10000]:
                b = [rng.choice([True, False]) for _ in range(n)]
                s = [rng.choice([True, False]) for _ in range(n)]
                expected = dataclasses.asdict(
                    paired_exact_match_statistics(
                        b, s, confidence_interval_method=method
                    )
                )
                actual = self.native(
                    operation="exact_match", baseline=b, subject=s, method=method
                )
                assert_exact(self, actual["ok"], expected)

    def test_missing_minimum_and_arithmetic_errors(self):
        rows = [{"id": "a", "input": "x", "expected": "yes", "output": "yes"}]
        b, s = run(rows), run(rows)
        p = policy()
        p["metrics"][0]["minimum_count"] = 2
        self.comparison(b, s, p)
        s["records"][0]["output"] = None
        self.comparison(b, s, policy())
        s["records"][0]["error"] = "retained failure"
        self.comparison(b, s, policy())
        s = run(rows)
        s["records"][0]["id"] = "other"
        self.error(b, s, policy())
        s = run(rows)
        s["records"][0]["expected"] = "changed"
        self.error(b, s, policy())
        s = run(rows)
        s["records"].append(copy.deepcopy(s["records"][0]))
        self.error(b, s, policy())
        b["records"][0]["expected"] = 3
        s = copy.deepcopy(b)
        self.error(b, s, policy())

    def test_likelihood_complete_binding_and_missingness(self):
        from tests.evaluation_comparison.test_likelihood import policy, row, run

        b = run([row("a", -4), row("b", -8), row("c", -2)])
        s = run([row("a", -3), row("b", -9), row("c", -1)])
        self.comparison(b, s, policy())
        altered = copy.deepcopy(s)
        del altered["records"][1]["likelihood"]
        self.comparison(b, altered, policy())
        for field in [
            "input_digest",
            "reference_digest",
            "artifact_digest",
            "configuration_digest",
            "tokenizer_digest",
        ]:
            altered = copy.deepcopy(s)
            altered["records"][1]["likelihood"][field] = digest("changed")
            self.error(b, altered, policy())

    def test_native_report_versions_and_policies(self):
        from invarlock.evidence_pack_contract import build_comparison_report

        rng = random.Random(144)
        for kind in ["exact_match", "normalized_nll_per_utf8_byte"]:
            for n in [1, 7, 21]:
                paired = {
                    "format": "invarlock/paired-records-v1",
                    "metric": kind,
                    "schedule_sha256": "ab" * 32,
                    "records": [
                        {
                            "baseline": {
                                "score": rng.choice([0.0, 1.0])
                                if kind == "exact_match"
                                else rng.uniform(0.1, 3)
                            },
                            "subject": {
                                "score": rng.choice([0.0, 1.0])
                                if kind == "exact_match"
                                else rng.uniform(0.1, 3)
                            },
                        }
                        for _ in range(n)
                    ],
                }
                if kind != "exact_match":
                    paired["derived_measurements"] = {
                        "perplexity_ratio": {
                            "status": "unavailable",
                            "basis": "authenticated_target_likelihood",
                            "method": "target_token_weighted_perplexity_ratio_v1",
                            "reason": "target_token_counts_unavailable",
                        }
                    }
                for version in [1, 2, 3]:
                    selected = (
                        {"delta_min_pp": -10}
                        if kind == "exact_match"
                        else {"ratio_max": 1.1}
                    )
                    selected.update(minimum_record_count=2)
                    selected[
                        "maximum_interval_width_pp"
                        if kind == "exact_match"
                        else "maximum_interval_width_ratio"
                    ] = 100
                    if kind == "exact_match" and version == 3:
                        selected["minimum_side_accuracy"] = 0.2
                    kwargs = {
                        "comparison_id": "test-report",
                        "paired_records": paired,
                        "policy": {"resolved_policy": {"metrics": {kind: selected}}},
                        "policy_digest": digest("policy"),
                        "report_format": f"invarlock/comparison-report-v{version}",
                    }
                    actual = self.native(operation="report", **kwargs)
                    self.assertIn("ok", actual, actual)
                    assert_exact(self, actual["ok"], build_comparison_report(**kwargs))

    def test_judge_decimal_hoeffding_intervals(self):
        from decimal import Decimal
        from fractions import Fraction

        from invarlock.judge_measurements.statistics import hoeffding_interval

        rng = random.Random(910)
        for count in [1, 2, 3, 17, 89, 999]:
            for bounds in [("0", "1"), ("-1", "1"), ("0", "0"), ("0.123", "0.777")]:
                lower, upper = map(Decimal, bounds)
                values = [
                    Fraction(lower)
                    + (Fraction(upper) - Fraction(lower))
                    * Fraction(rng.randrange(20), 19)
                    for _ in range(count)
                ]
                kwargs = {
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "alpha": Decimal("0.05"),
                    "comparisons": 2,
                }
                expected = dataclasses.asdict(hoeffding_interval(values, **kwargs))
                expected = {
                    k: str(v) if isinstance(v, Decimal) else v
                    for k, v in expected.items()
                }
                actual = self.native(
                    operation="judge",
                    statistic="hoeffding_interval",
                    values=[
                        {
                            "numerator": str(v.numerator),
                            "denominator": str(v.denominator),
                        }
                        for v in values
                    ],
                    lower_bound=str(lower),
                    upper_bound=str(upper),
                    alpha="0.05",
                    comparisons=2,
                )
                self.assertIn("ok", actual, actual)
                assert_exact(self, actual["ok"], expected)

    def test_judge_exact_unit_collapse(self):
        from decimal import Decimal

        from invarlock.judge_measurements.statistics import (
            ScheduledCase,
            Score,
            collapse_scores,
        )

        cases = [
            ScheduledCase("a", "u"),
            ScheduledCase("b", "u"),
            ScheduledCase("c", "v"),
        ]
        scores = [
            Score(case.case_id, side, r, Decimal(value))
            for case, value in zip(cases, ["0.1", "0.3", "0.9"], strict=True)
            for side in ["baseline", "subject"]
            for r in range(3)
        ]
        expected = collapse_scores(
            cases, scores, repetitions=3, lower_bound=Decimal(0), upper_bound=Decimal(1)
        )
        actual = self.native(
            operation="judge",
            statistic="collapse_scores",
            cases=[dataclasses.asdict(c) for c in cases],
            scores=[
                {
                    "case_id": s.case_id,
                    "side": s.side,
                    "repetition": s.repetition,
                    "value": str(s.value),
                }
                for s in scores
            ],
            repetitions=3,
            lower_bound="0",
            upper_bound="1",
        )
        expected = {
            "units": [
                {
                    "unit_id": u.unit_id,
                    "baseline": {
                        "numerator": str(u.baseline.numerator),
                        "denominator": str(u.baseline.denominator),
                    },
                    "subject": {
                        "numerator": str(u.subject.numerator),
                        "denominator": str(u.subject.denominator),
                    },
                }
                for u in expected.units
            ],
            "case_count": expected.case_count,
            "repetitions": expected.repetitions,
        }
        assert_exact(self, actual["ok"], expected)

    def test_full_judge_analysis_and_retained_source_mutations(self):
        from invarlock.judge_measurements.analysis import (
            analyze_measurements,
            decode_analysis_policy,
        )
        from tests.judge_measurements.test_analysis import _bundle, _retain, _runs

        for groups in [("u",), ("u", "u", "v"), tuple(f"u{i}" for i in range(30))]:
            plan, data = _bundle(groups=groups, repetitions=2)
            baseline, subject = _runs(plan)
            policy = {
                "format": "invarlock/judge-analysis-policy-v1",
                "plan_sha256": data["plan_sha256"],
                "metric_name": "quality",
                "decision_role": "required",
                "direction": "higher",
                "method": "fixed-benchmark-hoeffding-v1",
                "alpha": "0.05",
                "comparison_family_size": 2,
                "minimum_units": 2,
                "maximum_interval_width": "2",
                "allowed_degradation": "0.1",
                "subject_bound": "0.5",
            }
            for missing in [False, True]:
                measured = copy.deepcopy(data)
                if missing:
                    trial = measured["trials"][0]
                    trial.update(
                        status="incomplete",
                        selected_attempt=None,
                        attempts=[],
                        parse={"status": "unavailable", "rating": None, "value": None},
                    )
                    _retain(measured)
                expected = analyze_measurements(
                    plan,
                    measured,
                    decode_analysis_policy(policy, plan=plan),
                    baseline_run=baseline,
                    subject_run=subject,
                ).to_dict()
                actual = self.native(
                    operation="judge_analysis",
                    plan=plan,
                    measurements=measured,
                    policy=policy,
                    baseline_run=baseline,
                    subject_run=subject,
                )
                self.assertIn("ok", actual, actual)
                assert_exact(self, actual["ok"], expected)
            for field in ["answer_sha256", "trial_id", "parse", "source_content"]:
                altered = copy.deepcopy(data)
                if field == "parse":
                    altered["trials"][0]["parse"]["value"] = "0.25"
                    _retain(altered)
                elif field == "source_content":
                    altered["sources"][0]["content"] += " "
                else:
                    altered["trials"][0][field] = (
                        "a" * 64 if field == "answer_sha256" else "trial-" + "a" * 64
                    )
                    _retain(altered)
                with self.assertRaises(ValueError) as raised:
                    analyze_measurements(
                        plan,
                        altered,
                        decode_analysis_policy(policy, plan=plan),
                        baseline_run=baseline,
                        subject_run=subject,
                    )
                actual = self.native(
                    operation="judge_analysis",
                    plan=plan,
                    measurements=altered,
                    policy=policy,
                    baseline_run=baseline,
                    subject_run=subject,
                )
                self.assertEqual(actual["error"], str(raised.exception))

    def test_judge_precision_whole_result(self):
        from decimal import Decimal

        from invarlock.judge_measurements.analysis import JudgeAnalysisPolicy
        from invarlock.judge_measurements.precision import plan_precision
        from tests.judge_measurements.test_analysis import _bundle

        plan, _ = _bundle(groups=tuple(f"u{i}" for i in range(20)))
        policy = JudgeAnalysisPolicy(
            direction="higher",
            allowed_degradation=Decimal("0.1"),
            maximum_interval_width=Decimal("0.3"),
        )
        expected = plan_precision(plan, policy)
        actual = self.native(
            operation="judge_precision",
            plan=plan,
            policy={
                k: str(v) if isinstance(v, Decimal) else v
                for k, v in dataclasses.asdict(policy).items()
            },
        )
        self.assertIn("ok", actual, actual)
        assert_exact(self, actual["ok"], expected)

    def test_inspect_retained_event_replay(self):
        import hashlib

        from invarlock.judge_measurements.analysis import (
            analyze_measurements,
            decode_analysis_policy,
        )
        from invarlock.judge_measurements.contracts import canonical_payload
        from tests.judge_measurements.test_inspect_judge import (
            data,
            frozen_runs,
            ingest,
        )

        fixture = data.__wrapped__()
        plan = fixture[0]
        measurements = ingest(fixture)
        runs = frozen_runs(fixture)
        policy = {
            "format": "invarlock/judge-analysis-policy-v1",
            "plan_sha256": measurements["plan_sha256"],
            "metric_name": "quality",
            "decision_role": "required",
            "direction": "higher",
            "method": "fixed-benchmark-hoeffding-v1",
            "alpha": "0.05",
            "comparison_family_size": 2,
            "minimum_units": 1,
            "maximum_interval_width": "2",
            "allowed_degradation": "0.1",
            "subject_bound": None,
        }
        expected = analyze_measurements(
            plan, measurements, decode_analysis_policy(policy, plan=plan), **runs
        ).to_dict()
        actual = self.native(
            operation="judge_analysis",
            plan=plan,
            measurements=measurements,
            policy=policy,
            **runs,
        )
        self.assertIn("ok", actual, actual)
        assert_exact(self, actual["ok"], expected)
        for mutation in [
            "cache",
            "provider_request",
            "usage",
            "event_id",
            "model",
            "budget",
        ]:
            altered = copy.deepcopy(measurements)
            source = json.loads(altered["sources"][0]["content"])
            event = source["records"][0]["events"][0]
            if mutation == "cache":
                event["cache"] = "reused"
            if mutation == "provider_request":
                event["call"]["request"]["config"]["temperature"] = "1"
            if mutation == "usage":
                event["output"]["usage"]["output_tokens"] += 1
            if mutation == "event_id":
                event["uuid"] = "different"
            if mutation == "model":
                event["output"]["model"] = "different"
            if mutation == "budget":
                source["collection"]["max_calls"] = 1
            raw = canonical_payload(source)
            altered["sources"][0].update(
                content=raw.decode(),
                byte_size=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
            with self.assertRaises(ValueError) as raised:
                analyze_measurements(
                    plan, altered, decode_analysis_policy(policy, plan=plan), **runs
                )
            actual = self.native(
                operation="judge_analysis",
                plan=plan,
                measurements=altered,
                policy=policy,
                **runs,
            )
            self.assertEqual(actual["error"], str(raised.exception))

    def test_inspect_import_complete_outputs_and_errors(self):
        from invarlock.judge_measurements import import_export
        from invarlock.judge_measurements.collector import _import_sample
        from invarlock.judge_measurements.contracts import (
            canonical_payload,
            measurement_plan_digest,
        )
        from tests.judge_measurements.test_inspect_judge import (
            data,
            frozen_runs,
            ingest,
        )

        fixture = data.__wrapped__()
        plan, export, frozen, options = fixture
        runs = frozen_runs(fixture)
        actual = self.native(
            operation="import_inspect",
            export=export,
            plan=plan,
            options=dataclasses.asdict(options),
            **runs,
        )
        self.assertIn("ok", actual, actual)
        assert_exact(self, actual["ok"], ingest(fixture))
        expected = _import_sample(
            export["samples"][0],
            plan=plan,
            options=options,
            digest=measurement_plan_digest(plan),
            bindings={r["case_id"]: r for r in plan["answer_bindings"]},
            frozen_inputs=frozen,
            record_index=0,
            event_ids=set(),
        )
        actual = self.native(
            operation="import_inspect_sample",
            sample=export["samples"][0],
            plan=plan,
            options=dataclasses.asdict(options),
            **runs,
        )
        self.assertIn("ok", actual, actual)
        assert_exact(self, actual["ok"], expected)
        for mutation in [
            "epoch",
            "sample_id",
            "plan_digest",
            "duplicate_event",
            "request",
            "cache",
            "calls",
            "missing",
        ]:
            changed = copy.deepcopy(export)
            changed_options = options
            if mutation == "epoch":
                changed["samples"][0]["epoch"] = True
            if mutation == "sample_id":
                changed["samples"][0]["id"] = "altered"
            if mutation == "plan_digest":
                changed["samples"][0]["metadata"]["plan_sha256"] = "a" * 64
            if mutation == "duplicate_event":
                changed["samples"][1]["events"][0]["uuid"] = changed["samples"][0][
                    "events"
                ][0]["uuid"]
            if mutation == "request":
                changed["samples"][0]["events"][0]["call"]["request"]["config"][
                    "temperature"
                ] = "1"
            if mutation == "cache":
                changed["samples"][0]["events"][0]["cache"] = "reused"
            if mutation == "calls":
                changed_options = dataclasses.replace(options, max_calls=1)
                changed["collection"] = dataclasses.asdict(changed_options)
            if mutation == "missing":
                changed["samples"].pop()
            with self.assertRaises(ValueError) as raised:
                import_export(
                    canonical_payload(changed),
                    plan=plan,
                    options=changed_options,
                    **runs,
                )
            actual = self.native(
                operation="import_inspect",
                export=changed,
                plan=plan,
                options=dataclasses.asdict(changed_options),
                **runs,
            )
            self.assertEqual(actual.get("error"), str(raised.exception), mutation)

    def test_retained_collection_prefix(self):
        import tempfile

        import pytest

        from tests.judge_measurements.test_openai_compatible_collector import _collect

        with (
            tempfile.TemporaryDirectory() as directory,
            pytest.MonkeyPatch.context() as monkeypatch,
        ):
            measurements, (plan, _, baseline, subject), _ = _collect(
                Path(directory).resolve(), monkeypatch
            )
        for count in range(len(measurements["sources"]) + 1):
            actual = self.native(
                operation="collection_prefix",
                sources=measurements["sources"][:count],
                plan=plan,
                baseline_run=baseline,
                subject_run=subject,
            )
            self.assertEqual(actual.get("ok"), {"valid": True}, actual)
        changed = copy.deepcopy(measurements["sources"][:1])
        changed[0]["sha256"] = "a" * 64
        actual = self.native(
            operation="collection_prefix",
            sources=changed,
            plan=plan,
            baseline_run=baseline,
            subject_run=subject,
        )
        self.assertIn("digest does not match", actual.get("error", ""))

    def test_compatible_retained_http_replay(self):
        import hashlib
        import tempfile

        import pytest

        from invarlock.judge_measurements.analysis import (
            analyze_measurements,
            decode_analysis_policy,
        )
        from invarlock.judge_measurements.contracts import canonical_payload
        from tests.judge_measurements.test_openai_compatible_collector import _collect

        for transport in [{}, {"status": 500}, {"body": b"malformed"}]:
            with (
                tempfile.TemporaryDirectory() as directory,
                pytest.MonkeyPatch.context() as monkeypatch,
            ):
                measurements, (plan, _, baseline, subject), _ = _collect(
                    Path(directory).resolve(), monkeypatch, **transport
                )
            policy = {
                "format": "invarlock/judge-analysis-policy-v1",
                "plan_sha256": measurements["plan_sha256"],
                "metric_name": "quality",
                "decision_role": "required",
                "direction": "higher",
                "method": "fixed-benchmark-hoeffding-v1",
                "alpha": "0.05",
                "comparison_family_size": 2,
                "minimum_units": 1,
                "maximum_interval_width": "2",
                "allowed_degradation": "0.1",
                "subject_bound": None,
            }
            expected = analyze_measurements(
                plan,
                measurements,
                decode_analysis_policy(policy, plan=plan),
                baseline_run=baseline,
                subject_run=subject,
            ).to_dict()
            actual = self.native(
                operation="judge_analysis",
                plan=plan,
                measurements=measurements,
                policy=policy,
                baseline_run=baseline,
                subject_run=subject,
            )
            self.assertIn("ok", actual, actual)
            assert_exact(self, actual["ok"], expected)
            if not transport:
                for mutation in [
                    "wire",
                    "identity",
                    "credential",
                    "body",
                    "header",
                    "outcome",
                ]:
                    altered = copy.deepcopy(measurements)
                    source = json.loads(altered["sources"][0]["content"])
                    if mutation == "wire":
                        source["http"]["request"]["temperature"] = 1.0
                    if mutation == "identity":
                        source["service_identity"]["endpoint_sha256"] = "a" * 64
                    if mutation == "credential":
                        source["http"]["request"]["authorization"] = "retained"
                    if mutation == "body":
                        source["http"]["response_body"]["sha256"] = "a" * 64
                    if mutation == "header":
                        source["http"]["response_headers"]["x-private"] = "unsupported"
                    if mutation == "outcome":
                        source["http"]["outcome"] = "unknown"
                    raw = canonical_payload(source)
                    altered["sources"][0].update(
                        content=raw.decode(),
                        byte_size=len(raw),
                        sha256=hashlib.sha256(raw).hexdigest(),
                    )
                    with self.assertRaises(ValueError) as raised:
                        analyze_measurements(
                            plan,
                            altered,
                            decode_analysis_policy(policy, plan=plan),
                            baseline_run=baseline,
                            subject_run=subject,
                        )
                    actual = self.native(
                        operation="judge_analysis",
                        plan=plan,
                        measurements=altered,
                        policy=policy,
                        baseline_run=baseline,
                        subject_run=subject,
                    )
                    self.assertEqual(actual["error"], str(raised.exception))

    def test_inspect_recipe_budget_preflight(self):
        from invarlock.judge_measurements.workflow import _collection_budgets
        from tests.judge_measurements.test_inspect_judge import data

        plan, _, _, options = data.__wrapped__()
        for change in [
            None,
            "version",
            "fields",
            "grader",
            "epochs",
            "attempts",
            "budget",
            "anthropic-new",
            "anthropic-old",
            "google-new",
            "google-old",
        ]:
            p = copy.deepcopy(plan)
            c = dataclasses.asdict(options)
            c["inspect_version"] = "0.3.263"
            if change == "version":
                c["inspect_version"] = "0.3.254"
            if change == "fields":
                c["extra"] = True
            if change == "grader":
                c["grader"] = None
            if change == "epochs":
                c["epochs"] = True
            if change == "attempts":
                p["schedule"]["max_attempts"] = 2
            if change == "budget":
                c["max_calls"] = 1.0
            if change and change.startswith(("anthropic", "google")):
                c["grader"] = {
                    "anthropic-new": "anthropic/claude-sonnet-4-6",
                    "anthropic-old": "anthropic/claude-sonnet-4-5",
                    "google-new": "google/gemini-3.1-pro",
                    "google-old": "google/gemini-2.5-flash",
                }[change]
                p["judge"]["requested_model"] = c["grader"]
                p["judge"]["config"].update(seed=None, reasoning_effort=None)
            actual = self.native(operation="inspect_budgets", collection=c, plan=p)
            try:
                expected = _collection_budgets(c, p)
            except ValueError as exc:
                self.assertEqual(actual.get("error"), str(exc), change)
            else:
                self.assertEqual(actual.get("ok"), expected, change)

    def test_inspect_import_real_source_sharding(self):
        import hashlib

        from invarlock.evaluation_records.cases import case_set_digest
        from invarlock.evaluation_records.io import run_digest
        from invarlock.judge_measurements import import_export
        from invarlock.judge_measurements.contracts import (
            canonical_payload,
            expected_trial_id,
            measurement_plan_digest,
        )
        from tests.judge_measurements.test_inspect_judge import data, frozen_runs

        fixture = data.__wrapped__()
        plan, original, _, options = copy.deepcopy(fixture)
        runs = frozen_runs(fixture)
        cases = [f"case-{index:04d}" for index in range(3864)]
        plan["sampling"]["case_units"] = [{"case_id": c, "unit_id": c} for c in cases]
        plan["answer_bindings"] = [
            dict(plan["answer_bindings"][0], case_id=c) for c in cases
        ]
        plan["schedule"]["expected_trials"] = 7728
        for side in ["baseline", "subject"]:
            run = runs[f"{side}_run"]
            run["records"] = [dict(run["records"][0], id=c) for c in cases]
            plan[f"{side}_run_sha256"] = run_digest(run)
        plan["case_set_sha256"] = case_set_digest(
            {
                "format": "invarlock/evaluation-case-set-v1",
                "cases": [
                    {k: r[k] for k in ["id", "input", "expected", "metadata"]}
                    for r in runs["baseline_run"]["records"]
                ],
            }
        )
        digest = measurement_plan_digest(plan)
        export = dict(original, samples=[])
        for case in cases:
            for template in original["samples"]:
                sample = copy.deepcopy(template)
                sample["metadata"].update(case_id=case, plan_sha256=digest)
                sample["id"] = expected_trial_id(
                    digest, case, sample["metadata"]["side"], 1
                )
                sample["events"][0]["uuid"] = f"event-{len(export['samples'])}"
                sample["events"][0]["output"]["request_id"] = (
                    f"response-{len(export['samples'])}"
                )
                export["samples"].append(sample)
        options = dataclasses.replace(
            options,
            max_calls=7728,
            max_input_tokens=10**9,
            max_output_tokens=10**9,
            max_cost_microusd=10**9,
        )
        export["collection"] = dataclasses.asdict(options)
        expected = import_export(
            canonical_payload(export), plan=plan, options=options, **runs
        )
        actual = self.native(
            operation="import_inspect",
            export=export,
            plan=plan,
            options=dataclasses.asdict(options),
            **runs,
        )
        self.assertIn("ok", actual, actual)
        actual = actual["ok"]
        self.assertGreater(len(actual["sources"]), 1)
        self.assertEqual(actual["completeness"]["completed_trials"], 7728)
        self.assertTrue(
            all(s["byte_size"] <= 16 * 1024 * 1024 for s in actual["sources"])
        )
        # Complete canonical byte equality includes all trials, per-shard mappings and raw retained facts.
        self.assertEqual(
            hashlib.sha256(canonical_payload(actual)).digest(),
            hashlib.sha256(canonical_payload(expected)).digest(),
        )

    def test_judge_signed_zero_and_large_family(self):
        from decimal import Decimal

        from invarlock.judge_measurements.statistics import hoeffding_interval

        for lower, upper, values, comparisons in [
            ("-1", "1", ["-0.00000000000000001"], 2),
            ("-0", "-0", ["0"], 2),
            (
                "-0.00000000000000001",
                "-0.00000000000000001",
                ["-0.00000000000000001"],
                2,
            ),
            ("0", "1", ["0.2", "0.8"], 10**100),
        ]:
            expected = dataclasses.asdict(
                hoeffding_interval(
                    list(map(Decimal, values)),
                    lower_bound=Decimal(lower),
                    upper_bound=Decimal(upper),
                    comparisons=comparisons,
                )
            )
            expected = {
                k: str(v) if isinstance(v, Decimal) else v for k, v in expected.items()
            }
            actual = self.native(
                operation="judge",
                statistic="hoeffding_interval",
                values=values,
                lower_bound=lower,
                upper_bound=upper,
                comparisons=comparisons,
            )
            self.assertIn("ok", actual, actual)
            assert_exact(self, actual["ok"], expected)

    def test_compatible_endpoint_normalization_and_preflight(self):
        import tempfile

        import pytest

        from invarlock.judge_measurements.openai_compatible import (
            validate_openai_compatible_collection,
        )
        from invarlock.judge_measurements.openai_compatible_contract import (
            normalized_base_url,
        )
        from tests.judge_measurements.test_openai_compatible_collector import _collect

        urls = [
            "http://LOCALHOST:80/v1///",
            "https://example.test:443/a/v1",
            "http://[::1]:8080/v1",
            "https://bücher.example/v1",
            "https://faß.example/v1",
            "https://ςσ.example/v1",
            "https://한글.example/v1",
            "https://例え。テスト/v1",
            "https://Kelvin.example/v1",
            "https://אבג.example/v1",
            "https://aאב.example/v1",
            "https://example.test/v1?",
            "https://example.test/v1#",
            "http://localhost:0/v1",
            "https://example.test./v1",
            "https://api.openai.com/v1",
            "ftp://host/v1",
            "http://user@host/v1",
            "http://host:99999/v1",
            "http://host/a//v1",
            "http://host/%2f/v1",
            "http://host/v1?q=1",
            "http://[broken]/v1",
            "http://ab／cd/v1",
            "http://ab：cd/v1",
            "http://host/../v1",
            "http://host",
            "http://a..b/v1",
            "http://" + "a" * 64 + "/v1",
            "https://xn--é.example/v1",
            "http://[v1.local]/v1",
        ]
        randomizer = random.Random(305)
        chars = list("abc012-") + list("äéßΣςαбйאב漢字한글K\u00ad\u200c\u212b")
        urls += [
            "https://"
            + "".join(randomizer.choice(chars) for _ in range(8))
            + ".example/v1"
            for _ in range(80)
        ]
        for url in urls:
            actual = self.native(operation="compatible_base_url", base_url=url)
            try:
                expected = normalized_base_url(url)
            except ValueError as exc:
                self.assertEqual(actual.get("error"), str(exc), url)
            else:
                self.assertEqual(actual.get("ok"), expected, url)
        with (
            tempfile.TemporaryDirectory() as directory,
            pytest.MonkeyPatch.context() as monkeypatch,
        ):
            measurements, (plan, _, _, _), _ = _collect(
                Path(directory).resolve(), monkeypatch
            )
        collection = json.loads(measurements["sources"][0]["content"])["collection"]
        for mutation in [
            None,
            "field",
            "profile",
            "service",
            "auth",
            "format",
            "max_calls",
            "model",
            "identity",
            "budget",
            "reasoning",
        ]:
            changed = copy.deepcopy(collection)
            p = copy.deepcopy(plan)
            if mutation == "field":
                changed["unknown"] = True
            if mutation == "profile":
                changed["profile"] = "unknown"
            if mutation == "service":
                changed["service"] = "unknown"
            if mutation == "auth":
                changed["authentication"] = "unknown"
            if mutation == "format":
                changed["response_format"] = "unknown"
            if mutation == "max_calls":
                changed["max_calls"] = True
            if mutation == "model":
                changed["model"] = "bad model"
            if mutation == "identity":
                changed["base_url"] = "http://localhost:9090/v1"
            if mutation == "budget":
                changed["max_calls"] = 1
            if mutation == "reasoning":
                p["judge"]["config"]["seed"] = None
            actual = self.native(
                operation="compatible_collection", collection=changed, plan=p
            )
            try:
                expected = validate_openai_compatible_collection(changed, p)
            except ValueError as exc:
                self.assertEqual(actual.get("error"), str(exc), mutation)
            else:
                assert_exact(self, actual["ok"], expected)

    def test_large_integer_thresholds_keep_python_numeric_order(self):
        rows = [{"id": "a", "input": "x", "expected": "yes", "output": "yes"}]
        b, s = run(rows), run(rows)
        p = policy()
        p["metrics"][0].update(
            maximum_regression=10**400,
            maximum_interval_width=10**400,
            subject_minimum=10**400,
        )
        self.comparison(b, s, p)
        p["metrics"][0].update(subject_minimum=-(10**400), subject_maximum=10**400)
        self.comparison(b, s, p)
        p["metrics"][0].update(subject_minimum=10**400, subject_maximum=-(10**400))
        self.error(b, s, p)
        p = policy()
        p["metrics"][0]["minimum_count"] = 1.0
        self.comparison(b, s, p)
        p["expected_case_set_digest"] = digest("different cases")
        self.error(b, s, p)

    def test_projection_and_case_set_identity(self):
        from invarlock.evaluation_records.cases import case_set_digest
        from invarlock.evaluator_capture import capture_evaluator_run

        rows = [
            {
                "id": "a",
                "input": {"question": "hello"},
                "expected": "yes",
                "output": "yes",
                "context": {"retained": 1},
            }
        ]
        b = capture_evaluator_run(
            rows,
            source={"name": "source", "version": "1"},
            run_id="test",
            artifact_digest=digest("model"),
            input_projection={"kind": "json-pointer", "pointer": "/input/question"},
        )
        s = copy.deepcopy(b)
        p = policy()
        self.comparison(b, s, p)
        p["expected_case_set_digest"] = case_set_digest(
            {
                "format": "invarlock/evaluation-case-set-v1",
                "cases": [
                    {k: r[k] for k in ["id", "input", "expected", "metadata"]}
                    for r in b["records"]
                ],
            }
        )
        self.comparison(b, s, p)
        altered = copy.deepcopy(s)
        altered["records"][0]["context"]["input_projection"]["source"]["input"][
            "question"
        ] = "changed"
        self.error(b, altered, p)


if __name__ == "__main__":
    unittest.main()
