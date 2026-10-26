"""Exact presentation parity; these checks do not authenticate source evidence."""

import copy
import dataclasses
import json
import os
import subprocess
import unittest

from invarlock.report_presentation import (
    AuthoredReview,
    CheckView,
    IntervalView,
    MetricView,
    ReportView,
    render_html,
    render_markdown,
)


class ReportingDifferential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executable = os.environ.get("INVARLOCK_ATTESTATION_BIN")
        if not executable:
            raise unittest.SkipTest("INVARLOCK_ATTESTATION_BIN is unset")
        cls.process = subprocess.Popen(
            [executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )

    @classmethod
    def tearDownClass(cls):
        cls.process.stdin.close()
        cls.process.wait(timeout=10)
        cls.process.stdout.close()

    def native(self, **request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        self.assertTrue(line, "native process terminated")
        return json.loads(line)

    def check(self, view):
        for op, expected in [
            ("html", render_html(view)),
            ("markdown", render_markdown(view)),
            ("markdown-details", render_markdown(view, include_details=True)),
        ]:
            actual = self.native(
                operation=op.split("-")[0],
                view=dataclasses.asdict(view),
                include_details=op.endswith("details"),
            )
            self.assertIn("ok", actual, actual)
            if actual["ok"] != expected:
                from pathlib import Path

                Path("/tmp/report-expected.txt").write_text(expected)
                Path("/tmp/report-actual.txt").write_text(actual["ok"])
                i = next(
                    (
                        i
                        for i, (a, b) in enumerate(
                            zip(actual["ok"], expected, strict=False)
                        )
                        if a != b
                    ),
                    min(len(actual["ok"]), len(expected)),
                )
                self.fail(
                    f"{op} differs at {i}: actual {actual['ok'][i - 80 : i + 160]!r}; expected {expected[i - 80 : i + 160]!r}"
                )

    def test_shared_views_and_escape_cases(self):
        metric = MetricView(
            name="exact_match",
            scope="all_cases",
            decision="pass",
            baseline="0.5",
            candidate="0.75",
            change="+25 pp",
            count="4",
            explanation="Exact match is 0.75, from 0.5, a +25 pp change.",
            checks=(CheckView("Bound", "-5", "≥ -10", True, "Approved bound"),),
            interval=IntervalView(
                -5,
                40,
                25,
                -10,
                "Paired 95% confidence interval",
                "pp",
                threshold_direction="minimum",
                neutral=0.0,
                method="test",
                basis=("Exact paired cases.",),
            ),
            notes=("These are retained cases.",),
            evidence_source="paired.json",
        )
        basic = ReportView(
            title="Model comparison",
            family="Captured results",
            decision="pass",
            summary="Exact match improved from 0.5 to 0.75, a +25 pp change.",
            metrics=(metric,),
            assurance=(("Signature", "Checked"),),
            identity=(("Policy", "sha256:abc"),),
            subjects=(("Baseline", "Model A"), ("Subject", "Model B")),
            context=(
                ("Baseline deployment", "region-a"),
                ("Subject deployment", "region-b"),
                ("Baseline request", "same"),
                ("Subject request", "same"),
                ("Scope", "fixed"),
            ),
            changes=("Model changed.", "Deployment changed."),
            technical={"value": 0.5, "nested": [1, None, True]},
            details=(
                (
                    "Policy",
                    {
                        "metrics": [
                            {
                                "name": "exact_match",
                                "configuration": {
                                    "preview_only": True,
                                    "text": "{…}",
                                    "limits": {"depth": 6},
                                    "note": "Preview only",
                                },
                            }
                        ]
                    },
                ),
            ),
            next_steps=("Review scope",),
            limitations=("No acceptance inferred",),
        )
        self.check(basic)
        self.check(
            dataclasses.replace(
                basic,
                metrics=(),
                assurance=(),
                identity=(),
                subjects=(),
                context=(),
                changes=(),
                limitations=(),
                details=(),
                next_steps=(),
            )
        )
        for decision in ["regression", "insufficient_evidence", "unknown"]:
            self.check(
                dataclasses.replace(
                    basic,
                    decision=decision,
                    metrics=(
                        dataclasses.replace(
                            metric, decision=decision, checks=(), interval=None
                        ),
                    ),
                )
            )
        metrics = tuple(
            dataclasses.replace(
                metric,
                name="token_f1" if i % 2 else "exact_match",
                scope=f"slice_{i}",
                decision="pass" if i == 0 else "regression",
            )
            for i in range(7)
        )
        self.check(dataclasses.replace(basic, metrics=metrics, decision="regression"))
        self.check(
            dataclasses.replace(
                basic,
                review=AuthoredReview(
                    label="Owner review",
                    byline="Signed by owner",
                    context=(("Purpose", "QA"),),
                    recommendation="Review next",
                    decision="Pending",
                    basis=(("Evidence", "Retained"),),
                    next_action="Resolve gaps",
                ),
            )
        )
        hostile = "A< & > \" quote ' x\x1b\u202e\t\n _alone_ internal_word [link](url) `code` \\ | *strong*"
        self.check(
            dataclasses.replace(
                basic,
                title=hostile,
                summary=hostile,
                context=((hostile, hostile),),
                subjects=(),
                metrics=(
                    dataclasses.replace(
                        metric, name=hostile, scope=hostile, notes=(hostile,)
                    ),
                ),
            )
        )

    def test_comparison_view_and_junit(self):
        from test_analysis_differential import metric, policy, run

        from invarlock.evaluation_comparison.comparison import compare_runs
        from invarlock.record_reporting import _view, render_junit

        for kind in ["exact_match", "token_f1", "numeric_tolerance"]:
            for outputs in [["yes", "yes"], ["no", "yes"], ["no", "no"]]:
                rows = [
                    {
                        "id": str(i),
                        "input": "q",
                        "output": "yes",
                        "expected": "yes",
                        "scores": {"quality": 1},
                    }
                    for i in range(2)
                ]
                if kind == "numeric_tolerance":
                    for row in rows:
                        row.update(output="1", expected="1")
                changed = copy.deepcopy(rows)
                for row, output in zip(changed, outputs, strict=True):
                    row.update(
                        output=("1" if output == "yes" else "0")
                        if kind == "numeric_tolerance"
                        else output,
                        scores={"quality": float(output == "yes")},
                    )
                p = policy(
                    metric(
                        kind,
                        {"absolute": 0.1, "relative": 0}
                        if kind == "numeric_tolerance"
                        else {"unicode_version": "15.0.0"}
                        if kind == "token_f1"
                        else {},
                    )
                )
                comparison = compare_runs(
                    run(rows, "baseline"), run(changed, "subject"), p
                )
                expected = json.loads(
                    json.dumps(dataclasses.asdict(_view(comparison, None)))
                )
                actual = self.native(operation="comparison_view", comparison=comparison)
                self.assertIn("ok", actual, actual)
                self.assertEqual(actual["ok"], expected, kind)
                self.assertEqual(
                    self.native(operation="junit", comparison=comparison).get("ok"),
                    render_junit(comparison).decode(),
                )

    def test_captured_snapshot_view(self):
        import tempfile
        from pathlib import Path

        from invarlock.captured_reporting import _load, _view
        from tests.evidence_sets.test_verification import fixture

        for context in [False, True]:

            def change(baseline, subject):
                for side, run in [("baseline", baseline), ("subject", subject)]:
                    for row in run["records"]:
                        row["context"] = {
                            "model_key": "model_a",
                            "workflow": "quality",
                            "effective_messages": [
                                {"role": "user", "content": row["input"]}
                            ],
                        }
                        if side == "subject":
                            row["context"]["effective_messages"].insert(
                                0, {"role": "system", "content": "Be exact. <&>"}
                            )

            with tempfile.TemporaryDirectory() as directory:
                root, *_ = fixture(
                    Path(directory).resolve(),
                    captured_change=change if context else None,
                )
                path = root / "deterministic"
                manifest, payloads, signer, _ = _load(path)
                expected = json.loads(
                    json.dumps(dataclasses.asdict(_view(manifest, payloads, signer)))
                )
                actual = self.native(operation="captured_view", path=str(path))
                self.assertIn("ok", actual, actual)
                self.assertEqual(actual["ok"]["view"], expected)

    def test_native_snapshot_and_closed_report(self):
        import tempfile
        from pathlib import Path

        from invarlock.evidence_pack import EvidenceObservation
        from invarlock.evidence_pack_contract import (
            build_comparison_report,
            canonical_json_bytes,
            sha256_digest,
        )
        from invarlock.evidence_reporting import _render_native_evidence
        from tests.evidence_packs.test_evidence_pack import (
            _publish,
            _rebind_and_resign_pack,
        )

        modes = [
            "valid",
            "fail",
            "observation",
            "v1",
            "v2",
            "verdict",
            "mean",
            "count",
            "paired_count",
            "paired_p",
            "paired_interval",
            "uncertainty",
            "scope",
            "mass",
            "extra",
            "limit",
            "comparison",
            "bad_type",
            "sample",
            "side_accuracy",
        ]
        for mode in modes:
            with tempfile.TemporaryDirectory() as directory:
                observation = (
                    (
                        EvidenceObservation(
                            "note",
                            "comparison",
                            "note",
                            canonical_json_bytes({"note": "retained <&>"}),
                        ),
                    )
                    if mode == "observation"
                    else ()
                )
                pack, policy, _, _, key, args = _publish(
                    Path(directory).resolve(),
                    delta_min_pp=-1 if mode == "fail" else -100,
                    observations=observation,
                )
                path = pack / "reports/evaluation.report.json"
                report = json.loads(path.read_bytes())
                if mode in ["v1", "v2"]:
                    report = build_comparison_report(
                        comparison_id="single-comparison",
                        paired_records=json.loads(
                            (pack / "records/paired-records.json").read_bytes()
                        ),
                        policy=json.loads(policy.read_bytes()),
                        policy_digest=sha256_digest(policy.read_bytes()),
                        report_format="invarlock/comparison-report-" + mode,
                    )
                if mode == "verdict":
                    report["verdict"] = "fail"
                if mode == "mean":
                    report["baseline"]["mean_score"] = 0.7
                if mode == "count":
                    report["record_count"] = 3
                if mode == "paired_count":
                    report["paired_binary"]["both_pass"] += 1
                if mode == "paired_p":
                    report["paired_binary"]["mcnemar_exact_two_sided_p_value"] = 0.123
                if mode == "paired_interval":
                    report["paired_binary"]["effect_size_confidence_interval"][
                        "lower_pp"
                    ] += 1
                if mode == "uncertainty":
                    report["uncertainty"]["lower"] += 1
                if mode == "scope":
                    report["uncertainty"]["scope"] = "other"
                if mode == "mass":
                    report["uncertainty"]["interval_mass"] = 0.8
                if mode == "extra":
                    report["unexpected"] = True
                if mode == "limit":
                    report["comparison"]["minimum"] = -101
                if mode == "comparison":
                    report["comparison"]["value"] = 10
                if mode == "bad_type":
                    report["baseline"]["mean_score"] = True
                if mode == "sample":
                    report["sample_qualification"] = {}
                if mode == "side_accuracy":
                    report["side_accuracy"] = {}
                if mode not in ["valid", "fail", "observation"]:
                    path.chmod(0o600)
                    path.write_bytes(canonical_json_bytes(report))
                    _rebind_and_resign_pack(pack, key)
                actual = self.native(operation="native_view", path=str(pack))
                try:
                    view, digest = _render_native_evidence(pack)
                except ValueError as error:
                    self.assertEqual(actual.get("error"), str(error), mode)
                else:
                    self.assertIn("ok", actual, actual)
                    self.assertEqual(
                        actual["ok"]["view"],
                        json.loads(json.dumps(dataclasses.asdict(view))),
                        mode,
                    )
                    self.assertEqual(actual["ok"]["manifest_digest"], digest)
                    self.check(view)

    def test_native_likelihood_and_sample_qualification(self):
        import tempfile
        from pathlib import Path

        from invarlock.evidence_pack import InputIdentity, publish_comparison_evidence
        from invarlock.evidence_pack_contract import canonical_json_bytes, sha256_digest
        from invarlock.evidence_reporting import _render_native_evidence
        from tests.evidence_packs.test_evidence_pack import (
            _nll_policy_bytes,
            _publish,
            _side_evidence,
        )

        for mode in ["nll", "sample"]:
            with tempfile.TemporaryDirectory() as directory:
                d = Path(directory).resolve()
                _, policy, _, runtimes, _, args = _publish(d)
                if mode == "nll":
                    policy.write_bytes(_nll_policy_bytes(1.3))
                    args["normalized_request"]["comparison"]["metric"] = (
                        "normalized_nll_per_utf8_byte"
                    )
                else:
                    policy.write_bytes(
                        canonical_json_bytes(
                            {
                                "resolved_policy": {
                                    "metrics": {
                                        "exact_match": {
                                            "delta_min_pp": -100.0,
                                            "minimum_record_count": 3,
                                            "maximum_interval_width_pp": 200.0,
                                            "minimum_side_accuracy": 0.8,
                                        }
                                    }
                                }
                            }
                        )
                    )
                pd = sha256_digest(policy.read_bytes())
                args.update(
                    policy_bytes=policy.read_bytes(),
                    policy=InputIdentity(pd, locator="inputs/policy.json"),
                )
                for side, marker, probs in [
                    ("baseline", "c", (-2.0, -2.0)),
                    ("subject", "d", (-2.5, -2.0)),
                ]:
                    options = (
                        {"logprob_sums": probs}
                        if mode == "nll"
                        else {
                            "outputs": ("A", "B") if side == "baseline" else ("A", "C")
                        }
                    )
                    args[side + "_evidence"] = _side_evidence(
                        d / (mode + side),
                        schedule=args["schedule"],
                        image_digest=runtimes[side],
                        artifact_marker=marker,
                        role=side,
                        policy_digest=pd,
                        **options,
                    )
                pack = d / mode
                publish_comparison_evidence(pack, **args)
                view, digest = _render_native_evidence(pack)
                actual = self.native(operation="native_view", path=str(pack))
                self.assertIn("ok", actual, actual)
                self.assertEqual(
                    actual["ok"]["view"],
                    json.loads(json.dumps(dataclasses.asdict(view))),
                )
                self.check(view)

    def test_judge_policy_views(self):
        import tempfile
        from pathlib import Path

        from invarlock.judge_measurements.reporting import _snapshot, _view
        from tests.judge_measurements.test_evidence_acceptance import _publish

        for options in [
            {},
            {"baseline": 1, "subject": 0},
            {"incomplete": True},
            {"role": "advisory"},
            {"signed": False},
            {"policy_changes": {"minimum_units": 10**100}},
            {"policy_changes": {"subject_bound": "0.5"}},
            {"policy_changes": {"direction": "lower", "subject_bound": "0.5"}},
            {"policy_changes": {"maximum_interval_width": "0.01"}},
            {"policy_changes": {"alpha": "0.001", "comparison_family_size": 7}},
        ]:
            with tempfile.TemporaryDirectory() as directory:
                publication, _ = _publish(Path(directory).resolve(), **options)
                snapshot, artifacts = _snapshot(publication.path)
                view, facts = _view(snapshot, artifacts)
                actual = self.native(operation="judge_view", path=str(publication.path))
                self.assertIn("ok", actual, actual)
                self.assertEqual(
                    actual["ok"]["view"],
                    json.loads(json.dumps(dataclasses.asdict(view))),
                    options,
                )
                self.assertEqual(
                    actual["ok"]["facts"], json.loads(json.dumps(facts)), options
                )
                self.check(view)

    def test_report_publication_transactions(self):
        import tempfile
        from pathlib import Path

        from invarlock.evidence_reporting import EvidenceReportError, render_evidence
        from tests.evidence_sets.test_verification import fixture

        for family in ["deterministic", "judge", "set"]:
            for mode in ["success", "collision", "existing", "inside"]:
                with tempfile.TemporaryDirectory() as directory:
                    d = Path(directory).resolve()
                    root, _ = fixture(d)
                    evidence = root if family == "set" else root / family
                    outputs = {
                        "html_path": d / "out.html",
                        "markdown_path": d / "out.md",
                        "junit_path": d / "out.xml",
                    }
                    if mode == "collision":
                        outputs["markdown_path"] = outputs["html_path"]
                    if mode == "existing":
                        outputs["html_path"].write_text("existing")
                    if mode == "inside":
                        outputs["html_path"] = evidence / "report.html"
                    try:
                        expected = render_evidence(evidence, **outputs, explain=True)
                    except EvidenceReportError as error:
                        actual = self.native(
                            operation="report",
                            evidence_path=str(evidence),
                            explain=True,
                            **{k: str(v) for k, v in outputs.items()},
                        )
                        self.assertEqual(
                            actual["ok"]["payload"], error.payload, (family, mode)
                        )
                        continue
                    files = {k: v.read_bytes() for k, v in outputs.items()}
                    for v in outputs.values():
                        v.unlink()
                    actual = self.native(
                        operation="report",
                        evidence_path=str(evidence),
                        explain=True,
                        **{k: str(v) for k, v in outputs.items()},
                    )
                    self.assertEqual(actual["ok"]["status"], 0, actual)
                    self.assertEqual(actual["ok"]["text"], expected.text)
                    self.assertEqual(
                        actual["ok"]["payload"], json.loads(expected.as_json())
                    )
                    for k, v in outputs.items():
                        self.assertEqual(v.read_bytes(), files[k], (family, k))

    def test_interval_geometry_and_display(self):
        for bounds in [
            (0.0, 0.0, 0.0, None, None),
            (-1.0, 1.0, 0.0, 0.0, 0.0),
            (0.9, 1.1, 1.01, 1.2, 1.0),
            (1e-9, 2e-9, 1.5e-9, 1.7e-9, 0.0),
            (-1e308, 1e308, 0.0, 1e308, 0.0),
            (-0.00009999, 0.00010001, 0.0001, 0.0, 0.0),
            (-0.0, 0.0, -0.0, None, -0.0),
        ]:
            lower, upper, estimate, threshold, neutral = bounds
            m = MetricView(
                name="normalized_nll",
                scope="global",
                decision="pass",
                baseline="1",
                candidate="1",
                change="0",
                count="12",
                explanation="Recorded data",
                interval=IntervalView(
                    lower,
                    upper,
                    estimate,
                    threshold,
                    "95% confidence interval",
                    "ratio",
                    threshold_direction="maximum",
                    neutral=neutral,
                ),
            )
            self.check(
                ReportView(
                    title="Interval",
                    family="Diagnostic",
                    decision="pass",
                    summary="1 and 0 observed.",
                    metrics=(m,),
                    assurance=(),
                )
            )


if __name__ == "__main__":
    unittest.main()
